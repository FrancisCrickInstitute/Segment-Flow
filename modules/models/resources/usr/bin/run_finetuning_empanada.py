import torch
import argparse
import yaml
from utils_finetuning import patchify, FinetuningDataset, Patch2D, PanopticLoss
from torch.utils.data import DataLoader
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

augmentations = sorted(
    name
    for name in A.__dict__
    if callable(A.__dict__[name]) and not name.startswith("__") and name[0].isupper()
)

augmentations_dict = [
    {"aug": "RandomScale", "scale_limit": [-0.9, 1]},
    {
        "aug": "PadIfNeeded",
        "min_height": 128,
        "min_width": 128,
        "border_mode": 0,
    },
    {"aug": "RandomCrop", "height": 128, "width": 128},
    {"aug": "Rotate", "limit": 180, "border_mode": 0},
    {
        "aug": "RandomBrightnessContrast",
        "brightness_limit": 0.3,
        "contrast_limit": 0.3,
    },
    {"aug": "HorizontalFlip"},
    {"aug": "VerticalFlip"},
]

# set the training image augmentations
norms = {"mean": 0.57571, "std": 0.12765}
aug_string = []
dataset_augs = []
for aug_params in augmentations_dict:
    aug_name = aug_params["aug"]

    assert (
        aug_name in augmentations or aug_name == "CopyPaste"
    ), f"{aug_name} is not a valid albumentations augmentation!"

    aug_string.append(aug_params["aug"])
    del aug_params["aug"]
    dataset_augs.append(A.__dict__[aug_name](**aug_params))

aug_string = ",".join(aug_string)

tfs = A.Compose([*dataset_augs, A.Normalize(**norms), ToTensorV2()])


def finetune(config):
    print("finetuning")
    # device = config.get("device", "cpu")
    # if there is a device key and it is truthy
    device = config.get("device") or "cpu"

    train_dir = config["TRAIN"]["train_dir"]
    model_dir = config["TRAIN"]["model_dir"]
    save_dir = config["TRAIN"]["save_dir"]
    save_name = config["TRAIN"]["save_name"]
    epochs = config["TRAIN"]["epochs"]
    finetune_layer = config["TRAIN"]["layers"]
    batch_size = config["TRAIN"].get("batch_size") or 16
    patch_size = config["TRAIN"].get("patch_size") or (64, 64)

    # patch the images
    patcher = Patch2D(patch_size)
    data = patchify(train_dir, patcher)

    print(data["image"][0].shape)
    print(len(data["image"]))
    print(data["image"][0].dtype)

    data_cls = FinetuningDataset
    train_dataset = data_cls(data, transforms=tfs, weight_gamma=0.7)

    print("--- train dataset ---")
    print(train_dataset.__getitem__(0)["image"].shape)
    print(train_dataset.__getitem__(0)["image"].dtype)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=False, drop_last=True)

    print("--- dataloader ---")
    print("train_dataloader: ", train_loader)
    print(next(iter(train_loader))["image"].shape)
    eval_loader = DataLoader(train_dataset, 1, shuffle=False, drop_last=True)

    model = torch.jit.load(model_dir, map_location=device)

    # freeze all encoder layers
    for pname, param in model.named_parameters():
        if "encoder" in pname:
            param.requires_grad = False

    # freeze specific layers
    if finetune_layer == "none":
        pass
    elif finetune_layer == "all":
        for pname, param in model.named_parameters():
            if "encoder" in pname:
                param.requires_grad = True
    else:
        # unfreeze is cumulative from layer 1 to chosen layer
        layers = ["layer1", "layer2", "layer3", "layer4"]
        for layer_name in layers[layers.index(finetune_layer) :]:
            for pname, param in model.named_parameters():
                if layer_name in pname:
                    param.requires_grad = True
    num_trainable = sum(
        p[1].numel() for p in model.named_parameters() if p[1].requires_grad
    )
    print(f"Training {num_trainable} parameters.")

    optimizer = configure_optimizer(model, "AdamW", weight_decay=0.1, lr=0.00001)

    criterion = PanopticLoss()
    for epoch in range(epochs):
        train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
        )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.jit.save(model, save_dir + "/" + save_name + ".pth")


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch=None,
    config=None,
    device="cpu",
):
    model.train()
    for i, batch in enumerate(train_loader):
        images = batch["image"]
        target = {k: v for k, v in batch.items() if k not in ["image", "fname"]}

        # images = images.permute(0, 3, 1, 2).float()
        images = images.float()
        images = images.to(device, non_blocking=True)
        target = {
            k: tensor.to(device, non_blocking=True) for k, tensor in target.items()
        }

        print(f"shape: {images.shape}")

        optimizer.zero_grad()

        output = model(images)

        for k, v in output.items():
            print(f"{k=}, shape: {v.shape}")
        loss, aux_loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"epoch = {epoch}")
        print(aux_loss)
        print(loss.item())


def configure_optimizer(model, opt_name, **opt_params):
    """
    Takes an optimizer and separates parameters into two groups
    that either use weight decay or are exempt.

    Only BatchNorm parameters and biases are excluded.
    """

    # easy if there's no weight_decay
    if "weight_decay" not in opt_params:
        return optim.__dict__[opt_name](model.parameters(), **opt_params)
    elif opt_params["weight_decay"] == 0:
        return optim.__dict__[opt_name](model.parameters(), **opt_params)

    decay = set()
    no_decay = set()
    param_dict = {}

    blacklist = (torch.nn.BatchNorm2d,)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            full_name = "%s.%s" % (mn, pn) if mn else pn

            if full_name.endswith("bias"):
                no_decay.add(full_name)
            elif full_name.endswith("weight") and isinstance(m, blacklist):
                no_decay.add(full_name)
            else:
                decay.add(full_name)

            param_dict[full_name] = p

    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "Overlapping decay and no decay"
    assert len(param_dict.keys() - union_params) == 0, "Missing decay parameters"

    decay_params = [param_dict[pn] for pn in sorted(list(decay))]
    no_decay_params = [param_dict[pn] for pn in sorted(list(no_decay))]

    param_groups = [
        {"params": decay_params, **opt_params},
        {"params": no_decay_params, **opt_params},
    ]
    param_groups[1]["weight_decay"] = 0  # overwrite default to 0 for no_decay group

    return optim.__dict__[opt_name](param_groups, **opt_params)


def run_finetuning(train_dir, model_dir, save_dir, save_name, layers, epochs):
    print("run_finetuning in finetune_widget")
    # collect user input of the finetuning widget and convert into a config

    config_location = (
        "/Users/ahmedn/Work/sandbox/finetuning/scripts/finetuning_config.yml"
    )

    try:
        with open(config_location, mode="r") as handle:
            finetuning_config = yaml.load(handle, Loader=yaml.FullLoader)
    except:
        print(
            "Couldn't open finetuning config file ensure you have finetune_config.yml"
        )

    finetuning_config["TRAIN"]["train_dir"] = train_dir
    finetuning_config["TRAIN"]["model_dir"] = model_dir
    finetuning_config["TRAIN"]["save_dir"] = save_dir
    finetuning_config["TRAIN"]["save_name"] = save_name
    finetuning_config["TRAIN"]["layers"] = layers
    finetuning_config["TRAIN"]["epochs"] = epochs
    finetuning_config["TRAIN"]["device"] = "cpu"
    finetuning_config["EVAL"]["epochs_per_eval"] = 1
    finetuning_config["EVAL"]["class_names"] = {1: "mito"}

    finetuning_config["EVAL"]["engine"] = "PanopticDeepLabEngine"

    finetuning_config["EVAL"]["engine_params"] = {
        "confidence_thr": 0.5,
        "label_divisor": 1000,
        "nms_kernel": 7,
        "nms_threshold": 0.1,
        "stuff_area": 64,
        "thing_list": [1],
        "void_label": 0,
    }
    finetuning_config["EVAL"]["print_freq"] = 0

    finetune(finetuning_config)


def create_argparser_finetune():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dir", required=True, help="Path to ground truth for finetuning"
    )
    parser.add_argument("--model_chkpt", required=True, help="Base model Checkpoint")
    parser.add_argument(
        "--model_save_name", required=True, help="Name of the final finetuned model"
    )
    parser.add_argument(
        "--model_save_dir", required=True, help="Where to save the finetuned models"
    )
    parser.add_argument(
        "--layers", required=True, help="Layers to be unfrozen when fine-tuning"
    )
    parser.add_argument(
        "--epochs", required=True, help="Number of epochs to finetune for"
    )

    return parser


if __name__ == "__main__":
    print("Running Finetuning!")
    parser = create_argparser_finetune()
    cli_args = parser.parse_args()
    print(cli_args)
    run_finetuning(
        cli_args.train_dir,
        cli_args.model_chkpt,
        cli_args.model_save_dir,
        cli_args.model_save_name,
        cli_args.layers,
        int(cli_args.epochs),
    )
    print("Finetuning complete, please save model to local model registry to use.")
