import os
from pathlib import Path

import albumentations as A
import torch
import torch.optim as optim
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from utils import create_argparser_finetune
from utils_finetuning import FinetuningDataset, PanopticLoss, Patch2D, patchify

augmentations = sorted(
    name
    for name in A.__dict__
    if callable(A.__dict__[name]) and not name.startswith("__") and name[0].isupper()
)


def setup_augmentations(augs, norms):
    # set the training image augmentations
    aug_string = []
    dataset_augs = []
    for aug_params in augs:
        aug_name = aug_params["aug"]

        assert (
            aug_name in augmentations or aug_name == "CopyPaste"
        ), f"{aug_name} is not a valid albumentations augmentation!"

        aug_string.append(aug_params["aug"])
        del aug_params["aug"]
        dataset_augs.append(A.__dict__[aug_name](**aug_params))

    aug_string = ",".join(aug_string)

    tfs = A.Compose([*dataset_augs, A.Normalize(**norms), ToTensorV2()])
    return tfs


def update_training_metrics_file(epoch, average_loss, fpath):
    mode = "w" if epoch == 1 else "a"  # overwrite the file if first epoch
    with open(fpath, mode) as f:
        f.write(f"{epoch},{average_loss:.6f}\n")


def finetune(config):
    print("starting finetuning")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dir = config["TRAIN"]["train_dir"]
    model_dir = config["TRAIN"]["model_dir"]
    save_dir = config["TRAIN"]["save_dir"]
    save_name = config["TRAIN"]["save_name"]
    epochs = config["TRAIN"]["epochs"]
    finetune_layer = config["TRAIN"]["layers"]
    patch_size = config["TRAIN"]["patch_size"]
    transforms = config["TRAIN"]["transforms"]
    # not user selected atm more of a computational concern
    batch_size = config["TRAIN"].get("batch_size") or 16
    num_workers = config["TRAIN"].get("num_workers")

    # patch the images
    patcher = Patch2D(patch_size)
    data = patchify(train_dir, patcher)

    batch_size = min(
        batch_size, len(data["image"])
    )  # Batch size shouldn't be more than patch size

    data_cls = FinetuningDataset
    train_dataset = data_cls(data, transforms=transforms, weight_gamma=0.7)

    # Use provided num_workers or fall back to auto-detection
    if num_workers is None:
        cpu_count = os.cpu_count()
        num_workers = min(4, cpu_count // 2)
    print(f"Using {num_workers} workers for data loading")

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    model = torch.jit.load(model_dir, map_location=device)

    # freeze all encoder layers
    for pname, param in model.named_parameters():
        if "encoder" in pname:
            param.requires_grad = False

    # freeze specific layers
    if finetune_layer == "decoder":
        pass
    elif finetune_layer == "all":
        for pname, param in model.named_parameters():
            if "encoder" in pname:
                param.requires_grad = True
    else:
        # unfreeze is cumulative from layer 1 to chosen layer
        layers = ["layer1", "layer2", "layer3", "layer4"]
        for layer_name in layers[: layers.index(finetune_layer) + 1]:
            for pname, param in model.named_parameters():
                if layer_name in pname:
                    param.requires_grad = True
    num_trainable = sum(
        p[1].numel() for p in model.named_parameters() if p[1].requires_grad
    )
    print(f"Training {num_trainable} parameters.")

    # Create save directory if not present
    p = Path(save_dir)
    base_dir = Path(os.path.realpath(p))
    base_dir.mkdir(parents=True, exist_ok=True)

    metrics_fpath = p / "training_metrics.csv"

    optimizer = configure_optimizer(model, "AdamW", weight_decay=0.1, lr=0.00001)

    criterion = PanopticLoss()
    for epoch in range(1, epochs + 1):
        average_loss = train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            device=device,
        )
        update_training_metrics_file(epoch, average_loss, metrics_fpath)
    torch.jit.save(model, save_dir + "/" + save_name + ".pth")


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    config,
    epoch,
    device,
):
    total_loss = 0.0
    batch_count = 0

    model.train()
    for i, batch in enumerate(train_loader):
        images = batch["image"]
        target = {k: v for k, v in batch.items() if k not in ["image", "fname"]}

        images = images.float()
        images = images.to(device, non_blocking=True)
        target = {
            k: tensor.to(device, non_blocking=True) for k, tensor in target.items()
        }

        optimizer.zero_grad()

        output = model(images)

        loss, aux_loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        batch_count += 1

    average_loss = total_loss / batch_count
    return average_loss


def configure_optimizer(model, opt_name, **opt_params):
    """
    Takes an optimizer and separates parameters into two groups
    that either use weight decay or are exempt.

    Only BatchNorm parameters and biases are excluded.
    """

    # if theres not weight decay return all trainable params
    if "weight_decay" not in opt_params or opt_params["weight_decay"] == 0:
        return optim.__dict__[opt_name](
            (p for p in model.parameters() if p.requires_grad), **opt_params
        )

    decay = set()
    no_decay = set()
    param_dict = {}

    blacklist = (torch.nn.BatchNorm2d,)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue  # skip params which won't be trained (frozen)
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


def setup_finetuning(
    train_dir,
    model_dir,
    model_type,
    save_dir,
    save_name,
    patch_size,
    layers,
    epochs,
    num_workers=None,
):
    print("run_finetuning in finetune_widget")

    config_location = (
        Path(__file__).parent.resolve() / "finetune_configs" / "finetuning_config.yml"
    )

    with open(config_location, mode="r") as handle:
        finetuning_config = yaml.load(handle, Loader=yaml.FullLoader)

    # print(finetuning_config)
    augs = finetuning_config["TRAIN"]["augmentations"]
    norms = finetuning_config["TRAIN"]["norms"]
    transforms = setup_augmentations(augs, norms)

    finetuning_config["TRAIN"]["train_dir"] = train_dir
    finetuning_config["TRAIN"]["model_dir"] = model_dir
    finetuning_config["TRAIN"]["save_dir"] = save_dir
    finetuning_config["TRAIN"]["save_name"] = save_name
    finetuning_config["TRAIN"]["layers"] = layers
    finetuning_config["TRAIN"]["epochs"] = epochs
    finetuning_config["TRAIN"]["patch_size"] = [int(i) for i in patch_size.split(",")]
    finetuning_config["TRAIN"]["transforms"] = transforms
    finetuning_config["TRAIN"]["num_workers"] = num_workers

    return finetuning_config


if __name__ == "__main__":
    print("Running Finetuning!")
    parser = create_argparser_finetune()
    cli_args = parser.parse_args()
    print(cli_args)
    config = setup_finetuning(
        cli_args.train_dir,
        cli_args.model_chkpt,
        cli_args.model_type,  # For future variations between model types
        cli_args.model_save_dir,
        cli_args.model_save_name,
        cli_args.patch_size,
        cli_args.layers,
        int(cli_args.epochs),
        cli_args.num_workers,
    )
    finetune(config)
    print("Finetuning complete, please save model to local model registry to use.")
