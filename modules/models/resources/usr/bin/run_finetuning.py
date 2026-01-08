import torch
import time
import cv2
from utils_finetuning import patchify, FinetuningDataset, Patch2D, PanopticLoss
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from empanada.inference import engines
from empanada import metrics


# from sklearn import measure
# from skimage import io

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
    # why not
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
        is_val_epoch = True  # (epoch + 1) % config["EVAL"]["epochs_per_eval"]
        if is_val_epoch:
            validate(
                eval_loader=eval_loader,  # only supports batch size of 1?
                model=model,
                criterion=criterion,
                config=config,
            )
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


def validate(eval_loader, model, criterion, config):
    print("running validation :)")
    # validation metrics to track
    class_names = config["EVAL"]["class_names"]
    metric_dict = {}
    for metric_params in config["EVAL"]["metrics"]:
        reg_name = metric_params["name"]
        metric_name = metric_params["metric"]
        metric_params = {
            k: v for k, v in metric_params.items() if k not in ["name", "metric"]
        }
        metric_dict[reg_name] = metrics.__dict__[metric_name](
            metrics.AverageMeter, **metric_params
        )

    meters = metrics.ComposeMetrics(metric_dict, class_names)

    # validation tracking
    batch_time = ProgressAverageMeter("Time", ":6.3f")
    loss_meters = None

    progress = ProgressMeter(len(eval_loader), [batch_time], prefix="Validation: ")

    # create the Inference Engine
    engine_name = config["EVAL"]["engine"]
    engine = engines.__dict__[engine_name](model, **config["EVAL"]["engine_params"])

    for i, batch in enumerate(eval_loader):
        end = time.time()
        images = batch["image"]
        target = {k: v for k, v in batch.items() if k not in ["image", "fname"]}

        images = images.to(config["TRAIN"]["device"], non_blocking=True)
        target = {
            k: tensor.to(config["TRAIN"]["device"], non_blocking=True)
            for k, tensor in target.items()
        }

        # compute panoptic segmentations
        # from prediction and ground truth
        output = engine.infer(images)
        semantic = engine._harden_seg(output["sem"])
        output["pan_seg"] = engine.postprocess(
            semantic, output["ctr_hmp"], output["offsets"]
        )
        target["pan_seg"] = engine.postprocess(
            target["sem"].unsqueeze(1), target["ctr_hmp"], target["offsets"]
        )

        loss, aux_loss = criterion(output, target)

        # record losses
        if loss_meters is None:
            loss_meters = {}
            for k, v in aux_loss.items():
                loss_meters[k] = ProgressAverageMeter(k, ":.4e")
                loss_meters[k].update(v)
                # add to progress
                progress.meters.append(loss_meters[k])
        else:
            for k, v in aux_loss.items():
                loss_meters[k].update(v)

        # compute metrics
        with torch.no_grad():
            meters.evaluate(output, target)

        batch_time.update(time.time() - end)

        # if i % config["EVAL"]["print_freq"] == 0:
        progress.display(i)

    # end of epoch print evaluation metrics
    print("\n")
    print(f"Validation results:")
    meters.display()
    print("\n")


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


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class ProgressAverageMeter(metrics.AverageMeter):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        super().__init__()

    def __str__(self):
        fmtstr = "{name} {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)
