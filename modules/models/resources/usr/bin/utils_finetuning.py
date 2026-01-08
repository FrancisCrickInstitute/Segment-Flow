import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage import io
from torch.utils.data import Dataset
from copy import deepcopy
from glob import glob
from skimage import measure
import matplotlib.pyplot as plt
from typing import Union, Optional

# recreate data set class


class Patch2D(nn.Module):
    def __init__(
        self,
        patch_size: Union[list, tuple] = (512, 512),
        stride: Optional[Union[list, tuple, int]] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        if stride is None:
            self.stride = patch_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=self.stride, padding=0)

    def forward(self, x, mask):
        # Currently CxHxW, need BxCxHxW
        if x.ndim == 3:
            x = x.unsqueeze(0)
        # Masks are just HxW, so need to add a channel dimension
        elif x.ndim == 2:
            # https://github.com/pytorch/pytorch/issues/44989
            # Unfold doesn't support uint8, so need to convert to float
            x = x.to(torch.float32).unsqueeze(0).unsqueeze(0)
        # Unfold the image
        B, C, H, W = x.shape
        x = self.unfold(x)
        # Reshape (currently BxCxHxWxpatches)
        # Then squeeze to remove leading batch dim
        x = x.view(B, C, *self.patch_size, -1).permute(0, 4, 1, 2, 3).squeeze(0)
        if mask:
            # No channels, so remove and convert dtype
            return x.squeeze(1).to(torch.uint8)
        else:
            return x.squeeze(1)


def patchify(data_dir, patcher):
    subdirs = []
    for sd in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, sd)):
            subdirs.append(sd)

    impaths_dict = {}
    mskpaths_dict = {}

    for sd in subdirs:
        impaths_dict[sd] = glob(os.path.join(data_dir, f"{sd}/images/*"))
        mskpaths_dict[sd] = glob(os.path.join(data_dir, f"{sd}/masks/*"))

    impaths = []
    for paths in impaths_dict.values():
        impaths.extend(paths)

    mskpaths = []
    for paths in mskpaths_dict.values():
        mskpaths.extend(paths)

    data = {"image": [], "mask": []}

    print(f"found {len(impaths)} images and found {len(mskpaths)} masks")

    for f in impaths:
        image = cv2.imread(f, 0)
        image = torch.from_numpy(image)
        patches = patcher(image, mask=False)
        for patch in patches:
            data["image"].append(patch)

    for f in mskpaths:
        mask = io.imread(f)
        mask = torch.from_numpy(mask)
        patches = patcher(mask, mask=True)
        for patch in patches:
            data["mask"].append(patch)

    print(f"made {len( data['image'] )} images and made {len( data['mask'] )} masks")
    return data


def heatmap_and_offsets(sl2d, heatmap_sigma=6):
    r"""Creates center heatmap and offsets for panoptic deeplab
    training.

    Args:
        sl2d: Array of (h, w) defining an instance segmentation.

        heatmap_sigma: Float. Standard deviation of the Guassian filter
        that is applied to create the center heatmap.

    Returns:
        heatmap: Array of (1, h, w) defining the heatmap of instance centers.

        offsets: Array of (2, h, w) defining the offsets from each pixel
        to the associated instance center. Channels are up-down and
        left-right offsets respectively.

    """
    # make sure, the input is numpy
    convert = False
    if type(sl2d) == torch.Tensor:
        sl2d = sl2d.numpy()
        convert = True

    h, w = sl2d.shape
    centers = np.zeros((2, h, w), dtype=np.float32)
    heatmap = np.zeros((h, w), dtype=np.float32)

    # loop over the instance labels and store
    # relevant centers for each
    rp = measure.regionprops(sl2d)
    for r in rp:
        sub_label = r.label
        y, x = r.centroid
        heatmap[int(y), int(x)] = 1
        centers[0, sl2d == sub_label] = y
        centers[1, sl2d == sub_label] = x

    # apply a gaussian filter to spread the centers
    heatmap = cv2.GaussianBlur(
        heatmap,
        ksize=(0, 0),
        sigmaX=heatmap_sigma,
        sigmaY=heatmap_sigma,
        borderType=cv2.BORDER_CONSTANT,
    )

    hmax = heatmap.max()
    if hmax > 0:
        heatmap = heatmap / hmax

    # convert from centers to offsets
    yindices = np.arange(0, h, dtype=np.float32)
    xindices = np.arange(0, w, dtype=np.float32)

    # add the y indices to the first channel
    # in the output and x indices to the second channel
    offsets = np.zeros_like(centers)
    offsets[0] = centers[0] - yindices[:, None]
    offsets[1] = centers[1] - xindices[None, :]
    offsets[:, sl2d == 0] = 0

    # add empty dimension to heatmap
    heatmap = heatmap[None]  # (1, H, W)

    if convert:
        heatmap = torch.from_numpy(heatmap)
        offsets = torch.from_numpy(offsets)

    return heatmap, offsets


# patcher = Patch2D(patch_size=(32, 32))

# data_dir = "/Users/ahmedn/Work/finetuning/c_elegans_mitos/"
# data = patchify(data_dir=data_dir, patcher=patcher)
# print(data.keys())
# print(len(data["image"]))
# print(len(data["mask"]))
# print(data)


class FinetuningDataset:
    def __init__(self, data, transforms=None, heatmap_sigma=6, weight_gamma=0.3):
        self.data = data
        self.transforms = transforms
        self.heatmap_sigma = heatmap_sigma
        self.weight_gamma = (
            weight_gamma  # what is this and how is it used in the base dataclass
        )
        if weight_gamma is not None:
            pass
            # self.weights = self._example_weights(self.data, self.weight_gamma)

    def __len__(self):
        return len(self.data["image"])

    @staticmethod
    def _example_weights(
        paths_dict, gamma=0.3
    ):  # need to edit this to use data not the paths dict : )
        # counts by source subdirectory
        counts = np.array([len(paths) for paths in paths_dict.values()])

        # invert and gamma the distribution
        weights = 1 / counts
        weights = weights ** (gamma)

        # for interpretation, normalize weights
        # s.t. they sum to 1
        total_weights = weights.sum()
        weights /= total_weights

        # repeat weights per n images
        example_weights = []
        for w, c in zip(weights, counts):
            example_weights.extend([w] * c)

        return torch.tensor(example_weights)

    def __getitem__(self, idx):
        image = self.data["image"][idx]
        mask = self.data["mask"][idx]

        if image.ndim == 2:
            image = image[..., None]

        if self.transforms:
            output = self.transforms(image=image.numpy(), mask=mask.numpy())
        else:
            output = {"image": image, "mask": mask}

        # add all the other stuff like heatmap and offsets
        # ---
        mask = output["mask"]
        heatmap, offsets = heatmap_and_offsets(mask, self.heatmap_sigma)
        output["ctr_hmp"] = heatmap
        output["offsets"] = offsets

        # adding sem
        if isinstance(mask, torch.Tensor):
            output["sem"] = (mask > 0).float()
        elif isinstance(mask, np.ndarray):
            output["sem"] = (mask > 0).astype(np.float32)
        else:
            raise Exception(f"Invalid mask type {type(mask)}. Expect tensor or ndarry")

        return output


# DATASET STUFF


class _BaseDataset(Dataset):
    r"""Pytorch Dataset class that supports addition of multiple
    segmentation datasets and computes sampling weights based
    on the count of images within subdirectories and a weight_gamma
    factor.
    """

    def __init__(self, data_dir, transforms=None, weight_gamma=None):
        super(_BaseDataset, self).__init__()
        self.data_dir = data_dir

        self.subdirs = []
        for sd in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, sd)):
                self.subdirs.append(sd)

        # images and masks as dicts ordered by subdirectory
        self.impaths_dict = {}
        self.mskpaths_dict = {}

        for sd in self.subdirs:
            self.impaths_dict[sd] = glob(os.path.join(data_dir, f"{sd}/images/*"))
            self.mskpaths_dict[sd] = glob(os.path.join(data_dir, f"{sd}/masks/*"))

        # calculate weights per example, if weight gamma is not None
        self.weight_gamma = weight_gamma
        if weight_gamma is not None:
            self.weights = self._example_weights(self.impaths_dict, gamma=weight_gamma)
        else:
            self.weights = None

        # unpack dicts to lists of images
        self.impaths = []
        for paths in self.impaths_dict.values():
            self.impaths.extend(paths)

        self.mskpaths = []
        for paths in self.mskpaths_dict.values():
            self.mskpaths.extend(paths)

        print(
            f"Found {len(self.subdirs)} image subdirectories with {len(self.impaths)} images."
        )

        self.transforms = transforms

    def __len__(self):
        return len(self.impaths)

    def __add__(self, add_dataset):
        # make a copy of self
        merged_dataset = deepcopy(self)

        # add the dicts and append lists/dicts
        for sd in add_dataset.impaths_dict.keys():
            if sd in merged_dataset.impaths_dict:
                # concat lists of paths together
                merged_dataset.impaths_dict[sd] += add_dataset.impaths_dict[sd]
                merged_dataset.mskpaths_dict[sd] += add_dataset.mskpaths_dict[sd]
            else:
                merged_dataset.impaths_dict[sd] = add_dataset.impaths_dict[sd]
                merged_dataset.mskpaths_dict[sd] = add_dataset.mskpaths_dict[sd]

        # recalculate weights
        if merged_dataset.weight_gamma is not None:
            merged_dataset.weights = self._example_weights(
                merged_dataset.impaths_dict, gamma=merged_dataset.weight_gamma
            )
        else:
            merged_dataset.weights = None

        # unpack dicts to lists of images
        merged_dataset.impaths = []
        for paths in merged_dataset.impaths_dict.values():
            merged_dataset.impaths.extend(paths)

        merged_dataset.mskpaths = []
        for paths in merged_dataset.mskpaths_dict.values():
            merged_dataset.mskpaths.extend(paths)

        return merged_dataset

    @staticmethod
    def _example_weights(paths_dict, gamma=0.3):
        # counts by source subdirectory
        counts = np.array([len(paths) for paths in paths_dict.values()])

        # invert and gamma the distribution
        weights = 1 / counts
        weights = weights ** (gamma)

        # for interpretation, normalize weights
        # s.t. they sum to 1
        total_weights = weights.sum()
        weights /= total_weights

        # repeat weights per n images
        example_weights = []
        for w, c in zip(weights, counts):
            example_weights.extend([w] * c)

        return torch.tensor(example_weights)

    def __getitem__(self, idx):
        raise NotImplementedError


__all__ = ["SingleClassInstanceDataset"]


class SingleClassInstanceDataset(_BaseDataset):
    r"""Dataset for panoptic deeplab that supports a single instance
    class only.

    Args:
        data_dir: Str. Directory containing image/mask pairs. Structure should
        be data_dir -> source_datasets -> images/masks.

        transforms: Albumentations transforms to apply to images and masks.

        heatmap_sigma: Float. The standard deviation used for the gaussian
        blurring filter when converting object centers to a heatmap.

        weight_gamma: Float (0-1). Parameter than controls sampling of images
        within different source_datasets based on the number of images
        that that directory contains. Default is 0.3.

    """

    def __init__(
        self,
        data_dir,
        transforms=None,
        heatmap_sigma=6,
        weight_gamma=0.3,
    ):
        super(SingleClassInstanceDataset, self).__init__(
            data_dir, transforms, weight_gamma
        )
        self.heatmap_sigma = heatmap_sigma

    def __getitem__(self, idx):
        # transformed and paste example
        f = self.impaths[idx]
        image = cv2.imread(f, 0)
        print(f"{image.shape=}")
        mask = io.imread(self.mskpaths[idx])

        # add channel dimension if needed
        if image.ndim == 2:
            image = image[..., None]

        if self.transforms is not None:
            output = self.transforms(image=image, mask=mask)
        else:
            output = {"image": image, "mask": mask}

        mask = output["mask"]
        heatmap, offsets = heatmap_and_offsets(mask, self.heatmap_sigma)
        output["ctr_hmp"] = heatmap
        output["offsets"] = offsets
        output["fname"] = f

        # the last step is to binarize the mask for semantic segmentation
        if isinstance(mask, torch.Tensor):
            output["sem"] = (mask > 0).float()
        elif isinstance(mask, np.ndarray):
            output["sem"] = (mask > 0).astype(np.float32)
        else:
            raise Exception(
                f"Invalid mask type {type(mask)}. Expect tensor or ndarray."
            )

        return output


# LOSS FUNCTIONS - criterion


class BootstrapCE(nn.Module):
    r"""Standard (binary) cross-entropy loss where only the top
    k percent of largest loss values are averaged.

    Args:
        top_k_percent_pixels: Float, fraction of largest loss values
            to average. Default 0.2

    """

    def __init__(self, top_k_percent_pixels=0.2):
        super(BootstrapCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, labels):
        if logits.size(1) == 1:
            # add channel dim for BCE
            # (N, H, W) -> (N, 1, H, W)
            labels = labels.unsqueeze(1)
            pixel_losses = self.bce(logits, labels)
        else:
            pixel_losses = self.ce(logits, labels)

        pixel_losses = pixel_losses.contiguous().view(-1)

        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.mean()

        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)

        return pixel_losses.mean()


class HeatmapMSE(nn.Module):
    r"""
    Mean squared error (MSE) loss for instance center heatmaps
    """

    def __init__(self):
        super(HeatmapMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        return self.mse(output, target)


class OffsetL1(nn.Module):
    r"""
    L1 loss for instance center offsets. Loss is only calculated
    within the confines of the semantic segmentation.
    """

    def __init__(self):
        super(OffsetL1, self).__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, output, target, offset_weights):
        l1 = self.l1(output, target) * offset_weights

        weight_sum = offset_weights.sum()
        if weight_sum == 0:
            return l1.sum() * 0
        else:
            return l1.sum() / weight_sum


@torch.jit.script
def point_sample(
    features, point_coords, mode: str = "bilinear", align_corners: bool = False
):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)

    output = F.grid_sample(
        features,
        2.0 * point_coords - 1.0,
        mode=mode,
        align_corners=align_corners,
    )

    if add_dim:
        output = output.squeeze(3)

    return output


class PointRendLoss(nn.Module):
    r"""Standard (binary) cross-entropy between logits at
    points sampled by the point rend module.
    """

    def __init__(self):
        super(PointRendLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, point_logits, point_coords, labels):
        # sample the labels at the given coordinates
        point_labels = point_sample(
            labels.unsqueeze(1).float(),
            point_coords,
            mode="nearest",
            align_corners=False,
        )

        if point_logits.size(1) == 1:
            point_losses = self.bce(point_logits, point_labels)
        else:
            point_labels = point_labels.squeeze(1).long()
            point_losses = self.ce(point_logits, point_labels)

        return point_losses


class PanopticLoss(nn.Module):
    r"""Defines the overall panoptic loss function which combines
    semantic segmentation, instance centers and offsets.

    Args:
        ce_weight: Float, weight to apply to the semantic segmentation loss.

        mse_weight: Float, weight to apply to the centers heatmap loss.

        l1_weight: Float, weight to apply to the center offsets loss.

        pr_weight: Float, weight to apply to the point rend semantic
            segmentation loss. Only applies if using a Point Rend enabled model.

        top_k_percent: Float, fraction of largest semantic segmentation
            loss values to consider in BootstrapCE.

    """

    def __init__(
        self,
        ce_weight=1,
        mse_weight=200,
        l1_weight=0.01,
        pr_weight=1,
        top_k_percent=0.2,
    ):
        super(PanopticLoss, self).__init__()
        self.mse_loss = HeatmapMSE()
        self.l1_loss = OffsetL1()
        self.ce_loss = BootstrapCE(top_k_percent)
        self.pr_loss = PointRendLoss()

        self.ce_weight = ce_weight
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.pr_weight = pr_weight

    def forward(self, output, target):
        mse = self.mse_loss(output["ctr_hmp"], target["ctr_hmp"])
        ce = self.ce_loss(output["sem_logits"], target["sem"])

        # only evaluate loss inside of ground truth segmentation
        offset_weights = (target["sem"] > 0).unsqueeze(1)
        l1 = self.l1_loss(output["offsets"], target["offsets"], offset_weights)

        aux_loss = {"ce": ce.item(), "l1": l1.item(), "mse": mse.item()}
        total_loss = self.ce_weight * ce + self.mse_weight * mse + self.l1_weight * l1

        if "sem_points" in output:
            pr_ce = self.pr_loss(
                output["sem_points"], output["point_coords"], target["sem"]
            )
            aux_loss["pointrend_ce"] = pr_ce.item()
            total_loss += self.pr_weight * pr_ce

        aux_loss["total_loss"] = total_loss.item()
        return total_loss, aux_loss
