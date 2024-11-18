from pathlib import Path
from typing import Union
import yaml
import warnings

import hydra
from hydra import initialize_config_dir
import numpy as np
import requests
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from utils import (
    save_masks,
    create_argparser_inference,
    guess_rgb,
    load_img,
    extract_idxs,
)
from model_utils import get_device

# NOTE: Placeholder until internalisation from Meta, or us
BASE_CONFIGS = {
    "hiera_base": {
        "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/82b026cd5578af78757323ab99a0b5c8dc456cff/sam2_configs/sam2_hiera_b%2B.yaml",
        "fname": "sam2_hiera_b+.yaml",
    },
    "hiera_large": {
        "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/82b026cd5578af78757323ab99a0b5c8dc456cff/sam2_configs/sam2_hiera_l.yaml",
        "fname": "sam2_hiera_l.yaml",
    },
    "hiera_small": {
        "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/82b026cd5578af78757323ab99a0b5c8dc456cff/sam2_configs/sam2_hiera_s.yaml",
        "fname": "sam2_hiera_s.yaml",
    },
    "hiera_tiny": {
        "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/82b026cd5578af78757323ab99a0b5c8dc456cff/sam2_configs/sam2_hiera_t.yaml",
        "fname": "sam2_hiera_t.yaml",
    },
}


def run_sam2(
    img: np.ndarray,
    save_dir: Union[Path, str],
    save_name: str,
    model_type: str,
    model_chkpt: Union[Path, str],
    model_config: dict,
    idxs: list[int, ...],
):
    # Need to handle model types and get the appropriate model
    if model_type == "default":
        model_type = "hiera_base"
    if model_type not in BASE_CONFIGS:
        raise ValueError(f"Model type {model_type} not found!")
    base_config = get_sam2_config(model_type)
    device = get_device()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(Path.cwd())):
        sam2 = build_sam2(
            config_file=base_config,
            ckpt_path=model_chkpt,
            device=device,
            mode="eval",
            apply_postprocessing=False,  # TODO: Why?
        )
    # Create the AMG model
    model = SAM2AutomaticMaskGenerator(sam2, **model_config)
    if img.max() > 255:
        warnings.warn(
            "Image values are greater than 255, converting to uint8. This may result in loss of information."
        )
        img = img.astype(np.uint8)
    # Extract the dimensions
    ndim = img.ndim
    # Reduce ndims if RGB (i.e. it's a single RGB image, not a stack)
    if guess_rgb(img.shape, dim=-1):
        ndim -= 1
    # Send the image to the corresponding run func based on slice or stack
    if ndim == 2:
        all_masks = _run_sam2_slice(img, model)
    elif ndim == 3:
        all_masks = _run_sam2_stack(img, model)
    elif ndim == 4:
        if img.shape[0] == 1:
            img = img.squeeze()
            all_masks = _run_sam2_stack(img, model)
        else:
            raise ValueError("Cannot handle a stack of multi-channel images")
    else:
        raise ValueError("Can only handle an image, or stack of images!")
    save_masks(save_dir, save_name, all_masks, idxs=idxs, mask_type="instance")
    return img, all_masks


def get_sam2_config(model_type):
    req = requests.get(BASE_CONFIGS[model_type]["url"])
    req.raise_for_status()
    # Load the YAML
    cfg = yaml.safe_load(req.text)
    # Dump it into current working folder for Hydra to 'discover'
    fname = BASE_CONFIGS[model_type]["fname"]
    with open(fname, "w") as f:
        yaml.dump(cfg, f)
    return fname


def _run_sam2_slice(img_slice, model):
    # Expand to 3-channel if not rgb
    if not guess_rgb(img_slice.shape, dim=-1):
        img_slice = np.stack((img_slice,) * 3, axis=-1)
    img_slice = img_slice[..., :3]
    masks = model.generate(img_slice)
    # Convert the masks into a napari-friendly format
    mask_img = create_mask_arr(masks, img_slice.shape[:2])
    return mask_img


def _run_sam2_stack(img_stack, model):
    # Initialize the container of all masks
    if guess_rgb(img_stack.shape, dim=-1):
        all_masks = np.zeros(img_stack.shape[:3], dtype=int)
    else:
        all_masks = np.zeros(img_stack.shape, dtype=int)
    # Get the contrast limits
    if img_stack.dtype == np.uint8:
        contrast_limits = (0, 255)
    else:
        min_val = np.nanmin(img_stack)
        max_val = np.nanmax(img_stack)
        if min_val == max_val:
            contrast_limits = (0, 1)
        else:
            contrast_limits = (float(min_val), float(max_val))
    # Loop over each stack and run
    for idx in range(img_stack.shape[0]):
        img_slice = img_stack[idx]
        # Expand to 3-channel if not rgb
        if not guess_rgb(img_slice.shape, dim=-1):
            img_slice = np.stack((img_slice,) * 3, axis=-1)
        img_slice = img_slice[..., :3]
        # Normalize the slice
        # Convert to uint8 just in case
        img_slice = normalize_slice(
            img_slice, source_limits=contrast_limits, target_limits=(0, 255)
        ).astype(np.uint8)
        # Actually run the model on this slice
        masks = _run_sam2_slice(img_slice, model)
        # Insert the masks for this slice
        all_masks[idx, ...] = masks
    return all_masks


def normalize_slice(img_slice, source_limits, target_limits):
    # From https://github.com/MIC-DKFZ/napari-sam/blob/main/src/napari_sam/utils.py
    if source_limits is None:
        source_limits = (img_slice.min(), img_slice.max())

    if target_limits is None:
        target_limits = (0, 1)

    if source_limits[0] == source_limits[1] or target_limits[0] == target_limits[1]:
        return img_slice * 0
    else:
        x_std = (img_slice - source_limits[0]) / (source_limits[1] - source_limits[0])
        x_scaled = x_std * (target_limits[1] - target_limits[0]) + target_limits[0]
        return x_scaled


def create_mask_arr(masks: list, slice_shape):
    # No segmentations, return empty array
    if len(masks) == 0:
        return np.zeros(slice_shape, dtype=int)
    # Sort the masks/annotations by area to allow overwriting/lapping
    sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    # Need integers for napari Labels layer
    # NOTE: Seen argmax used in other libraries, but wrong due to double-use of 0 (if included!)
    # That will result in 1 less mask present
    mask_img = np.zeros(slice_shape, dtype=int)
    for i, mask in enumerate(sorted_anns):
        mask_img[mask["segmentation"]] = i + 1
    return mask_img


if __name__ == "__main__":
    parser = create_argparser_inference()
    cli_args = parser.parse_args()

    with open(cli_args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    img = load_img(
        fpath=cli_args.img_path,
        idxs=cli_args.idxs,
        channels=cli_args.channels,
        num_slices=cli_args.num_slices,
        preprocess_params=cli_args.preprocess_params,
        dim_order="ZYXC",
    )

    # Squeze it!
    img = np.squeeze(img)

    img, masks = run_sam2(
        img=img,
        save_dir=Path(cli_args.output_dir),
        save_name=cli_args.mask_fname,
        model_type=cli_args.model_type,
        model_chkpt=cli_args.model_chkpt,
        model_config=model_config,
        idxs=cli_args.idxs,
    )
