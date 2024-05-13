from pathlib import Path
from typing import Union
import yaml
import warnings

import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm.auto import tqdm

from utils import (
    save_masks,
    create_argparser_inference,
    guess_rgb,
    load_img,
    extract_idxs,
)
from model_utils import get_device


def run_sam(
    save_dir: Union[Path, str],
    save_name: str,
    fpath: Union[Path, str],
    model_type: str,
    model_chkpt: Union[Path, str],
    model_config: dict,
    idxs: list[int, ...],
):
    # Handle extra finetuned/other models
    if "MicroSAM" in model_type:
        model_type = "vit_b"
    elif model_type == "MedSAM":
        model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=model_chkpt)
    sam.to(get_device())
    # Create the model
    model = SamAutomaticMaskGenerator(sam, **model_config)
    # Load the image
    img = load_img(fpath, idxs)
    if img.max() > 255:
        warnings.warn(
            "Image values are greater than 255, converting to uint8. This may result in loss of information."
        )
        img = img.astype(np.uint8)
    # Extract the dimensions
    ndim = img.ndim
    # Reduce ndims if RGB (i.e. it's a single RGB image, not a stack)
    if guess_rgb(img.shape):
        ndim -= 1
    # Get the start and end indices
    _, _, _, _, start_z, end_z = extract_idxs(idxs)
    # Create the progress bar for this stack
    pbar = tqdm(total=end_z - start_z, desc=f"{Path(fpath).stem}")
    # Send the image to the corresponding run func based on slice or stack
    if ndim == 2:
        all_masks = _run_sam_slice(img, model, pbar)
    elif ndim == 3:
        all_masks = _run_sam_stack(save_dir, save_name, img, model, pbar)
    elif ndim == 4:
        if img.shape[0] == 1:
            img = img.squeeze()
            all_masks = _run_sam_stack(save_dir, save_name, img, model, pbar)
        else:
            raise ValueError("Cannot handle a stack of multi-channel images")
    else:
        raise ValueError("Can only handle an image, or stack of images!")
    save_masks(save_dir, save_name, all_masks, idxs=idxs)
    pbar.close()
    return img, all_masks


def _run_sam_slice(img_slice, model, pbar):
    # Expand to 3-channel if not rgb
    if not guess_rgb(img_slice.shape):
        img_slice = np.stack((img_slice,) * 3, axis=-1)
    img_slice = img_slice[..., :3]
    masks = model.generate(img_slice)
    # Convert the masks into a napari-friendly format
    mask_img = create_mask_arr(masks, img_slice.shape[:2])
    # Update progress bar
    pbar.update(1)
    return mask_img


def _run_sam_stack(save_dir, save_name, img_stack, model, pbar):
    # Initialize the container of all masks
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
        if not guess_rgb(img_slice.shape):
            img_slice = np.stack((img_slice,) * 3, axis=-1)
        img_slice = img_slice[..., :3]
        # Normalize the slice
        # Convert to uint8 just in case
        img_slice = normalize_slice(
            img_slice, source_limits=contrast_limits, target_limits=(0, 255)
        ).astype(np.uint8)
        # Actually run the model on this slice
        masks = _run_sam_slice(img_slice, model, pbar)
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


def create_mask_arr(masks, slice_shape):
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

    img, masks = run_sam(
        save_dir=Path(cli_args.output_dir),
        save_name=cli_args.mask_fname,
        fpath=cli_args.img_path,
        model_type=cli_args.model_type,
        model_chkpt=cli_args.model_chkpt,
        model_config=model_config,
        idxs=cli_args.idxs,
    )
