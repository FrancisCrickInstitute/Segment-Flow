from pathlib import Path
from typing import Union
import yaml

import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm.auto import tqdm

from utils import (
    save_masks,
    create_argparser_inference,
    guess_rgb,
    load_img,
    align_segment_labels,
)
from model_utils import get_device


def run_sam(
    save_dir: Union[Path, str],
    save_name: str,
    fpath: Union[Path, str],
    model_type: str,
    model_chkpt: Union[Path, str],
    model_config: dict,
    start_idx: int,
    end_idx: int,
):
    sam = sam_model_registry[model_type](checkpoint=model_chkpt)
    sam.to(get_device())
    # Create the model
    model = SamAutomaticMaskGenerator(sam, **model_config)
    # Load the image
    img = load_img(fpath, start_idx, end_idx)
    # Extract the dimensions
    ndim = img.ndim
    # Reduce ndims if RGB (i.e. it's a single RGB image, not a stack)
    if guess_rgb(img.shape):
        ndim -= 1
    # Create the progress bar for this stack
    pbar = tqdm(total=end_idx - start_idx, desc=f"{Path(fpath).stem}")
    # Send the image to the corresponding run func based on slice or stack
    if ndim == 2:
        all_masks = _run_sam_slice(img, model, pbar)
    elif ndim == 3:
        all_masks = _run_sam_stack(save_dir, save_name, img, model, pbar, start_idx)
    elif ndim == 4:
        if img.shape[0] == 1:
            img = img.squeeze()
            all_masks = _run_sam_stack(save_dir, save_name, img, model, pbar, start_idx)
        else:
            raise ValueError("Cannot handle a stack of multi-channel images")
    else:
        raise ValueError("Can only handle an image, or stack of images!")
    save_masks(
        save_dir,
        save_name,
        all_masks,
        curr_idx=end_idx - start_idx,
        start_idx=start_idx,
    )
    pbar.close()
    return img, all_masks


def _run_sam_slice(img_slice, model, pbar):
    # Expand to 3-channel if not rgb
    if not guess_rgb(img_slice.shape):
        img_slice = np.stack((img_slice,) * 3, axis=-1)
    img_slice = img_slice[..., :3]
    masks = model.generate(img_slice)
    # Convert the masks into a napari-friendly format
    mask_img = create_mask_arr(masks)
    # Update progress bar
    pbar.update(1)
    return mask_img


def _run_sam_stack(save_dir, save_name, img_stack, model, pbar, start_idx):
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
        # Save this slice (and all previous slices)
        save_masks(
            save_dir=save_dir,
            save_name=save_name,
            masks=all_masks,
            curr_idx=idx + 1,
            start_idx=start_idx,
        )
    # Align masks to ensure consistent colouration
    all_masks = align_segment_labels(all_masks)
    return all_masks


# def align_segment_labels(all_masks):
#     aligned_masks = all_masks.copy()

#     for idx in range(all_masks.shape[0] - 1):
#         curr_slice = all_masks[idx]
#         next_slice = all_masks[idx+1]
#         curr_labels, curr_counts = np.unique(curr_slice, return_counts=True)
#         next_labels, next_counts = np.unique(next_slice, return_counts=True)

#         cost_matrix = np.zeros((
#             curr_labels.size,
#             next_labels.size
#         ), dtype=int)

#         for label in next_labels:
#             cost_matrix[curr_slice == label] += 1


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


def create_mask_arr(masks):
    # Sort the masks/annotations by area to allow overwriting/lapping
    sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    # Need integers for napari Labels layer
    # NOTE: argmax used in other libraries, but wrong due to double-use of 0
    # That will result in 1 less mask present
    mask_img = np.zeros(sorted_anns[0]["segmentation"].shape, dtype=int)
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
        start_idx=cli_args.start_idx,
        end_idx=cli_args.end_idx,
    )
