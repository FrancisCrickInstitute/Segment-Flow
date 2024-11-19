import os
from pathlib import Path
import psutil
from typing import Union

from aiod_utils.preprocess import get_output_shape, get_downsample_factor
from aiod_utils.io import extract_idxs_from_fname
import aiod_utils.rle as aiod_rle
import dask.array as da
import dask_image.ndmeasure
from numba import jit, prange
from numba.core import types
from numba.typed import Dict
import numpy as np
import skimage.measure
from skimage.segmentation import relabel_sequential
from tqdm import tqdm

from utils import reduce_dtype, align_segment_labels


def combine_masks(
    masks: list[str],
    overlap: list[float, ...],
    image_size: tuple[int, ...],
    preprocess_params: Union[str, Path],
):
    """
    Combine masks from each of the substacks into a single array/dataset.

    If overlap is 0, then the masks are simply inserted into their relevant indices.

    If overlap is >0, then the masks need to be combined.
    """
    # Get the chunk size from the first file
    start_x, end_x, start_y, end_y, start_z, end_z = extract_idxs_from_fname(masks[0])
    chunk_size = (end_x - start_x, end_y - start_y, end_z - start_z)
    # Check if there is XY tiling (at least one must be true for any given substack)
    xy_tiling = (
        start_x > 0 or end_x < image_size[1] or start_y > 0 or end_y < image_size[2]
    )
    # Create the array to hold the masks
    # NOTE: Using uint16 to be safe, but ideally should be taken from inputs (but slight chicken & egg)
    # Ensure image size appropriate to given dims
    if image_size[0] == 1:
        image_size = image_size[1:]
        is_2d = True
    else:
        is_2d = False
    # Get output shape given preprocessing
    output_shape = get_output_shape(options=preprocess_params, input_shape=image_size)
    # Need to handle the indices as well if downsampling occurred
    downsample_factor = get_downsample_factor(preprocess_params)
    # Create the array to hold the masks (uint16 should be fine...?)
    all_masks = np.zeros(output_shape, dtype=np.uint16)
    # Loop over each mask and insert into the array
    # Add the masks together if overlap is >0
    # NOTE: Adding together only really makes sense for binary masks
    overlap = [float(val) for val in overlap]
    if sum(overlap) == 0.0:
        for mask_path in masks:
            idxs = extract_idxs_from_fname(
                mask_path, downsample_factor=downsample_factor
            )
            mask = aiod_rle.load_encoding(mask_path)
            mask, _ = aiod_rle.decode(mask)
            # Cast boolean to allow addition
            if mask.dtype == bool:
                mask = mask.astype(np.uint8)
            all_masks = insert_mask(
                all_masks=all_masks,
                mask=mask,
                idxs=idxs,
                xy_tiling=xy_tiling,
                is_overlap=False,
                is_2d=is_2d,
            )
    # TODO: Extract this, and handle binary/labelled masks properly, with specified vote mechanism
    else:
        # Combine the masks
        for mask_path in masks:
            start_x, end_x, start_y, end_y, start_z, end_z = extract_idxs_from_fname(
                mask_path, downsample_factor=downsample_factor
            )
            mask = aiod_rle.load_encoding(mask_path)
            mask, _ = aiod_rle.decode(mask)
            # Cast boolean to allow addition
            if mask.dtype == bool:
                mask = mask.astype(np.uint8)
            # Just sum, naive method
            if is_2d:
                all_masks[start_x:end_x, start_y:end_y] += mask
            else:
                all_masks[start_z:end_z, start_x:end_x, start_y:end_y] += mask
    return reduce_dtype(all_masks)


def insert_mask(
    all_masks,
    mask,
    idxs: tuple[int, int, int, int, int, int],
    xy_tiling: bool,
    is_overlap: bool,
    is_2d: bool,
):
    # Extract the indices
    start_x, end_x, start_y, end_y, start_z, end_z = idxs
    # Ensure labels are unique across a slice
    if xy_tiling:
        # Get the current maximum value across the relevant slices
        if is_2d:
            max_val = all_masks.max()
        else:
            max_val = all_masks[start_z:end_z, ...].max()
        # # Check if we need to upcast the array
        # if max_val + mask.max() > np.iinfo(all_masks.dtype).max:
        #     all_masks = all_masks.astype(np.uint32, copy=False)
        # Add a constant to all non-zero values to ensure uniqueness
        mask[mask > 0] += max_val
        # Insert the mask into the array
        if is_2d:
            all_masks[start_x:end_x, start_y:end_y] = mask
        else:
            all_masks[start_z:end_z, start_x:end_x, start_y:end_y] = mask
    else:
        # Insert the mask into the array
        if is_2d:
            all_masks[start_x:end_x, start_y:end_y] = mask
        else:
            all_masks[start_z:end_z, start_x:end_x, start_y:end_y] = mask
    return all_masks


def connect_components(all_masks: np.ndarray):
    # Convert to dask array
    all_masks = da.from_array(all_masks)
    # Get the connected components, combining masks from consecutive frames
    labelled, num_holes = dask_image.ndmeasure.label(all_masks)
    labelled = labelled.compute()
    num_holes = int(num_holes)
    # Get the appropriate dtype from the number of holes, and convert to numpy array
    return reduce_dtype(labelled, max_val=num_holes)


@jit(nopython=True, parallel=True, fastmath=True)
def mask_iou_batch(
    box_matches, curr_slice_bool, next_slice_bool, curr_label_dict, next_label_dict
):
    # Initialize the array to store the IoUs
    n = len(box_matches)
    ious = np.zeros(n)
    # Parallel loop over the box matches
    for i in prange(n):
        # Extract the boolean masks for the current and next labels
        curr_label, next_label = box_matches[i]
        curr_mask = curr_slice_bool[..., curr_label_dict[curr_label]]
        next_mask = next_slice_bool[..., next_label_dict[next_label]]
        # Calculate the IoU
        # Inlined here to help numba optimise
        union = np.count_nonzero(np.logical_or(curr_mask, next_mask))
        if union == 0:
            ious[i] = 0.0
        else:
            intersection = np.count_nonzero(np.logical_and(curr_mask, next_mask))
            ious[i] = intersection / union
    return ious


def filter_overlaps(curr_slice, next_slice):
    # Get the bounding boxes for each region in the current and next slices
    rps = skimage.measure.regionprops(curr_slice)
    boxes1 = np.array([rp.bbox for rp in rps])
    labels1 = np.array([rp.label for rp in rps])

    rps = skimage.measure.regionprops(next_slice)
    boxes2 = np.array([rp.bbox for rp in rps])
    labels2 = np.array([rp.label for rp in rps])

    # Check for overlaps between the boxes in the two slices
    box_matches = []

    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            res = check_overlap(box1, box2)
            if res:
                box_matches.append((labels1[i], labels2[j]))
    return box_matches


def check_overlap(box1, box2):
    # Box: [min_row, min_col, max_row, max_col]
    # https://stackoverflow.com/a/40795835
    # We compare x & y coords of bottom-left & top-right corners
    # Bottom-left: min_col (x), max_row (y)
    # Top-right: max_col (x), min_row (y)
    # Note that higher y is lower in the image: (0,0) is top-left
    return not (
        box1[3] < box2[1] or box1[1] > box2[3] or box1[2] < box2[0] or box1[0] > box2[2]
    )


def connect_sam(all_masks, iou_threshold):
    for idx in tqdm(range(all_masks.shape[0] - 1)):
        # Create a matrix to store all combinations of IoUs
        curr_slice = all_masks[idx]
        next_slice = all_masks[idx + 1]

        # Get the unique labels in the current and next slices
        curr_labels = np.unique(curr_slice)
        next_labels = np.unique(next_slice)
        # Get a numba-compatible dictionary for the labels to allow for later indexing
        curr_label_dict = Dict.empty(key_type=types.uint16, value_type=types.uint16)
        next_label_dict = Dict.empty(key_type=types.uint16, value_type=types.uint16)
        curr_label_dict.update(
            {label: np.uint16(i) for i, label in enumerate(curr_labels)}
        )
        next_label_dict.update(
            {label: np.uint16(i) for i, label in enumerate(next_labels)}
        )

        # Restrict to only overlapping boxes
        box_matches = filter_overlaps(curr_slice, next_slice)

        # No matches, skip
        if len(box_matches) > 0:
            # Create boolean masks for each label in the current and next slices
            # Effectively converts (H, W) int array into (H, W, N) boolean where N is the number of labels
            curr_slice_bool = curr_slice[..., None] == curr_labels
            next_slice_bool = next_slice[..., None] == next_labels

            # Calculate IoUs for all pairs of overlapping boxes
            ious = mask_iou_batch(
                box_matches,
                curr_slice_bool,
                next_slice_bool,
                curr_label_dict,
                next_label_dict,
            )
            # Get the max label from the current slice to assign to to ensure no conflict
            max_label = curr_labels.max() + 1
            # Create an array mapping the next labels to the current labels
            mapping_arr = np.full(
                int(next_labels.max() + 1), fill_value=0, dtype=np.uint16
            )
            # Iterate over the matches and check which ones sufficiently overlap
            for iou, (curr_label, next_label) in zip(ious, box_matches):
                # If threshold met, remap label
                if iou >= iou_threshold:
                    mapping_arr[next_label] = curr_label
            # Need to account for all other labels
            for i, val in enumerate(mapping_arr):
                # Fill in the labels that were not matched
                if val == 0:
                    # Skip background
                    if i == 0:
                        continue
                    # Set to the next available label
                    mapping_arr[i] = max_label
                    max_label += 1
            # Remap the labels in the next slice
            # Fancy mapping: https://stackoverflow.com/a/55950051
            all_masks[idx + 1] = mapping_arr[next_slice.copy()]
    # Relabel the masks to get consecutive labels from 1 to N
    (
        all_masks,
        _,
        _,
    ) = relabel_sequential(all_masks)
    return reduce_dtype(all_masks)


def mask_iou(masks1: np.ndarray, masks2: np.ndarray):
    intersection = np.sum(np.logical_and(masks1, masks2))
    union = np.sum(np.logical_or(masks1, masks2))
    if union == 0:
        return 0.0
    else:
        return intersection / union


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-fname", required=True, help="Mask save filename")
    parser.add_argument("--output-dir", required=True, help="Mask output directory")
    parser.add_argument(
        "--masks",
        required=True,
        nargs="+",
        help="Masks to combine",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model used to generate masks",
    )
    parser.add_argument(
        "--image-size",
        nargs=3,
        type=int,
        required=True,
        help="Size of the image stack, in array (i.e. D x H x W) format.",
    )
    parser.add_argument(
        "--overlap",
        required=True,
        nargs=3,
        help="Overlap in each dimension (default is 0). Assumed H x W x D.",
    )
    parser.add_argument(
        "--postprocess",
        required=False,
        action="store_true",
        help="Run postprocessing on the masks",
    )
    parser.add_argument(
        "--iou-threshold",
        required=False,
        type=float,
        default=0.8,
        help="IoU threshold for aligning masks (in SAM)",
    )
    parser.add_argument(
        "--preprocess-params", help="Preprocessing parameters YAML file"
    )

    cli_args = parser.parse_args()

    mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
    print(f"Memory used before loading stack: {mem_used:.2f} GB")
    # Combine the masks
    if len(cli_args.masks) > 1:
        combined_masks = combine_masks(
            cli_args.masks,
            overlap=cli_args.overlap,
            image_size=cli_args.image_size,
            preprocess_params=cli_args.preprocess_params,
        )
        mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
        print(f"Memory used after loading stack: {mem_used:.2f} GB")
    else:
        combined_masks = aiod_rle.load_encoding(cli_args.masks[0])
        # NOTE: Extract metadata later from preprocess params
        combined_masks, _ = aiod_rle.decode(combined_masks)
    print(f"Combined masks shape: {combined_masks.shape}")
    if cli_args.postprocess:
        print("Postprocessing masks...")
        if cli_args.model == "sam" or cli_args.model == "sam2":
            # No need to align over slices if there are none! Labels consecutive already
            if combined_masks.ndim > 2:
                combined_masks = connect_sam(
                    combined_masks, iou_threshold=cli_args.iou_threshold
                )
        else:
            combined_masks = connect_components(combined_masks)
    # Ensure the dtype is always reduced if possible
    # NOTE: Postprocessing funcs above handle this themselves
    else:
        combined_masks = reduce_dtype(combined_masks)
    # Squeeze the array in case there is only one slice
    combined_masks = np.squeeze(combined_masks)
    mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
    print(f"Memory used in combination: {mem_used:.2f} GB")
    # Save the masks
    save_path = Path(cli_args.output_dir) / f"{cli_args.mask_fname}_all.rle"
    # Get downsample factor for metadata if used
    downsample_factor = get_downsample_factor(cli_args.preprocess_params)
    if downsample_factor is not None:
        metadata = {"downsample_factor": downsample_factor}
    else:
        metadata = {}
    encoded_masks = aiod_rle.encode(combined_masks, metadata=metadata)
    # Free up memory (though too late at this point)
    del combined_masks
    # Save the masks
    aiod_rle.save_encoding(rle=encoded_masks, fpath=save_path)
    # Remove the individual masks now that they are combined
    for mask_path in cli_args.masks:
        # Remove the mask
        (Path(cli_args.output_dir) / mask_path).unlink()
