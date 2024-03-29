from pathlib import Path
import os
import psutil

import dask.array as da
import dask_image.ndmeasure
import numpy as np
import skimage.measure

from utils import reduce_dtype, align_segment_labels, extract_idxs_from_fname


def combine_masks(
    masks: list[str], overlap: list[float, ...], image_size: tuple[int, ...]
):
    """
    Combine masks from each of the substacks into a single array/dataset.

    If overlap is 0, then the masks are simply inserted into their relevant indices.

    If overlap is >0, then the masks need to be combined.
    """
    # Get the chunk size from the first file
    start_x, end_x, start_y, end_y, start_z, end_z = extract_idxs_from_fname(masks[0])
    chunk_size = (end_x - start_x, end_y - start_y, end_z - start_z)
    # Create the array to hold the masks
    all_masks = np.zeros(image_size, dtype=np.uint16)
    # Loop over each mask and insert into the array
    if all([val == 0 for val in overlap]):
        for mask in masks:
            start_x, end_x, start_y, end_y, start_z, end_z = extract_idxs_from_fname(
                mask
            )
            mask = np.load(mask)
            all_masks[start_z:end_z, start_x:end_x, start_y:end_y] = mask
    else:
        # Combine the masks
        for mask in masks:
            start_x, end_x, start_y, end_y, start_z, end_z = extract_idxs_from_fname(
                mask
            )
            mask = np.load(mask)
            # Just sum, naive method
            all_masks[start_z:end_z, start_x:end_x, start_y:end_y] += mask
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


# def connect_sam(all_masks: np.ndarray):
#     print("Special SAM stuff!")
#     # Align overlapping segment labels
#     all_masks = align_segment_labels(all_masks)
#     return reduce_dtype(all_masks)


def mask_iou(masks1: np.ndarray, masks2: np.ndarray):
    intersection = np.sum(np.logical_and(masks1, masks2))
    union = np.sum(np.logical_or(masks1, masks2))
    return intersection / union


def filter_overlaps(curr_slice, next_slice):
    rps = skimage.measure.regionprops(curr_slice)
    boxes1 = np.array([rp.bbox for rp in rps])
    labels1 = np.array([rp.label for rp in rps])

    rps = skimage.measure.regionprops(next_slice)
    boxes2 = np.array([rp.bbox for rp in rps])
    labels2 = np.array([rp.label for rp in rps])

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
    for idx in range(all_masks.shape[0] - 1):
        # Create a matrix to store all combinations of IoUs
        curr_slice = all_masks[idx]
        next_slice = all_masks[idx + 1]

        # curr_labels = np.unique(curr_slice)
        next_labels = np.unique(next_slice)

        # Store the label mappings
        label_maps = []
        # Get the max label in the next slice to assign new labels
        max_label = next_labels.max() + 1

        # Restrict to only overlapping boxes
        box_matches = filter_overlaps(curr_slice, next_slice)

        # Iterate over the matches and check which ones sufficiently overlap
        for curr_label, next_label in box_matches:
            # Get the current and next masks
            curr_mask = curr_slice == curr_label
            next_mask = next_slice == next_label
            # Compute the IOU
            iou = mask_iou(curr_mask, next_mask)
            # If threshold met, remap label
            if iou > iou_threshold:
                # Skip if it's already the same label
                if curr_label == next_label:
                    continue
                # If new label already present, map it to next available label
                if curr_label in next_labels:
                    label_maps.append((curr_label, max_label))
                    max_label += 1
                # Otherwise, map the label in the next slice to the value in the current
                label_maps.append((next_label, curr_label))
        # Remap the labels if matches found
        if label_maps:
            new_next_slice = next_slice.copy()
            for new_label, next_label in label_maps:
                new_next_slice[next_slice == new_label] = next_label
            # NOTE: This makes next_slice now point to new_next_slice
            all_masks[idx + 1] = new_next_slice
    return reduce_dtype(all_masks)


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

    cli_args = parser.parse_args()

    mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
    print(f"Memory used before loading stack: {mem_used:.2f} GB")
    # Combine the masks
    if len(cli_args.masks) > 1:
        combined_masks = combine_masks(
            cli_args.masks, overlap=cli_args.overlap, image_size=cli_args.image_size
        )
        mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
        print(f"Memory used after loading stack: {mem_used:.2f} GB")
    else:
        combined_masks = np.load(cli_args.masks[0])
    print(f"Combined masks shape: {combined_masks.shape}")
    if cli_args.postprocess:
        print("Postprocessing masks...")
        if cli_args.model == "sam":
            combined_masks = connect_sam(
                combined_masks, iou_threshold=cli_args.iou_threshold
            )
        else:
            combined_masks = connect_components(combined_masks)
    mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
    print(f"Memory used in combination: {mem_used:.2f} GB")
    # Save the masks
    save_path = Path(cli_args.output_dir) / f"{cli_args.mask_fname}_all.npy"
    np.save(save_path, combined_masks)
    # Remove the individual masks now that they are combined
    for mask_path in cli_args.masks:
        # Remove the mask
        (Path(cli_args.output_dir) / mask_path).unlink()
