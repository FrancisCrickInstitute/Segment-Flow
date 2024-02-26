from pathlib import Path
import os
import psutil

import dask.array as da
import dask_image.ndmeasure
import numpy as np

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


def connect_sam(all_masks: np.ndarray):
    # Align overlapping segment labels
    all_masks = align_segment_labels(all_masks)
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
        help="Size of the image stack",
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

    cli_args = parser.parse_args()

    # Load the masks
    masks = []
    mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
    print(f"Memory used before loading stack: {mem_used:.2f} GB")
    for mask_path in cli_args.masks:
        masks.append(np.load(mask_path))
    # Combine the masks
    if len(masks) > 1:
        combined_masks = combine_masks(
            masks, overlap=cli_args.overlap, image_size=cli_args.image_size
        )
        mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
        print(f"Memory used after loading stack: {mem_used:.2f} GB")
    else:
        combined_masks = masks[0]
    print(f"Combined masks shape: {combined_masks.shape}")
    if cli_args.postprocess:
        print("Postprocessing masks...")
        if cli_args.model == "sam":
            combined_masks = connect_sam(combined_masks)
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
