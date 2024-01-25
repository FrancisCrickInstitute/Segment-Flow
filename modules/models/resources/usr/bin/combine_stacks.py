from pathlib import Path
import os
import psutil

import dask.array as da
import dask_image.ndmeasure
import numpy as np

from utils import reduce_dtype, align_segment_labels


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

    cli_args = parser.parse_args()

    # Sort masks by index to ensure correct order
    # Our start_idx is the final component of the filename, so we can sort by that
    cli_args.masks.sort(key=lambda x: int(Path(x).stem.split("_")[-1]))
    # Load the masks
    masks = []
    for mask_path in cli_args.masks:
        masks.append(np.load(mask_path))
    # Combine the masks
    if len(masks) > 1:
        combined_masks = np.concatenate(masks)
        if cli_args.model == "sam":
            combined_masks = connect_sam(combined_masks)
        else:
            combined_masks = connect_components(combined_masks)
    else:
        combined_masks = masks[0]
    mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
    print(f"Memory used in combination: {mem_used:.2f} GB")
    # Save the masks
    save_path = Path(cli_args.output_dir) / f"{cli_args.mask_fname}_all.npy"
    np.save(save_path, combined_masks)
    # Remove the individual masks now that they are combined
    for mask_path in cli_args.masks:
        # Remove the mask
        (Path(cli_args.output_dir) / mask_path).unlink()
