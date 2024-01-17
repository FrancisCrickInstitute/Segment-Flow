from pathlib import Path
import os
import psutil

import numpy as np

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

    cli_args = parser.parse_args()

    # Sort masks by index to ensure correct order
    cli_args.masks.sort(key=lambda x: int(Path(x).stem.split("_")[-1]))
    # Load the masks
    masks = []
    for mask_path in cli_args.masks:
        masks.append(np.load(mask_path))
    # Combine the masks
    if len(masks) > 1:
        combined_masks = np.concatenate(masks)
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
