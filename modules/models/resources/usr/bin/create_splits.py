from collections import defaultdict
from pathlib import Path

import pandas as pd

from aiod_utils.stacks import Stack, calc_num_stacks, generate_stack_indices


if __name__ == "__main__":
    # Get the command line arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img-csv", required=True, help="Path to csv file")
    parser.add_argument(
        "--num-substacks",
        required=True,
        nargs=3,
        help="Number of stacks in each dimension (default is 'auto'). Assumed H x W x D.",
    )
    parser.add_argument(
        "--overlap",
        required=True,
        nargs=3,
        help="Overlap in each dimension (default is 0). Assumed H x W x D.",
    )
    parser.add_argument(
        "--output-csv",
        required=False,
        default="all_img_paths.csv",
        type=str,
        help="Output csv file with stack indices",
    )

    args = parser.parse_args()

    # Load the csv file
    img_csv_fpath = Path(args.img_csv)
    img_df = pd.read_csv(img_csv_fpath)

    # Check that the csv has the required columns
    required_columns = ["img_path", "height", "width", "num_slices", "channels"]
    for col in required_columns:
        if col not in img_df.columns:
            raise ValueError(
                f"Column '{col}' not found in input image path csv file ({img_csv_fpath})."
            )

    # Drop the stack info if it exists
    img_df = img_df.drop(
        columns=[
            "stack_idx",
            "start_h",
            "end_h",
            "start_w",
            "end_w",
            "start_d",
            "end_d",
        ],
        errors="ignore",
    )
    # Remove any rows with the same image path (caused by previous runs/expansions)
    img_df = img_df.drop_duplicates(subset=["img_path"])

    new_csv = defaultdict(list)

    # Loop over every image file in the csv
    for idx, row in img_df.iterrows():
        img_path = Path(row["img_path"])
        # Extract the image shape from the row
        img_shape = Stack(
            height=int(row["height"]),
            width=int(row["width"]),
            depth=int(row["num_slices"]),
            channels=int(row["channels"]),
        )
        # Get the requested number of substacks (either int or 'auto' for each dimension)
        num_substacks = Stack(*args.num_substacks)
        # Ensure overlap is a tuple of floats
        overlap_fraction = Stack(*map(float, args.overlap))
        # Calculate the number of stacks and the effective shape
        num_substacks, eff_shape = calc_num_stacks(
            img_shape, num_substacks, overlap_fraction
        )
        # Generate the stack indices
        stack_indices, num_stacks, stack_size = generate_stack_indices(
            image_shape=img_shape,
            num_substacks=num_substacks,
            overlap_fraction=overlap_fraction,
            eff_shape=eff_shape,
        )

        for i, stack in enumerate(stack_indices):
            # Insert all info from the row
            for key, value in row.items():
                new_csv[key].append(value)
            # Add the stack info
            new_csv["stack_idx"].append(i)
            new_csv["start_h"].append(stack[0][0])
            new_csv["end_h"].append(stack[0][1])
            new_csv["start_w"].append(stack[1][0])
            new_csv["end_w"].append(stack[1][1])
            new_csv["start_d"].append(stack[2][0])
            new_csv["end_d"].append(stack[2][1])

    # Overwrite the csv with the new info
    new_csv_df = pd.DataFrame(new_csv)
    new_csv_df.to_csv(Path.cwd() / args.output_csv, index=False)
