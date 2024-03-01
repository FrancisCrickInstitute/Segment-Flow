from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# TODO: These constants vary by GPU/available memory, and also scale by number of channels which is not considered here
# Named substack to avoid patch/tile confusion, and to avoid chunk confusion
MAX_SUBSTACK_SIZE = {
    "height": 3000,
    "width": 3000,
    "depth": 100,
}


def auto_substack_size(image_shape: tuple[int, ...], num_substacks: tuple[int, ...]):
    """
    Calculate the number of substacks to use for a given image size, for any dimension specified as 'auto'.

    Assumes the image shape is a tuple of integers representing the dimensions of the image (D, H, W).

    This uses the constants defined at the top of script to create chunks for each job within a given size, keeping
    down the size (but increasing number) of the submitted jobs to avoid memory issues.
    """
    # Extract the dimensions
    depth, height, width = image_shape
    # Extract the number of substacks
    x_substacks, y_substacks, z_substacks = num_substacks
    if x_substacks == "auto":
        x_substacks = (height // MAX_SUBSTACK_SIZE["height"]) + 1
    else:
        x_substacks = check_sensible_num_substacks(
            int(x_substacks), height, dim="height"
        )
    if y_substacks == "auto":
        y_substacks = (width // MAX_SUBSTACK_SIZE["width"]) + 1
    else:
        y_substacks = check_sensible_num_substacks(int(y_substacks), width, dim="width")
    if z_substacks == "auto":
        z_substacks = (depth // MAX_SUBSTACK_SIZE["depth"]) + 1
    else:
        z_substacks = check_sensible_num_substacks(int(z_substacks), depth, dim="depth")
    return (x_substacks, y_substacks, z_substacks)


def check_sensible_num_substacks(num_stacks_dim: int, dim_size: int, dim: str):
    """
    Make sure the input number of substacks is sensible for the given dimension size.
    """
    # Make sure the number of stacks is at least 1
    if num_stacks_dim < 1:
        num_stacks_dim = 1
    # More stacks than pixels? Use auto method
    if num_stacks_dim > dim_size:
        return (dim_size // MAX_SUBSTACK_SIZE[dim]) + 1
    # Make sure it's not too small, defined as 1% of the max stacks size
    if (dim_size // num_stacks_dim) < (MAX_SUBSTACK_SIZE[dim] / 100):
        # NOTE: We are just ignoring the user here
        # We could default to MAX_SUBSTACK_SIZE[dim] / 100 as a lower bound
        # But I think I'd rather just set our default stacks size and ignore it!
        return (dim_size // MAX_SUBSTACK_SIZE[dim]) + 1
    # Make sure it's not too big, defined as 5x the max stacks size
    if (dim_size // num_stacks_dim) > (MAX_SUBSTACK_SIZE[dim] * 5):
        # NOTE: Again we are just ignoring the user here
        # We could default to MAX_SUBSTACK_SIZE[dim] * 5 as an upper bound...
        return (dim_size // MAX_SUBSTACK_SIZE[dim]) + 1
    return num_stacks_dim


def generate_stack_indices(
    image_shape: tuple[int, ...],
    num_substacks: tuple[int, ...],
    overlap_fraction: tuple[float, ...],
) -> list[tuple[tuple[int, int], ...]]:
    """
    Generate the indices for every stack for a given image size, desired number of substacks, and overlap fraction.

    Note that the overlap fraction is a float between 0 and 1, and the number of substacks is a tuple of integers, both of which should represent the same number of dimensions and meaning of the image_shape, which is expected to be a tuple of integers representing the dimensions of the image (D, H, W).

    Also note that the output is not guaranteed to completely satisfy the given arguments, as it may not be satisfiable. In this case, the overlap created will be different, but the number of substacks is guaranteed.
    """
    # Extract the dimensions
    # TODO: Handle the case where the image is 2D
    # TODO: Handle the case where the image is 4D
    # TODO: Handle the image shape being e.g. H x W x D instead of D x H x W
    # In the above, we can probably assume D is minimum size
    depth, height, width = image_shape
    num_substacks_height, num_substacks_width, num_substacks_depth = num_substacks
    overlap_fraction_height, overlap_fraction_width, overlap_fraction_depth = (
        overlap_fraction
    )
    # The effective size of the image after multiply by the overlap added
    # This helps create the appropriate stack size, amd we later create the overlap with offset
    # This only counts if we have more than 1 substack in that dimension
    eff_height = (
        round(height * (1 + overlap_fraction_height))
        if num_substacks_height > 1
        else height
    )
    eff_width = (
        round(width * (1 + overlap_fraction_width))
        if num_substacks_width > 1
        else width
    )
    eff_depth = (
        round(depth * (1 + overlap_fraction_depth))
        if num_substacks_depth > 1
        else depth
    )
    # Calculate the stack size based on the number of substacks
    stack_height = eff_height // num_substacks_height
    stack_width = eff_width // num_substacks_width
    stack_depth = eff_depth // num_substacks_depth

    # Calculate overlap size based on fraction
    overlap_height = (
        0
        if overlap_fraction_height == 0
        else max(int(stack_height * overlap_fraction_height), 1)
    )
    overlap_width = (
        0
        if overlap_fraction_width == 0
        else max(int(stack_width * overlap_fraction_width), 1)
    )
    overlap_depth = (
        0
        if overlap_fraction_depth == 0
        else max(int(stack_depth * overlap_fraction_depth), 1)
    )

    # Generate indices for the substacks
    stack_indices = []

    for i in range(num_substacks_height):
        for j in range(num_substacks_width):
            for k in range(num_substacks_depth):
                start_h = i * (stack_height - overlap_height)
                end_h = (
                    min(start_h + stack_height, height)
                    if i < max(num_substacks_height, 1) - 1
                    else height
                )
                start_w = j * (stack_width - overlap_width)
                end_w = (
                    min(start_w + stack_width, width)
                    if j < max(num_substacks_width, 1) - 1
                    else width
                )
                start_d = k * (stack_depth - overlap_depth)
                end_d = (
                    min(start_d + stack_depth, depth)
                    if k < max(num_substacks_depth, 1) - 1
                    else depth
                )

                stack_indices.append(
                    ((start_h, end_h), (start_w, end_w), (start_d, end_d))
                )
    return (
        stack_indices,
        len(stack_indices),
        (stack_depth, stack_height, stack_width),
    )


def check_sizes(stack_indices):
    "Quick check to see if the stack sizes are all the same."
    sizes = []
    for stack in stack_indices:
        stack_size = 0
        for stack_dim in stack:
            stack_size += stack_dim[1] - stack_dim[0]
        sizes.append(stack_size)
    if len(set(sizes)) == 1:
        return True
    else:
        return False


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
        img_shape = (row["num_slices"], row["height"], row["width"])
        # Ensure the image shape is a tuple of integers
        img_shape = tuple([int(val) for val in img_shape])
        num_substacks = auto_substack_size(
            img_shape,
            args.num_substacks,
        )
        # Ensure overlap is a tuple of floats
        overlap_fraction = tuple([float(val) for val in args.overlap])

        # Generate the stack indices
        stack_indices, num_stacks, stack_size = generate_stack_indices(
            img_shape,
            num_substacks=num_substacks,
            overlap_fraction=overlap_fraction,
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
