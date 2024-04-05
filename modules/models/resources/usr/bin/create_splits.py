from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Union
import warnings

import pandas as pd

# TODO: Handle the case where the image is 2D (slice) or 3D (slice with channels)
# TODO: Handle the case where the image is 4D (volume + channels)
# TODO: Handle the case where the image is 5D (volume + time + channels)
Stack = namedtuple("Stack", ["height", "width", "depth"])
# TODO: These constants vary by GPU/available memory, and also scale by number of channels which is not considered here
# Named substack to avoid patch/tile confusion, and to avoid chunk confusion
MAX_SUBSTACK_SIZE = Stack(
    height=5000,
    width=5000,
    depth=50,
)


def auto_size(size: int, max_size: Union[int, float]) -> int:
    """
    Calculate the number of stacks to use for a given size, based on a maximum size.
    """
    num_stacks = size // max_size
    # If the size is less than the max size, we still want to use 1 stack
    if num_stacks == 0:
        return 1
    # We want to ensure that max_size is not exceeded
    if (size / num_stacks) > max_size:
        num_stacks += 1
    return int(num_stacks)


def calc_num_stacks_dim(
    dim_size: int, req_stacks_dim: int, overlap: float, dim: str
) -> tuple[int, int]:
    """
    Determine the number of stacks to use for a given dimension, based on the size of the dimension, the number of stacks requested, the overlap fraction requested, and our maximum size constraints set at the top.
    """
    # Calculate the effective size of the image after multiplying by the added overlap
    # This is ignored/irrelevant if only 1 stack is requested or if the overlap is 0 respectively
    eff_size = round(dim_size * (1 + overlap))
    if req_stacks_dim == "auto":
        num_stacks_dim = auto_size(eff_size, getattr(MAX_SUBSTACK_SIZE, dim))
        if num_stacks_dim == 1 and overlap > 0:
            warnings.warn(
                f"'Auto' determined only 1 stack is needed, but overlap is set to {overlap}. This will result in overlap being ignored."
            )
    else:
        num_stacks_dim = int(req_stacks_dim)
    # Check that whatever num_stacks_dim set is sensible
    # Ignore zero or negative numbers
    if num_stacks_dim < 1:
        num_stacks_dim = 1
    # More stacks than pixels? Use auto method
    if num_stacks_dim > eff_size:
        num_stacks_dim = auto_size(eff_size, getattr(MAX_SUBSTACK_SIZE, dim))
    # Make sure it's not too small, defined as 5% of the max stacks size
    elif (eff_size // num_stacks_dim) < (getattr(MAX_SUBSTACK_SIZE, dim) / 20):
        # NOTE: We are just ignoring the user here
        # We could default to getattr(MAX_SUBSTACK_SIZE, dim) / 20 as a lower bound
        # But I think I'd rather just set our default stacks size and ignore it!
        num_stacks_dim = auto_size(eff_size, getattr(MAX_SUBSTACK_SIZE, dim))
    # Make sure it's not too big, defined as 3x the max stacks size
    elif (eff_size // num_stacks_dim) > (getattr(MAX_SUBSTACK_SIZE, dim) * 3):
        # NOTE: Again we are just ignoring the user here
        # We could default to getattr(MAX_SUBSTACK_SIZE, dim) * 3 as an upper bound...
        num_stacks_dim = auto_size(eff_size, getattr(MAX_SUBSTACK_SIZE, dim))
    # If num_stacks_dim is 1, and overlap is >0, warn the user
    if num_stacks_dim == 1 and overlap > 0:
        warnings.warn(
            f"You asked for 1 {dim} stack, but overlap is set to {overlap}. This will result in overlap being ignored."
        )
    return num_stacks_dim, eff_size


def calc_num_stacks(
    image_shape: Stack, req_stacks: Stack, overlap_fraction: Stack
) -> tuple[Stack, Stack]:
    num_stacks_height, eff_height = calc_num_stacks_dim(
        image_shape.height, req_stacks.height, overlap_fraction.height, "height"
    )
    num_stacks_width, eff_width = calc_num_stacks_dim(
        image_shape.width, req_stacks.width, overlap_fraction.width, "width"
    )
    num_stacks_depth, eff_depth = calc_num_stacks_dim(
        image_shape.depth, req_stacks.depth, overlap_fraction.depth, "depth"
    )
    num_stacks = Stack(
        height=num_stacks_height,
        width=num_stacks_width,
        depth=num_stacks_depth,
    )
    eff_shape = Stack(
        height=eff_height,
        width=eff_width,
        depth=eff_depth,
    )
    return num_stacks, eff_shape


def generate_stack_indices(
    image_shape: Stack,
    num_substacks: Stack,
    overlap_fraction: Stack,
    eff_shape: Stack,
) -> tuple[list[tuple[tuple[int, int], ...]], int, Stack]:
    """
    Generate the indices for every stack for a given image size, desired number of substacks, and overlap fraction.

    Note that the overlap fraction is a float between 0 and 1, and the number of substacks is a tuple of integers, both of which should represent the same number of dimensions and meaning of the image_shape, which is expected to be a tuple of integers representing the dimensions of the image (D, H, W).

    Also note that the output is not guaranteed to completely satisfy the given arguments, as it may not be satisfiable. In this case, the overlap created will be different, but the number of substacks is guaranteed.
    """
    # Calculate the stack size based on the number of substacks
    stack_height = eff_shape.height // num_substacks.height
    stack_width = eff_shape.width // num_substacks.width
    stack_depth = eff_shape.depth // num_substacks.depth

    # Calculate overlap size based on fraction
    overlap_height = (
        0
        if overlap_fraction.height == 0
        else max(int(stack_height * overlap_fraction.height), 1)
    )
    overlap_width = (
        0
        if overlap_fraction.width == 0
        else max(int(stack_width * overlap_fraction.width), 1)
    )
    overlap_depth = (
        0
        if overlap_fraction.depth == 0
        else max(int(stack_depth * overlap_fraction.depth), 1)
    )

    # Generate indices for the substacks
    stack_indices = []

    for i in range(num_substacks.height):
        for j in range(num_substacks.width):
            for k in range(num_substacks.depth):
                start_h = i * (stack_height - overlap_height)
                end_h = (
                    min(start_h + stack_height, image_shape.height)
                    if i < max(num_substacks.height, 1) - 1
                    else image_shape.height
                )
                start_w = j * (stack_width - overlap_width)
                end_w = (
                    min(start_w + stack_width, image_shape.width)
                    if j < max(num_substacks.width, 1) - 1
                    else image_shape.width
                )
                start_d = k * (stack_depth - overlap_depth)
                end_d = (
                    min(start_d + stack_depth, image_shape.depth)
                    if k < max(num_substacks.depth, 1) - 1
                    else image_shape.depth
                )

                stack_indices.append(
                    ((start_h, end_h), (start_w, end_w), (start_d, end_d))
                )
    return (
        stack_indices,
        len(stack_indices),
        Stack(stack_height, stack_width, stack_depth),
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
        img_shape = Stack(
            height=int(row["height"]),
            width=int(row["width"]),
            depth=int(row["num_slices"]),
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
