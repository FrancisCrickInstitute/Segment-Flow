from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# TODO: These constants vary by GPU/available memory, and also scale by number of channels which is not considered here
MAX_TILE_SIZE = {
    "height": 3000,
    "width": 3000,
    "depth": 100,
}


def auto_tile_size(image_shape: tuple[int, ...], num_tiles: tuple[int, ...]):
    """
    Calculate the number of tiles to use for a given image size, for any dimension specified as 'auto'.

    Assumes the image shape is a tuple of integers representing the dimensions of the image (e.g. (H, W, D)).

    This uses the constants defined at the top of script to create chunks for each job within a given size, keeping
    down the size (but increasing number) of the submitted jobs to avoid memory issues.
    """
    # Extract the dimensions
    height, width, depth = image_shape
    # Extract the number of tiles
    x_tiles, y_tiles, z_tiles = num_tiles
    if x_tiles == "auto":
        x_tiles = (height // MAX_TILE_SIZE["height"]) + 1
    else:
        x_tiles = check_sensible_num_tiles(int(x_tiles), height, dim="height")
    if y_tiles == "auto":
        y_tiles = (width // MAX_TILE_SIZE["width"]) + 1
    else:
        y_tiles = check_sensible_num_tiles(int(y_tiles), width, dim="width")
    if z_tiles == "auto":
        z_tiles = (depth // MAX_TILE_SIZE["depth"]) + 1
    else:
        z_tiles = check_sensible_num_tiles(int(z_tiles), depth, dim="depth")
    return (x_tiles, y_tiles, z_tiles)


def check_sensible_num_tiles(num_tiles_dim: int, dim_size: int, dim: str):
    """
    Make sure the input number of tiles is sensible for the given dimension size.
    """
    # Make sure the number of tiles is at least 1
    if num_tiles_dim < 1:
        return 1
    # More tiles than pixels? Use auto method
    if num_tiles_dim > dim_size:
        return (dim_size // MAX_TILE_SIZE[dim]) + 1
    # Make sure it's not too small, defined as 1% of the max tile size
    if (dim_size // num_tiles_dim) < (MAX_TILE_SIZE[dim] / 100):
        # NOTE: We are just ignoring the user here
        # We could default to MAX_TILE_SIZE[dim] / 100 as a lower bound
        # But I think I'd rather just set our default tile size and ignore it!
        return (dim_size // MAX_TILE_SIZE[dim]) + 1
    # Make sure it's not too big, defined as 5x the max tile size
    if (dim_size // num_tiles_dim) > (MAX_TILE_SIZE[dim] * 5):
        # NOTE: Again we are just ignoring the user here
        # We could default to MAX_TILE_SIZE[dim] * 5 as an upper bound...
        return (dim_size // MAX_TILE_SIZE[dim]) + 1
    return num_tiles_dim


def generate_patch_indices(
    image_shape: tuple[int, ...],
    num_tiles: tuple[int, ...],
    overlap_fraction: tuple[float, ...],
) -> list[tuple[tuple[int, int], ...]]:
    """
    Generate the indices for every patch for a given image size, desired number of tiles, and overlap fraction.

    Note that the overlap fraction is a float between 0 and 1, and the number of tiles is a tuple of integers, both of which should represent the same number of dimensions and meaning of the image_shape, which is expected to be a tuple of integers representing the dimensions of the image (e.g. (H, W, D)).

    Also note that the output is not guaranteed to completely satisfy the given arguments, as it may not be satisfiable. In this case, the overlap created will be different, but the number of tiles is guaranteed.
    """
    # Extract the dimensions
    # TODO: Handle the case where the image is 2D
    # TODO: Handle the case where the image is 4D
    # TODO: Handle the image shape being e.g. D x H x W instead of H x W x D
    # In the above, we can probably assume D is minimum size
    height, width, depth = image_shape
    num_tiles_height, num_tiles_width, num_tiles_depth = num_tiles
    overlap_fraction_height, overlap_fraction_width, overlap_fraction_depth = (
        overlap_fraction
    )
    # The effective size of the image after multiply by the overlap added
    # This helps create the appropriate patch size, amd we later create the overlap with offset
    eff_height = round(height * (1 + overlap_fraction_height))
    eff_width = round(width * (1 + overlap_fraction_width))
    eff_depth = round(depth * (1 + overlap_fraction_depth))

    # Calculate the patch size based on the number of tiles
    patch_height = eff_height // num_tiles_height
    patch_width = eff_width // num_tiles_width
    patch_depth = eff_depth // num_tiles_depth

    # Calculate overlap size based on fraction
    overlap_height = (
        0
        if overlap_fraction_height == 0
        else max(int(patch_height * overlap_fraction_height), 1)
    )
    overlap_width = (
        0
        if overlap_fraction_width == 0
        else max(int(patch_width * overlap_fraction_width), 1)
    )
    overlap_depth = (
        0
        if overlap_fraction_depth == 0
        else max(int(patch_depth * overlap_fraction_depth), 1)
    )

    # Generate indices for the patches
    patch_indices = []

    for i in range(num_tiles_height):
        for j in range(num_tiles_width):
            for k in range(num_tiles_depth):
                start_h = i * (patch_height - overlap_height)
                end_h = (
                    min(start_h + patch_height, height)
                    if i < max(num_tiles_height, 1) - 1
                    else height
                )
                start_w = j * (patch_width - overlap_width)
                end_w = (
                    min(start_w + patch_width, width)
                    if j < max(num_tiles_width, 1) - 1
                    else width
                )
                start_d = k * (patch_depth - overlap_depth)
                end_d = (
                    min(start_d + patch_depth, depth)
                    if k < max(num_tiles_depth, 1) - 1
                    else depth
                )

                patch_indices.append(
                    ((start_h, end_h), (start_w, end_w), (start_d, end_d))
                )

    return patch_indices, len(patch_indices)


if __name__ == "__main__":
    # Get the command line arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img-csv", required=True, help="Path to csv file")
    parser.add_argument(
        "--num-tiles",
        required=True,
        nargs=3,
        help="Number of tiles in each dimension (default is 'auto'). Assumed H x W x D.",
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
        help="Output csv file with patch indices",
    )

    args = parser.parse_args()

    # Load the csv file
    img_csv_fpath = Path(args.img_csv)
    img_df = pd.read_csv(img_csv_fpath)

    # Drop the patch info if it exists
    img_df = img_df.drop(
        columns=[
            "patch_idx",
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
        img_shape = (row["height"], row["width"], row["num_slices"])
        # Ensure the image shape is a tuple of integers
        img_shape = tuple([int(val) for val in img_shape])
        num_tiles = auto_tile_size(
            img_shape,
            args.num_tiles,
        )
        # Ensure overlap is a tuple of floats
        overlap_fraction = tuple([float(val) for val in args.overlap])

        # Generate the patch indices
        patch_indices, _ = generate_patch_indices(
            img_shape,
            num_tiles=num_tiles,
            overlap_fraction=overlap_fraction,
        )

        for i, patch in enumerate(patch_indices):
            # Insert all info from the row
            for key, value in row.items():
                new_csv[key].append(value)
            # Add the patch info
            new_csv["patch_idx"].append(i)
            new_csv["start_h"].append(patch[0][0])
            new_csv["end_h"].append(patch[0][1])
            new_csv["start_w"].append(patch[1][0])
            new_csv["end_w"].append(patch[1][1])
            new_csv["start_d"].append(patch[2][0])
            new_csv["end_d"].append(patch[2][1])

    # Overwrite the csv with the new info
    new_csv_df = pd.DataFrame(new_csv)
    new_csv_df.to_csv(Path.cwd() / args.output_csv, index=False)
