from collections import defaultdict
from pathlib import Path

import pandas as pd
from aiod_utils.io import load_image
from aiod_utils.stacks import (
    MAX_SUBSTACK_SIZE,
    Stack,
    calc_num_stacks,
    compute_max_substack_size,
    generate_stack_indices,
)

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
    parser.add_argument(
        "--memory-per-job",
        required=False,
        default=None,
        type=int,
        help="Memory available per runModel job in bytes (used to compute substack size dynamically).",
    )

    def _cap_value(s: str) -> int | None:
        """Parse a --max-substack token: 'null'/'none'/'' -> None (no cap), else int."""
        return None if s.strip().lower() in ("null", "none", "") else int(s)

    parser.add_argument(
        "--max-substack",
        required=False,
        default=None,
        nargs=3,
        type=_cap_value,
        metavar=("H", "W", "D"),
        help=(
            "Per-dimension compute cap on substack size (H W D). Bounds per-job wall time "
            "(D/slices is the dominant factor). Element-wise min'd with the memory-derived size. "
            "A value of 'null' leaves that axis uncapped (deferred to the memory budget / whole image) - e.g. 'null null 8' caps depth only. "
            "If the whole flag is omitted, only the memory cap applies."
        ),
    )

    args = parser.parse_args()

    def _min_dim(a: int | None, b: int | None) -> int | None:
        """min of two per-axis caps; None means 'no cap on this axis' -> take the other."""
        if a is None:
            return b
        if b is None:
            return a
        return min(a, b)

    def min_stack(a: Stack, b: Stack) -> Stack:
        """Element-wise min of two caps; None on an axis = uncapped (keep the other)."""
        return Stack(
            height=_min_dim(a.height, b.height),
            width=_min_dim(a.width, b.width),
            depth=_min_dim(a.depth, b.depth),
            channels=a.channels if a.channels is not None else b.channels,
        )

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
    fetch_dtype = "dtype" not in img_df.columns

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
    for _idx, row in img_df.iterrows():
        img_path = Path(row["img_path"])
        # Extract the image shape from the row
        img_shape = Stack(
            height=int(row["height"]),
            width=int(row["width"]),
            depth=int(row["num_slices"]),
            channels=int(row["channels"]),
        )
        # TODO modify napari csv export to include dtype column
        # NOTE: If BioImage.dtype fails, will it return None? What are the possible failures here
        if fetch_dtype:
            try:
                img_dtype = str(load_image(img_path).dtype)
            except Exception:  # noqa: BLE001
                img_dtype = "float32"  # conservative fallback
        else:
            img_dtype = (
                str(row["dtype"]) if pd.notna(row["dtype"]) else "float32"
            )  # conservative fallback
        # Compute the maximum substack size
        # Dynamic if memory_per_job provided, else use constant
        # Two independent constraints bound the substack size:
        #   * memory  ("will it fit in memory?")  -> compute_max_substack_size
        #   * compute ("will it finish in time?") -> the per-model --max-substack cap
        # Calc compute cap first, to keep substack H/W large if poss
        compute_cap = (
            Stack(*args.max_substack, channels=img_shape.channels)
            if args.max_substack is not None
            else None
        )
        # Only when memory still binds after compute-capping do both caps apply (element-wise min of the two).
        # compute_max_substack_size returns its input shape unchanged iff that shape fits the budget
        if args.memory_per_job is not None:
            if compute_cap is None:
                max_substack_size = compute_max_substack_size(
                    memory_bytes=args.memory_per_job,
                    dtype=img_dtype,
                    image_shape=img_shape,
                )
            else:
                capped_shape = min_stack(img_shape, compute_cap)
                fits_memory = (
                    compute_max_substack_size(
                        memory_bytes=args.memory_per_job,
                        dtype=img_dtype,
                        image_shape=capped_shape,
                    )
                    == capped_shape
                )
                if fits_memory:
                    max_substack_size = capped_shape
                else:
                    mem_cap = compute_max_substack_size(
                        memory_bytes=args.memory_per_job,
                        dtype=img_dtype,
                        image_shape=img_shape,
                    )
                    max_substack_size = min_stack(mem_cap, compute_cap)
        elif compute_cap is not None:
            max_substack_size = min_stack(MAX_SUBSTACK_SIZE, compute_cap)
        else:
            max_substack_size = MAX_SUBSTACK_SIZE
        # Get the requested number of substacks (either int or 'auto' for each dimension)
        num_substacks = Stack(*args.num_substacks)
        # Ensure overlap is a tuple of floats
        overlap_fraction = Stack(*map(float, args.overlap))
        # Calculate the number of stacks and the effective shape
        num_substacks, eff_shape = calc_num_stacks(
            img_shape, num_substacks, overlap_fraction, max_substack_size
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
            # Ensure a dtype column exists downstream (used to calibrate memory).
            # Only add it here if the input CSV didn't already carry one (else row.items() did).
            if fetch_dtype:
                new_csv["dtype"].append(img_dtype)
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
