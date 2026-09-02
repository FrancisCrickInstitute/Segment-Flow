"""
For this first version, where we will implement this new pipeline for preprocessing sets, we will just use the existing functions and run them over the whole image naively without Dask/chunk thoughts
"""

from pathlib import Path

import numpy as np
import pandas as pd
from aiod_utils.io import get_mask_prefix, load_image_data, save_image
from aiod_utils.preprocess import (
    get_params_str,
    get_prep_hash,
    load_methods,
    run_preprocess,
)
from utils import DEFAULT_DIM_ORDER as DIM_ORDER
from utils import read_img_csv


def construct_fname(image_id, prep_hash):
    # image_id + a short hash of the preprocessing params (not the raw
    # string) keeps names short and stops them accumulating a fresh
    # extension-like segment per pipeline stage. Always output OME-Zarr.
    # Same identity + prep rule the mask names use, so both stay in step.
    return Path(f"{get_mask_prefix(image_id, prep_hash)}.ome.zarr")


def save_preprocessed_image(image_id, prep_hash, prep_image, save_dims):
    fname = construct_fname(image_id, prep_hash)
    save_image(prep_image, fname, dim_order=save_dims)
    return fname


if __name__ == "__main__":
    import argparse

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Preprocess an image")
    parser.add_argument("--img-path", required=True, type=str, help="Path to image")
    # NOTE: We require the params here, as this step will not be run if not
    parser.add_argument(
        "--preprocess-params", required=True, help="Preprocessing parameters YAML file"
    )
    parser.add_argument("--img-csv", required=True, help="Path to csv file")
    args = parser.parse_args()

    # Read image CSV, and filter only to the file for this process
    csv_path = Path(args.img_csv)
    df_img = read_img_csv(csv_path)
    # Reconstruct full path and match with DF to only get the row for this image
    df_img["img_path"] = df_img["img_path"].apply(lambda x: Path(x).name)
    df_img = df_img.loc[df_img.img_path == args.img_path]
    if len(df_img) == 0:
        raise ValueError(f"No matching image found in CSV for {args.img_path}")
    elif len(df_img) > 1:
        raise ValueError(f"Multiple matching images found in CSV for {args.img_path}")
    # Preprocess the image
    # TODO: Switch to return_dask, map over blocks, and check output as described at top
    # Load with explicit CZYX ordering so axis identity is preserved for all image types,
    # including RGB images where the S (samples) dimension is mapped to C, giving (C, Z, H, W).
    image_4d = load_image_data(args.img_path, dim_order=DIM_ORDER)
    # Record which axes are singleton before squeezing so we can reconstruct CZYX afterwards
    squeezed_axes = [i for i, s in enumerate(image_4d.shape) if s == 1]
    image = image_4d.squeeze()
    # Extract all preprocessing sets (except empty no-ops)
    preprocess_methods = load_methods(args.preprocess_params, filter_noop=True)
    # Create a new dataframe to store the new images, repeating rows per preprocessing set
    df_new = pd.concat([df_img.copy()] * len(preprocess_methods), ignore_index=True)
    # Loop over each set and preprocess
    # Derive the dim_order that matches the squeezed data
    save_dims = "".join(d for i, d in enumerate(DIM_ORDER) if i not in squeezed_axes)
    # Retrieve image_id for naming
    image_id = df_img["image_id"].iloc[0]
    # Container to store the hash with the preprocessing params for logging
    legend_lines = []
    for i, preprocess_dict in enumerate(preprocess_methods):
        prep_image = run_preprocess(image, methods=preprocess_dict, parse=False)
        prep_hash = get_prep_hash(preprocess_dict)
        # Get the new filename, identified by a short hash of the params
        fname = save_preprocessed_image(
            image_id=image_id,
            prep_hash=prep_hash,
            prep_image=prep_image,
            save_dims=save_dims,
        )
        # Update the dataframe with the new image path, ensuring full path given
        df_new.loc[i, "img_path"] = fname
        # Embed image_id/prep_hash to use throughout for branching etc.
        df_new.loc[i, "image_id"] = image_id
        df_new.loc[i, "prep_hash"] = prep_hash
        legend_lines.append(
            f"[{prep_hash}] {get_params_str(preprocess_dict, to_save=False)}"
        )
        # Update shape info in the dataframe if downsampled/modified
        if image.shape != prep_image.shape:
            # Re-insert the singleton axes that were squeezed out to restore CZYX identity.
            # This is critical for images like RGB (C=3, Z=1) where squeeze gives (C, H, W):
            # without re-expansion, axis 0 (C) would be misread as num_slices.
            prep_4d = (
                np.expand_dims(prep_image, axis=squeezed_axes)
                if squeezed_axes
                else prep_image
            )
            # CZYX order is always: axis0=channels, axis1=num_slices, axis2=height, axis3=width
            _, new_slices, new_height, new_width = prep_4d.shape
            df_new.loc[i, "num_slices"] = new_slices
            df_new.loc[i, "height"] = new_height
            df_new.loc[i, "width"] = new_width
    # Save the new dataframe, matching the img_csv glob in modules/models/main.nf
    df_new.to_csv(f"{image_id}.csv", index=False)
    # Write the hashes to a file so we can log it with Nextflow instead of printing
    # NOTE: I thought logging with Nxf would avoid stdout eating, ensures
    # it's in the nextflow.log and not reliant on debug=true
    Path("preprocess_hashes.txt").write_text("\n".join(legend_lines) + "\n")
