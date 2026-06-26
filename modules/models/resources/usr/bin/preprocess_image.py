"""
For this first version, where we will implement this new pipeline for preprocessing sets, we will just use the existing functions and run them over the whole image naively without Dask/chunk thoughts
"""

from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io
from aiod_utils.io import load_image_data
from aiod_utils.preprocess import get_params_str, load_methods, run_preprocess


def construct_fname(img_path, preprocess_params):
    suffix = get_params_str(preprocess_params, to_save=True)
    img_path = Path(img_path)
    return f"{img_path.stem}_{suffix}{img_path.suffix}"


def save_preprocessed_image(img_path, preprocess_params, prep_image):
    fname = construct_fname(img_path, preprocess_params)
    # Save the image
    # TODO: Switch over to bioio when fully sorted, standardising to OME-TIFF internally within Segment-Flow
    skimage.io.imsave(fname, prep_image)
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
    df_img = pd.read_csv(csv_path)
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
    image_4d = load_image_data(args.img_path, dim_order="CZYX")
    # Record which axes are singleton before squeezing so we can reconstruct CZYX afterwards
    squeezed_axes = [i for i, s in enumerate(image_4d.shape) if s == 1]
    image = image_4d.squeeze()
    # Extract all preprocessing sets (except empty no-ops)
    preprocess_methods = load_methods(args.preprocess_params, filter_noop=True)
    # Create a new dataframe to store the new images, repeating rows per preprocessing set
    df_new = pd.concat([df_img.copy()] * len(preprocess_methods), ignore_index=True)
    # Loop over each set and preprocess
    for i, preprocess_dict in enumerate(preprocess_methods):
        prep_image = run_preprocess(image, methods=preprocess_dict, parse=False)
        # Get the new filename with embedded preprocessing params
        fname = save_preprocessed_image(
            img_path=args.img_path,
            preprocess_params=preprocess_dict,
            prep_image=prep_image,
        )
        # Update the dataframe with the new image path, ensuring full path given
        df_new.loc[i, "img_path"] = fname
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
    # Save the new dataframe
    df_new.to_csv(f"{Path(args.img_path).stem}.csv", index=False)
