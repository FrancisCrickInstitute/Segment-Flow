"""
For this first version, where we will implement this new pipeline for preprocessing sets, we will just use the existing functions and run them over the whole image naively without Dask/chunk thoughts 
"""

from pathlib import Path

import pandas as pd
import skimage.io

from aiod_utils.preprocess import get_preprocess_params, run_preprocess, load_methods
from aiod_utils.io import load_image


def construct_fname(img_path, preprocess_params):
    suffix = get_preprocess_params(preprocess_params, to_save=True)
    img_path = Path(img_path)
    return f"{img_path.stem}_{suffix}{img_path.suffix}"


def save_preprocessed_image(img_path, preprocess_params, prep_image):
    fname = construct_fname(img_path, preprocess_params)
    # Save the image
    # TODO: Switch over to bioio when fully sorted
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
    # Preprocess the image
    # TODO: Switch to return_dask, map over blocks, and check output as described at top
    image = load_image(fpath=args.img_path, return_array=True).squeeze()
    # Extract all preprocessing sets
    preprocess_methods = load_methods(args.preprocess_params)
    # Create a new dataframe to store the new images, repeating rows per preprocessing set
    df_new = pd.concat([df_img] * len(preprocess_methods), ignore_index=True)
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
        # TODO: Update size and other metadata in the dataframe, needed if downsampled
        if image.shape != prep_image.shape:
            # Find the column for each of the elements, and overwrite
            # NOTE: If we add a preprocessing function that modifies number of channels, this will need to be updated
            # TODO: Reorder columns to match order of image.shape, regardless
            cols = ["num_slices", "height", "width"]
            orig_shape = tuple(df_new.loc[i, cols])
            for j, val in enumerate(prep_image.shape):
                if val not in orig_shape:
                    df_new.loc[i, cols[j]] = val
    # Save the new dataframe
    df_new.to_csv(f"{Path(args.img_path).stem}.csv", index=False)
