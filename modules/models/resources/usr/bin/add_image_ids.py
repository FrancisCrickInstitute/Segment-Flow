"""
Adds a stable image_id and prep_hash column to every row of
the input image CSV, before any preprocessing branching happens.

image_id derived from aiod_utils' bioio-based extension recognition
Necessary as we can't do this in Nextflow/Groovy!

Preprocessed branches later overwrite prep_hash with its real value (see
preprocess_image.py)
Non-preprocessed rows keep placeholder so every row shares the same CSV cols
"""

import argparse
from pathlib import Path

import pandas as pd
from aiod_utils.io import get_image_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add a stable image_id to every row of an image CSV"
    )
    parser.add_argument("--img-csv", required=True, help="Path to the input CSV")
    parser.add_argument(
        "--output-csv", required=True, help="Path to write the augmented CSV"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.img_csv)
    df["image_id"] = df["img_path"].apply(get_image_id)
    if "prep_hash" not in df.columns:
        df["prep_hash"] = ""
    df.to_csv(Path(args.output_csv), index=False)
