"""
Adds a stable image_id (plus placeholder prep_hash/preprocess_params columns)
to every row of the input image CSV, before any preprocessing branching
happens. image_id relies on aiod_utils' bioio-based extension recognition,
which Nextflow/Groovy cannot replicate natively, so it's computed once here
in Python rather than re-derived independently wherever it's needed.

Preprocessed branches later overwrite prep_hash/preprocess_params with their
real values (see preprocess_image.py); no-op/non-preprocessed rows keep the
placeholders here, so every row in the pipeline shares the same CSV schema.
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
    if "preprocess_params" not in df.columns:
        df["preprocess_params"] = "[]"
    df.to_csv(Path(args.output_csv), index=False)
