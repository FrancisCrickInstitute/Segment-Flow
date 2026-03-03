import argparse
import csv
from pathlib import Path
from typing import Union, Optional
from aiod_registry import load_manifests
from urllib.parse import urlparse


def get_location_type(
    location: str,
):
    # Determine the type of location
    res = urlparse(location)
    if res.scheme in ("http", "https"):
        return "url"
    elif res.scheme in ("file", ""):
        return "file"
    else:
        raise TypeError(f"Cannot determine type (file/url) of location: {location}!")


def get_model_info(model_name: str, model_version: str, model_task: str):
    manifests = load_manifests(filter_access=False)
    versions = manifests[model_name].versions
    # model_version arrives sanitised from Nextflow (e.g. "MitoNet-v1"); resolve to manifest key
    resolved_version = model_version.replace("-", " ")
    model_info = versions[resolved_version].tasks[model_task]
    model_location = model_info.location
    print(f"{model_location=}")
    # print(dir(model_info))
    model_params_location = model_info.config_path
    print(model_params_location)

    model_location_type = get_location_type(model_location)
    if model_location_type == "url":
        # This parses the URL to get the root filename which we'll use
        res = urlparse(model_location)
        model_name = Path(res.path).name
    else:
        res = Path(model_location)
        model_name = res.name

    model_params_name = None
    if model_params_location:
        params_location_type = get_location_type(model_params_location)
        if params_location_type == "url":
            # This parses the URL to get the root filename which we'll use
            res = urlparse(model_params_location)
            model_params_name = Path(res.path).name
        else:
            res = Path(model_params_location)
            model_params_name = res.name

    print(f"{ model_name=}")
    print(f"{ model_location=}")

    header = [
        "name",
        "location",
        "loc_type",
        "file_type",
    ]

    rows = []
    rows.append([model_name, model_location, model_location_type, "model"])
    if model_params_name:
        rows.append(
            [model_params_name, model_params_location, params_location_type, "params"]
        )

    with open("model_info.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="The name of the model",
    )
    parser.add_argument(
        "--model_version",
        required=True,
        type=str,
        help="The version of the model",
    )
    parser.add_argument(
        "--task",
        required=True,
        type=str,
        help="The task the model will be used for",
    )
    args = parser.parse_args()

    get_model_info(args.model_name, args.model_version, args.task)
