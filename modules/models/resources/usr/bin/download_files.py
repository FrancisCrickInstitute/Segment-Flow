import argparse
import csv
from pathlib import Path
import shutil
from typing import Union, Optional
from aiod_registry import load_manifests
import requests
from tqdm.auto import tqdm
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


def get_file(
    fname: str,
    file_loc: str,
):
    # Check whether we are using a local path or a URL
    file_type = get_location_type(file_loc)
    print(f"{file_type=}")

    if file_type == "url":
        print(f"Downloading {file_loc}")
        download_from_url(
            file_loc, Path(fname)
        )  # file name should contain the file type .pth
        # using the same name as fname
    elif file_type == "file":
        # Handle case where directory containing checkpoint is given
        if not Path(file_loc).is_file():
            if Path(file_loc).is_dir():
                file_loc = Path(file_loc) / fname
            else:
                raise FileNotFoundError(f"Model checkpoint not found: {file_loc}")
        print(f"Copying {file_loc}")
        copy_from_path(file_loc, Path(fname))


def download_from_url(url: str, chkpt_fname: Union[Path, str]):
    # Open the URL and get the content length
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    req = requests.get(url, stream=True, headers=headers)
    req.raise_for_status()
    content_length = int(req.headers.get("Content-Length"))

    # Download the file and update the progress bar
    with open(chkpt_fname, "wb") as f:
        with tqdm(
            desc=f"Downloading {chkpt_fname.name}...",
            total=content_length,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in req.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    # Close request
    req.close()
    print(f"Done! Checkpoint saved to {chkpt_fname}")


def copy_from_path(fpath: Union[Path, str], chkpt_fname: Union[Path, str]):
    if not Path(fpath).is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {fpath}")
    # Copy the file from accessible path
    shutil.copy(fpath, chkpt_fname)


def get_model_info(model_name: str, model_version: str, model_task: str):
    manifests = load_manifests()
    versions = manifests[model_name].versions
    # model_version arrives sanitised from Nextflow (e.g. "MitoNet-v1"); resolve to manifest key
    resolved_version = model_version.replace("-", " ")
    model_info = versions[resolved_version].tasks[model_task]
    location = model_info.location
    config_path = model_info.config_path
    param_location = None
    metadata = model_info.metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        required=False,
        type=str,
        help="The name of the model",
    )
    parser.add_argument(
        "--model_location",
        required=False,
        type=str,
        help="The name of the model",
    )
    parser.add_argument(
        "--model_param_name",
        required=False,
        type=str,
        help="The name of the model",
    )
    parser.add_argument(
        "--model_param_location",
        required=False,
        type=str,
        help="The name of the model",
    )
    args = parser.parse_args()

    if args.model_name and args.model_location:
        get_file(args.model_name, args.model_location)
    else:
        print("Model already downloaded?")
    if args.model_param_name and args.model_param_location:
        get_file(args.model_param_name, args.model_param_location)
    else:
        print("Model params already downloaded?")
