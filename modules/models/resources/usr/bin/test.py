import argparse
import csv
import os
import requests
import shutil
from pathlib import Path
from tqdm.auto import tqdm
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


def get_file(
    fname: str,
    file_loc: str,
    file_type: str,
):
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


def main(model_name: str, model_version: str, model_task: str, cache_dir: str):
    manifests = load_manifests(filter_access=False)
    versions = manifests[model_name].versions
    # model_version arrives sanitised from Nextflow (e.g. "MitoNet-v1"); resolve to manifest key
    resolved_version = model_version.replace("-", " ")
    model_info = versions[resolved_version].tasks[model_task]
    model_location = model_info.location
    print(f"{model_location=}")
    # print(dir(model_info))
    model_config_location = model_info.config_path
    print(model_config_location)

    model_location_type = get_location_type(model_location)
    if model_location_type == "url":
        # This parses the URL to get the root filename which we'll use
        res = urlparse(model_location)
        full_model_name = Path(res.path).name
    else:
        res = Path(model_location)
        full_model_name = res.name
    full_model_name = full_model_name.replace("_", "-")
    cache_model_loc = Path(cache_dir) / full_model_name

    if cache_model_loc.exists():
        loc_path = cache_model_loc
        symlink_path = Path.cwd() / loc_path.name
        if not symlink_path.exists():
            os.symlink(loc_path, symlink_path)
            print(f"Created symlink: {symlink_path} -> {loc_path}")
        else:
            print(f"Symlink already exists: {symlink_path}")
    else:
        get_file(full_model_name, model_location, model_location_type)

    if model_config_location:
        config_location_type = get_location_type(model_config_location)

        model_config_name = model_version + "_config.yml"
        print("this is the model config name:", model_config_name)

        cache_config_loc = Path(cache_dir) / model_config_name

        if cache_config_loc.exists():
            loc_path = cache_model_loc
            symlink_path = Path.cwd() / loc_path.name
            if not symlink_path.exists():
                os.symlink(loc_path, symlink_path)
                print(f"Created symlink: {symlink_path} -> {loc_path}")
            else:
                print(f"Symlink already exists: {symlink_path}")
        else:
            get_file(model_config_name, model_config_location, config_location_type)
        # get the config file


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
    parser.add_argument(
        "--cache_loc",
        required=True,
        type=str,
        help="AIOD Cache location",
    )
    args = parser.parse_args()

    main(args.model_name, args.model_version, args.task, args.cache_loc)
