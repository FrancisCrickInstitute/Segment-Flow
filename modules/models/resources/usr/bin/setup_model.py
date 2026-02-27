import argparse
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


def get_model_file(
    output_dir: Union[Path, str],
    file_loc: str,
):
    # Check whether we are using a local path or a URL
    file_type = get_location_type(file_loc)

    if file_type == "url":
        print(f"Downloading {file_loc}")
        download_from_url(file_loc, Path(output_dir))
    elif file_type == "file":
        # Handle case where directory containing checkpoint is given
        if not Path(file_loc).is_file():
            if Path(file_loc).is_dir():
                file_loc = Path(file_loc) / output_dir
            else:
                raise FileNotFoundError(f"Model checkpoint not found: {file_loc}")
        print(f"Copying {file_loc}")
        copy_from_path(file_loc, Path(output_dir))


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


def get_model_info(
    model_name: str, model_version: str, model_task: str, model_dir: str
):
    manifests = load_manifests()
    versions = manifests[model_name].versions
    # model_version arrives sanitised from Nextflow (e.g. "MitoNet-v1"); resolve to manifest key
    resolved_version = model_version.replace("-", " ")
    model_info = versions[resolved_version].tasks[model_task]
    location = model_info.location
    config_path = model_info.config_path
    param_location = None  # TODO: what is the attribute called?
    metadata = model_info.metadata

    print(f"{location=}")
    print(f"{config_path=}")
    print(f"{param_location=}")
    print(f"{metadata=}")

    # check if files already exist
    print(
        "checking:", (Path(model_dir) / "checkpoints" / (str(model_version) + ".pth")/Users/ahmedn/.nextflow/aiod/aiod_cache/empanada/checkpoints/MitoNet-v1.pth)
    )
    if not (Path(model_dir) / "checkpoints" / (str(model_version) + ".pth")).exists():
        # download checkpoints
        get_model_file(output_dir=model_version + ".pth", file_loc=location)
    else:
        print("file exists")
    if param_location:
        if not (
            Path(model_dir) / "checkpoints" / (str(model_version) + ".pth")
        ).exists():
            # download model param_location.yml file if available
            get_model_file(
                param_location,
                param_location,
            )
    # TODO: Where should this be saved? - /params?

    # get_model_file(output_dir, param_loc, cache_folder="finetuning_cache")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        required=False,  # TODO: change back to required
        type=str,
        help="The name of the model",
    )
    parser.add_argument(
        "--model_version",
        required=False,
        type=str,
        help="The version of the model",
    )
    parser.add_argument(
        "--task",
        required=False,
        type=str,
        help="The task the model will be used for",
    )
    parser.add_argument(
        "--model_dir",
        required=False,
        type=str,
        help="Where the model checkpoints and the param files will be saved",
    )

    args = parser.parse_args()

    model_details = get_model_info(
        model_name=args.model_name,
        model_version=args.model_version,
        model_task=args.task,
        model_dir=args.model_dir,
    )

    # model_location = "https://zenodo.org/record/6861565/files/MitoNet_v1.pth?download=1"
    # model_location = "/Users/ahmedn/Work/finetuning/models/MitoNet_v1.pth"

    # output_dir = "/Users/ahmedn/Desktop/AIOD-172"
