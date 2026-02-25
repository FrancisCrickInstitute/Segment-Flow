import argparse
from pathlib import Path
import shutil
from typing import Union, Optional
from aiod_registry import load_manifests
import download_model
import requests
from tqdm.auto import tqdm
from urllib.parse import urlparse


"""
Retrieves all required information from model registry before running model
python /Users/ahmedn/Work/ai-on-demand/src/ai_on_demand/Segment-Flow/modules/models/resources/usr/bin/setup_model.py
"""


def get_location_type(
    location: Union[str, list[str]],
    location_type: Optional[Union[str, list[str]]] = None,
):
    # Skip if provided
    if location_type is not None:
        # If a single location, convert to list
        if isinstance(location_type, str):
            location_type = [location_type]
        return location_type
    if not isinstance(location, list):
        location = [location]
    # Create a list to store the type of location
    location_type = []
    # Determine the type of location
    for loc in location:
        res = urlparse(loc)
        if res.scheme in ("http", "https"):
            location_type.append("url")
        elif res.scheme in ("file", ""):
            location_type.append("file")
        else:
            # NOTE: Because of including "" above, it is unlikely this will be reached
            raise TypeError(
                f"Cannot determine type (file/url) of location: {location}!"
            )
    return location_type


def download_model_files(
    output_dir: Union[str, Path],
    chkpt_loc: Union[str, Path],
    chkpt_loc_type: Optional[str] = None,
    param_loc: Optional[str] = None,
    param_loc_type: Union[str, Path] = None,
):
    # TODO: check if the files have already been downloaded here? or in the get_model_file?

    # download checkpoint
    get_model_file(
        output_dir=output_dir, file_loc=chkpt_loc, cache_folder="checkpoints"
    )
    if param_loc:
        get_model_file(output_dir, param_loc, cache_folder="checkpoints")

    # get_model_file(output_dir, param_loc, cache_folder="finetuning_cache")


def get_model_file(
    output_dir: Union[Path, str],
    file_loc: str,
    cache_folder: str,
    file_loc_type: Optional[str] = None,
):
    # NOTE: Using output_dir here as that's where Nextflow will copy the result to
    # TODO: don't downlaod if the files already downlaoded - filter is also done in the nextflow side

    file_type = get_location_type(file_loc, file_loc_type)[
        0
    ]  # TODO: what if its multiple files?
    file_output_dir = Path(output_dir) / cache_folder

    # Check whether we are using a local path or a URL
    if file_type == "url":
        # This parses the URL to get the root filename
        res = urlparse(file_loc)
        file_fname = Path(res.path).name
        file_fname = file_output_dir / file_fname
        if file_fname.exists():
            return
        print(f"Downloading {file_loc}")
        download_from_url(file_loc, Path(file_fname))
    elif file_type == "file":
        # Handle case where directory containing checkpoint is given
        res = Path(file_loc)
        file_loc = str(res.parent)
        file_fname = res.name

        if not Path(file_loc).is_file():
            if Path(file_loc).is_dir():
                file_loc = Path(file_loc) / file_fname
            else:
                raise FileNotFoundError(f"Model checkpoint not found: {file_loc}")
        file_fname = file_output_dir / file_fname
        if file_fname.exists():
            return
        print(f"Copying {file_loc}")
        copy_from_path(file_loc, Path(file_fname))


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


def get_model_details(model_name: str, model_version: str, task: str):
    model_name = "empanada"
    model_version = "MitoNet v1"
    task = "mito"
    manifests = load_manifests()
    print(type(manifests))
    for key in manifests.keys():
        print(key)
    print(type(manifests[model_name].versions))
    model_info = manifests[model_name].versions[model_version].tasks[task]
    location = model_info.location
    config_path = model_info.config_path
    params = model_info.params
    location_type = model_info.location_type  # Maybe provided?
    metadata = model_info.metadata

    print(f"{location=}")
    print(f"{config_path=}")
    print(f"{params=}")
    print(f"{location_type=}")
    print(f"{metadata=}")

    # TODO: move this into the same file
    download_model.get_model_checkpoint(
        output_dir="/Users/ahmedn/.nextflow/aiod/aiod_cache/empanada/checkpoints/",  # TODO: use the one provided by nextflow
        chkpt_fname=f"/Users/ahmedn/.nextflow/aiod/aiod_cache/empanada/checkpoints/{model_version}.pth",
        chkpt_loc=model_info.location,
        chkpt_type=model_info.location_type,
    )


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

    # model_details = get_model_details(
    #     model_name=args.model_name,
    #     model_version=args.model_version,
    #     model_task=args.task,
    # )

    model_location = "https://zenodo.org/record/6861565/files/MitoNet_v1.pth?download=1"
    model_location = "/Users/ahmedn/Work/finetuning/models/MitoNet_v1.pth"

    output_dir = "/Users/ahmedn/Desktop/AIOD-172"
    download_model_files(output_dir, chkpt_loc=model_location)
