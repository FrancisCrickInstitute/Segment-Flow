import argparse
from pathlib import Path
import shutil
from typing import Union

import requests
from tqdm.auto import tqdm


def get_model_checkpoint(
    chkpt_output_dir: Union[Path, str], chkpt_fname: str, chkpt_loc: str, chkpt_type: str, 
):
    # Get the model dict
    # model_dict = MODEL_BANK[model_name][task]
    # Get the checkpoint filename
    # chkpt_fname = Path(model_dict[model_type]["filename"])
    # Just return if this already exists
    # NOTE: Using chkpt_output_dir here as that's where Nextflow will copy the result to
    if (Path(chkpt_output_dir) / chkpt_fname).exists():
        return
    # Check whether we are using a local path or a URL
    if chkpt_type == "url":
        print(f"Downloading {chkpt_loc}")
        download_from_url(chkpt_loc, Path(chkpt_fname))
    elif chkpt_type == "dir":
        # Handle case where directory containing checkpoint is given
        if not Path(chkpt_loc).is_file():
            if Path(chkpt_loc).is_dir():
                chkpt_loc = Path(chkpt_loc) / chkpt_fname
            else:
                raise FileNotFoundError(f"Model checkpoint not found: {chkpt_loc}")
        print(f"Copying {chkpt_loc}")
        copy_from_path(chkpt_loc, Path(chkpt_fname))
    else:
        raise KeyError(
            f"Either 'url' or 'dir' must be specified!"
        )


def download_from_url(url: str, chkpt_fname: Union[Path, str]):
    # Open the URL and get the content length
    req = requests.get(url, stream=True)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chkpt-path",
        required=True,
        type=str,
        help="Full path to model checkpoint (for saving)",
    )
    parser.add_argument(
        "--chkpt-loc",
        required=True,
        type=str,
        help="Location of model checkpoint (source)",
    )
    parser.add_argument(
        "--chkpt-type",
        required=True,
        type=str,
        choices=["url", "dir"],
        help="Type of model checkpoint location",
    )
    parser.add_argument(
        "--chkpt-fname",
        required=True,
        type=str,
        help="Filename of the checkpoint",
    )

    args = parser.parse_args()

    chkpt_output_dir = Path(args.chkpt_path).parent

    get_model_checkpoint(
        chkpt_output_dir=chkpt_output_dir,
        chkpt_fname=args.chkpt_fname,
        chkpt_loc=args.chkpt_loc,
        chkpt_type=args.chkpt_type,
    )
