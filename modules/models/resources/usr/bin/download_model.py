import argparse
from pathlib import Path
import shutil

import requests
from tqdm.auto import tqdm


def get_model_checkpoint(
    chkpt_fname: str,
    chkpt_loc: str,
    chkpt_type: str,
):
    """Download or copy a model artifact into the current working directory.

    Existence checking is intentionally absent: Nextflow's storeDir directive
    handles it externally — this script is only ever called when the file is
    genuinely missing from the cache.
    """
    if chkpt_type == "url":
        print(f"Downloading {chkpt_loc}")
        download_from_url(chkpt_loc, Path(chkpt_fname))
    elif chkpt_type == "file":
        # Handle case where directory containing checkpoint is given
        if not Path(chkpt_loc).is_file():
            if Path(chkpt_loc).is_dir():
                chkpt_loc = Path(chkpt_loc) / chkpt_fname
            else:
                raise FileNotFoundError(f"Model artifact not found: {chkpt_loc}")
        print(f"Copying {chkpt_loc}")
        copy_from_path(chkpt_loc, Path(chkpt_fname))


def download_from_url(url: str, chkpt_fname: Path | str):
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


def copy_from_path(fpath: Path | str, chkpt_fname: Path | str):
    if not Path(fpath).is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {fpath}")
    # Copy the file from accessible path
    shutil.copy(fpath, chkpt_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download or copy a model artifact into the current working directory. "
            "Cache-hit detection is handled externally by Nextflow's storeDir directive."
        )
    )
    parser.add_argument(
        "--chkpt-loc",
        required=True,
        type=str,
        help="Source location of the artifact (URL or file path)",
    )
    parser.add_argument(
        "--chkpt-type",
        required=True,
        type=str,
        choices=["url", "file"],
        help="Type of source location",
    )
    parser.add_argument(
        "--chkpt-fname",
        required=True,
        type=str,
        help="Destination filename (written to the current working directory)",
    )

    args = parser.parse_args()

    get_model_checkpoint(
        chkpt_fname=args.chkpt_fname,
        chkpt_loc=args.chkpt_loc,
        chkpt_type=args.chkpt_type,
    )
