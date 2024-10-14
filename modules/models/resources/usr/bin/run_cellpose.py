from pathlib import Path
from typing import Union

from cellpose import models

import yaml

from utils import save_masks, create_argparser_inference, load_img
from model_utils import get_device


def run_cellpose(
    save_dir: Union[Path, str],
    save_name: str,
    idxs: list[int, ...],
    config: dict,
):
    # Extract model config arguments
    masks, _, _, _ = cp_model.eval(
        img,
        diameter=config["diameter"],
        channels=config["channels"],
        batch_size=config["batch_size"],
        channel_axis=config["channel_axis"],
        z_axis=config["z_axis"],
        do_3D=config["do_3D"],
        anisotropy=config["anisotropy"],
        stitch_threshold=config["stitch_threshold"],
        cellprob_threshold=config["cellprob_threshold"],
        flow_threshold=config["flow_threshold"],
        niter=config["niter"],
        min_size=config["min_size"],
    )

    save_masks(Path(save_dir), save_name, masks, idxs=idxs)


if __name__ == "__main__":
    parser = create_argparser_inference()
    cli_args = parser.parse_args()

    with open(cli_args.model_config, "r") as f:
        config = yaml.safe_load(f)

    # Load image and apply preprocessing if specified
    img = load_img(
        fpath=cli_args.img_path,
        idxs=cli_args.idxs,
        channels=cli_args.channels,
        num_slices=cli_args.num_slices,
        preprocess_params=cli_args.preprocess_params,
        dim_order="CZYX",
    )
    # Ensure config has the correct channel and z axis
    shape = img.shape
    if shape[0] == cli_args.channels:
        config["channel_axis"] = 0
    elif shape[1] == cli_args.channels:
        config["channel_axis"] = 1
    else:
        config["channel_axis"] = None
    if shape[0] == cli_args.num_slices:
        config["z_axis"] = 0
    elif shape[1] == cli_args.num_slices:
        config["z_axis"] = 1
    else:
        config["z_axis"] = None

    # Extract the segment and nucleus channels
    config["channels"] = [
        config["segment_channel"],
        config["nucleus_channel"],
    ]

    # Create the Cellpose model with available device
    cp_model = models.Cellpose(model_type=cli_args.model_type, device=get_device())

    run_cellpose(
        save_dir=cli_args.output_dir,
        save_name=cli_args.mask_fname,
        idxs=cli_args.idxs,
        config=config,
    )
