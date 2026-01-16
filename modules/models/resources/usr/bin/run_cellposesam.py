from pathlib import Path
from typing import Union

from cellpose import models
import torch
import yaml

from utils import save_masks, create_argparser_inference, load_img, get_model_name_type
from model_utils import get_device


def run_cellpose(
    save_dir: Union[Path, str],
    save_name: str,
    idxs: list[int, ...],
    config: dict,
    model_chkpt: str,
    device: torch.device,
):
    # NOTE: This is just a link to the Cellpose-SAM model, but circumvents Cellpose's fixed model location
    # Initialize Cellpose-SAM model
    cp_model = models.CellposeModel(
        gpu=device.type == "cuda",
        pretrained_model=str(Path(model_chkpt).readlink()),
    )
    # Extract model config arguments
    # NOTE: rescale and channels not used anymore
    # NOTE: more than 3 channels not used
    masks, _, _ = cp_model.eval(
        img,
        batch_size=config["batch_size"],
        flow_threshold=config["flow_threshold"],
        cellprob_threshold=config["cellprob_threshold"],
        z_axis=config["z_axis"],
        channel_axis=config["channel_axis"],
        do_3D=config["do_3D"],
        anisotropy=config["anisotropy"],
        stitch_threshold=config["stitch_threshold"],
        min_size=config["min_size"],
        max_size_fraction=config["max_size_fraction"],
    )

    save_masks(Path(save_dir), save_name, masks, idxs=idxs, mask_type="instance")


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
        dim_order="CZYX",
    )
    # Squeeze the image to avoid specifying a z or channel axis for single slice/channel images
    # As Cellpose does some checking underneath which leads to errors
    img = img.squeeze()
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
    # Ensure 3D properly set
    # TODO: Does Cellpose-SAM still need this or does it better handle 3D in 2D mode?
    config["do_3D"] = True if cli_args.num_slices > 1 else False

    device = get_device(model_type=get_model_name_type(cli_args.model_type))

    run_cellpose(
        save_dir=cli_args.output_dir,
        save_name=cli_args.mask_fname,
        idxs=cli_args.idxs,
        config=config,
        model_chkpt=cli_args.model_chkpt,
        device=device,
    )
