from pathlib import Path
from typing import Union, Any
import numpy as np

from cellpose import models
import yaml

from utils import save_masks, create_argparser_inference, load_img, get_model_name_type
from model_utils import get_device


def run_cellpose(
    save_dir: Union[Path, str],
    save_name: str,
    idxs: list[int],
    config: dict[str, Any],
) -> None:
    # Extract model config arguments
    masks, _, _ = cp_model.eval(
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

    config["do_3D"] = True if cli_args.num_slices > 1 else False

    # Extract the segment and nucleus channels
    config["channels"] = [
        config["segment_channel"],
        config["nucleus_channel"],
    ]

    device = get_device(model_type=get_model_name_type(cli_args.model_type))

    # Initialize CellposeModel directly with finetuned checkpoint
    cp_model_inner = models.CellposeModel(
        pretrained_model=cli_args.model_chkpt,
        device=device,
    )

    # Now create a wrapper-like object that includes a SizeModel based on base_model
    from cellpose.models import size_model_path, SizeModel

    base_size_model_path = size_model_path(cli_args.base_model)

    # Create a simple wrapper object that acts like Cellpose
    class CellposeWrapper:
        """
        Wraps a fine-tuned CellposeModel with the base model's SizeModel.

        This allows inference with a fine-tuned checkpoint while using the
        base model's diameter estimation capability.

        Args:
            cp_model: Fine-tuned CellposeModel instance
            size_model: SizeModel from base model (e.g., 'cellpose')
        """

        def __init__(self, cp_model: Any, size_model: Any) -> None:
            self.cp = cp_model
            self.sz = size_model
            self.device = cp_model.device
            self.diam_mean = cp_model.diam_mean
            self.pretrained_size = size_model.pretrained_size

        def eval(
            self,
            x: np.ndarray,
            batch_size: int = 8,
            channels: list[int] = [0, 0],
            channel_axis: int | None = None,
            invert: bool = False,
            normalize: bool = True,
            diameter: float | list[float] | None = 30.0,
            do_3D: bool = False,
            **kwargs: Any,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

            diam0 = (
                diameter[0] if isinstance(diameter, (np.ndarray, list)) else diameter
            )
            estimate_size = True if (diameter is None or diam0 == 0) else False

            if (
                estimate_size
                and self.pretrained_size is not None
                and not do_3D
                and x[0].ndim < 4
            ):
                diams, _ = self.sz.eval(
                    x,
                    channels=channels,
                    channel_axis=channel_axis,
                    batch_size=batch_size,
                    normalize=normalize,
                    invert=invert,
                )
                diameter = None
            elif estimate_size:
                if self.pretrained_size is None:
                    reason = "no pretrained size model specified in model Cellpose"
                else:
                    reason = "does not work on non-2D images"
                print(f"could not estimate diameter, {reason}")
                diams = self.diam_mean
            else:
                diams = diameter

                # delegate the rest to the CellposeModel
                return self.cp.eval(
                    x,
                    channels=channels,
                    channel_axis=channel_axis,
                    batch_size=batch_size,
                    normalize=normalize,
                    invert=invert,
                    diameter=diams,
                    do_3D=do_3D,
                    **kwargs,
                )

    size_model = SizeModel(
        device=device, pretrained_size=base_size_model_path, cp_model=cp_model_inner
    )
    cp_model = CellposeWrapper(cp_model_inner, size_model)

    run_cellpose(
        save_dir=cli_args.output_dir,
        save_name=cli_args.mask_fname,
        idxs=cli_args.idxs,
        config=config,
    )
