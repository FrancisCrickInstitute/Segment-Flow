from pathlib import Path
from typing import Union

import numpy as np
import yaml

from stardist.models import StarDist2D, StarDist3D
from csbdeep.utils import normalize

from utils import save_masks, create_argparser_inference, load_img


def run_stardist(
    save_dir: Union[Path, str],
    save_name: str,
    idxs: list[int, ...],
    img: np.ndarray,
    model_type: str,
    model_chkpt: Union[Path, str],
    config: dict,
    is_3d: bool = False,
):
    """Run StarDist segmentation pipeline.

    Args:
        save_dir: Directory to save the output masks
        save_name: Base name for saved files
        idxs: Slice indices being processed
        img: Input image array
        model_type: Model type/version to use
        model_chkpt: Path to model checkpoint directory
        config: Configuration dictionary containing model parameters
        is_3d: Whether to use 3D StarDist model
    """
    print(f"Running StarDist {'3D' if is_3d else '2D'} segmentation...")

    # Load the appropriate model
    if is_3d:
        # For 3D, load from checkpoint directory
        if model_chkpt and Path(model_chkpt).is_dir():
            model_path = Path(model_chkpt)
            model = StarDist3D(None, name=model_type, basedir=str(model_path.parent))
        else:
            # Try to load pretrained model
            try:
                model = StarDist3D.from_pretrained(model_type)
            except Exception as e:
                print(f"Failed to load pretrained model: {e}")
                raise ValueError(f"Could not load StarDist3D model '{model_type}'")
    else:
        # For 2D, load from checkpoint directory or pretrained
        if model_chkpt and Path(model_chkpt).is_dir():
            model_path = Path(model_chkpt)
            model = StarDist2D(None, name=model_type, basedir=str(model_path.parent))
        else:
            # Try to load pretrained model
            try:
                model = StarDist2D.from_pretrained(model_type)
            except Exception as e:
                print(f"Failed to load pretrained model: {e}")
                raise ValueError(f"Could not load StarDist2D model '{model_type}'")

    print(f"Model loaded: {model_type}")
    print(f"Input image shape: {img.shape}")

    # Normalize the image
    normalize_percentiles = config.get("normalize", [1, 99.8])
    if config.get("normalize_img", True):
        img_normalized = normalize(img, *normalize_percentiles)
    else:
        img_normalized = img

    # Extract prediction parameters
    prob_thresh = config.get("prob_thresh", None)
    nms_thresh = config.get("nms_thresh", None)
    n_tiles = config.get("n_tiles", None)
    if n_tiles is not None:
        n_tiles = tuple(n_tiles)

    # Additional parameters
    scale = config.get("scale", None)

    # Run prediction
    print("Running prediction...")
    labels, details = model.predict_instances(
        img_normalized,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        n_tiles=n_tiles,
        scale=scale,
    )

    print(f"Segmentation complete. Labels shape: {labels.shape}, unique labels: {len(np.unique(labels))}")

    # Save the final segmentation
    save_masks(Path(save_dir), save_name, labels, idxs=idxs, mask_type="instance")


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

    # Determine if this is 2D or 3D based on the number of slices
    is_3d = cli_args.num_slices > 1

    # StarDist expects specific input formats
    if is_3d:
        # 3D: expects ZYX format for single channel
        if cli_args.channels == 1:
            img = img[0]  # Remove the channel dimension
            print(f"3D mode: image shape after channel squeeze: {img.shape}")
        else:
            raise ValueError("StarDist does not support multi-channel 3D images directly")
    else:
        # 2D: expects YX for grayscale or YXC for RGB
        if cli_args.channels == 1:
            # Squeeze both channel and z dimensions for single channel 2D
            img = img[0, 0]
            print(f"2D mode: image shape after squeeze: {img.shape}")
        elif cli_args.channels == 3:
            # RGB image: squeeze z dimension but keep channels, then move to last axis
            img = img[:, 0, :, :]  # Remove z dimension but keep channels (CYX)
            img = np.moveaxis(img, 0, -1)  # Move channels to last axis (YXC)
            print(f"2D RGB mode: image shape: {img.shape}")
        else:
            raise ValueError(f"StarDist 2D supports 1 or 3 channels, got {cli_args.channels}")

    run_stardist(
        save_dir=cli_args.output_dir,
        save_name=cli_args.mask_fname,
        idxs=cli_args.idxs,
        img=img,
        model_type=cli_args.model_type,
        model_chkpt=cli_args.model_chkpt,
        config=config,
        is_3d=is_3d,
    )
