from pathlib import Path
from typing import Union

import numpy as np
import yaml

from stardist.models import StarDist2D, StarDist3D
from csbdeep.utils import normalize

from utils import save_masks, create_argparser_inference, load_img, guess_rgb


def _load_stardist_model(model_type: str, model_chkpt: Union[Path, str], is_3d: bool):
    """Load StarDist model (2D or 3D).

    Args:
        model_type: Model type/version to use
        model_chkpt: Path to model checkpoint directory
        is_3d: Whether to use 3D StarDist model

    Returns:
        Loaded StarDist model
    """
    if is_3d:
        # For 3D, load from checkpoint directory
        if model_chkpt and Path(model_chkpt).is_dir():
            model_path = Path(model_chkpt)
            model = StarDist3D(None, name=model_type, basedir=str(model_path.parent))
        else:
            # Try to load pretrained model
            model = StarDist3D.from_pretrained(model_type)
            if model is None:
                raise ValueError(f"Could not load StarDist3D model '{model_type}'")
    else:
        # For 2D, load from checkpoint directory or pretrained
        if model_chkpt and Path(model_chkpt).is_dir():
            model_path = Path(model_chkpt)
            model = StarDist2D(None, name=model_type, basedir=str(model_path.parent))
        else:
            # Try to load pretrained model
            model = StarDist2D.from_pretrained(model_type)
            if model is None:
                raise ValueError(f"Could not load StarDist2D model '{model_type}'")
    return model


def _run_stardist_2d(
    img: np.ndarray,
    model: StarDist2D,
    config: dict,
) -> np.ndarray:
    """Run 2D StarDist prediction on a single slice.

    Args:
        img: 2D image (YX) or RGB image (YXC)
        model: StarDist2D model
        config: Configuration dictionary containing model parameters

    Returns:
        2D label image
    """
    # Normalize the image
    normalize_pmin = config.get("normalize_pmin", 1)
    normalize_pmax = config.get("normalize_pmax", 99.8)
    if config.get("normalize_img", True):
        img_normalized = normalize(img, normalize_pmin, normalize_pmax)
    else:
        img_normalized = img

    # Extract prediction parameters
    prob_thresh = config.get("prob_thresh", None)
    nms_thresh = config.get("nms_thresh", None)
    n_tiles = config.get("n_tiles", None)
    if n_tiles is not None:
        n_tiles = tuple(n_tiles)
    scale = config.get("scale", None)

    # Run prediction
    labels, details = model.predict_instances(
        img_normalized,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        n_tiles=n_tiles,
        scale=scale,
    )

    return labels


def _run_stardist_2d_stack(
    img_stack: np.ndarray,
    model: StarDist2D,
    config: dict,
) -> np.ndarray:
    """Run 2D StarDist prediction on each slice of a 3D stack.

    Args:
        img_stack: 3D image stack (ZYX)
        model: StarDist2D model
        config: Configuration dictionary containing model parameters

    Returns:
        3D label image (ZYX)
    """
    print(f"Running 2D model on {img_stack.shape[0]} slices...")
    all_labels = np.zeros(img_stack.shape, dtype=np.uint16)

    for z_idx in range(img_stack.shape[0]):
        img_slice = img_stack[z_idx]
        labels = _run_stardist_2d(img_slice, model, config)
        all_labels[z_idx] = labels

    return all_labels


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
        img: Input image array (shape varies based on input, see below)
        model_type: Model type/version to use
        model_chkpt: Path to model checkpoint directory
        config: Configuration dictionary containing model parameters
        is_3d: Whether to use 3D StarDist model

    Expected input shapes:
        2D model:
            - 2D grayscale: (Y, X)
            - 2D RGB: (Y, X, C) where C=3
            - 3D grayscale stack: (Z, Y, X) - runs model on each slice
        3D model:
            - 3D grayscale: (Z, Y, X)
    """
    print(f"Running StarDist {'3D' if is_3d else '2D'} segmentation...")
    print(f"Input image shape: {img.shape}")

    # Load the appropriate model
    model = _load_stardist_model(model_type, model_chkpt, is_3d)
    print(f"Model loaded: {model_type}")

    # Run prediction based on model type and data shape
    if is_3d:
        # 3D model: expects ZYX format
        if img.ndim != 3:
            raise ValueError(f"3D model expects 3D input (ZYX), got shape {img.shape}")

        # Normalize the image
        normalize_pmin = config.get("normalize_pmin", 1)
        normalize_pmax = config.get("normalize_pmax", 99.8)
        if config.get("normalize_img", True):
            img_normalized = normalize(img, normalize_pmin, normalize_pmax)
        else:
            img_normalized = img

        # Extract prediction parameters
        prob_thresh = config.get("prob_thresh", None)
        nms_thresh = config.get("nms_thresh", None)
        n_tiles = config.get("n_tiles", None)
        if n_tiles is not None:
            n_tiles = tuple(n_tiles)
        scale = config.get("scale", None)

        print("Running 3D prediction...")
        labels, details = model.predict_instances(
            img_normalized,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            n_tiles=n_tiles,
            scale=scale,
        )
    else:
        # 2D model: expects YX or YXC format
        if img.ndim == 2:
            # Single 2D grayscale image
            print("Running 2D prediction on single slice...")
            labels = _run_stardist_2d(img, model, config)
        elif img.ndim == 3:
            # Could be RGB (YXC) or 3D stack (ZYX)
            if guess_rgb(img.shape, dim=-1):
                # RGB image (YXC)
                print("Running 2D prediction on RGB image...")
                labels = _run_stardist_2d(img, model, config)
            else:
                # 3D stack - run 2D model on each slice
                labels = _run_stardist_2d_stack(img, model, config)
        else:
            raise ValueError(f"2D model expects 2D (YX), RGB (YXC), or 3D stack (ZYX) input, got shape {img.shape}")

    print(f"Segmentation complete. Labels shape: {labels.shape}, unique labels: {len(np.unique(labels))}")

    # Save the final segmentation
    save_masks(Path(save_dir), save_name, labels, idxs=idxs, mask_type="instance")


if __name__ == "__main__":
    parser = create_argparser_inference()
    parser.add_argument(
        "--channel-idx",
        type=int,
        default=0,
        help="Channel index to use for multi-channel images (default: 0)",
    )
    parser.add_argument(
        "--use-3d-model",
        action="store_true",
        help="Use 3D StarDist model instead of 2D (default: False)",
    )
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

    # Determine model dimensionality from CLI argument
    is_3d = cli_args.use_3d_model

    print(f"Input data: channels={cli_args.channels}, num_slices={cli_args.num_slices}")
    print(f"Loaded image shape (CZYX): {img.shape}")

    # Prepare image based on model type and data characteristics
    if is_3d:
        # 3D model requested
        if cli_args.num_slices == 1:
            raise ValueError("3D model requires multiple slices (num_slices > 1)")

        if cli_args.channels == 1:
            # Single channel 3D: remove channel dimension -> ZYX
            img = img[0]
            print(f"3D model: prepared shape (ZYX): {img.shape}")
        elif cli_args.channels == 3:
            raise ValueError("3D model does not support RGB images")
        elif cli_args.channels > 1:
            # Multi-channel 3D: select channel
            if cli_args.channel_idx >= cli_args.channels:
                raise ValueError(f"channel_idx={cli_args.channel_idx} exceeds available channels={cli_args.channels}")
            img = img[cli_args.channel_idx]
            print(f"3D model: selected channel {cli_args.channel_idx}, prepared shape (ZYX): {img.shape}")
        else:
            raise ValueError(f"Unexpected channel configuration: channels={cli_args.channels}")
    else:
        # 2D model requested
        if cli_args.channels == 1:
            if cli_args.num_slices == 1:
                # Single 2D slice: remove both channel and z dimensions -> YX
                img = img[0, 0]
                print(f"2D model (single slice): prepared shape (YX): {img.shape}")
            else:
                # 3D stack: remove channel dimension, keep z for per-slice processing -> ZYX
                img = img[0]
                print(f"2D model (stack): prepared shape (ZYX): {img.shape}")
        elif cli_args.channels == 3 and cli_args.num_slices == 1:
            # RGB 2D image: remove z dimension, move channels to last axis -> YXC
            img = img[:, 0, :, :]  # Remove z dimension (CYX)
            img = np.moveaxis(img, 0, -1)  # Move channels to last axis (YXC)
            print(f"2D model (RGB): prepared shape (YXC): {img.shape}")
        elif cli_args.channels > 1 and cli_args.num_slices == 1:
            # Multi-channel 2D: select channel, remove z dimension -> YX
            if cli_args.channel_idx >= cli_args.channels:
                raise ValueError(f"channel_idx={cli_args.channel_idx} exceeds available channels={cli_args.channels}")
            img = img[cli_args.channel_idx, 0]
            print(
                f"2D model (multi-channel): selected channel {cli_args.channel_idx}, prepared shape (YX): {img.shape}"
            )
        elif cli_args.channels > 1 and cli_args.num_slices > 1:
            # 4D data with 2D model: select channel, keep z for per-slice processing -> ZYX
            if cli_args.channel_idx >= cli_args.channels:
                raise ValueError(f"channel_idx={cli_args.channel_idx} exceeds available channels={cli_args.channels}")
            img = img[cli_args.channel_idx]
            print(f"2D model (4D data): selected channel {cli_args.channel_idx}, prepared shape (ZYX): {img.shape}")
        else:
            raise ValueError(
                f"Unexpected data configuration: channels={cli_args.channels}, num_slices={cli_args.num_slices}"
            )

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
