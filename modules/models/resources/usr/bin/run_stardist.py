import shutil
import zipfile
from pathlib import Path

import numpy as np
import yaml
from csbdeep.utils import normalize
from stardist.models import StarDist2D, StarDist3D
from utils import create_argparser_inference, load_img, save_masks

print("Latest!!")

STARDIST_MODEL_FILES = ("config.json", "thresholds.json")


def _find_stardist_model_dir(search_root: Path) -> Path:
    """Find the extracted StarDist model directory containing config files."""
    if search_root.is_dir():
        if any((search_root / fname).exists() for fname in STARDIST_MODEL_FILES):
            return search_root

        for candidate in sorted(search_root.rglob("config.json")):
            if candidate.parent.is_dir():
                return candidate.parent

    raise FileNotFoundError(f"Could not find extracted StarDist model files in {search_root}.")


def _extract_stardist_archive(archive_path: Path, model_type: str) -> Path:
    """Extract a StarDist archive into a stable cache directory and return the model directory."""
    extract_root = archive_path.parent / f"{model_type}_extracted"
    marker_path = extract_root / ".aiod_extracted"

    if not marker_path.exists():
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_root)
        marker_path.touch()

    return _find_stardist_model_dir(extract_root)


def _resolve_stardist_model_dir(model_chkpt: Path | str, model_type: str) -> Path | None:
    """Resolve a local checkpoint input into the actual StarDist model directory."""
    if not model_chkpt:
        return None

    model_path = Path(model_chkpt)
    if not model_path.exists():
        return None

    if model_path.is_dir():
        return _find_stardist_model_dir(model_path)

    if model_path.suffix.lower() == ".zip":
        return _extract_stardist_archive(model_path, model_type)

    return None


def _load_stardist_model(model_type: str, model_chkpt: Path | str, model_axes: str):
    """Load StarDist model from the pipeline-managed checkpoint.

    Args:
        model_type: Model type/version to use
        model_chkpt: Path to the downloaded checkpoint artifact
        model_axes: Expected model axes

    Returns:
        Loaded StarDist model
    """
    model_class = _get_model_class(model_axes)
    model_dir = _resolve_stardist_model_dir(model_chkpt, model_type)

    if model_dir is None:
        raise FileNotFoundError(
            f"Could not resolve a downloaded StarDist model from checkpoint artifact: {model_chkpt}"
        )

    print(f"Loading StarDist model from local directory: {model_dir}")
    return model_class(None, name=model_dir.name, basedir=str(model_dir.parent))


def _get_model_class(model_axes: str):
    return StarDist3D if _spatial_ndim(model_axes) == 3 else StarDist2D


def _get_prediction_n_tiles(model, img: np.ndarray, config: dict):
    """Use configured tiling if provided, otherwise ask StarDist to choose."""
    n_tiles = config.get("n_tiles")
    if n_tiles is not None:
        return tuple(n_tiles)

    if hasattr(model, "_guess_n_tiles"):
        return model._guess_n_tiles(img)  # pylint: disable=protected-access

    return None


def _normalise_axes(axes: str | None) -> str | None:
    if not isinstance(axes, str):
        return None
    axes = "".join(ch for ch in axes.upper() if ch.isalpha())
    return axes or None


def _get_model_axes(config: dict) -> str:
    """Get the expected input axes for the selected model version."""
    axes = _normalise_axes(config.get("axes"))
    if axes is None:
        raise ValueError(
            "StarDist model axes are missing from the model config. Please provide 'axes' in the selected model version."
        )
    return axes


def _spatial_ndim(axes: str) -> int:
    return sum(axis != "C" for axis in axes)


def _infer_input_axes(channels: int, num_slices: int) -> str:
    has_channels = channels is not None and int(channels) > 1
    has_z = num_slices is not None and int(num_slices) > 1

    axis_lookup = {
        (False, False): "YX",
        (False, True): "ZYX",
        (True, False): "CYX",
        (True, True): "CZYX",
    }
    return axis_lookup[(has_channels, has_z)]


def _compact_loaded_image(img: np.ndarray, channels: int, num_slices: int) -> tuple[np.ndarray, str]:
    input_axes = _infer_input_axes(channels, num_slices)

    if input_axes == "CZYX":
        return img, input_axes
    if input_axes == "CYX":
        return img[:, 0, :, :], input_axes
    if input_axes == "ZYX":
        return img[0, :, :, :], input_axes
    return img[0, 0, :, :], input_axes


def _transpose_to_axes(img: np.ndarray, source_axes: str, target_axes: str) -> np.ndarray:
    if source_axes == target_axes:
        return img
    if sorted(source_axes) != sorted(target_axes):
        raise ValueError(f"Cannot transpose from axes {source_axes} to incompatible axes {target_axes}.")
    axis_order = [source_axes.index(axis) for axis in target_axes]
    return np.transpose(img, axes=axis_order)


def _select_channel(
    img: np.ndarray,
    input_axes: str,
    channel_idx: int,
) -> tuple[np.ndarray, str]:
    if "C" not in input_axes:
        return img, input_axes

    # -1 means use the original image as-is (no channel extraction)
    if channel_idx == -1:
        return img, input_axes

    channel_axis = input_axes.index("C")
    num_channels = img.shape[channel_axis]
    if channel_idx < 0 or channel_idx >= num_channels:
        raise ValueError(f"channel_idx={channel_idx} exceeds available channels={num_channels}.")

    img = np.take(img, indices=channel_idx, axis=channel_axis)
    return img, input_axes.replace("C", "")


def _prepare_input_for_model(
    img: np.ndarray,
    input_axes: str,
    model_axes: str,
    channel_idx: int,
) -> tuple[np.ndarray, str, bool]:
    """Prepare image data to match the selected model axes.

    Returns:
        prepared_img: Input ready for direct prediction, or the full stack for slice-wise 2D.
        prepared_axes: Axes of prepared_img.
        run_over_slices: Whether a 2D model should be applied slice-by-slice over Z.
    """
    prepared_img = img
    prepared_axes = input_axes

    if "C" in prepared_axes and "C" not in model_axes:
        if channel_idx == -1:
            raise ValueError(
                f"Model axes {model_axes} has no channel axis, but channel_idx=-1 (original) "
                f"would keep {prepared_axes}. Select a specific channel index (0 to N-1)."
            )
        prepared_img, prepared_axes = _select_channel(prepared_img, prepared_axes, channel_idx)
    elif "C" not in prepared_axes and "C" in model_axes:
        raise ValueError(f"Model expects channel axis ({model_axes}) but input data axes are {input_axes}.")

    model_spatial_ndim = _spatial_ndim(model_axes)
    input_spatial_ndim = _spatial_ndim(prepared_axes)

    if input_spatial_ndim < model_spatial_ndim:
        raise ValueError(
            f"Model expects {model_axes} input, but received lower-dimensional data with axes {prepared_axes}."
        )

    if input_spatial_ndim == model_spatial_ndim:
        prepared_img = _transpose_to_axes(prepared_img, prepared_axes, model_axes)
        return prepared_img, model_axes, False

    if model_spatial_ndim == 2 and input_spatial_ndim == 3 and "Z" in prepared_axes:
        return prepared_img, prepared_axes, True

    raise ValueError(f"Cannot use model axes {model_axes} with input data axes {prepared_axes}.")


def _predict_instances(
    img: np.ndarray,
    model,
    config: dict,
) -> np.ndarray:
    """Normalize the input image and run StarDist prediction."""
    normalize_pmin = config.get("normalize_pmin", 1)
    normalize_pmax = config.get("normalize_pmax", 99.8)
    img_normalized = normalize(img, normalize_pmin, normalize_pmax) if config.get("normalize_img", True) else img

    prob_thresh = config.get("prob_thresh")
    nms_thresh = config.get("nms_thresh")
    n_tiles = _get_prediction_n_tiles(model, img_normalized, config)
    scale = config.get("scale")

    labels, _ = model.predict_instances(
        img_normalized,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        n_tiles=n_tiles,
        scale=scale,
    )
    return labels


def _run_stardist_2d_stack(
    img_stack: np.ndarray,
    stack_axes: str,
    model,
    config: dict,
    model_axes: str,
) -> np.ndarray:
    """Run 2D StarDist prediction on each slice of a 3D stack.

    Args:
        img_stack: Image stack with a Z axis and optional channel axis
        model: StarDist2D model
        config: Configuration dictionary containing model parameters

    Returns:
        3D label image (ZYX)
    """
    z_axis = stack_axes.index("Z")
    slice_axes = stack_axes.replace("Z", "")
    num_slices = img_stack.shape[z_axis]

    print(f"Running 2D model on {num_slices} slices...")
    all_labels = []

    for z_idx in range(num_slices):
        img_slice = np.take(img_stack, indices=z_idx, axis=z_axis)
        img_slice = _transpose_to_axes(img_slice, slice_axes, model_axes)
        all_labels.append(_predict_instances(img_slice, model, config))

    return np.stack(all_labels, axis=0)


def run_stardist(
    save_dir: Path | str,
    save_name: str,
    idxs: list[int],
    img: np.ndarray,
    model_type: str,
    model_chkpt: Path | str,
    config: dict,
    channels: int,
    num_slices: int,
    channel_idx: int = 0,
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
        channels: Number of image channels in the source data
        num_slices: Number of Z slices in the source data
        channel_idx: Channel index to use when channel selection is required
    """
    save_dir = Path(save_dir)
    model_axes = _get_model_axes(config)

    print(f"Running StarDist with model axes {model_axes}...")
    print(f"Loaded image shape (CZYX): {img.shape}")

    model = _load_stardist_model(model_type, model_chkpt, model_axes)
    print(f"Model loaded: {model_type}")
    compact_img, input_axes = _compact_loaded_image(img, channels, num_slices)
    print(f"Detected input data axes: {input_axes}; compact shape: {compact_img.shape}")

    prepared_img, prepared_axes, run_over_slices = _prepare_input_for_model(
        compact_img,
        input_axes,
        model_axes,
        channel_idx,
    )

    channel_desc = "original (all channels)" if channel_idx == -1 else f"channel {channel_idx}"
    print(f"Running inference on {channel_desc}; prepared axes: {prepared_axes}, shape: {prepared_img.shape}")

    if run_over_slices:
        print(f"Running 2D model slice-by-slice over Z axis.")
        labels = _run_stardist_2d_stack(
            prepared_img,
            prepared_axes,
            model,
            config,
            model_axes,
        )
    else:
        labels = _predict_instances(prepared_img, model, config)

    print(f"Segmentation complete. Labels shape: {labels.shape}, unique labels: {len(np.unique(labels))}")

    save_masks(save_dir, save_name, labels, idxs=idxs, mask_type="instance")


if __name__ == "__main__":
    parser = create_argparser_inference()
    parser.add_argument(
        "--channel-idx",
        type=int,
        default=0,
        help="Channel index to use for multi-channel images (default: 0)",
    )
    cli_args = parser.parse_args()

    with open(cli_args.model_config) as f:
        config = yaml.safe_load(f)

    # Load image and apply preprocessing if specified
    img = load_img(
        fpath=cli_args.img_path,
        idxs=cli_args.idxs,
        channels=cli_args.channels,
        num_slices=cli_args.num_slices,
        dim_order="CZYX",
    )

    print(f"Input data metadata: channels={cli_args.channels}, num_slices={cli_args.num_slices}")

    run_stardist(
        save_dir=cli_args.output_dir,
        save_name=cli_args.mask_fname,
        idxs=cli_args.idxs,
        img=img,
        model_type=cli_args.model_type,
        model_chkpt=cli_args.model_chkpt,
        config=config,
        channels=cli_args.channels,
        num_slices=cli_args.num_slices,
        channel_idx=config.get("channel_idx", cli_args.channel_idx),
    )
