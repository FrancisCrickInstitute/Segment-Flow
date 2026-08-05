import shutil
import zipfile
from pathlib import Path

import numpy as np
import yaml
from csbdeep.utils import normalize
from stardist.models import StarDist2D, StarDist3D
from utils import create_argparser_inference, load_img, save_masks

STARDIST_MODEL_FILES = ("config.json", "thresholds.json")


def _find_stardist_model_dir(search_root: Path) -> Path:
    """Find the extracted StarDist model directory containing config files."""
    if search_root.is_dir():
        if any((search_root / fname).exists() for fname in STARDIST_MODEL_FILES):
            return search_root

        for candidate in sorted(search_root.rglob("config.json")):
            if candidate.parent.is_dir():
                return candidate.parent

    raise FileNotFoundError(
        f"Could not find extracted StarDist model files in {search_root}."
    )


def _extract_stardist_archive(archive_path: Path, model_type: str) -> Path:
    """Extract a StarDist archive into a cache directory and return the model directory."""
    # TODO: Look to handle unzipping at setup to avoid per-substack unzipping
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


def _resolve_stardist_model_dir(
    model_chkpt: Path | str, model_type: str
) -> Path | None:
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
    model_class = StarDist3D if _spatial_ndim(model_axes) == 3 else StarDist2D
    model_dir = _resolve_stardist_model_dir(model_chkpt, model_type)

    if model_dir is None:
        raise FileNotFoundError(
            f"Could not resolve a downloaded StarDist model from checkpoint artifact: {model_chkpt}"
        )

    print(f"Loading StarDist model from local directory: {model_dir}")
    return model_class(None, name=model_dir.name, basedir=str(model_dir.parent))


def _spatial_ndim(axes: str) -> int:
    return sum(axis != "C" for axis in axes)


def _resolve_model_axes(raw: str | None) -> str:
    """Validate the model axes piped in from the registry via setupModel."""
    # TODO: Delete this func once we centralise model<->input validation into it's own process
    if not raw:
        raise ValueError(
            "No model axes supplied via --model-axes. This should be resolved "
            "from the registry model version's 'axes' field at the setupModel stage."
        )
    axes = raw.upper()
    unsupported = set(axes) - set("CZYX")
    if unsupported:
        raise ValueError(
            f"Unsupported model axes {axes!r}: cannot handle {sorted(unsupported)}."
        )
    return axes


def _resolve_channel(
    img: np.ndarray, model_axes: str, channels: int, channel_idx: int
) -> tuple[np.ndarray, str]:
    """Reduce the loaded CZYX image's channel axis to what the model needs.

    Returns the (possibly channel-reduced) image and its current axis order:
    'CZYX' if the model wants a channel axis, otherwise 'ZYX'.
    """
    if "C" in model_axes:
        return img, "CZYX"
    if channels > 1:
        if channel_idx < 0 or channel_idx >= channels:
            raise ValueError(
                f"Image has {channels} channels but model axes {model_axes!r} have "
                f"no channel axis. Select a channel index (0 to {channels - 1}), "
                f"got {channel_idx}."
            )
        return img[channel_idx], "ZYX"
    return img[0], "ZYX"


def _transpose_to_axes(
    img: np.ndarray, source_axes: str, target_axes: str
) -> np.ndarray:
    if source_axes == target_axes:
        return img
    return np.transpose(img, axes=[source_axes.index(a) for a in target_axes])


def _get_prediction_n_tiles(model, img: np.ndarray, config: dict):
    """Use configured tiling if provided, otherwise ask StarDist to choose."""
    n_tiles = config.get("n_tiles")
    if n_tiles is not None:
        return tuple(n_tiles)

    if hasattr(model, "_guess_n_tiles"):
        return model._guess_n_tiles(img)  # noqa: SLF001

    return None


def _predict_instances(img: np.ndarray, model, config: dict) -> np.ndarray:
    """Normalize the input image and run StarDist prediction."""
    normalize_pmin = config.get("normalize_pmin", 1)
    normalize_pmax = config.get("normalize_pmax", 99.8)
    img_normalized = (
        normalize(img, normalize_pmin, normalize_pmax)
        if config.get("normalize_img", True)
        else img
    )

    labels, _ = model.predict_instances(
        img_normalized,
        prob_thresh=config.get("prob_thresh"),
        nms_thresh=config.get("nms_thresh"),
        n_tiles=_get_prediction_n_tiles(model, img_normalized, config),
        scale=config.get("scale"),
    )
    return labels


def run_stardist(
    save_dir: Path | str,
    save_name: str,
    idxs: list[int],
    img: np.ndarray,
    model_type: str,
    model_chkpt: Path | str,
    model_axes: str,
    config: dict,
    channels: int,
    num_slices: int,
    channel_idx: int = -1,
    output_mask_type: str = "instance",
):
    """Run StarDist segmentation pipeline.

    Args:
        save_dir: Directory to save the output masks
        save_name: Base name for saved files
        idxs: Slice indices being processed
        img: Input image array, loaded as CZYX
        model_type: Model type/version to use
        model_chkpt: Path to model checkpoint directory or zip archive
        model_axes: Axes the model expects (e.g. 'YX', 'ZYX', 'YXC'), resolved
            from the registry at the setupModel stage
        config: Configuration dictionary containing model parameters
        channels: Number of channels in the source image
        num_slices: Number of Z slices in the source image
        channel_idx: Channel to select when the model has no channel axis (-1 = none)
        output_mask_type: Mask type to save ('binary', 'instance')
    """
    save_dir = Path(save_dir)
    print(f"Loaded image shape (CZYX): {img.shape}")

    if "Z" in model_axes and num_slices <= 1:
        # TODO: Delete this part once we centralise model<->input validation into it's own process
        raise ValueError(
            f"Model expects {model_axes} (3D) input, but the image has only 1 Z slice."
        )

    model = _load_stardist_model(model_type, model_chkpt, model_axes)
    print(f"Model loaded: {model_type}; expects axes {model_axes}")

    img, axes = _resolve_channel(img, model_axes, channels, channel_idx)

    # Flag for (spatial) 2D model with (spatial) 3D data
    run_over_slices = "Z" not in model_axes and num_slices > 1

    if run_over_slices:
        z_axis = axes.index("Z")
        slice_axes = axes.replace("Z", "")
        print(f"Running 2D model slice-by-slice over {img.shape[z_axis]} Z slices...")
        labels = np.stack(
            [
                _predict_instances(
                    _transpose_to_axes(
                        np.take(img, indices=z_idx, axis=z_axis), slice_axes, model_axes
                    ),
                    model,
                    config,
                )
                for z_idx in range(img.shape[z_axis])
            ],
            axis=0,
        )
    else:
        # Remove placeholder Z for (spatial) 2D data
        if "Z" not in model_axes:
            img = np.take(img, indices=0, axis=axes.index("Z"))
            axes = axes.replace("Z", "")
        img = _transpose_to_axes(img, axes, model_axes)
        labels = _predict_instances(img, model, config)

    print(
        f"Segmentation complete. Labels shape: {labels.shape}, unique labels: {len(np.unique(labels))}"
    )

    save_masks(save_dir, save_name, labels, idxs=idxs, mask_type=output_mask_type)


if __name__ == "__main__":
    parser = create_argparser_inference()
    cli_args = parser.parse_args()

    with open(cli_args.model_config) as f:
        config = yaml.safe_load(f)

    model_axes = _resolve_model_axes(cli_args.model_axes)

    # Load as CZYX, like every other model script; reshaped in run_stardist() to
    # match model_axes.
    img = load_img(
        fpath=cli_args.img_path,
        idxs=cli_args.idxs,
        channels=cli_args.channels,
        num_slices=cli_args.num_slices,
        dim_order="CZYX",
    )

    print(
        f"Input data metadata: channels={cli_args.channels}, num_slices={cli_args.num_slices}"
    )

    run_stardist(
        save_dir=cli_args.output_dir,
        save_name=cli_args.mask_fname,
        idxs=cli_args.idxs,
        img=img,
        model_type=cli_args.model_type,
        model_chkpt=cli_args.model_chkpt,
        model_axes=model_axes,
        config=config,
        channels=cli_args.channels,
        num_slices=cli_args.num_slices,
        channel_idx=config.get("channel_idx", -1),
        output_mask_type=cli_args.output_mask_type
        if cli_args.output_mask_type != "auto"
        else "instance",
    )
