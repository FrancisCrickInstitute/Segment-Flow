"""Run InstanSeg (https://github.com/instanseg/instanseg) within Segment-Flow.

InstanSeg is a 2D method, so a substack carrying several Z slices is segmented
slice by slice, as run_stardist.py does for its 2D model variants.

Three InstanSeg-specific wrinkles are handled here:

  * Checkpoints with two output heads (nuclei and cells) map one head onto one
    AIoD task, so --task selects it. A cells target is silently ignored on a
    single-head checkpoint, which would file nuclei masks as cytoplasm, so that
    pairing is rejected outright.
  * A 2D model numbers each slice's instances from 1 independently. Nothing
    downstream separates aliased labels -- relabel_sequential remaps the value
    set rather than spatially disjoint components, the RLE round-trip preserves
    label values exactly, and combine_stacks only offsets between substacks --
    so the slices are offset here before stacking.
  * InstanSeg infers the channel axis as the smallest one and squeezes away
    singleton axes, so each slice is handed over as one explicit 2D plane
    rather than a stack, and the geometry is checked before and after.

Note that params.overlap must stay at 0: a non-zero overlap sends
combine_stacks down its summing path, which is meaningless for instance labels.
"""

import io
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import yaml
from utils import (
    create_argparser_inference,
    load_img,
    resolve_channel,
    resolve_model_axes,
    save_masks,
    transpose_to_axes,
)

TORCHSCRIPT_NAME = "instanseg.pt"
# Every published model declares y/x size {min: 32, step: 1} in its rdf.yaml
MIN_SPATIAL = 32
# _sliding_window_inference overlaps tiles by 2 * (80 + 20) px and asserts the
# tile is larger than that
MIN_TILE_SIZE = 256
# Channel counts each published checkpoint accepts, from its rdf.yaml. None
# means the network is channel-invariant and takes any number.
EXPECTED_CHANNELS = {
    "brightfield_nuclei": 3,
    "single_channel_nuclei": 1,
    "fluorescence_nuclei_and_cells": None,
}
# Post-processing kwargs the scripted InstanSeg_Torchscript.forward accepts.
# InstanSeg filters kwargs by parsing its own graph and drops unknown names
# without complaint, so keep this explicit and log what is actually sent.
INSTANSEG_KWARGS = ("mask_threshold", "min_size")
TASK_TO_TARGET = {"nuclei": "nuclei", "cyto": "cells"}


def _find_member(archive: zipfile.ZipFile, name: str) -> str:
    """Locate a file at the archive root, falling back to any nesting depth."""
    names = archive.namelist()
    if name in names:
        return name
    nested = sorted(n for n in names if PurePosixPath(n).name == name)
    if not nested:
        raise FileNotFoundError(
            f"{name} not found in {archive.filename}; archive contains: {names[:20]}"
        )
    return nested[0]


def _load_torchscript(model_chkpt: Path | str) -> torch.jit.ScriptModule:
    """Resolve the staged checkpoint artifact into a loaded TorchScript module.

    Handles the release .zip, an already-extracted directory, and a bare .pt.

    The zip is read in memory rather than extracted: --model-chkpt is a symlink
    Nextflow stages into the per-task work directory, so extracting beside it
    would repeat the work for every substack.
    """
    path = Path(model_chkpt)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    if path.is_dir():
        candidates = sorted(path.rglob(TORCHSCRIPT_NAME))
        if not candidates:
            raise FileNotFoundError(f"No {TORCHSCRIPT_NAME} found under {path}")
        return torch.jit.load(str(candidates[0]), map_location="cpu")

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            weights = io.BytesIO(archive.read(_find_member(archive, TORCHSCRIPT_NAME)))
        return torch.jit.load(weights, map_location="cpu")

    return torch.jit.load(str(path), map_location="cpu")


def _resolve_target(task: str | None, cells_and_nuclei: bool) -> str:
    """Map the AIoD task onto InstanSeg's target, rejecting impossible pairings.

    InstanSeg only honours `target` when the network has both heads, so asking a
    nuclei-only checkpoint for cells would quietly return nuclei instead.
    """
    if not cells_and_nuclei:
        if task == "cyto":
            raise ValueError(
                "Task 'cyto' was requested but this InstanSeg checkpoint has a "
                "single output head, so it can only produce nuclei. Use the "
                "fluorescence_nuclei_and_cells version for whole cells."
            )
        return "all_outputs"
    if task not in TASK_TO_TARGET:
        raise ValueError(
            f"Task {task!r} is not supported by InstanSeg. This checkpoint "
            f"segments nuclei and cells; expected one of {sorted(TASK_TO_TARGET)}."
        )
    return TASK_TO_TARGET[task]


def _resolve_channel_ids(raw, n_channels: int) -> list[int] | None:
    """Validate the optional channel subset ahead of InstanSeg.

    InstanSeg's own check is off-by-one (it allows an index equal to the channel
    count), which would fail later with a bare IndexError.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, str):
        raw = [tok for tok in raw.replace(" ", "").split(",") if tok]
    ids = [int(i) for i in raw]
    out_of_range = [i for i in ids if i < 0 or i >= n_channels]
    if out_of_range:
        raise ValueError(
            f"Channel indices {out_of_range} are out of range for a "
            f"{n_channels}-channel image (valid: 0 to {n_channels - 1})."
        )
    return ids


def _check_channels(
    model_type: str, model_axes: str, channels: int, img: np.ndarray
) -> np.ndarray:
    """Check the image's channel count against what this checkpoint accepts.

    Only meaningful when the network takes a channel axis. Without one a single
    channel is picked upstream by resolve_channel, so the image's own channel
    count says nothing about what the network will see.

    RGBA is tolerated for the 3-channel model by dropping the alpha channel,
    since load_img treats a 4th sample as a channel.
    """
    expected = EXPECTED_CHANNELS.get(model_type)
    if "C" not in model_axes or expected is None or channels == expected:
        return img
    if expected == 3 and channels == 4:
        print("Image has 4 channels; dropping the last one as alpha for RGB input")
        return img[:3]
    raise ValueError(
        f"InstanSeg version {model_type!r} needs exactly {expected} channel(s) "
        f"but the image has {channels}. Choose a version matching your data: "
        "brightfield_nuclei for RGB histology, single_channel_nuclei for one "
        "fluorescence channel, fluorescence_nuclei_and_cells for any number."
    )


def _check_plane(plane: np.ndarray, model_axes: str) -> tuple[int, int]:
    """Validate one 2D plane's geometry, returning its (height, width).

    Guards two InstanSeg behaviours that fail silently rather than loudly: it
    picks the channel axis as the smallest axis, and it squeezes singleton axes.
    """
    channels, (height, width) = (
        (plane.shape[0], plane.shape[1:]) if "C" in model_axes else (1, plane.shape)
    )
    if min(height, width) < MIN_SPATIAL:
        raise ValueError(
            f"Substack slice is {height}x{width}, below InstanSeg's minimum of "
            f"{MIN_SPATIAL}x{MIN_SPATIAL}. Raise params.model_max_substack or "
            "params.memory_per_job so substacks are larger."
        )
    if channels >= min(height, width):
        raise ValueError(
            f"Image has {channels} channels but each slice is only "
            f"{height}x{width}. InstanSeg infers the channel axis as the "
            "smallest one, so it would transpose this image. Use larger "
            "substacks, or select fewer channels with 'channel_ids'."
        )
    return height, width


def _build_eval_kwargs(config: dict) -> dict:
    """Collect the post-processing kwargs InstanSeg accepts.

    A None means "use the value baked into this checkpoint", which differs
    between versions, so those are dropped rather than forwarded.
    """
    kwargs = {
        name: config[name] for name in INSTANSEG_KWARGS if config.get(name) is not None
    }
    print(f"InstanSeg post-processing kwargs: {kwargs or 'checkpoint defaults'}")
    return kwargs


def _resolve_pixel_size(raw) -> float | None:
    """Coerce the configured pixel size, where blank means "do not rescale".

    A user-supplied config can carry a string, and a non-positive value would
    reach InstanSeg as a scale factor and quietly ruin the output, so both are
    handled here rather than downstream.
    """
    if raw is None or (isinstance(raw, str) and raw.strip() in ("", "None", "none")):
        return None
    try:
        pixel_size = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"pixel_size must be a number in microns per pixel, got {raw!r}."
        ) from exc
    if pixel_size <= 0:
        raise ValueError(
            f"pixel_size must be greater than 0, got {pixel_size}. Leave it blank "
            "to segment the image without rescaling."
        )
    return pixel_size


def _resolve_tile_size(config: dict) -> int:
    tile_size = int(config.get("tile_size") or 512)
    if tile_size < MIN_TILE_SIZE:
        raise ValueError(
            f"tile_size must be at least {MIN_TILE_SIZE} (got {tile_size}); "
            "InstanSeg overlaps tiles by 200 px and rejects anything smaller."
        )
    return tile_size


def _predict_plane(model, plane: np.ndarray, model_axes: str, config: dict, **shared):
    """Segment one 2D plane, returning an (H, W) integer label image."""
    height, width = _check_plane(plane, model_axes)

    method = config.get("processing_method") or "auto"
    if method == "auto":
        # Mirrors InstanSeg's own dispatch. The small path normalises on the
        # device, where torch.quantile rejects tensors above 2**24 elements, so
        # the threshold is a memory guard as much as a speed one.
        method = "small" if plane.size < model.small_image_threshold else "medium"

    if method == "small":
        instances = model.eval_small_image(plane, **shared)
    elif method == "medium":
        instances = model.eval_medium_image(
            plane,
            tile_size=_resolve_tile_size(config),
            batch_size=int(config.get("batch_size") or 1),
            normalisation_subsampling_factor=int(
                config.get("normalisation_subsampling_factor") or 1
            ),
            **shared,
        )
    else:
        raise ValueError(
            f"Unsupported processing_method {method!r}; expected 'auto', "
            "'small' or 'medium'. InstanSeg's whole-slide path needs a file "
            "path and extra dependencies, so it is not offered here."
        )

    labels = instances.numpy()
    if labels.shape[:2] != (1, 1):
        raise RuntimeError(
            f"Expected a single (1, 1, H, W) output but got {labels.shape}; "
            "more than one output head was active."
        )
    labels = labels[0, 0]
    if labels.shape != (height, width):
        raise RuntimeError(
            f"InstanSeg returned labels of shape {labels.shape} for a "
            f"{height}x{width} input, so the channel axis was misread."
        )
    return labels


def _stack_slices(slices: list[np.ndarray]) -> np.ndarray:
    """Stack 2D label images into ZYX, keeping label IDs unique across slices.

    Each slice is numbered from 1 independently, so a plain stack would alias
    distinct objects onto one label. save_masks() compacts the offset IDs again
    with relabel_sequential.
    """
    stacked = []
    running_max = 0
    for labels in slices:
        labels = np.asarray(labels, dtype=np.int64).copy()
        foreground = labels > 0
        if foreground.any():
            labels[foreground] += running_max
            running_max = int(labels.max())
        stacked.append(labels)
    return np.stack(stacked, axis=0)


def run_instanseg(
    save_dir: Path | str,
    save_name: str,
    idxs: list[int],
    img: np.ndarray,
    task: str | None,
    model_type: str,
    model_chkpt: Path | str,
    model_axes: str,
    config: dict,
    channels: int,
    channel_idx: int = -1,
    output_mask_type: str = "instance",
):
    """Run the InstanSeg segmentation pipeline.

    Args:
        save_dir: Directory to save the output masks
        save_name: Base name for saved files
        idxs: Slice indices being processed
        img: Input image array, loaded as CZYX
        task: AIoD task this run was requested for, selecting the output head
        model_type: Model type/version to use
        model_chkpt: Path to the downloaded checkpoint artifact
        model_axes: Axes the model expects, resolved from the registry
        config: Configuration dictionary containing model parameters
        channels: Number of channels in the source image
        channel_idx: Channel to select when the model has no channel axis
        output_mask_type: Mask type to save ('binary', 'instance')
    """
    from instanseg import InstanSeg

    save_dir = Path(save_dir)
    print(f"Loaded image shape (CZYX): {img.shape}")

    if "Z" in model_axes:
        raise ValueError(
            f"InstanSeg is a 2D method, but registry axes {model_axes!r} declare "
            "a Z axis."
        )

    module = _load_torchscript(model_chkpt)
    target = _resolve_target(task, bool(module.cells_and_nuclei))
    device = config.get("device") or "auto"
    # InstanSeg picks cuda, then MPS, then CPU, and its scripted graph moves the
    # operations MPS lacks onto the CPU itself. model_utils.get_device() is
    # deliberately not used here as it would resolve MPS to CPU.
    model = InstanSeg(
        model_type=module,
        device=None if device == "auto" else device,
        verbosity=1,
    )
    print(
        f"InstanSeg {model_type}: cells_and_nuclei={bool(module.cells_and_nuclei)}, "
        f"trained at {module.pixel_size} um/px, task={task} -> target={target}"
    )

    img = _check_channels(model_type, model_axes, channels, img)
    channels = img.shape[0]
    img, axes = resolve_channel(img, model_axes, channels, channel_idx)

    shared = {
        "pixel_size": _resolve_pixel_size(config.get("pixel_size")),
        "normalise": bool(config.get("normalise", True)),
        "rescale_output": bool(config.get("rescale_output", True)),
        "target": target,
        # Defaults to True, which would return a copy of the input alongside
        "return_image_tensor": False,
        **_build_eval_kwargs(config),
    }
    channel_ids = (
        _resolve_channel_ids(config.get("channel_ids"), channels)
        if "C" in model_axes
        else None
    )
    if channel_ids is not None:
        shared["channel_ids"] = channel_ids
    if shared["pixel_size"] is None:
        print(
            "No pixel size given, so the image is segmented as-is. Set it if the "
            f"image is not already near {module.pixel_size} um/px."
        )

    z_axis = axes.index("Z")
    plane_axes = axes.replace("Z", "")
    method = config.get("processing_method") or "auto"
    print(f"Segmenting {img.shape[z_axis]} slice(s) with the {method!r} method")
    labels = _stack_slices(
        [
            _predict_plane(
                model,
                transpose_to_axes(
                    np.take(img, indices=z_idx, axis=z_axis), plane_axes, model_axes
                ),
                model_axes,
                config,
                **shared,
            )
            for z_idx in range(img.shape[z_axis])
        ]
    )

    print(
        f"Segmentation complete. Labels shape: {labels.shape}, "
        f"unique labels: {len(np.unique(labels))}"
    )

    save_masks(save_dir, save_name, labels, idxs=idxs, mask_type=output_mask_type)


if __name__ == "__main__":
    parser = create_argparser_inference()
    cli_args = parser.parse_args()

    with open(cli_args.model_config) as f:
        config = yaml.safe_load(f)

    model_axes = resolve_model_axes(cli_args.model_axes)

    # Load as CZYX, like every other model script; each slice is reshaped to
    # match model_axes in run_instanseg().
    img = load_img(
        fpath=cli_args.img_path,
        idxs=cli_args.idxs,
        channels=cli_args.channels,
        num_slices=cli_args.num_slices,
        dim_order="CZYX",
    )

    print(
        f"Input data metadata: channels={cli_args.channels}, "
        f"num_slices={cli_args.num_slices}, task={cli_args.task}"
    )

    run_instanseg(
        save_dir=cli_args.output_dir,
        save_name=cli_args.mask_fname,
        idxs=cli_args.idxs,
        img=img,
        task=cli_args.task,
        model_type=cli_args.model_type,
        model_chkpt=cli_args.model_chkpt,
        model_axes=model_axes,
        config=config,
        channels=cli_args.channels,
        channel_idx=config.get("channel_idx", -1),
        output_mask_type=cli_args.output_mask_type
        if cli_args.output_mask_type != "auto"
        else "instance",
    )
