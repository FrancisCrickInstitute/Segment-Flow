import os
import tempfile
from pathlib import Path

import aiod_utils.rle as aiod_rle
import dask
import dask.array as da
import dask_image.ndmeasure
import numpy as np
import psutil
import skimage.measure
import tifffile
import zarr
from aiod_utils.io import check_dtype, extract_idxs_from_fname, reduce_dtype
from aiod_utils.preprocess import get_downsample_factor
from numba import jit, prange
from numba.core import types
from numba.typed import Dict
from skimage.segmentation import relabel_sequential
from tqdm import tqdm

# Default number of slices read/written at a time for the Zarr-backed combined-mask
# store, so a max/page scan never materialises more than this many slices. Overridable
# via --slab-size. TODO: calculate a reasonable size based on memory and shape similar to compute_max_substack_size
SLAB_SIZE = 64


def _log_mem(label: str):
    """Print current RSS, to track that the memory fix actually holds at runtime."""
    mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
    print(f"Memory used {label}: {mem_used:.2f} GB")


def _open_store(path, shape, dtype, chunks):
    """Create a chunked, compressed Zarr array on local scratch to hold a mask volume."""
    shape = tuple(int(s) for s in shape)
    # Never chunk larger than the array itself
    chunks = tuple(min(c, s) for c, s in zip(chunks[-len(shape) :], shape, strict=True))
    return zarr.open_array(
        store=str(path), mode="w", shape=shape, chunks=chunks, dtype=dtype
    )


def iter_slabs(arr, slab=SLAB_SIZE):
    """Yield blocks over the leading axis, as in-memory numpy arrays.

    Works for both numpy and Zarr arrays. 2D arrays are yielded whole.
    """
    if arr.ndim == 2:
        yield np.asarray(arr)
        return
    for z0 in range(0, arr.shape[0], slab):
        z1 = min(z0 + slab, arr.shape[0])
        yield np.asarray(arr[z0:z1])


def iter_pages(arr, transform=None, slab=SLAB_SIZE):
    """Yield individual 2D pages, reading in slabs so a Zarr store isn't hit per-slice."""
    for block in iter_slabs(arr, slab):
        if transform is not None:
            block = transform(block)
        if block.ndim == 2:
            yield block
        else:
            yield from block  # N - Yielding slab at at time


def write_blocks(store, block, origin, add=False, slab=SLAB_SIZE):
    """Write `block` into `store` at `origin`, a z-slab at a time."""
    if len(origin) == 2:
        y0, x0 = origin
        sl = (
            slice(
                y0, y0 + block.shape[0]
            ),  # N - create a slice object from origin to the size of the stack in y
            slice(x0, x0 + block.shape[1]),  # N - same for x
        )
        store[sl] = (store[sl] + block) if add else block
        return
    z0, y0, x0 = origin
    ys = slice(y0, y0 + block.shape[1])
    xs = slice(x0, x0 + block.shape[2])
    for k in range(0, block.shape[0], slab):
        k1 = min(k + slab, block.shape[0])
        sl = (slice(z0 + k, z0 + k1), ys, xs)
        store[sl] = (store[sl] + block[k:k1]) if add else block[k:k1]


def slab_max(arr, start=0, end=None, slab=SLAB_SIZE):
    """
    Max over arr[start:end] along the leading axis, read in slabs so the range is
    never fully materialised. `end=None` covers the whole array.
    """
    if end is None:
        end = arr.shape[0]
    best = 0
    for z0 in range(start, end, slab):
        z1 = min(z0 + slab, end)
        block = np.asarray(arr[z0:z1])
        if block.size:
            best = max(best, int(block.max()))
    return best


def resolve_mask_type(masks, mask_type, slab_size=SLAB_SIZE):
    """Resolve the mask type once for the whole volume, without materialising it."""
    if mask_type is not None:
        return mask_type
    if masks.dtype == bool:
        return "binary"
    seen = set()
    for block in iter_slabs(masks, slab_size):
        seen.update(np.unique(block).tolist())
        if len(seen) > 2:
            return "instance"
    return "binary"


def encode_slicewise(masks, mask_type, metadata, slab_size=SLAB_SIZE):
    """Per-slice equivalent of aiod_rle.encode(masks, mask_type, metadata).

    TODO (future iteration): this accumulates every slice's runs in one list and copies
    it per slice via [:-1]. Left as-is because the encoded form is small relative to the
    volume, but it could stream to disk or fan the slices out over a process pool if the
    encode ever shows up in a profile.
    """
    if mask_type is None:
        raise ValueError("mask_type must be resolved before slice-wise encode")
    if masks.ndim == 2:
        return aiod_rle.encode(
            np.asarray(masks), mask_type=mask_type, metadata=metadata
        )
    out = []
    for page in iter_pages(masks, slab=slab_size):
        # Each call appends its own trailing metadata dict, which is dropped here and
        # re-added exactly once at the end.
        out.extend(
            aiod_rle.encode(page, mask_type=mask_type, metadata=dict(metadata))[:-1]
        )
    out.append({"metadata": {**metadata, "mask_type": mask_type}})
    return out


def _peek_mask_type(mask_path):
    """Read the mask type from a patch's trailing metadata, without decoding it."""
    return aiod_rle.load_encoding(mask_path)[-1].get("metadata", {}).get("mask_type")


def combine_masks(
    masks: list[str],
    overlap: list[float, ...],
    image_size: tuple[int, ...],
    model: str,
    store_path: str | Path,
    slab_size: int = SLAB_SIZE,
):
    """Combine masks from each of the substacks into a single array/dataset.

    If overlap is 0, masks are simply inserted at their substack indices. If overlap
    is >0, overlapping regions are summed.

    Returns:
        tuple[zarr.Array, str | None]: Combined mask store and mask type ("binary",
        "instance", or None).

    TODO (future iteration): with overlap == 0 and no postprocessing, the inputs and the
    output are both per-slice RLE, so tiles could be merged in RLE space without ever
    materialising a dense voxel.
    """
    start_x, end_x, start_y, end_y, start_z, end_z = extract_idxs_from_fname(masks[0])
    chunk_shape = (min(slab_size, end_z - start_z), end_y - start_y, end_x - start_x)
    xy_tiling = (
        start_x > 0 or end_x < image_size[1] or start_y > 0 or end_y < image_size[2]
    )
    if image_size[0] == 1:
        image_size = image_size[1:]
        is_2d = True
    else:
        is_2d = False
    overlap = [float(val) for val in overlap]
    is_overlap = sum(overlap) != 0.0
    # Peeked before the store is opened so a binary volume can use a 1-byte dtype
    # Label offsetting is pointless for binary masks encode re-binarises anyway
    is_binary = _peek_mask_type(masks[0]) == "binary"
    # overlap > 0 sums into the store, so bool would clip the vote count
    dtype = np.bool_ if is_binary and not is_overlap else np.uint16
    relabel = xy_tiling and not is_binary
    all_masks = _open_store(store_path, image_size, dtype, chunks=chunk_shape)
    mask_types_seen = set()
    # (start_z, end_z, max_label) per substack inserted so far; see insert_mask.
    label_ranges = []

    # TODO: for overlap > 0, handle binary/labelled masks properly with a specified vote
    # mechanism rather than a plain sum.
    for mask_path in masks:
        idxs = extract_idxs_from_fname(mask_path)
        # TODO: Perslab decoding?
        mask, metadata = aiod_rle.decode(aiod_rle.load_encoding(mask_path))
        current_mask_type = metadata.get("metadata", {}).get("mask_type")
        if current_mask_type is not None:
            mask_types_seen.add(current_mask_type)
        # Cast boolean to allow addition
        if mask.dtype == bool and all_masks.dtype != bool:
            mask = mask.astype(np.uint8)
        all_masks = insert_mask(
            all_masks=all_masks,
            mask=mask,
            idxs=idxs,
            relabel=relabel,
            is_2d=is_2d,
            add=is_overlap,
            slab_size=slab_size,
            label_ranges=label_ranges,
        )

    if len(mask_types_seen) > 1:
        raise ValueError(
            f"Inconsistent mask types found across mask files: {mask_types_seen}. "
            "All mask files must have the same mask type."
        )
    mask_type = mask_types_seen.pop() if mask_types_seen else None

    return all_masks, mask_type


def insert_mask(
    all_masks,
    mask,
    idxs: tuple[int, int, int, int, int, int],
    relabel: bool,
    is_2d: bool,
    add: bool = False,
    slab_size: int = SLAB_SIZE,
    label_ranges: list[tuple[int, int, int]] | None = None,
):
    start_x, end_x, start_y, end_y, start_z, end_z = idxs
    # Ensure labels are unique across a slice
    if relabel:
        # Max label already placed where this substack lands (anywhere in the store for a
        # 2D volume), taken from what we have written rather than read back with slab_max:
        # the store holds nothing but these substacks, so the max over intersecting
        # records is the same answer, and reading it back dominated large volumes.
        if label_ranges is None:
            label_ranges = []
        max_val = max(
            (m for z0, z1, m in label_ranges if is_2d or (z0 < end_z and start_z < z1)),
            default=0,
        )
        # TODO: Handle the below, why is it commented out? (`mask_max` below now makes
        # the overflow check free -- no second pass over the substack needed.)
        # # Check if we need to upcast the array
        # if max_val + mask.max() > np.iinfo(all_masks.dtype).max:
        #     all_masks = all_masks.astype(np.uint32, copy=False)
        mask_max = int(mask.max()) if mask.size else 0
        # Add a constant to all non-zero values to ensure uniqueness
        # NOTE: Equivalent to `mask[mask > 0] += max_val` but writes in place via `where`,
        # avoiding the fancy-index gather/scatter pair (one fewer substack-sized temporary)
        np.add(mask, max_val, out=mask, where=mask > 0)
        # Empty substack -> mask_max is 0 and nothing was offset, so this records the
        # incoming max unchanged, which is still a correct bound for the range.
        label_ranges.append((start_z, end_z, max_val + mask_max))
    origin = (start_y, start_x) if is_2d else (start_z, start_y, start_x)
    write_blocks(all_masks, mask, origin, add=add, slab=slab_size)
    return all_masks


def connect_components(all_masks, store_path: str | Path):
    """Label connected components across slices, keeping the volume out of RAM.

    Written straight to a Zarr store instead of materialised with .compute(). The
    write and the component count are evaluated in a single dask.compute call so the
    label graph is only traversed once.
    """
    if isinstance(all_masks, np.ndarray):
        arr = da.from_array(all_masks)
    else:
        arr = da.from_zarr(all_masks)
    labelled, num_holes = dask_image.ndmeasure.label(arr)
    # dtype is left as dask_image produced it: aiod_rle.encode reduces per slab, so
    # rewriting the whole store just to downcast it would buy nothing.
    out = _open_store(
        store_path, labelled.shape, labelled.dtype, chunks=labelled.chunksize
    )
    write = da.to_zarr(labelled, out, compute=False)
    _, num_holes = dask.compute(write, num_holes)
    return out, int(num_holes)


@jit(nopython=True, parallel=True, fastmath=True)
def mask_iou_batch(
    box_matches, curr_slice_bool, next_slice_bool, curr_label_dict, next_label_dict
):
    # Initialize the array to store the IoUs
    n = len(box_matches)
    ious = np.zeros(n)
    # Parallel loop over the box matches
    for i in prange(n):
        # Extract the boolean masks for the current and next labels
        curr_label, next_label = box_matches[i]
        curr_mask = curr_slice_bool[..., curr_label_dict[curr_label]]
        next_mask = next_slice_bool[..., next_label_dict[next_label]]
        # Calculate the IoU
        # Inlined here to help numba optimise
        union = np.count_nonzero(np.logical_or(curr_mask, next_mask))
        if union == 0:
            ious[i] = 0.0
        else:
            intersection = np.count_nonzero(np.logical_and(curr_mask, next_mask))
            ious[i] = intersection / union
    return ious


def filter_overlaps(curr_slice, next_slice):
    # Get the bounding boxes for each region in the current and next slices
    rps = skimage.measure.regionprops(curr_slice)
    boxes1 = np.array([rp.bbox for rp in rps])
    labels1 = np.array([rp.label for rp in rps])

    rps = skimage.measure.regionprops(next_slice)
    boxes2 = np.array([rp.bbox for rp in rps])
    labels2 = np.array([rp.label for rp in rps])

    # Check for overlaps between the boxes in the two slices
    box_matches = []

    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            res = check_overlap(box1, box2)
            if res:
                box_matches.append((labels1[i], labels2[j]))
    return box_matches


def check_overlap(box1, box2):
    # Box: [min_row, min_col, max_row, max_col]
    # https://stackoverflow.com/a/40795835
    # We compare x & y coords of bottom-left & top-right corners
    # Bottom-left: min_col (x), max_row (y)
    # Top-right: max_col (x), min_row (y)
    # Note that higher y is lower in the image: (0,0) is top-left
    return not (
        box1[3] < box2[1] or box1[1] > box2[3] or box1[2] < box2[0] or box1[0] > box2[2]
    )


def connect_sam(all_masks, iou_threshold):
    for idx in tqdm(range(all_masks.shape[0] - 1)):
        # Create a matrix to store all combinations of IoUs
        curr_slice = all_masks[idx]
        next_slice = all_masks[idx + 1]

        # Get the unique labels in the current and next slices
        curr_labels = np.unique(curr_slice)
        next_labels = np.unique(next_slice)
        # Get a numba-compatible dictionary for the labels to allow for later indexing
        curr_label_dict = Dict.empty(key_type=types.uint16, value_type=types.uint16)
        next_label_dict = Dict.empty(key_type=types.uint16, value_type=types.uint16)
        curr_label_dict.update(
            {label: np.uint16(i) for i, label in enumerate(curr_labels)}
        )
        next_label_dict.update(
            {label: np.uint16(i) for i, label in enumerate(next_labels)}
        )

        # Restrict to only overlapping boxes
        box_matches = filter_overlaps(curr_slice, next_slice)

        # No matches, skip
        if len(box_matches) > 0:
            # Create boolean masks for each label in the current and next slices
            # Effectively converts (H, W) int array into (H, W, N) boolean where N is the number of labels
            curr_slice_bool = curr_slice[..., None] == curr_labels
            next_slice_bool = next_slice[..., None] == next_labels

            # Calculate IoUs for all pairs of overlapping boxes
            ious = mask_iou_batch(
                box_matches,
                curr_slice_bool,
                next_slice_bool,
                curr_label_dict,
                next_label_dict,
            )
            # Get the max label from the current slice to assign to to ensure no conflict
            max_label = curr_labels.max() + 1
            # Create an array mapping the next labels to the current labels
            mapping_arr = np.full(
                int(next_labels.max() + 1), fill_value=0, dtype=np.uint16
            )
            # Iterate over the matches and check which ones sufficiently overlap
            for iou, (curr_label, next_label) in zip(ious, box_matches, strict=True):
                # If threshold met, remap label
                if iou >= iou_threshold:
                    mapping_arr[next_label] = curr_label
            # Need to account for all other labels
            for i, val in enumerate(mapping_arr):
                # Fill in the labels that were not matched
                if val == 0:
                    # Skip background
                    if i == 0:
                        continue
                    # Set to the next available label
                    mapping_arr[i] = max_label
                    max_label += 1
            # Remap the labels in the next slice
            # Fancy mapping: https://stackoverflow.com/a/55950051
            all_masks[idx + 1] = mapping_arr[next_slice.copy()]
    # Relabel the masks to get consecutive labels from 1 to N
    (
        all_masks,
        _,
        _,
    ) = relabel_sequential(all_masks)
    return reduce_dtype(all_masks)


def mask_iou(masks1: np.ndarray, masks2: np.ndarray):
    intersection = np.sum(np.logical_and(masks1, masks2))
    union = np.sum(np.logical_or(masks1, masks2))
    if union == 0:
        return 0.0
    else:
        return intersection / union


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-fname", required=True, help="Mask save filename")
    parser.add_argument("--output-dir", required=True, help="Mask output directory")
    parser.add_argument(
        "--masks",
        required=True,
        nargs="+",
        help="Masks to combine",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model used to generate masks",
    )
    parser.add_argument(
        "--image-size",
        nargs=3,
        type=int,
        required=True,
        help="Size of the image stack, in array (i.e. D x H x W) format.",
    )
    parser.add_argument(
        "--overlap",
        required=True,
        nargs=3,
        help="Overlap in each dimension (default is 0). Assumed H x W x D.",
    )
    parser.add_argument(
        "--postprocess",
        required=False,
        action="store_true",
        help="Run postprocessing on the masks",
    )
    parser.add_argument(
        "--iou-threshold",
        required=False,
        type=float,
        default=0.8,
        help="IoU threshold for aligning masks (in SAM)",
    )
    parser.add_argument(
        "--output-format",
        required=False,
        default="rle",
        choices=["rle", "tiff"],
        help="Output format for the combined masks ('rle' or 'tiff')",
    )
    parser.add_argument(
        "--output-mask-type",
        required=False,
        default="instance",
        choices=["auto", "binary", "instance"],
        help="Mask type of the combined output ('binary' or 'instance')",
    )
    parser.add_argument(
        "--slab-size",
        required=False,
        type=int,
        default=SLAB_SIZE,
        help="Number of z-slices read/written at a time for the Zarr-backed combined-mask "
        "store; bounds how much of the volume is materialised in memory at once "
        f"(default: {SLAB_SIZE}).",
    )

    cli_args = parser.parse_args()
    if cli_args.slab_size < 1:
        raise ValueError(f"--slab-size must be >= 1, got {cli_args.slab_size}")
    slab_size = cli_args.slab_size

    _log_mem("before loading stack")
    # use /tmp in the node that the combine stacks process is running on if available for faster I/O
    scratch = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or ".")
    print(f"Scratch stores in: {scratch.name}")
    combine_store = Path(scratch.name) / "all_masks.zarr"
    labelled_store = Path(scratch.name) / "all_masks_labelled.zarr"

    # Combine the masks
    if len(cli_args.masks) > 1:
        combined_masks, mask_type_from_file = combine_masks(
            cli_args.masks,
            overlap=cli_args.overlap,
            image_size=cli_args.image_size,
            model=cli_args.model,
            store_path=combine_store,
            slab_size=slab_size,
        )
        _log_mem("after loading stack")
    else:
        combined_masks = aiod_rle.load_encoding(cli_args.masks[0])
        # NOTE: Extract metadata later from preprocess params
        combined_masks, decoded_metadata = aiod_rle.decode(combined_masks)
        # Extract mask_type from metadata to avoid expensive check_mask_type() later
        mask_type_from_file = decoded_metadata.get("metadata", {}).get("mask_type")
    print(f"Combined masks shape: {combined_masks.shape}")
    if cli_args.postprocess:
        print("Postprocessing masks...")
        if cli_args.model == "sam" or cli_args.model == "sam2":
            # No need to align over slices if there are none! Labels consecutive already
            if combined_masks.ndim > 2:
                # NOTE: connect_sam remains dense -- relabel_sequential needs the whole
                # volume in memory, and SAM volumes are small enough for that to be fine.
                combined_masks = connect_sam(
                    np.asarray(combined_masks), iou_threshold=cli_args.iou_threshold
                )
        else:
            # Q - shouldn't we also have a guard againt combined_masks.ndim > 2?
            combined_masks, _num_holes = connect_components(
                combined_masks, store_path=labelled_store
            )
    # combined_masks is a plain np.ndarray only in the single-substack path above
    # (aiod_rle.decode returns numpy directly, so combine_masks() is never called); the
    # multi-substack Zarr path reduces dtype and squeezes itself (slab_max/resolve_mask_type
    # stream it, and image_size is already squeezed to 2D in combine_masks when needed).
    elif isinstance(combined_masks, np.ndarray):
        combined_masks = reduce_dtype(combined_masks)
    # Squeeze the array in case there is only one slice (numpy path only, see above)
    if isinstance(combined_masks, np.ndarray):
        combined_masks = np.squeeze(combined_masks)
    _log_mem("in combination")
    # Save the masks
    output_format = cli_args.output_format.lower()
    save_path = f"{cli_args.mask_fname}_all.{output_format}"
    # Get downsample factor for metadata if used
    # NOTE: Our Napari plugin uses this as an identifier to rescale for visualization
    # FIXME: This is brittle and poor, the params should be extracted from the preprocess params
    # Which themselves should be tied to the image that they produced
    if "Downsample" in cli_args.mask_fname:
        downsample_factor = get_downsample_factor(filename=cli_args.mask_fname)
        metadata = {"downsample_factor": downsample_factor}
    else:
        metadata = {}
    try:
        if output_format == "tiff":
            # Resolve 'auto' using the mask type recorded in the individual patches
            resolved_mask_type = (
                mask_type_from_file
                if cli_args.output_mask_type == "auto"
                else cli_args.output_mask_type
            )
            # Convert to uint8 0/255 for clean display in downstream tools
            if resolved_mask_type == "binary":
                out_dtype = np.uint8

                def _to_display(block):
                    return (block > 0) * np.uint8(255)

            else:
                # Pick the narrowest dtype that fits (a TIFF needs one dtype up front,
                # hence the streaming max rather than a full-volume np.max)
                out_dtype = check_dtype(
                    combined_masks, max_val=slab_max(combined_masks, slab=slab_size)
                )

                def _to_display(block):
                    return block.astype(out_dtype, copy=False)

            # metadata is serialised as JSON into the TIFF ImageDescription tag
            # pages are written from a generator function so the volume is never resident
            tifffile.imwrite(
                save_path,
                iter_pages(combined_masks, transform=_to_display, slab=slab_size),
                shape=combined_masks.shape,
                dtype=out_dtype,
                metadata=metadata,
                imagej=True,
            )
        else:
            # Reuse mask_type from decoded patches; fall back to CLI value (skip if 'auto' and absent)
            resolved_mask_type = mask_type_from_file or (
                cli_args.output_mask_type
                if cli_args.output_mask_type != "auto"
                else None
            )
            resolved_mask_type = resolve_mask_type(
                combined_masks, resolved_mask_type, slab_size=slab_size
            )
            encoded_masks = encode_slicewise(
                combined_masks, mask_type=resolved_mask_type, metadata=metadata
            )
            aiod_rle.save_encoding(rle=encoded_masks, fpath=save_path)
        _log_mem("after saving")
        # Remove the (symlinked) individual masks now that they are combined.
        # missing_ok because Nextflow does not re-publish a cached task's outputs on
        # -resume: once this has run, the published masks are gone for good, and a
        # re-run of this task would otherwise die here after doing all the work.
        for mask_path in cli_args.masks:
            (Path(cli_args.output_dir) / mask_path).unlink(missing_ok=True)
    finally:
        # Remove the scratch stores; they are pure intermediates
        scratch.cleanup()
