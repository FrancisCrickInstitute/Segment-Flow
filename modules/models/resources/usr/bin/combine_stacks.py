import os
import shutil
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
import zarr

# Number of z-slices materialised at a time when reading/encoding the combined volume.
# aiod_rle.encode makes three transient full-size copies of whatever it is handed
# (astype(bool), the transpose+reshape copy, and the XOR output), so peak memory during
# encoding is ~4x one slab. Measured to hold down to ~64 slices; below that the
# accumulated output list dominates and there is nothing further to gain.
# See AIOD-266_fix_plan.md sections 3 and 6.
ENCODE_SLAB = 64
# Chunk shape for the Zarr-backed combined-mask store (~8 MB per chunk at uint16). The z
# extent matches ENCODE_SLAB so a slab read maps onto exactly one chunk layer. Keeping y/x
# well below the frame size matters: chunks spanning the full frame make every substack
# write a partial-chunk read-modify-write of a large chunk, which measured 5x worse.
ZARR_CHUNKS = (64, 256, 256)


def _open_store(path, shape, dtype, chunks=ZARR_CHUNKS):
    """Create a chunked, compressed Zarr array on local scratch to hold a mask volume.

    The default compressor is deliberately left alone: mask data is highly compressible
    (in testing a 480 MB volume wrote out as 0.3 MB), and the codec keyword differs
    between zarr 2.x and 3.x.
    """
    shape = tuple(int(s) for s in shape)
    # Match chunk rank to array rank, and never chunk larger than the array itself
    chunks = tuple(min(c, s) for c, s in zip(chunks[-len(shape) :], shape, strict=True))
    return zarr.open_array(
        store=str(path), mode="w", shape=shape, chunks=chunks, dtype=dtype
    )


def iter_slabs(arr, slab=ENCODE_SLAB):
    """Yield (z0, z1, block) over the leading axis, as in-memory numpy arrays.

    Works for numpy arrays and for Zarr arrays alike, so callers never need to know
    whether the volume is resident or on disk. 2D arrays are yielded whole.
    """
    if arr.ndim == 2:
        yield 0, 1, np.asarray(arr)
        return
    for z0 in range(0, arr.shape[0], slab):
        z1 = min(z0 + slab, arr.shape[0])
        yield z0, z1, np.asarray(arr[z0:z1])


def iter_pages(arr, transform=None, slab=ENCODE_SLAB):
    """Yield individual 2D pages, reading in slabs so a Zarr store isn't hit per-slice."""
    for _, _, block in iter_slabs(arr, slab):
        if transform is not None:
            block = transform(block)
        if block.ndim == 2:
            yield block
        else:
            yield from block


def max_over_z(arr, start_z, end_z, slab=ENCODE_SLAB):
    """Max over arr[start_z:end_z], read in slabs so the range is never materialised."""
    best = 0
    for z0 in range(start_z, end_z, slab):
        z1 = min(z0 + slab, end_z)
        block_max = int(np.asarray(arr[z0:z1]).max())
        best = max(best, block_max)
    return best


def write_blocks(store, block, origin, add=False, slab=ENCODE_SLAB):
    """Write `block` into `store` at `origin`, a z-slab at a time.

    Writing a whole substack in a single __setitem__ leaves many chunks in flight inside
    zarr at once; going a slab at a time bounds that (measured 3-5x lower peak). `add=True`
    accumulates rather than overwrites, for the overlap > 0 path.
    """
    if len(origin) == 2:
        y0, x0 = origin
        sl = (
            slice(y0, y0 + block.shape[0]),
            slice(x0, x0 + block.shape[1]),
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


def global_max(arr, slab=ENCODE_SLAB):
    """Max over the whole array, read in slabs so it is never fully materialised."""
    best = 0
    for _, _, block in iter_slabs(arr, slab):
        if block.size:
            best = max(best, int(block.max()))
    return best


def resolve_mask_type(masks, mask_type):
    """Resolve the mask type up front, without materialising the whole volume.

    Must be done once for the whole volume rather than per slab: aiod_rle.encode would
    otherwise infer independently for each slab via check_mask_type(), which could
    classify a sparse slab as "binary" and a busy one as "instance" and so produce a
    structurally invalid file.

    Deliberately avoids aiod_rle.check_mask_type, which calls np.unique on the full
    array; that sorts a copy, i.e. one more full-volume allocation. Scanning slabs and
    exiting as soon as a third distinct value appears costs one slab instead.
    """
    if mask_type is not None:
        return mask_type
    if masks.dtype == bool:
        return "binary"
    seen = set()
    for _, _, block in iter_slabs(masks):
        seen.update(np.unique(block).tolist())
        if len(seen) > 2:
            return "instance"
    return "binary"


def encode_streaming(masks, mask_type, metadata, slab=ENCODE_SLAB):
    """Slab-wise equivalent of aiod_rle.encode(masks, mask_type, metadata).

    aiod_rle.encode returns one entry per z-slice plus a trailing metadata dict, and each
    slice's entry depends only on that slice's own pixels (see _encode_binary: row i of
    `diff` derives solely from mask[i]). Encoding in z-slabs and concatenating the lists
    in z order therefore reproduces the identical list, element for element, while
    holding only `slab` slices plus encode's three transient copies of them in RAM.

    `masks` may be a numpy array, a Zarr array, or anything supporting [z0:z1] slicing
    that returns a numpy array.
    """
    if mask_type is None:
        raise ValueError("mask_type must be resolved before streaming encode")
    if masks.ndim == 2:
        return aiod_rle.encode(
            np.asarray(masks), mask_type=mask_type, metadata=metadata
        )
    out = []
    for _, _, block in iter_slabs(masks, slab):
        # Each call appends its own trailing metadata dict, which is dropped here and
        # re-added exactly once at the end.
        out.extend(
            aiod_rle.encode(block, mask_type=mask_type, metadata=dict(metadata))[:-1]
        )
    out.append({"metadata": {**metadata, "mask_type": mask_type}})
    return out


def combine_masks(
    masks: list[str],
    overlap: list[float, ...],
    image_size: tuple[int, ...],
    model: str,
    store_path: str | Path,
):
    """
    Combine masks from each of the substacks into a single array/dataset.

    If overlap is 0, then the masks are simply inserted into their relevant indices.

    If overlap is >0, then the masks need to be combined.

    The combined volume is written into a chunked, compressed Zarr store at
    `store_path` rather than a dense in-RAM array, so peak memory is bounded by one
    decoded substack instead of the whole volume.

    Returns:
        tuple[zarr.Array, str | None]: Combined mask store and mask type ("binary",
        "instance", or None).
    """
    # Get the chunk size from the first file
    start_x, end_x, start_y, end_y, start_z, end_z = extract_idxs_from_fname(masks[0])
    _chunk_size = (end_x - start_x, end_y - start_y, end_z - start_z)
    # Check if there is XY tiling (at least one must be true for any given substack)
    xy_tiling = (
        start_x > 0 or end_x < image_size[1] or start_y > 0 or end_y < image_size[2]
    )
    # Create the array to hold the masks
    # NOTE: Using uint16 to be safe, but ideally should be taken from inputs (but slight chicken & egg)
    # Ensure image size appropriate to given dims
    if image_size[0] == 1:
        image_size = image_size[1:]
        is_2d = True
    else:
        is_2d = False
    # NOTE: image_size will now always take account of downsampling
    # Create the store to hold the masks (uint16 should be fine...?)
    # NOTE: Storage-backed rather than a dense np.zeros: the full volume never has to be
    # resident. uint16 costs almost nothing on disk once compressed, and per-slab dtype
    # reduction happens inside aiod_rle.encode, so there is no eager full-volume downcast.
    all_masks = _open_store(store_path, image_size, np.uint16)
    # Loop over each mask and insert into the array
    # Add the masks together if overlap is >0
    # NOTE: Adding together only really makes sense for binary masks
    overlap = [float(val) for val in overlap]
    mask_types_seen = set()

    if sum(overlap) == 0.0:
        for mask_path in masks:
            idxs = extract_idxs_from_fname(mask_path)
            encoding = aiod_rle.load_encoding(mask_path)
            mask, metadata = aiod_rle.decode(encoding)
            current_mask_type = metadata.get("metadata", {}).get("mask_type")
            if current_mask_type is not None:
                mask_types_seen.add(current_mask_type)
            # Cast boolean to allow addition
            if mask.dtype == bool:
                mask = mask.astype(np.uint8)
            all_masks = insert_mask(
                all_masks=all_masks,
                mask=mask,
                idxs=idxs,
                xy_tiling=xy_tiling,
                is_overlap=False,
                is_2d=is_2d,
            )
    # TODO: Extract this, and handle binary/labelled masks properly, with specified vote mechanism
    else:
        # Combine the masks
        for mask_path in masks:
            start_x, end_x, start_y, end_y, start_z, end_z = extract_idxs_from_fname(
                mask_path
            )
            encoding = aiod_rle.load_encoding(mask_path)
            mask, metadata = aiod_rle.decode(encoding)
            current_mask_type = metadata.get("metadata", {}).get("mask_type")
            if current_mask_type is not None:
                mask_types_seen.add(current_mask_type)
            # Cast boolean to allow addition
            if mask.dtype == bool:
                mask = mask.astype(np.uint8)
            # Just sum, naive method
            # NOTE: Read-modify-write a slab at a time; only the substack is resident
            origin = (start_y, start_x) if is_2d else (start_z, start_y, start_x)
            write_blocks(all_masks, mask, origin, add=True)

    # Validate mask type consistency across all mask files
    if len(mask_types_seen) > 1:
        raise ValueError(
            f"Inconsistent mask types found across mask files: {mask_types_seen}. "
            "All mask files must have the same mask type."
        )
    mask_type = mask_types_seen.pop() if mask_types_seen else None

    # NOTE: No eager reduce_dtype here -- that would rewrite the whole store to save
    # nothing (it is already compressed). aiod_rle.encode reduces each slab as it goes.
    return all_masks, mask_type


def insert_mask(
    all_masks,
    mask,
    idxs: tuple[int, int, int, int, int, int],
    xy_tiling: bool,
    is_overlap: bool,
    is_2d: bool,
):
    # Extract the indices
    start_x, end_x, start_y, end_y, start_z, end_z = idxs
    # Ensure labels are unique across a slice
    if xy_tiling:
        # Get the current maximum value across the relevant slices
        # NOTE: Read in slabs so a Zarr-backed store is never fully materialised
        max_val = (
            int(np.asarray(all_masks).max())
            if is_2d
            else max_over_z(all_masks, start_z, end_z)
        )
        # TODO: Handle the below, why is it commented out?
        # # Check if we need to upcast the array
        # if max_val + mask.max() > np.iinfo(all_masks.dtype).max:
        #     all_masks = all_masks.astype(np.uint32, copy=False)
        # Add a constant to all non-zero values to ensure uniqueness
        # NOTE: Equivalent to `mask[mask > 0] += max_val` but writes in place via `where`,
        # avoiding the fancy-index gather/scatter pair (one fewer substack-sized temporary)
        np.add(mask, max_val, out=mask, where=mask > 0)
    # Insert the mask into the array, a slab at a time
    origin = (start_y, start_x) if is_2d else (start_z, start_y, start_x)
    write_blocks(all_masks, mask, origin)
    return all_masks


def connect_components(all_masks, store_path: str | Path):
    """Label connected components across slices, keeping the volume out of RAM.

    The labelled result is written straight back to a Zarr store instead of being
    materialised with .compute(). The write and the component count are evaluated in a
    single dask.compute call so the label graph is only traversed once.
    """
    # Convert to dask array, preserving the store's chunking where there is one
    if isinstance(all_masks, np.ndarray):
        arr = da.from_array(all_masks)
    else:
        arr = da.from_zarr(all_masks)
    # Get the connected components, combining masks from consecutive frames
    labelled, num_holes = dask_image.ndmeasure.label(arr)
    # NOTE: dtype is left as dask_image produced it; aiod_rle.encode reduces per slab, so
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

    cli_args = parser.parse_args()

    # Scratch Zarr stores for the combined (and optionally relabelled) volume. These live
    # in the task's working directory and are removed in the finally block below.
    combine_store = Path("all_masks.zarr")
    labelled_store = Path("all_masks_labelled.zarr")

    mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
    print(f"Memory used before loading stack: {mem_used:.2f} GB")
    # Combine the masks
    if len(cli_args.masks) > 1:
        combined_masks, mask_type_from_file = combine_masks(
            cli_args.masks,
            overlap=cli_args.overlap,
            image_size=cli_args.image_size,
            model=cli_args.model,
            store_path=combine_store,
        )
        mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
        print(f"Memory used after loading stack: {mem_used:.2f} GB")
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
            combined_masks, _num_holes = connect_components(
                combined_masks, store_path=labelled_store
            )
    # Ensure the dtype is always reduced if possible
    # NOTE: Postprocessing funcs above handle this themselves
    elif isinstance(combined_masks, np.ndarray):
        # NOTE: Only meaningful for the dense single-mask path; a Zarr-backed volume is
        # reduced per slab inside aiod_rle.encode instead.
        combined_masks = reduce_dtype(combined_masks)
    # Squeeze the array in case there is only one slice
    # NOTE: Zarr-backed stores are already created at the right rank (combine_masks drops
    # the leading axis for 2D input), and np.squeeze would materialise the whole volume.
    if isinstance(combined_masks, np.ndarray):
        combined_masks = np.squeeze(combined_masks)
    mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
    print(f"Memory used in combination: {mem_used:.2f} GB")
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
    if output_format == "tiff":
        # Resolve 'auto' using the mask type recorded in the individual patches
        resolved_mask_type = (
            mask_type_from_file
            if cli_args.output_mask_type == "auto"
            else cli_args.output_mask_type
        )
        # Convert binary masks to uint8 0/255 for clean display
        if resolved_mask_type == "binary":
            # Convert to binary uint8 with 0/255 values for clean display in downstream tools
            combined_masks = (combined_masks > 0) * np.uint8(255)
        # metadata dict is serialised as JSON into the TIFF ImageDescription tag
        tifffile.imwrite(
            save_path, combined_masks, metadata=metadata, imagej=True
        )  # can we tifffile imwrite from a zarr store?
    else:
        # Reuse mask_type from decoded patches; fall back to CLI value (skip if 'auto' and absent)
        resolved_mask_type = mask_type_from_file or (
            cli_args.output_mask_type if cli_args.output_mask_type != "auto" else None
        )
        # here we want to iterate and pass in each chunk (z slice)

        zarr_combined_masks = zarr.open(
            "combined_stack.zarr",
            mode="w",
            shape=combined_masks.shape,
            chunks=(
                combined_masks.shape[0],
                1,
                1,
            ),  # can we consistently get Z? What if 2D?e
            dtype=combined_masks.dtype,
        )
        zarr_combined_masks = combined_masks

        rle_combined_masks = []
        for i in range(combined_masks.shape[0]):
            slice = zarr_combined_masks[i]
            rle_slice = aiod_rle.encode(slice)
            rle_combined_masks += rle_slice

        # metadata = rle_combined_masks[0][2]
        metadata = {"metadata": {"mask_type": "binary"}}

        rle_combined_masks = [
            item for item in rle_combined_masks if "counts" in item
        ]  # will counts always be there?

        rle_combined_masks.append(metadata)  # is this reliable?
        rle_combined_masks.insert(0, {"size": combined_masks.shape})

        # --- OLD METHOD ---
        # encoded_masks = aiod_rle.encode(
        #     combined_masks,
        #     mask_type=resolved_mask_type,
        #     metadata=metadata,
        # )

        # Free up memory (though too late at this point)
        # aiod_rle.save_encoding(rle=encoded_masks, fpath=save_path)

        aiod_rle.save_encoding(rle=rle_combined_masks, fpath=save_path)

    del combined_masks
    # Remove the (symlinked) individual masks now that they are combined
    for mask_path in cli_args.masks:
        (Path(cli_args.output_dir) / mask_path).unlink()
