import os
from collections import defaultdict
from functools import partial
from pathlib import Path

import aiod_utils.rle as aiod_rle
import dask.array as da
import dask_image.ndmeasure
import numpy as np
import psutil
import skimage.measure
import tifffile
from aiod_utils.io import (
    check_dtype,
    extract_idxs_from_fname,
    get_combined_mask_name,
    get_mask_name,
    reduce_dtype,
)
from aiod_utils.preprocess import (
    get_downsample_factor,
    get_prep_hash,
    load_methods,
)
from numba import jit, prange
from numba.core import types
from numba.typed import Dict
from skimage.segmentation import relabel_sequential
from tqdm import tqdm

def get_preprocess_methods(config_path: str, prep_hash: str) -> list[dict]:
    """
    Find the specific preprocessing methods set matching prep_hash within the
    run's full preprocessing config (params.preprocess). Matches by
    recomputing the same hash used to create prep_hash in the first place
    (see preprocess_image.py), rather than threading the set itself through
    the pipeline's CSV, which can't safely carry raw JSON.
    """
    if not prep_hash:
        return []
    candidates = load_methods(config_path, filter_noop=True)
    for methods in candidates:
        if get_prep_hash(methods) == prep_hash:
            return methods
    raise ValueError(
        f"No preprocessing set in {config_path} matches prep_hash '{prep_hash}'"
    )


def connect_components(all_masks: np.ndarray):
    # Convert to dask array
    all_masks = da.from_array(all_masks)
    # Get the connected components, combining masks from consecutive frames
    labelled, num_holes = dask_image.ndmeasure.label(all_masks)
    labelled = labelled.compute()
    num_holes = int(num_holes)
    # Get the appropriate dtype from the number of holes, and convert to numpy array
    return reduce_dtype(labelled, max_val=num_holes)


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


class RLETile:
    """One substack's RLE, decoded a single slice at a time.

    tile[i] decodes slice i (each slice is read once, then dropped from the
    RLE). tile.labels and tile.dtype describe the decoded values, taken from
    the RLE without decoding it.
    """

    def __init__(self, rle: list):
        self.rle = rle
        self.metadata = rle[-1]
        # mask_type as recorded in the file, None if absent
        self.meta_mask_type = self.metadata.get("metadata", {}).get("mask_type")
        self.mask_type = self.meta_mask_type or aiod_rle.check_rle_type(rle)

    @classmethod
    def load(cls, path: str | Path) -> "RLETile":
        return cls(aiod_rle.load_encoding(path))

    @property
    def dtype(self):
        return np.dtype(bool) if self.mask_type == "binary" else np.dtype(np.uint16)

    @property
    def labels(self) -> np.ndarray:
        """Non-zero values present in the decoded substack."""
        if self.mask_type == "binary":
            # Odd-indexed counts are foreground runs
            has_fg = any(sum(entry["counts"][1::2]) for entry in self.rle[:-1])
            return np.array([1] if has_fg else [], dtype=np.int64)
        idxs = np.fromiter(
            (entry["idx"] for rle_slice in self.rle[:-1] for entry in rle_slice),
            dtype=np.int64,
        )
        return np.unique(idxs[idxs > 0])

    def __getitem__(self, i: int) -> np.ndarray:
        mask, _ = aiod_rle.decode(
            [self.rle[i], self.metadata], mask_type=self.mask_type
        )
        self.rle[i] = None
        return mask


def label_offsets(tiles: list) -> list[int]:
    """Offset for each tile's non-zero labels, for tiles sharing one z-range.

    Each tile's offset is the largest value held by the tiles before it (in
    the given order) once their own offsets are added. The addition wraps in
    the dtype the tile is combined in (uint8 for binary, uint16 for instance).
    """
    offsets, top = [], 0
    for tile in tiles:
        offsets.append(top)
        labels = tile.labels
        if labels.size:
            work_dtype = np.uint8 if tile.dtype == bool else tile.dtype
            modulus = np.iinfo(work_dtype).max + 1
            top = max(top, int(((labels + top) % modulus).max()))
    return offsets


def uses_label_offsets(tiles: list, image_size: tuple[int, int, int], overlap: bool):
    """Whether labels are offset: XY tiling, judged from the first tile, and no overlap."""
    _, H, W = image_size
    start_x, end_x, start_y, end_y, _, _ = tiles[0][0]
    # TODO: end_x is compared against H and end_y against W
    xy_tiling = start_x > 0 or end_x < H or start_y > 0 or end_y < W
    return xy_tiling and not overlap


def combined_max(tiles: list, image_size: tuple[int, int, int]) -> int:
    """Largest value iter_combined_slices yields with no overlap, without decoding.

    Each tile's values are its labels plus its offset (see label_offsets).
    """
    offset_labels = uses_label_offsets(tiles, image_size, overlap=False)
    starts = defaultdict(list)
    for idxs, open_fn in tiles:
        if idxs[4] < image_size[0]:
            starts[idxs[4]].append(open_fn)
    top = 0
    for open_fns in starts.values():
        opened = [open_fn() for open_fn in open_fns]
        offsets = label_offsets(opened) if offset_labels else [0] * len(opened)
        for tile, offset in zip(opened, offsets, strict=True):
            labels = tile.labels
            if labels.size:
                work_dtype = np.uint8 if tile.dtype == bool else tile.dtype
                modulus = np.iinfo(work_dtype).max + 1
                top = max(top, int(((labels + offset) % modulus).max()))
    return top


def iter_combined_slices(tiles: list, image_size: tuple[int, int, int], overlap: bool):
    """Yield the combined (H, W) uint16 plane for each z in 0..D-1.

    tiles: one (idxs, open_fn) per substack, in CLI order. idxs is
    (start_x, end_x, start_y, end_y, start_z, end_z); open_fn() returns a tile
    where tile[i] is the substack's 2D slice i (see RLETile). Each tile is
    opened when z reaches its start_z and released after its end_z.

    With overlap, covering tiles are summed. Without, each tile is written
    into its region, with labels offset (see label_offsets) if there is XY
    tiling.
    """
    D, H, W = image_size
    offset_labels = uses_label_offsets(tiles, image_size, overlap)
    starts = defaultdict(list)
    for k, (idxs, _) in enumerate(tiles):
        starts[idxs[4]].append(k)
    # k -> (tile, label offset)
    open_tiles = {}
    mask_types_seen = set()
    for z in range(D):
        for k in [k for k in open_tiles if tiles[k][0][5] <= z]:
            del open_tiles[k]
        if starts.get(z):
            opened = [tiles[k][1]() for k in starts[z]]
            for tile in opened:
                mask_type = getattr(tile, "meta_mask_type", None)
                if mask_type is not None:
                    mask_types_seen.add(mask_type)
            if len(mask_types_seen) > 1:
                raise ValueError(
                    f"Inconsistent mask types found across mask files: {mask_types_seen}. "
                    "All mask files must have the same mask type."
                )
            # NOTE: tiles starting together are assumed to share their z-range
            offsets = label_offsets(opened) if offset_labels else [0] * len(opened)
            open_tiles.update(zip(starts[z], zip(opened, offsets)))
        plane = np.zeros((H, W), dtype=np.uint16)
        for k in sorted(open_tiles):
            tile, offset = open_tiles[k]
            x0, x1, y0, y1, z0, _ = tiles[k][0]
            t = np.asarray(tile[z - z0]).reshape(y1 - y0, x1 - x0)
            # Cast boolean to allow addition
            if t.dtype == bool:
                t = t.astype(np.uint8)
            if overlap:
                # Just sum, naive method
                plane[y0:y1, x0:x1] += t
            else:
                if offset:
                    t[t > 0] += t.dtype.type(offset % (np.iinfo(t.dtype).max + 1))
                plane[y0:y1, x0:x1] = t
        yield plane


def infer_mask_type(planes) -> str:
    """aiod_rle.check_mask_type over all planes: binary if <= 2 unique values."""
    values = set()
    for plane in planes:
        values.update(np.unique(plane).tolist())
        if len(values) > 2:
            return "instance"
    return "binary"


def write_rle(planes, save_path: str, mask_type: str, metadata: dict):
    """Encode each (H, W) plane and save them as one RLE file."""
    rle = []
    for plane in planes:
        rle.extend(aiod_rle.encode(plane, mask_type=mask_type, metadata={})[:-1])
    rle.append({"metadata": {**metadata, "mask_type": mask_type}})
    aiod_rle.save_encoding(rle=rle, fpath=save_path)


def write_tiff(
    make_planes,
    shape: tuple[int, ...],
    save_path: str,
    binary: bool,
    metadata: dict,
    dtype=None,
    max_val: int | None = None,
):
    """Write the planes from make_planes() as TIFF pages.

    Binary masks are written as uint8 0/255. Otherwise the pages use dtype,
    or the smallest dtype fitting max_val, the planes' maximum. Without
    either, the maximum takes an extra pass over make_planes().
    """
    if binary:
        dtype = np.uint8

        def to_page(plane):
            return (plane > 0) * np.uint8(255)

    else:
        if dtype is None:
            if max_val is None:
                max_val = 0
                for plane in make_planes():
                    max_val = max(max_val, int(plane.max()))
            dtype = check_dtype(None, max_val=max_val)

        def to_page(plane):
            return plane.astype(dtype, copy=False)

    # metadata dict is serialised as JSON into the TIFF ImageDescription tag
    tifffile.imwrite(
        save_path,
        map(to_page, make_planes()),
        shape=shape,
        dtype=dtype,
        metadata=metadata,
        imagej=True,
    )


def collect_dense(planes, image_size: tuple[int, int, int], single: bool, mask_type: str):
    """Stack the planes into one array: (D, H, W), or (H, W) if D == 1.

    Multiple substacks give the smallest dtype that fits. A single substack
    keeps its decoded dtype (bool for binary, uint16 for instance).
    """
    D, H, W = image_size
    dense = np.empty((D, H, W), dtype=np.uint16)
    for z, plane in enumerate(planes):
        dense[z] = plane
    if D == 1:
        dense = dense[0]
    if not single:
        return reduce_dtype(dense)
    return dense.astype(bool) if mask_type == "binary" else dense


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
        "--preprocess-config",
        required=True,
        help=(
            "Path to a JSON file of the run's full params.preprocess config "
            "(same file preprocessImage receives); used with --prep-hash to "
            "recover e.g. the downsample factor for output metadata."
        ),
    )
    parser.add_argument(
        "--prep-hash",
        required=False,
        default="",
        help="Hash identifying this image's preprocessing branch, empty if none",
    )
    parser.add_argument(
        "--image-id",
        required=True,
        help="Image identity this mask belongs to, used to verify --mask-fname",
    )
    parser.add_argument(
        "--param-hash",
        required=True,
        help="Run hash the pipeline resolved, used to verify --mask-fname",
    )

    cli_args = parser.parse_args()

    # getMaskName in main.nf has to build the mask filename independently -
    # Nextflow needs output patterns before the script runs, which Groovy cannot
    # get from Python. Check the two agree rather than trusting they do: a silent
    # divergence writes masks that aiod_napari's watcher never matches.
    expected_fname = get_mask_name(
        run_hash=cli_args.param_hash,
        image_id=cli_args.image_id,
        prep_hash=cli_args.prep_hash or None,
    )
    if cli_args.mask_fname != expected_fname:
        raise ValueError(
            f"Mask filename from the pipeline ('{cli_args.mask_fname}') does not "
            f"match the one aiod_utils builds ('{expected_fname}'). "
            "getMaskName in main.nf has drifted from aiod_utils.io.get_mask_name - "
            "update both to the same format."
        )

    mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
    print(f"Memory used before loading stack: {mem_used:.2f} GB")
    image_size = tuple(cli_args.image_size)
    is_overlap = sum(float(val) for val in cli_args.overlap) > 0
    tiles = [
        (extract_idxs_from_fname(mask_path), partial(RLETile.load, mask_path))
        for mask_path in cli_args.masks
    ]
    # Mask type from the first file: as recorded, and as decoded
    first_tile = RLETile.load(cli_args.masks[0])
    mask_type_from_file = first_tile.meta_mask_type
    decoded_mask_type = first_tile.mask_type
    del first_tile

    def make_planes():
        return iter_combined_slices(tiles, image_size, overlap=is_overlap)

    out_shape = image_size[1:] if image_size[0] == 1 else image_size
    print(f"Combined masks shape: {out_shape}")
    combined_masks = None
    if cli_args.postprocess:
        combined_masks = collect_dense(
            make_planes(),
            image_size,
            single=len(cli_args.masks) == 1,
            mask_type=decoded_mask_type,
        )
        print("Postprocessing masks...")
        if cli_args.model == "sam" or cli_args.model == "sam2":
            # No need to align over slices if there are none! Labels consecutive already
            if combined_masks.ndim > 2:
                combined_masks = connect_sam(
                    combined_masks, iou_threshold=cli_args.iou_threshold
                )
        else:
            combined_masks = connect_components(combined_masks)
        # Squeeze the array in case there is only one slice
        combined_masks = np.squeeze(combined_masks)
        out_shape = combined_masks.shape

        def make_planes():
            return iter(combined_masks if combined_masks.ndim > 2 else [combined_masks])

    # Save the masks
    output_format = cli_args.output_format.lower()
    save_path = get_combined_mask_name(cli_args.mask_fname, output_format)
    # Get downsample factor for metadata if used.
    # NOTE: Our Napari plugin uses this as an identifier to rescale for visualization
    # Recover this branch's preprocessing set from the run's full config by
    # matching prep_hash
    preprocess_methods = get_preprocess_methods(
        cli_args.preprocess_config, cli_args.prep_hash
    )
    downsample_factor = (
        get_downsample_factor(methods=preprocess_methods)
        if preprocess_methods
        else None
    )
    metadata = (
        {"downsample_factor": downsample_factor}
        if downsample_factor is not None
        else {}
    )
    if output_format == "tiff":
        # Resolve 'auto' using the mask type recorded in the individual patches
        resolved_mask_type = (
            mask_type_from_file
            if cli_args.output_mask_type == "auto"
            else cli_args.output_mask_type
        )
        write_tiff(
            make_planes,
            out_shape,
            save_path,
            # Binary masks are written as uint8 0/255 for clean display
            binary=resolved_mask_type == "binary",
            metadata=metadata,
            dtype=None if combined_masks is None else combined_masks.dtype,
            # Known without decoding when nothing overlaps
            max_val=(
                combined_max(tiles, image_size)
                if combined_masks is None
                and not is_overlap
                and resolved_mask_type != "binary"
                else None
            ),
        )
    else:
        # Reuse mask_type from decoded patches; fall back to CLI value (infer if 'auto' and absent)
        resolved_mask_type = mask_type_from_file or (
            cli_args.output_mask_type if cli_args.output_mask_type != "auto" else None
        )
        if resolved_mask_type is None:
            resolved_mask_type = (
                infer_mask_type(make_planes())
                if combined_masks is None
                else aiod_rle.check_mask_type(combined_masks)
            )
        write_rle(make_planes(), save_path, resolved_mask_type, metadata)
    del combined_masks
    # Remove the (symlinked) individual masks now that they are combined
    for mask_path in cli_args.masks:
        (Path(cli_args.output_dir) / mask_path).unlink()
