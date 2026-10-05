#!/usr/bin/env python
"""combine_stacks.py with Zarr substacks in place of .rle ones (harness only).

Same CLI as combine_stacks.py, but --masks are .zarr / .zarr.zip substacks as
written by gen_substacks.py. The combine (iter_combined_slices) and the output
sinks are imported from combine_stacks.py unchanged; only the tile reader
differs, so comparing this against the worktree target isolates the cost of
reading RLE vs Zarr inputs. No --postprocess.

combine_stacks.py is taken from $COMBINE_STACKS_SCRIPT, else the worktree copy.
"""

import argparse
import importlib.util
import os
from functools import partial
from pathlib import Path

import numpy as np
import zarr
from aiod_utils.io import extract_idxs_from_fname, get_combined_mask_name

HARNESS_DIR = Path(__file__).resolve().parent
SCRIPT = Path(
    os.environ.get(
        "COMBINE_STACKS_SCRIPT",
        HARNESS_DIR.parents[1] / "modules/models/resources/usr/bin/combine_stacks.py",
    )
)
_spec = importlib.util.spec_from_file_location("combine_stacks", SCRIPT)
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


class ZarrTile:
    """One Zarr substack; tile[i] reads slice i, one chunk at a time."""

    def __init__(self, path):
        path = str(path)
        store = zarr.storage.ZipStore(path, mode="r") if path.endswith(".zip") else path
        self.arr = zarr.open_array(store, mode="r")
        self.labels = np.asarray(self.arr.attrs["labels"], dtype=np.int64)
        binary = self.arr.dtype == bool
        # Same decoded dtype and mask type as RLETile
        self.dtype = np.dtype(bool) if binary else np.dtype(np.uint16)
        self.meta_mask_type = "binary" if binary else "instance"
        self.chunk_depth = self.arr.chunks[0]
        self._block, self._block_start = None, None

    def __getitem__(self, i):
        start = (i // self.chunk_depth) * self.chunk_depth
        if self._block_start != start:
            self._block = self.arr[start : start + self.chunk_depth]
            self._block_start = start
        out = self._block[i - start]
        return out if out.dtype == bool else out.astype(np.uint16)


def zarr_idxs(path):
    name = Path(path).name.removesuffix(".zip").removesuffix(".zarr")
    return extract_idxs_from_fname(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mask-fname", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--masks", required=True, nargs="+")
    p.add_argument("--model", required=True)
    p.add_argument("--image-size", nargs=3, type=int, required=True)
    p.add_argument("--overlap", required=True, nargs=3)
    p.add_argument("--postprocess", action="store_true")
    p.add_argument("--iou-threshold", type=float, default=0.8)
    p.add_argument("--output-format", default="rle", choices=["rle", "tiff"])
    p.add_argument("--output-mask-type", default="instance",
                   choices=["auto", "binary", "instance"])
    p.add_argument("--preprocess-config")
    p.add_argument("--prep-hash", default="")
    p.add_argument("--image-id")
    p.add_argument("--param-hash")
    args = p.parse_args()
    if args.postprocess:
        raise SystemExit("zarr_variant does not support --postprocess")

    image_size = tuple(args.image_size)
    is_overlap = sum(float(v) for v in args.overlap) > 0
    tiles = [(zarr_idxs(m), partial(ZarrTile, m)) for m in args.masks]
    mask_type_from_file = ZarrTile(args.masks[0]).meta_mask_type

    def make_planes():
        return cs.iter_combined_slices(tiles, image_size, overlap=is_overlap)

    out_shape = image_size[1:] if image_size[0] == 1 else image_size
    save_path = get_combined_mask_name(args.mask_fname, args.output_format)
    if args.output_format == "tiff":
        resolved = (
            mask_type_from_file
            if args.output_mask_type == "auto"
            else args.output_mask_type
        )
        cs.write_tiff(make_planes, out_shape, save_path, binary=resolved == "binary",
                      metadata={})
    else:
        cs.write_rle(make_planes(), save_path, mask_type_from_file, {})
    for m in args.masks:
        (Path(args.output_dir) / m).unlink()


if __name__ == "__main__":
    main()
