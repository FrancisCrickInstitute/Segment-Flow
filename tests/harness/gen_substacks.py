#!/usr/bin/env python
"""Synthetic per-substack masks for the combine_stacks harness.

Substack geometry comes from aiod_utils.stacks.generate_stack_indices, as in
create_splits.py. Content is a deterministic function of global coordinates: a
blocky grid where each block's foreground/id comes from a hash of its global
block index. Neighbouring substacks therefore agree in their overlaps, and no
global dense volume is ever built.

Each substack is relabelled and dtype-reduced as utils.save_masks does, then
written from the same array in every requested format:
  rle                       .rle, as runModel writes them
  zarr-<dir|zip>-<zstd|lz4>-c<depth>   one Zarr array per substack

Outputs are cached under a directory keyed by the parameters, so repeated runs
and different targets reuse them. Use as a module (see bench_combine.py) or:

    python gen_substacks.py --shape 64 512 512 --grid 4 2 2 --formats rle zarr-dir-zstd-c1
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import aiod_utils.rle as aiod_rle
import numpy as np
from aiod_utils.io import get_mask_name, reduce_dtype
from aiod_utils.stacks import Stack, generate_stack_indices
from skimage.segmentation import relabel_sequential

HARNESS_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = Path(os.environ.get("COMBINE_BENCH_CACHE", HARNESS_DIR / "cache"))

# Names the bench uses for --mask-fname / --image-id / --param-hash
IMAGE_ID = "bench_tif"
RUN_HASH = "benchhash"
MASK_FNAME = get_mask_name(run_hash=RUN_HASH, image_id=IMAGE_ID)

# Bumped whenever the content or file layout changes, so stale caches are not reused
GEN_VERSION = 1


def substack_indices(shape, grid, overlap):
    """Substack (x0, x1, y0, y1, z0, z1) tuples, in create_splits.py's order."""
    D, H, W = shape
    nz, ny, nx = grid
    image_shape = Stack(height=H, width=W, depth=D)
    overlap_fraction = Stack(height=overlap, width=overlap, depth=overlap)
    # Effective shape as calc_num_stacks_dim computes it, without its size caps,
    # so the requested grid is used as given
    eff_shape = Stack(
        height=round(H * (1 + overlap)),
        width=round(W * (1 + overlap)),
        depth=round(D * (1 + overlap)),
    )
    stack_indices, _, _ = generate_stack_indices(
        image_shape=image_shape,
        num_substacks=Stack(height=ny, width=nx, depth=nz),
        overlap_fraction=overlap_fraction,
        eff_shape=eff_shape,
    )
    return [(w0, w1, h0, h1, d0, d1) for (h0, h1), (w0, w1), (d0, d1) in stack_indices]


def _hash(keys, seed):
    """splitmix64 over uint64 keys."""
    with np.errstate(over="ignore"):
        x = keys.astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def block_content(idxs, mask_type, block, block_z, fg_frac, n_ids, seed):
    """Dense content of one substack, a function of global coordinates only."""
    x0, x1, y0, y1, z0, z1 = idxs
    bz = np.arange(z0, z1) // block_z
    by = np.arange(y0, y1) // block
    bx = np.arange(x0, x1) // block
    # Hash only the distinct blocks touching this substack, then expand
    uz, iz = np.unique(bz, return_inverse=True)
    uy, iy = np.unique(by, return_inverse=True)
    ux, ix = np.unique(bx, return_inverse=True)
    gz, gy, gx = np.meshgrid(uz, uy, ux, indexing="ij")
    keys = (gz.astype(np.uint64) << np.uint64(42)) | (
        gy.astype(np.uint64) << np.uint64(21)
    ) | gx.astype(np.uint64)
    h = _hash(keys, seed)
    fg = (h % np.uint64(1_000_000)) < np.uint64(int(fg_frac * 1_000_000))
    if mask_type == "binary":
        coarse = fg.astype(np.uint8)
    else:
        ids = (h >> np.uint64(32)) % np.uint64(n_ids) + np.uint64(1)
        coarse = np.where(fg, ids, 0).astype(np.uint32)
    return coarse[np.ix_(iz, iy, ix)]


def to_saved_array(arr, mask_type):
    """What utils.save_masks hands to aiod_rle.encode."""
    arr, _, _ = relabel_sequential(arr)
    arr = reduce_dtype(arr)
    if mask_type == "binary":
        arr = arr.astype(bool)
    return arr


def parse_zarr_format(fmt):
    """'zarr-dir-zstd-c1' -> ('dir', 'zstd', 1)."""
    _, store, codec, depth = fmt.split("-")
    if store not in ("dir", "zip") or codec not in ("zstd", "lz4") or depth[0] != "c":
        raise ValueError(f"bad zarr format '{fmt}'")
    return store, codec, int(depth[1:])


def write_zarr(arr, path, store, codec, chunk_depth, labels):
    import zarr
    from zarr.codecs import BloscCodec, ZstdCodec

    compressor = ZstdCodec() if codec == "zstd" else BloscCodec(cname="lz4")
    chunks = (min(chunk_depth, arr.shape[0]), arr.shape[1], arr.shape[2])
    zstore = zarr.storage.ZipStore(path, mode="w") if store == "zip" else str(path)
    z = zarr.create_array(
        store=zstore,
        shape=arr.shape,
        dtype=arr.dtype,
        chunks=chunks,
        compressors=[compressor],
        attributes={"labels": [int(v) for v in labels]},
        overwrite=True,
    )
    z[:] = arr
    if store == "zip":
        zstore.close()


def path_size(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size, 1
    files = [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files), len(files)


def params_key(params):
    blob = json.dumps({**params, "gen_version": GEN_VERSION}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def generate(
    shape,
    grid,
    overlap=0.0,
    mask_type="instance",
    block=16,
    block_z=None,
    fg_frac=0.4,
    n_ids=1000,
    seed=0,
    formats=("rle",),
    cache_dir=DEFAULT_CACHE,
    quiet=False,
):
    """Generate (or reuse) the substacks; returns the manifest dict."""
    params = {
        "shape": [int(s) for s in shape],
        "grid": [int(g) for g in grid],
        "overlap": float(overlap),
        "mask_type": mask_type,
        "block": int(block),
        "block_z": int(block_z or block),
        "fg_frac": float(fg_frac),
        "n_ids": int(n_ids),
        "seed": int(seed),
    }
    out_dir = Path(cache_dir) / params_key(params)
    manifest_path = out_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"params": params, "dir": str(out_dir), "substacks": [], "formats": {}}
    )
    todo = [f for f in formats if f not in manifest["formats"]]
    if not todo:
        return manifest

    idx_list = substack_indices(params["shape"], params["grid"], params["overlap"])
    names = [
        f"{MASK_FNAME}_x{x0}-{x1}_y{y0}-{y1}_z{z0}-{z1}"
        for x0, x1, y0, y1, z0, z1 in idx_list
    ]
    stats = {f: {"write_s": [], "bytes": [], "files": []} for f in todo}
    for f in todo:
        (out_dir / f).mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    for idxs, name in zip(idx_list, names, strict=True):
        arr = block_content(
            idxs,
            mask_type,
            params["block"],
            params["block_z"],
            params["fg_frac"],
            params["n_ids"],
            params["seed"],
        )
        arr = to_saved_array(arr, mask_type)
        labels = np.arange(1, int(arr.max()) + 1) if arr.any() else np.array([], int)
        for f in todo:
            t0 = time.perf_counter()
            if f == "rle":
                path = out_dir / f / f"{name}.rle"
                enc = aiod_rle.encode(arr, mask_type=mask_type, metadata={})
                aiod_rle.save_encoding(enc, path)
            else:
                store, codec, depth = parse_zarr_format(f)
                ext = ".zarr.zip" if store == "zip" else ".zarr"
                path = out_dir / f / f"{name}{ext}"
                write_zarr(arr, path, store, codec, depth, labels)
            stats[f]["write_s"].append(time.perf_counter() - t0)
            nbytes, nfiles = path_size(path)
            stats[f]["bytes"].append(nbytes)
            stats[f]["files"].append(nfiles)
    manifest["substacks"] = [
        {"name": n, "idxs": list(i)} for n, i in zip(names, idx_list, strict=True)
    ]
    for f in todo:
        s = stats[f]
        manifest["formats"][f] = {
            "write_s_mean": float(np.mean(s["write_s"])),
            "write_s_max": float(np.max(s["write_s"])),
            "write_s_total": float(np.sum(s["write_s"])),
            "bytes_total": int(np.sum(s["bytes"])),
            "files_total": int(np.sum(s["files"])),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if not quiet:
        print(
            f"generated {len(names)} substacks ({', '.join(todo)}) in "
            f"{time.perf_counter() - t_all:.1f}s -> {out_dir}"
        )
    return manifest


def substack_paths(manifest, fmt):
    """Paths of the substacks of one format, in generation order."""
    out_dir = Path(manifest["dir"]) / fmt
    if fmt == "rle":
        ext = ".rle"
    else:
        ext = ".zarr.zip" if parse_zarr_format(fmt)[0] == "zip" else ".zarr"
    return [out_dir / f"{s['name']}{ext}" for s in manifest["substacks"]]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--shape", nargs=3, type=int, required=True, metavar=("D", "H", "W"))
    p.add_argument("--grid", nargs=3, type=int, required=True, metavar=("ND", "NH", "NW"))
    p.add_argument("--overlap", type=float, default=0.0, help="Overlap fraction")
    p.add_argument("--mask-type", choices=["binary", "instance"], default="instance")
    p.add_argument("--block", type=int, default=16, help="XY block size (smaller = more runs)")
    p.add_argument("--block-z", type=int, default=None, help="Z block size (default --block)")
    p.add_argument("--fg-frac", type=float, default=0.4, help="Foreground block fraction")
    p.add_argument("--n-ids", type=int, default=1000, help="Distinct ids before relabelling")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--formats", nargs="+", default=["rle"])
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p.add_argument("--clear", action="store_true", help="Delete this entry first")
    args = p.parse_args()
    kwargs = dict(
        shape=args.shape,
        grid=args.grid,
        overlap=args.overlap,
        mask_type=args.mask_type,
        block=args.block,
        block_z=args.block_z,
        fg_frac=args.fg_frac,
        n_ids=args.n_ids,
        seed=args.seed,
        formats=args.formats,
        cache_dir=args.cache_dir,
    )
    if args.clear:
        manifest = generate(**{**kwargs, "formats": ()}, quiet=True)
        shutil.rmtree(manifest["dir"], ignore_errors=True)
    manifest = generate(**kwargs)
    print(json.dumps(manifest["formats"], indent=2))


if __name__ == "__main__":
    main()
