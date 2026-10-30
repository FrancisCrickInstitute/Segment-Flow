"""Unit tests for the slice-wise combine in combine_stacks.py.

Run inside the combine_stacks conda env (CI does not run Python tests):
    pytest tests/test_combine_stacks.py
"""

import importlib.util
import random
from pathlib import Path

import aiod_utils.rle as aiod_rle
import numpy as np
import pytest
from aiod_utils.io import get_mask_name, reduce_dtype
from aiod_utils.stacks import Stack, generate_stack_indices
from skimage.segmentation import relabel_sequential

TESTS_DIR = Path(__file__).parent
SCRIPT = TESTS_DIR.parent / "modules/models/resources/usr/bin/combine_stacks.py"

IMAGE_ID = "test_tif"
RUN_HASH = "testhash"

_spec = importlib.util.spec_from_file_location("combine_stacks", SCRIPT)
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


# ---------------------------------------------------------------- inputs


def tile_grid(shape, grid, overlap):
    """Substack idxs (x0, x1, y0, y1, z0, z1), as create_splits.py lays them out."""
    D, H, W = shape
    nz, ny, nx = grid
    eff = Stack(
        height=round(H * (1 + overlap)),
        width=round(W * (1 + overlap)),
        depth=round(D * (1 + overlap)),
    )
    indices, _, _ = generate_stack_indices(
        image_shape=Stack(height=H, width=W, depth=D),
        num_substacks=Stack(height=ny, width=nx, depth=nz),
        overlap_fraction=Stack(height=overlap, width=overlap, depth=overlap),
        eff_shape=eff,
    )
    return [(w0, w1, h0, h1, d0, d1) for (h0, h1), (w0, w1), (d0, d1) in indices]


def tile_content(shape, mask_type, rng, n_labels=12, empty=False):
    """Blocky random content for one substack, with some empty slices."""
    if empty:
        return np.zeros(shape, dtype=np.uint16)
    coarse = tuple(max(1, s // 3) for s in shape)
    if mask_type == "binary":
        arr = (rng.random(coarse) < 0.4).astype(np.uint16)
    else:
        arr = rng.integers(0, n_labels + 1, size=coarse).astype(np.uint16)
    for ax, s in enumerate(shape):
        arr = np.repeat(arr, -(-s // arr.shape[ax]), axis=ax)
    arr = arr[: shape[0], : shape[1], : shape[2]].copy()
    if shape[0] > 2:
        arr[1] = 0
    return arr


def save_substack(path, arr, mask_type, keep_mask_type=True):
    """Encode one substack as utils.save_masks does."""
    arr, _, _ = relabel_sequential(arr)
    arr = reduce_dtype(arr)
    if mask_type == "binary":
        arr = arr.astype(bool)
    rle = aiod_rle.encode(arr, mask_type=mask_type, metadata={})
    if not keep_mask_type:
        del rle[-1]["metadata"]["mask_type"]
    aiod_rle.save_encoding(rle, path)


def make_inputs(
    src, shape, grid, overlap, mask_type, seed=0, empty_tiles=(), keep_mask_type=True,
    prep_hash=None, content=None,
):
    mask_fname = get_mask_name(run_hash=RUN_HASH, image_id=IMAGE_ID, prep_hash=prep_hash)
    rng = np.random.default_rng(seed)
    names = []
    for k, (x0, x1, y0, y1, z0, z1) in enumerate(tile_grid(shape, grid, overlap)):
        tshape = (z1 - z0, y1 - y0, x1 - x0)
        if content is not None:
            arr = content(k, tshape)
        else:
            arr = tile_content(tshape, mask_type, rng, empty=k in empty_tiles)
        name = f"{mask_fname}_x{x0}-{x1}_y{y0}-{y1}_z{z0}-{z1}.rle"
        save_substack(src / name, arr, mask_type, keep_mask_type)
        names.append(name)
    return mask_fname, names




GEOMETRIES = {
    "z-only": ((12, 20, 20), (3, 1, 1)),
    "xy": ((6, 20, 20), (2, 2, 2)),
    "xy-H>W": ((5, 24, 14), (2, 3, 2)),
    "xy-W>H": ((5, 14, 24), (1, 2, 3)),
    "z-only-H>W": ((8, 26, 10), (2, 1, 1)),
    "z-only-W>H": ((8, 10, 26), (2, 1, 1)),
    "last-tile-larger": ((13, 23, 22), (3, 2, 2)),
    "single-3d": ((7, 15, 15), (1, 1, 1)),
    "2d-single": ((1, 20, 24), (1, 1, 1)),
    "2d-xy": ((1, 20, 24), (1, 2, 3)),
}


# ---------------------------------------------------------------- units


@pytest.mark.parametrize("mask_type", ["instance", "binary"])
def test_rle_tile_matches_decode(mask_type):
    rng = np.random.default_rng(3)
    arr = tile_content((6, 9, 11), mask_type, rng)
    arr, _, _ = relabel_sequential(arr)
    if mask_type == "binary":
        arr = arr.astype(bool)
    rle = aiod_rle.encode(arr, mask_type=mask_type, metadata={})
    expected, _ = aiod_rle.decode(rle)
    tile = cs.RLETile(rle)
    assert tile.dtype == expected.dtype
    np.testing.assert_array_equal(tile.labels, np.unique(expected[expected > 0]).astype(int))
    for i in range(arr.shape[0]):
        np.testing.assert_array_equal(tile[i], expected[i])


def test_label_offsets():
    class Fake:
        def __init__(self, labels, dtype=np.uint16):
            self.labels = np.array(labels, dtype=np.int64)
            self.dtype = np.dtype(dtype)

    tiles = [Fake([1, 2, 3]), Fake([]), Fake([1, 5]), Fake([1])]
    assert cs.label_offsets(tiles) == [0, 3, 3, 8]
    # Wraps in the tile's combine dtype
    assert cs.label_offsets([Fake([65535]), Fake([1, 2])]) == [0, 65535]
    assert cs.label_offsets([Fake([1], bool)] * 3) == [0, 1, 2]
    # A tile wrapping to 0 leaves the running maximum where it was
    assert cs.label_offsets([Fake([1], bool)] * 258)[-4:] == [254, 255, 255, 255]


def test_tiles_open_lazily_and_release():
    """Only tiles covering the current z are held."""
    alive, opened = set(), []

    class Tile:
        dtype = np.dtype(np.uint16)
        labels = np.array([1])

        def __init__(self, k, shape):
            self.k, self.shape = k, shape
            alive.add(k)
            opened.append(k)

        def __getitem__(self, i):
            return np.ones(self.shape, dtype=np.uint16)

        def __del__(self):
            alive.discard(self.k)

    idx_list = tile_grid((12, 8, 8), (3, 2, 1), 0.0)
    tiles = [
        (idxs, lambda k=k, idxs=idxs: Tile(k, (idxs[3] - idxs[2], idxs[1] - idxs[0])))
        for k, idxs in enumerate(idx_list)
    ]
    max_alive = 0
    for z, plane in enumerate(cs.iter_combined_slices(tiles, (12, 8, 8), overlap=False)):
        max_alive = max(max_alive, len(alive))
        assert plane.shape == (8, 8)
    assert sorted(opened) == list(range(len(tiles)))
    # One z-layer of the 3x2x1 grid
    assert max_alive == 2


@pytest.mark.parametrize("mask_type", ["instance", "binary"])
@pytest.mark.parametrize("geometry", ["z-only", "xy", "xy-H>W", "2d-xy", "single-3d"])
def test_combined_max_matches_planes(tmp_path, geometry, mask_type):
    shape, grid = GEOMETRIES[geometry]
    _, names = make_inputs(tmp_path, shape, grid, 0.0, mask_type, empty_tiles=(1,))
    random.Random(0).shuffle(names)
    tiles = [
        (cs.extract_idxs_from_fname(n), lambda n=n: cs.RLETile.load(tmp_path / n))
        for n in names
    ]
    planes = cs.iter_combined_slices(tiles, shape, overlap=False)
    assert cs.combined_max(tiles, shape) == max(int(p.max()) for p in planes)
