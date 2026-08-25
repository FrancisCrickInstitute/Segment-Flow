"""Unit tests for the slab-wise/slice-wise helpers in combine_stacks.py.

The invariant that matters: encode_slicewise must produce output identical to handing
the whole volume to aiod_rle.encode. That is what makes the memory fix safe -- the
.rle format is unchanged, so nothing downstream needs to know about it.

NOTE: CI does not currently run Python tests (see .github/workflows/), so run these
manually inside the combine_stacks conda environment:
    pytest tests/test_combine_stacks.py
"""

import importlib.util
from pathlib import Path

import aiod_utils.rle as aiod_rle
import numpy as np
import pytest

SCRIPT = (
    Path(__file__).parent.parent / "modules/models/resources/usr/bin/combine_stacks.py"
)
_spec = importlib.util.spec_from_file_location("combine_stacks", SCRIPT)
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def _binary(shape, density=0.1, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < density).astype(np.uint8)


def _instance(shape, n_labels=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_labels, shape).astype(np.uint16)


VOLUMES = {
    "binary_sparse": (_binary((37, 41, 53), 0.05), "binary"),
    "binary_dense": (_binary((20, 17, 19), 0.7), "binary"),
    "binary_zeros": (np.zeros((11, 13, 17), np.uint8), "binary"),
    "binary_ones": (np.ones((11, 13, 17), np.uint8), "binary"),
    "binary_single_slice": (_binary((1, 13, 17), 0.5), "binary"),
    "binary_slab_exact": (_binary((cs.SLAB_SIZE, 9, 11), 0.2), "binary"),
    "binary_slab_plus_one": (_binary((cs.SLAB_SIZE + 1, 9, 11), 0.2), "binary"),
    "instance": (_instance((23, 29, 31)), "instance"),
    "instance_uint16_labels": (_instance((13, 11, 17), 300), "instance"),
}


@pytest.mark.parametrize("name", list(VOLUMES))
def test_encode_slicewise_matches_encode(name):
    """Slice-wise encoding is identical to whole-volume encoding."""
    vol, mask_type = VOLUMES[name]
    expected = aiod_rle.encode(vol.copy(), mask_type=mask_type, metadata={})
    got = cs.encode_slicewise(vol.copy(), mask_type=mask_type, metadata={})
    assert got == expected


@pytest.mark.parametrize("name", list(VOLUMES))
def test_encode_slicewise_roundtrips(name):
    """Decoding slice-wise output reproduces the same pixels as decoding whole output."""
    vol, mask_type = VOLUMES[name]
    expected, _ = aiod_rle.decode(
        aiod_rle.encode(vol.copy(), mask_type=mask_type, metadata={})
    )
    got, _ = aiod_rle.decode(
        cs.encode_slicewise(vol.copy(), mask_type=mask_type, metadata={})
    )
    assert np.array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("name", list(VOLUMES))
def test_encode_slicewise_from_zarr_store(tmp_path, name):
    """A Zarr-backed volume encodes identically to the equivalent numpy array."""
    vol, mask_type = VOLUMES[name]
    expected = aiod_rle.encode(vol.copy(), mask_type=mask_type, metadata={})
    store = cs._open_store(tmp_path / f"{name}.zarr", vol.shape, vol.dtype, chunks=vol.shape)
    store[:] = vol
    got = cs.encode_slicewise(store, mask_type=mask_type, metadata={})
    assert got == expected


def test_encode_slicewise_2d():
    """2D input is delegated straight through, matching encode's own 2D handling."""
    vol = _binary((13, 17), 0.3)
    expected = aiod_rle.encode(vol.copy(), mask_type="binary", metadata={})
    got = cs.encode_slicewise(vol.copy(), mask_type="binary", metadata={})
    assert got == expected


def test_encode_slicewise_preserves_extra_metadata():
    """Caller metadata (e.g. downsample_factor) survives, alongside mask_type."""
    vol = _binary((9, 11, 13), 0.2)
    meta = {"downsample_factor": (1, 2, 2)}
    got = cs.encode_slicewise(vol.copy(), mask_type="binary", metadata=meta)
    assert got[-1]["metadata"]["downsample_factor"] == (1, 2, 2)
    assert got[-1]["metadata"]["mask_type"] == "binary"
    # Exactly one trailing metadata dict, and one entry per z-slice
    assert len(got) == vol.shape[0] + 1
    assert sum("metadata" in entry for entry in got if isinstance(entry, dict)) == 1


def test_encode_slicewise_rejects_unresolved_mask_type():
    """Refuses to guess: per-slice inference could mix types within one file."""
    with pytest.raises(ValueError, match="must be resolved"):
        cs.encode_slicewise(_binary((5, 7, 9)), mask_type=None, metadata={})


@pytest.mark.parametrize("name", list(VOLUMES))
def test_rechunk_slicewise_preserves_data(tmp_path, name):
    """Rechunking to one z-slice per chunk doesn't change the array's contents."""
    vol, _ = VOLUMES[name]
    if vol.ndim != 3:
        pytest.skip("rechunk_slicewise is only used for 3D stores")
    store = cs._open_store(
        tmp_path / f"{name}_src.zarr", vol.shape, vol.dtype, chunks=vol.shape
    )
    store[:] = vol
    out = cs.rechunk_slicewise(store, tmp_path / f"{name}_out.zarr")
    assert out.shape == vol.shape
    assert out.chunks == (1, *vol.shape[1:])
    assert np.array_equal(np.asarray(out), vol)


@pytest.mark.parametrize(
    ("vol", "expected"),
    [
        (np.zeros((9, 5, 5), np.uint8), "binary"),
        (np.ones((9, 5, 5), np.uint8), "binary"),
        (_binary((9, 5, 5), 0.4), "binary"),
        (np.zeros((9, 5, 5), bool), "binary"),
        (_instance((9, 5, 5), 8), "instance"),
    ],
)
def test_resolve_mask_type_matches_check_mask_type(vol, expected):
    """Streaming inference agrees with aiod_rle.check_mask_type on the whole array."""
    assert cs.resolve_mask_type(vol, None) == expected
    assert aiod_rle.check_mask_type(vol) == expected


def test_resolve_mask_type_passes_through_explicit_value():
    assert cs.resolve_mask_type(_instance((5, 5, 5)), "binary") == "binary"


@pytest.mark.parametrize("slab", [1, 5, 64])
def test_write_blocks_matches_direct_assignment(tmp_path, slab):
    """Slab-wise writes land the same bytes as a single __setitem__."""
    vol = _instance((30, 24, 26), 50)
    store = cs._open_store(tmp_path / "w.zarr", vol.shape, np.uint16, chunks=vol.shape)
    block = vol[5:20, 4:18, 6:22]
    cs.write_blocks(store, block, (5, 4, 6), slab=slab)
    assert np.array_equal(store[5:20, 4:18, 6:22], block)


def test_write_blocks_add_accumulates(tmp_path):
    """add=True sums into the region, as the overlap > 0 path requires."""
    shape = (20, 10, 10)
    store = cs._open_store(tmp_path / "a.zarr", shape, np.uint16, chunks=shape)
    block = np.ones((8, 6, 6), dtype=np.uint8)
    cs.write_blocks(store, block, (2, 1, 1), add=True, slab=3)
    cs.write_blocks(store, block, (2, 1, 1), add=True, slab=3)
    assert np.array_equal(store[2:10, 1:7, 1:7], np.full((8, 6, 6), 2))


def test_slab_max_over_range_and_whole_array(tmp_path):
    """slab_max(arr) covers the whole array; slab_max(arr, start, end) restricts to a range."""
    vol = _instance((40, 12, 14), 100)
    vol[7, 3, 4] = 999
    store = cs._open_store(tmp_path / "m.zarr", vol.shape, np.uint16, chunks=vol.shape)
    store[:] = vol
    assert cs.slab_max(store) == int(vol.max())
    assert cs.slab_max(store, 0, 10) == int(vol[0:10].max())
    assert cs.slab_max(store, 20, 30) == int(vol[20:30].max())


def test_iter_pages_yields_every_slice_in_order(tmp_path):
    vol = _instance((70, 8, 9), 20)
    store = cs._open_store(tmp_path / "p.zarr", vol.shape, np.uint16, chunks=vol.shape)
    store[:] = vol
    pages = list(cs.iter_pages(store))
    assert len(pages) == vol.shape[0]
    assert np.array_equal(np.stack(pages), vol)


def test_iter_slabs_covers_array_exactly(tmp_path):
    vol = _instance((70, 6, 7), 20)
    store = cs._open_store(tmp_path / "s.zarr", vol.shape, np.uint16, chunks=vol.shape)
    store[:] = vol
    seen = [(z0, z1) for z0, z1, _ in cs.iter_slabs(store, slab=16)]
    assert seen[0][0] == 0
    assert seen[-1][1] == vol.shape[0]
    # Pairwise over consecutive slabs, so the second sequence is shorter by one
    assert all(a[1] == b[0] for a, b in zip(seen, seen[1:], strict=False))
    assert np.array_equal(
        np.concatenate([b for _, _, b in cs.iter_slabs(store, slab=16)]), vol
    )


def test_open_store_clamps_chunks_to_shape(tmp_path):
    """Chunks must never exceed the array, including for 2D volumes."""
    store3d = cs._open_store(
        tmp_path / "c3.zarr", (10, 20, 30), np.uint16, chunks=(100, 100, 100)
    )
    assert all(c <= s for c, s in zip(store3d.chunks, store3d.shape, strict=True))
    store2d = cs._open_store(
        tmp_path / "c2.zarr", (20, 30), np.uint16, chunks=(1, 100, 100)
    )
    assert len(store2d.chunks) == 2
    assert all(c <= s for c, s in zip(store2d.chunks, store2d.shape, strict=True))


def test_zarr_store_is_compressed(tmp_path):
    """Sanity check that mask data actually compresses; the fix relies on it."""
    vol = _binary((64, 256, 256), 0.02, seed=1)
    store = cs._open_store(tmp_path / "z.zarr", vol.shape, np.uint16, chunks=vol.shape)
    store[:] = vol
    on_disk = sum(
        p.stat().st_size for p in (tmp_path / "z.zarr").rglob("*") if p.is_file()
    )
    assert on_disk < vol.nbytes * 2 / 4, (
        f"expected strong compression, got {on_disk} bytes for {vol.nbytes * 2} uncompressed"
    )


def test_insert_mask_forwards_slab_size(tmp_path, monkeypatch):
    """--slab-size is resolved once in __main__ and passed down explicitly as a plain
    argument (slab_size on combine_masks/insert_mask, slab on write_blocks/slab_max) --
    not via a mutable module global. This checks insert_mask actually forwards the
    value it's given to both of its own slab-based calls, rather than silently falling
    back to the SLAB_SIZE default.
    """
    shape = (10, 5, 5)
    store = cs._open_store(tmp_path / "sm.zarr", shape, np.uint16, chunks=shape)
    seen_slabs = []
    real_slab_max, real_write_blocks = cs.slab_max, cs.write_blocks

    def spy_slab_max(*args, **kwargs):
        seen_slabs.append(kwargs.get("slab"))
        return real_slab_max(*args, **kwargs)

    def spy_write_blocks(*args, **kwargs):
        seen_slabs.append(kwargs.get("slab"))
        return real_write_blocks(*args, **kwargs)

    monkeypatch.setattr(cs, "slab_max", spy_slab_max)
    monkeypatch.setattr(cs, "write_blocks", spy_write_blocks)

    mask = _instance((5, 5, 5), 3)
    cs.insert_mask(
        all_masks=store,
        mask=mask,
        idxs=(0, 5, 0, 5, 0, 5),
        xy_tiling=True,
        is_overlap=False,
        is_2d=False,
        slab_size=7,
    )
    assert seen_slabs == [7, 7]  # one slab_max call, one write_blocks call, both slab=7


def test_insert_mask_and_write_blocks_default_to_slab_size_constant(tmp_path):
    """With no slab_size passed, behaviour still matches the SLAB_SIZE constant --
    the default argument value, not a runtime lookup."""
    shape = (5, 5, 5)
    store = cs._open_store(tmp_path / "d.zarr", shape, np.uint16, chunks=shape)
    mask = _instance((5, 5, 5), 3)
    result = cs.insert_mask(
        all_masks=store,
        mask=mask.copy(),
        idxs=(0, 5, 0, 5, 0, 5),
        xy_tiling=False,
        is_overlap=False,
        is_2d=False,
    )
    assert np.array_equal(np.asarray(result[:]), mask)
