"""Unit tests for the slab-wise (streaming) helpers in combine_stacks.py.

The invariant that matters: encode_streaming must produce output identical to handing the
whole volume to aiod_rle.encode. That is what makes the memory fix in AIOD-266 safe --
the .rle format is unchanged, so nothing downstream needs to know about it.

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
    "binary_slab_exact": (_binary((cs.ENCODE_SLAB, 9, 11), 0.2), "binary"),
    "binary_slab_plus_one": (_binary((cs.ENCODE_SLAB + 1, 9, 11), 0.2), "binary"),
    "instance": (_instance((23, 29, 31)), "instance"),
    "instance_uint16_labels": (_instance((13, 11, 17), 300), "instance"),
}


@pytest.mark.parametrize("name", list(VOLUMES))
@pytest.mark.parametrize("slab", [1, 2, 3, 7, 64, 10_000])
def test_encode_streaming_matches_encode(name, slab):
    """Slab-wise encoding is identical to whole-volume encoding, at any slab size."""
    vol, mask_type = VOLUMES[name]
    expected = aiod_rle.encode(vol.copy(), mask_type=mask_type, metadata={})
    got = cs.encode_streaming(vol.copy(), mask_type=mask_type, metadata={}, slab=slab)
    assert got == expected


@pytest.mark.parametrize("name", list(VOLUMES))
def test_encode_streaming_roundtrips(name):
    """Decoding streamed output reproduces the same pixels as decoding whole output."""
    vol, mask_type = VOLUMES[name]
    expected, _ = aiod_rle.decode(
        aiod_rle.encode(vol.copy(), mask_type=mask_type, metadata={})
    )
    got, _ = aiod_rle.decode(
        cs.encode_streaming(vol.copy(), mask_type=mask_type, metadata={}, slab=4)
    )
    assert np.array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("name", list(VOLUMES))
def test_encode_streaming_from_zarr_store(tmp_path, name):
    """A Zarr-backed volume encodes identically to the equivalent numpy array."""
    vol, mask_type = VOLUMES[name]
    expected = aiod_rle.encode(vol.copy(), mask_type=mask_type, metadata={})
    store = cs._open_store(tmp_path / f"{name}.zarr", vol.shape, vol.dtype)
    store[:] = vol
    got = cs.encode_streaming(store, mask_type=mask_type, metadata={}, slab=4)
    assert got == expected


def test_encode_streaming_2d():
    """2D input is delegated straight through, matching encode's own 2D handling."""
    vol = _binary((13, 17), 0.3)
    expected = aiod_rle.encode(vol.copy(), mask_type="binary", metadata={})
    got = cs.encode_streaming(vol.copy(), mask_type="binary", metadata={})
    assert got == expected


def test_encode_streaming_preserves_extra_metadata():
    """Caller metadata (e.g. downsample_factor) survives, alongside mask_type."""
    vol = _binary((9, 11, 13), 0.2)
    meta = {"downsample_factor": (1, 2, 2)}
    got = cs.encode_streaming(vol.copy(), mask_type="binary", metadata=meta, slab=2)
    assert got[-1]["metadata"]["downsample_factor"] == (1, 2, 2)
    assert got[-1]["metadata"]["mask_type"] == "binary"
    # Exactly one trailing metadata dict, and one entry per z-slice
    assert len(got) == vol.shape[0] + 1
    assert sum("metadata" in entry for entry in got if isinstance(entry, dict)) == 1


def test_encode_streaming_rejects_unresolved_mask_type():
    """Refuses to guess: per-slab inference could mix types within one file."""
    with pytest.raises(ValueError, match="must be resolved"):
        cs.encode_streaming(_binary((5, 7, 9)), mask_type=None, metadata={})


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
    store = cs._open_store(tmp_path / "w.zarr", vol.shape, np.uint16)
    block = vol[5:20, 4:18, 6:22]
    cs.write_blocks(store, block, (5, 4, 6), slab=slab)
    assert np.array_equal(store[5:20, 4:18, 6:22], block)


def test_write_blocks_add_accumulates(tmp_path):
    """add=True sums into the region, as the overlap > 0 path requires."""
    store = cs._open_store(tmp_path / "a.zarr", (20, 10, 10), np.uint16)
    block = np.ones((8, 6, 6), dtype=np.uint8)
    cs.write_blocks(store, block, (2, 1, 1), add=True, slab=3)
    cs.write_blocks(store, block, (2, 1, 1), add=True, slab=3)
    assert np.array_equal(store[2:10, 1:7, 1:7], np.full((8, 6, 6), 2))


def test_max_over_z_and_global_max(tmp_path):
    vol = _instance((40, 12, 14), 100)
    vol[7, 3, 4] = 999
    store = cs._open_store(tmp_path / "m.zarr", vol.shape, np.uint16)
    store[:] = vol
    assert cs.global_max(store) == int(vol.max())
    assert cs.max_over_z(store, 0, 10) == int(vol[0:10].max())
    assert cs.max_over_z(store, 20, 30) == int(vol[20:30].max())


def test_iter_pages_yields_every_slice_in_order(tmp_path):
    vol = _instance((70, 8, 9), 20)
    store = cs._open_store(tmp_path / "p.zarr", vol.shape, np.uint16)
    store[:] = vol
    pages = list(cs.iter_pages(store))
    assert len(pages) == vol.shape[0]
    assert np.array_equal(np.stack(pages), vol)


def test_iter_slabs_covers_array_exactly(tmp_path):
    vol = _instance((70, 6, 7), 20)
    store = cs._open_store(tmp_path / "s.zarr", vol.shape, np.uint16)
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
    store3d = cs._open_store(tmp_path / "c3.zarr", (10, 20, 30), np.uint16)
    assert all(c <= s for c, s in zip(store3d.chunks, store3d.shape, strict=True))
    store2d = cs._open_store(tmp_path / "c2.zarr", (20, 30), np.uint16)
    assert len(store2d.chunks) == 2
    assert all(c <= s for c, s in zip(store2d.chunks, store2d.shape, strict=True))


def test_zarr_store_is_compressed(tmp_path):
    """Sanity check that mask data actually compresses; the fix relies on it."""
    vol = _binary((64, 256, 256), 0.02, seed=1)
    store = cs._open_store(tmp_path / "z.zarr", vol.shape, np.uint16)
    store[:] = vol
    on_disk = sum(
        p.stat().st_size for p in (tmp_path / "z.zarr").rglob("*") if p.is_file()
    )
    assert on_disk < vol.nbytes * 2 / 4, (
        f"expected strong compression, got {on_disk} bytes for {vol.nbytes * 2} uncompressed"
    )
