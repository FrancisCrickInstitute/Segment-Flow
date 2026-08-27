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
    store = cs._open_store(
        tmp_path / f"{name}.zarr", vol.shape, vol.dtype, chunks=vol.shape
    )
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
    blocks = list(cs.iter_slabs(store, slab=16))
    assert [b.shape[0] for b in blocks] == [16, 16, 16, 16, 6]
    assert np.array_equal(np.concatenate(blocks), vol)


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


def _patch_one_substack(monkeypatch, mask, mask_type):
    """Stand in for a single substack file covering a quarter tile of a 2x larger volume."""
    nz, ny, nx = mask.shape
    monkeypatch.setattr(cs, "extract_idxs_from_fname", lambda _: (0, nx, 0, ny, 0, nz))
    monkeypatch.setattr(
        cs.aiod_rle, "load_encoding", lambda _: [{"metadata": {"mask_type": mask_type}}]
    )
    monkeypatch.setattr(
        cs.aiod_rle,
        "decode",
        lambda _: (mask, {"metadata": {"mask_type": mask_type}}),
    )


def test_combine_masks_chunks_to_one_write_region(tmp_path, monkeypatch):
    """Chunked to what one write_blocks call covers, so writes and the slab reads in
    encode_slicewise are both whole-chunk and no rechunk pass is needed."""
    _patch_one_substack(monkeypatch, np.zeros((100, 12, 16), np.uint16), "instance")
    store, _ = cs.combine_masks(
        ["a.rle"],
        overlap=[0.0, 0.0, 0.0],
        image_size=(200, 24, 32),
        model="empanada",
        store_path=tmp_path / "cm.zarr",
        slab_size=8,
    )
    assert store.chunks == (8, 12, 16)


def test_combine_masks_binary_uses_bool_store_and_skips_relabel(tmp_path, monkeypatch):
    """Binary masks get a 1-byte store and no label offsetting -- encode re-binarises,
    so offsetting per x/y tile is two full passes over the substack thrown away."""
    mask = np.ones((10, 12, 16), bool)
    _patch_one_substack(monkeypatch, mask, "binary")
    store, mask_type = cs.combine_masks(
        ["a.rle"],
        overlap=[0.0, 0.0, 0.0],
        image_size=(20, 24, 32),
        model="empanada",
        store_path=tmp_path / "b.zarr",
        slab_size=8,
    )
    assert store.dtype == bool
    assert mask_type == "binary"
    # Untouched by any offset: still 0/1, not 0/k
    assert set(np.unique(np.asarray(store))) == {False, True}


def test_combine_masks_instance_uses_uint16_store(tmp_path, monkeypatch):
    _patch_one_substack(monkeypatch, _instance((10, 12, 16), 5), "instance")
    store, mask_type = cs.combine_masks(
        ["a.rle"],
        overlap=[0.0, 0.0, 0.0],
        image_size=(20, 24, 32),
        model="empanada",
        store_path=tmp_path / "i.zarr",
        slab_size=8,
    )
    assert store.dtype == np.uint16
    assert mask_type == "instance"


def test_combine_masks_binary_with_overlap_stays_summable(tmp_path, monkeypatch):
    """overlap > 0 sums overlapping regions, so a bool store would clip the vote count."""
    _patch_one_substack(monkeypatch, np.ones((10, 12, 16), bool), "binary")
    store, _ = cs.combine_masks(
        ["a.rle", "b.rle"],
        overlap=[0.1, 0.1, 0.1],
        image_size=(20, 24, 32),
        model="empanada",
        store_path=tmp_path / "bo.zarr",
        slab_size=8,
    )
    assert store.dtype == np.uint16
    # Both stubs land on the same tile, so the shared region accumulates to 2
    assert int(np.asarray(store).max()) == 2


def test_encode_slicewise_matches_encode_across_chunkings(tmp_path):
    """Reading via slabs must give byte-identical output whatever the store chunking --
    that independence is what lets combine_masks pick chunks for write efficiency."""
    vol, mask_type = VOLUMES["instance"]
    expected = aiod_rle.encode(vol.copy(), mask_type=mask_type, metadata={})
    for i, chunks in enumerate([(1, *vol.shape[1:]), (7, 5, 6), vol.shape]):
        store = cs._open_store(tmp_path / f"ck{i}.zarr", vol.shape, vol.dtype, chunks)
        store[:] = vol
        for slab in (1, 5, 64):
            got = cs.encode_slicewise(
                store, mask_type=mask_type, metadata={}, slab_size=slab
            )
            assert got == expected, f"chunks={chunks} slab={slab}"


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


def test_insert_mask_offset_matches_store_readback(tmp_path):
    """insert_mask tracks the max label in `label_ranges` instead of re-reading the
    store. The offset it applies must equal what slab_max would have read back over the
    substack's z-range, for every tile of a 2x2x2 layout -- tiles sharing a z-range see
    each other's labels, tiles in a different z-range do not.
    """
    shape = (20, 8, 8)
    store = cs._open_store(tmp_path / "lr.zarr", shape, np.uint16, chunks=(5, 4, 4))
    label_ranges = []
    for start_z, end_z in ((0, 10), (10, 20)):
        for start_y, end_y in ((0, 4), (4, 8)):
            for start_x, end_x in ((0, 4), (4, 8)):
                # What the old readback would have returned, taken before we write
                expected_offset = cs.slab_max(store, start_z, end_z)
                mask = _instance((end_z - start_z, 4, 4), 5)
                before = mask.copy()
                cs.insert_mask(
                    all_masks=store,
                    mask=mask,
                    idxs=(start_x, end_x, start_y, end_y, start_z, end_z),
                    relabel=True,
                    is_2d=False,
                    label_ranges=label_ranges,
                )
                written = np.asarray(store[start_z:end_z, start_y:end_y, start_x:end_x])
                assert np.array_equal(
                    written, np.where(before > 0, before + expected_offset, 0)
                )


def test_insert_mask_offset_matches_store_readback_2d(tmp_path):
    """For a 2D volume the offset covers the whole store, not a z-range."""
    store = cs._open_store(tmp_path / "lr2.zarr", (8, 8), np.uint16, chunks=(4, 4))
    label_ranges = []
    for start_y, end_y in ((0, 4), (4, 8)):
        for start_x, end_x in ((0, 4), (4, 8)):
            expected_offset = cs.slab_max(store)
            mask = _instance((4, 4), 5)
            before = mask.copy()
            cs.insert_mask(
                all_masks=store,
                mask=mask,
                idxs=(start_x, end_x, start_y, end_y, 0, 1),
                relabel=True,
                is_2d=True,
                label_ranges=label_ranges,
            )
            written = np.asarray(store[start_y:end_y, start_x:end_x])
            assert np.array_equal(
                written, np.where(before > 0, before + expected_offset, 0)
            )


def test_insert_mask_empty_substack_does_not_consume_labels(tmp_path):
    """An all-zero substack offsets nothing, so it must not advance the running max."""
    shape = (10, 4, 8)
    store = cs._open_store(tmp_path / "lre.zarr", shape, np.uint16, chunks=(5, 4, 4))
    label_ranges = []
    cs.insert_mask(
        all_masks=store,
        mask=np.zeros((10, 4, 4), np.uint16),
        idxs=(0, 4, 0, 4, 0, 10),
        relabel=True,
        is_2d=False,
        label_ranges=label_ranges,
    )
    mask = _instance((10, 4, 4), 5)
    before = mask.copy()
    cs.insert_mask(
        all_masks=store,
        mask=mask,
        idxs=(4, 8, 0, 4, 0, 10),
        relabel=True,
        is_2d=False,
        label_ranges=label_ranges,
    )
    assert np.array_equal(np.asarray(store[:, :, 4:8]), before)


def test_insert_mask_forwards_slab_size(tmp_path, monkeypatch):
    """--slab-size is resolved once in __main__ and passed down explicitly as a plain
    argument (slab_size on combine_masks/insert_mask, slab on write_blocks) -- not via a
    mutable module global. This checks insert_mask actually forwards the value it's
    given, rather than silently falling back to the SLAB_SIZE default.
    """
    shape = (10, 5, 5)
    store = cs._open_store(tmp_path / "sm.zarr", shape, np.uint16, chunks=shape)
    seen_slabs = []
    real_write_blocks = cs.write_blocks

    def spy_write_blocks(*args, **kwargs):
        seen_slabs.append(kwargs.get("slab"))
        return real_write_blocks(*args, **kwargs)

    def fail_slab_max(*args, **kwargs):
        raise AssertionError("insert_mask must not read the store back to find the max")

    monkeypatch.setattr(cs, "slab_max", fail_slab_max)
    monkeypatch.setattr(cs, "write_blocks", spy_write_blocks)

    mask = _instance((5, 5, 5), 3)
    cs.insert_mask(
        all_masks=store,
        mask=mask,
        idxs=(0, 5, 0, 5, 0, 5),
        relabel=True,
        is_2d=False,
        slab_size=7,
        label_ranges=[],
    )
    assert seen_slabs == [7]  # one write_blocks call, slab=7


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
        relabel=False,
        is_2d=False,
    )
    assert np.array_equal(np.asarray(result[:]), mask)
