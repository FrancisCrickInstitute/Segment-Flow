"""Equivalence tests for combine_stacks.py against master's dense implementation.

Both scripts run end to end, in-process, on the same small substacks, and
their outputs must match exactly: RLE lists ==, TIFFs pixel-, dtype- and
shape-equal. The reference is a verbatim copy of master's script in
tests/fixtures/combine_stacks_master.py.

Run inside the combine_stacks conda env (CI does not run Python tests):
    pytest tests/test_combine_stacks_vs_master.py
"""

import json
import random
import runpy
import sys
import warnings
from pathlib import Path

import aiod_utils.rle as aiod_rle
import numpy as np
import pytest
import tifffile
from aiod_utils.io import get_combined_mask_name, get_mask_name, reduce_dtype
from aiod_utils.preprocess import get_prep_hash
from aiod_utils.stacks import Stack, generate_stack_indices
from skimage.segmentation import relabel_sequential

TESTS_DIR = Path(__file__).parent
SCRIPT = TESTS_DIR.parent / "modules/models/resources/usr/bin/combine_stacks.py"
REFERENCE = TESTS_DIR / "fixtures/combine_stacks_master.py"

IMAGE_ID = "test_tif"
RUN_HASH = "testhash"


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


# ---------------------------------------------------------------- running


def run_script(
    script, run_dir, src, names, mask_fname, shape, overlap, output_format="rle",
    output_mask_type="auto", postprocess=False, model="empanada", prep=None,
    prep_hash=None, monkeypatch=None,
):
    """Run one script's __main__ in run_dir, on symlinks to src/names."""
    run_dir.mkdir()
    for name in names:
        (run_dir / name).symlink_to(src / name)
    (run_dir / "prep.json").write_text(json.dumps(prep or []))
    argv = [
        str(script),
        "--mask-fname", mask_fname,
        "--output-dir", ".",
        "--masks", *names,
        "--model", model,
        "--image-size", *map(str, shape),
        "--overlap", *[str(overlap)] * 3,
        "--output-format", output_format,
        "--output-mask-type", output_mask_type,
        "--preprocess-config", "prep.json",
        "--image-id", IMAGE_ID,
        "--param-hash", RUN_HASH,
    ]
    if prep_hash:
        argv += ["--prep-hash", prep_hash]
    if postprocess:
        argv.append("--postprocess")
    with monkeypatch.context() as m:
        m.chdir(run_dir)
        m.setattr(sys, "argv", argv)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runpy.run_path(str(script), run_name="__main__")
    # Inputs are unlinked by the script once combined
    assert not any((run_dir / name).exists() for name in names)
    return run_dir / get_combined_mask_name(mask_fname, output_format)


def assert_same_output(ref_path, new_path, output_format):
    if output_format == "rle":
        ref = aiod_rle.load_encoding(ref_path)
        new = aiod_rle.load_encoding(new_path)
        assert len(new) == len(ref)
        assert new[-1] == ref[-1]
        for i, (r, n) in enumerate(zip(ref, new, strict=True)):
            assert n == r, f"slice {i} differs"
    else:
        with tifffile.TiffFile(ref_path) as rt, tifffile.TiffFile(new_path) as nt:
            ref, new = rt.asarray(), nt.asarray()
            assert rt.imagej_metadata == nt.imagej_metadata
            assert len(rt.pages) == len(nt.pages)
        assert new.dtype == ref.dtype
        assert new.shape == ref.shape
        np.testing.assert_array_equal(new, ref)


@pytest.fixture
def compare(tmp_path, monkeypatch):
    """compare(shape, grid, overlap, mask_type, **opts): run both, assert equal."""

    def _compare(
        shape, grid, overlap, mask_type, output_format="rle", output_mask_type="auto",
        postprocess=False, model="empanada", order="gen", seed=0, empty_tiles=(),
        keep_mask_type=True, prep=None, prep_hash=None, content=None,
    ):
        src = tmp_path / "src"
        src.mkdir(exist_ok=True)
        mask_fname, names = make_inputs(
            src, shape, grid, overlap, mask_type, seed=seed, empty_tiles=empty_tiles,
            keep_mask_type=keep_mask_type, prep_hash=prep_hash, content=content,
        )
        if order == "shuffle":
            random.Random(seed).shuffle(names)
        kwargs = dict(
            src=src, names=names, mask_fname=mask_fname, shape=shape, overlap=overlap,
            output_format=output_format, output_mask_type=output_mask_type,
            postprocess=postprocess, model=model, prep=prep, prep_hash=prep_hash,
            monkeypatch=monkeypatch,
        )
        ref = run_script(REFERENCE, tmp_path / "ref", **kwargs)
        new = run_script(SCRIPT, tmp_path / "new", **kwargs)
        assert_same_output(ref, new, output_format)
        return ref, new

    return _compare


# ---------------------------------------------------------------- equivalence

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


@pytest.mark.parametrize("output_format", ["rle", "tiff"])
@pytest.mark.parametrize("mask_type", ["instance", "binary"])
@pytest.mark.parametrize("overlap", [0.0, 0.2])
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_matches_master(compare, geometry, overlap, mask_type, output_format):
    shape, grid = GEOMETRIES[geometry]
    compare(shape, grid, overlap, mask_type, output_format=output_format, empty_tiles=(1,))


@pytest.mark.parametrize("output_format", ["rle", "tiff"])
@pytest.mark.parametrize("mask_type", ["instance", "binary"])
def test_corner_covered_by_8_substacks(compare, mask_type, output_format):
    # With overlap, the centre voxel lies in all 8 substacks of a 2x2x2 grid
    compare((10, 20, 20), (2, 2, 2), 0.3, mask_type, output_format=output_format)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("geometry", ["xy", "xy-H>W", "2d-xy"])
def test_shuffled_order(compare, geometry, seed):
    # Label offsets follow the --masks order, which Nextflow does not fix
    shape, grid = GEOMETRIES[geometry]
    compare(shape, grid, 0.0, "instance", order="shuffle", seed=seed, empty_tiles=(0, 3))


@pytest.mark.parametrize("output_format", ["rle", "tiff"])
@pytest.mark.parametrize("mask_type", ["instance", "binary"])
def test_all_empty(compare, mask_type, output_format):
    shape, grid = GEOMETRIES["xy"]
    compare(shape, grid, 0.0, mask_type, output_format=output_format,
            empty_tiles=range(8))


@pytest.mark.parametrize("geometry", ["xy", "2d-xy"])
def test_binary_as_instance_tiff(compare, geometry):
    # Binary substacks written as an instance TIFF keep their label offsets
    shape, grid = GEOMETRIES[geometry]
    compare(shape, grid, 0.0, "binary", output_format="tiff", output_mask_type="instance")


@pytest.mark.parametrize("output_mask_type", ["auto", "instance", "binary"])
@pytest.mark.parametrize("output_format", ["rle", "tiff"])
@pytest.mark.parametrize("mask_type", ["instance", "binary"])
def test_missing_mask_type(compare, mask_type, output_format, output_mask_type):
    shape, grid = GEOMETRIES["xy"]
    compare(shape, grid, 0.0, mask_type, output_format=output_format,
            output_mask_type=output_mask_type, keep_mask_type=False)


@pytest.mark.parametrize("output_mask_type", ["instance", "binary"])
@pytest.mark.parametrize("output_format", ["rle", "tiff"])
def test_cli_mask_type_differs_from_files(compare, output_format, output_mask_type):
    shape, grid = GEOMETRIES["xy"]
    for mask_type in ("instance", "binary"):
        if mask_type != output_mask_type:
            compare(shape, grid, 0.0, mask_type, output_format=output_format,
                    output_mask_type=output_mask_type)


@pytest.mark.parametrize("output_format", ["rle", "tiff"])
def test_uint16_label_wrap(compare, output_format):
    # Two 2D tiles of 40000 labels each: the second's offset labels wrap in uint16
    def content(k, tshape):
        return np.arange(1, np.prod(tshape) + 1, dtype=np.uint32).reshape(tshape)

    compare((1, 200, 400), (1, 1, 2), 0.0, "instance", output_format=output_format,
            content=content)


def test_uint8_binary_offset_wrap(compare):
    # 272 binary tiles in one plane: offsets pass 255 and wrap in uint8
    def content(k, tshape):
        return np.ones(tshape, dtype=np.uint8)

    compare((1, 32, 34), (1, 16, 17), 0.0, "binary", output_format="tiff",
            output_mask_type="instance", content=content)


def test_downsample_metadata(compare):
    methods = [{"name": "Downsample", "params": {"block_size": [1, 2, 2], "method": "max"}}]
    prep_hash = get_prep_hash(methods)
    ref, new = compare((4, 12, 12), (2, 1, 1), 0.0, "instance", prep=[methods],
                       prep_hash=prep_hash)
    assert aiod_rle.load_encoding(new)[-1]["metadata"]["downsample_factor"] == (1, 2, 2)


@pytest.mark.parametrize("output_format", ["rle", "tiff"])
@pytest.mark.parametrize(
    ("model", "mask_type", "geometry"),
    [
        ("empanada", "binary", "xy"),
        ("empanada", "instance", "z-only"),
        ("empanada", "binary", "single-3d"),
        ("sam", "instance", "xy"),
        ("sam", "instance", "single-3d"),
        ("sam", "instance", "2d-xy"),
        ("sam", "instance", "2d-single"),
    ],
)
def test_postprocess(compare, model, mask_type, geometry, output_format):
    shape, grid = GEOMETRIES[geometry]
    compare(shape, grid, 0.0, mask_type, output_format=output_format, postprocess=True,
            model=model)


def test_missing_mask_type_auto_single_label(compare):
    # Inferred from the combined values: one label is 'binary'
    def content(k, tshape):
        return np.ones(tshape, dtype=np.uint16)

    compare((4, 10, 10), (2, 1, 1), 0.0, "instance", keep_mask_type=False,
            content=content)


def test_inconsistent_mask_types_raise(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    mask_fname, names = make_inputs(src, (4, 10, 10), (2, 1, 1), 0.0, "instance")
    # Re-save the second substack as binary
    mask, _ = aiod_rle.decode(aiod_rle.load_encoding(src / names[1]))
    save_substack(src / names[1], (mask > 0).astype(np.uint8), "binary")
    for script, name in ((REFERENCE, "ref"), (SCRIPT, "new")):
        with pytest.raises(ValueError, match="Inconsistent mask types"):
            run_script(script, tmp_path / name, src, names, mask_fname, (4, 10, 10),
                       0.0, monkeypatch=monkeypatch)
