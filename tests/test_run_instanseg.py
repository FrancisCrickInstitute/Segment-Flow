"""Smoke tests for run_instanseg.py against the tensors InstanSeg ships.

Every InstanSeg release zip carries a test-input.npy and the
test-output_instance_segmentation.npy its authors produced from it, so the
whole call path -- checkpoint loading, target selection, channel layout and
output unpacking -- can be checked without any of our own reference data.

The tests skip when the checkpoints are not in the AIoD cache, so a machine
that has never run the pipeline stays green. Point AIOD_INSTANSEG_MODEL_DIR at
a directory of release zips (or extracted model directories) to run them.
"""

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

BIN_DIR = Path(__file__).resolve().parents[1] / "modules/models/resources/usr/bin"
sys.path.insert(0, str(BIN_DIR))

# Where downloadArtifact leaves the checkpoints, and an override for anyone
# holding the zips elsewhere
DEFAULT_CACHE = Path.home() / ".nextflow/aiod/aiod_cache/instanseg/checkpoints"
MODEL_DIR = Path(os.environ.get("AIOD_INSTANSEG_MODEL_DIR", DEFAULT_CACHE))

pytest.importorskip("torch", reason="InstanSeg environment not active")
pytest.importorskip("instanseg", reason="InstanSeg environment not active")

import run_instanseg  # noqa: E402


def _find_checkpoint(version: str) -> Path:
    """Locate a checkpoint for `version`, whatever name it was cached under."""
    if not MODEL_DIR.is_dir():
        pytest.skip(f"No InstanSeg checkpoints at {MODEL_DIR}")
    matches = sorted(p for p in MODEL_DIR.glob(f"*{version}*") if p.is_file())
    if not matches:
        pytest.skip(f"No checkpoint for {version} in {MODEL_DIR}")
    return matches[0]


def _reference_tensors(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the (input, expected output) pair bundled with a checkpoint zip."""
    if not zipfile.is_zipfile(checkpoint):
        pytest.skip(f"{checkpoint} is not a zip, so it carries no test tensors")
    with zipfile.ZipFile(checkpoint) as archive:
        import io

        def load(name):
            member = run_instanseg._find_member(archive, name)
            return np.load(io.BytesIO(archive.read(member)))

        return load("test-input.npy"), load("test-output_instance_segmentation.npy")


def _segment(checkpoint: Path, image: np.ndarray, task: str, **config):
    """Run one plane through the script the way the pipeline would."""
    from instanseg import InstanSeg

    module = run_instanseg._load_torchscript(checkpoint)
    target = run_instanseg._resolve_target(task, bool(module.cells_and_nuclei))
    model = InstanSeg(model_type=module, device="cpu", verbosity=0)
    shared = {
        "pixel_size": None,
        "normalise": True,
        "rescale_output": True,
        "target": target,
        "return_image_tensor": False,
    }
    return run_instanseg._predict_plane(model, image, "CYX", config, **shared)


def _count(labels: np.ndarray) -> int:
    return len(np.unique(labels)) - (1 if (labels == 0).any() else 0)


@pytest.mark.parametrize(
    ("version", "task", "channel"),
    [
        ("brightfield_nuclei", "nuclei", 0),
        ("fluorescence_nuclei_and_cells", "nuclei", 0),
        ("fluorescence_nuclei_and_cells", "cyto", 1),
    ],
)
def test_matches_reference_object_count(version, task, channel):
    """Each task must reproduce its own channel of the reference output.

    Counts rather than exact labels: InstanSeg is only deterministic for a
    given device and precision, and the references were made on the authors'.
    """
    checkpoint = _find_checkpoint(version)
    image, expected = _reference_tensors(checkpoint)

    labels = _segment(checkpoint, image[0], task)
    expected_count = _count(expected[0, channel])
    actual_count = _count(labels)

    assert labels.shape == image.shape[-2:]
    assert expected_count > 0, "reference output is empty; wrong fixture?"
    assert actual_count == pytest.approx(expected_count, rel=0.1), (
        f"{version}/{task}: found {actual_count} objects, "
        f"reference has {expected_count}"
    )


def test_cyto_rejected_on_single_head_checkpoint():
    """A cells target is silently ignored upstream, so we must refuse it."""
    with pytest.raises(ValueError, match="single output head"):
        run_instanseg._resolve_target("cyto", cells_and_nuclei=False)


def test_targets_select_different_outputs():
    """Nuclei and cells must not come back as the same mask."""
    checkpoint = _find_checkpoint("fluorescence_nuclei_and_cells")
    image, _ = _reference_tensors(checkpoint)

    nuclei = _segment(checkpoint, image[0], "nuclei")
    cells = _segment(checkpoint, image[0], "cyto")

    assert not np.array_equal(nuclei, cells)
    # A cell contains its nucleus, so cells cover at least as much area
    assert (cells > 0).sum() >= (nuclei > 0).sum()


def test_stack_slices_keeps_labels_unique():
    """Per-slice IDs restart at 1, so stacking must offset them."""
    first = np.array([[0, 1], [1, 2]])
    second = np.array([[1, 1], [0, 2]])

    stacked = run_instanseg._stack_slices([first, second])

    assert stacked.shape == (2, 2, 2)
    assert set(np.unique(stacked[0])) == {0, 1, 2}
    assert set(np.unique(stacked[1])) == {0, 3, 4}
    # Background must stay background
    assert stacked[0][0, 0] == 0
    assert stacked[1][1, 0] == 0


def test_stack_slices_handles_empty_slices():
    """An empty slice must not consume label values or break the running max."""
    stacked = run_instanseg._stack_slices(
        [np.array([[0, 1]]), np.zeros((1, 2), dtype=int), np.array([[0, 1]])]
    )

    assert set(np.unique(stacked)) == {0, 1, 2}


def test_channel_ids_rejects_out_of_range():
    """InstanSeg's own bound check is off by one, so we check first."""
    assert run_instanseg._resolve_channel_ids("0, 2", 3) == [0, 2]
    assert run_instanseg._resolve_channel_ids([0, 2], 3) == [0, 2]
    assert run_instanseg._resolve_channel_ids(None, 3) is None
    assert run_instanseg._resolve_channel_ids("", 3) is None
    with pytest.raises(ValueError, match="out of range"):
        run_instanseg._resolve_channel_ids("3", 3)


def test_plane_checks_catch_ambiguous_geometry():
    """Guard the two shapes InstanSeg would misread rather than reject."""
    with pytest.raises(ValueError, match="below InstanSeg's minimum"):
        run_instanseg._check_plane(np.zeros((3, 16, 256)), "CYX")
    with pytest.raises(ValueError, match="infers the channel axis"):
        run_instanseg._check_plane(np.zeros((64, 64, 64)), "CYX")
    assert run_instanseg._check_plane(np.zeros((3, 256, 256)), "CYX") == (256, 256)
    assert run_instanseg._check_plane(np.zeros((256, 256)), "YX") == (256, 256)


def test_channel_count_mismatch_is_explicit():
    """A wrong version choice must not reach TorchScript as a shape error."""
    check = run_instanseg._check_channels
    rgb = np.zeros((3, 1, 64, 64))
    rgba = np.zeros((4, 1, 64, 64))

    assert check("brightfield_nuclei", "CYX", 3, rgb).shape[0] == 3
    # RGBA is tolerated by dropping alpha, since load_img counts it as a channel
    assert check("brightfield_nuclei", "CYX", 4, rgba).shape[0] == 3
    # The channel-invariant version accepts anything
    assert check("fluorescence_nuclei_and_cells", "CYX", 7, rgb) is rgb

    with pytest.raises(ValueError, match="needs exactly 3"):
        check("brightfield_nuclei", "CYX", 1, np.zeros((1, 1, 8, 8)))


def test_channel_count_unchecked_when_model_selects_a_channel():
    """A YX model has one channel picked for it, so the image's count is moot."""
    multi = np.zeros((5, 1, 64, 64))

    assert (
        run_instanseg._check_channels("single_channel_nuclei", "YX", 5, multi) is multi
    )


def test_tile_size_floor_is_enforced():
    """InstanSeg asserts on small tiles from deep inside its tiling code."""
    assert run_instanseg._resolve_tile_size({}) == 512
    assert run_instanseg._resolve_tile_size({"tile_size": 256}) == 256
    with pytest.raises(ValueError, match="at least 256"):
        run_instanseg._resolve_tile_size({"tile_size": 128})


def test_eval_kwargs_drop_unset_thresholds():
    """None means 'use this checkpoint's own default', which differs by version."""
    assert run_instanseg._build_eval_kwargs({"min_size": None}) == {}
    assert run_instanseg._build_eval_kwargs({"min_size": 25}) == {"min_size": 25}


def test_pixel_size_is_validated():
    """Blank means no rescaling; a bad value must not reach InstanSeg."""
    resolve = run_instanseg._resolve_pixel_size
    assert resolve(None) is None
    assert resolve("") is None
    assert resolve("None") is None
    assert resolve(0.5) == 0.5
    # A user-supplied config can carry a string
    assert resolve("0.25") == 0.25
    with pytest.raises(ValueError, match="greater than 0"):
        resolve(0)
    with pytest.raises(ValueError, match="must be a number"):
        resolve("half a micron")
