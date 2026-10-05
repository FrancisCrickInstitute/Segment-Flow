#!/usr/bin/env python
"""Bench harness for combine_stacks.py: generate inputs, run one target, measure.

  1. INPUTS: synthetic substacks from gen_substacks.py (cached, keyed by their
     parameters, shared by every target), or real .rle substacks via --inputs-from.
  2. RUN one target on them in a scratch dir of symlinks, as Nextflow stages them,
     optionally under a profiler. Each run is its own child process, forked by
     a small launcher; peak RSS comes from the launcher's os.wait4 on it.
  3. REPORT wall time, peak RSS, the target's phase table (if it prints one),
     output size, and a per-slice fingerprint of the output, to
     results/<tag>/summary.json (stdout/stderr in results/<tag>/log.txt).

Targets (--target):
  worktree        combine_stacks.py in this checkout
  git:<ref>       combine_stacks.py at <ref>, e.g. git:origin/master, git:origin/AIOD-312
  path:<file>     any copy of the script
  zarr:<format>   zarr_variant.py (worktree combine_stacks.py reading Zarr substacks),
                  e.g. zarr:zarr-dir-zstd-c1; see gen_substacks.py for formats

The CLI dialect is detected from the script's --help: master's (--preprocess-config,
--image-id, --param-hash) or the legacy AIOD-312 one (--slab-size).

Run inside the combine_stacks conda env (envs/conda_combine_stacks.yml).

Examples:
    python bench_combine.py --preset tiny
    python bench_combine.py --preset small --target git:origin/master
    python bench_combine.py --preset small --mask-type binary --overlap 0.1
    python bench_combine.py --preset small --postprocess --output-format tiff
    python bench_combine.py --preset small --target git:origin/AIOD-312 --slab-size 16
    python bench_combine.py --preset small --profiler memray
    python bench_combine.py --inputs-from /path/to/aiod_cache/masks --target worktree
"""

import argparse
import hashlib
import json
import os
import pickle
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import aiod_utils.rle as aiod_rle
import numpy as np
from aiod_utils.io import extract_idxs_from_fname, get_combined_mask_name

import gen_substacks as gen

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parents[1]
SCRIPT_RELPATH = "modules/models/resources/usr/bin/combine_stacks.py"
COMBINE = REPO_ROOT / SCRIPT_RELPATH
ZARR_VARIANT = HARNESS_DIR / "zarr_variant.py"
RESULTS_DIR = HARNESS_DIR / "results"

# (shape D H W, grid ND NH NW). large & huge are for the cluster.
PRESETS = {
    "tiny": ((16, 128, 128), (2, 2, 2)),
    "small": ((64, 512, 512), (4, 2, 2)),
    "medium": ((256, 1024, 1024), (8, 2, 2)),
    "large": ((512, 2048, 2048), (8, 4, 4)),
    "huge": ((1024, 3072, 3072), (16, 4, 4)),
}

PHASE_HEADER = "--- combine_stacks phase timings ---"
PHASE_ROW = re.compile(r"^\s+(.+?)\s+([\d.]+)s\s+[\d.]+m\s+[\d.]+%$")


# ---------------------------------------------------------------- targets


def resolve_target(spec, cache_dir):
    """Return (script path, label) for a --target spec."""
    if spec == "worktree":
        return COMBINE, "worktree"
    kind, _, value = spec.partition(":")
    if kind == "path":
        return Path(value).resolve(), f"path-{Path(value).stem}"
    if kind == "zarr":
        gen.parse_zarr_format(value)
        return ZARR_VARIANT, f"zarr-{value}"
    if kind == "git":
        sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", value],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        out = Path(cache_dir) / "targets" / f"combine_stacks_{sha}.py"
        if not out.exists():
            src = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "show", f"{value}:{SCRIPT_RELPATH}"],
                check=True, capture_output=True, text=True,
            ).stdout
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(src)
        return out, f"git-{value.replace('/', '-')}-{sha}"
    raise ValueError(f"unknown target '{spec}'")


_DIALECTS = {}


def detect_dialect(script):
    """'master' or 'legacy', from the script's --help."""
    script = Path(script)
    if script not in _DIALECTS:
        res = subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True, text=True
        )
        if res.returncode != 0:
            raise RuntimeError(f"{script} --help failed:\n{res.stderr}")
        if "--slab-size" in res.stdout:
            _DIALECTS[script] = "legacy"
        elif "--preprocess-config" in res.stdout:
            _DIALECTS[script] = "master"
        else:
            raise RuntimeError(f"unrecognised CLI dialect for {script}")
    return _DIALECTS[script]


def build_cmd(script, dialect, masks, image_size, overlap, opts, result_dir, workdir):
    D, H, W = image_size
    core = [
        str(script),
        "--mask-fname", gen.MASK_FNAME,
        "--output-dir", ".",
        "--masks", *masks,
        "--model", opts["model"],
        "--image-size", str(D), str(H), str(W),
        "--overlap", *[str(overlap)] * 3,
        "--iou-threshold", str(opts["iou_threshold"]),
        "--output-format", opts["output_format"],
        "--output-mask-type", opts["output_mask_type"],
    ]
    if dialect == "master":
        prep = Path(workdir) / "preprocess_config.json"
        prep.write_text("[]")
        core += [
            "--preprocess-config", prep.name,
            "--image-id", gen.IMAGE_ID,
            "--param-hash", gen.RUN_HASH,
        ]
    elif dialect == "legacy":
        core += ["--slab-size", str(opts["slab_size"])]
    if opts["postprocess"]:
        core.append("--postprocess")

    profiler = opts["profiler"]
    if profiler == "memray":
        out = result_dir / "memray.bin"
        return ["memray", "run", "--force", "-o", str(out), *core], out
    if profiler == "pyspy":
        out = result_dir / "pyspy.svg"
        return ["py-spy", "record", "-o", str(out), "-f", "flamegraph",
                "--", sys.executable, *core], out
    if profiler == "cprofile":
        out = result_dir / "combine.pstats"
        return [sys.executable, "-m", "cProfile", "-o", str(out), *core], out
    return [sys.executable, *core], None


# ---------------------------------------------------------------- inputs


def real_inputs(inputs_from):
    """Image size / overlap / mask type of a directory of real .rle substacks."""
    paths = sorted(Path(inputs_from).glob("*.rle"))
    if not paths:
        raise SystemExit(f"no .rle files in {inputs_from}")
    idxs = [extract_idxs_from_fname(p.name) for p in paths]
    D = max(i[5] for i in idxs)
    H = max(i[3] for i in idxs)
    W = max(i[1] for i in idxs)
    covered = sum((i[1] - i[0]) * (i[3] - i[2]) * (i[5] - i[4]) for i in idxs)
    overlap = 1.0 if covered > D * H * W else 0.0
    meta = aiod_rle.load_encoding(paths[0])[-1]
    mask_type = meta.get("metadata", {}).get("mask_type")
    return paths, (D, H, W), overlap, mask_type


# ---------------------------------------------------------------- measuring


def norm_rss(raw):
    """ru_maxrss in bytes (Linux reports kB, macOS bytes)."""
    return raw * 1024 if platform.system() == "Linux" else raw


# Runs argv[2:] as its own child and writes that child's ru_maxrss to argv[1].
# On Linux ru_maxrss survives exec: a process forked from this one starts at this
# one's RSS, which in-process generation and fingerprinting (sweep.py) inflate.
# Forking the target from a fresh, small interpreter keeps that floor at ~10 MB.
LAUNCHER = """
import os, signal, sys
pid = os.fork()
if pid == 0:
    os.execvp(sys.argv[2], sys.argv[2:])
_, status, rusage = os.wait4(pid, 0)
with open(sys.argv[1], "w") as f:
    f.write(str(rusage.ru_maxrss))
code = os.waitstatus_to_exitcode(status)
if code < 0:
    signal.signal(-code, signal.SIG_DFL)
    os.kill(os.getpid(), -code)
sys.exit(code)
"""


def run_child(cmd, cwd, env, log_path):
    """Run cmd to completion; returns (exit code, wall s, peak RSS bytes)."""
    with tempfile.NamedTemporaryFile("r", suffix=".maxrss") as rss_file:
        with open(log_path, "w") as log:
            log.write("cmd: " + " ".join(cmd) + "\n\n")
            log.flush()
            t0 = time.perf_counter()
            launch = [sys.executable, "-c", LAUNCHER, rss_file.name, *cmd]
            returncode = subprocess.run(
                launch, cwd=cwd, env=env, stdout=log, stderr=log
            ).returncode
            wall = time.perf_counter() - t0
        # Empty if the launcher itself died
        raw = rss_file.read().strip() or "0"
    return returncode, wall, norm_rss(int(raw))


def parse_phases(log_text):
    """{phase: seconds} from a combine_stacks phase table, or {} if absent."""
    phases, in_table = {}, False
    for line in log_text.splitlines():
        if line.strip() == PHASE_HEADER:
            in_table = True
            continue
        if in_table:
            m = PHASE_ROW.match(line)
            if not m:
                break
            phases[m.group(1)] = float(m.group(2))
    return phases


def _digest(arr):
    h = hashlib.sha1()
    h.update(f"{arr.dtype.str}{arr.shape}".encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def fingerprint(path, output_format):
    """Per-slice digest of an output, read one slice at a time."""
    slice_digests = []
    out = {}
    if output_format == "rle":
        rle = aiod_rle.load_encoding(path)
        meta = rle[-1]
        mask_type = meta.get("metadata", {}).get("mask_type")
        arr = None
        for i in range(len(rle) - 1):
            arr, _ = aiod_rle.decode([rle[i], meta], mask_type=mask_type)
            slice_digests.append(_digest(arr))
        n = len(rle) - 1
        out["dtype"] = str(arr.dtype)
        out["shape"] = [n, *arr.shape] if n > 1 else list(arr.shape)
        out["metadata"] = meta.get("metadata", {})
        out["encoding_digest"] = hashlib.sha1(pickle.dumps(rle)).hexdigest()
    else:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            for page in tif.pages:
                arr = page.asarray()
                slice_digests.append(_digest(arr))
            out["dtype"] = str(tif.pages[0].dtype)
            out["shape"] = list(tif.series[0].shape)
            out["imagej_metadata"] = {
                k: v for k, v in (tif.imagej_metadata or {}).items() if k != "ImageJ"
            }
    out["slice_digests"] = slice_digests
    out["digest"] = hashlib.sha1(
        (out["dtype"] + str(out["shape"]) + "".join(slice_digests)).encode()
    ).hexdigest()
    return out


# ---------------------------------------------------------------- one run


DEFAULTS = {
    "target": "worktree",
    "preset": None,
    "shape": None,
    "grid": None,
    "overlap": 0.0,
    "mask_type": "instance",
    "block": 16,
    "block_z": None,
    "fg_frac": 0.4,
    "n_ids": 1000,
    "seed": 0,
    "order": "gen",
    "inputs_from": None,
    "model": "empanada",
    "iou_threshold": 0.8,
    "output_format": "rle",
    "output_mask_type": "auto",
    "postprocess": False,
    "profiler": "none",
    "slab_size": 64,
    "store": "disk",
    "scratch": os.environ.get("TMPDIR") or tempfile.gettempdir(),
    "cache_dir": str(gen.DEFAULT_CACHE),
    "keep": False,
    "tag": None,
    "quiet": False,
}


def run_bench(**cfg):
    """Run one target on one scenario; returns the summary dict."""
    opts = {**DEFAULTS, **cfg}
    say = (lambda *a: None) if opts["quiet"] else print
    script, label = resolve_target(opts["target"], opts["cache_dir"])
    is_zarr = opts["target"].startswith("zarr:")
    dialect = "master" if is_zarr else detect_dialect(script)

    if opts["inputs_from"]:
        if is_zarr:
            raise SystemExit("--inputs-from is RLE only")
        paths, shape, overlap, mask_type = real_inputs(opts["inputs_from"])
        grid, manifest = None, None
        input_fmt = "rle"
    else:
        if opts["preset"]:
            shape, grid = PRESETS[opts["preset"]]
        else:
            shape, grid = tuple(opts["shape"]), tuple(opts["grid"])
        overlap, mask_type = opts["overlap"], opts["mask_type"]
        input_fmt = opts["target"].split(":", 1)[1] if is_zarr else "rle"
        manifest = gen.generate(
            shape, grid, overlap=overlap, mask_type=mask_type,
            block=opts["block"], block_z=opts["block_z"], fg_frac=opts["fg_frac"],
            n_ids=opts["n_ids"], seed=opts["seed"], formats=(input_fmt,),
            cache_dir=opts["cache_dir"], quiet=opts["quiet"],
        )
        paths = gen.substack_paths(manifest, input_fmt)
    if opts["order"] == "shuffle":
        paths = list(paths)
        random.Random(opts["seed"]).shuffle(paths)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    grid_s = "x".join(map(str, grid)) if grid else "real"
    tag = opts["tag"] or (
        f"{opts['preset'] or 'custom'}_{mask_type}_g{grid_s}_ov{overlap}"
        f"_{opts['output_format']}{'_pp' if opts['postprocess'] else ''}_{label}_{stamp}"
    )
    result_dir = RESULTS_DIR / tag
    result_dir.mkdir(parents=True, exist_ok=True)

    workdir = Path(tempfile.mkdtemp(prefix="combine_bench_", dir=opts["scratch"])).resolve()
    try:
        masks = []
        for p in paths:
            (workdir / Path(p).name).symlink_to(Path(p).resolve())
            masks.append(Path(p).name)
        cmd, artifact = build_cmd(
            script, dialect, masks, shape, overlap, opts, result_dir, workdir
        )
        if shutil.which(cmd[0]) is None:
            raise SystemExit(f"'{cmd[0]}' not found on PATH")
        env = dict(os.environ, TMPDIR=str(workdir), COMBINE_STACKS_STORE=opts["store"])
        if is_zarr:
            env["COMBINE_STACKS_SCRIPT"] = str(COMBINE)
        say(f"[{label}] shape={tuple(shape)} grid={grid} substacks={len(masks)} -> {workdir}")
        log_path = result_dir / "log.txt"
        rc, wall, peak = run_child(cmd, workdir, env, log_path)
        log_text = log_path.read_text()

        out_path = workdir / get_combined_mask_name(gen.MASK_FNAME, opts["output_format"])
        fp = None
        if rc == 0 and out_path.exists():
            fp = fingerprint(out_path, opts["output_format"])
        input_stats = (
            manifest["formats"][input_fmt]
            if manifest
            else {"bytes_total": sum(Path(p).stat().st_size for p in paths),
                  "files_total": len(paths)}
        )
        summary = {
            "tag": tag,
            "target": opts["target"],
            "target_label": label,
            "dialect": dialect,
            "preset": opts["preset"],
            "shape": list(shape),
            "grid": list(grid) if grid else None,
            "substacks": len(masks),
            "overlap": overlap,
            "mask_type": mask_type,
            "block": opts["block"],
            "order": opts["order"],
            "inputs_from": opts["inputs_from"],
            "input_format": input_fmt,
            "input_stats": input_stats,
            "model": opts["model"],
            "output_format": opts["output_format"],
            "output_mask_type": opts["output_mask_type"],
            "postprocess": opts["postprocess"],
            "profiler": opts["profiler"],
            "slab_size": opts["slab_size"] if dialect == "legacy" else None,
            "returncode": rc,
            "wall_seconds": round(wall, 3),
            "peak_rss_gb": round(peak / 1e9, 4),
            "phases": parse_phases(log_text),
            "output_mb": round(out_path.stat().st_size / 1e6, 3) if out_path.exists() else None,
            "fingerprint": fp,
            "host": platform.node(),
        }
        (result_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if rc != 0:
            say(f"[{label}] FAILED (exit {rc}); last lines of {log_path}:")
            say("\n".join(log_text.splitlines()[-15:]))
        else:
            say(
                f"[{label}] wall={wall:.2f}s peak_rss={peak / 1e9:.3f}GB "
                f"digest={fp['digest'][:12] if fp else None} -> {result_dir}"
            )
        if opts["profiler"] == "memray" and artifact and artifact.exists():
            html = result_dir / "memray_flamegraph.html"
            subprocess.run(["memray", "flamegraph", "-f", "-o", str(html), str(artifact)],
                           check=False, capture_output=True)
            say(f"open {html}")
        elif opts["profiler"] in ("pyspy", "cprofile") and artifact and artifact.exists():
            say(f"profile: {artifact}")
        return summary
    finally:
        if opts["keep"]:
            say(f"kept scratch: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target", default="worktree")
    p.add_argument("--preset", choices=list(PRESETS))
    p.add_argument("--shape", nargs=3, type=int, metavar=("D", "H", "W"))
    p.add_argument("--grid", nargs=3, type=int, metavar=("ND", "NH", "NW"))
    p.add_argument("--inputs-from", help="Directory of real .rle substacks")
    p.add_argument("--mask-type", choices=["binary", "instance"], default="instance")
    p.add_argument("--overlap", type=float, default=0.0, help="Overlap fraction")
    p.add_argument("--block", type=int, default=16, help="XY block size (smaller = more runs)")
    p.add_argument("--block-z", type=int, default=None)
    p.add_argument("--fg-frac", type=float, default=0.4)
    p.add_argument("--n-ids", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--order", choices=["gen", "shuffle"], default="gen",
                   help="--masks order: generation order, or shuffled as Nextflow may")
    p.add_argument("--model", default="empanada", help="empanada|sam|... (postprocess route)")
    p.add_argument("--iou-threshold", type=float, default=0.8)
    p.add_argument("--output-format", choices=["rle", "tiff"], default="rle")
    p.add_argument("--output-mask-type", choices=["auto", "binary", "instance"], default="auto")
    p.add_argument("--postprocess", action="store_true")
    p.add_argument("--profiler", choices=["none", "memray", "pyspy", "cprofile"], default="none")
    p.add_argument("--slab-size", type=int, default=64, help="Legacy dialect only")
    p.add_argument("--store", choices=["disk", "ram"], default="disk", help="Legacy dialect only")
    p.add_argument("--scratch", default=DEFAULTS["scratch"],
                   help="Where the run's symlinks and output live (node-local on a cluster)")
    p.add_argument("--cache-dir", default=DEFAULTS["cache_dir"], help="Generated inputs cache")
    p.add_argument("--keep", action="store_true", help="Do not scrap the scratch dir")
    args = p.parse_args()
    if not (args.inputs_from or args.preset or (args.shape and args.grid)):
        p.error("give --preset, --shape and --grid, or --inputs-from")
    summary = run_bench(**vars(args))
    return summary["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
