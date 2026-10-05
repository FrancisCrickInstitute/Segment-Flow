#!/usr/bin/env python
"""Run a matrix of targets x scenarios through bench_combine.run_bench.

Writes results/sweep_<stamp>.csv and results/sweep_<stamp>.md: wall time, peak
RSS, phase split, and whether each output's fingerprint matches the reference
target's (the first --targets entry unless --ref is given).

Default scenarios, per preset:
  {instance, binary} x {z-only, xy grid} x {overlap 0, 0.1} x {3D, 2D} x {rle, tiff}
  + instance xy overlap-0 with shuffled --masks order (label offsets follow it)
  + postprocess: empanada (connect_components) on binary, sam (connect_sam) on instance

Examples:
    python sweep.py --presets small
    python sweep.py --presets small medium --targets git:origin/master worktree git:origin/AIOD-312
    python sweep.py --presets small --filter binary --filter tiff
    python sweep.py --presets medium --exclude pp-sam
    python sweep.py --presets small --targets worktree zarr:zarr-dir-zstd-c1 --suite zarr
"""

import argparse
import csv
import json
from datetime import datetime

import bench_combine as bench

ZARR_FORMATS = [
    "zarr-dir-zstd-c1",
    "zarr-dir-zstd-c8",
    "zarr-dir-lz4-c1",
    "zarr-zip-zstd-c1",
    "zarr-zip-lz4-c8",
]


def default_scenarios(preset):
    shape, grid = bench.PRESETS[preset]
    D, H, W = shape
    nz, ny, nx = grid
    grids = {
        "3d": {"z": (nz, 1, 1), "xy": (nz, ny, nx)},
        "2d": {"z": (1, 1, 1), "xy": (1, ny, nx)},
    }
    shapes = {"3d": (D, H, W), "2d": (1, H, W)}
    out = []
    for mask_type in ("instance", "binary"):
        for tiling in ("z", "xy"):
            for overlap in (0.0, 0.1):
                for dim in ("3d", "2d"):
                    for fmt in ("rle", "tiff"):
                        out.append({
                            "name": f"{preset}-{mask_type}-{tiling}-ov{overlap}-{dim}-{fmt}",
                            "shape": shapes[dim], "grid": grids[dim][tiling],
                            "mask_type": mask_type, "overlap": overlap,
                            "output_format": fmt,
                        })
    out.append({
        "name": f"{preset}-instance-xy-ov0.0-3d-rle-shuffled",
        "shape": shapes["3d"], "grid": grids["3d"]["xy"], "mask_type": "instance",
        "overlap": 0.0, "output_format": "rle", "order": "shuffle",
    })
    for model, mask_type in (("empanada", "binary"), ("sam", "instance")):
        out.append({
            "name": f"{preset}-{mask_type}-xy-ov0.0-3d-rle-pp-{model}",
            "shape": shapes["3d"], "grid": grids["3d"]["xy"], "mask_type": mask_type,
            "overlap": 0.0, "output_format": "rle", "postprocess": True, "model": model,
        })
    return out


def zarr_scenarios(preset):
    """Step-3 evaluation: run density, tiling and overlap, RLE output."""
    shape, grid = bench.PRESETS[preset]
    nz, ny, nx = grid
    out = []
    for mask_type in ("instance", "binary"):
        for block in (4, 16, 64):
            for tiling, g in (("z", (nz, 1, 1)), ("xy", grid)):
                for overlap in (0.0, 0.1):
                    out.append({
                        "name": f"{preset}-{mask_type}-b{block}-{tiling}-ov{overlap}",
                        "shape": shape, "grid": g, "mask_type": mask_type,
                        "overlap": overlap, "block": block, "output_format": "rle",
                    })
    return out


def flat_row(scenario, summary, ref_digest):
    fp = summary.get("fingerprint") or {}
    digest = fp.get("digest")
    return {
        "scenario": scenario["name"],
        "target": summary["target"],
        "target_label": summary["target_label"],
        "returncode": summary["returncode"],
        "wall_seconds": summary["wall_seconds"],
        "peak_rss_gb": summary["peak_rss_gb"],
        "output_mb": summary["output_mb"],
        "input_mb": round(summary["input_stats"]["bytes_total"] / 1e6, 3),
        "input_files": summary["input_stats"]["files_total"],
        "input_write_s_mean": summary["input_stats"].get("write_s_mean"),
        "input_write_s_max": summary["input_stats"].get("write_s_max"),
        "digest": digest,
        "matches_ref": (digest == ref_digest) if (digest and ref_digest) else None,
        "phases": json.dumps(summary["phases"]),
        "tag": summary["tag"],
    }


def write_markdown(path, rows, targets, ref):
    by_scenario = {}
    for r in rows:
        by_scenario.setdefault(r["scenario"], {})[r["target"]] = r
    head = "| scenario | " + " | ".join(targets) + " |"
    sep = "|---|" + "---|" * len(targets)
    lines = [
        f"Reference: `{ref}`. Cells: wall s / peak RSS GB / output matches reference.",
        "",
        head,
        sep,
    ]
    for name, cells in by_scenario.items():
        parts = []
        for t in targets:
            r = cells.get(t)
            if r is None:
                parts.append("")
            elif r["returncode"] != 0:
                parts.append(f"FAIL ({r['returncode']})")
            else:
                mark = {True: "=", False: "**DIFF**", None: "?"}[r["matches_ref"]]
                parts.append(f"{r['wall_seconds']:.2f} / {r['peak_rss_gb']:.3f} / {mark}")
        lines.append(f"| {name} | " + " | ".join(parts) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--presets", nargs="+", default=["small"], choices=list(bench.PRESETS))
    p.add_argument("--targets", nargs="+", default=["git:origin/master", "worktree"])
    p.add_argument("--ref", help="Target whose output the others are compared with")
    p.add_argument("--suite", choices=["default", "zarr"], default="default")
    p.add_argument("--zarr-formats", nargs="+", default=ZARR_FORMATS,
                   help="With --suite zarr, adds zarr:<format> targets for each")
    p.add_argument("--filter", action="append", default=[],
                   help="Only scenarios whose name contains every given substring")
    p.add_argument("--exclude", action="append", default=[],
                   help="Skip scenarios whose name contains any given substring")
    p.add_argument("--scratch", default=bench.DEFAULTS["scratch"])
    p.add_argument("--cache-dir", default=bench.DEFAULTS["cache_dir"])
    p.add_argument("--profiler", default="none", choices=["none", "memray", "pyspy", "cprofile"])
    args = p.parse_args()

    targets = list(args.targets)
    if args.suite == "zarr":
        targets += [f"zarr:{f}" for f in args.zarr_formats if f"zarr:{f}" not in targets]
    ref = args.ref or targets[0]
    if ref not in targets:
        targets.insert(0, ref)
    make = zarr_scenarios if args.suite == "zarr" else default_scenarios
    scenarios = [s for preset in args.presets for s in make(preset)]
    scenarios = [s for s in scenarios if all(f in s["name"] for f in args.filter)]
    scenarios = [s for s in scenarios if not any(x in s["name"] for x in args.exclude)]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bench.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = bench.RESULTS_DIR / f"sweep_{stamp}.csv"
    md_path = bench.RESULTS_DIR / f"sweep_{stamp}.md"
    rows = []
    print(f"{len(scenarios)} scenarios x {len(targets)} targets, reference {ref}")
    for i, sc in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {sc['name']}")
        cfg = {k: v for k, v in sc.items() if k != "name"}
        ordered = [ref] + [t for t in targets if t != ref]
        ref_digest = None
        for t in ordered:
            summary = bench.run_bench(
                target=t, scratch=args.scratch, cache_dir=args.cache_dir,
                profiler=args.profiler, tag=f"sweep_{stamp}/{sc['name']}/{t.replace(':', '-').replace('/', '-')}",
                quiet=True, **cfg,
            )
            fp = summary.get("fingerprint") or {}
            if t == ref:
                ref_digest = fp.get("digest")
            row = flat_row(sc, summary, ref_digest)
            rows.append(row)
            mark = {True: "=", False: "DIFF", None: "?"}[row["matches_ref"]]
            status = "ok" if row["returncode"] == 0 else f"FAIL({row['returncode']})"
            print(f"  {t:<28} {status:<8} wall={row['wall_seconds']:>8.2f}s "
                  f"rss={row['peak_rss_gb']:.3f}GB {mark}")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        write_markdown(md_path, rows, targets, ref)
    print(f"\n{csv_path}\n{md_path}")
    n_diff = sum(r["matches_ref"] is False for r in rows)
    n_fail = sum(r["returncode"] != 0 for r in rows)
    print(f"{n_diff} digest mismatches, {n_fail} failures")
    return 1 if (n_diff or n_fail) else 0


if __name__ == "__main__":
    raise SystemExit(main())
