# combine_stacks bench harness (AIOD-312)

Runs `combine_stacks.py` on its own, with no images, models or Nextflow. It
generates substacks, runs one or more versions of the script on them, and
reports wall time, peak RSS, the phase split and a fingerprint of the output.
The fingerprint shows whether two versions produce the same result.

| File | Purpose |
|------|---------|
| `gen_substacks.py` | Synthetic substacks (RLE and/or Zarr), cached by their parameters |
| `bench_combine.py` | One target on one scenario; writes `results/<tag>/{summary.json,log.txt}` |
| `sweep.py` | Targets × scenarios matrix; writes `results/sweep_<stamp>.{csv,md}` |
| `zarr_variant.py` | Worktree `combine_stacks.py` reading Zarr substacks instead of `.rle` |
| `submit.sbatch` | Runs a sweep (or one bench) on the cluster |

The equivalence unit tests are in `tests/test_combine_stacks.py` (pytest).

## Setup

Run inside the combine_stacks conda env:

    mamba env create -f ../../modules/models/envs/conda_combine_stacks.yml   # first time
    mamba activate combine_conda

A Nextflow-built copy works too, e.g. `~/.nextflow/aiod/conda/env-8a070ce1...`.
The env yml does not list `zarr`, `memray`, `py-spy` or `pytest`. The Zarr
targets need `zarr` 3; the profilers need their tools on `PATH`.

## Inputs

`gen_substacks.py` lays substacks out with `aiod_utils.stacks.generate_stack_indices`,
as `create_splits.py` does. The `--grid` is used as given, without
`calc_num_stacks`' size caps. The content is blocky, and each block's id and
foreground come from a hash of its global block index. Neighbours therefore agree
in their overlaps, and no global dense volume is built. Each substack is relabelled
and dtype-reduced as `utils.save_masks` does. Every requested format is written
from the same array, and the generator records the write time, bytes and file count.

Knobs: `--shape D H W`, `--grid ND NH NW`, `--overlap` (fraction), `--mask-type`,
`--block` / `--block-z` (block size; smaller means more runs), `--fg-frac`,
`--n-ids`, `--seed`. Outputs are cached in `cache/<params-hash>/` (override with
`COMBINE_BENCH_CACHE` or `--cache-dir`) and reused by every target.

`--inputs-from DIR` uses real `.rle` substacks instead, e.g. copied from a
cluster run's mask cache. The image size comes from the largest substack
indices, and overlap is assumed if the substacks cover more voxels than the image.

## Targets

| `--target` | Runs |
|------------|------|
| `worktree` (default) | `combine_stacks.py` in this checkout |
| `git:<ref>` | the script at `<ref>`, extracted with `git show`, e.g. `git:origin/master`, `git:origin/AIOD-312` |
| `path:<file>` | any copy of the script |
| `zarr:<format>` | `zarr_variant.py`, e.g. `zarr:zarr-dir-zstd-c1` |

The CLI dialect is detected from `--help`:
- **master:** adds `--preprocess-config` (a dummy `[]`), `--image-id` and `--param-hash`;
- **legacy AIOD-312:** adds `--slab-size`, and `--store` sets `COMBINE_STACKS_STORE`.

Zarr formats are `zarr-<dir|zip>-<zstd|lz4>-c<chunk depth>`. `zip` writes one file
per substack, and `lz4` is Blosc-LZ4.

## Measurements

- **Peak RSS** is `os.wait4` on the one child process, normalised to bytes. Under a
  profiler, this is the profiler process, so it is not comparable with plain runs.
  The child is forked by a small launcher, not by the harness: on Linux
  `ru_maxrss` survives exec, so a child forked from the harness never reports
  less than the harness's own RSS. Generating inputs and fingerprinting outputs
  in-process (as `sweep.py` does) inflate that well above the target's real peak.
  Results from before 2026-10-02 afternoon (jobs 59062343/4) have this floor.
- **Wall** is the whole child, including the interpreter start-up and imports
  (~1.7 s, ~0.32 GB here).
- **Phases** come from the target's own `--- combine_stacks phase timings ---` table.
  Phases are self-time and nested phases are excluded, so rows sum to the total.
  Only AIOD-312 prints this table now; for the slice-wise script, use a profiler.
  The phase splits in the results below came from a since-removed instrumented
  version of the slice-wise script.
- **Fingerprint** decodes the output one slice at a time (TIFF: one page at a time)
  and hashes each slice's dtype, shape and bytes. It also records the shape and
  metadata, and for RLE output a hash of the whole encoding. `sweep.py` compares
  `digest` against the reference target.

Profilers: `--profiler memray` (writes `memray.bin` and a flamegraph HTML),
`pyspy`, or `cprofile`.

## Quick start

    python bench_combine.py --preset tiny
    python bench_combine.py --preset small --target git:origin/master
    python bench_combine.py --preset small --mask-type binary --overlap 0.1 --output-format tiff
    python bench_combine.py --preset small --order shuffle      # --masks order as Nextflow may give it
    python sweep.py --presets small                              # master vs worktree
    python sweep.py --presets medium --targets git:origin/master worktree git:origin/AIOD-312 --exclude pp-sam
    python sweep.py --presets small --targets worktree --suite zarr

Presets (shape D×H×W, grid): tiny 16×128×128 (2,2,2), small 64×512² (4,2,2),
medium 256×1024² (8,2,2), large 512×2048² (8,4,4), huge 1024×3072² (16,4,4).
Use large and huge on the cluster.

Default sweep scenarios:
- {instance, binary} × {z-only, xy grid} × {overlap 0, 0.1} × {3D, 2D} × {rle, tiff}.
  The 2D z-only case is a single substack.
- A shuffled `--masks` order.
- Postprocessing for empanada (`connect_components`, binary) and sam
  (`connect_sam`, instance). The sam case takes ~2 min at small with master's
  algorithm, so exclude `pp-sam` at medium and above.

## Cluster

    sbatch submit.sbatch                                     # large, master vs worktree
    sbatch submit.sbatch --presets huge --targets worktree --exclude pp-sam
    BENCH="bench_combine.py --preset large --profiler memray" sbatch submit.sbatch

Submit from this directory. Keep `--mem` above master's peak if master is a target.

## Results (local, Apple M-series, 24 GB)

### Terms

- **z-only grid:** substacks are split along z only. Each substack is a run of
  whole slices (full H×W), e.g. grid (8, 1, 1).
- **xy grid:** substacks are also split in H and/or W. Each substack covers part
  of each of its slices, e.g. grid (8, 2, 2). Only xy grids with overlap 0 offset
  labels.
- **overlap:** fraction of the substack size shared with each neighbour. With
  overlap > 0, the overlapping region is summed.
- **block (b4 / b16 / b64):** side length in voxels of the synthetic blocks.
  Smaller blocks mean more runs per slice and more instances, so larger RLE and
  slower encode and decode. The default is 16.
- **Zarr format:** `dir` is one directory per substack and `zip` one file per
  substack; `zstd` / `lz4` are the compressors; `c1` / `c8` are the slices per chunk.
- **Wall** and **peak RSS** are for the whole process. Both include Python
  start-up and imports: about 1.7 s and 0.32 GB.

At medium (256×1024×1024 = 268 M voxels), one dense uint16 copy of the volume
is 0.54 GB.

### Pipeline: where the heavy data is, per strategy

```
  INPUT: N substack .rle files on disk (written by runModel, compressed)
  ═══════════════════════════════════════════════════════════════════════════════════
  MASTER (dense)
  [1] combine   for each substack, in --masks order:
                  load the whole .rle  → decode the WHOLE substack (binary: + uint8 copy)
                  → write into all_masks: offset labels (xy, overlap 0), add (overlap > 0)
                  RAM: all_masks = WHOLE VOLUME, uint16 (2 B/vox)  + 1 decoded substack
  [2] dtype     reduce_dtype(all_masks)             RAM: + uint8 volume copy (if max < 256)
  [3] postproc  empanada: connect_components        RAM: + int32 labels (4 B/vox) + dask
                  (dask label, then .compute() into RAM)
                sam:      connect_sam               RAM: slice pairs in place,
                  (relabel_sequential at the end)        + a volume copy at the end
  [4] write     rle:  encode the WHOLE volume
                  binary   vectorised: fast,  RAM: + ~1 B/vox temporaries
                  instance per slice, per instance: slow, little extra RAM
                  output RLE list held in RAM, then pickled
                tiff: imwrite the volume            (binary: + uint8 0/255 copy)
  ───────────────────────────────────────────────────────────────────────────────────
  AIOD-312 (scratch Zarr store)
  [1] combine   decode each substack → write into a Zarr store on scratch disk
                  RAM: 1 decoded substack   DISK: whole volume, compressed (write + read back)
  [2] dtype     max scan over the store, --slab-size slices at a time
  [3] postproc  connect_components: dask reads the store → labels → a 2nd Zarr store
                  RAM: dask chunks only   cost: compress, write, read (8× slower here)
                connect_sam: still dense, as master
  [4] write     encode / TIFF pages slab by slab from the store   RAM: 1 slab
  ───────────────────────────────────────────────────────────────────────────────────
  SLICE-WISE (this branch)
  [1+4] for z = 0 … D-1:
          open the substacks starting at z (load their .rle)   RAM: RLE of 1 z-layer
          decode ONE slice of each substack covering z
          combine into one plane (offset / add)                RAM: 1 plane, H×W uint16
          encode the plane → append to output RLE list         RAM: output RLE grows
        tiff: pages written as planes are made
          instance dtype: from RLE labels (overlap 0), or an extra decode pass (overlap > 0)
  [2]   no volume to reduce
  [3]   postproc: planes collected into a DENSE VOLUME → same as master [3] + [4]
                  (memory unchanged: the streaming postprocess is not done yet)
  ───────────────────────────────────────────────────────────────────────────────────
  ZARR SUBSTACKS (step 3, harness only)
  as slice-wise, but [1] reads one chunk of a .zarr substack instead of decoding RLE
    → decode time ≈ 0; encode [4] unchanged; runModel writes .zarr instead of RLE-encoding
```

What each strategy changes, compared with master:

| stage | AIOD-312 | slice-wise | Zarr substacks |
|---|---|---|---|
| [1] combine: RAM | 1 decoded substack + scratch store | RLE of 1 z-layer + 1 plane | as slice-wise |
| [1] combine: time | + store write and read | same | decode ≈ 0 |
| [2] dtype | slab scan of the store | none, or from labels | as slice-wise |
| [3] postprocess: RAM | empanada: dask chunks; sam: dense | dense (unchanged) | not supported |
| [3] postprocess: time | empanada 8× slower | same | – |
| [4] write: RAM | 1 slab | 1 plane + output RLE | as slice-wise |
| [4] write: time | + store read | same | same |
| runModel side | unchanged | unchanged | Zarr write replaces RLE encode |

Master's memory per step at medium, measured in isolation on a dense uint16 volume
(0.54 GB) with the default content (block 16):

| step | extra RAM above the volume (GB) | time (s) |
|---|---|---|
| encode, binary | 0.50 | 0.9 |
| encode, instance | 0.00 | 63.8 |
| connect_components (binary input) | 3.22 | 8.6 |

### Step 1: slice-wise `combine_stacks.py` vs master

Every scenario gives output identical to master's. AIOD-312 differs from master
only on instance + xy + overlap > 0. Sources: `results/sweep_20261001-122444`
(small), `results/sweep_20261001-123419` and `results/sweep_20261001-130245`
(medium).

Medium (256×1024×1024), overlap 0 unless stated:

| scenario | wall, master (s) | wall, slice-wise (s) | wall, AIOD-312 (s) | peak RSS, master (GB) | peak RSS, slice-wise (GB) | peak RSS, AIOD-312 (GB) |
|---|---|---|---|---|---|---|
| instance, z-only, rle | 79.5 | 79.1 | 82.6 | 1.14 | 0.74 | 0.95 |
| instance, xy, rle | 15.8 | 15.5 | 17.3 | 1.24 | 0.78 | 1.06 |
| instance, z-only, tiff | 14.9 | 14.8 | 15.6 | 1.24 | 0.51 | 1.16 |
| instance, z-only, tiff, overlap 0.1 | 15.6 | 28.2 | 17.7 | 1.27 | 0.52 | 1.17 |
| binary, z-only, rle | 4.4 | 4.1 | 4.9 | 1.60 | 0.45 | 0.71 |
| binary, xy, rle, overlap 0.1 | 4.8 | 4.4 | 8.4 | 1.50 | 0.45 | 0.89 |
| binary, xy, tiff | 4.2 | 3.8 | 4.1 | 1.25 | 0.39 | 0.84 |
| binary, xy, rle, empanada postprocess | 12.8 | 11.8 | 97.9 | 5.16 | 5.23 | 2.47 |

All peak RSS figures include ~0.32 GB of interpreter baseline.

**Why binary is faster than instance, but uses more memory on master.** The
dense volume is the same 0.54 GB uint16 for both; the difference is the encode,
which uses a different algorithm for each mask type:
- **Binary** is one vectorised numpy pass over the whole volume (transpose copy,
  XOR of shifted copies, `argwhere`). It is fast, but it allocates full-volume
  temporaries: +0.50 GB, 0.9 s. Master also keeps a uint8 copy of the volume
  from `reduce_dtype`.
- **Instance** loops in Python over every slice and every instance in it
  (`find_objects`, then one encode per instance). The temporaries are small but
  it is slow: 63.8 s, no measurable extra RAM.

For the same reason, instance runs spend most of their time encoding the output
(65 of 77 s in the slice-wise z-only run).

**Why postprocess memory does not drop with slice-wise, but halves on AIOD-312.**
Postprocess needs the whole volume at once, so slice-wise collects its planes into
the same dense array master builds. It then runs master's `connect_components`
unchanged, which takes `dask_image` labels and `.compute()`s them into RAM: an int32
volume (4 B/vox, 1.07 GB) plus dask's intermediates, +3.2 GB in all. Slice-wise
gains nothing here by design: step 1 kept master's postprocess algorithms, with
streaming left as a follow-up.

AIOD-312 never holds the volume: dask reads the scratch Zarr store and writes the
labels into a second store on disk. That halves the memory, but every chunk is
compressed, written and read back, so it is 8× slower (97.9 s vs 12.8 s). It
leaves `connect_sam` dense.

A streaming postprocess fits the slice-wise loop without a store:
- **`connect_components`:** label each plane in 2D, union labels that touch
  across consecutive planes, then relabel. That takes two passes over the input,
  with RAM of about 2 planes plus the label-equivalence table.
- **`connect_sam`:** it already works on consecutive slice pairs. The final
  `relabel_sequential` becomes a mapping applied while encoding.

**Instance TIFF with overlap > 0** decodes everything twice. The first pass finds
the output dtype, because summed labels cannot be predicted from the RLE. With
overlap 0, the maximum comes from the labels and offsets without decoding.

### Step 1, large preset on the cluster (partial)

Large (512×2048×2048, grid 8×1×1, instance, block 16, overlap 0), `ncpu` node,
`results/sweep_20261002-112735` (job 59062343, still running at writing):

| scenario | wall, master (s) | wall, slice-wise (s) | peak RSS, master (GB) | peak RSS, slice-wise, harness (GB) | RSS after save, slice-wise, in-script (GB) |
|---|---|---|---|---|---|
| rle  | 4811 | 4628 | 8.44 | 3.93* | 1.27 |
| tiff | 1415 | 1826† | 6.15 | 3.93* | 0.46 |

Outputs are identical.
- \* is the harness's own RSS, not slice-wise's (see Measurements). Master's peaks
  are above that floor, so they stand.
- † the script's phase table totals 978 s; the other ~850 s are outside it
  (start-up/exit) and unexplained, likely the node or filesystem.
- **Time.** Instance runs are bound by encode time, not memory: encode is 3668 s
  (79%) and decode 935 s, both master's algorithms.
- **Remaining memory.** Slice-wise holds the output RLE list in RAM until it is
  pickled, ~3.5× the file size: 297 MB → ~1 GB here.

### AIOD-312 on real data

EMPIAR mito, 3598×3944×4455 (63 Gvox, 126 GB as uint16), 8 substacks (2×2×2),
empanada, no postprocess, from ahmedn's `stresstest_large_resume.out`:
- 12.0 min in all: store insert 341 s, read + encode 330 s, decode 50 s.
- Output 23.5 MB; 50 GB requested; logged RSS ≤ 0.69 GB.
- AIOD-312 still decodes each whole substack (~7.9 Gvox) before writing it to the
  store, so its true peak is far above the logged RSS (the memray profile was not
  readable to check). Slice-wise decodes one slice per substack and skips the
  store, which was 47% of the time.
- Master's dense volume (126 GB) would not fit in the 50 GB request at all.

**Postprocess** is the open part: slice-wise still builds the dense volume for it.
The streaming approach above (2D label + union across planes; `connect_sam` on
slice pairs) needs no Zarr store, and AIOD-312's store-based empanada postprocess
is 8× slower and leaves `connect_sam` dense.

### Step 3: Zarr substacks

Every Zarr output is identical to the RLE route's, and peak RSS is within
±0.1 GB of it. Sources: `results/sweep_20261001-130527` (small, 5 formats,
overlap 0 and 0.1) and `results/sweep_20261001-134832` (medium, 3 formats,
overlap 0). All rows below are medium, overlap 0. Wall time includes ~1.7 s of
start-up.

Combine time, RLE inputs vs Zarr (dir, zstd, chunk depth 8):

| scenario | RLE: wall (s) | RLE: decode (s) | RLE: encode (s) | Zarr: wall (s) | Zarr: read (s) | Zarr: encode (s) | wall change (%) |
|---|---|---|---|---|---|---|---|
| instance, block 4, z-only | 588 | 82 | 500 | 512 | 0.27 | 508 | −13 |
| instance, block 4, xy | 372 | 55 | 309 | 317 | 0.31 | 312 | −15 |
| instance, block 16, z-only | 79.3 | 11.3 | 64.5 | 67.4 | 0.15 | 64.5 | −15 |
| instance, block 16, xy | 16.4 | 2.1 | 10.4 | 13.2 | 0.20 | 10.4 | −20 |
| instance, block 64, z-only | 3.96 | 0.25 | 1.48 | 3.91 | 0.17 | 1.70 | −1 |
| binary, block 4, z-only | 8.90 | 5.25 | 1.04 | 3.80 | 0.25 | 1.00 | −57 |
| binary, block 16, z-only | 4.34 | 1.31 | 0.75 | 2.82 | 0.13 | 0.75 | −35 |
| binary, block 64, z-only | 3.08 | 0.33 | 0.71 | 2.55 | 0.08 | 0.70 | −17 |

Substack write cost (what each runModel task pays) and input size on disk, for
the same scenarios:

| scenario | write per substack, RLE (s) | write per substack, Zarr dir zstd c8 (s) | inputs, RLE (MB) | files, RLE | inputs, Zarr dir zstd c1 (MB) | files, Zarr dir zstd c1 | inputs, Zarr dir zstd c8 (MB) | files, Zarr dir zstd c8 | inputs, Zarr zip lz4 c8 (MB) | files, Zarr zip lz4 c8 |
|---|---|---|---|---|---|---|---|---|---|---|
| instance, block 4, z-only | 62.3 | 0.08 | 148 | 8 | 21.1 | 264 | 21.0 | 40 | 59.4 | 8 |
| instance, block 4, xy | 10.4 | 0.02 | 183 | 32 | 22.0 | 1056 | 5.9 | 160 | 58.5 | 32 |
| instance, block 16, z-only | 8.27 | 0.07 | 36.2 | 8 | 3.33 | 264 | 2.96 | 40 | 29.5 | 8 |
| instance, block 16, xy | 0.35 | 0.02 | 27.1 | 32 | 3.71 | 1056 | 0.79 | 160 | 18.4 | 32 |
| instance, block 64, z-only | 0.19 | 0.07 | 2.05 | 8 | 2.85 | 264 | 2.85 | 40 | 24.4 | 8 |
| binary, block 4, z-only | 0.17 | 0.08 | 64.5 | 8 | 23.4 | 264 | 23.4 | 40 | 7.59 | 8 |
| binary, block 16, z-only | 0.11 | 0.07 | 16.1 | 8 | 3.91 | 264 | 3.91 | 40 | 8.94 | 8 |
| binary, block 64, z-only | 0.10 | 0.08 | 4.29 | 8 | 1.91 | 264 | 1.90 | 40 | 7.52 | 8 |

**What this shows:**
- **Decode.** Zarr removes nearly all of the input-decode time.
- **Instance masks are limited by the encode.** Encoding the output is unchanged
  and dominates instance runs, so the combine is only 13–20% faster.
- **Binary masks gain more.** Decode is a large share of the work, so the combine
  is 35–57% faster for blocks of 4–16. At block 64 there is little to decode.
- **Substack writes.** Writing Zarr is far cheaper than RLE-encoding instance
  substacks (8–62 s vs under 0.1 s). The runModel side would get faster.
- **File count.** Chunk depth 1 in a dir store writes one file per slice per
  substack (1056 files for 32 substacks). Chunk depth 8, or a zip store, stays
  close to RLE's count.
- **Compressors.** zstd is smaller than Blosc-LZ4 in most cases.

**Against the proposed rule** (≥ 25% combine-time drop at medium and above,
net of the runModel-side cost):
- binary masks pass;
- instance masks do not, on combine time alone;
- the runModel cost is negative, so the remaining cost is the change itself:
  `utils.save_masks`, every `run_*.py` through it, napari's live per-substack
  `.rle` watcher, and the file count (use chunk depth ≥ 8 or zip).

The synthetic content at block 4 with 1000 ids is denser than real masks.
Re-check with real substacks:

    python bench_combine.py --inputs-from DIR

`--inputs-from` reads RLE only, so a Zarr comparison on real data needs the
substacks converted first.
