# AIOD-266: combine_masks memory investigation — session notes

## Ticket goal
`combine_masks` (in `modules/models/resources/usr/bin/combine_stacks.py`) is memory-problematic.
Plan: get a big dataset (EMPIAR-12627, Fib-SEM, single 117GB .tif), run Segment-Flow as a real
Nextflow/Slurm job with Empanada, profile `combine_stacks.py` with memray, confirm it should use a
memory-mapped/chunked store (Zarr preferred over raw memmap) instead of a dense in-RAM array, then
use real numbers to fix the dynamic memory request for the `combineStacks` process.

HPC path: `/nemo/stp/ddt/working/ahmedn/aiod-266/Segment-Flow` (Nemo/Crick cluster).
Local checkout: `/Users/ahmedn/aiod/Segment-Flow`.

## Concepts clarified
- **"Run as a job"** = run the real pipeline via `nextflow run ... -profile crick`, not the script
  standalone — each pipeline stage is a Nextflow process, submitted to Slurm on the crick profile,
  with its own memory/time request. Must profile under real Slurm memory constraints, not ad hoc.
- **Memory-mapped** (`numpy.memmap`): array bytes live on disk, OS pages them into RAM on demand.
- **Storage-backed/chunked** (Dask/Zarr): array split into chunks stored/compressed on disk, never
  fully materialized. Zarr preferred: proper chunked format + compression + plays natively with
  `dask.array.from_zarr`/`.to_zarr()` for lazy compute.
- Known hotspots in `combine_stacks.py`:
  - `combine_masks()` line 53: `all_masks = np.zeros(image_size, dtype=np.uint16)` — dense,
    full-volume, fully in-RAM. Prime OOM suspect.
  - `connect_components()` (~145-153): wraps in `da.from_array(...)` then immediately calls
    `.compute()` — discards Dask's laziness right away, defeats the point of using Dask at all.
  - `insert_mask()` (~122-130): commented-out uint16 overflow guard on label-uniqueness offset —
    separate **correctness** bug (silent wraparound), not memory. Worth its own ticket.

## Changes already made this session (uncommitted, local checkout only)
1. `nextflow.config`: added `params.profile_memory = false` (new "Debug parameters" section).
2. `modules/models/envs/conda_combine_stacks.yml`: added `memray` as a conda-forge dependency.
3. `modules/models/main.nf`, `combineStacks` process script: when `params.profile_memory` is true,
   wraps the python call as `memray run --native -o ${mask_fname}_memray.bin ...` and afterwards
   runs `memray flamegraph` + `memray stats` into `${mask_fname}_memray_flamegraph.html` /
   `_stats.txt`. **Not** added as declared Nextflow `output:` — these files land in the task's
   ephemeral work dir only (`find work/ -name "*_memray*"` or `nextflow log <run> -f workdir,name`
   to locate), deliberately, to avoid changing the process's output channel shape.

## HPC run log — issues hit, in order
1. **Conda cache lock "Permission denied"**: `/flask/conda/ddt/.combine_conda-<hash>.lock`.
   Cause: editing `conda_combine_stacks.yml` (adding memray) changed the env's content hash, so
   Nextflow needed to build a *new* cache entry in the shared `/flask/conda/ddt/` dir
   (`conda.cacheDir`, set in `profiles/crick.conf:31`) — likely only writable by whoever seeded it.
   `splitStacks` hit it first only because it reuses the same yml and runs earliest in the DAG, not
   because it needs memray itself. Fix suggested: override `conda.cacheDir` to a personal writable
   dir via a `-c my_overrides.config` file. **Appears resolved** (pipeline progressed further).
2. **`ModuleNotFoundError: No module named 'aiod_utils'`** in `create_splits.py`, despite
   `aiod_utils==0.1` being listed under `pip:` in the yml. Two live hypotheses, unconfirmed:
   (a) stale/incomplete cached env reused from an earlier failed attempt (Nextflow only checks dir
   existence, not that every package installed); (b) pip install of aiod_utils silently failed
   during env build because the **compute node** (not login node) lacks PyPI/internet access —
   common on HPC — and conda/mamba didn't hard-fail the whole env build on that pip subsection.
   Diagnostics given but not yet run/confirmed: check env's `site-packages` for aiod_utils directly,
   or manually `mamba env create -f conda_combine_stacks.yml -p /tmp/test_env` on a compute node.
   **Status: pipeline got past this eventually (runModel ran), root cause not confirmed.**
3. **`runModel` OOM, exit 137 (SIGKILL)** — currently blocking, **unresolved, no decision made**.
   - `runModel` has no explicit `memory{}`; inherits from `withLabel: gpu_process` in
     `profiles/crick.conf:16-21`: `memory = { params.memory_per_job * task.attempt }`.
   - `params.memory_per_job = 50.GB` default (`profiles/crick.conf:3`).
   - `errorStrategy` retries exit codes 135-143 (`profiles/crick.conf:13`), `maxRetries = 3`
     (`profiles/crick.conf:14`) → escalates to `50 × 3 = 150GB` before terminating. Still OOM'd.
   - Root cause hypothesis: `create_splits.py --memory-per-job <bytes>` (passed from
     `modules/models/main.nf:44-46`) sizes substacks assuming raw substack byte size ≈ memory
     budget, but Empanada/MitoNet's actual peak memory during 3D inference is apparently a large
     multiplier above that. The same `memory_per_job` param drives both the substack-sizing target
     *and* the literal Slurm memory request for the GPU job — coupling that has no accounting for
     the model's real memory multiplier.
   - Pasted `WorkflowStats` log line literally contained `peakMemory=1.6 TB` and `peakRunning=8`
     (these two are directly-stated facts from the log, not inferred). Dividing them for a
     "~200GB/task" estimate was **my own unverified arithmetic**, layered on top — not confirmed,
     and I'm not certain what Nextflow's `WorkflowStatsObserver.peakMemory` actually aggregates
     (sum of requested vs. actual usage across concurrent tasks, at what sampling granularity).
   - When asked to choose a path forward (pragmatic memory bump / get real `seff`/`sacct` MaxRSS
     first / properly decouple substack-sizing budget from job memory request into two params),
     **user declined to pick ("exit") — decision explicitly deferred, not resolved.**

## Self-correction note
Earlier in the session I cited `profiles/crick.conf` line numbers from memory (`:2`, `:18-19`)
without re-reading the file — they were wrong (`:3`, `:13` respectively). Root cause: file was
first viewed via `cat` (no line numbers), then cited later without a fresh `Read`. Lesson: always
re-`Read` a file immediately before citing precise line numbers, don't rely on memory of prior
`cat` output.

## Open decisions / next steps
1. Decide how to unblock `runModel` OOM (see three options above — still an open fork).
2. Confirm the `aiod_utils` ModuleNotFoundError root cause (network vs. stale cache) if it recurs.
3. Once `runModel` → `combineStacks` runs end-to-end on the real 117GB-derived substacks, do the
   actual profiling run: `-profile crick --profile_memory true`, then `memray flamegraph`/`stats`
   on the captured `.bin`, cross-checked against `seff <jobid>`/`sacct --format=MaxRSS` for ground
   truth (memray shows *where* in the code memory goes; Slurm's MaxRSS is what actually gets
   enforced/OOM-killed on, and is the number to design the memory formula around).
4. Fix `combine_stacks.py`: back `all_masks` with a Zarr store instead of dense `np.zeros(...)`;
   rework `connect_components` to use `da.from_zarr(...)` / `.to_zarr(...)` instead of `.compute()`.
5. Re-profile post-fix, then rewrite the `combineStacks` memory formula
   (`modules/models/main.nf:153`, currently `Math.max(5GB, masks*.size().sum() * 10000) * attempt`
   — a poor proxy since it scales off input RLE-compressed size, not the actual output volume) using
   the real measured numbers.

## Getting image dims/dtype for the 117GB tif without loading it
```python
import tifffile
with tifffile.TiffFile("/path/to/big.tif") as tf:
    s = tf.series[0]
    print(s.shape, s.dtype, s.axes)   # header/IFD only, no pixel decode; handles BigTIFF transparently
```
Check `axes` to map shape entries onto the `height,width,num_slices,channels` CSV columns needed by
`create_splits.py` (required columns: `img_path,height,width,num_slices,channels`, optional `dtype`
— supplying `dtype` yourself skips `create_splits.py`'s own `aiod_utils.load_image(...).dtype` call
at line 94, avoiding an extra load path on the huge file).
