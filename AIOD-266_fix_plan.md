# AIOD-266 — `combineStacks` memory fix: plan and rationale

**Status:** analysis complete, profiling done, **both phases implemented** (see §7 and §13).
Not yet re-profiled on the cluster — that is the one remaining verification step.
**Dataset used:** EMPIAR-12627 Fib-SEM, single 117 GiB `.tif`, `3598 × 3944 × 4455` uint16.
**Profiled run:** `-profile crick --profile_memory true`, empanada/MitoNet, 8 substacks, overlap 0,
binary masks, **no** `--postprocess`. Duration 11m 48s. Peak RSS **253.5 GB**.

---

## 1. TL;DR

| | |
|---|---|
| **Measured peak** | 253.5 GB, i.e. **4.01 bytes per voxel** |
| **What that is** | four copies of the whole mask volume, alive at the same time |
| **Where** | three of the four are inside `aiod_rle.encode`, *not* in `combine_masks` |
| **Ticket's original plan** | put `all_masks` in a Zarr store |
| **Problem with that** | on its own it changes the peak by **0%** |
| **Actual fix** | Zarr store **+** encode the volume in z-slabs. Both halves, or neither works. |
| **Expected result** | ~253 GB → **~25 GB**, and **flat in slice count** instead of linear |
| **Output format** | **unchanged**. Byte-identical `.rle` — verified across 13 configurations. |

> **Revised down from an earlier "< 10 GB" estimate.** Once the volume is out of RAM, the
> peak is set by **decoding one input substack**, not by the encode. `aiod_utils`'
> `_decode_binary` holds a per-slice list *and* the `np.stack` copy of it, so one
> 1799×1972×2227 substack costs ~15 GiB, and the `astype(uint8)` plus the `insert_mask`
> temporary push that to ~22 GiB. Getting below that needs a streaming *decode* too
> (§11 item 2), which is out of scope here. The ~25 GB figure is a projection from
> measured scaling behaviour, not yet a cluster measurement.

---

## 2. Background: the three pieces you need to know

### 2.1 What the mask volume is

The pipeline splits the image into substacks, runs the model on each, then `combineStacks`
reassembles them into one mask volume the same shape as the input image.

```
image_size          = 3598 × 3944 × 4455  = 63.22 billion voxels
1 byte  per voxel   = 63.22 GB  (58.88 GiB)
2 bytes per voxel   = 126.4 GB  (117.75 GiB)   ← this is the 117 GiB input file
```

Everything below is easier if you hold onto one number: **one full-volume byte array = 63 GB.**

### 2.2 What RLE is

Run-length encoding stores "how long each run of identical pixels is" instead of the pixels.
A row `0 0 0 1 1 0` becomes `[3, 2, 1]`. For sparse masks this is a massive saving —
in a test below, a 480 MB volume compressed to **0.3 MB**.

### 2.3 What `aiod_rle.encode` returns

This is the fact the whole fix rests on. It returns **a flat Python list with one dict per
z-slice**, plus one trailing metadata dict at the end.

A tiny real example — a 4-slice, 3×5 volume with blobs on z=0,1,2 and z=3 empty:

```
encode(whole volume) -> a list of 5 items:
  [0] {'size': [3, 5], 'counts': [4, 1, 2, 1, 7]}     <- z=0
  [1] {'size': [3, 5], 'counts': [0, 1, 14]}          <- z=1
  [2] {'size': [3, 5], 'counts': [11, 1, 2, 1]}       <- z=2
  [3] {'size': [3, 5], 'counts': [15]}                <- z=3  (empty slice = one run of 15)
  [4] {'metadata': {'mask_type': 'binary'}}           <- one trailing dict
```

Nothing clever is happening. It is a list, in z order, one entry per slice.

---

## 3. Why memory blows up

### 3.1 The four copies

`combine_stacks.py:419` hands the **entire** volume to `aiod_rle.encode` in one call. Inside
`encode` → `_encode_binary`, three whole-array numpy operations each allocate a fresh full-volume
array, and none of them can be freed until the function returns.

Line numbers are current `aiod_utils/rle.py` (post-AIOD-341). The flame graph shows `41` rather
than `42` because it profiled `aiod_utils==0.1`, before AIOD-341 shifted that line by one.

| # | Code | Why it allocates | Full volume | 64-slice slab |
|---|---|---|---|---|
| 1 | `combine_stacks.py:108` `reduce_dtype(all_masks)` | the combined result itself | 63.22 GB | — (stays on disk) |
| 2 | `rle.py:42` `mask = mask.astype(bool)` | dtype change ⇒ numpy must copy | +63.22 GB | +1.12 GB |
| 3 | `rle.py:69` `mask.transpose(0,2,1).reshape(b,-1)` | transpose makes it non-contiguous, so `reshape` **cannot** return a view — it copies | +63.22 GB | +1.12 GB |
| 4 | `rle.py:73` `diff = mask[:,1:] ^ mask[:,:-1]` | output buffer for the XOR | +63.22 GB | +1.12 GB |
| | | **total live at once** | **253 GB** | **~4.5 GB** |

`4 × 63.22 GB = 253 GB`, and the measured peak was `253.5 GB`. The model is exact, not fitted.

**These three copies are inherent to how the algorithm is written. The fix is not to remove them.
The fix is to make sure they only ever apply to a small slab.**

### 3.2 The timeline, from the temporal flame graph

Allocation lifetimes, 1 snapshot = 71 ms:

```
GiB
236 ┤                                              ┌──────────────────┐
    │                                              │ XOR temp  :73    │
177 ┤                          ┌───┐          ┌────┴──────────────────┤
    │                          │   │          │ reshape copy   :69    │
118 ┤ ┌────────────────────────┴─┐ │     ┌────┴───────────────────────┤
    │ │  np.zeros :53  117.75GiB │ │     │ astype(bool)   :42         │
 59 ┤ │  (alive 57.5s -> 426.7s) │ ├─────┴────────────────────────────┤
    │ │                          │ │ reduce_dtype io.py (the result)  │
  0 ┼─┴──────────────────────────┴─┴──────────────────────────────────┘
     57s        ...8 substacks...      426s   442s  450s   503s     707s
                                        ^            ^
                              np.zeros freed    all four live: 204s at 236 GiB
```

Two things worth noticing:

1. **`np.zeros` at `combine_stacks.py:53` — the ticket's original prime suspect — dies 280 seconds
   before the peak.** It is genuinely 117.75 GiB, but it is *not* what sets the high-water mark.
   It is invisible in the default (peak-snapshot) flame graph for exactly this reason.
2. The run sat at 236 GiB for **204 seconds — 29% of its wall time.** Slurm OOM-kills on the
   *height* of that plateau, not its duration.

### 3.3 Why AIOD-341 does not change this

`aiod_utils` AIOD-341 ("RLE Optimisation", squash-merged to `origin/main` as `5ba443c`) replaced an
`O(b × total_changes)` per-slice rescan with `searchsorted` + `np.split`. `np.split` returns
**views**, so it allocates nothing. Rows 2, 3 and 4 of the table above are untouched by that diff.

> **Binary-path peak memory is unchanged by AIOD-341.** What it fixes is the 204-second plateau —
> same height, much narrower. Good for the time budget, irrelevant to the memory request.

For **instance** masks AIOD-341 *is* a large memory win: it deleted `mask.astype(np.int64)`, which
was an 8 bytes/voxel copy — **471 GiB** on this dataset. Any earlier estimate of ~9–10 bytes/voxel
for the instance path applies to `0.1` only and should not be reused.

**Blocker:** PyPI has **only `aiod_utils==0.1`**, and `modules/models/envs/conda_combine_stacks.yml`
pins `aiod_utils==0.1`. AIOD-341 is unreleased, so the pipeline cannot see it. A release + pin bump
is a prerequisite for any instance-path work.

**Compatibility note:** the new instance format adds `offset` / `full_size`, and `size` now means
the *bbox* size. New code reads old files fine (`.get()` fallbacks); **old `0.1` cannot read new
files** — its `_decode_binary` would `np.stack` mismatched shapes and raise. Since `.rle` files are
cached in `aiod_cache` across runs, a half-upgraded fleet can hit this. The binary format is
unchanged, so existing binary caches are safe.

---

## 4. The key insight

> **Each z-slice's encoded entry depends only on that slice's own pixels.**

This is verifiable by reading `_encode_binary` in `aiod_utils/rle.py`:

```
out[i]["counts"]  (line 109)
  <- groups[i]                              (line 92)
  <- col split at row boundaries            (lines 82-84)
  <- np.argwhere(diff)                      (line 75)
  <- diff = mask[:,1:] ^ mask[:,:-1]        (line 73)   row i of diff comes only from mask[i]
  plus mask[i, 0]                           (line 103)
```

There is no term anywhere in that function that mixes slice *i* with slice *j*.
`_encode_instance` is an explicit per-slice loop, so the same holds there.

**Encoding is therefore a *map* over slices. There is no global state to preserve.**

But it is *implemented* as "hand me the whole 3D array, and I will do three whole-array numpy
operations before I start looping." That mismatch — a per-slice problem implemented as a
whole-volume operation — is the entire bug.

---

## 5. The fix, in two halves

### Half 1 — Streaming encode: call `encode` on slabs, glue the lists

```python
out = []
for z0 in range(0, vol.shape[0], 64):       # 64 slices at a time
    slab_rle = aiod_rle.encode(vol[z0:z0+64], mask_type=resolved, metadata={})
    out += slab_rle[:-1]                    # drop this slab's trailing metadata dict
out.append({"metadata": {"mask_type": resolved}})   # add exactly one at the end
```

Verified on the tiny 4-slice example, 2 slices per slab:

```
  slab z=0:2 -> 3 items (2 slice entries + 1 metadata)
  slab z=2:4 -> 3 items (2 slice entries + 1 metadata)

IDENTICAL: True
decode(streamed) == original volume : True
decode(streamed) == decode(whole)   : True
```

Because the output is a per-slice list in z order, concatenating slabs in z order reproduces the
*same list, element for element*. **The `.rle` file is byte-identical.**

**Analogy:** you don't need the whole book in memory to write down a word count for each page.
Read a page, count it, append the number, move on. Same output; one page of RAM instead of one book.

### Half 2 — Zarr-backed `all_masks`: stop holding the volume in RAM at all

Replace the dense `np.zeros(image_size, dtype=np.uint16)` at `combine_stacks.py:53` with a chunked,
compressed Zarr array on node-local scratch. Substacks get written into their region; `store[z0:z1]`
later reads back only those chunks.

### Why you need both

| Approach | kills | what's left | peak |
|---|---|---|---|
| Zarr only (original ticket plan) | the `1 × V` dense array | `encode()` requires an ndarray (`rle.py:19`), so you read all of V back in and re-create all three temporaries | **253 GB — no change** |
| Streaming only | the `3 × V` temporaries | you still built a dense `all_masks` | ~184 GB |
| **Both** | both | one slab + one substack | **< 10 GB** |

- **Streaming** decides *how much you touch at once*.
- **Zarr** decides *where the full volume lives*.

Neither alone is sufficient. This is the single most important correction to the original ticket.

---

## 6. Measurements backing all of the above

All run against the real `aiod_utils/rle.py`, on a 600 × 800 × 1000 uint8 volume (480 MB) of sparse
blobs. Peak RSS via `ru_maxrss`, three separate processes.

```
today          volume= 480.0 MB  peak_rss= 2002.5 MB  => 4.04 x volume
stream-only    volume= 480.0 MB  peak_rss=  763.6 MB  => 1.46 x volume
zarr-stream    volume= 480.0 MB  peak_rss=  489.0 MB  => 0.89 x volume

today == stream-only : True
today == zarr-stream : True
```

`today` reproduces **4.04 × volume** at 1/130th the scale of the cluster run's 4.01 — strong evidence
the four-copy model is correct rather than coincidental.

`stream-only` is 1.46 × rather than 0.46 × precisely because `all_masks` is still dense in RAM.
That residual `1 ×` **is** `combine_stacks.py:53`. This is the empirical proof that half 2 is required.

### Choosing the slab size

Encode phase only, volume already on disk in Zarr (480 MB uncompressed → **0.3 MB** compressed):

```
  SLAB= 600 slices (480.0 MB)  peak=1974.1 MB  = 4.11 x volume  =  4.11 x slab
  SLAB= 128 slices (102.4 MB)  peak= 421.1 MB  = 0.88 x volume  =  4.11 x slab
  SLAB=  64 slices ( 51.2 MB)  peak= 221.9 MB  = 0.46 x volume  =  4.33 x slab
  SLAB=  16 slices ( 12.8 MB)  peak= 118.3 MB  = 0.25 x volume  =  9.24 x slab
  SLAB=   4 slices (  3.2 MB)  peak=  70.4 MB  = 0.15 x volume  = 22.02 x slab
```

- `SLAB = 600` **is today's code** — one slab covering the whole volume.
- The `4 × slab` law holds exactly down to ~64 slices. **Peak is `4 × slab`, and you choose the slab.**
- Below ~16 slices the ratio climbs because you hit a floor: the accumulated output RLE list
  (360,600 Python ints here) plus the interpreter. Absolute peak still falls, but with diminishing
  returns and more Python-loop overhead.
- **Use 64.** On the real volume one z-slice is 17.57 MB, so 64 slices = 1.12 GB and encode peak
  ≈ **4.5 GB** instead of 253 GB.

---

## 7. Implementation

### Phase 1 — streaming encode (small, self-contained, no Zarr required)

Add to `combine_stacks.py`:

```python
def encode_streaming(masks, mask_type, metadata, slab=64):
    """
    Slab-wise equivalent of aiod_rle.encode(masks, mask_type, metadata).

    encode() returns one entry per z-slice plus a trailing metadata dict, and each
    slice's entry depends only on that slice's pixels -- so encoding in z-slabs and
    concatenating is byte-identical, while holding only `slab` slices (and encode's
    three transient full-size copies of them) in RAM at a time.

    `masks` may be a numpy array, a zarr Array, or anything supporting [z0:z1]
    slicing that returns a numpy array.

    mask_type MUST be resolved by the caller: encode() would otherwise infer it per
    slab via check_mask_type(), which can classify a sparse slab as "binary" and a
    busy one as "instance", producing a structurally invalid file.
    """
    if mask_type is None:
        raise ValueError("mask_type must be resolved before streaming encode")
    if masks.ndim == 2:
        return aiod_rle.encode(np.asarray(masks), mask_type=mask_type, metadata=metadata)
    out = []
    for z0 in range(0, masks.shape[0], slab):
        chunk = np.asarray(masks[z0 : z0 + slab])
        out.extend(
            aiod_rle.encode(chunk, mask_type=mask_type, metadata=dict(metadata))[:-1]
        )
    out.append({"metadata": {**metadata, "mask_type": mask_type}})
    return out
```

Then at `combine_stacks.py:419`:

```diff
-        encoded_masks = aiod_rle.encode(
-            combined_masks,
-            mask_type=resolved_mask_type,
-            metadata=metadata,
-        )
+        encoded_masks = encode_streaming(
+            combined_masks,
+            mask_type=resolved_mask_type,
+            metadata=metadata,
+        )
```

**Already tested:** 8 mask shapes (sparse/dense/all-zero/all-ones/single-slice/instance/uint16
labels) × 6 slab sizes (1, 2, 3, 7, 64, all) × both mask types × sourced from both numpy arrays and
Zarr stores → **0 failures, byte-identical output in every case.**

Note `np.asarray(masks[z0:z0+slab])` **already accepts a Zarr array**, so phase 2 does not have to
touch the encode path again.

### Phase 2 — Zarr-backed `all_masks`

1. `combine_stacks.py:53` — replace `np.zeros(image_size, dtype=np.uint16)` with a Zarr array on
   node-local scratch, chunked to align with the substack grid (e.g. `(64, 1024, 1024)`) with
   Blosc/Zstd compression. Mask data compresses enormously (480 MB → 0.3 MB in the test above).
2. `insert_mask:124` — `all_masks[start_z:end_z, ...].max()` must not materialise the slab. Use
   `da.from_zarr(store)[start_z:end_z].max().compute()`.
3. `combine_masks:108` — drop the eager full-volume `reduce_dtype`. Choose the store dtype up front
   (uint16 costs almost nothing once compressed) and downcast only the small slabs you materialise.
4. `connect_components:145-153` — replace `da.from_array(...)` + `.compute()` with
   `da.from_zarr(...)` → `dask_image.ndmeasure.label(...)` → `.to_zarr(...)`. Never `.compute()`.
5. `combine_stacks.py:401-413` (TIFF branch) — `tifffile.imwrite` accepts an iterator of slices, so
   it can stream too.
6. Keep the existing dense path for SAM (`connect_sam` is slice-wise and `relabel_sequential` needs
   dense; SAM data is small). Gate on `model`.

---

## 8. Gotchas that must be in the implementation

1. **`resolved_mask_type` can be `None`.** `combine_stacks.py:416-418` yields `None` when
   `--output-mask-type auto` is set *and* the patches carry no `mask_type` metadata. Today `encode`
   infers once for the whole volume; per-slab it would infer independently and could mix types
   within one file. Hence the explicit `raise` in `encode_streaming`. Resolve once, up front.
2. **Do not fix (1) by calling `check_mask_type` on the full volume.** `rle.py:55` does
   `np.unique(mask)`, which sorts a copy — another full 63 GB allocation. (The comment at
   `combine_stacks.py:368` already hints at this cost.) Resolve incrementally: scan slabs
   accumulating distinct values, early-exit once more than 2 are seen.
3. **Strip each slab's trailing metadata dict** (`rle.py:49` appends one per call) and append
   exactly one at the end. Getting this wrong silently corrupts the file.
4. **Slab size is a constant, not a per-dataset tunable.** 64.

---

## 9. The dynamic memory request (`modules/models/main.nf:153`)

### What's wrong with it now

```groovy
memory { (Math.max((5.GB).toBytes(), masks*.size().sum() * 10000) * task.attempt) as MemoryUnit }
```

The `× 10000` is documented as "buffer (10) × average compression factor (1000)". The constant isn't
the problem — **the independent variable is.** Peak RSS is a function of *voxel count*; the input is
*RLE-compressed bytes*. The ratio between them is the mask's compression factor, which swings by
orders of magnitude with mask complexity (0.3 MB from 480 MB in one test above). The formula has no
error bound in either direction. It happened not to OOM on this run; that was luck.

The right variable is already in scope — `main.nf:188` passes
`--image-size ${meta.num_slices} ${meta.height} ${meta.width}`.

### Interim formula (if the code fix is deferred)

```groovy
memory {
    // Peak RSS is driven by voxel count, not RLE-compressed input size.
    // Measured AIOD-266 (EMPIAR-12627, 3598x3944x4455 = 63.2 Gvoxel, binary, no postprocess):
    //   253.5 GB peak = 4.01 B/voxel == 4 live full-volume uint8 arrays
    //   (result + astype(bool) + transpose/reshape copy + XOR temp) inside aiod_rle.encode.
    def voxels = (meta.num_slices as long) * (meta.height as long) * (meta.width as long)
    def bpv = params.postprocess ? 10 : (output_mask_type == 'binary' ? 4 : 10)
    (Math.max((5.GB).toBytes(), (2.GB).toBytes() + voxels * bpv) * task.attempt) as MemoryUnit
}
```

| Path | bytes/voxel | Basis |
|---|---|---|
| binary, no postprocess | **4.01** | **measured**, and matches the 4-array model exactly |
| combine phase alone | ~3 | measured 184 GiB ≈ 2 (uint16 store) + 1 (downcast copy) |
| instance, `aiod_utils==0.1` | ~9–10 | derived: `astype(np.int64)` = 8 B/vox + result |
| instance, post-AIOD-341 | ~2–3 | derived: int64 upcast removed; **unmeasured** |
| postprocess | ~10+ | derived: int32 label `.compute()` + input + downcast, then encode; **unmeasured** |

`output_mask_type` can be `'auto'`, resolved at *runtime* from patch metadata — Nextflow cannot know
it, so `'auto'` must be costed as instance.

**But be honest in the ticket:** at 10 B/voxel this volume asks for **632 GB**, and with
`maxRetries = 3` the escalation ladder tops 1.8 TB. That is not a request you can put on `ncpu`.
**The formula is the symptom, not the fix.**

### Post-fix formula — the real target

Once combination is Zarr-backed and encoding is streamed, peak is bounded by *one substack decode*
plus *one encode slab*, and becomes **independent of total volume**. The substack extents are
already in the mask filenames, so the bound can be computed rather than guessed:

```groovy
memory {
    // Post-fix: peak is bounded by the largest single substack decode + one encode slab,
    // NOT by total volume. Extents come from the *_x{a}-{b}_y{c}-{d}_z{e}-{f}.rle filenames.
    def big = masks.collect { m ->
        def g = (m.name =~ /_x(\d+)-(\d+)_y(\d+)-(\d+)_z(\d+)-(\d+)\./)
        g ? (g[0][2].toLong() - g[0][1].toLong()) *
            (g[0][4].toLong() - g[0][3].toLong()) *
            (g[0][6].toLong() - g[0][5].toLong()) : 0L
    }.max() ?: 0L
    // ~3 B/voxel for aiod_utils decode transients, + slab/Zarr-cache/interpreter headroom
    (Math.max((8.GB).toBytes(), (6.GB).toBytes() + big * 3) * task.attempt) as MemoryUnit
}
```

For this dataset: **~30 GB instead of 253 GB**, and it stays ~30 GB for a 1 TB image.

### Validate against Slurm before committing to constants

memray's tracked heap (253.36 GB) and RSS (253.50 GB) agree to 0.05%, so the 4 B/voxel figure is
solid — but Slurm enforces on its own accounting. From the run already completed:

```bash
sacct -j <jobid> --format=JobID,JobName,ReqMem,MaxRSS,MaxVMSize,Elapsed,State
seff <jobid>
grep -h 'Memory used' work/*/*/.command.out     # combine_stacks.py's own psutil prints (353/363/388)
grep -h '^[0-9.]* [KMGT]B$' work/*/*/.command.log   # the `echo ${task.memory}` at main.nf:182
```

Once Zarr is in: put the store on node-local scratch. cgroup `memory.current` counts page cache for
those files (reclaimable, normally harmless, but it looks alarming); `sacct` MaxRSS does not. The two
diverging is expected, not a regression.

---

## 9b. Two findings that only showed up during implementation

Neither was in the original plan; both were measured, and both are now encoded as constants
with comments in `combine_stacks.py`.

**1. Zarr chunk shape matters more than expected.** A first attempt used
`ZARR_CHUNKS = (32, 512, 512)`. On a 512³ test volume that made each chunk span the entire
frame, so *every* substack write became a partial-chunk read-modify-write of a large chunk —
and zarr performs those concurrently. Writing 8 substacks of 256³ into a 512³ uint16 store:

```
  chunks=(32,512,512) [ 16.8 MB]  whole-substack   peak=  492.0 MB
  chunks=(64,256,256) [  8.4 MB]  whole-substack   peak=  102.6 MB
  chunks=(32,256,256) [  4.2 MB]  whole-substack   peak=   88.5 MB
```

**2. A whole-substack `__setitem__` leaves many chunks in flight.** Writing the same data a
z-slab at a time bounds it, independently of chunk shape:

```
  chunks=(32,512,512) [ 16.8 MB]  slab-wise        peak=  170.3 MB   (was 492.0)
  chunks=(64,256,256) [  8.4 MB]  slab-wise        peak=   40.0 MB   (was 102.6)
```

Both fixes are applied: `ZARR_CHUNKS = (64, 256, 256)` (z matched to `ENCODE_SLAB`, y/x kept
well below the frame size) and all region writes go through `write_blocks`, which slabs them.
Before these two changes the implementation measured **4.0 bytes/voxel** — i.e. the Zarr
store had bought nothing at all, because the write path had quietly reintroduced a
whole-volume-scale working set. This is worth knowing for anyone tuning the constants.

## 10. Testing / acceptance criteria

- [x] Unit test: `encode_streaming(...) == aiod_rle.encode(...)` across slab sizes `{1, 2, 3, 7, 64,
      all}` for both mask types, including all-zero, all-ones, single-slice and 2D inputs.
      → `tests/test_combine_stacks.py`, 90 tests passing.
- [x] Round-trip test: `decode(encode_streaming(v)) == v`.
- [x] Regression fixture: `.rle` output byte-identical to the pre-change implementation across
      13 end-to-end configurations (see §13).
- [x] Zarr-source test: `encode_streaming` over a Zarr array == over the equivalent numpy array.
- [ ] **Cluster re-profile with `--profile_memory true`** — the one outstanding item. Expect
      peak RSS ~25 GB with `--postprocess` off, cross-checked against `sacct` MaxRSS.
- [ ] `nextflow.config` / `main.nf`: memory formula switched to the post-fix form above
      (§9). **Not yet applied** — deliberately held until the cluster re-profile confirms
      the constant, since the current formula is over-provisioning rather than
      under-provisioning and is therefore not urgent.
- [ ] Note: CI does not run Python tests (`.github/workflows/` has only `update_envs.yml`),
      so `tests/test_combine_stacks.py` must be run manually for now.

---

## 11. Out of scope here, but found during profiling

1. **`insert_mask:130` `mask[mask > 0] += max_val`** allocates a full extra substack-sized temporary
   — visible in the temporal data as 7.36 GiB held for ~20 s on each of the 8 substacks. Not on the
   critical path today, but after phase 2 the substack working set *becomes* the peak. Also: for
   binary masks the uniqueness offset is pointless work, since `astype(bool)` at `rle.py:42`
   collapses it straight back.
2. **Decode holds 2 × substack.** `_decode_run_length` builds a per-slice list (7.36 GiB accumulated)
   and then `np.stack` copies it (another 7.36 GiB) at `rle.py:216`. A `decode_into(out=...)` API in
   `aiod_utils` would remove that, and it is the natural phase-3 companion to streaming encode —
   after phases 1 and 2, this decode is the new dominant term.
3. **`connect_components:150` `.compute()`** materialises the full int32 label volume — structurally
   the worst thing left in the file, and completely unprofiled. Needs its own measurement run.
4. **uint16 overflow guard commented out at `insert_mask:126-128`** — a silent-wraparound
   *correctness* bug on the instance path, not a memory bug. Deserves its own ticket.
5. **`aiod_utils` release + pin bump** — prerequisite for anything on the instance path (§3.3).
6. **`combine_masks` `xy_tiling` check looks transposed** (`combine_stacks.py:40-42`):
   `end_x < image_size[1]` compares an x extent against the *height*, and
   `end_y < image_size[2]` compares a y extent against the *width*, while `image_size` is
   (D, H, W). It happens not to matter on the profiled data (`start_x > 0` is true for half
   the substacks, so `xy_tiling` is True regardless), but it would misfire for a single
   tile on a non-square frame. Left unchanged here — fixing it changes behaviour and
   belongs in its own ticket.
7. **Labels are discarded after `--postprocess` when the patches are binary.**
   `resolved_mask_type` comes from the patch metadata (`combine_stacks.py:416-418`), so it
   is `"binary"` even after `connect_components` has produced a labelled volume — and
   `encode` then does `astype(bool)`, throwing the labels away. Confirmed empirically
   during this work. Pre-existing, preserved deliberately, but it means the postprocess
   path currently does expensive work for no benefit on binary input.

---

## 12. Suggested ticket split

| Ticket | Scope | Size |
|---|---|---|
| AIOD-266a | Streaming encode in `combine_stacks.py` + equivalence tests | **done** |
| AIOD-266b | Zarr-backed `all_masks`; completes the fix | **done** |
| AIOD-266c | Rewrite the `combineStacks` memory formula using §9 | small, pending re-profile |
| AIOD-266d | Release `aiod_utils` with AIOD-341, bump the pin; re-profile the instance path | small |
| AIOD-266e | `connect_components` / `--postprocess` path: profile, then fix `.compute()` | medium, unmeasured |
| AIOD-266f | uint16 overflow guard in `insert_mask` (correctness) | small |

---

## 13. Implementation record — what was measured

All figures below are from actual runs of the modified `combine_stacks.py` against a
reference implementation of the pre-change dense path (`reference_old.py`, replicating
`git show HEAD:...combine_stacks.py`), using the real `aiod_utils/rle.py`.

### Output equivalence: 13/13 configurations byte-identical

```
  binary 3D tiled 2x2x2 (profiled cfg)   new=3327B    old=3327B    IDENTICAL
  binary 3D, 200 slices (>3 slabs)       new=5973B    old=5973B    IDENTICAL
  binary 3D, Z=64 (slab boundary)        new=2909B    old=2909B    IDENTICAL
  binary 3D, Z=65 (slab boundary)        new=2947B    old=2947B    IDENTICAL
  instance 3D tiled                      new=6507B    old=6507B    IDENTICAL
  binary, z-split only (no xy tiling)    new=3521B    old=3521B    IDENTICAL
  binary all-zero                        new=1606B    old=1606B    IDENTICAL
  binary all-ones                        new=1816B    old=1816B    IDENTICAL
  binary, overlap>0 (sum branch)         new=3283B    old=3283B    IDENTICAL
  binary -> tiff                         new=173466B  old=173466B  IDENTICAL
  instance -> tiff                       new=151806B  old=151806B  IDENTICAL
  binary, explicit output-mask-type      new=3282B    old=3282B    IDENTICAL
  binary 2D (image_size[0] == 1)         new=149B     old=149B     IDENTICAL
```

`--postprocess` has no byte-comparable reference (the old path called `.compute()` then
`reduce_dtype`; the new one writes Zarr at dask_image's native dtype), so it was checked
against a dense reference by decoded pixel content instead: **matches**, 13 labels found.

### Memory: flat in slice count

Frame fixed at 256×256, substack fixed at 128³, slice count growing 16×. Peaks via
`/usr/bin/time -l`; the import-only baseline for this dependency set measured 121.8 MB.

```
       Z    Mvox  tiles |  old peak above base |  new peak above base | ratio  identical
     128     8.4      4 |    109.6M     -12.2M |    181.4M      59.6M |  0.60x  True
     512    33.6     16 |    261.5M     139.7M |    182.7M      60.9M |  1.43x  True
    1024    67.1     32 |    466.0M     344.2M |    189.2M      67.4M |  2.46x  True
    2048   134.2     64 |    872.8M     751.0M |    189.3M      67.5M |  4.61x  True
```

**Old grows linearly with slice count (140 → 751 MB); new is flat (60 → 68 MB).** That is
the property that matters: the profiled dataset has 3598 slices.

Two honest caveats on these numbers:

1. **At small volumes the new code is *worse*** (0.60× at Z=128) — the Zarr store, its
   compression buffers and the import baseline cost more than a small dense array. The
   crossover is around Z≈300 at this frame size. Not a concern for the datasets this
   ticket exists for, but it means "always faster/leaner" is false.
2. **New peak is flat in Z, not flat in volume.** It still scales with *frame area*, via
   the `4 × ENCODE_SLAB × H × W` encode-slab term. Growing the frame at fixed Z:
   256³ → 384³ → 512³ gave new peaks of 182 → 274 → 334 MB. For the real frame
   (3944×4455) that term is 4 × 64 × 17.57 MB ≈ **4.5 GB** — real, but well under the
   substack decode floor, which is why ~25 GB is the projection.

### Wall time

Roughly 1.5× slower on the 512³ test (0.86s → 1.27s), from compression and chunk I/O.
Against this, AIOD-341 removes the 204-second `_encode_binary` rescan plateau from the real
run, so net wall time on the cluster should improve, not regress.
