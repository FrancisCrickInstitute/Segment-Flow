---
name: dev-swap-conda-pkg
description: Test a local dev branch of a pinned pip dependency (aiod_utils, aiod_registry) against Segment-Flow's Nextflow-cached conda envs, with a guaranteed restore - and audit/fix those same envs when they've drifted from their declared pin. Use whenever you'd otherwise be tempted to `pip install -e` directly into a `~/.nextflow/aiod/conda/env-*` directory, or to hand-edit files inside one, to verify a cross-repo change before its dependency is released/pinned-bumped. Also use when checking whether cached envs actually match what their `.yml` says, or fixing them if not.
---

# Swapping a local dev branch into a cached conda env

Segment-Flow's conda envs are cached by content-hash of their `.yml` file
(`conda.cacheDir`, e.g. `~/.nextflow/aiod/conda/env-<hash>`) and reused across
every future run against that cache. That reuse is exactly what makes it
tempting to edit one in place - install a local branch of `aiod_utils` or
`aiod_registry` there to test a cross-repo change before it's released and
the pin bumped - but leaving it swapped silently breaks the cache's whole
premise: every future run against that hash quietly uses the wrong code,
and nobody will notice until something inexplicable breaks much later.

**Never hand-edit or `pip install -e` directly into a cached env.** Always go
through `scripts/dev_swap_conda_pkg.sh`, which makes the swap-and-restore one
guaranteed unit instead of two manual steps whose second half is easy to
forget - it backs up the current install, swaps in the local checkout, and
restores from that exact backup on ANY exit path (success, failure, or
Ctrl-C/SIGTERM), verified by testing interrupts at multiple points including
mid-`pip install`.

## Usage

Find which cached env(s) currently have the package installed:

```
./scripts/dev_swap_conda_pkg.sh list aiod_utils
```

Run the actual test - this is the entry point to use almost always:

```
./scripts/dev_swap_conda_pkg.sh run \
  --env ~/.nextflow/aiod/conda/env-<hash> \
  --package aiod_utils \
  --local-path ~/Documents/ai_ondemand/aiod_utils \
  -- nextflow run test_something.nf -profile local
```

This backs up the env's current install, `pip install -e`s the local
checkout, runs the given command, then restores the exact original files -
whether the command succeeds, fails, or is interrupted. The wrapped
command's own exit code is preserved as the script's exit code.

`swap`/`restore` also exist as separate subcommands for interactive
exploration that isn't a single command. If you use them directly, restore
is on you - prefer `run`.

## After running

Always confirm the env is actually back to its pinned state before moving
on - don't just trust that `run` printed `RESTORED`:

```
./scripts/dev_swap_conda_pkg.sh list aiod_utils
```

A `state=pinned` entry (not `SWAPPED`) for the env you used means it's clean.
If a session ever ends before `restore` could run (e.g. the whole process
was killed with `SIGKILL`, which no trap can catch), `list` will show
`state=SWAPPED (see <manifest>)` - read that manifest and run `restore`
manually with the same `--env`/`--package` before trusting that env again.

## Auditing for drift (a separate, related problem)

Independent of anything swapped through this script, cached envs can drift
from what their `.yml` actually declares - almost always leftover from a
manual swap done before this script existed, or one done by hand bypassing
it. Check for that with:

```
./scripts/dev_swap_conda_pkg.sh audit aiod_utils
```

This reports every cached env whose installed version doesn't match the
version declared in `modules/*/envs/*.yml` (parsed directly from the `- pkg==version`
pip lines - the source of truth), and separately flags any env with a stray
`.dev_swap` manifest sitting in it (a swap that was started but never
`restore`d). Fix what it finds with:

```
./scripts/dev_swap_conda_pkg.sh fix-drift aiod_utils --yes
```

`fix-drift` defaults to a dry run (omit `--yes` to just see the plan first).
For each drifted env it either restores from a stray manifest if one exists
(the exact original files - more faithful than a fresh install), or
force-reinstalls the exact pinned version otherwise.

**Both commands require an explicit package name - they deliberately don't
scan "every pinned dependency."** This is really only meaningful for
packages pinned identically everywhere in the repo, which in practice means
our own packages (`aiod_utils`, `aiod_registry`). Third-party ML deps like
`torch`/`torchvision`/`numba` are pinned separately per `cuda/`/`generic/`
env variant (mutually exclusive, never expected to match each other), so
auditing those isn't meaningful the same way - if their pins ever do
disagree across `.yml` files, both commands detect that and refuse
(`INCONSISTENT pins`) rather than guess which one is "correct" for a given
env. Don't widen either command's scope to cover those without solving that
problem first.
