#!/bin/bash
#SBATCH --job-name=combine_bench
#SBATCH --output=combine_bench_%j.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
# --mem is the cluster's hard cap. Master's dense combine needs a few times the
# volume (huge: 1024x3072x3072 uint16 = 19 GB per copy), so keep it well above
# that, or drop master from --targets for the biggest presets.
#
# Usage:
#   sbatch submit.sbatch                                   # default sweep below
#   sbatch submit.sbatch --presets huge --exclude pp-sam   # args go to sweep.py
#   BENCH="bench_combine.py --preset large --profiler memray" sbatch submit.sbatch

set -euo pipefail

# --- activate the combine_stacks conda env (edit for your site) ---
source /camp/apps/eb/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh
conda activate /flask/conda/ddt/combine_conda-f6816e510d80ad61a43ff5576391ab44

# Slurm runs a copy of this script, so $0 is not in the harness dir: submit from it
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# Inputs are cached across runs; keep the cache on fast shared scratch
export COMBINE_BENCH_CACHE="${COMBINE_BENCH_CACHE:-$PWD/cache}"

if [[ -n "${BENCH:-}" ]]; then
    # shellcheck disable=SC2086
    python -u $BENCH
elif [[ $# -gt 0 ]]; then
    python -u sweep.py "$@"
else
    python -u sweep.py --presets large --targets git:origin/master worktree --exclude pp-sam
fi
