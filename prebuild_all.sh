#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# prebuild_all.sh — Pre-build all Segment-Flow conda environments
# ---------------------------------------------------------------------------
# Discovers available models by scanning run_*.py scripts (mirrors main.nf
# logic) and calls prebuild.nf once per model so that Nextflow populates the
# shared conda cacheDir.  The two fixed environments (combine_stacks,
# setup_model) are built on the first iteration and are cache-hits thereafter.
#
# Usage:
#   ./prebuild_all.sh [PROFILE]
#
# Arguments:
#   PROFILE   Nextflow profile to use (default: crick)
#
# Example:
#   ./prebuild_all.sh crick
# ---------------------------------------------------------------------------
set -euo pipefail

PROFILE="${1:-crick}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/modules/models/resources/usr/bin"

# Discover available models — same logic as main.nf
models=()
while IFS= read -r model; do
    models+=("$model")
done < <(
    for f in "${BIN_DIR}"/run_*.py; do
        basename "$f" .py | sed 's/^run_//'
    done | sort
)

if [[ ${#models[@]} -eq 0 ]]; then
    echo "ERROR: No run_*.py scripts found in ${BIN_DIR}" >&2
    exit 1
fi

echo "=============================================="
echo "  Segment-Flow conda environment prebuild"
echo "=============================================="
echo "Profile : ${PROFILE}"
echo "Models  : ${models[*]}"
echo ""

for model in "${models[@]}"; do
    echo "--- Building envs for model: ${model} ---"
    nextflow run "${SCRIPT_DIR}/prebuild.nf" \
        -profile "${PROFILE}" \
        --model  "${model}"
    echo ""
done

echo "=============================================="
echo "  All conda environments built successfully."
echo "=============================================="
