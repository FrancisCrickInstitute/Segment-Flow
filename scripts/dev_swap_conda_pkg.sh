#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# dev_swap_conda_pkg.sh — temporarily point a Nextflow-cached conda env at a
# local editable checkout of one of its pinned pip packages (e.g. aiod_utils),
# then guarantee it's put back exactly as it was.
#
# Nextflow's conda envs are cached by content-hash of their .yml file and
# reused across runs (conda.cacheDir, e.g. ~/.nextflow/aiod/conda/env-<hash>).
# Editing one in place to test a local branch is useful, but leaving it
# swapped breaks that hash's promise of being the exact env its .yml
# describes - every future run against that cache silently uses the wrong
# code. This script makes the swap-and-restore a single guaranteed unit
# instead of two manual steps that are easy to forget the second half of.
#
# Usage:
#   dev_swap_conda_pkg.sh list <package> [--cache-dir DIR]
#   dev_swap_conda_pkg.sh run --env DIR --package NAME --local-path PATH -- CMD...
#   dev_swap_conda_pkg.sh swap --env DIR --package NAME --local-path PATH
#   dev_swap_conda_pkg.sh restore --env DIR --package NAME
#   dev_swap_conda_pkg.sh audit <package> [--cache-dir DIR] [--repo-root DIR]
#   dev_swap_conda_pkg.sh fix-drift <package> [--yes] [--cache-dir DIR] [--repo-root DIR]
#
# `run` is the intended entry point for testing a local branch: it swaps,
# runs CMD, and always restores (even if CMD fails, or the script is
# interrupted). `swap`/`restore` exist for cases where the test itself isn't
# a single command (e.g. exploring interactively) - if you use them
# directly, YOU are responsible for calling `restore` before you're done.
# Prefer `run`.
#
# `audit`/`fix-drift` are a separate concern: they don't assume anything was
# swapped via this script - for ONE given package, they compare every cached
# env's actually installed version against what modules/*/envs/*.yml
# declares it should be, to catch drift left over from swaps done before
# this script existed (or done by hand, bypassing it). They take a package
# by design, not "every pinned dependency" - packages like torch/numba have
# legitimately different pins between the cuda/ and generic/ env variants
# (mutually exclusive, never expected to match), so a repo-wide sweep would
# just be noise. This is really only meaningful for our own packages
# (aiod_utils, aiod_registry), which are pinned identically everywhere.
# `fix-drift` defaults to a dry run; pass --yes to actually change anything.
#
# Example:
#   ./scripts/dev_swap_conda_pkg.sh list aiod_utils
#   ./scripts/dev_swap_conda_pkg.sh run \
#     --env ~/.nextflow/aiod/conda/env-747a7b9... \
#     --package aiod_utils \
#     --local-path ~/Documents/ai_ondemand/aiod_utils \
#     -- nextflow run test_preprocess.nf -profile local
#   ./scripts/dev_swap_conda_pkg.sh audit aiod_utils
#   ./scripts/dev_swap_conda_pkg.sh fix-drift aiod_utils --yes
# ---------------------------------------------------------------------------
set -euo pipefail

DEFAULT_CACHE_DIR="${NXF_CONDA_CACHE_DIR:-$HOME/.nextflow/aiod/conda}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    sed -n '2,50p' "$0"
    exit "${1:-1}"
}

find_site_packages() {
    local env_dir="$1"
    local sp
    sp=$(find "$env_dir/lib" -maxdepth 1 -type d -name 'python3.*' 2>/dev/null | head -1)
    [ -n "$sp" ] || die "No python3.* dir found under $env_dir/lib - is this a real conda env?"
    echo "$sp/site-packages"
}

# Unique top-level entries (package dir, dist-info dir, .pth files, etc.)
# that `pip show -f` reports as belonging to a package. These are exactly
# the units we move to/from the backup - not individual files - so the
# swap/restore is atomic per top-level path.
pkg_top_level_paths() {
    local py="$1" package="$2"
    "$py" -m pip show -f "$package" 2>/dev/null \
        | sed -n '/^Files:/,$p' | tail -n +2 \
        | sed 's/^ *//' \
        | cut -d/ -f1 \
        | sort -u
}

find_repo_root() {
    local script_dir
    script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    (cd "$script_dir/.." && pwd)
}

# Emits "<package> <version>" lines (one per distinct pin found), by
# scanning every `- <package>==<version>` pip entry under modules/*/envs
# .yml files - the source of truth for what SHOULD be installed. A package
# pinned to more than one version across different .yml files emits more
# than one line for it here; callers must treat that as ambiguous.
find_pinned_versions() {
    local repo_root="$1"
    grep -rhoE '^[[:space:]]*-[[:space:]]*[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+[[:space:]]*$' \
        "$repo_root"/modules/*/envs 2>/dev/null \
        | sed -E 's/^[[:space:]]*-[[:space:]]*//; s/[[:space:]]*$//; s/==/ /' \
        | sort -u
}

# Distinct pinned versions for one package - normally exactly one line;
# more than one means the .yml files themselves disagree.
pinned_versions_for_package() {
    local repo_root="$1" package="$2"
    find_pinned_versions "$repo_root" | awk -v p="$package" '$1==p {print $2}' | sort -u
}

cmd_list() {
    local package="${1:?package name required}"
    shift
    local cache_dir="$DEFAULT_CACHE_DIR"
    while [ $# -gt 0 ]; do
        case "$1" in
        --cache-dir)
            cache_dir="$2"
            shift 2
            ;;
        *) die "Unknown argument: $1" ;;
        esac
    done
    [ -d "$cache_dir" ] || die "Cache dir not found: $cache_dir"

    echo "Searching $cache_dir for envs with '$package' installed..."
    local found=0
    for env_dir in "$cache_dir"/*/; do
        env_dir="${env_dir%/}"
        local py="$env_dir/bin/python3"
        [ -x "$py" ] || continue
        local info
        info=$("$py" -m pip show "$package" 2>/dev/null) || continue
        found=1
        local version location
        version=$(echo "$info" | sed -n 's/^Version: //p')
        location=$(echo "$info" | sed -n 's/^Location: //p')
        local swap_marker="$env_dir/.dev_swap/${package}.manifest"
        local state="pinned"
        [ -f "$swap_marker" ] && state="SWAPPED (see $swap_marker)"
        printf '  %s\n      version=%s state=%s\n' "$env_dir" "$version" "$state"
    done
    [ "$found" -eq 1 ] || echo "  (no matching envs found)"
}

cmd_swap() {
    local env_dir="" package="" local_path=""
    while [ $# -gt 0 ]; do
        case "$1" in
        --env) env_dir="$2"; shift 2 ;;
        --package) package="$2"; shift 2 ;;
        --local-path) local_path="$2"; shift 2 ;;
        *) die "Unknown argument: $1" ;;
        esac
    done
    [ -n "$env_dir" ] && [ -n "$package" ] && [ -n "$local_path" ] \
        || die "swap requires --env, --package and --local-path"
    [ -d "$env_dir" ] || die "Env dir not found: $env_dir"
    [ -d "$local_path" ] || die "Local path not found: $local_path"

    local py="$env_dir/bin/python3"
    [ -x "$py" ] || die "No python3 found in $env_dir/bin"

    local backup_dir="$env_dir/.dev_swap"
    local manifest="$backup_dir/${package}.manifest"
    [ -f "$manifest" ] && die "Already swapped (manifest exists: $manifest). Run 'restore' first."

    local site_packages
    site_packages=$(find_site_packages "$env_dir")

    local top_level_paths
    top_level_paths=$(pkg_top_level_paths "$py" "$package")
    [ -n "$top_level_paths" ] || die "Could not find '$package' installed in $env_dir - nothing to back up."

    mkdir -p "$backup_dir/files"
    # Write the manifest BEFORE moving anything: if we're interrupted mid-move
    # or mid-install below, restore can still find it and knows what's safe
    # to put back (its own loop skips any entry that never actually got
    # backed up, so a partial manifest is safe to act on).
    {
        echo "package=$package"
        echo "env_dir=$env_dir"
        echo "local_path=$local_path"
        echo "site_packages=$site_packages"
        echo "top_level_paths=$(echo "$top_level_paths" | tr '\n' ',')"
    } >"$manifest"

    echo "Backing up current install of '$package' from $site_packages:"
    while IFS= read -r entry; do
        [ -n "$entry" ] || continue
        [ -e "$site_packages/$entry" ] || continue
        echo "  $entry"
        mv "$site_packages/$entry" "$backup_dir/files/$entry"
    done <<<"$top_level_paths"

    echo "Installing local editable checkout: $local_path"
    "$py" -m pip install --no-deps --disable-pip-version-check -e "$local_path"

    echo
    echo "SWAPPED: $env_dir now uses local '$package' from $local_path"
    echo "This env's conda-cache hash no longer matches its .yml until restored."
    echo "Run: $0 restore --env '$env_dir' --package '$package'"
}

cmd_restore() {
    local env_dir="" package=""
    while [ $# -gt 0 ]; do
        case "$1" in
        --env) env_dir="$2"; shift 2 ;;
        --package) package="$2"; shift 2 ;;
        *) die "Unknown argument: $1" ;;
        esac
    done
    [ -n "$env_dir" ] && [ -n "$package" ] || die "restore requires --env and --package"

    # Ignore further INT/TERM once restore itself is underway (e.g. an
    # impatient second Ctrl-C) - this is the one section that must run to
    # completion rather than be interrupted again mid-way.
    trap '' INT TERM

    local backup_dir="$env_dir/.dev_swap"
    local manifest="$backup_dir/${package}.manifest"
    [ -f "$manifest" ] || die "No swap recorded for '$package' in $env_dir (nothing to restore)."

    local site_packages
    site_packages=$(sed -n 's/^site_packages=//p' "$manifest")
    local top_level_paths
    top_level_paths=$(sed -n 's/^top_level_paths=//p' "$manifest" | tr ',' '\n')

    local py="$env_dir/bin/python3"
    echo "Removing local editable install of '$package'..."
    "$py" -m pip uninstall -y --disable-pip-version-check "$package" >/dev/null 2>&1 || true

    echo "Restoring backed-up files to $site_packages:"
    while IFS= read -r entry; do
        [ -n "$entry" ] || continue
        [ -e "$backup_dir/files/$entry" ] || continue
        echo "  $entry"
        rm -rf "$site_packages/$entry"
        mv "$backup_dir/files/$entry" "$site_packages/$entry"
    done <<<"$top_level_paths"

    rm -rf "$backup_dir"

    local restored_version
    restored_version=$("$py" -m pip show "$package" 2>/dev/null | sed -n 's/^Version: //p')
    [ -n "$restored_version" ] || die "Restore left '$package' NOT INSTALLED in $env_dir - env is broken, investigate before reusing it."
    echo "RESTORED: '$package' back to version $restored_version in $env_dir"
}

cmd_audit() {
    local package="${1:?package name required}"
    shift
    local cache_dir="$DEFAULT_CACHE_DIR" repo_root
    repo_root=$(find_repo_root)
    while [ $# -gt 0 ]; do
        case "$1" in
        --cache-dir) cache_dir="$2"; shift 2 ;;
        --repo-root) repo_root="$2"; shift 2 ;;
        *) die "Unknown argument: $1" ;;
        esac
    done
    [ -d "$cache_dir" ] || die "Cache dir not found: $cache_dir"

    local versions
    versions=$(pinned_versions_for_package "$repo_root" "$package")
    [ -n "$versions" ] || die "No '- $package==version' pip pin found under $repo_root/modules/*/envs"
    if [ "$(grep -c . <<<"$versions")" -gt 1 ]; then
        die "'$package' has INCONSISTENT pins across .yml files ($(tr '\n' ' ' <<<"$versions")) - not meaningful to audit until the .yml files agree on one version."
    fi
    echo "'$package' is pinned to $versions under $repo_root/modules/*/envs"

    echo
    echo "Scanning $cache_dir for drift..."
    local drift_found=0 env_dir py info version manifest flags
    for env_dir in "$cache_dir"/*/; do
        env_dir="${env_dir%/}"
        py="$env_dir/bin/python3"
        [ -x "$py" ] || continue
        info=$("$py" -m pip show "$package" 2>/dev/null) || continue
        version=$(sed -n 's/^Version: //p' <<<"$info")
        manifest="$env_dir/.dev_swap/${package}.manifest"
        flags=""
        [ -f "$manifest" ] && flags+="STRAY SWAP MANIFEST ($manifest); "
        [ "$version" != "$versions" ] && flags+="installed=$version pinned=$versions"
        if [ -n "$flags" ]; then
            drift_found=1
            echo "  $env_dir: $flags"
        fi
    done
    [ "$drift_found" -eq 1 ] || echo "  (no drift found - every env with '$package' matches its pin)"
}

cmd_fix_drift() {
    local package="${1:?package name required}"
    shift
    local cache_dir="$DEFAULT_CACHE_DIR" repo_root apply=0
    repo_root=$(find_repo_root)
    while [ $# -gt 0 ]; do
        case "$1" in
        --cache-dir) cache_dir="$2"; shift 2 ;;
        --repo-root) repo_root="$2"; shift 2 ;;
        --yes) apply=1; shift ;;
        *) die "Unknown argument: $1" ;;
        esac
    done
    [ -d "$cache_dir" ] || die "Cache dir not found: $cache_dir"

    local versions
    versions=$(pinned_versions_for_package "$repo_root" "$package")
    [ -n "$versions" ] || die "No '- $package==version' pip pin found under $repo_root/modules/*/envs"
    [ "$(grep -c . <<<"$versions")" -eq 1 ] \
        || die "'$package' has INCONSISTENT pins across .yml files ($(tr '\n' ' ' <<<"$versions")) - fix the .yml files first."
    local target_version="$versions"

    [ "$apply" -eq 1 ] || echo "DRY RUN - pass --yes to actually make changes."
    echo

    local env_dir py info version manifest changed=0
    for env_dir in "$cache_dir"/*/; do
        env_dir="${env_dir%/}"
        py="$env_dir/bin/python3"
        [ -x "$py" ] || continue
        info=$("$py" -m pip show "$package" 2>/dev/null) || continue
        version=$(sed -n 's/^Version: //p' <<<"$info")
        manifest="$env_dir/.dev_swap/${package}.manifest"

        if [ -f "$manifest" ]; then
            changed=1
            echo "$env_dir: stray swap manifest found -> restore from its exact backup"
            [ "$apply" -eq 1 ] && cmd_restore --env "$env_dir" --package "$package"
        elif [ "$version" != "$target_version" ]; then
            changed=1
            echo "$env_dir: $version -> $target_version (force reinstall from pin)"
            if [ "$apply" -eq 1 ]; then
                "$py" -m pip install --no-deps --force-reinstall --disable-pip-version-check \
                    "${package}==${target_version}"
            fi
        fi
    done

    echo
    if [ "$changed" -eq 0 ]; then
        echo "Nothing to do - every env already matches its pin."
    elif [ "$apply" -eq 1 ]; then
        echo "Done. Run 'audit $package' to confirm."
    else
        echo "Dry run complete. Re-run with --yes to apply the above."
    fi
}

cmd_run() {
    # Deliberately NOT `local`: the EXIT trap below is a string evaluated
    # after this function returns, by which point any `local` var here
    # would already be out of scope (set -u would fail on it as unbound).
    env_dir="" package="" local_path=""
    local args=("$@")
    local cmd=()
    local i=0
    while [ $i -lt ${#args[@]} ]; do
        case "${args[$i]}" in
        --env) env_dir="${args[$((i + 1))]}"; i=$((i + 2)) ;;
        --package) package="${args[$((i + 1))]}"; i=$((i + 2)) ;;
        --local-path) local_path="${args[$((i + 1))]}"; i=$((i + 2)) ;;
        --)
            i=$((i + 1))
            cmd=("${args[@]:$i}")
            break
            ;;
        *) die "Unknown argument: ${args[$i]}" ;;
        esac
    done
    [ -n "$env_dir" ] && [ -n "$package" ] && [ -n "$local_path" ] \
        || die "run requires --env, --package and --local-path"
    [ ${#cmd[@]} -gt 0 ] || die "run requires a command after '--'"

    # Traps must be armed BEFORE cmd_swap runs, not after: cmd_swap's own
    # `pip install -e` takes real time, and a signal arriving during it
    # (rather than during CMD below) would otherwise have no handler yet.
    # EXIT fires exactly once at actual process exit and bash preserves the
    # exit code that triggered it, so this restores whether cmd_swap/CMD
    # succeeds, fails (set -e kills the script), or is interrupted.
    trap safe_restore_trap EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    cmd_swap --env "$env_dir" --package "$package" --local-path "$local_path"

    echo
    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
}

# Trap target: only restores if a swap actually got far enough to record a
# manifest - an interrupt before/during arg parsing has nothing to undo.
safe_restore_trap() {
    local manifest="${env_dir:-}/.dev_swap/${package:-}.manifest"
    if [ -n "${env_dir:-}" ] && [ -n "${package:-}" ] && [ -f "$manifest" ]; then
        echo
        echo "Restoring '$package' in $env_dir..."
        cmd_restore --env "$env_dir" --package "$package"
    fi
}

main() {
    [ $# -ge 1 ] || usage
    local sub="$1"
    shift
    case "$sub" in
    list) cmd_list "$@" ;;
    swap) cmd_swap "$@" ;;
    restore) cmd_restore "$@" ;;
    run) cmd_run "$@" ;;
    audit) cmd_audit "$@" ;;
    fix-drift) cmd_fix_drift "$@" ;;
    -h | --help) usage 0 ;;
    *) usage ;;
    esac
}

main "$@"
