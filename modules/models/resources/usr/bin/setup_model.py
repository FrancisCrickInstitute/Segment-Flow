import argparse
import os
import json
from pathlib import Path
from aiod_registry import load_manifests
from aiod_registry.utils import generate_default_config, is_accessible, resolve_version
from urllib.parse import urlparse


def get_location_type(location: str) -> str:
    """Determine whether a location is a URL or a local file path."""
    res = urlparse(location)
    if res.scheme in ("http", "https"):
        return "url"
    elif res.scheme in ("file", ""):
        return "file"
    else:
        raise TypeError(f"Cannot determine type (file/url) of location: {location}!")


def artifact_extension(location: str, location_type: str) -> str:
    """Return the file extension for an artifact given its location."""
    if location_type == "url":
        return Path(urlparse(location).path).suffix
    return Path(location).suffix


def write_meta(fname: str, name: str, location: str, loc_type: str) -> None:
    """Write an artifact metadata JSON file for consumption by Nextflow."""
    with open(fname, "w") as f:
        json.dump({"name": name, "location": location, "type": loc_type}, f)
    print(f"Written metadata for '{name}' -> {fname}")


def check_access(location: str, loc_type: str) -> bool:
    if loc_type == "file":
        return os.access(location, os.F_OK | os.R_OK)
    return True


def config_stem(source: str) -> str:
    """Return the stem of a config source filename (works for both paths and URLs).

    Appended to config artifact filenames so that storeDir's existence check
    becomes source-specific: changing params.model_config to a file with a
    different name produces a different artifact filename and therefore a cache
    miss, forcing the new config to be fetched.
    Checkpoints are intentionally excluded from this scheme — they are immutable
    and should always be served from the storeDir cache.
    """
    parsed = urlparse(source)
    # For URLs use the URL path component; for local paths use the source directly
    path_part = parsed.path if parsed.scheme in ("http", "https") else source
    return Path(path_part).stem


def main(model_name: str, model_version: str, model_task: str, user_config: str | None = None):
    # Load full registry without accessibility filtering to allow precise error reporting
    manifests = load_manifests(filter_access=False)

    # User input check: model name
    if model_name not in manifests:
        raise KeyError(
            f"Model '{model_name}' not found in the registry. "
            f"Available models: {list(manifests.keys())}"
        )

    versions = manifests[model_name].versions
    # User input check: version and task (accepts exact name or slug)
    try:
        model_version = resolve_version(manifests[model_name], model_version)
        model_info = versions[model_version].tasks[model_task]
    except KeyError as e:
        raise KeyError(
            f"Model version '{model_version}' with task '{model_task}' not found in the registry! "
            f"Model version must be one of {list(versions.keys())}"
        ) from e
    # Use the slug for all filesystem/metadata names to avoid spaces
    model_version_slug = versions[model_version].slug

    # Environment check: at least one location must be accessible
    accessible_entry = next(
        (entry for entry in model_info.locations if is_accessible(entry.location)),
        None,
    )
    if accessible_entry is None:
        raise PermissionError(
            f"Model '{model_name}' / '{model_version}' / '{model_task}' exists in the registry "
            f"but none of its locations are accessible on this machine:\n"
            + "\n".join(f"  {e.location}" for e in model_info.locations)
        )

    # Extract required data from the accessible entry
    model_location = accessible_entry.location
    model_location_type = get_location_type(model_location)
    model_config_location = accessible_entry.config_path
    # Derive the canonical checkpoint filename (version + extension from source)
    ext = artifact_extension(model_location, model_location_type)
    full_model_name = model_version_slug + "_" + model_task + ext

    # Write one metadata JSON per artifact; Nextflow reads these to decide whether
    # to stage from the external cache (storeDir) or run downloadModelData.
    write_meta(
        "model_chkpt_meta.json", full_model_name, model_location, model_location_type
    )

    if user_config:
        # Route 1: user-supplied config path or URL
        config_location_type = get_location_type(user_config)
        if not check_access(user_config, config_location_type):
            raise FileNotFoundError(
                f"User-supplied config is not accessible: {user_config}"
            )
        model_config_name = model_version_slug + "_" + model_task + f"_config_{config_stem(user_config)}.yml"
        write_meta(
            "model_config_meta.json", model_config_name, user_config, config_location_type
        )
        print(f"Using user-supplied config: {user_config}")
    elif model_info.params:
        # Route 2: generate default config from registry params — content is deterministic
        # for a given model version/task, so no source tag is needed.
        default_yaml = generate_default_config(manifests[model_name], model_version, model_task)
        model_config_name = model_version_slug + "_" + model_task + "_config.yml"
        config_abs_path = Path.cwd() / model_config_name
        config_abs_path.write_text(default_yaml, encoding="utf-8")
        write_meta(
            "model_config_meta.json", model_config_name, str(config_abs_path), "file"
        )
        print(f"Generated default config from registry params -> {config_abs_path}")
    elif model_config_location:
        # Route 3: model's registry config_path (may be a local path not available to all users)
        config_location_type = get_location_type(model_config_location)
        if not check_access(model_config_location, config_location_type):
            raise PermissionError(
                f"Config required but not accessible via any route:\n"
                f"  Route 1 (user-supplied): not provided\n"
                f"  Route 2 (registry default): no params defined for this model\n"
                f"  Route 3 (registry config_path): '{model_config_location}' is not accessible"
            )
        model_config_name = model_version_slug + "_" + model_task + f"_config_{config_stem(model_config_location)}.yml"
        write_meta(
            "model_config_meta.json", model_config_name, model_config_location, config_location_type
        )
        print(f"Using registry config_path: {model_config_location}")
    else:
        print("No config available via any route — model likely requires no config file.")

    model_finetuning_location = getattr(model_info, "finetuning_path", None)
    if model_finetuning_location:
        finetuning_location_type = get_location_type(model_finetuning_location)
        if not check_access(model_finetuning_location, finetuning_location_type):
            raise PermissionError(
                f"Finetuning artifact is not accessible: {model_finetuning_location} (type: {finetuning_location_type})"
            )
        finetuning_ext = artifact_extension(
            model_finetuning_location, finetuning_location_type
        )
        finetuning_name = (
            model_version_slug + "_" + model_task + "_finetuning_meta" + finetuning_ext
        )
        write_meta(
            "model_finetuning_meta.json",
            finetuning_name,
            model_finetuning_location,
            finetuning_location_type,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Query the AIoD model registry and write JSON metadata files for each "
            "model artifact (checkpoint, config, finetuning). Downloading is handled "
            "separately so that Nextflow can cache results via storeDir."
        )
    )
    parser.add_argument(
        "--model_name", required=True, type=str, help="The name of the model"
    )
    parser.add_argument(
        "--model_version", required=True, type=str, help="The version of the model"
    )
    parser.add_argument(
        "--task", required=True, type=str, help="The task the model will be used for"
    )
    parser.add_argument(
        "--user-config",
        required=False,
        default=None,
        type=str,
        help="User-supplied config path or URL. If omitted, a default config is resolved from the registry.",
    )

    args = parser.parse_args()
    main(args.model_name, args.model_version, args.task, args.user_config)
