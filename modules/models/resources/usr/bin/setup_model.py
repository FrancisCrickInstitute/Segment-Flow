import argparse
import json
from pathlib import Path
from aiod_registry import load_manifests
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


def main(model_name: str, model_version: str, model_task: str):
    manifests = load_manifests(filter_access=False)
    versions = manifests[model_name].versions
    # model_version arrives sanitised from Nextflow (e.g. "MitoNet-v1"); resolve to manifest key
    resolved_version = model_version.replace("-", " ")

    # Extract required data from the manifest
    model_info = versions[resolved_version].tasks[model_task]
    model_location = model_info.location
    model_config_location = getattr(model_info, "config_path", None)
    model_finetuning_location = getattr(model_info, "finetuning_path", None)
    model_location_type = get_location_type(model_location)

    # Derive the canonical checkpoint filename (version + extension from source)
    ext = artifact_extension(model_location, model_location_type)
    full_model_name = model_version + "_" + model_task + ext

    # Write one metadata JSON per artifact; Nextflow reads these to decide whether
    # to stage from the external cache (storeDir) or run downloadModelData.
    write_meta(
        "model_chkpt_meta.json", full_model_name, model_location, model_location_type
    )

    if model_config_location:
        config_location_type = get_location_type(model_config_location)
        model_config_name = model_version + "_" + model_task + "_config.yml"
        write_meta(
            "model_config_meta.json",
            model_config_name,
            model_config_location,
            config_location_type,
        )

    if model_finetuning_location:
        finetuning_location_type = get_location_type(model_finetuning_location)
        finetuning_ext = artifact_extension(
            model_finetuning_location, finetuning_location_type
        )
        finetuning_name = (
            model_version + "_" + model_task + "_finetuning_meta" + finetuning_ext
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

    args = parser.parse_args()
    main(args.model_name, args.model_version, args.task)
