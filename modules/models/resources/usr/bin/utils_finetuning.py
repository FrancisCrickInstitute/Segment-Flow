import shutil
from pathlib import Path


def save_finetuned_model_artifact(
    source_model_path: str | Path,
    save_dir: str | Path,
    model_name: str,
) -> Path:
    """
    Copy a finetuned model artifact into the save directory.
    """
    source_path = Path(source_model_path)
    destination_path = Path(save_dir) / f"{model_name}.pth"

    if source_path.exists():
        shutil.copy2(source_path, destination_path)

    return destination_path
