from pathlib import Path

import numpy as np
import torch


def save_masks(
    save_dir, save_name, masks, stack_slice=False, all=False, idx=None
):
    save_dir.mkdir(parents=True, exist_ok=True)
    # Cannot save a slice of a stack and all slice(s)
    assert not (stack_slice and all)
    # Incrementally save the masks of a slice from a stack
    if stack_slice:
        save_path = save_dir / f"{save_name}_{idx}.npy"
        # Remove file for previous mask iteration
        if idx > 0:
            (save_dir / f"{save_name}_{idx-1}.npy").unlink()
    # Specify path for img or all slices, indicating finished
    if all:
        save_path = save_dir / f"{save_name}_all.npy"
        # Remove any previous files
        for f in save_dir.glob(f"{save_name}_*.npy"):
            f.unlink()
    # Save the complete masks!
    np.save(save_path, masks)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
