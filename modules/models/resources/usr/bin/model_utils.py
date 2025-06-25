from typing import Optional

import torch


def get_device(model_type: Optional[str] = None, verbose: bool = False) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type is not None:
        print(f"Using device: {device} for model type: {model_type}")
    else:
        print(f"Using device: {device}")
    if verbose:
        print(f"Device type: {device.type}")
        if device.type == "cuda":
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(device)}")
            print(f"CUDA device properties: {torch.cuda.get_device_properties(device)}")
    return device
