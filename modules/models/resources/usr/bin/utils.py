import argparse

import numpy as np
import skimage.io
from skimage.segmentation import relabel_sequential
import torch


def save_masks(save_dir, save_name, masks, curr_idx: int, start_idx: int):
    save_dir.mkdir(parents=True, exist_ok=True)
    # Define path, where start_idx + curr_idx is the most recent slice to save
    save_path = save_dir / f"{save_name}_{curr_idx}_{start_idx}.npy"
    # TODO: Use the max value to determine appropriate dtype to minimize size
    # Relabel the inputs to minimise int size and thus output file size
    masks, _, _ = relabel_sequential(masks)
    # Determine appropriate dtype
    best_dtype = np.result_type(np.min_scalar_type(masks.min()), masks.max())
    masks = masks.astype(best_dtype)
    # TODO: Longer-term, use zarr/dask to save to disk
    np.save(save_path, masks)
    # Get path for previous slice
    prev_path = save_dir / f"{save_name}_{curr_idx-1}_{start_idx}.npy"
    # Remove previous save if it exists
    if prev_path.is_file():
        prev_path.unlink()


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    return device


def create_argparser_inference():
    parser = argparse.ArgumentParser()

    parser.add_argument("--img-path", required=True, help="Path to image")
    parser.add_argument("--mask-fname", required=True, help="Mask save filename")
    parser.add_argument("--output-dir", required=True, help="Mask output directory")
    parser.add_argument("--model-chkpt", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--start-idx", type=int, required=True, help="Start index for stack"
    )
    parser.add_argument(
        "--end-idx", type=int, required=True, help="End index for stack"
    )
    parser.add_argument("--model-type", help="Select model type", default="default")
    parser.add_argument("--model-config", help="Model config path")

    return parser


def guess_rgb(img_shape):
    # https://github.com/napari/napari/blob/26dcda8c2cb545948f01be7949fadf79a2927e91/napari/layers/image/_image_utils.py#L13
    ndim = len(img_shape)
    last_dim = img_shape[-1]
    return ndim > 2 and last_dim in (3, 4)


def load_img(img_path, start_idx: int, end_idx: int):
    img = skimage.io.imread(img_path)

    # No slicing to be done if not a stack
    if img.ndim == 2:
        return img
    else:
        return img[start_idx:end_idx, ...]
