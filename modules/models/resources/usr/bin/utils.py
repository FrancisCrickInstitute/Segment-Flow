import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
from skimage.segmentation import relabel_sequential

import aiod_utils
import aiod_utils.io as aiod_io


def save_masks(save_dir, save_name, masks, idxs: list[int, ...]):
    save_dir.mkdir(parents=True, exist_ok=True)
    # Extract the start and end indices in each dim
    start_x, end_x, start_y, end_y, start_z, end_z = extract_idxs(idxs)
    # Define path with all the indices
    save_path = (
        save_dir
        / f"{save_name}_x{start_x}-{end_x}_y{start_y}-{end_y}_z{start_z}-{end_z}.npy"
    )
    # Relabel the inputs to minimise int size and thus output file size
    masks, _, _ = relabel_sequential(masks)
    # Reduce dtype to save space
    masks = reduce_dtype(masks)
    # TODO: Longer-term, use zarr/dask to save to disk
    np.save(save_path, masks)
    return save_path


def extract_idxs(idxs: list[int, ...]):
    # Standardise expected idxs format and extraction
    start_x, end_x, start_y, end_y, start_z, end_z = idxs
    return start_x, end_x, start_y, end_y, start_z, end_z


def extract_idxs_from_fname(fname: str):
    # Extract the indices from the filename
    idx_ranges = Path(fname).stem.split("_")[-3:]
    start_x, end_x = map(int, idx_ranges[0].split("x")[1].split("-"))
    start_y, end_y = map(int, idx_ranges[1].split("y")[1].split("-"))
    start_z, end_z = map(int, idx_ranges[2].split("z")[1].split("-"))
    return start_x, end_x, start_y, end_y, start_z, end_z


def create_argparser_inference():
    parser = argparse.ArgumentParser()

    parser.add_argument("--img-path", required=True, help="Path to image")
    parser.add_argument("--mask-fname", required=True, help="Mask save filename")
    parser.add_argument("--output-dir", required=True, help="Mask output directory")
    parser.add_argument("--model-chkpt", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--idxs",
        nargs=6,
        type=int,
        required=True,
        help="Start and end indices for stack",
    )
    parser.add_argument("--model-type", help="Select model type", default="default")
    parser.add_argument("--model-config", help="Model config path")
    parser.add_argument(
        "--preprocess-params", help="Preprocessing parameters YAML file"
    )

    return parser


def guess_rgb(img_shape, dim: int = 0):
    # Unified load func aims for [CD]HW format, so check for RGB(A) in first dim
    ndim = len(img_shape)
    channel_dim = img_shape[dim]
    return ndim > 2 and channel_dim in (3, 4)


def load_img(
    fpath, idxs: list[int, ...], preprocess_params: Union[str, Path], **kwargs
):
    # By default we return array in [CD]HW format, depending on input
    # Squeeze by default to remove any singleton dimensions (primarily C=1)
    img = aiod_io.load_image(fpath, return_array=True, **kwargs).squeeze()
    # Extract the start and end indices in each dim
    start_x, end_x, start_y, end_y, start_z, end_z = extract_idxs(idxs)

    # Handle case where no depth, but multiple channels
    if start_z == 0 and end_z == 1:
        depth = False
    else:
        depth = True

    # No slicing to be done if not a stack
    if img.ndim == 2:
        img = img[start_x:end_x, start_y:end_y]
    elif (img.ndim == 3) and depth:
        img = img[start_z:end_z, start_x:end_x, start_y:end_y]
    elif (img.ndim == 3) and not depth:
        # If RGB(A) in last dim (i.e. from skimage), slice accordingly
        if guess_rgb(img.shape, dim=-1):
            img = img[start_x:end_x, start_y:end_y, ...]
        # Otherwise assume channel is first dim
        else:
            img = img[..., start_x:end_x, start_y:end_y]
    else:
        img = img[..., start_z:end_z, start_x:end_x, start_y:end_y]

    # Apply preprocessing if provided
    img = aiod_utils.run_preprocess(img, preprocess_params)
    return img


def reduce_dtype(arr: np.ndarray, max_val: Optional[int] = None):
    # Get the max value in the array if not provided
    if max_val is None:
        max_val = arr.max()
    # Get the appropriate dtype from the max value
    if max_val < 256:
        best_dtype = np.uint8
    elif max_val < 65536:
        best_dtype = np.uint16
    # Surely it doesn't need more than 32 bits...
    else:
        best_dtype = np.uint32
    return arr.astype(best_dtype, copy=False)


def align_segment_labels(all_masks: np.ndarray, threshold: float = 0.5):
    # From https://github.com/MIC-DKFZ/napari-sam/blob/main/src/napari_sam/_widget.py#L1118
    """
    There is a potentially better way to do this, using the Hungarian algorithm
    It will, however, still require computing the "cost" (i.e. overlap, defined as
    the count of co-occurences between every numerical label between two slices)
    The Hungarian algorithm itself can be easily done using scipy.optimize.linear_sum_assignment
    It's just that then the optimal assignment will be found, rather than using this
    thresholded approach. Can revise later as needed.

    TODO: Abstract out into separate nextflow process?
    """
    for i in range(all_masks.shape[0] - 1):
        current_slice = all_masks[i]
        next_slice = all_masks[i + 1]
        next_labels, next_label_counts = np.unique(next_slice, return_counts=True)
        next_label_counts = next_label_counts[next_labels != 0]
        next_labels = next_labels[next_labels != 0]
        new_next_slice = np.zeros_like(next_slice)
        if len(next_labels) > 0:
            for next_label, next_label_count in zip(next_labels, next_label_counts):
                current_roi_labels = current_slice[next_slice == next_label]
                current_roi_labels, current_roi_label_counts = np.unique(
                    current_roi_labels, return_counts=True
                )
                current_roi_label_counts = current_roi_label_counts[
                    current_roi_labels != 0
                ]
                current_roi_labels = current_roi_labels[current_roi_labels != 0]
                if len(current_roi_labels) > 0:
                    current_max_count = np.max(current_roi_label_counts)
                    current_max_count_label = current_roi_labels[
                        np.argmax(current_roi_label_counts)
                    ]
                    overlap = current_max_count / next_label_count
                    if overlap >= threshold:
                        new_next_slice[next_slice == next_label] = (
                            current_max_count_label
                        )
                    else:
                        new_next_slice[next_slice == next_label] = next_label
                else:
                    new_next_slice[next_slice == next_label] = next_label
            all_masks[i + 1] = new_next_slice
    return all_masks
