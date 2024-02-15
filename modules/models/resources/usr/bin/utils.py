import argparse
from typing import Optional

import numpy as np
import skimage.io
from skimage.segmentation import relabel_sequential


def save_masks(save_dir, save_name, masks, curr_idx: int, start_idx: int):
    save_dir.mkdir(parents=True, exist_ok=True)
    # Define path, where start_idx + curr_idx is the most recent slice to save
    save_path = save_dir / f"{save_name}_{curr_idx}_{start_idx}.npy"
    # Relabel the inputs to minimise int size and thus output file size
    masks, _, _ = relabel_sequential(masks)
    # Reduce dtype to save space
    masks = reduce_dtype(masks)
    # TODO: Longer-term, use zarr/dask to save to disk
    np.save(save_path, masks)
    # Get path for previous slice
    prev_path = save_dir / f"{save_name}_{curr_idx-1}_{start_idx}.npy"
    # Remove previous save if it exists
    if prev_path.is_file():
        prev_path.unlink()


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
    # TODO: Use zarr/dask to load from disk, or something else
    img = skimage.io.imread(img_path)
    # TODO: Ensure first dim is slices? Difficult to generalise...

    # No slicing to be done if not a stack
    if img.ndim == 2:
        return img
    else:
        return img[start_idx:end_idx, ...]


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
    return arr.astype(best_dtype)


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
