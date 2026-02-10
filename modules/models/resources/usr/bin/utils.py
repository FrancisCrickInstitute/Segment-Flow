import argparse
import inspect
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from skimage.segmentation import relabel_sequential

import aiod_utils.io as aiod_io
import aiod_utils.rle as aiod_rle


def save_masks(
    save_dir, save_name, masks, idxs: list[int, ...], metadata: dict = {}, **kwargs
):
    save_dir.mkdir(parents=True, exist_ok=True)
    # Extract the start and end indices in each dim
    start_x, end_x, start_y, end_y, start_z, end_z = idxs
    # Define path with all the indices
    save_path = f"{save_name}_x{start_x}-{end_x}_y{start_y}-{end_y}_z{start_z}-{end_z}"
    # Relabel the inputs to minimise int size and thus output file size
    masks, _, _ = relabel_sequential(masks)
    # Reduce dtype to save space
    masks = aiod_io.reduce_dtype(masks)
    # Encode the masks and save them (inserting metadata if provided)
    # NOTE: kwargs is there to allow specifying mask encoding type — otherwise inferred from masks
    encoded_masks = aiod_rle.encode(
        masks,
        metadata=metadata,
        **kwargs,
    )
    aiod_rle.save_encoding(rle=encoded_masks, fpath=str(save_path) + ".rle")
    # TODO: For what use is returning the save path?
    return save_path


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
        "--channels", type=int, help="Number of channels in the input image"
    )
    parser.add_argument(
        "--num-slices", type=int, help="Number of Z slices in the input image"
    )

    return parser


def guess_rgb(img_shape, dim: int = 0):
    # Unified load func aims for [CD]HW format, so check for RGB(A) in first dim
    ndim = len(img_shape)
    channel_dim = img_shape[dim]
    return ndim > 2 and channel_dim in (3, 4)


def load_img(
    fpath,
    idxs: list[int, ...],
    channels: Optional[int] = None,
    num_slices: Optional[int] = None,
    **kwargs,
):
    # Caller should specify desired dimension ordering (model dependent)
    dim_order = kwargs.pop("dim_order", "CZYX")
    # TODO: Better to return Dask and index as needed?
    # FIXME: here rbg converted to channels; napari treats rbg separately
    img = aiod_io.load_image_data(fpath, dim_order=dim_order, rgb_as_channels=True, **kwargs)
    # Extract the start and end indices in each dim
    start_x, end_x, start_y, end_y, start_z, end_z = idxs
    # Validate array shape against expected channels and slices
    img = validate_dims(img, dim_order, channels, num_slices)
    assert img.ndim == len(dim_order), (
        "Something has gone wrong with the image dimensions!"
    )
    slices = {
        "C": np.s_[:],
        "Z": np.s_[start_z:end_z],
        "X": np.s_[start_x:end_x],
        "Y": np.s_[start_y:end_y],
    }
    # Slice the image based on the given indices
    return img[tuple(slices[dim] for dim in dim_order)]


def validate_dims(
    img,
    dim_order: str,
    channels: Optional[int] = 1,
    num_slices: Optional[int] = 1,
):
    """
    Validate and potentially fix dimension order of image array.

    Raises ValueError if dimensions don't match and can't be obviously swapped.
    """

    errmsg = f"Image shape { ({d: img.shape[i] for i, d in enumerate(dim_order)}) } does not match expected channels and slices {channels, num_slices}."

    has_c, has_z = "C" in dim_order, "Z" in dim_order

    # 2D
    if not has_c and not has_z:
        return img
    # 3D validate
    if has_c ^ has_z:
        dimsize = img.shape[dim_order.index("C" if has_c else "Z")]
        expected = channels if has_c else num_slices
        if dimsize != expected:
            raise ValueError(errmsg)
        return img

    # 4D case - both C and Z
    c_idx, z_idx = dim_order.index("C"), dim_order.index("Z")
    size_c, size_z = img.shape[c_idx], img.shape[z_idx]

    if size_c == channels and size_z == num_slices:
        return img

    if size_c == num_slices and size_z == channels and channels != num_slices:
        warnings.warn(
            f"Swapping C and Z: detected C={size_c}, Z={size_z} but expected C={channels}, Z={num_slices}"
        )
        return np.swapaxes(img, c_idx, z_idx)
    raise ValueError(errmsg)



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


def get_mask_type_from_model(model_type: str) -> str:
    instance_models = ["sam", "cellpose", "cellposesam"]
    for model in instance_models:
        if model in model_type:
            return "instance"
    else:
        return "binary"


def get_model_name_type(model_type: str) -> str:
    """
    Get the model name from the script that called his function and model type.

    model_type: str
        The variant/version of the model.
    """
    # Get the file name of the script that called this function
    # NOTE: This should generally work, though inspecting with e.g. pdb will change the results
    f = inspect.stack()[1].filename
    # Strip the model name from the file path
    # NOTE: Stick to current run_<MODEL_NAME> convention for future script types
    model_name = Path(f).stem.split("_")[-1]
    # Add the model type to the name
    return f"{model_name}_{model_type}"
