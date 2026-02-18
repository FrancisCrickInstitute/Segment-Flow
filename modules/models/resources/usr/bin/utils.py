import argparse
import inspect
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
    start_x, end_x, start_y, end_y, start_z, end_z = extract_idxs(idxs)
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


def extract_idxs(idxs: list[int, ...]):
    # Standardise expected idxs format and extraction
    start_x, end_x, start_y, end_y, start_z, end_z = idxs
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
    # By default we return array in [CD]HW format, depending on input
    if "dim_order" in kwargs:
        dim_order = kwargs.pop("dim_order")
    else:
        dim_order = "CZYX"
    # TODO: Better to return Dask and index as needed?
    img = aiod_io.load_image(fpath, return_array=True, dim_order=dim_order, **kwargs)
    # Extract the start and end indices in each dim
    start_x, end_x, start_y, end_y, start_z, end_z = extract_idxs(idxs)

    img = transpose_dims(img, dim_order, channels, num_slices)

    slices = [
        np.s_[:],
        np.s_[start_z:end_z],
        np.s_[start_x:end_x],
        np.s_[start_y:end_y],
    ]
    #
    trans_dim_order = dim_order
    # Remove slice objects as and if necessary
    # Keep HW only
    if img.ndim == 2:
        slices = slices[2:]
        trans_dim_order = trans_dim_order.replace("Z", "").replace("C", "")
    elif img.ndim == 3:
        # Keep CHW only if no slices (or 1 slice and channels)
        if (num_slices is None) or (num_slices == 1 and channels is not None):
            slices = [slices[0]] + slices[2:]
            trans_dim_order = trans_dim_order.replace("Z", "")
        # Keep DHW only if no channels
        elif channels is None:
            slices = slices[1:]
            trans_dim_order = trans_dim_order.replace("C", "")

    # Reorder slices based on dim_order
    slices = translate_from_order(slices, trans_dim_order)
    # Slice the image based on the given indices
    img = img[tuple(slices)]

    # Preprocessing currently expects a squeezed image
    # As most of them don't use channel info
    if channels is not None and channels == 1:
        # Get channel axis
        channel_axis = dim_order.index("C")
        img = np.squeeze(img, axis=channel_axis)
        squeezed = True
    else:
        squeezed = False

    if squeezed:
        img = np.expand_dims(img, axis=channel_axis)
    return img


def transpose_dims(
    img,
    dim_order: str,
    channels: Optional[int] = None,
    num_slices: Optional[int] = None,
):
    # TODO: dim_order has already been used in the load img
    # This function is about verifying that has happened, so remove
    # the default expectations below and check using the provided dim_order
    # Nothing to do if no channels or slices provided
    if channels is None and num_slices is None:
        return img
    # If channels and slices are both 1, nothing to do
    if channels == 1 and num_slices == 1:
        return img
    # For 2D, we can just return the image
    if img.ndim == 2:
        return img
    # FIXME: Check if ndim == 3 and which is missing, then remove from dim_order
    if img.ndim == 3:
        # Remove channels if None, or if 1 and has slices
        # NOTE: Could cause issues because not squeezing?
        if channels is None or (
            channels == 1 and num_slices is not None and num_slices > 1
        ):
            dim_order = dim_order.replace("C", "")
        # If slices is 1, then we remove the z axis
        if num_slices is None or (num_slices <= 1 and channels is not None):
            dim_order = dim_order.replace("Z", "")
    # Get shape and containers
    shape = list(img.shape)
    source = []
    dest = []
    if channels is not None:
        channel_axis = dim_order.index("C")
        channel_curr_idx = shape.index(channels)
        if channel_curr_idx != channel_axis:
            source.append(channel_curr_idx)
            dest.append(channel_axis)
    # Expectation is that slices are second dim
    if num_slices is not None and num_slices > 1:
        z_axis = dim_order.index("Z")
        z_curr_idx = shape.index(num_slices)
        if z_curr_idx != z_axis:
            source.append(z_curr_idx)
            dest.append(z_axis)
    # We ignore HW for now, as assumed these are not ever changed
    # Move the axes to the correct positions if necessary
    if source:
        img = np.moveaxis(img, source, dest)
    return img


def translate_from_order(l: list, dim_order: str, default_order: str = "CZYX"):
    # Get the translation from given order to default order
    mapper = {k: v for v, k in enumerate(default_order) if k in dim_order}
    # Ensure consecutive indices
    mapper = {k: i for i, k in enumerate(mapper)}
    # Get the translation from given order to default order
    trans = [mapper[dim] for dim in dim_order]
    return [l[i] for i in trans]


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
