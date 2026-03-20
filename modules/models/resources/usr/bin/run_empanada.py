from pathlib import Path
import platform

from empanada.data import VolumeDataset
from empanada.data.utils import resize_by_factor
from empanada.inference.engines import (
    PanopticDeepLabRenderEngine,
    PanopticDeepLabRenderEngine3d,
)
from empanada.inference.tracker import InstanceTracker
from empanada.inference.patterns import (
    create_matchers,
    forward_matching,
    backward_matching,
    update_trackers,
    finish_tracking,
    create_instance_consensus,
    create_semantic_consensus,
    fill_volume,
    get_axis_trackers_by_class,
)
import numpy as np
import skimage
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import yaml

from utils import save_masks, create_argparser_inference, load_img, get_model_name_type
from model_utils import get_device

# IoU / IoA thresholds used for instance matching across planes
_MERGE_IOU_THR = 0.25
_MERGE_IOA_THR = 0.25
_AXES = {"xy": 0, "xz": 1, "yz": 2}


def normalize(img, mean, std):
    max_pixel_value = np.iinfo(img.dtype).max

    mean = np.array(mean, dtype=np.float32) * max_pixel_value
    std = np.array(std, dtype=np.float32) * max_pixel_value
    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


class Preprocessor:
    # Follows empanada-napari's Preprocessor class
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image=None):
        assert image is not None
        if np.issubdtype(image.dtype, np.floating):
            raise ValueError("Input image cannot be float type!")
        # Normalize image, move channel dim, and convert to tensor
        return {"image": torch.from_numpy(normalize(image, self.mean, self.std)[None])}


def force_connected(pan_seg, thing_list, label_divisor=1000):
    for label in thing_list:
        min_id = label * label_divisor
        max_id = min_id + label_divisor

        instance_seg = pan_seg.copy()
        outside_mask = np.logical_or(pan_seg < min_id, pan_seg >= max_id)
        instance_seg[outside_mask] = 0

        # NOTE: Underneath, empanada uses skimage.measure.label or cc3d's connected_components if avail
        # Just use skimage here to keep deps down
        instance_seg = skimage.measure.label(instance_seg).astype(np.uint16)
        instance_seg[instance_seg > 0] += min_id
        pan_seg[instance_seg > 0] = instance_seg[instance_seg > 0]
    return pan_seg


def infer_2d(engine, img, norms, inference_kwargs):
    # Get the starting/original size
    orig_size = img.shape
    # Downsampling has been requested, so resize accordingly
    inference_scale = inference_kwargs.get("inference_scale", 1)
    if inference_scale != 1:
        img = resize_by_factor(img, scale_factor=inference_scale)
    # Preprocess
    img = Preprocessor(norms["mean"], norms["std"])(image=img)["image"].unsqueeze(0)
    # Run inference
    pan_seg = engine(img, orig_size, upsampling=inference_scale)
    # Postprocess
    pan_seg = force_connected(
        pan_seg.squeeze().detach().cpu().numpy().astype(np.uint16),
        engine.thing_list,
        engine.label_divisor,
    )
    return pan_seg


def run_2d(engine, img, norms, inference_kwargs):
    # First handle special case of [B, C, H, W]
    if img.ndim == 4:
        if img.shape[0] != 1:
            raise ValueError(
                f"Can only handle an image, or stack of images, not {img.shape}!"
            )
        else:
            img = img.squeeze()
    full_mask = np.zeros(img.shape, dtype=np.int32)
    # Stack of slices
    if img.ndim == 3:
        for plane, img_slice in enumerate(img):
            seg = infer_2d(engine, img_slice, norms, inference_kwargs)
            full_mask[plane, ...] = seg
        return full_mask
    # Single slice
    elif img.ndim == 2:
        return infer_2d(engine, img, norms, inference_kwargs)
    else:
        raise ValueError("Can only handle a 2D image, or stack of images!")


def create_trackers(engine, img_shape, axis_name, labels):
    return [
        InstanceTracker(label, engine.label_divisor, img_shape, axis_name)
        for label in labels
    ]


def infer_3d(engine: PanopticDeepLabRenderEngine3d, img, norms, inference_kwargs):
    axis_name = inference_kwargs["inference_plane"].lower()
    axis = _AXES[axis_name]
    dataset = VolumeDataset(
        img,
        axis,
        Preprocessor(norms["mean"], norms["std"]),
        scale=inference_kwargs["inference_scale"],
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )
    trackers = create_trackers(engine, img.shape, axis_name, labels)
    matchers = create_matchers(
        thing_list=engine.thing_list,
        label_divisor=engine.label_divisor,
        merge_iou_thr=_MERGE_IOU_THR,
        merge_ioa_thr=_MERGE_IOA_THR,
    )
    # NOTE: Skip the panoptic_stack/zarr_store stuff for now
    # Safeguard check for MacOS
    if platform.system() == "Darwin":
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    queue = mp.Queue()
    rle_stack = []
    matcher_out, matcher_in = mp.Pipe()
    matcher_proc = mp.Process(
        target=forward_matching,
        args=(
            matchers,
            queue,
            rle_stack,
            matcher_in,
            labels,
            engine.label_divisor,
            engine.thing_list,
        ),
    )
    matcher_proc.start()

    for batch in dataloader:
        image_batch = batch["image"]
        size = batch["size"]
        pan_seg = engine(image_batch, size, inference_kwargs["inference_scale"])

        if pan_seg is None:
            queue.put(None)
            continue
        else:
            queue.put(pan_seg.squeeze().cpu().numpy())

    final_segs = engine.end(inference_kwargs["inference_scale"])
    if final_segs:
        for pan_seg in final_segs:
            queue.put(pan_seg.squeeze().cpu().numpy())
    queue.put("finish")
    rle_stack = matcher_out.recv()[0]
    matcher_proc.join()
    # Now propagate labels backward through stack
    axis_len = img.shape[axis]
    for idx, rle_seg in backward_matching(rle_stack, matchers, axis_len):
        update_trackers(rle_seg, idx, trackers)
    finish_tracking(trackers)

    stack = np.zeros(img.shape, dtype=np.uint32)
    for tracker in trackers:
        fill_volume(stack, tracker.instances)
    engine.reset()
    return stack, trackers


def instance_relabel(tracker):
    """Relabels instances starting from 1
    Directly taken from empanada-napari
    """
    instance_id = 1
    instances = {}
    for instance_attr in tracker.instances.values():
        # vote on indices that should belong to an object
        runs_cat = np.stack([instance_attr["starts"], instance_attr["runs"]], axis=1)

        sort_idx = np.argsort(runs_cat[:, 0], kind="stable")
        runs_cat = runs_cat[sort_idx]

        # TODO: technically this could break the zarr_fill_instances function
        # if an object has a pixel in the bottom right corner of the Nth z slice
        # and a pixel in the top left corner of the N+1th z slice
        # only applies to yz axis
        instances[instance_id] = {}
        instances[instance_id]["box"] = instance_attr["box"]
        instances[instance_id]["starts"] = runs_cat[:, 0]
        instances[instance_id]["runs"] = runs_cat[:, 1]
        instance_id += 1

    return instances


def postprocess_volume(engine, trackers_dict, img_shape, inference_kwargs):
    class_names = inference_kwargs["class_names"]

    for class_id, class_name in class_names.items():
        class_tracker = get_axis_trackers_by_class(trackers_dict, class_id)[0]
        shape3d = class_tracker.shape3d

        stack_tracker = InstanceTracker(
            class_id,
            engine.label_divisor,
            shape3d,
            "xy",  # As this isn't ortho-plane...I think; the authors did it!
        )
        stack_tracker.instances = instance_relabel(class_tracker)

    stack_vol = np.zeros(img_shape, dtype=np.uint16)
    fill_volume(stack_vol, stack_tracker.instances) # pyright: ignore[reportPossiblyUnboundVariable]
    return stack_vol


def consensus_volume(engine, trackers_dict, inference_kwargs):
    thing_list = engine.thing_list
    class_names = inference_kwargs["class_names"]

    shape3d = next(iter(trackers_dict.values()))[0].shape3d
    consensus_vol = np.zeros(shape3d, dtype=np.uint16)

    for class_id, class_name in class_names.items():
        class_trackers = get_axis_trackers_by_class(trackers_dict, class_id)

        if class_id in thing_list:
            # TODO: Check allow_one_view
            consensus_tracker = create_instance_consensus(
                class_trackers,
                inference_kwargs["pixel_vote_thr"],
                cluster_iou_thr=0.75,  # Hard-coded in empanada-napari
                bypass=inference_kwargs["allow_one_view"],
            )
        else:
            consensus_tracker = create_semantic_consensus(
                class_trackers, inference_kwargs["pixel_vote_thr"]
            )
        fill_volume(consensus_vol, consensus_tracker.instances)
    return consensus_vol


def run_3d(engine, img, norms, inference_kwargs):
    # NOTE: Long-term, the pipeline would fail long before here if the data doesn't match the requested model+params
    if img.ndim == 2:
        raise ValueError("Image must be a stack for ortho/3D inference!")

    trackers_dict = {}
    if inference_kwargs["inference_plane"] != "All":
        # TODO: Check if we still need this and how they do it!
        # Reorient the image if necessary
        if inference_kwargs["inference_plane"] == "XZ":
            img = np.moveaxis(img, 1, 0)
        elif inference_kwargs["inference_plane"] == "YZ":
            img = np.moveaxis(img, 2, 0)
        stack, trackers = infer_3d(engine, img, norms, inference_kwargs)
        trackers_dict[inference_kwargs["inference_plane"].lower()] = trackers
        stack = postprocess_volume(engine, trackers_dict, img.shape, inference_kwargs)
    else:
        trackers_dict = {}
        stacks = []
        for plane in _AXES.keys():
            inference_kwargs["inference_plane"] = plane
            stack, trackers = infer_3d(engine, img, norms, inference_kwargs)
            trackers_dict[plane] = trackers
            stacks.append(stack)
        stack = consensus_volume(engine, trackers_dict, inference_kwargs)
    return stack


if __name__ == "__main__":
    parser = create_argparser_inference()
    cli_args = parser.parse_args()

    img = load_img(
        fpath=cli_args.img_path,
        idxs=cli_args.idxs,
        channels=cli_args.channels,
        num_slices=cli_args.num_slices,
        dim_order="CZYX",
    )

    with open(cli_args.model_config, "r") as f:
        config = yaml.safe_load(f)
    device = get_device(model_type=get_model_name_type(cli_args.model_type))
    model = torch.jit.load(cli_args.model_chkpt, map_location=device)
    # Set model to eval mode (just in case)
    model.eval()
    # Try to extract model specifics from config, with some reasonable defaults
    norms = config.get("norms", {"mean": 0.57571, "std": 0.12765})
    padding_factor = config.get(
        "padding_factor", 128 if "mini" in cli_args.model_type.lower() else 16
    )
    thing_list = [] if config["semantic_only"] else config.get("thing_list", [1])
    labels = config.get("labels", thing_list if thing_list else [1])
    class_names = config.get(
        "class_names", {label: f"class_{label}" for label in labels}
    )
    inference_kwargs = {
        "semantic_only": config["semantic_only"],
        "inference_scale": config["downsampling"],
        "labels": labels,
        "class_names": class_names,
        "label_erosion": config["label_erosion"],
        "label_dilation": config["label_dilation"],
        "pixel_vote_thr": config["pixel_vote_thr"],
        "allow_one_view": config["allow_one_view"],
    }

    if img.squeeze().ndim == 2:
        # TODO: Unify plane and inference_plane
        if config["plane"] == "All":
            raise ValueError("Cannot run ortho-plane inference on a single slice!")
        # Create the inference engine
        engine = PanopticDeepLabRenderEngine(
            model=model,
            thing_list=thing_list,
            padding_factor=padding_factor,
            nms_kernel=config["min_distance"],
            nms_threshold=config["center_threshold"],
            confidence_thr=config["conf_threshold"],
            coarse_boundaries=not config["fine_boundaries"],
            label_divisor=config["max_objects"],
        )
        # Run inference
        pan_seg = run_2d(engine, img.squeeze(), norms, inference_kwargs)
    elif img.squeeze().ndim == 3:
        engine = PanopticDeepLabRenderEngine3d(
            model=model,
            thing_list=thing_list,
            median_kernel_size=config["median_slices"],
            label_divisor=config["max_objects"],
            nms_threshold=config["center_threshold"],
            nms_kernel=config["min_distance"],
            confidence_thr=config["conf_threshold"],
            padding_factor=padding_factor,
            coarse_boundaries=not config["fine_boundaries"],
        )
        inference_kwargs["inference_plane"] = config["plane"]
        pan_seg = run_3d(engine, img.squeeze(), norms, inference_kwargs)
    else:
        raise ValueError(
            f"Can only handle a 2D image, or stack of images, not {img.squeeze().shape}!"
        )
    # Save the stack
    save_masks(
        save_dir=Path(cli_args.output_dir),
        save_name=cli_args.mask_fname,
        masks=pan_seg,
        idxs=cli_args.idxs,
        mask_type="binary",  # Much faster this way
    )
