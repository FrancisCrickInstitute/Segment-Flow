from pathlib import Path

from empanada.inference.engines import (
    PanopticDeepLabRenderEngine,
    PanopticDeepLabRenderEngine3d,
)
import numpy as np
import skimage
import torch
import yaml

from utils import save_masks, create_argparser_inference, load_img, get_model_name_type
from model_utils import get_device


def normalize(img, mean, std):
    # Isn't the image a tensor?
    max_pixel_value = np.iinfo(img.dtype).max

    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def force_connected(pan_seg, thing_list, label_divisor=1000):
    for label in thing_list:
        min_id = label * label_divisor
        max_id = min_id + label_divisor

        instance_seg = pan_seg.copy()
        outside_mask = np.logical_or(pan_seg < min_id, pan_seg >= max_id)
        instance_seg[outside_mask] = 0

        instance_seg = skimage.measure.label(instance_seg).astype(np.uint16)
        instance_seg[instance_seg > 0] += min_id
        pan_seg[instance_seg > 0] = instance_seg[instance_seg > 0]
    return pan_seg


def preprocess(img, norms, device):
    # TODO: It should be just an image right?
    assert img.ndim == 2
    return (
        torch.from_numpy(normalize(img, norms["mean"], norms["std"])[None])
        .unsqueeze(0)
        .to(device)
    )


def infer_2d(engine, img, norms, device):
    # Resize if doing - SKIP
    orig_size = img.shape
    # Preprocess
    img = preprocess(img, norms, device)
    # Run inference
    pan_seg = engine(img, orig_size, upsampling=1)
    # Postprocess
    pan_seg = force_connected(
        pan_seg.squeeze().detach().cpu().numpy().astype(np.uint16),
        engine.thing_list,
        engine.label_divisor,
    )
    return pan_seg


def run_2d(engine, img, norms, device, inference_kwargs):
    # First handle special case of [B, C, H, W]
    if img.ndim == 4:
        if img.shape[0] != 1:
            raise ValueError(
                f"Can only handle an image, or stack of images, not {img.shape}!"
            )
        else:
            img = img.squeeze()
    full_mask = np.zeros(img.shape, dtype=int)
    # Stack of slices
    if img.ndim == 3:
        for plane, img_slice in enumerate(img):
            seg = infer_2d(engine, img_slice, norms, device)
            full_mask[plane, ...] = seg
        # # Pad segmentations  # NOTE: Why did they do this?
        # max_h = max(seg.shape[0] for seg in segs)
        # max_w = max(seg.shape[1] for seg in segs)
        # padded = []
        # for seg in segs:
        #     h, w = seg.shape
        #     pad_h = max_h - h
        #     pad_w = max_w - w
        #     padded.append(
        #         np.pad(seg, ((0, pad_h), (0, pad_w)))
        #     )
        # return np.stack(padded, axis=0)
        return full_mask
    # Single slice
    elif img.ndim == 2:
        return infer_2d(engine, img, norms, device)
    else:
        raise ValueError("Can only handle an image, or stack of images!")


def infer_3d():
    pass


def run_3d(engine, img, norms, inference_kwargs):
    pass


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
    # NOTE: These are fixed from MitoNet configs, no matter the version
    norms = {"mean": 0.57571, "std": 0.12765}
    thing_list = [] if config["semantic_only"] else [1]

    if "mini" in cli_args.model_type.lower():
        padding_factor = 128
    else:
        padding_factor = 16

    inference_kwargs = {
        "semantic_only": config["semantic_only"],
        "inference_scale": config["downsampling"],
    }

    if config["plane"] != "All":
        # Reorient the image if necessary
        if config["plane"] == "XZ":
            img = np.moveaxis(img, 1, 0)
        elif config["plane"] == "YZ":
            img = np.moveaxis(img, 2, 0)
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
        pan_seg = run_2d(engine, img, norms, device, inference_kwargs)
    else:
        raise NotImplementedError
        inference_plane = config["inference_plane"]
        # Create trackers

        # Create matchers

        # Create stack to insert into

        # Create the inference engine

        # Run inference

        # Extra step, apparently
        # final_segs = self.engine.end(self.inference_scale)
        # if final_segs:
        #     for i, pan_seg in enumerate(final_segs):
        #         pan_seg = pan_seg.squeeze().cpu().numpy()
        #         queue.put(pan_seg)

        # Postprocessing
        # Backward matching
        # NOTE: This should probably be separate, and allowed to do for all models
        # This also involves updating the trackers

        # FInish tracking

        # Filtering (ignore, should be separated)

        # Fill panoptic volume / save stack

        # Stack postprocessing
        # Relabels and filters each class
    # Save the stack
    save_masks(
        save_dir=Path(cli_args.output_dir),
        save_name=cli_args.mask_fname,
        masks=pan_seg,
        idxs=cli_args.idxs,
        mask_type="binary",  # Much faster this way
    )

# python run_mitonet.py --model-chkpt /Users/shandc/Documents/ai-on-demand/src/ai_on_demand/.nextflow/cache/mitonet/mitonet_mini.pt --img-path /Users/shandc/Documents/data/napari_examples/example_stack.tiff


"""
TODO:
- Integrate 3D mode
    - May be easier to split into classes as they did
- Add semantic only segmentation
- Integrate downsampling
    - Empanada uses cv2.resize for this
"""
