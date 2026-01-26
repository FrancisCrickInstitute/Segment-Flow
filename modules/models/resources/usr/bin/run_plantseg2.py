from pathlib import Path
from typing import Union

import numpy as np
import yaml

from utils import save_masks, create_argparser_inference, load_img, get_model_name_type
from model_utils import get_device


def run_plantseg2(
    save_dir: Union[Path, str],
    save_name: str,
    idxs: list[int, ...],
    img: np.ndarray,
    config: dict,
):
    """Run PlantSeg2 pipeline: prediction -> watershed -> GASP.

    Args:
        save_dir: Directory to save the output masks
        save_name: Base name for saved files
        idxs: Slice indices being processed
        img: Input image array
        config: Configuration dictionary containing model and segmentation parameters
    """
    from plantseg.functionals.prediction.prediction import unet_prediction
    from plantseg.functionals.segmentation.segmentation import dt_watershed, gasp

    # Step 1: Run prediction to get boundary probability maps
    print("Running prediction...")

    # Prepare prediction parameters
    model_name = config.get("model_name", None)
    model_id = config.get("model_id", None)
    patch = config.get("patch", None)
    if patch is not None:
        patch = tuple(patch)
    patch_halo = config.get("patch_halo", None)
    if patch_halo is not None:
        patch_halo = tuple(patch_halo)

    device = config.get("device", "cuda")
    input_layout = config.get("input_layout", "ZYX")

    # Run prediction
    boundary_pmaps = unet_prediction(
        raw=img,
        input_layout=input_layout,
        model_name=model_name,
        model_id=model_id,
        patch=patch,
        patch_halo=patch_halo,
        single_batch_mode=config.get("single_batch_mode", True),
        device=device,
        model_update=config.get("model_update", False),
        disable_tqdm=config.get("disable_tqdm", False),
    )

    # If multi-channel output, select the first channel
    if boundary_pmaps.ndim == 4:
        boundary_pmaps = boundary_pmaps[0]

    print(f"Prediction complete. Boundary pmaps shape: {boundary_pmaps.shape}")

    # Step 2: Run watershed segmentation
    print("Running watershed segmentation...")

    superpixels = dt_watershed(
        boundary_pmaps=boundary_pmaps,
        threshold=config.get("ws_threshold", 0.5),
        sigma_seeds=config.get("ws_sigma_seeds", 1.0),
        stacked=config.get("ws_stacked", False),
        sigma_weights=config.get("ws_sigma_weights", 2.0),
        min_size=config.get("ws_min_size", 100),
        alpha=config.get("ws_alpha", 1.0),
        pixel_pitch=config.get("ws_pixel_pitch", None),
        apply_nonmax_suppression=config.get("ws_apply_nonmax_suppression", False),
        n_threads=config.get("n_threads", None),
        mask=config.get("mask", None),
    )

    print(f"Watershed complete. Superpixels shape: {superpixels.shape}, unique labels: {len(np.unique(superpixels))}")

    # Step 3: Run GASP segmentation
    print("Running GASP segmentation...")

    segmentation = gasp(
        boundary_pmaps=boundary_pmaps,
        superpixels=superpixels,
        gasp_linkage_criteria=config.get("gasp_linkage_criteria", "average"),
        beta=config.get("gasp_beta", 0.5),
        post_minsize=config.get("gasp_post_minsize", 100),
        n_threads=config.get("n_threads", 6),
    )

    print(f"GASP complete. Segmentation shape: {segmentation.shape}, unique labels: {len(np.unique(segmentation))}")

    # Save the final segmentation
    save_masks(Path(save_dir), save_name, segmentation, idxs=idxs, mask_type="instance")

    # Also save as TIFF using PlantSeg's create_tiff
    from plantseg.io.tiff import create_tiff
    from plantseg.io.voxelsize import VoxelSize

    start_x, end_x, start_y, end_y, start_z, end_z = idxs
    tiff_path = Path(save_dir) / f"{save_name}_x{start_x}-{end_x}_y{start_y}-{end_y}_z{start_z}-{end_z}.tif"

    # Get voxel size from config if available, otherwise use default
    voxel_size_tuple = config.get("voxel_size", None)
    voxel_size_unit = config.get("voxel_size_unit", "um")
    voxel_size = VoxelSize(voxels_size=voxel_size_tuple, unit=voxel_size_unit)

    # Determine layout based on segmentation dimensions
    if segmentation.ndim == 3:
        layout = "ZYX"
    elif segmentation.ndim == 2:
        layout = "YX"
    else:
        layout = "ZYX"  # fallback

    create_tiff(tiff_path, segmentation.astype(np.uint16), voxel_size, layout=layout)
    print(f"Segmentation saved to {tiff_path}")


if __name__ == "__main__":
    parser = create_argparser_inference()
    cli_args = parser.parse_args()

    with open(cli_args.model_config, "r") as f:
        config = yaml.safe_load(f)

    # Load image and apply preprocessing if specified
    img = load_img(
        fpath=cli_args.img_path,
        idxs=cli_args.idxs,
        channels=cli_args.channels,
        num_slices=cli_args.num_slices,
        dim_order="CZYX",
    )

    # Determine input layout based on image shape
    # Since load_img always returns CZYX format, we need to handle the layout accordingly
    if cli_args.num_slices > 1:
        # 3D data
        if cli_args.channels == 1:
            # Squeeze channel dimension for single channel 3D data
            img = img[0]  # Remove the channel dimension
            input_layout = "ZYX"
        else:
            # Multi-channel 3D data
            input_layout = "CZYX"
    else:
        # 2D data
        if cli_args.channels == 1:
            # Squeeze both channel and z dimensions for single channel 2D data
            img = img[0, 0]  # Remove channel and z dimensions
            input_layout = "YX"
        else:
            # Multi-channel 2D data
            img = img[:, 0, :, :]  # Remove z dimension but keep channels
            input_layout = "CYX"

    config["input_layout"] = input_layout

    # Set device based on model type if not specified in config
    if "device" not in config:
        config["device"] = get_device(model_type=get_model_name_type(cli_args.model_type))

    run_plantseg2(
        save_dir=cli_args.output_dir,
        save_name=cli_args.mask_fname,
        idxs=cli_args.idxs,
        img=img,
        config=config,
    )
