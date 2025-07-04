from pathlib import Path

import numpy as np
import skimage.measure
import skimage.io

from em_segment.modules.loading import load_from_yaml
from em_segment.predictions import do_predictions
from model_utils import get_device
from utils import save_masks, create_argparser_inference, load_img, get_model_name_type

if __name__ == "__main__":
    parser = create_argparser_inference()
    cli_args = parser.parse_args()

    chkpt_path = Path(cli_args.model_chkpt)
    assert chkpt_path.is_file()
    config_path = Path(cli_args.model_config)
    assert config_path.is_file()

    # Load the trainer/model etc. from yaml config
    trainer, evaluators, config_obj = load_from_yaml(config_path)

    # Load the image
    stack = load_img(
        fpath=cli_args.img_path,
        idxs=cli_args.idxs,
        channels=cli_args.channels,
        num_slices=cli_args.num_slices,
        dim_order="CZYX",
    )
    stack = np.squeeze(stack)
    # Get the segmentations
    preds = do_predictions(
        trainer=trainer,
        config=config_obj,
        stack=stack,
        stack_name=Path(cli_args.img_path).stem,
        stack_filepath=cli_args.img_path,
        chkpt_path=chkpt_path,
        load_kwargs={
            "map_location": get_device(
                model_type=get_model_name_type(get_model_name_type(cli_args.model_type))
            )
        },
    )
    # Get connected components
    labelled_stack = skimage.measure.label(preds > 0.5)
    # Save the stack
    save_masks(
        save_dir=Path(cli_args.output_dir),
        save_name=cli_args.mask_fname,
        masks=labelled_stack,
        idxs=cli_args.idxs,
        mask_type="binary",
    )
