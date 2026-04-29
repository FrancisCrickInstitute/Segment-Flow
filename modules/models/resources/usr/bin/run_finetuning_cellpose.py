import logging
from pathlib import Path
from cellpose import io, models, train
from utils import create_argparser_finetune
from utils_finetuning import save_finetuned_model_artifact
import yaml


class CSVLoggingHandler(logging.Handler):
    """Custom handler that writes loss metrics to CSV."""

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        # Initialize CSV file
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,test_loss\n")

    def emit(self, record):
        """Extract and log loss values from training messages."""
        msg = record.getMessage()
        # Cellpose logs: "epoch, train_loss=X.XXXX, test_loss=Y.YYYY, ..."
        if "train_loss=" in msg and "test_loss=" in msg:
            try:
                parts = msg.split(", ")
                epoch = parts[0]
                train_loss = parts[1].split("=")[1]
                test_loss = parts[2].split("=")[1]

                with open(self.csv_path, "a", encoding="utf-8") as f:
                    f.write(f"{epoch},{train_loss},{test_loss}\n")
                    f.flush()
            except (IndexError, ValueError):
                pass


def update_training_metrics_file(train_losses, fpath: Path):
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss\n")
        for epoch, loss in enumerate(train_losses, start=1):
            f.write(f"{epoch},{float(loss):.6f}\n")


def finetune_cellpose(cli_args):
    io.logger_setup()

    train_dir = cli_args.train_dir
    test_dir = cli_args.test_dir if cli_args.test_dir else None

    save_dir = Path(cli_args.model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(cli_args.model_config, "r") as f:
        config = yaml.safe_load(f)

    # Extract the segment and nucleus channels
    channels = [
        config["segment_channel"],
        config["nucleus_channel"],
    ]

    metrics_path = save_dir / "training_metrics.csv"
    csv_handler = CSVLoggingHandler(metrics_path)
    cellpose_logger = logging.getLogger("cellpose.train")
    cellpose_logger.addHandler(csv_handler)

    n_epochs = int(cli_args.epochs)
    batch_size = 1
    learning_rate = float(cli_args.learning_rate)
    weight_decay = float(cli_args.weight_decay)
    use_sgd = bool(cli_args.sdg)
    momentum = float(cli_args.momentum)
    masks_ext = "_seg"

    # Load train data and optional test data in Cellpose expected format.
    images, labels, _, test_images, test_labels, _ = io.load_train_test_data(
        train_dir,
        test_dir,
        mask_filter=masks_ext,
        look_one_level_down=False,
    )

    if len(images) == 0:
        raise ValueError(
            f"No training images found in '{train_dir}'. "
            f"Expected image files with masks ending in '{masks_ext}'."
        )

    model = models.CellposeModel(
        model_type=cli_args.model_type,
        pretrained_model=cli_args.model_chkpt,
    )

    model_name = cli_args.model_save_name

    new_model_path, train_losses, _ = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        test_data=test_images,
        test_labels=test_labels,
        normalize=True,
        channels=channels,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        SGD=use_sgd,
        min_train_masks=1,
        model_name=model_name,
    )

    final_model_copy = save_finetuned_model_artifact(
        source_model_path=new_model_path,
        save_dir=save_dir,
        model_name=model_name,
    )

    # Cellpose writes model checkpoints under its own training directory.
    # Persist the final resolved path for downstream registration/inspection.
    final_model_pointer = save_dir / f"{model_name}.txt"
    with open(final_model_pointer, "w", encoding="utf-8") as f:
        f.write(str(new_model_path))

    print(f"Finished Cellpose finetuning. Final model path: {final_model_copy}")
    print(f"Training metrics written to: {metrics_path}")


if __name__ == "__main__":
    print("Running Cellpose Finetuning")
    parser = create_argparser_finetune()
    cli_args = parser.parse_args()
    finetune_cellpose(cli_args)
    print("Finetuning complete, please save model to local model registry to use.")
