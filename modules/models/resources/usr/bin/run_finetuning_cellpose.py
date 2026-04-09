from pathlib import Path
import shutil

from cellpose import io, models, train

from utils import create_argparser_finetune


def update_training_metrics_file(train_losses, fpath: Path):
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss\n")
        for epoch, loss in enumerate(train_losses, start=1):
            f.write(f"{epoch},{float(loss):.6f}\n")


def finetune_cellpose(cli_args):
    io.logger_setup()

    train_dir = Path(cli_args.train_dir)
    save_dir = Path(cli_args.model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = int(cli_args.epochs)
    batch_size = 1
    learning_rate = 1e-3
    weight_decay = 1e-4
    masks_ext = "_seg"  # TODO: move these params to a default config AIOD-258

    # Load train data in Cellpose expected format; use train split as test split fallback.
    images, labels, *_ = io.load_train_test_data(
        str(train_dir),
        str(train_dir),  # TODO: Add testing data option
        mask_filter=masks_ext,
        look_one_level_down=False,
    )

    if len(images) == 0:
        raise ValueError(
            f"No training images found in '{train_dir}'. "
            f"Expected image files with masks ending in '{masks_ext}'."
        )

    # NOTE: Segment-Flow passes args like --patch_size and --layers for all models.
    # They are currently not used by Cellpose finetuning.
    model = models.CellposeModel(
        model_type=cli_args.model_type,
        pretrained_model=cli_args.model_chkpt,
    )

    model_name = cli_args.model_save_name

    new_model_path, train_losses, _ = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        normalize=True,
        channels=[0, 0],
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        SGD=True,
        min_train_masks=1,
        model_name=model_name,
    )

    metrics_path = save_dir / "training_metrics.csv"
    update_training_metrics_file(train_losses, metrics_path)

    # Copy final model artifact into the caller-specified save directory.
    final_model_copy = save_dir / f"{model_name}.pth"
    if Path(new_model_path).exists():
        shutil.copy2(new_model_path, final_model_copy)

    # Cellpose writes model checkpoints under its own training directory.
    # Persist the final resolved path for downstream registration/inspection.
    final_model_pointer = save_dir / f"{model_name}.txt"
    with open(final_model_pointer, "w", encoding="utf-8") as f:
        f.write(str(new_model_path))

    print(f"Finished Cellpose finetuning. Final model path: {new_model_path}")
    if final_model_copy.exists():
        print(f"Copied final model to: {final_model_copy}")
    print(f"Training metrics written to: {metrics_path}")


if __name__ == "__main__":
    print("Running Cellpose Finetuning")
    parser = create_argparser_finetune()
    cli_args = parser.parse_args()
    finetune_cellpose(cli_args)
    print("Finetuning complete, please save model to local model registry to use.")
