import time
import argparse


def update_training_metrics_file(epoch, average_loss, fpath):
    mode = "w" if epoch == 1 else "a"  # overwrite the file if first epoch
    with open(fpath, mode) as f:
        f.write(f"{epoch},{average_loss:.6f}\n")


def clear_training_metrics_file(fpath):
    f = open(fpath, "w+")
    f.close


def run_finetuning_test(epochs):
    for i in range(1, epochs + 1):
        average_loss = 0
        update_training_metrics_file(
            i, average_loss, "/Users/ahmedn/Desktop/training_metrics.csv"
        )
        time.sleep(5)
    clear_training_metrics_file("/Users/ahmedn/Desktop/training_metrics.csv")


def create_argparser_finetune():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dir", required=True, help="Path to ground truth for finetuning"
    )
    parser.add_argument("--model_chkpt", required=True, help="Base model Checkpoint")
    parser.add_argument(
        "--model_save_name", required=True, help="Name of the final finetuned model"
    )
    parser.add_argument(
        "--model_save_dir", required=True, help="Where to save the finetuned models"
    )
    parser.add_argument(
        "--layers", required=True, help="Layers to be unfrozen when fine-tuning"
    )
    parser.add_argument(
        "--epochs", required=True, help="Number of epochs to finetune for"
    )

    return parser


if __name__ == "__main__":
    print("Running Finetuning TEST")
    parser = create_argparser_finetune()
    cli_args = parser.parse_args()
    run_finetuning_test(int(cli_args.epochs))
    print("Finetuning complete, please save model to local model registry to use.")
