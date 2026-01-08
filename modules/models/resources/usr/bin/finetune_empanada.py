print("run_finetuning!")
from run_finetuning import finetune
import yaml


def run_finetuning(train_dir, model_dir, save_dir, save_name, layers, epochs):
    print("run_finetuning in finetune_widget")
    # collect user input of the finetuning widget and convert into a config

    config_location = (
        "/Users/ahmedn/Work/sandbox/finetuning/scripts/finetuning_config.yml"
    )

    try:
        with open(config_location, mode="r") as handle:
            finetuning_config = yaml.load(handle, Loader=yaml.FullLoader)
    except:
        print(
            "Couldn't open finetuning config file ensure you have finetune_config.yml"
        )

    finetuning_config["TRAIN"]["train_dir"] = train_dir
    finetuning_config["TRAIN"]["model_dir"] = model_dir
    finetuning_config["TRAIN"]["save_dir"] = save_dir
    finetuning_config["TRAIN"]["save_name"] = save_name
    finetuning_config["TRAIN"]["layers"] = layers
    finetuning_config["TRAIN"]["epochs"] = epochs
    finetuning_config["TRAIN"]["device"] = "cpu"
    finetuning_config["EVAL"]["epochs_per_eval"] = 1
    finetuning_config["EVAL"]["class_names"] = {1: "mito"}

    finetuning_config["EVAL"]["engine"] = "PanopticDeepLabEngine"

    finetuning_config["EVAL"]["engine_params"] = {
        "confidence_thr": 0.5,
        "label_divisor": 1000,
        "nms_kernel": 7,
        "nms_threshold": 0.1,
        "stuff_area": 64,
        "thing_list": [1],
        "void_label": 0,
    }
    finetuning_config["EVAL"]["print_freq"] = 0

    finetune(finetuning_config)


train_dir = "/Users/ahmedn/Work/finetuning/c_elegans_mitos"
model_dir = "/Users/ahmedn/Work/finetuning/models/MitoNet_v1_mini.pth"
save_dir = "/Users/ahmedn/Work/finetuning/models"
save_name = "mini_finetuned"
layers = "none"
epochs = 1

run_finetuning(train_dir, model_dir, save_dir, save_name, layers, epochs)
