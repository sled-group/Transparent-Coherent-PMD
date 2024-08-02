import json
import os
import yaml

from travel.constants import CONFIG_PATH

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

EGO4D_ANNOTATION_PATH = config["data"]["ego4d"]["annotation_path"]
EGO4D_SPLIT_PATHS = {
    "train": os.path.join(config["data"]["ego4d"]["split_path"], "fho_main_train.json"),
    "val": os.path.join(config["data"]["ego4d"]["split_path"], "fho_main_val.json"),
    "test": os.path.join(config["data"]["ego4d"]["split_path"], "fho_main_test.json")
}
EGO4D_VIDEO_PATH = config["data"]["ego4d"]["video_path"]

MISALIGNSRL_PATH = config["data"]["ego4d"]["misalignsrl_path"]