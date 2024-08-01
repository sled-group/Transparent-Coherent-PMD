import json
import os
import yaml

from travel.constants import CONFIG_PATH

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

ANNOTATIONS_DIR = config["data"]["captaincook4d"]["annotations_dir"]
VIDEO_DIR = config["data"]["captaincook4d"]["video_dir"] # Directory containing CaptainCook4D mp4s

# NOTE: there may be several ways to split the CaptainCook4D data, as mentioned in their paper:
# A. Split up recordings arbitrarily without any dependency
# B. By recording environment (save some environments for testing)
# C. By recipe (save some recipes for testing)
# D. By person (i.e., person following the recipe)
# Recipe might make more sense in the future to target generalization. But for now, let's just split arbitrarily to 
DATA_SPLITS = json.load(open(os.path.join(ANNOTATIONS_DIR, "data_splits/recordings_data_split_combined.json"), "r"))

ERROR_CATEGORIES = json.load(open(os.path.join(ANNOTATIONS_DIR, "annotation_json/error_category_idx.json"), "r"))

RECIPE_STEPS = json.load(open(os.path.join(ANNOTATIONS_DIR, "annotation_json/step_idx_description.json"), "r"))
RECIPE_STEPS = {int(k): "-".join(v.split("-")[1:]).strip() for k, v in RECIPE_STEPS.items()}