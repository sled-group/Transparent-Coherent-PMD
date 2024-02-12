import json
import os
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

ANNOTATIONS_DIR = config["captaincook4d"]["annotations_dir"]
VIDEO_DIR = config["captaincook4d"]["video_dir"] # Directory containing CaptainCook4D mp4s
ERROR_CATEGORIES = json.load(open(os.path.join(ANNOTATIONS_DIR, "annotation_json/error_category_idx.json"), "r"))
