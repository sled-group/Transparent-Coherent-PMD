from typing import Any
import yaml

from travel.constants import CONFIG_PATH
from travel.data.utils import generate_float_series

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
DETECTION_FRAMES_PROPORTION = float(config["mistake_detection_strategies"]["frames_proportion"]) # Use last N% of frames for frame-based mistake detection strategies
MISTAKE_DETECTION_THRESHOLDS = [round(threshold, 2) for threshold in generate_float_series(0.01, 0.99, 0.01)]
DIV_MODEL_PATH = str(config['mistake_detection_strategies']['diversity_model_path'])