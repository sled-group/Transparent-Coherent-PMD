import os
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

DATA_CACHE_DIR = config["cache"]["data_cache_dir"] # Directory to cache model outputs and other temporary data
MODEL_CACHE_DIR = config["cache"]["model_cache_dir"] # Directory to cache pre-trained models from Hugging Face
RESULTS_DIR = config["cache"]["results_dir"] # Directory to cache saved mistake detection results that should not be deleted

for dir in DATA_CACHE_DIR, MODEL_CACHE_DIR, RESULTS_DIR:
    if not os.path.exists(dir):
        os.makedirs(dir)