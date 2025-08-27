import os
import yaml

# Can configure an alternate config path in os.environ if needed
if "TRAVEl_config_path" in os.environ:
    CONFIG_PATH = os.environ["TRAVEl_config_path"]
else:
    CONFIG_PATH = "config.yml"

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

DATA_CACHE_DIR = config["cache"]["data_cache_dir"] # Directory to cache model outputs and other temporary data
MODEL_CACHE_DIR = config["cache"]["model_cache_dir"] # Directory to cache pre-trained models from Hugging Face
CACHE_FREQUENCY = int(config["cache"]["cache_frequency"])
RESULTS_DIR = config["cache"]["results_dir"] # Directory to cache saved mistake detection results that should not be deleted
HF_TOKEN = config["hf_token"]
RANDOM_SEED = int(config["random_seed"])
IMAGES_CHUNK_SIZE = int(config["data"]["images_chunk_size"]) # When processing large numbers of images, this is the maximum number of images to keep in memory at once
QUALTRICS_API_TOKEN = str(config["data"]["qualtrics_api_token"])
QUALTRICS_DATA_CENTER = str(config["data"]["qualtrics_data_center"])
QUALTRICS_LIBRARY_NAME = str(config["data"]["qualtrics_library_name"])

for dir in DATA_CACHE_DIR, MODEL_CACHE_DIR, RESULTS_DIR:
    if not os.path.exists(dir):
        os.makedirs(dir)

DEFAULT_WANDB_PROJECT = "TRAVEl"