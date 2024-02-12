import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

DATA_CACHE_DIR = config["cache"]["data_cache_dir"] # Directory to cache model outputs and other data
MODEL_CACHE_DIR = config["cache"]["model_cache_dir"] # Directory to cache pre-trained models from Hugging Face
