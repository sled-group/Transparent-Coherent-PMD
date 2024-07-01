import numpy as np
import os
import random
import resource
import torch

from travel.constants import RANDOM_SEED, MODEL_CACHE_DIR

# Set random seed
def set_random_seed(random_seed=RANDOM_SEED):
    """Sets random seed in random, numpy, and torch from the one specified in config.yml file."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

def set_hf_cache():
    """Sets Hugging Face cache directory as configured in config.yml. This must be called before importing transformers anywhere else."""
    os.environ['HF_HOME'] = MODEL_CACHE_DIR

def configure_wandb(wandb_project_name: str="TRAVEl"):
    """Sets Weights & Biases project name."""
    os.environ["WANDB_PROJECT"] = wandb_project_name

def init_travel(wandb_project_name: str="TRAVEl"):
    """
    Call this method at the beginning of any script to set random seeds and redirect the Hugging Face cache.

    :param wandb_project_name: Name for project in Weights & Biases (`wandb`).
    """
    set_random_seed()
    set_hf_cache()
    configure_wandb(wandb_project_name)

def set_memory_limit(n_bytes):

    """Force Python to raise an exception when it uses more than n_bytes bytes of memory. From https://metabob.com/blog-articles/chasing-memory-spikes-and-leaks-in-python.html."""
    if n_bytes <= 0: 
        return

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)

    resource.setrlimit(resource.RLIMIT_AS, (n_bytes, hard))

    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)

    if n_bytes < soft*1024:

        resource.setrlimit(resource.RLIMIT_DATA, (n_bytes, hard))