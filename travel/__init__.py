import numpy as np
import os
import random
import torch

from travel.constants import RANDOM_SEED, MODEL_CACHE_DIR

# Set random seed
def set_random_seed():
    """Sets random seed in random, numpy, and torch from the one specified in config.yml file."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

def set_hf_cache():
    """Sets Hugging Face cache directory as configured in config.yml. This must be called before importing transformers anywhere else."""
    os.environ['HF_HOME'] = MODEL_CACHE_DIR

def configure_wandb():
    """Sets Weights & Biases project name."""
    os.environ["WANDB_PROJECT"] = "TRAVEl"

def init_travel():
    """Call this method at the beginning of any script to set random seedsd and redirect the Hugging Face cache."""
    set_random_seed()
    set_hf_cache()
    configure_wandb()