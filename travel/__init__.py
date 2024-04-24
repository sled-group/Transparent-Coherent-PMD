import numpy as np
import random
import torch

from travel.constants import RANDOM_SEED

# Set random seed
def set_random_seed():
    """Sets random seed in random, numpy, and torch from the one specified in config.yml file."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)