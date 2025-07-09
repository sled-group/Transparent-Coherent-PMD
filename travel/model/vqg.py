import numpy as np
import torch
from tqdm import tqdm
from transformers import TextGenerationPipeline
from transformers.pipelines.pt_utils import KeyDataset
from typing import Optional, Union
import yaml

def cleanup_generated_question(question):
    """Cleanup method for generated questions in iterative VQA pipeline."""
    question = question.split("?")[0].strip() + "?"
    if "." in question:
        question = question.split(".")[1].strip()    
    return question