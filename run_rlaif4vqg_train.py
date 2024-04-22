import argparse
import datetime
import json
import os
import shutil
import torch
from tqdm import tqdm

from travel.constants import MODEL_CACHE_DIR, RESULTS_DIR, HF_TOKEN
from travel.model.vqg import VQG_DEMONSTRATIONS, generate_vqg_prompt_icl, VQGOutputs, save_vqg_outputs, parse_vqg_outputs
from travel.data.mistake_detection import MistakeDetectionTasks, get_cutoff_time_by_proportion
from travel.data.captaincook4d import CaptainCook4DDataset
from travel.data.captaincook4d.constants import RECIPE_STEPS

os.environ['HF_HOME'] = MODEL_CACHE_DIR
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

# parser = argparse.ArgumentParser()
# parser.add_argument("--task", type=str, default="captaincook4d", choices=[task.value for task in MistakeDetectionTasks])
# parser.add_argument("--lm_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Name or path to Hugging Face model for LM. Can be a fine-tuned LM for VQG.")
# parser.add_argument("--n_demonstrations", type=int, default=5, choices=range(1, len(VQG_DEMONSTRATIONS) + 1), help="Number of demonstrations of VQG for in-context learning. Must be <= the number of demonstrations available in travel.model.vqg.VQG_DEMONSTRATIONS.")
# # parser.add_argument("--n_questions_to_generate", type=int, default=2, choices=range(1, len(VQG_DEMONSTRATIONS[0].questions) + 1), help="Number of questions to generate per procedure.")
# parser.add_argument("--temperature", type=float, default=0.4, help="Temperature for language generation, i.e., degree of randomness to use in sampling words.")
# parser.add_argument("--top_p", type=float, default=0.9, help="top_p for language generation, i.e., top percentage of words to consider in terms of likelihood.")
# parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
# args = parser.parse_args()

# TODO: complete