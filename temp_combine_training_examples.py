# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
import json
from memory_profiler import profile
import os
from pympler.tracker import SummaryTracker
import shutil
from tqdm import tqdm

from travel.constants import IMAGES_CHUNK_SIZE
from travel.data.utils import split_list_into_partitions
from travel.data.utils.image import resize_with_aspect, CACHED_FRAME_DIMENSION
from travel.data.vqa import VQAOutputs
from travel.data.vqg_learning import load_frameVQA_examples, save_vqg_training_examples, load_vqg_training_examples, FrameVQAMistakeDetectionExample, VQGTrainingExample
from travel.model.grounding import VisualFilterTypes
from travel.model.vqa import save_vqa_outputs
from travel.model.vqg_learning import FrameVQAMistakeDetectionScorer

TARGET_DIR = "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqg_learning/VQG_data_Llama-2-7b-hf_icl20_0.0_0.5_1.0_1.0/llava-1.5-7b-hf/VQA_data_llava-1.5-7b-hf_spatial1.0_20240706025209"
PARTITION = "train"

all_examples = []
for d in os.listdir(os.path.join(TARGET_DIR, f"VQA_scoring_cache_{PARTITION}")):
    if os.path.isdir(os.path.join(TARGET_DIR, f"VQA_scoring_cache_{PARTITION}", d)):
        this_examples = load_vqg_training_examples(os.path.join(TARGET_DIR, f"VQA_scoring_cache_{PARTITION}", d), PARTITION)
        all_examples += this_examples
        print(f"Got examples from {d}.")

save_vqg_training_examples(all_examples, TARGET_DIR, PARTITION)
print("Saved combined examples! :)")
        
    