from travel import init_travel
init_travel()

import argparse
from collections import defaultdict, Counter
import json
import numpy as np
import os
import pickle
from PIL import Image
from pprint import pprint
import shutil
import spacy
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer, PhrasalConstraint           

from travel.constants import RESULTS_DIR, IMAGES_CHUNK_SIZE, HF_TOKEN, CONFIG_PATH
from travel.data.captaincook4d import CaptainCook4DDataset
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.data.vqa import VQAResponse, get_vqa_response_token_ids, VQAOutputs, DIALOG_START_TOKENS, IMAGE_TOKENS, USER_START_TOKENS, USER_END_TOKENS, ASSISTANT_START_TOKENS, ASSISTANT_END_TOKENS, IVQA_PREAMBLE, IVQA_SUCCESS_QUESTION
from travel.data.vqg import generate_vqg_prompt_icl
from travel.model import simple_lm_prompt_beam_search, simple_vlm_prompt_beam_search, compute_completion_log_likelihoods, compute_completion_log_likelihoods_encoder_decoder, compute_completion_log_likelihoods_vlm
from travel.model.grounding import VisualFilterTypes, ContrastiveRegionFilter, VisualContrastiveFilter, SpatialVisualFilter, AGLAFilter, ImageMaskTypes
from travel.model.metrics import mistake_detection_metrics, question_coherence_metrics_nli, question_coherence_metrics_vlm, generate_det_curve, generate_tiered_metric_curves, entropy, compile_accuracy_and_coherence_metrics
from travel.model.mistake_detection import MISTAKE_DETECTION_THRESHOLDS
from travel.model.nli import NLI_MODEL_PATH, NLI_BATCH_SIZE
from travel.model.vqa import run_vqa_with_visual_filter
from travel.model.vqg import cleanup_generated_question

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt-4o", help="Name of GPT model to use.")
parser.add_argument("--vqg_adapter_path", type=str, help="Name or path to adapter of VLM's LM to be used for VQG. This is for fine-tuned VQG models. Adapter base model should match the model used by the VLM specified in `vlm_name`.")
parser.add_argument("--task", type=str, default="ego4d_single", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--eval_partition", type=str, default="val", choices=["train", "val", "test"])
parser.add_argument("--max_iterations", type=int, default=10, help="Maximum number of questions to generate before making a final mistake detection decision.")
parser.add_argument("--num_beams", type=int, default=8, choices=list(range(21)), help="Number of beams in beam search.")
parser.add_argument("--num_return_sequences", type=int, default=4, choices=list(range(21)), help="Number of generation candidates to return from beam search. Recommend setting this to be less than number of beams due to generation constraints.")
parser.add_argument("--n_icl_demonstrations", type=int, default=0, choices=list(range(21)), help="Pass this argument to generate an extra pool of candidate questions using n in-context VQG examples (doesn't incorporate answers to previous questions).")
parser.add_argument("--condition_questions_with_frames", action="store_true", help="Pass this argument to pass frame into VLM while generating questions (usually off by default since this hurts performance).")
parser.add_argument("--question_selection_strategy", type=str, default="likelihood", choices=["likelihood", "relevance", "informativeness", "coherence"], help="Strategy to use to choose question to generate from beam search candidates.")
parser.add_argument("--exclude_history_from_vqa", action="store_true", help="Pass this argument to exclude the dialog history from VQA, and instead directly ask only questions.")
parser.add_argument("--coherence_evaluation_strategy", type=str, default="nli", choices=["vlm", "nli"], help="Strategy to use to perform final coherence evaluation of dialog.")
parser.add_argument("--early_stop_delta", type=int, default=0.1, help="If success probability changes less than this over 3 turns, stop generating questions.")
parser.add_argument("--confident_range", type=int, default=0.05, help="If success probability is within this from 0.0 or 1.0, stop early due to high confidence.")
parser.add_argument("--unsure_range", type=int, default=0.1, help="A VQA output will be considered unsure if the probability of yes and no are within this range of 50 percent (exclusive).")
parser.add_argument("--visual_filter_mode", type=str, required=False, choices=[t.value for t in VisualFilterTypes], help="Visual attention filter mode.")
parser.add_argument("--visual_filter_strength", type=float, required=False, default=1.0, help="Float strength for masks used in visual filters. Depending on the visual filter type, this may be interpreted as a percentage darkness or a Gaussian blur kernel size.")
parser.add_argument("--generation_batch_size", type=int, default=10, help="Batch size for question generation with LM.")
parser.add_argument("--vqa_batch_size", type=int, default=10, help="Batch size for VQA with VLM.")
parser.add_argument("--nli_batch_size", type=int, default=NLI_BATCH_SIZE, help="Batch size for scoring candidate questions with NLI model.")
parser.add_argument("--run_id", type=str, required=False, help="Unique ID for this run, which will be used to create the output directory (and should be shared across any parallel processes).")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of iterative VQA.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
parser.add_argument("--get_negated_success_probs", action="store_true", help="Pass this argument to calculate success probabilities for negated answers to questions.")
parser.add_argument("--run_allturns_metrics", action="store_true", help="Pass this argument run an additional set of metrics without early stopping.")
parser.add_argument("--cache_vqa_frames", action="store_true", help="Pass this argument to cache frames in VQA outputs (e.g., to inspect visual filter resuilts). This consumes a lot of disk space for large datasets.")
parser.add_argument("--print_prompts", action="store_true", help="Pass this argument to print some sample prompts during execution (for debugging purposes).")
args = parser.parse_args()

if args.cache_vqa_frames and args.visual_filter_mode is None:
    print("Warning: --cache_vqa_frames only applies to frames modified by visual filters (configured through --visual_filter_mode and --visual_filter_strength).")
if args.run_allturns_metrics and args.coherence_evaluation_strategy != "nli":
    print("Warning: --run_allturns_metrics only works with NLI-based coherence evaluation. Will not run all-turns metrics.")

# Set up results directory
if args.resume_dir is None:
    vlm_name = args.model_name
    task_name = args.task
    if args.debug:
        task_name += f"_debug{args.debug_n_examples}" if args.task != "captaincook4d" else "_debug"
    this_results_dir = os.path.join(task_name, vlm_name, f"IterativeVQA_q{args.max_iterations}_{task_name}")
    this_results_dir += f"_{vlm_name}"
    # Needed?
    this_results_dir += f"_beam{args.num_beams}-{args.num_return_sequences}"
    this_results_dir += f"_{args.question_selection_strategy}"
    if args.condition_questions_with_frames:
        this_results_dir += f"_cqframe"    
    if args.n_icl_demonstrations > 0:
        this_results_dir += f"_icl{args.n_icl_demonstrations}"
    if args.exclude_history_from_vqa:
        this_results_dir += "_nohistory"
    if args.visual_filter_mode is not None:
        this_results_dir += f"_{args.visual_filter_mode}{args.visual_filter_strength}"
    if args.vqg_adapter_path is not None:
        this_results_dir += "_dpo"
    this_results_dir += f"_{args.run_id}"
    this_results_dir = os.path.join(RESULTS_DIR, "vqa_mistake_detection", this_results_dir)
    if not os.path.exists(this_results_dir):
        os.makedirs(this_results_dir)
else:
    this_results_dir = args.resume_dir

# Set up visual filter if needed
visual_filter = None
nlp = None
if args.visual_filter_mode is not None:
    if VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Spatial_NoRephrase:
        visual_filter = SpatialVisualFilter(rephrase_questions=False, mask_strength=args.visual_filter_strength, mask_type=ImageMaskTypes.Darkness, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Spatial_Blur:
        visual_filter = SpatialVisualFilter(rephrase_questions=False, mask_strength=args.visual_filter_strength, mask_type=ImageMaskTypes.Blur, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')            
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Spatial:
        visual_filter = SpatialVisualFilter(rephrase_questions=True, mask_strength=args.visual_filter_strength, mask_type=ImageMaskTypes.Darkness, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')            
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Contrastive_Region:
        visual_filter = ContrastiveRegionFilter(mask_strength=args.visual_filter_strength, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Visual_Contrastive:
        visual_filter = VisualContrastiveFilter(alpha=args.visual_filter_strength, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')            
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.AGLA:
        visual_filter = AGLAFilter(alpha=args.visual_filter_strength, device=f"cuda:0")
        nlp = None
    else:
        raise NotImplementedError(f"Visual filter type {args.visual_filter_mode} is not compatible with iterative VQA!")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# NLI model to score consistency and verifiability
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)


# Load approopriate evaluation dataset
dataset = None
for retry in range(5):
    print(f"Loading evaluation dataset (try {retry})...")
    try:
        if MistakeDetectionTasks(args.task) == MistakeDetectionTasks.CaptainCook4D:
            dataset = CaptainCook4DDataset(data_split=args.eval_partition, debug_n_examples_per_class=args.debug_n_examples if args.debug else None)
        elif MistakeDetectionTasks(args.task) == MistakeDetectionTasks.Ego4D_Single:
            dataset = Ego4DMistakeDetectionDataset(data_split=args.eval_partition, 
                                                   mismatch_augmentation=True,
                                                   multi_frame=False,
                                                   debug_n_examples_per_class=args.debug_n_examples if args.debug else None)
        else:
            raise NotImplementedError(f"Haven't implemented usage of {args.task} dataset yet!")
        break
    except Exception as e:
        print("Encountered error during data loading:")
        pprint(e)
        time.sleep(60)
if dataset is None:
    raise ValueError("Could not load dataset after retrying!")

print(f"Beginning iterative VQA inference...")
all_questions = []
all_candidate_questions = []
all_candidate_questions_scores = []
all_candidate_questions_sources = []
all_scores = []
all_answers = []
all_answer_probs = []
all_success_probs = []
all_success_probs_negated = []

all_example_ids = []
all_procedures = []
all_labels = []

cache_path = os.path.join(this_results_dir, f"cached_outputs.pkl")
is_complete = False
last_batch_idx = -1
if os.path.exists(cache_path):
    is_complete, last_batch_idx, all_questions, all_candidate_questions, all_candidate_questions_scores, all_candidate_questions_sources, all_scores, all_answers, all_answer_probs, all_success_probs, all_success_probs_negated, all_example_ids, all_procedures, all_labels = pickle.load(open(cache_path, "rb"))

