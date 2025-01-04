from travel import init_travel
init_travel()

import argparse
from collections import defaultdict, Counter
from copy import deepcopy
from itertools import product
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
from travel.model.metrics import question_coherence_metrics_nli, question_coherence_metrics_vlm, generate_det_curve, compile_accuracy_and_coherence_metrics, generate_3d_overview_graph
from travel.model.mistake_detection import MISTAKE_DETECTION_THRESHOLDS
from travel.model.nli import NLI_MODEL_PATH, NLI_BATCH_SIZE
from travel.model.vqa import run_vqa_with_visual_filter
from travel.model.vqg import cleanup_generated_question


parser = argparse.ArgumentParser()
parser.add_argument("--vlm_name", type=str, default=None, help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--hf_hub_revision", type=str, default=None, help="Optional revision ID for VLM in Hugging Face Hub.")
parser.add_argument("--vqg_adapter_path", type=str, help="Name or path to adapter of VLM's LM to be used for VQG. This is for fine-tuned VQG models. Adapter base model should match the model used by the VLM specified in `vlm_name`.")
parser.add_argument("--task", type=str, default="ego4d_single", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--eval_partition", type=str, default="val", choices=["train", "val", "test"])
parser.add_argument("--max_iterations", type=int, default=10, help="Maximum number of questions to generate before making a final mistake detection decision.")
parser.add_argument("--length_penalty", type=float, default=1.0, help="Exponential length penalty for generation (> 0.0 promotes long sequences, < 0.0 promotes short sequences).")
parser.add_argument("--restrict_q_words", action="store_true", help="Pass this argument to restrict first words of generated questions to 'is', 'are', 'do', and 'does'.")
parser.add_argument("--num_beams", type=int, default=8, choices=list(range(21)), help="Number of beams in beam search.")
parser.add_argument("--num_return_sequences", type=int, default=4, choices=list(range(21)), help="Number of generation candidates to return from beam search. Recommend setting this to be less than number of beams due to generation constraints.")
parser.add_argument("--n_icl_demonstrations", type=int, default=0, choices=list(range(21)), help="Pass this argument to generate an extra pool of candidate questions using n in-context VQG examples (doesn't incorporate answers to previous questions).")
parser.add_argument("--condition_questions_with_frames", action="store_true", help="Pass this argument to pass frame into VLM while generating questions (usually off by default since this hurts performance).")
parser.add_argument("--question_selection_strategy", type=str, default="likelihood", choices=["likelihood", "relevance", "informativeness", "coherence"], help="Strategy to use to choose question to generate from beam search candidates.")
parser.add_argument("--exclude_history_from_vqa", action="store_true", help="Pass this argument to exclude the dialog history from VQA, and instead directly ask only questions.")
parser.add_argument("--early_stop_delta", nargs='+', type=float, default=[0.05, 0.1, 0.2, 0.4], help="List of early_stop_delta values to consider, separated by spaces. If success probability changes less than this over 3 turns, stop generating questions.")
parser.add_argument("--confident_range", nargs='+', type=float, default=[0.025, 0.05, 0.1, 0.2], help="List of confident_range values to consider, separated by spaces. If success probability is within this from 0.0 or 1.0, stop early due to high confidence.")
parser.add_argument("--unsure_range", type=float, default=0.1, help="A VQA output will be considered unsure if the probability of yes and no are within this range of 50 percent (exclusive).")
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
parser.add_argument("--cache_vqa_frames", action="store_true", help="Pass this argument to cache frames in VQA outputs (e.g., to inspect visual filter resuilts). This consumes a lot of disk space for large datasets.")
parser.add_argument("--print_prompts", action="store_true", help="Pass this argument to print some sample prompts during execution (for debugging purposes).")
args = parser.parse_args()

if args.vlm_name is None:
    raise ValueError("You must pass an HF VLM name.")

assert torch.cuda.device_count() == 1, "Iterative VQA requires exactly 1 GPU per process; use `srun` to enable multi-GPU parallelization."
if args.cache_vqa_frames and args.visual_filter_mode is None:
    print("Warning: --cache_vqa_frames only applies to frames modified by visual filters (configured through --visual_filter_mode and --visual_filter_strength).")

# Get parallelization details from srun if any
if "SLURM_PROCID" in os.environ and "SLURM_NPROCS" in os.environ:
    worker_index = int(os.environ["SLURM_PROCID"])
    n_workers = int(os.environ["SLURM_NPROCS"])
else:
    worker_index = 0
    n_workers = 1
# NOTE: if resuming from a previous run, must have the same number of GPUs as original run

# Set up results directory
if args.resume_dir is None:
    vlm_name = args.vlm_name.split('/')[-1]
    task_name = args.task
    if args.debug:
        task_name += f"_debug{args.debug_n_examples}" if args.task != "captaincook4d" else "_debug"
    this_results_dir = os.path.join(task_name, vlm_name, f"IterativeVQA_q{args.max_iterations}_{task_name}")
    this_results_dir += f"_{vlm_name}"
    this_results_dir += f"_beam{args.num_beams}-{args.num_return_sequences}"
    if args.length_penalty != 1.0:
        this_results_dir += f"_lp{args.length_penalty}"
    if args.restrict_q_words:
        this_results_dir += "_qw"
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
    if worker_index == 0 and not os.path.exists(this_results_dir):
        os.makedirs(this_results_dir)
else:
    this_results_dir = args.resume_dir


print(f"({worker_index}) Setting up models...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load VLM - some VLMs may be under AutoModelForVision2Seq, some may be under AutoModelForCausalLM
try:
    vlm = AutoModelForVision2Seq.from_pretrained(args.vlm_name, quantization_config=bnb_config, trust_remote_code=True, token=HF_TOKEN, revision=args.hf_hub_revision)   
except Exception as e:
    print("Encountered exception when trying to load model with AutoModelForVision2Seq:")
    pprint(e)
    
    vlm = AutoModelForCausalLM.from_pretrained(args.vlm_name, quantization_config=bnb_config, trust_remote_code=True, token=HF_TOKEN, revision=args.hf_hub_revision)
vlm_processor = AutoProcessor.from_pretrained(args.vlm_name, trust_remote_code=True, token=HF_TOKEN, revision=args.hf_hub_revision)
vlm_processor.tokenizer.padding_side = "left"
response_token_ids = get_vqa_response_token_ids(vlm_processor.tokenizer)

# We'll use VLM's LM directly to generate questions
if getattr(vlm, "language_model", None):
    lm = vlm.language_model
else:
    lm = vlm
tokenizer = vlm_processor.tokenizer
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load adapter for VQG if there is one
if args.vqg_adapter_path is not None:
    assert not args.condition_questions_with_frames, "VQG adapters are only supported for image-free VQG."
    lm.load_adapter(args.vqg_adapter_path, adapter_name="vqg")
    print("Loaded VQG adapter at", args.vqg_adapter_path)
    print(lm.active_adapters())

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

# Shared generation kwargs

# kwargs to force question generations to have a "?" and start with words that would typically begin a yes/no question
question_generation_constraints = [    
    PhrasalConstraint(
        [vlm_processor.tokenizer("Is it blue?", add_special_tokens=False).input_ids[-1]]
    ),
]
if not args.restrict_q_words:
    yes_no_q_tokens = [
        vlm_processor.tokenizer("Is it blue?", add_special_tokens=False).input_ids[0], 
        vlm_processor.tokenizer("Was it blue?", add_special_tokens=False).input_ids[0],
        vlm_processor.tokenizer("Are they blue?", add_special_tokens=False).input_ids[0], 
        vlm_processor.tokenizer("Were they blue?", add_special_tokens=False).input_ids[0],
        vlm_processor.tokenizer("Does it look blue?", add_special_tokens=False).input_ids[0],
        vlm_processor.tokenizer("Do they look blue?", add_special_tokens=False).input_ids[0],
        vlm_processor.tokenizer("Did they look blue?", add_special_tokens=False).input_ids[0],
        vlm_processor.tokenizer("Has the oven turned on?", add_special_tokens=False).input_ids[0],
        vlm_processor.tokenizer("Have the eggs boiled?", add_special_tokens=False).input_ids[0],
        vlm_processor.tokenizer("Had the eggs boiled?", add_special_tokens=False).input_ids[0],
    ]
else:
    yes_no_q_tokens = [
        vlm_processor.tokenizer("Is it blue?", add_special_tokens=False).input_ids[0], 
        vlm_processor.tokenizer("Are they blue?", add_special_tokens=False).input_ids[0], 
        vlm_processor.tokenizer("Does it look blue?", add_special_tokens=False).input_ids[0],
        vlm_processor.tokenizer("Do they look blue?", add_special_tokens=False).input_ids[0],
    ]
begin_suppress_tokens = [t for t in list(range(vlm_processor.tokenizer.vocab_size)) if t not in yes_no_q_tokens]
bad_words_ids = [[vlm_processor.tokenizer("Yes or no?", add_special_tokens=False).input_ids[1]], 
                 vlm_processor.tokenizer("successful", add_special_tokens=False).input_ids, 
                 vlm_processor.tokenizer("successfully", add_special_tokens=False).input_ids, 
                 vlm_processor.tokenizer("completed", add_special_tokens=False).input_ids,
                 vlm_processor.tokenizer("procedure", add_special_tokens=False).input_ids]

generation_kwargs = {
    "do_sample": False,
    "num_beams": args.num_beams,
    "num_return_sequences": args.num_return_sequences,
    "constraints": question_generation_constraints,
    "begin_suppress_tokens": begin_suppress_tokens,   
    "bad_words_ids": bad_words_ids, 
    "pad_token_id": tokenizer.eos_token_id,
    "length_penalty": args.length_penalty,
}

# NLI model to score consistency and verifiability
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)


# Load approopriate evaluation dataset
dataset = None
for retry in range(5):
    print(f"({worker_index}) Loading evaluation dataset (try {retry})...")
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


print(f"({worker_index}) Beginning iterative VQA inference...")
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

cache_path = os.path.join(this_results_dir, f"cached_outputs{worker_index}.pkl")
is_complete = False
last_batch_idx = -1
if os.path.exists(cache_path):
    is_complete, last_batch_idx, all_questions, all_candidate_questions, all_candidate_questions_scores, all_candidate_questions_sources, all_scores, all_answers, all_answer_probs, all_success_probs, all_success_probs_negated, all_example_ids, all_procedures, all_labels = pickle.load(open(cache_path, "rb"))

batch_idx = None
if not is_complete:
    for batch_idx, batch_examples in tqdm(enumerate(dataset.get_batches(IMAGES_CHUNK_SIZE, 
                                                                        n_workers=n_workers, 
                                                                        worker_index=worker_index,
                                                                        load_frames=False)), 
                                                    desc="running iterative VQA inference"):

        # If already in cache, skip this batch
        if batch_idx <= last_batch_idx:
            continue    

        # Take first frame (expect there to only be one frame)
        batch_procedures = [example.procedure_description for example in batch_examples]
        batch_frames = [Image.open(example.frames[0]) for example in batch_examples]

        this_batch_size = len(batch_examples)

        prompts = [
            f'{DIALOG_START_TOKENS[args.vlm_name]}{USER_START_TOKENS[args.vlm_name]}{IMAGE_TOKENS[args.vlm_name]}{IVQA_PREAMBLE.format(procedure=procedure)}' 
            for procedure in batch_procedures
        ]
        if args.print_prompts:
            pprint(prompts[0])
        questions = [[] for _ in range(this_batch_size)]
        frames = [[] for _ in range(this_batch_size)]
        candidate_questions = [[] for _ in range(this_batch_size)]
        candidate_questions_scores = [[] for _ in range(this_batch_size)]
        candidate_questions_sources = [[] for _ in range(this_batch_size)]
        scores = [[] for _ in range(this_batch_size)]
        answer_probs = [[] for _ in range(this_batch_size)] 
        answers = [[] for _ in range(this_batch_size)]
        success_probs = [[] for _ in range(this_batch_size)]
        success_probs_negated = [[] for _ in range(this_batch_size)]

        # Iteratively generate questions
        for question_idx in tqdm(range(args.max_iterations), desc="running iterative QA"):

            # If we have an adapter available for VQG, enable it (this should only be used for the dialog-based VQG, not in-context learning)
            if args.vqg_adapter_path is not None:
                lm.enable_adapters()

            # Generate a question (with beam search so we have several candidates)
            prompts_q = [prompt + f"{ASSISTANT_END_TOKENS[args.vlm_name] if question_idx != 0 else USER_END_TOKENS[args.vlm_name]}{USER_START_TOKENS[args.vlm_name]}Q:" for prompt in prompts]
            if not args.condition_questions_with_frames:
                new_questions, _ = simple_lm_prompt_beam_search(lm,
                                                                tokenizer,
                                                                [prompt.replace(IMAGE_TOKENS[args.vlm_name], "") for prompt in prompts_q],
                                                                max_new_tokens=20,
                                                                batch_size=max(args.generation_batch_size // (2 ** question_idx), 1),
                                                                generation_kwargs=generation_kwargs)
            else:
                new_questions = simple_vlm_prompt_beam_search(vlm,
                                                              vlm_processor,
                                                              prompts_q,
                                                              batch_frames,
                                                              IMAGE_TOKENS[args.vlm_name],
                                                              max_new_tokens=20,
                                                              batch_size=max(args.generation_batch_size // (2 ** question_idx), 1),
                                                              generation_kwargs=generation_kwargs)
            new_questions = [[cleanup_generated_question(question) for question in beam_search_questions] for beam_search_questions in new_questions]                                
            new_questions_sources = [["vlm"] * len(beam_search_questions) for beam_search_questions in new_questions]

            if args.vqg_adapter_path is not None:
                lm.disable_adapters()

            # Optionally inject more candidates from original VQG ICL code
            if args.n_icl_demonstrations > 0:
                icl_prompts = [generate_vqg_prompt_icl(procedure, args.n_icl_demonstrations, include_answers=False) for procedure in batch_procedures] # Create ICL prompt
                icl_prompts = [
                    prompt + '\n'.join([str(pqi+1) + ' ' + pq for pqi, pq in enumerate(previous_questions[-2:])]) + ("\n" if len(previous_questions) > 0 else "") + f"{len(previous_questions) + 1}. " 
                    for prompt, previous_questions in zip(icl_prompts, questions)
                ] # Add some previous questions if possible (take last 2 that were asked)
                icl_new_questions, _ = simple_lm_prompt_beam_search(lm,
                                                                    tokenizer,
                                                                    icl_prompts,
                                                                    max_new_tokens=20,
                                                                    batch_size=max(args.generation_batch_size // args.n_icl_demonstrations, 1),
                                                                    generation_kwargs=generation_kwargs)
                
                icl_new_questions = [[cleanup_generated_question(question) for question in beam_search_questions] for beam_search_questions in icl_new_questions]
                
                for batch_sub_idx in range(this_batch_size):
                    new_questions[batch_sub_idx] += icl_new_questions[batch_sub_idx]
                    new_questions_sources[batch_sub_idx] += ["icl"] * len(icl_new_questions[batch_sub_idx])

            # Remove duplicate candidates
            keep_idxs = [[question_idx for question_idx, question in enumerate(beam_search_outputs) if question not in beam_search_outputs[:question_idx]] for beam_search_outputs in new_questions]

            # Try to remove any candidates that we've seen before (if we've seen all the candidates before, don't remove any)
            keep_idxs_filtered = [[question_idx for question_idx, question in enumerate(beam_search_outputs) if question_idx in keep_idxs[batch_sub_idx] and question not in questions[batch_sub_idx]] for batch_sub_idx, beam_search_outputs in enumerate(new_questions)]
            keep_idxs = [keep_idxs_filtered[batch_sub_idx] if len(keep_idxs_filtered[batch_sub_idx]) > 0 else keep_idxs[batch_sub_idx] for batch_sub_idx in range(this_batch_size)]

            # Apply kept indices to new questions and their sources
            new_questions = [[new_questions[batch_sub_idx][question_idx] for question_idx in this_keep_idxs] for batch_sub_idx, this_keep_idxs in enumerate(keep_idxs)]
            new_questions_sources = [[new_questions_sources[batch_sub_idx][question_idx] for question_idx in this_keep_idxs] for batch_sub_idx, this_keep_idxs in enumerate(keep_idxs)]

            # Save all candidates from beam search
            for batch_sub_idx in range(len(candidate_questions)):
                candidate_questions[batch_sub_idx].append(new_questions[batch_sub_idx])
                candidate_questions_sources[batch_sub_idx].append(new_questions_sources[batch_sub_idx])

            # Select best candidate question from pool
            if args.question_selection_strategy == "likelihood":

                # First recalculate likelihood of each cleaned up question in iterative VQA context

                # (this is mostly important when we're using ICL injected questions)
                # NOTE: for some (unknown at this time) reason, LLaVA's LM likelihoods calculated during beam search (returned by output.scores) do not
                # match the likelihoods we get from a forward pass (which is unlike other models, e.g., T5). As such, we need to recompute completion
                # log likelihoods no matter what when using the "likelihood" candidate selection strategy. 
                # 
                # While this doesn't seem to affect the ordering of contextually generated candidates, the scores used here are slightly different. It's also 
                # worth noting that this step is especially important when ranking ICL-generated candidates among contextually generated candidates, as they may 
                # reasonably have higher likelihood than beam search candidates, but we need to be sure scores are calculated consistently. Further, since 
                # questions aren't "cleaned up" in initial generation, scores can be affected by this and at minimum need to be clipped at "?" token that ends
                # each question.
                # 
                # Some relevant discussion on this issue here: https://discuss.huggingface.co/t/compute-log-probabilities-of-any-sequence-provided/11710/10
                if not args.condition_questions_with_frames:
                    if "-t5-" not in args.vlm_name:
                        generation_scores = compute_completion_log_likelihoods(lm, tokenizer, [prompt.replace(IMAGE_TOKENS[args.vlm_name], "") for prompt in prompts_q], new_questions, batch_size=max(args.generation_batch_size // max(args.n_icl_demonstrations, 1), 1))
                    else:
                        generation_scores = compute_completion_log_likelihoods_encoder_decoder(lm, tokenizer, [prompt.replace(IMAGE_TOKENS[args.vlm_name], "") for prompt in prompts_q], new_questions, batch_size=max(args.generation_batch_size // max(args.n_icl_demonstrations, 1), 1))
                else:
                    generation_scores = compute_completion_log_likelihoods_vlm(vlm, vlm_processor, prompts_q, batch_frames, new_questions, batch_size=max(args.generation_batch_size // max(args.n_icl_demonstrations, 1), 1))

                # Select most likely question (first one in list)
                selected_questions = []
                new_scores = []
                for batch_sub_idx, (beam_search_questions, beam_search_scores) in enumerate(zip(new_questions, generation_scores)):                    
                    assert len(beam_search_questions) == len(beam_search_scores), "Expected candidate questions and their scores to have the same shape!"

                    # Save all candidate scores
                    candidate_questions_scores[batch_sub_idx].append(beam_search_scores)

                    candidate_idxs = list(range(len(beam_search_questions)))

                    # Then pick candidate with highest score
                    best_candidate = max(candidate_idxs, key=lambda x: beam_search_scores[x] == max(beam_search_scores))
                    selected_questions.append(beam_search_questions[best_candidate])
                    new_scores.append(beam_search_scores[best_candidate])

                new_questions = selected_questions

            elif args.question_selection_strategy in ["relevance", "informativeness", "coherence"]:
                # Calculate coherence metrics for each candidate question
                nli_outputs = question_coherence_metrics_nli(
                    nli_tokenizer, 
                    nli_model,
                    tokenizer,
                    lm,
                    [procedure for procedure, beam_search_questions in zip(batch_procedures, new_questions) for _ in beam_search_questions],
                    [question for beam_search_questions in new_questions for question in beam_search_questions],
                    previous_questions=[[q for qi, q in enumerate(batch_idx_questions) if batch_idx_answers[qi] != "Unsure"] for batch_idx_questions, batch_idx_answers, beam_search_questions in zip(questions, answers, new_questions) for _ in beam_search_questions],
                    previous_answers=[[a for a in batch_idx_answers if a != "Unsure"] for batch_idx_answers, beam_search_questions in zip(answers, new_questions) for _ in beam_search_questions],
                    rephrase_batch_size=args.generation_batch_size
                )

                # Select best candidate based on coherence metrics
                selected_questions = []
                new_scores = []
                parallel_idx = 0
                ranking_key_mapping = {
                    "relevance": "relevance_marginal",
                    "informativeness": "informativeness_marginal",
                    "coherence": "informativeness_marginal_x_relevance_marginal",
                }
                for batch_sub_idx, beam_search_questions in enumerate(new_questions):
                    this_nli_outputs = [{k: round(float(nli_outputs[k][i]), 3) if type(nli_outputs[k][i]) != str else nli_outputs[k][i] for k in nli_outputs} for i in range(parallel_idx, parallel_idx + len(beam_search_questions))]
                    candidate_questions_scores[batch_sub_idx].append(this_nli_outputs)
                    parallel_idx += len(beam_search_questions)

                    # Use marginal relevance (consistency) and expected informativeness (verifiability) to rank candidates
                    candidate_scores = np.array(
                        [candidate_metrics[ranking_key_mapping[args.question_selection_strategy]] for candidate_metrics in this_nli_outputs]
                    )

                    best_candidate = np.argmax(candidate_scores)
                    selected_questions.append(beam_search_questions[best_candidate])
                    new_scores.append(round(float(candidate_scores[best_candidate]), 6))
                
                new_questions = selected_questions
                    
            # Save scores for best questions
            for batch_sub_idx in range(this_batch_size):
                scores[batch_sub_idx].append(new_scores[batch_sub_idx])

            # Save generated questions
            for batch_sub_idx in range(this_batch_size):
                questions[batch_sub_idx].append(new_questions[batch_sub_idx])

            # Run VQA with generated questions (and optional spatial filter)
            prompts_a = [prompt + f' {question}{USER_END_TOKENS[args.vlm_name]}{ASSISTANT_START_TOKENS[args.vlm_name]}A:' for prompt, question in zip(prompts_q, new_questions)]
            if args.print_prompts:
                pprint(prompts_a[0])

            # Effective prompt for VQA depends on whether we want to exclude dialog history from prompt
            if not args.exclude_history_from_vqa:
                use_prompts_a = prompts_a
            else:
                use_prompts_a = [f'{USER_START_TOKENS[args.vlm_name]}{IMAGE_TOKENS[args.vlm_name]}Q: {question}{USER_END_TOKENS[args.vlm_name]}{ASSISTANT_START_TOKENS[args.vlm_name]}A:' for prompt, question in zip(prompts_q, new_questions)]

            new_answers_logits = run_vqa_with_visual_filter(vlm_processor=vlm_processor, 
                                                            vlm=vlm, 
                                                            batch_examples=batch_examples, 
                                                            batch_frames=batch_frames, 
                                                            prompts_a=use_prompts_a, 
                                                            new_questions=new_questions, 
                                                            question_idx=question_idx,
                                                            batch_size=max(args.vqa_batch_size // (2 ** question_idx), 1) if not args.exclude_history_from_vqa else args.vqa_batch_size,
                                                            visual_filter=visual_filter,
                                                            nlp=nlp,
                                                            visual_filter_mode=VisualFilterTypes(args.visual_filter_mode) if visual_filter else None,
                                                            frame_cache_dir=this_results_dir if args.cache_vqa_frames else None,
                                                            is_encoder_decoder="-t5-" in args.vlm_name.lower())

            # Gather up VQA outputs (which automatically calculates answer probabilities from logits)
            new_answers = [
                VQAOutputs(
                    task_name=MistakeDetectionTasks(args.task),
                    example_id=example.example_id,
                    procedure_id=example.procedure_id,
                    frame=example.frames[0],
                    prompt=prompt,
                    expected_answer=None,
                    response_token_ids=response_token_ids,
                    logits=logits,
                    question=question,
                ) for logits, example, prompt, question in zip(new_answers_logits, batch_examples, prompts_a, new_questions)
            ]
            new_answers_str = [output.predicted_answer.name if np.abs(output.answer_probs[VQAResponse.Yes] - 0.5) >= args.unsure_range else "Unsure" for output in new_answers]

            # Save answers and their probabilities
            for batch_sub_idx in range(this_batch_size):
                answer_probs[batch_sub_idx].append([round(float(new_answers[batch_sub_idx].answer_probs[VQAResponse(answer_idx)]), 6) for answer_idx in range(2)])
                answers[batch_sub_idx].append(new_answers_str[batch_sub_idx])
            
            
            prompts = [prompt + " " + output for prompt, output in zip(prompts_a, new_answers_str)]

            # Ask VLM probability of success
            questions_success = [
                IVQA_SUCCESS_QUESTION.format(procedure=procedure)
                for procedure in batch_procedures
            ]
            prompts_success = [
                prompt + f'{ASSISTANT_END_TOKENS[args.vlm_name]}{USER_START_TOKENS[args.vlm_name]}Q: {question}{USER_END_TOKENS[args.vlm_name]}{ASSISTANT_START_TOKENS[args.vlm_name]}A: '
                for prompt, question in zip(prompts, questions_success)
            ]
            if args.print_prompts:
                pprint(prompts_success[0])
            success_vqa_outputs = run_vqa_with_visual_filter(vlm_processor=vlm_processor, 
                                                             vlm=vlm, 
                                                             batch_examples=batch_examples, 
                                                             batch_frames=batch_frames, 
                                                             prompts_a=prompts_success, 
                                                             new_questions=questions_success, 
                                                             question_idx=f"{question_idx}_success",
                                                             batch_size=max(args.vqa_batch_size // (2 ** question_idx), 1),
                                                             visual_filter=visual_filter if visual_filter and VisualFilterTypes(args.visual_filter_mode) not in [VisualFilterTypes.Spatial_NoRephrase, VisualFilterTypes.Spatial_Blur] else None, # Don't use spatial filter for SuccessVQA step, since this may remove important information
                                                             nlp=nlp,
                                                             visual_filter_mode=VisualFilterTypes(args.visual_filter_mode) if visual_filter else None,
                                                             frame_cache_dir=this_results_dir if args.cache_vqa_frames else None,
                                                             is_encoder_decoder="-t5-" in args.vlm_name.lower())
            success_vqa_outputs = [
                VQAOutputs(
                    task_name=MistakeDetectionTasks(args.task),
                    example_id=example.example_id,
                    procedure_id=example.procedure_id,
                    frame=example.frames[0],
                    prompt=prompt,
                    expected_answer=None,
                    response_token_ids=response_token_ids,
                    logits=logits,
                    question=question,
                ) for logits, example, prompt, question in zip(success_vqa_outputs, batch_examples, prompts_a, new_questions)
            ]               

            # Save success probability for this turn
            for batch_sub_idx in range(this_batch_size):
                success_probs[batch_sub_idx].append(
                    round(float(success_vqa_outputs[batch_sub_idx].answer_probs[VQAResponse.Yes]), 6)
                )

            # Clear out VQA outputs now because they occupy a lot of memory
            del new_answers
            del success_vqa_outputs


        # Update global lists of tracked outputs
        all_questions += questions
        all_candidate_questions += candidate_questions
        all_candidate_questions_scores += candidate_questions_scores
        all_candidate_questions_sources += candidate_questions_sources
        all_scores += scores
        all_answers += answers
        all_answer_probs += answer_probs
        all_success_probs += success_probs
        all_success_probs_negated += success_probs_negated
        all_example_ids += [example.example_id for example in batch_examples]
        all_procedures += [example.procedure_description for example in batch_examples]
        all_labels += [example.mistake_type for example in batch_examples]

        for frame in batch_frames:
            frame.close()
        del batch_frames

        # And cache tracked outputs
        pickle.dump((    
            False,
            batch_idx,
            all_questions, 
            all_candidate_questions, 
            all_candidate_questions_scores, 
            all_candidate_questions_sources,
            all_scores, 
            all_answers, 
            all_answer_probs, 
            all_success_probs,
            all_success_probs_negated,
            all_example_ids,
            all_procedures,
            all_labels,
        ), open(cache_path, "wb"))

# Verify we got correct number of outputs
all_results = [
    all_questions, 
    all_candidate_questions, 
    all_candidate_questions_scores, 
    all_candidate_questions_sources,
    all_scores, 
    all_answers, 
    all_answer_probs, 
    all_success_probs,
    all_success_probs_negated,
    all_example_ids,
    all_procedures,
    all_labels,
]
assert all(len(l) == len(all_results[0]) for l in all_results), f"Expected to get same number of all outputs! ({', '.join([str(len(l)) for l in all_results])})"

# Cache one more time to indicate the generation is finished
if batch_idx is not None:
    pickle.dump((    
        True,
        batch_idx,
        all_questions, 
        all_candidate_questions, 
        all_candidate_questions_scores, 
        all_candidate_questions_sources,
        all_scores, 
        all_answers, 
        all_answer_probs, 
        all_success_probs,
        all_success_probs_negated,
        all_example_ids,
        all_procedures,
        all_labels,
    ), open(cache_path, "wb"))

print(f"({worker_index}) Done running iterative VQA inference!")


# Gather up results across processes and evaluate
if worker_index == 0:
    print(f"({worker_index}) Gathering all results...")
    for other_worker_index in range(1, n_workers):
        print(f"({worker_index}) Gathering results from worker {other_worker_index}...")
        delay_per_try = 10
        delay_so_far = 0
        max_delay = 7200 if args.resume_dir is not None else 1800 # Allow a longer delay in case some processes are already finished in resumed run
        while True:
            other_cache_path = os.path.join(this_results_dir, f"cached_outputs{other_worker_index}.pkl")
            if os.path.exists(other_cache_path):
                is_complete, \
                _, \
                other_questions, \
                other_candidate_questions, \
                other_candidate_questions_scores, \
                other_candidate_questions_sources, \
                other_scores, \
                other_answers, \
                other_answer_probs, \
                other_success_probs, \
                other_success_probs_negated, \
                other_example_ids, \
                other_procedures, \
                other_labels = pickle.load(open(other_cache_path, "rb"))
                if is_complete:
                    # Add other process results to our results
                    all_questions += other_questions
                    all_candidate_questions += other_candidate_questions
                    all_candidate_questions_scores += other_candidate_questions_scores
                    all_candidate_questions_sources += other_candidate_questions_sources
                    all_scores += other_scores
                    all_answers += other_answers
                    all_answer_probs += other_answer_probs
                    all_success_probs += other_success_probs
                    all_success_probs_negated += other_success_probs_negated
                    all_example_ids += other_example_ids
                    all_procedures += other_procedures
                    all_labels += other_labels
                    print(f"({worker_index}) Collected results from worker {other_worker_index}.")
                    break

            # Decide whether to try again
            if delay_so_far >= max_delay:
                raise TimeoutError(f"Waited for {max_delay} seconds for results from worker {other_worker_index}. Process may have failed.")
            print(f"({worker_index}) Still waiting for results from worker {other_worker_index} ({delay_so_far} sec.)!")
            time.sleep(delay_per_try)
            delay_so_far += delay_per_try

    # Collect key information from results rollouts and final success probabilities after n iterations
    all_results_dicts = {}
    all_probs = []
    for questions, candidate_questions, candidate_questions_scores, candidate_questions_sources, scores, answers, answer_probs, success_probs, success_probs_negated, example_id, procedure, label \
        in tqdm(zip(all_questions,
                    all_candidate_questions,
                    all_candidate_questions_scores,
                    all_candidate_questions_sources,
                    all_scores,
                    all_answers,
                    all_answer_probs,
                    all_success_probs,
                    all_success_probs_negated,
                    all_example_ids,
                    all_procedures,
                    all_labels), desc="compiling results"):
        
        final_success_prob = success_probs[args.max_iterations - 1]
        all_probs.append(round(final_success_prob, 6))   

        results_dict = {
            "procedure": procedure,
            "mistake": True if label is not None else False,
            "mistake_type": label,
            "questions": questions,
            "frame_dir": os.path.join(this_results_dir, f"vqa_frames/{example_id}") if args.cache_vqa_frames else dataset.get_example_dir(example_id),
            "answers": answers,
            "answer_probs": answer_probs,
            "scores": scores,
            "success_probs": success_probs,
            "success_probs_negated": success_probs_negated,
            "final_turn": args.max_iterations - 1,
            "final_success_prob": final_success_prob,
            "candidate_questions": candidate_questions,
            "candidate_questions_scores": candidate_questions_scores,
            "candidate_questions_sources": candidate_questions_sources,
        }
        all_results_dicts[example_id] = results_dict

    json.dump(all_results_dicts, 
            open(os.path.join(this_results_dir, f"outputs_{args.eval_partition}.json"), "w"),
            indent=4)


    print(f"({worker_index}) Evaluating outputs...")
    metrics = {}

    # Calculate coherence metrics of full rollouts
    all_chosen_questions = [question for results_dict in all_results_dicts.values() for question in results_dict['questions'][:results_dict['final_turn'] + 1]]
    all_previous_questions = [[q for qi, q in enumerate(results_dict['questions'][:question_idx]) if results_dict['answers'][qi] != "Unsure"] for results_dict in all_results_dicts.values() for question_idx in range(results_dict['final_turn'] + 1)]

    label_answer_mapping = {0: "No", 1: "Yes"}
    all_predicted_answers = [label_answer_mapping[np.argmax(answer_probs)] for results_dict in all_results_dicts.values() for answer_probs in results_dict['answer_probs'][:results_dict['final_turn'] + 1]]
    all_previous_answers = [[a for a in results_dict['answers'][:question_idx] if a != "Unsure"] for results_dict in all_results_dicts.values() for question_idx in range(results_dict['final_turn'] + 1)]

    all_coherence_metrics = question_coherence_metrics_nli(nli_tokenizer,
                                            nli_model,
                                            tokenizer,
                                            lm,                                         
                                            [procedure for results_dict, procedure in zip(all_results_dicts.values(), all_procedures) for _ in range(results_dict['final_turn'] + 1)],
                                            all_chosen_questions,
                                            answers=all_predicted_answers,
                                            previous_questions=all_previous_questions,
                                            previous_answers=all_previous_answers,
                                            mistake_labels=[results_dict['mistake'] for results_dict in all_results_dicts.values() for _ in range(results_dict['final_turn'] + 1)],
                                            rephrase_batch_size=args.generation_batch_size)

    # Tune stopping criteria (early stop delta and confident range) to maximize accuracy and coherence of explanations
    tuning_results_dir = os.path.join(this_results_dir, "stopping_criteria_tuning")
    if not os.path.exists(tuning_results_dir):
        os.makedirs(tuning_results_dir)

    cand_max_iterations = [args.max_iterations]
    cand_early_stop_delta = args.early_stop_delta
    cand_confident_range = args.confident_range
    cand_criteria = product(cand_max_iterations, cand_early_stop_delta, cand_confident_range)

    best_performance = None

    performance_by_criteria = {}

    # print("Tuning stopping criteria over combinations:")
    # pprint(list(cand_criteria))
    
    for mi, esd, cd in tqdm(cand_criteria, desc="tuning stopping criteria"):
        all_probs = []
        all_labels = []
        all_procedures = []
        for example_id, output in tqdm(all_results_dicts.items(), desc="outputs"):
            final_success_prob = None
            success_probs = output['success_probs']
            for success_prob_idx, success_prob in enumerate(success_probs[:mi]): 
                # Early stopping mechanism: 
                # if success score doesn't change enough over 3 turns, stop incorporating questions
                # (we still run inference across all questions for efficiency and simplicity, but later can make a proper demo script)
                final_success_prob = success_prob
                if success_prob_idx >= 2 and success_prob_idx < len(success_probs) - 1:
                    if np.abs(success_probs[success_prob_idx-1] - success_probs[success_prob_idx-2]) < esd and np.abs(success_probs[success_prob_idx] - success_probs[success_prob_idx-1]) < esd:
                        break
                # OR if success score is within confident delta, stop
                if success_prob < cd or 1.0 - success_prob < cd:
                    break           
                    
            output['final_turn'] = success_prob_idx
            all_results_dicts[example_id] = output
            all_probs.append(final_success_prob)
            all_labels.append(output['mistake_type'])
            all_procedures.append(output['procedure'])
                    
        # Calculate coherence metrics of updated rollouts
        all_chosen_questions = [question for results_dict in all_results_dicts.values() for question in results_dict['questions'][:10]]
        all_previous_questions = [[q for qi, q in enumerate(results_dict['questions'][:question_idx]) if results_dict['answers'][qi] != "Unsure"] for results_dict in all_results_dicts.values() for question_idx in range(10)]

        label_answer_mapping = {0: "No", 1: "Yes"}
        all_predicted_answers = [label_answer_mapping[np.argmax(answer_probs)] for results_dict in all_results_dicts.values() for answer_probs in results_dict['answer_probs'][:10]]
        all_previous_answers = [[a for a in results_dict['answers'][:question_idx] if a != "Unsure"] for results_dict in all_results_dicts.values() for question_idx in range(10)]
            
        # Adjust all_coherence_metrics for the specific final turns we chose here
        readjusted_all_coherence_metrics = {}
        for k in all_coherence_metrics:
            parallel_idx = 0
            this_metrics = []
            for results_dict in all_results_dicts.values():
                for question_idx in range(args.max_iterations):

                    # Skip over the turns we don't want for this set of criteria
                    if question_idx > results_dict['final_turn']:
                        parallel_idx += 1
                        continue
                    else:
                        if type(k) != str:
                            this_metrics.append(max(round(float(all_coherence_metrics[k][parallel_idx]), 6), 0.0)) # If negative, just round up to 0.0 for aggregated metrics
                        else:
                            this_metrics.append(all_coherence_metrics[k][parallel_idx])
                        parallel_idx += 1
                    
            readjusted_all_coherence_metrics[k] = this_metrics
        
        # Compile accuracy and coherence metrics
        accuracy_metrics_by_threshold, coherence_metrics, other_metrics = compile_accuracy_and_coherence_metrics(all_labels, all_probs, readjusted_all_coherence_metrics, all_results_dicts, MISTAKE_DETECTION_THRESHOLDS, 0.1)
        coherence_metrics_by_threshold = coherence_metrics['metrics_by_threshold']
        
        performance_by_criteria[str((mi, esd, cd))] = {
            "accuracy": accuracy_metrics_by_threshold['best_metrics']['accuracy'],
            "consistency": coherence_metrics_by_threshold[accuracy_metrics_by_threshold['best_threshold']]['consistency'],
            "verifiability": coherence_metrics_by_threshold[accuracy_metrics_by_threshold['best_threshold']]['verifiability'],
            "relevance_marginal": coherence_metrics['relevance_marginal'],
            "informativeness_marginal": coherence_metrics['informativeness_marginal'],
            "informativeness_marginal_ref": coherence_metrics['informativeness_marginal_ref'],
        }
        this_performance = performance_by_criteria[str((mi, esd, cd))]["verifiability"]
        if best_performance is None or this_performance > best_performance:
            best_performance = this_performance
            best_metrics = (accuracy_metrics_by_threshold, readjusted_all_coherence_metrics, coherence_metrics, coherence_metrics_by_threshold, other_metrics, deepcopy(all_results_dicts))
            best_criteria = (mi, esd, cd)

        # Save info for this combo
        subdir_path = os.path.join(tuning_results_dir, f"{mi}_{esd}_{cd}")
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        json.dump(all_results_dicts, 
                open(os.path.join(subdir_path, f"outputs_{args.eval_partition}.json"), "w"),
                indent=4)    
        
        json.dump(accuracy_metrics_by_threshold, 
                open(os.path.join(subdir_path, f"metrics_accuracy_{args.eval_partition}.json"), "w"),
                indent=4)

        json.dump(coherence_metrics, 
                open(os.path.join(subdir_path, f"metrics_coherence_nli_{args.eval_partition}.json"), "w"),
                indent=4)

        json.dump(readjusted_all_coherence_metrics, 
                open(os.path.join(subdir_path, f"metrics_coherence_raw_nli_{args.eval_partition}.json"), "w"),
                indent=4)          

        json.dump(other_metrics, 
                open(os.path.join(subdir_path, f"metrics_other_{args.eval_partition}.json"), "w"),
                indent=4)              

    # Save tuning results
    mi, esd, cd = best_criteria
    json.dump({"max_iterations": mi, "early_stop_delta": esd, "confident_range": cd},
            open(os.path.join(tuning_results_dir, "tuned_stopping_criteria.json"), "w"),
            indent=4,
    )

    json.dump(performance_by_criteria, open(os.path.join(tuning_results_dir, "performance_by_criteria.json"), "w"), indent=4)

    # Grab best metrics and save them in main results directory
    accuracy_metrics_by_threshold, readjusted_all_coherence_metrics, coherence_metrics, coherence_metrics_by_theshold, other_metrics, all_results_dicts = best_metrics

    json.dump(all_results_dicts, 
            open(os.path.join(this_results_dir, f"outputs_{args.eval_partition}.json"), "w"),
            indent=4)

    json.dump(accuracy_metrics_by_threshold, 
            open(os.path.join(this_results_dir, f"metrics_accuracy_{args.eval_partition}.json"), "w"),
            indent=4)
    
    json.dump(coherence_metrics, 
            open(os.path.join(this_results_dir, f"metrics_coherence_nli_{args.eval_partition}.json"), "w"),
            indent=4)

    json.dump(all_coherence_metrics, 
            open(os.path.join(this_results_dir, f"metrics_coherence_raw_nli_{args.eval_partition}.json"), "w"),
            indent=4)        

    json.dump(other_metrics, 
            open(os.path.join(this_results_dir, f"metrics_other_{args.eval_partition}.json"), "w"),
            indent=4)        
    
    # Grab metrics that go in results tables
    table_metrics = {
        "accuracy": accuracy_metrics_by_threshold['best_metrics']['accuracy'],
        "relevance": coherence_metrics['relevance_marginal'],
        "informativeness": coherence_metrics['informativeness_marginal'],
        "n_iterations": other_metrics['n_iterations'],
        "info_gain": other_metrics['dialog_info_gain'],
    }
    json.dump(table_metrics, 
              open(os.path.join(this_results_dir, f"metrics_table_{args.eval_partition}.json"), "w"),
              indent=4)

    # Generate DET curves for accuracy
    generate_det_curve(accuracy_metrics_by_threshold, os.path.join(this_results_dir, f"det_accuracy_{args.eval_partition}.pdf"))

    # Generate 3D scatter plot of decision error, relevance, and informativeness
    graph_name = args.question_selection_strategy
    if args.n_icl_demonstrations > 0:
        graph_name += f"_icl{args.n_icl_demonstrations}"
    if args.vqg_adapter_path is not None:
        graph_name += "_dpo"
    generate_3d_overview_graph(coherence_metrics, all_results_dicts, dataset, this_results_dir, graph_name=graph_name)

    # Save args and config
    shutil.copy(CONFIG_PATH, os.path.join(this_results_dir, "config.yml"))
    json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)

    print(f"({worker_index}) Done!")