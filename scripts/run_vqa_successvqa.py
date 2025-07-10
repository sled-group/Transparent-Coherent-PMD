from travel import init_travel
init_travel()

import argparse
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
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig       

from travel.constants import RESULTS_DIR, IMAGES_CHUNK_SIZE, HF_TOKEN, CONFIG_PATH
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.data.vqa import VQAResponse, get_vqa_response_token_ids, VQAOutputs, DIALOG_START_TOKENS, IMAGE_TOKENS, USER_START_TOKENS, USER_END_TOKENS, ASSISTANT_START_TOKENS, ASSISTANT_END_TOKENS, SVQA_PREAMBLE, SVQA_SUCCESS_QUESTION
from travel.model.grounding import VisualFilterTypes, ContrastiveRegionFilter, VisualContrastiveFilter, SpatialVisualFilter, ImageMaskTypes
from travel.model.metrics import mistake_detection_metrics, generate_det_curve, entropy
from travel.model.mistake_detection import MISTAKE_DETECTION_THRESHOLDS
from travel.model.vqa import run_vqa_with_visual_filter


parser = argparse.ArgumentParser()
parser.add_argument("--vlm_name", type=str, default=None, help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--hf_hub_revision", type=str, default=None, help="Optional revision ID for VLM in Hugging Face Hub.")
parser.add_argument("--task", type=str, default="ego4d_single", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--eval_partition", type=str, default="val", choices=["train", "val", "test"])
parser.add_argument("--visual_filter_mode", type=str, required=False, choices=[t.value for t in VisualFilterTypes], help="Visual attention filter mode.")
parser.add_argument("--visual_filter_strength", type=float, required=False, default=1.0, help="Float strength for masks used in visual filters. Depending on the visual filter type, this may be interpreted as a percentage darkness or a Gaussian blur kernel size.")
parser.add_argument("--vqa_batch_size", type=int, default=10, help="Batch size for VQA with VLM.")
parser.add_argument("--run_id", type=str, required=False, help="Unique ID for this run, which will be used to create the output directory (and should be shared across any parallel processes).")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of iterative VQA.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
parser.add_argument("--cache_vqa_frames", action="store_true", help="Pass this argument to cache frames in VQA outputs (e.g., to inspect visual filter resuilts). This consumes a lot of disk space for large datasets.")
parser.add_argument("--print_prompts", action="store_true", help="Pass this argument to print some sample prompts during execution (for debugging purposes).")
args = parser.parse_args()

print("ARGUMENTS:")
pprint(args)
print('\n')

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
        task_name += f"_debug{args.debug_n_examples}"
    this_results_dir = os.path.join(task_name, vlm_name, f"SuccessVQA_{task_name}")
    this_results_dir += f"_{vlm_name}"
    if args.visual_filter_mode is not None:
        this_results_dir += f"_{args.visual_filter_mode}{args.visual_filter_strength}"
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
    else:
        raise NotImplementedError(f"Visual filter type {args.visual_filter_mode} is not compatible with iterative VQA!")

# Load approopriate evaluation dataset
dataset = None
for retry in range(5):
    print(f"({worker_index}) Loading evaluation dataset (try {retry})...")
    try:
        if MistakeDetectionTasks(args.task) == MistakeDetectionTasks.Ego4D_Single:
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
all_success_probs = []

all_example_ids = []
all_procedures = []
all_labels = []

cache_path = os.path.join(this_results_dir, f"cached_outputs{worker_index}.pkl")
is_complete = False
last_batch_idx = -1
if os.path.exists(cache_path):
    is_complete, last_batch_idx, all_success_probs, all_example_ids, all_procedures, all_labels = pickle.load(open(cache_path, "rb"))

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
            f'{DIALOG_START_TOKENS[args.vlm_name]}{USER_START_TOKENS[args.vlm_name]}{IMAGE_TOKENS[args.vlm_name]}{SVQA_PREAMBLE.format(procedure=procedure)}' 
            for procedure in batch_procedures
        ]
        if args.print_prompts:
            pprint(prompts[0])
        frames = [[] for _ in range(this_batch_size)]
        success_probs = [[] for _ in range(this_batch_size)]

        # Ask VLM probability of success
        questions_success = [
            SVQA_SUCCESS_QUESTION.format(procedure=procedure)
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
                                                            question_idx=f"-1_success",
                                                            batch_size=args.vqa_batch_size,
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
            ) for logits, example, prompt, question in zip(success_vqa_outputs, batch_examples, prompts_success, questions_success)
        ]               

        # Save success probability for this turn
        for batch_sub_idx in range(this_batch_size):
            success_probs[batch_sub_idx].append(
                round(float(success_vqa_outputs[batch_sub_idx].answer_probs[VQAResponse.Yes]), 6)
            )

        # Update global lists of tracked outputs
        all_success_probs += success_probs
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
            all_success_probs,
            all_example_ids,
            all_procedures,
            all_labels,
        ), open(cache_path, "wb"))

# Verify we got correct number of outputs
all_results = [
    all_success_probs,
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
        all_success_probs,
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
        max_delay = 7200 if args.resume_dir is not None else 7200 # Allow a longer delay in case some processes are already finished in resumed run
        while True:
            other_cache_path = os.path.join(this_results_dir, f"cached_outputs{other_worker_index}.pkl")
            if os.path.exists(other_cache_path):
                is_complete, \
                _, \
                other_success_probs, \
                other_example_ids, \
                other_procedures, \
                other_labels = pickle.load(open(other_cache_path, "rb"))
                if is_complete:
                    # Add other process results to our results
                    all_success_probs += other_success_probs
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
    for success_probs, example_id, procedure, label \
        in tqdm(zip(all_success_probs,
                    all_example_ids,
                    all_procedures,
                    all_labels), desc="compiling results"):
        
        final_success_prob = success_probs[0]
        all_probs.append(round(final_success_prob, 6))   

        results_dict = {
            "procedure": procedure,
            "mistake": True if label is not None else False,
            "mistake_type": label,
            "questions": None,
            "frame_dir": os.path.join(this_results_dir, f"vqa_frames/{example_id}") if args.cache_vqa_frames else dataset.get_example_dir(example_id),
            "answers": None,
            "answer_probs": None,
            "scores": None,
            "success_probs": success_probs,
            "success_probs_negated": None,
            "final_turn": -1,
            "final_success_prob": final_success_prob,
            "candidate_questions": None,
            "candidate_questions_scores": None,
            "candidate_questions_sources": None,
        }
        all_results_dicts[example_id] = results_dict

    json.dump(all_results_dicts, 
            open(os.path.join(this_results_dir, f"outputs_{args.eval_partition}.json"), "w"),
            indent=4)


    print(f"({worker_index}) Evaluating...")
    metrics = {}

    accuracy_metrics_by_threshold = {}
    all_labels_binary = [True if l is not None else False for l in all_labels]
    best_metrics = None
    for threshold in MISTAKE_DETECTION_THRESHOLDS:
        preds = [1.0 - p >= threshold for p in all_probs] # Have to do 1.0 - probability since we got "success" probability from VLM
        assert len(preds) == len(all_probs) == len(all_labels), "Expected same number of preds, probs, and labels."
        this_metrics = mistake_detection_metrics(all_labels_binary, preds)
        accuracy_metrics_by_threshold[threshold] = this_metrics
        if best_metrics is None or (this_metrics['false_positive_rate'] + this_metrics['false_negative_rate']) < (best_metrics['false_positive_rate'] + best_metrics['false_negative_rate']):
            best_metrics = this_metrics
            best_threshold = threshold
    accuracy_metrics_by_threshold['best_metrics'] = best_metrics
    accuracy_metrics_by_threshold['best_threshold'] = best_threshold

    json.dump(accuracy_metrics_by_threshold, 
             open(os.path.join(this_results_dir, f"metrics_accuracy_{args.eval_partition}.json"), "w"),
             indent=4)
        
    # Grab metrics that go in results tables
    table_metrics = {
        "accuracy": accuracy_metrics_by_threshold['best_metrics']['accuracy'],
        "relevance": None,
        "informativeness": None,
        "n_iterations": 0.0,
        "info_gain": np.mean([1.0 - entropy(p) for p in all_probs]),
    }
    json.dump(table_metrics, 
              open(os.path.join(this_results_dir, f"metrics_table_{args.eval_partition}.json"), "w"),
              indent=4)

    # Generate DET curves for accuracy
    generate_det_curve(accuracy_metrics_by_threshold, os.path.join(this_results_dir, f"det_accuracy_{args.eval_partition}.pdf"))

    # Save args and config
    shutil.copy(CONFIG_PATH, os.path.join(this_results_dir, "config.yml"))
    json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)

    print(f"({worker_index}) Done!")