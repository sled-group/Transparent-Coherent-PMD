from travel import init_travel
init_travel()

import argparse
from collections import defaultdict
import json
import numpy as np
import os
import pickle
from pprint import pprint
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer
                         
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.data.vqa import VQAResponse, get_vqa_response_token_ids, VQAOutputs, IMAGE_TOKENS, USER_START_TOKENS, USER_END_TOKENS, ASSISTANT_START_TOKENS, ASSISTANT_END_TOKENS
from travel.model.metrics import mistake_detection_metrics, question_coherence_metrics_nli, generate_det_curve, generate_tiered_metric_curves
from travel.model.mistake_detection import MISTAKE_DETECTION_THRESHOLDS
from travel.model.nli import NLI_MODEL_PATH
from travel.model.vqa import run_qa 


parser = argparse.ArgumentParser()
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=["llava-hf/llava-1.5-7b-hf"], help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--task", type=str, default="ego4d_single", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--results_dir", type=str, help="Path to results directory for previous completed run of iterative VQA.")
parser.add_argument("--eval_partition", type=str, default="val", choices=["val", "test"])
parser.add_argument("--question_selection_strategy", type=str, default="likelihood", choices=["likelihood", "coherence"], help="Strategy to use to choose question to generate from beam search candidates.")
parser.add_argument("--early_stop_delta", type=int, default=0.1, help="If success probability changes less than this over 3 turns, stop generating questions.")
parser.add_argument("--generation_batch_size", type=int, default=10, help="Batch size for question generation with LM.")
parser.add_argument("--qa_batch_size", type=int, default=20, help="Batch size for QA with VLM.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
args = parser.parse_args()

assert torch.cuda.device_count() == 1, "Iterative VQA requires exactly 1 GPU per process; use `srun` to enable multi-GPU parallelization."

# Get parallelization details from srun if any
if "SLURM_PROCID" in os.environ and "SLURM_NPROCS" in os.environ:
    worker_index = int(os.environ["SLURM_PROCID"])
    n_workers = int(os.environ["SLURM_NPROCS"])
else:
    worker_index = 0
    n_workers = 1
# NOTE: must have the same number of GPUs as original run

# Set up results directory
this_results_dir = args.results_dir


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
    vlm = AutoModelForVision2Seq.from_pretrained(args.vlm_name, quantization_config=bnb_config)   
except:
    vlm = AutoModelForCausalLM.from_pretrained(args.vlm_name, quantization_config=bnb_config)
vlm_processor = AutoProcessor.from_pretrained(args.vlm_name)
vlm_processor.tokenizer.padding_side = "left"
response_token_ids = get_vqa_response_token_ids(vlm_processor.tokenizer)

# We'll use VLM's LM directly to generate questions
lm = vlm.language_model
tokenizer = vlm_processor.tokenizer
tokenizer.pad_token_id = tokenizer.eos_token_id

# NLI model to score consistency and verifiability
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)

print(f"({worker_index}) Beginning iterative VQA inference...")
cache_path = os.path.join(this_results_dir, f"cached_outputs{worker_index}.pkl")
is_complete, last_batch_idx, all_questions, all_frames, all_candidate_questions, all_candidate_questions_scores, all_candidate_questions_sources, all_scores, all_answers, all_answer_probs, all_success_probs, all_example_ids, all_procedures, all_labels = pickle.load(open(cache_path, "rb"))
assert is_complete, "Can only run noimg SuccessVQA on a completed run of iterative VQA."

n_questions_per_example = len(all_questions[0])
all_prompts = defaultdict(list)
for example_idx, (questions, frames, candidate_questions, candidate_questions_scores, candidate_questions_sources, scores, answers, answer_probs, success_probs, example_id, procedure, label) \
    in enumerate(tqdm(zip(all_questions,
                            all_frames,
                            all_candidate_questions,
                            all_candidate_questions_scores,
                            all_candidate_questions_sources,
                            all_scores,
                            all_answers,
                            all_answer_probs,
                            all_success_probs,
                            all_example_ids,
                            all_procedures,
                            all_labels), desc="running image-free success VQA")):
    
    assert len(questions) == n_questions_per_example
    question_success = f'Based on the above information, has the procedure "{procedure}" been successfully executed?'
    success_prompt = f'{ASSISTANT_END_TOKENS[type(vlm)]}{USER_START_TOKENS[type(vlm)]}Q: {question_success}{USER_END_TOKENS[type(vlm)]}{ASSISTANT_START_TOKENS[type(vlm)]}A:'

    # Iteratively generate questions
    for question_idx in tqdm(range(len(questions)), desc="running iterative QA"):
        prompt = f'{USER_START_TOKENS[type(vlm)]}{IMAGE_TOKENS[type(vlm)]}This is a photo of someone working on the procedure "{procedure}". I will ask a series of different yes/no questions about the state of the scene to determine whether the person has successfully executed the procedure. The goal is to extract as much relevant information as possible from the scene, so I will not repeat questions.'
        for question, answer in zip(questions[:question_idx + 1], answers[:question_idx + 1]):
            prompt += f"{ASSISTANT_END_TOKENS[type(vlm)] if question_idx != 0 else USER_END_TOKENS[type(vlm)]}{USER_START_TOKENS[type(vlm)]}Q: "
            prompt += f'{question}{USER_END_TOKENS[type(vlm)]}{ASSISTANT_START_TOKENS[type(vlm)]}A: {VQAResponse(answer).name}'

            all_prompts[question_idx].append(prompt + success_prompt)

# Update results dir to be a subdirectory of original results dir
this_results_dir = os.path.join(this_results_dir, "results_image_free_SuccessVQA")
if not os.path.exists(this_results_dir):
    os.makedirs(this_results_dir)

# Run QA and cache logits
cache_dir = os.path.join(this_results_dir, "noimg_qa_cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

logits = defaultdict(int)
for question_idx in range(n_questions_per_example):
    logits[question_idx] = run_qa(lm, 
                                  tokenizer, 
                                  all_prompts[question_idx], 
                                  batch_size=max(args.qa_batch_size // (2 ** question_idx), 1),
                                  cache_path=os.path.join(cache_dir, f"noimg_qa_logits{worker_index}-{question_idx}.pt"))

# Gather up results across processes and evaluate
if worker_index == 0:
    print(f"({worker_index}) Gathering all results...")
    for other_worker_index in range(1, n_workers):
        print(f"({worker_index}) Gathering results from worker {other_worker_index}...")
        for question_idx in len(logits):
            delay_per_try = 10
            delay_so_far = 0
            max_delay = 1800
            while True:
                other_cache_path = os.path.join(cache_dir, f"noimg_qa_logits{worker_index}-{question_idx}.pt")
                if os.path.exists(other_cache_path):
                    other_logits = torch.load(other_cache_path)
                    logits[question_idx] = torch.cat((logits[question_idx], other_logits), dim=0)

                # Decide whether to try again
                if delay_so_far >= max_delay:
                    raise TimeoutError(f"Waited for {max_delay} seconds for results from worker {other_worker_index}. Process may have failed.")
                print(f"({worker_index}) Still waiting for results from worker {other_worker_index} ({delay_so_far} sec.)!")
                time.sleep(delay_per_try)
                delay_so_far += delay_per_try

    # Update success probs with image-free QA results
    all_success_probs = [
        [
            # Use code in VQAOutputs class to calculate Yes/No probabilities from logits
            round(float(VQAOutputs(
                task_name=MistakeDetectionTasks(args.task),
                example_id="",
                procedure_id=-1,
                frame="",
                prompt="",
                expected_answer=None,
                response_token_ids=response_token_ids,
                logits=logits[question_idx][example_idx],
                question="",
            ).answer_probs[VQAResponse.Yes]), 6) for question_idx in range(n_questions_per_example)
        ] for example_idx in range(len(all_example_ids))
    ]

    # Collect key information from results rollouts and final success probabilities
    all_results_dicts = {}
    all_probs = []
    for questions, frames, candidate_questions, candidate_questions_scores, candidate_questions_sources, scores, answers, answer_probs, success_probs, example_id, procedure, label \
        in tqdm(zip(all_questions,
                    all_frames,
                    all_candidate_questions,
                    all_candidate_questions_scores,
                    all_candidate_questions_sources,
                    all_scores,
                    all_answers,
                    all_answer_probs,
                    all_success_probs,
                    all_example_ids,
                    all_procedures,
                    all_labels), desc="compiling results"):
        
        final_success_prob = None
        for success_prob_idx, success_prob in enumerate(success_probs):
            # Early stopping mechanism: 
            # if success score doesn't change enough over 3 turns, stop incorporating questions
            # (we still run inference across all questions for efficiency and simplicity, but later can make a proper demo script)
            final_success_prob = success_prob
            if success_prob_idx >= 2 and success_prob_idx < len(success_probs) - 1:
                if np.abs(success_probs[success_prob_idx-1] - success_probs[success_prob_idx-2]) < args.early_stop_delta and np.abs(success_probs[success_prob_idx] - success_probs[success_prob_idx-1]) < args.early_stop_delta:
                    break
        all_probs.append(round(final_success_prob, 6))   

        results_dict = {
            "procedure": procedure,
            "mistake": True if label is not None else False,
            "mistake_type": label,
            "questions": questions,
            "frame_paths": frames,
            "answers": [a.value for a in answers],
            "answer_probs": answer_probs,
            "scores": scores,
            "success_probs": success_probs,
            "final_turn": success_prob_idx,
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

    # Calculate coherence metrics of final rollouts
    all_chosen_questions = [question for results_dict in all_results_dicts.values() for question in results_dict['questions'][:results_dict['final_turn'] + 1]]
    all_previous_questions = [results_dict['questions'][:question_idx] for results_dict in all_results_dicts.values() for question_idx in range(results_dict['final_turn'] + 1)]

    all_predicted_answers = [VQAResponse(answer) for results_dict in all_results_dicts.values() for answer in results_dict['answers'][:results_dict['final_turn'] + 1]]
    all_previous_answers = [[VQAResponse(a) for a in results_dict['answers'][:question_idx]] for results_dict in all_results_dicts.values() for question_idx in range(results_dict['final_turn'] + 1)]

    all_coherence_metrics = question_coherence_metrics_nli(nli_tokenizer,
                                            nli_model,
                                            tokenizer,
                                            lm,                                         
                                            [procedure for results_dict, procedure in zip(all_results_dicts.values(), all_procedures) for _ in range(results_dict['final_turn'] + 1)],
                                            all_chosen_questions,
                                            answers=all_predicted_answers,
                                            previous_questions=all_previous_questions,
                                            previous_answers=all_previous_answers,
                                            rephrase_batch_size=args.generation_batch_size)

    parallel_idx = 0
    coherence_metrics_by_example = defaultdict(list)
    coherence_metrics_by_turn = defaultdict(list)
    coherence_metric_names = ['relevance', 'informativeness', 'relevance_marginal', 'informativeness_marginal', 'informativeness_marginal_x_relevance']
    for results_dict in all_results_dicts.values():
        for k in coherence_metric_names:
            if k in all_coherence_metrics:
                this_metrics = []
                for question_idx in range(results_dict['final_turn'] + 1):
                    this_metrics.append(round(float(all_coherence_metrics[k][parallel_idx]), 6))
                coherence_metrics_by_example[k + "_by_example"].append(round(float(np.mean(this_metrics)), 6))
                coherence_metrics_by_turn[k + "_by_turn"].append(this_metrics)
        parallel_idx += 1
    
    # Calculate accuracy metrics
    best_metrics = None
    best_threshold = None
    accuracy_metrics_by_threshold = {}
    coherence_metrics_by_threshold = {}
    all_labels_binary = [True if l is not None else False for l in all_labels]
    for threshold in MISTAKE_DETECTION_THRESHOLDS:
        preds = [1.0 - p >= threshold for p in all_probs] # Have to do 1.0 - probability since we got "success" probability from VLM
        assert len(preds) == len(all_probs) == len(all_labels), "Expected same number of preds, probs, and labels."
        this_metrics = mistake_detection_metrics(all_labels_binary, preds)
        accuracy_metrics_by_threshold[threshold] = this_metrics

        # Calculate consistency and verifiability for this example, which are conditional on correctness
        verifiability = np.mean([coherence_metrics_by_example['informativeness_marginal_x_relevance_by_example'][i] if preds[i] == all_labels_binary[i] else 0.0 for i in range(len(preds))])
        consistency = np.mean([coherence_metrics_by_example['relevance_by_example'][i] if preds[i] == all_labels_binary[i] else 0.0 for i in range(len(preds))])
        coherence_metrics_by_threshold[threshold] = {"verifiability": verifiability, "consistency": consistency,}

        if best_metrics is None or (this_metrics['false_positive_rate'] + this_metrics['false_negative_rate']) < (best_metrics['false_positive_rate'] + best_metrics['false_negative_rate']):
            best_metrics = this_metrics
            best_threshold = threshold

    accuracy_metrics_by_threshold['best_metrics'] = best_metrics
    accuracy_metrics_by_threshold['best_threshold'] = best_threshold

    # Save accuracy and coherence metrics
    json.dump(accuracy_metrics_by_threshold, 
            open(os.path.join(this_results_dir, f"metrics_accuracy_{args.eval_partition}.json"), "w"),
            indent=4)
    
    coherence_metrics = {
        k: round(float(np.mean(coherence_metrics_by_example[k + "_by_example"])), 6) for k in coherence_metric_names if k + "_by_example" in coherence_metrics_by_example
    } | {
        "metrics_by_threshold": coherence_metrics_by_threshold,
        "metrics_by_example": coherence_metrics_by_example,
        "metrics_by_turn": coherence_metrics_by_turn,
    }
    json.dump(coherence_metrics, 
            open(os.path.join(this_results_dir, f"metrics_coherence_{args.eval_partition}.json"), "w"),
            indent=4)

    # Generate DET curves for accuracy
    generate_det_curve(accuracy_metrics_by_threshold, os.path.join(this_results_dir, f"det_accuracy_{args.eval_partition}.pdf"))

    # Generate curves for all metrics by threshold
    generate_tiered_metric_curves(MISTAKE_DETECTION_THRESHOLDS, 
                                  [accuracy_metrics_by_threshold[t]['accuracy'] for t in MISTAKE_DETECTION_THRESHOLDS],
                                  [coherence_metrics_by_threshold[t]['consistency'] for t in MISTAKE_DETECTION_THRESHOLDS], 
                                  [coherence_metrics_by_threshold[t]['verifiability'] for t in MISTAKE_DETECTION_THRESHOLDS],
                                  [os.path.join(this_results_dir, f"graph_tiered_metrics_{args.eval_partition}.pdf")])
    
    print(f"({worker_index}) Done!")