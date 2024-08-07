from travel import init_travel
init_travel()

import argparse
from collections import defaultdict
import json
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer

from travel.constants import RESULTS_DIR
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.data.vqa import VQAResponse, get_vqa_response_token_ids
from travel.model.metrics import mistake_detection_metrics, question_coherence_metrics, generate_det_curve
from travel.model.mistake_detection import MISTAKE_DETECTION_THRESHOLDS
from travel.model.nli import NLI_MODEL_PATH, NLI_BATCH_SIZE

parser = argparse.ArgumentParser()
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=["llava-hf/llava-1.5-7b-hf"], help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--task", type=str, default="ego4d_single", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--eval_partition", type=str, default="val", choices=["val", "test"])
parser.add_argument("--early_stop_delta", type=int, default=0.1, help="If success probability changes less than this over 3 turns, stop generating questions.")
parser.add_argument("--generation_batch_size", type=int, default=60, help="Batch size for question generation with LM.")
parser.add_argument("--nli_batch_size", type=int, default=NLI_BATCH_SIZE, help="Batch size for scoring candidate questions with NLI model.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
args = parser.parse_args()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load VLM
vlm = AutoModelForVision2Seq.from_pretrained(args.vlm_name, quantization_config=bnb_config)   
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

# Set up results directory
vlm_name = args.vlm_name.split('/')[-1]
task_name = args.task
if args.debug:
    task_name += f"_debug{args.debug_n_examples}"
results_dir = os.path.join(RESULTS_DIR, "vqa_mistake_detection", task_name, vlm_name)

all_questions = []
all_frames = []
all_candidate_questions = []
all_candidate_questions_scores = []
all_scores = []
all_answers = []
all_answer_probs = []
all_success_probs = []

all_example_ids = []
all_procedures = []
all_labels = []

for this_results_dir in tqdm(os.listdir(results_dir)):
    if not os.path.exists(os.path.join(results_dir, this_results_dir, f"metrics_accuracy_{args.eval_partition}.json")):
        # Only rerun if we've evaluated before
        print(f"Could not find a metrics file {os.path.join(results_dir, this_results_dir, f'metrics_accuracy_{args.eval_partition}.json')}.")
        continue

    for fname in os.listdir(os.path.join(results_dir, this_results_dir)):
        if fname.endswith(".pkl"):
            is_complete, \
            _, \
            other_questions, \
            other_frames, \
            other_candidate_questions, \
            other_candidate_questions_scores, \
            other_scores, \
            other_answers, \
            other_answer_probs, \
            other_success_probs, \
            other_example_ids, \
            other_procedures, \
            other_labels = pickle.load(open(os.path.join(results_dir, this_results_dir, fname), "rb"))
            if is_complete:
                # Add other process results to our results
                all_questions += other_questions
                all_frames += other_frames
                all_candidate_questions += other_candidate_questions
                all_candidate_questions_scores += other_candidate_questions_scores
                all_scores += other_scores
                all_answers += other_answers
                all_answer_probs += other_answer_probs
                all_success_probs += other_success_probs
                all_example_ids += other_example_ids
                all_procedures += other_procedures
                all_labels += other_labels

    # Verify we got correct number of outputs
    all_results = [
        all_questions, 
        all_frames,
        all_candidate_questions, 
        all_candidate_questions_scores, 
        all_scores, 
        all_answers, 
        all_answer_probs, 
        all_success_probs,
        all_example_ids,
        all_procedures,
        all_labels,
    ]
    assert all(len(l) == len(all_results[0]) for l in all_results), f"Expected to get same number of all outputs! ({', '.join([str(len(l)) for l in all_results])})"

    # Collect key information from results rollouts and final success probabilities
    all_results_dicts = {}
    all_probs = []
    for questions, frames, candidate_questions, candidate_questions_scores, scores, answers, answer_probs, success_probs, example_id, procedure, label \
        in tqdm(zip(all_questions,
                    all_frames,
                    all_candidate_questions,
                    all_candidate_questions_scores,
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
                # TODO: this doesn't seem to be working as expected
                if np.abs(success_probs[success_prob_idx-1] - success_probs[success_prob_idx-2]) < args.early_stop_delta and np.abs(success_probs[success_prob_idx] - success_probs[success_prob_idx-1]) < args.early_stop_delta:
                    break
        all_probs.append(round(final_success_prob, 6))   

        results_dict = {
            "procedure": procedure,
            "mistake": label,
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
        }
        all_results_dicts[example_id] = results_dict

    json.dump(all_results_dicts, 
            open(os.path.join(results_dir, this_results_dir, f"outputs_{args.eval_partition}.json"), "w"),
            indent=4)

    metrics = {}

    # Calculate accuracy metrics
    best_metrics = None
    best_threshold = None
    accuracy_metrics = {}
    for threshold in MISTAKE_DETECTION_THRESHOLDS:
        preds = [1.0 - p >= threshold for p in all_probs]
        assert len(preds) == len(all_probs) == len(all_labels), "Expected same number of preds, probs, and labels."
        this_metrics = mistake_detection_metrics(all_labels, preds)
        accuracy_metrics[threshold] = this_metrics

        if best_metrics is None or (this_metrics['false_positive_rate'] + this_metrics['false_negative_rate']) < (best_metrics['false_positive_rate'] + best_metrics['false_negative_rate']):
            best_metrics = this_metrics
            best_threshold = threshold

    accuracy_metrics['best_metrics'] = best_metrics
    accuracy_metrics['best_threshold'] = best_threshold

    json.dump(accuracy_metrics, 
            open(os.path.join(results_dir, this_results_dir, f"metrics_accuracy_{args.eval_partition}.json"), "w"),
            indent=4)

    # Generate DET curve
    generate_det_curve(accuracy_metrics, os.path.join(results_dir, this_results_dir, f"det_accuracy_{args.eval_partition}.pdf"))

    # Calculate coherence metrics of final rollouts
    all_chosen_questions = [question for results_dict in all_results_dicts.values() for question in results_dict['questions'][:results_dict['final_turn'] + 1]]
    all_previous_questions = [results_dict['questions'][:question_idx] for results_dict in all_results_dicts.values() for question_idx in range(results_dict['final_turn'] + 1)]

    all_predicted_answers = [VQAResponse(answer) for results_dict in all_results_dicts.values() for answer in results_dict['answers'][:results_dict['final_turn'] + 1]]
    all_previous_answers = [[VQAResponse(a) for a in results_dict['answers'][:question_idx]] for results_dict in all_results_dicts.values() for question_idx in range(results_dict['final_turn'] + 1)]

    all_metrics = question_coherence_metrics(nli_tokenizer,
                                            nli_model,
                                            tokenizer,
                                            lm,                                         
                                            [procedure for results_dict, procedure in zip(all_results_dicts.values(), all_procedures) for _ in range(results_dict['final_turn'] + 1)],
                                            all_chosen_questions,
                                            answers=all_predicted_answers, # TODO: this returns an informativeness metric which says how much different success probability is for a yes or no answer - should we just look at probability of entailment if it's a success example, and contradiction if it's a mistake?
                                            previous_questions=all_previous_questions,
                                            previous_answers=all_previous_answers,
                                            rephrase_batch_size=args.generation_batch_size)

    parallel_idx = 0
    coherence_metrics_by_example = defaultdict(list)
    coherence_metric_names = ['relevance', 'informativeness', 'relevance_marginal', 'informativeness_marginal']
    for results_dict in all_results_dicts.values():
        for k in coherence_metric_names:
            if k in all_metrics:
                this_metrics = []
                for question_idx in range(results_dict['final_turn'] + 1):
                    this_metrics.append(all_metrics[k][parallel_idx])
                coherence_metrics_by_example[k].append(round(float(np.mean(this_metrics)), 6))
        parallel_idx += 1

    coherence_metrics = {
        k: round(float(np.mean(coherence_metrics_by_example[k])), 6) for k in coherence_metric_names if k in coherence_metrics_by_example
    } | {
        "metrics_by_example": coherence_metrics_by_example,
    }
    json.dump(coherence_metrics, 
            open(os.path.join(results_dir, this_results_dir, f"metrics_coherence_{args.eval_partition}.json"), "w"),
            indent=4)
