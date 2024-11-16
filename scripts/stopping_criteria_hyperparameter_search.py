from travel import init_travel
init_travel()

import argparse
from copy import deepcopy
from itertools import product
import json
import numpy as np
import os
from pprint import pprint
import torch
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer, PhrasalConstraint           

from travel.constants import HF_TOKEN
from travel.data.vqa import get_vqa_response_token_ids
from travel.model.metrics import question_coherence_metrics_nli, generate_det_curve, generate_tiered_metric_curves, compile_accuracy_and_coherence_metrics
from travel.model.mistake_detection import MISTAKE_DETECTION_THRESHOLDS
from travel.model.nli import NLI_MODEL_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--this_results_dir", type=str, help="Path to results directory for approach to re-tune.")
args = parser.parse_args()

VLM_NAME = args.vlm_name
this_results_dir = args.this_results_dir

with open(os.path.join(this_results_dir, "outputs_val.json"), "r") as f:
    all_results_dicts = json.load(f)

this_results_dir = os.path.join(this_results_dir, "stopping_criteria_tuning")
if not os.path.exists(this_results_dir):
    os.makedirs(this_results_dir)

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
    vlm = AutoModelForVision2Seq.from_pretrained(VLM_NAME, quantization_config=bnb_config, trust_remote_code=True, token=HF_TOKEN)   
except Exception as e:
    print("Encountered exception when trying to load model with AutoModelForVision2Seq:")
    pprint(e)
    
    vlm = AutoModelForCausalLM.from_pretrained(VLM_NAME, quantization_config=bnb_config, trust_remote_code=True, token=HF_TOKEN)
vlm_processor = AutoProcessor.from_pretrained(VLM_NAME, trust_remote_code=True, token=HF_TOKEN)
vlm_processor.tokenizer.padding_side = "left"
response_token_ids = get_vqa_response_token_ids(vlm_processor.tokenizer)

# We'll use VLM's LM directly to generate questions
if getattr(vlm, "language_model", None):
    lm = vlm.language_model
else:
    lm = vlm
tokenizer = vlm_processor.tokenizer
tokenizer.pad_token_id = tokenizer.eos_token_id

# NLI model to score consistency and verifiability
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)

cand_max_iterations = [2, 4, 6, 8, 10]
cand_early_stop_delta = [0.05, 0.1, 0.2, 0.4]
cand_confident_range = [0.025, 0.05, 0.1, 0.2]
cand_criteria = product(cand_max_iterations, cand_early_stop_delta, cand_confident_range)

best_performance = None

all_coherence_metrics = None

performance_by_criteria = {}

for mi, esd, cd in tqdm(cand_criteria, desc="candidate criteria"):
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

    # Get all coherence metrics for all turns (original run doesn't have this)
    if all_coherence_metrics is None:
        all_coherence_metrics = question_coherence_metrics_nli(nli_tokenizer,
                                                               nli_model,
                                                               tokenizer,
                                                               lm,                                         
                                                               [procedure for results_dict, procedure in zip(all_results_dicts.values(), all_procedures) for _ in range(10)],
                                                               all_chosen_questions,
                                                               answers=all_predicted_answers,
                                                               previous_questions=all_previous_questions,
                                                               previous_answers=all_previous_answers,
                                                               mistake_labels=[results_dict['mistake'] for results_dict in all_results_dicts.values() for _ in range(10)],
                                                               rephrase_batch_size=20)
        
    # Adjust all_coherence_metrics for the specific final turns we chose here
    readjusted_all_coherence_metrics = {}
    for k in all_coherence_metrics:
        parallel_idx = 0
        this_metrics = []
        for results_dict in all_results_dicts.values():
            for question_idx in range(10):

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
    accuracy_metrics_by_threshold, coherence_metrics = compile_accuracy_and_coherence_metrics(all_labels, all_probs, readjusted_all_coherence_metrics, all_results_dicts, MISTAKE_DETECTION_THRESHOLDS, 0.1)
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
        best_metrics = (accuracy_metrics_by_threshold, readjusted_all_coherence_metrics, coherence_metrics, coherence_metrics_by_threshold, deepcopy(all_results_dicts))
        best_criteria = (mi, esd, cd)

    # Save info for this combo
    subdir_path = os.path.join(this_results_dir, f"{mi}_{esd}_{cd}")
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)
    json.dump(all_results_dicts, 
            open(os.path.join(subdir_path, "outputs_val.json"), "w"),
            indent=4)    
    
    json.dump(accuracy_metrics_by_threshold, 
            open(os.path.join(subdir_path, "metrics_accuracy_val.json"), "w"),
            indent=4)

    json.dump(coherence_metrics, 
            open(os.path.join(subdir_path, "metrics_coherence_nli_val.json"), "w"),
            indent=4)

    json.dump(readjusted_all_coherence_metrics, 
            open(os.path.join(subdir_path, "metrics_coherence_raw_nli_val.json"), "w"),
            indent=4)            

accuracy_metrics_by_threshold, readjusted_all_coherence_metrics, coherence_metrics, coherence_metrics_by_theshold, all_results_dicts = best_metrics

# Save accuracy and coherence metrics and other outputs for best combo
json.dump(all_results_dicts, 
        open(os.path.join(this_results_dir, f"outputs_val.json"), "w"),
        indent=4)

json.dump(accuracy_metrics_by_threshold, 
        open(os.path.join(this_results_dir, f"metrics_accuracy_val.json"), "w"),
        indent=4)

json.dump(coherence_metrics, 
        open(os.path.join(this_results_dir, f"metrics_coherence_nli_val.json"), "w"),
        indent=4)

json.dump(readjusted_all_coherence_metrics, 
        open(os.path.join(this_results_dir, f"metrics_coherence_raw_nli_val.json"), "w"),
        indent=4)            

mi, esd, cd = best_criteria
json.dump({"max_iterations": mi, "early_stop_delta": esd, "confident_range": cd},
          open(os.path.join(this_results_dir, "tuned_stopping_criteria.json"), "w"),
          indent=4,
)

json.dump(performance_by_criteria, open(os.path.join(this_results_dir, "performance_by_criteria.json"), "w"), indent=4)

# Generate DET curves for accuracy
generate_det_curve(accuracy_metrics_by_threshold, os.path.join(this_results_dir, f"det_accuracy_val.pdf"))

# Generate curves for all metrics by threshold
generate_tiered_metric_curves(MISTAKE_DETECTION_THRESHOLDS, 
                              [accuracy_metrics_by_threshold[t]['accuracy'] for t in MISTAKE_DETECTION_THRESHOLDS],
                              [coherence_metrics_by_threshold[t]['consistency'] for t in MISTAKE_DETECTION_THRESHOLDS], 
                              [coherence_metrics_by_threshold[t]['verifiability'] for t in MISTAKE_DETECTION_THRESHOLDS],
                              [os.path.join(this_results_dir, f"graph_tiered_metrics_nli_val.pdf")])
