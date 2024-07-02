# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import json
import os
from pprint import pprint

from travel.model.mistake_detection import generate_det_curves

# Configure arguments here
results_fnames = [
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_20240701113527/metrics_heuristic_val.json",
    # "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240701115231/metrics_nli_val.json",
    # "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial1.0_20240701115730/metrics_nli_val.json",
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240701115231/metrics_heuristic_val.json",
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial1.0_20240701115730/metrics_heuristic_val.json",
    # "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase0.75_20240630222902/metrics_heuristic_val.json"
]
# results_fnames = [
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results/vqa_mistake_detection/captaincook4d/SuccessVQA_captaincook4d_llava-1.5-7b-hf_20240621093854/metrics_heuristic_val.json",
#     # "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results/vqa_mistake_detection/captaincook4d/VQG2VQA_captaincook4d_llava-1.5-7b-hf_20240620101539/metrics_nli_val.json",
#     # "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results/vqa_mistake_detection/captaincook4d/VQG2VQA_captaincook4d_llava-1.5-7b-hf_spatial1.0_20240620154228/metrics_nli_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results/vqa_mistake_detection/captaincook4d/VQG2VQA_captaincook4d_llava-1.5-7b-hf_20240621093406/metrics_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results/vqa_mistake_detection/captaincook4d/VQG2VQA_captaincook4d_llava-1.5-7b-hf_spatial1.0_20240621093359/metrics_heuristic_val.json",
# ]
results_names = [
    "SuccessVQA",
    # "VQG2VQA + NLI",
    # "VQG2VQA + NLI + Spatial",
    "VQG2VQA",
    "VQG2VQA + Spatial",
    # "VQG2VQA + Spatial (0.75 strength, no rephrase)"
]
output_fname_prefix = "det_comparison_val"

metrics = [{float(k): v for k, v in json.load(open(fname, "r")).items() if k not in ["best_metrics", "best_threshold"]} for fname in results_fnames]
pprint(metrics)
output_fname = f"{output_fname_prefix}_{'_'.join(results_names).replace(' ', '-')}.pdf"
save_paths = [os.path.join("/".join(fname.split("/")[:-1]), output_fname) for fname in results_fnames]

generate_det_curves(metrics,
                    results_names,
                    save_paths)
print("Graph generated!")