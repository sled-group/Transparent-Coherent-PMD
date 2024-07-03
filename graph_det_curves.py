# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import json
import os
from pprint import pprint

from travel.model.mistake_detection import generate_det_curves

# Configure results to graph here

# Graph 1: 3 main approaches
# results_fnames = [
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_20240701113527/metrics_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240701115231/metrics_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial1.0_20240701115730/metrics_heuristic_val.json",
# ]
# results_names = [
#     "SuccessVQA",
#     "VQG2VQA",
#     "VQG2VQA + Spatial",
# ]
# output_fname_prefix = "det_comparison_val"

# Graph 2: spatial filter parts breakdown (adding TOC to all 3 results, removing rephrase from spatial filter)
# results_fnames = [
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_20240701113527/metrics_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_target_object_counter1.0_20240702182300/metrics_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240701115231/metrics_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_target_object_counter1.0_20240702181455/metrics_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial1.0_20240701115730/metrics_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase1.0_20240701130520/metrics_heuristic_val.json",
# ]
# results_names = [
#     "SuccessVQA",
#     "SuccessVQA + TOC",
#     "VQG2VQA",
#     "VQG2VQA + TOC",
#     "VQG2VQA + Spatial",
#     "VQG2VQA + Spatial (no rephrase)",
# ]
# output_fname_prefix = "det_spatial_breakdown_val"

# Graph 3: Introducing NLI
# results_fnames = [
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_20240701113527/metrics_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240701115231/metrics_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240701115231/metrics_nli_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial1.0_20240701115730/metrics_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial1.0_20240701115730/metrics_nli_val.json",
# ]
# results_names = [
#     "SuccessVQA",
#     "VQG2VQA",
#     "VQG2VQA + NLI",
#     "VQG2VQA + Spatial",
#     "VQG2VQA + Spatial + NLI",
# ]
# output_fname_prefix = "det_nli_val"

results_fnames = [
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_target_object_counter1.0_20240702181455/metrics_heuristic_val.json",
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase0.25_20240702214841/metrics_heuristic_val.json",
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase0.5_20240701161456/metrics_heuristic_val.json",
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase0.75_20240702204247/metrics_heuristic_val.json",
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase1.0_20240701130520/metrics_heuristic_val.json",
]
results_names = [
    "VQG2VQA + TOC",
    "VQG2VQA + Spatial (0.25)",
    "VQG2VQA + Spatial (0.5)",
    "VQG2VQA + Spatial (0.75)",
    "VQG2VQA + Spatial (1.0)",
]
output_fname_prefix = "det_spatial_intensity_val"


# TODO: Graph 4: Spatial Filter Intensity


metrics = [{float(k): v for k, v in json.load(open(fname, "r")).items() if k not in ["best_metrics", "best_threshold"]} for fname in results_fnames]
pprint(metrics)
output_fname = f"{output_fname_prefix}_{'_'.join(results_names).replace(' ', '-')}.pdf"
save_paths = [os.path.join("/".join(fname.split("/")[:-1]), output_fname) for fname in results_fnames]

generate_det_curves(metrics,
                    results_names,
                    save_paths)
print("Graph generated!")