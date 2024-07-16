# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import datetime
import json
import os
from pprint import pprint

from travel.constants import RESULTS_DIR
from travel.model.mistake_detection import generate_det_curves, generate_roc_curves

# Configure results to graph here
TASK = "ego4d"
timestamp = datetime.datetime.now()
run_folder_name = f"classification_curves_{timestamp.strftime('%Y%m%d%H%M%S')}"
graph_dir = os.path.join(RESULTS_DIR, "analysis", TASK, run_folder_name)
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)

for results_fnames, results_names, output_fname_prefix in [
    (
        # Graph 1: main approaches to compare
        [
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_20240701113527/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240701115231/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase1.0_20240701130520/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase1.0_20240701130520/metrics_nli_val.json",
        ],
        [
            "SuccessVQA",
            "VQG2VQA",
            "VQG2VQA + Spatial",
            "VQG2VQA + Spatial + NLI"
        ],
        "main_val"
    ),
    (
        # Graph 2: spatial filter parts breakdown (adding TOC to all 3 results, removing rephrase from spatial filter)
        [
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_20240701113527/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_target_object_counter1.0_20240702182300/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240701115231/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_target_object_counter1.0_20240702181455/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase1.0_20240701130520/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial1.0_20240701115730/metrics_heuristic_val.json",
        ],
        [
            "SuccessVQA",
            "SuccessVQA + TOC",
            "VQG2VQA",
            "VQG2VQA + TOC",
            "VQG2VQA + TOC + Spatial",
            "VQG2VQA + TOC + Spatial + Rephrase",
        ],
        "spatial_breakdown_val"
    ),
    (
        # Graph 3: Introducing NLI
        [
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_20240701113527/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240701115231/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240701115231/metrics_nli_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase1.0_20240701130520/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase1.0_20240701130520/metrics_nli_val.json",
        ],
        [
            "SuccessVQA",
            "VQG2VQA",
            "VQG2VQA + NLI",
            "VQG2VQA + Spatial",
            "VQG2VQA + Spatial + NLI",
        ],
        "nli_val"
    ),
    (
        # Graph 4: Spatial Filter Darkness Intensity
        [
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_target_object_counter1.0_20240702181455/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase0.25_20240702214841/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase0.5_20240701161456/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase0.75_20240702204247/metrics_heuristic_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase1.0_20240701130520/metrics_heuristic_val.json",
        ],
        [
            "VQG2VQA + TOC",
            "VQG2VQA + Spatial (0.25)",
            "VQG2VQA + Spatial (0.5)",
            "VQG2VQA + Spatial (0.75)",
            "VQG2VQA + Spatial (1.0)",
        ],
        "spatial_intensity_val"
    ),
    (
        # Graph 5: Spatial Filter Blur Intensity
        [
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_blur5.0_20240709190600/metrics_nli_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_blur15.0_20240709212259/metrics_nli_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_blur25.0_20240709201418/metrics_nli_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_blur35.0_20240709223525/metrics_nli_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_blur45.0_20240709234703/metrics_nli_val.json"
        ],
        [
            "VQG2VQA + Spatial (Blur; k5) + NLI",
            "VQG2VQA + Spatial (Blur; k15) + NLI",
            "VQG2VQA + Spatial (Blur; k25) + NLI",
            "VQG2VQA + Spatial (Blur; k35) + NLI",
            "VQG2VQA + Spatial (Blur; k45) + NLI",
        ],
        "spatial_intensity_blur_val"
    ),
]:

    metrics = [{float(k): v for k, v in json.load(open(fname, "r")).items() if k not in ["best_metrics", "best_threshold"]} for fname in results_fnames]
    pprint(metrics)

    # Graph DET and ROC curves
    output_fname = f"det_comparison_{output_fname_prefix}_{'_'.join(results_names).replace(' ', '-')}.pdf"
    save_paths = [os.path.join("/".join(fname.split("/")[:-1]), run_folder_name, output_fname) for fname in results_fnames] + [os.path.join(graph_dir, output_fname)]
    generate_det_curves(metrics,
                        results_names,
                        save_paths)

    output_fname = f"roc_comparison_{output_fname_prefix}_{'_'.join(results_names).replace(' ', '-')}.pdf"
    save_paths = [os.path.join("/".join(fname.split("/")[:-1]), run_folder_name, output_fname) for fname in results_fnames] + [os.path.join(graph_dir, output_fname)]
    generate_roc_curves(metrics,
                        results_names,
                        save_paths)

    print(f"Graphs for {output_fname_prefix} generated!")