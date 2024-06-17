# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import json
import os
from pprint import pprint

from travel.model.mistake_detection import generate_det_curves

# Configure arguments here
results_fnames = [
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results/vqa_mistake_detection/captaincook4d/SuccessVQA_captaincook4d_llava-1.5-7b-hf_20240516113912/metrics_heuristic_val.json",
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results/vqa_mistake_detection/captaincook4d/VQG2VQA_captaincook4d_llava-1.5-7b-hf_20240617181122/metrics_nli_val.json",
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results/vqa_mistake_detection/captaincook4d/VQG2VQA_captaincook4d_llava-1.5-7b-hf_20240617181622_spatial/metrics_nli_val.json",
]
results_names = [
    "SuccessVQA",
    "VQG2VQA + NLI",
    "VQG2VQA + NLI + Spatial",
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