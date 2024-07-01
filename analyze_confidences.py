# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Configure arguments here
results_fnames = [
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results/vqa_mistake_detection/ego4d/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_20240627120709/preds_heuristic_val.json",
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240627122511/preds_heuristic_val.json",
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial1.0_20240627161437/preds_heuristic_val.json",
]
results_names = [
    "SuccessVQA",
    "VQG2VQA",
    "VQG2VQA + Spatial",
]
output_fname_prefix = "confidence_comparison_val"

confidences_correct = [[] for _ in results_fnames]
confidences_incorrect = [[] for _ in results_fnames]
for i, result_fname in enumerate(results_fnames):
    result_preds = json.load(open(result_fname, "r"))
    for pred in result_preds.values():
        mistake = pred["example"]["mistake"]
        
        mistake_probs = np.array(pred['mistake_detection']["0.0"]["mistake_probs"]).flatten()
        mistake_prob = np.max(mistake_probs) if len(mistake_probs) > 0 else 1.0
        if mistake:
            confidences_correct[i].append(mistake_prob)
        else:
            confidences_incorrect[i].append(mistake_prob)
        
x = np.arange(len(results_names))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
for i in range(len(results_fnames)):
    rects1 = ax.bar(
        x[i] - width/2, 
        np.mean(confidences_correct[i]), 
        width, 
        yerr=np.std(confidences_correct[i]), 
        label='Mistake Examples' if i == 0 else "", 
        color='green', 
        capsize=5
    )
    rects2 = ax.bar(
        x[i] + width/2, 
        np.mean(confidences_incorrect[i]), 
        width, 
        yerr=np.std(confidences_incorrect[i]), 
        label='Success Examples' if i == 0 else "", 
        color='red', 
        capsize=5
    )

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Result')
ax.set_ylabel('Confidence')
ax.set_xticks(x)
ax.set_xticklabels(results_names)
ax.legend()

fig.tight_layout()

# fig, ax = plt.subplots()
# for i in range(len(results_fnames)):
#     y_correct = confidences_correct[i]
#     y_incorrect = confidences_incorrect[i]
#     x_correct = np.full(len(y_correct), x[i] - 0.1)  # slight offset for visual clarity
#     x_incorrect = np.full(len(y_incorrect), x[i] + 0.1)  # slight offset for visual clarity
#     ax.scatter(x_correct, y_correct, color='green', label='Mistake Examples' if i == 0 else "")
#     ax.scatter(x_incorrect, y_incorrect, color='red', label='Success Examples' if i == 0 else "")

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Result')
# ax.set_ylabel('Confidence')
# ax.set_xticks(x)
# ax.set_xticklabels(results_names)
# ax.legend()

# fig.tight_layout()

output_fname = f"{output_fname_prefix}_{'_'.join(results_names).replace(' ', '-')}.pdf"
save_paths = [os.path.join("/".join(fname.split("/")[:-1]), output_fname) for fname in results_fnames]
for path in save_paths:
    fig.savefig(path)

print("Graph generated!")
