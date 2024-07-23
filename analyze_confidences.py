# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pprint import pprint
import random
from scipy.stats import pointbiserialr
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, auc
from tqdm import tqdm

from travel.constants import RESULTS_DIR
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.mistake_detection import MistakeDetectionExample
from travel.data.vqa import VQAOutputs, VQAResponse
from travel.model.mistake_detection import aggregate_mistake_probs_over_frames, DETECTION_FRAMES_PROPORTION, MISTAKE_DETECTION_THRESHOLDS, mistake_detection_metrics, generate_det_curve, HeuristicMistakeDetectionEvaluator, compile_mistake_detection_preds, generate_risk_coverage_plot, calculate_abstention_metrics, plot_abstention_metrics
from travel.model.utils import expected_calibration_error

# Configure results to graph here
TASK = "ego4d"
timestamp = datetime.datetime.now()
run_folder_name = f"confidence_analysis_{timestamp.strftime('%Y%m%d%H%M%S')}"
output_dir = os.path.join(RESULTS_DIR, f"analysis", TASK, run_folder_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configure arguments here
# results_fnames = [
#     "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_20240701113527/preds_heuristic_val.json",
#     "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_target_object_counter1.0_20240702182300/preds_heuristic_val.json",
#     "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240701115231/preds_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240701115231/preds_nli_val.json",
#     "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial1.0_20240701115730/preds_heuristic_val.json",
#     "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial1.0_20240701115730/preds_nli_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_blur45.0_20240709234703/preds_nli_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase1.0_20240701130520/preds_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase1.0_20240701130520/preds_nli_val.json",
# ]
# results_fnames = [
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/SuccessVQA_ego4d_debug500_llava-1.5-7b-hf_20240712182848/preds_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/VQG2VQA_ego4d_debug500_llava-1.5-7b-hf_20240713091808/preds_nli_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/VQG2VQA_ego4d_debug500_llava-1.5-7b-hf_20240712233421/preds_nli_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/VQG2VQA_ego4d_debug500_llava-1.5-7b-hf_spatial_norephrase1.0_20240713091801/preds_nli_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/VQG2VQA_ego4d_debug500_llava-1.5-7b-hf_spatial_norephrase1.0_20240712233432/preds_nli_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/VQG2VQA_ego4d_debug500_llava-1.5-7b-hf_20240713191222/preds_heuristic_val.json",
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/VQG2VQA_ego4d_debug500_llava-1.5-7b-hf_20240713191222/preds_nli_val.json",
#     # "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/VQG2VQA_ego4d_debug500_llava-1.5-7b-hf_spatial_norephrase1.0_20240713191244/preds_heuristic_val.json"
#     "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/VQG2VQA_ego4d_debug500_llava-1.5-7b-hf_spatial_norephrase1.0_20240713191244/preds_nli_val.json",
    
# ]
# results_fnames = [
#     "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/SuccessVQA_ego4d_debug500_llava-1.5-7b-hf_20240712182848/preds_heuristic_val.json",
#     "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/SuccessVQA_ego4d_debug500_llava-1.5-7b-hf_caption_20240717104130/preds_heuristic_val.json",
#     "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/VQG2VQA_ego4d_debug500_llava-1.5-7b-hf_20240713191222/preds_nli_val.json",
#     "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/VQG2VQA_ego4d_debug500_llava-1.5-7b-hf_caption_20240717103812/preds_nli_val.json",
#     "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/VQG2VQA_ego4d_debug500_llava-1.5-7b-hf_spatial_norephrase1.0_20240713191244/preds_nli_val.json",
#     "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug500/llava-1.5-7b-hf/VQG2VQA_ego4d_debug500_llava-1.5-7b-hf_spatial_norephrase1.0_caption_20240717103844/preds_nli_val.json"
# ]
results_fnames = [
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_20240701113527/preds_heuristic_val.json",
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_caption_20240722155838/preds_heuristic_val.json",
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA2SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_20240722153521/preds_heuristic_val.json",
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_debug250/llava-1.5-7b-hf/VQG2VQA2SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_spatial_norephrase1.0_20240722155446/preds_heuristic_val.json"
]
for results_fname in results_fnames:
    if not os.path.exists(os.path.join(os.path.join("/".join(results_fname.split("/")[:-1]), run_folder_name))):
        os.makedirs(os.path.join(os.path.join("/".join(results_fname.split("/")[:-1]), run_folder_name)))
                 
metrics = [
    json.load(open(fname.replace("preds_", "metrics_"), "r")) for fname in results_fnames
]
# results_names = [
#     "SuccessVQA",
#     "VQG2VQA + NLI",
#     "VQG2VQA (NLI corrected) + NLI",
#     "VQG2VQA + Spatial + NLI",
#     "VQG2VQA (NLI corrected) + Spatial + NLI",
#     "VQG2VQA (NLI corrected v2)",
#     "VQG2VQA (NLI corrected v2) + NLI",
#     # "VQG2VQA (NLI corrected v2) + Spatial",
#     "VQG2VQA (NLI corrected v2) + Spatial + NLI",
# ]
# results_names = [
#     "SuccessVQA",
#     "SuccessVQA + TOC",
#     "VQG2VQA",
#     "VQG2VQA + NLI",
#     "VQG2VQA + Spatial",
#     "VQG2VQA + Spatial + NLI",
#     "VQG2VQA + Spatial (Blur k45) + NLI",
#     "VQG2VQA + Spatial (No Rephrase)",
#     "VQG2VQA + Spatial (No Rephrase) + NLI"
# ]
# results_names = [
#     "SuccessVQA",
#     "SuccessVQA + Caption",
#     "VQG2VQA + NLI",
#     "VQG2VQA + Caption + NLI",
#     "VQG2VQA + Spatial + NLI",
#     "VQG2VQA + Caption + Spatial + NLI"
# ]
results_names = [
    "SuccessVQA",
    "Caption -> SuccessVQA",
    "VQG2VQA -> SuccessVQA",
    "VQG2VQA + Spatial -> SuccessVQA",
]

print("(0) Compiling pred data...")
mistake_probs = [[] for _ in results_fnames]
success_probs = [[] for _ in results_fnames]
all_probs = [[] for _ in results_fnames]
mistake_confidence = [[] for _ in results_fnames]
success_confidence = [[] for _ in results_fnames]
all_confidence = [[] for _ in results_fnames]
all_labels = [[] for _ in results_fnames]
all_correctness = [[] for _ in results_fnames]
all_error = [[] for _ in results_fnames]
mistake_error = [[] for _ in results_fnames]
success_error = [[] for _ in results_fnames]
for i, result_fname in enumerate(results_fnames):
    result_preds = json.load(open(result_fname, "r"))
    for pred in result_preds.values():
        example = MistakeDetectionExample.from_dict(pred["example"] | {"frames": ["" for _ in pred["example"]["frame_times"]]}, load_frames=False)
        example.cutoff_to_last_frames(DETECTION_FRAMES_PROPORTION)
        mistake = example.mistake
        
        example_mistake_probs = np.array(pred['mistake_detection']["0.0"]["mistake_probs"])
        mistake_prob = aggregate_mistake_probs_over_frames(example_mistake_probs, example.frame_times)

        if mistake:
            mistake_probs[i].append(mistake_prob)
            mistake_confidence[i].append(abs(mistake_prob - 0.5) * 2.0)
            mistake_error[i].append(mistake_prob if not mistake else 1 - mistake_prob)
        else:
            success_probs[i].append(mistake_prob)
            success_confidence[i].append(abs(mistake_prob - 0.5) * 2.0)
            success_error[i].append(mistake_prob if not mistake else 1 - mistake_prob)

        all_probs[i].append(mistake_prob)
        all_confidence[i].append(abs(mistake_prob - 0.5) * 2.0)
        all_labels[i].append(mistake)
        all_correctness[i].append(True if (mistake_prob >= metrics[i]["best_threshold"] and mistake) or (mistake_prob < metrics[i]["best_threshold"] and not mistake) else False)
        all_error[i].append(mistake_prob if not mistake else 1 - mistake_prob)


# # Analysis 0: Attempt to ensemble
# print("(0) Attempting ensemble...")
# successvqa_preds = json.load(open(results_fnames[0], "r"))
# vqg2vqa_preds = json.load(open(results_fnames[1], "r"))

# # Loop over examples
# ensemble_vqa_outputs = []
# ensemble_dataset = Ego4DMistakeDetectionDataset(data_split="val", 
#                                                       mismatch_augmentation=True,
#                                                       multi_frame=True,
#                                                       debug_n_examples_per_class=250)
# for k in successvqa_preds:
#     vqa_outputs_successvqa = successvqa_preds[k]["vqa"]
#     vqa_outputs_vqg2vqa = vqg2vqa_preds[k]["vqa"]
#     example = MistakeDetectionExample.from_dict(pred["example"] | {"frames": ["" for _ in pred["example"]["frame_times"]]}, load_frames=False)
#     example.cutoff_to_last_frames(DETECTION_FRAMES_PROPORTION)

#     # Loop over frames
#     example_vqa_outputs = []
#     for successvqa_outputs, vqg2vqa_outputs in zip(vqa_outputs_successvqa, vqa_outputs_vqg2vqa):
#         all_outputs = successvqa_outputs + vqg2vqa_outputs

#         # Select VQA output that had the highest confidence among ensembled approaches
#         max_confidence_output = max(all_outputs, key=lambda x: abs(x["answer_probs"]["0"]) - abs(x["answer_probs"]["1"]))
#         max_confidence_output["expected_answer"] = VQAResponse(max_confidence_output["expected_answer"])
#         max_confidence_output["predicted_answer"] = VQAResponse(max_confidence_output["predicted_answer"])
#         max_confidence_output["answer_probs"] = {VQAResponse(int(k)): float(v) for k, v in max_confidence_output["answer_probs"].items()}
#         example_vqa_outputs.append([VQAOutputs(**max_confidence_output, response_token_ids={}, logits=None)])
#         # mistake_answer = 1 - max_confidence_output["expected_answer"]
#         # mistake_prob = max_confidence_output["answer_probs"][str(mistake_answer)]

#     ensemble_vqa_outputs.append(example_vqa_outputs)


# evaluator = HeuristicMistakeDetectionEvaluator(ensemble_dataset, ensemble_vqa_outputs)
# mistake_detection_preds, metrics = evaluator.evaluate_mistake_detection()

# # Compile preds per mistake detection example
# preds = compile_mistake_detection_preds(ensemble_dataset, ensemble_vqa_outputs, mistake_detection_preds)

# output_fname = f"ensembled_det_curve_{'_'.join(results_names).replace(' ', '-')}.pdf"
# save_paths = [os.path.join("/".join(fname.split("/")[:-1]), output_fname) for fname in results_fnames]
# for path in results_fnames:
# # Save metrics, preds, DET curve, config file (which may have some parameters that vary over time), and command-line arguments
#     metrics_filename = f"metrics_ensemble_heuristic_val.json"
#     json.dump(metrics, open(os.path.join("/".join(path.split("/")[:-1]), metrics_filename), "w"), indent=4)

#     preds_filename = f"preds_ensemble_heuristic_val.json"
#     json.dump(preds, open(os.path.join("/".join(path.split("/")[:-1]), preds_filename), "w"), indent=4)

#     det_filename = f"det_ensemble_heuristic_val.pdf"
#     generate_det_curve(metrics, os.path.join(os.path.join("/".join(path.split("/")[:-1]), det_filename)))

# print("Done!")

# Analysis 1: Plot confidence and variance for model predictions on success and mistake predictions
print("(1) Beginning confidence graph generation...")

# Bar graph comparing mistake probability for mistake and success examples
x = np.arange(len(results_names))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
fig.set_figwidth(10)
for i in range(len(results_fnames)):
    rects1 = ax.bar(
        x[i] - width/2, 
        np.mean(mistake_probs[i]), 
        width, 
        yerr=np.std(mistake_probs[i]), 
        label='Mistake Examples' if i == 0 else "", 
        color='red', 
        capsize=5
    )
    rects2 = ax.bar(
        x[i] + width/2, 
        np.mean(success_probs[i]), 
        width, 
        yerr=np.std(success_probs[i]), 
        label='Success Examples' if i == 0 else "", 
        color='green', 
        capsize=5
    )

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Result')
ax.set_ylabel('Mistake Probability')
ax.set_xticks(x)
ax.set_xticklabels(results_names)
ax.legend()

fig.tight_layout()

output_fname = f"confidence_comparison1_val_{'_'.join(results_names).replace(' ', '-')}.pdf"
save_paths = [os.path.join("/".join(fname.split("/")[:-1]), run_folder_name, output_fname) for fname in results_fnames] + [os.path.join(output_dir, output_fname)]
for path in save_paths:
    fig.savefig(path)

# Scatter plot of mistake vs success confidence
x = np.arange(len(results_names))  # the label locations
fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(7)
for i in range(len(results_fnames)):
    y_mistake = mistake_confidence[i]
    y_success = success_confidence[i]
    x_mistake = np.full(len(y_mistake), x[i] - 0.1)  # slight offset for visual clarity
    x_success = np.full(len(y_success), x[i] + 0.1)  # slight offset for visual clarity
    marker_size_mistake = mistake_error[i]
    marker_size_success = success_error[i]

    ax.scatter(x_mistake, y_success, color='red', label='Mistake Examples' if i == 0 else "")
    ax.scatter(x_success, y_mistake, color='green', label='Success Examples' if i == 0 else "")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Result')
ax.set_ylabel('Confidence')
ax.set_xticks(x)
ax.set_xticklabels(results_names)
ax.legend()

fig.tight_layout()

output_fname = f"confidence_comparison2_val_{'_'.join(results_names).replace(' ', '-')}.pdf"
save_paths = [os.path.join("/".join(fname.split("/")[:-1]), run_folder_name, output_fname) for fname in results_fnames] + [os.path.join(output_dir, output_fname)]
for path in save_paths:
    fig.savefig(path)

for i, result_fname in enumerate(results_fnames):
    plt.clf()
    result_name = results_names[i]

    # Define the bins
    bins = np.linspace(0, 1.0, 11)

    # Compute histogram data
    hist1, _ = np.histogram(mistake_probs[i], bins=bins)
    hist2, _ = np.histogram(success_probs[i], bins=bins)

    # Plot the stacked histogram
    bars1 = plt.bar(bins[:-1], hist1, width=bins[1]-bins[0], label='Mistake Examples', color="red", align='edge')
    bars2 = plt.bar(bins[:-1], hist2, width=bins[1]-bins[0], bottom=hist1, label='Success Examples', color="green", align='edge')

    # Add labels with the proportion of mistake vs. success probs
    for bar1, bar2 in zip(bars1, bars2):
        total = bar1.get_height() + bar2.get_height()
        if total > 0:
            proportion1 = bar1.get_height() / total
            proportion2 = bar2.get_height() / total
            plt.text(
                bar1.get_x() + bar1.get_width() / 2,
                bar1.get_height() / 2,
                f'{proportion1:.2f}',
                ha='center',
                va='center',
                color='white'
            )
            plt.text(
                bar2.get_x() + bar2.get_width() / 2,
                bar1.get_height() + bar2.get_height() / 2,
                f'{proportion2:.2f}',
                ha='center',
                va='center',
                color='white'
            )

    # Add labels and title
    plt.xlabel('Mistake Probability')
    plt.ylabel('Count')
    plt.legend()

    # Display the plot
    plt.show()

    output_fname = f"confidence_histogram_val_{result_name.replace(' ', '-')}.pdf"
    save_paths = [os.path.join("/".join(result_fname.split("/")[:-1]), run_folder_name, output_fname)] + [os.path.join(output_dir, output_fname)]
    for path in save_paths:
        fig.savefig(path)

print("(1) Confidence graphs generated!")

# Analysis 2: Correlation of confidences with mistake labels, calibration curves, etc.
print("(2) Beginning correlation analysis of confidences...")
lines = []
for i in range(len(results_fnames)):
    # pprint(all_probs[i])
    # pprint(all_labels[i])
    lines.append(f"{results_names[i]} point biserial correlation between confidence and correctness:")
    lines.append(str(pointbiserialr(all_correctness[i], all_confidence[i])))
    lines.append("")
    lines.append(f"{results_names[i]} Brier score:")
    lines.append(str(brier_score_loss([1 if l else 0 for l in all_labels[i]], all_probs[i], pos_label=1)))
    lines.append("")
    lines.append(f"{results_names[i]} ECE (10 bins):")
    ece_probs = (1.0 - np.expand_dims(np.array(all_probs[i]), 1), np.expand_dims(np.array(all_probs[i]), 1))
    ece_probs = np.concatenate(ece_probs, axis=1)
    ece = expected_calibration_error(ece_probs, [1 if l else 0 for l in all_labels[i]])
    lines.append(str(ece))
    lines.append("\n")

output_fname = f"confidence_analysis_metrics_{'_'.join(results_names).replace(' ', '-')}.txt"
save_paths = [os.path.join("/".join(fname.split("/")[:-1]), run_folder_name, output_fname) for fname in results_fnames] + [os.path.join(output_dir, output_fname)]
for save_path in save_paths:
    with open(save_path, 'w') as f:
        f.write("\n".join(lines))

plt.clf()
for i in range(len(results_fnames)):
    fraction_of_positives, mean_predicted_value = calibration_curve(all_labels[i], all_probs[i], n_bins=20)
    plt.plot(mean_predicted_value, fraction_of_positives, label=results_names[i])

# Plot perfectly calibrated line
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plots (Reliability Diagrams)')
plt.legend(loc='best')
plt.grid()

output_fname = f"calibration_curves_val_{'_'.join(results_names).replace(' ', '-')}.pdf"
save_paths = [os.path.join("/".join(fname.split("/")[:-1]), run_folder_name, output_fname) for fname in results_fnames] + [os.path.join(output_dir, output_fname)]
for path in save_paths:
    plt.savefig(path)

print("(2) Done!")

# Analysis 3: Selective prediction metrics
penalty = 1
thresholds = np.linspace(0.0, 1.0, 101) # [0.0, 0.01, 0.02, 0.03, ..., 0.98, 0.99, 1.0]
thresholds = [t for t in thresholds if t >= 0.5] # [0.50, 0.51, ..., 0.98, 0.99, 1.0] (only keep thresholds at least 0.5 because every class is predicted with at least 0.5 likelihood in binary classification)

all_coverages = []
all_risks = []
for i in range(len(results_fnames)):
    coverages, risks, eff_reliabilities, sp_recalls = [], [], [], []
    for t in tqdm(thresholds, desc="thresholds"):
        c, r, e, spr, _ = calculate_abstention_metrics(all_probs[i], [1 if l else 0 for l in all_labels[i]], t, penalty)
        coverages.append(c)
        risks.append(r)
        eff_reliabilities.append(e)
        sp_recalls.append(spr)

    output_fname = f"selective_prediction_metrics_val_{results_names[i].replace(' ', '-')}.pdf"
    save_paths = [os.path.join("/".join(results_fnames[i].split("/")[:-1]), run_folder_name, output_fname)] + [os.path.join(output_dir, output_fname)]

    plot_abstention_metrics(thresholds, coverages, risks, eff_reliabilities, sp_recalls, results_names[i], save_paths)

    # Save coverage and risk for later risk-coverage curve
    all_coverages.append(coverages)
    all_risks.append(risks)

    # Apply Platt scaling and re-run
    calibration_indices = random.sample(list(range(len(all_probs[i]))), 100)

    calibration_probs = [prob for j, prob in enumerate(all_probs[i]) if j in calibration_indices]
    calibration_labels = [label for j, label in enumerate(all_labels[i]) if j in calibration_indices]

    evaluation_probs = [prob for j, prob in enumerate(all_probs[i]) if j not in calibration_indices]
    evaluation_labels = [label for j, label in enumerate(all_labels[i]) if j not in calibration_indices]

    clf = LogisticRegression().fit(np.expand_dims(np.array(calibration_probs), 1), calibration_labels)
    calibrated_probs = clf.predict_proba(np.expand_dims(np.array(evaluation_probs), 1))[:, 1]
    pprint(calibrated_probs)

    coverages, risks, eff_reliabilities, sp_recalls = [], [], [], []
    for t in tqdm(thresholds, desc="thresholds"):
        c, r, e, spr, _ = calculate_abstention_metrics(calibrated_probs, [1 if l else 0 for l in evaluation_labels], t, penalty)
        coverages.append(c)
        risks.append(r)
        eff_reliabilities.append(e)
        sp_recalls.append(spr)

    output_fname = f"selective_prediction_calibrated_metrics_val_{results_names[i].replace(' ', '-')}.pdf"
    save_paths = [os.path.join("/".join(results_fnames[i].split("/")[:-1]), run_folder_name, output_fname)] + [os.path.join(output_dir, output_fname)]

    plot_abstention_metrics(thresholds, coverages, risks, eff_reliabilities, sp_recalls, results_names[i], save_paths)    


output_fname = f"risk_coverage_val_{'_'.join(results_names).replace(' ', '-')}.pdf"
save_paths = [os.path.join("/".join(results_fnames[i].split("/")[:-1]), run_folder_name, output_fname) for i in range(len(results_fnames))] + [os.path.join(output_dir, output_fname)]
generate_risk_coverage_plot(all_coverages, all_risks, results_names, save_paths)

