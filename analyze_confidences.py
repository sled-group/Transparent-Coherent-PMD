# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pprint import pprint
from scipy.stats import pointbiserialr
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.mistake_detection import MistakeDetectionExample
from travel.data.vqa import VQAOutputs, VQAResponse
from travel.model.mistake_detection import aggregate_mistake_probs_over_frames, DETECTION_FRAMES_PROPORTION, MISTAKE_DETECTION_THRESHOLDS, mistake_detection_metrics, generate_det_curve, HeuristicMistakeDetectionEvaluator, compile_mistake_detection_preds

# Configure arguments here
results_fnames = [
    "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_20240701113527/preds_heuristic_val.json",
    "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/SuccessVQA_ego4d_debug250_llava-1.5-7b-hf_target_object_counter1.0_20240702182300/preds_heuristic_val.json",
    "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240701115231/preds_heuristic_val.json",
    "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_20240701115231/preds_nli_val.json",
    "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial1.0_20240701115730/preds_heuristic_val.json",
    "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d/VQG2VQA_ego4d_debug250_llava-1.5-7b-hf_spatial1.0_20240701115730/preds_nli_val.json",
]
metrics = [
    json.load(open(fname.replace("preds_", "metrics_"), "r")) for fname in results_fnames
]
results_names = [
    "SuccessVQA",
    "SuccessVQA + TOC",
    "VQG2VQA",
    "VQG2VQA + NLI",
    "VQG2VQA + Spatial",
    "VQG2VQA + Spatial + NLI"
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
save_paths = [os.path.join("/".join(fname.split("/")[:-1]), output_fname) for fname in results_fnames]
for path in save_paths:
    fig.savefig(path)

x = np.arange(len(results_names))  # the label locations

# Scatter plot of mistake vs success confidence
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
save_paths = [os.path.join("/".join(fname.split("/")[:-1]), output_fname) for fname in results_fnames]
for path in save_paths:
    fig.savefig(path)

print("(1) Confidence graphs generated!")

# Analysis 2: Correlation of confidences with mistake labels
print("(2) Beginning correlation analysis of confidences...")
for i in range(len(results_fnames)):
    # pprint(all_probs[i])
    # pprint(all_labels[i])
    print(f"{results_names[i]} point biserial correlation between confidence and correctness:")
    pprint(pointbiserialr(all_correctness[i], all_confidence[i]))
    print("")
    print(f"{results_names[i]} Brier score:")
    print(brier_score_loss([1 if l else 0 for l in all_labels[i]], all_probs[i], pos_label=1))
    print("\n")

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
save_paths = [os.path.join("/".join(fname.split("/")[:-1]), output_fname) for fname in results_fnames]
for path in save_paths:
    plt.savefig(path)

print("(2) Done!")