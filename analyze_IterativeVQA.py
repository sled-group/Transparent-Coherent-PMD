# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

from collections import Counter
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pprint import pprint
import random
from scipy.stats import pointbiserialr, spearmanr
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from tqdm import tqdm

from travel.constants import RESULTS_DIR
from travel.model.metrics import generate_risk_coverage_plot, calculate_abstention_metrics, plot_abstention_metrics, generate_det_curves, entropy
from travel.model.utils import expected_calibration_error

# Configure results to graph here
TASK = "ego4d_single"
timestamp = datetime.datetime.now()
run_folder_name = f"confidence_analysis_{timestamp.strftime('%Y%m%d%H%M%S')}"
parent_output_dir = os.path.join(RESULTS_DIR, f"analysis", TASK, run_folder_name)

for results_fnames, results_names, analysis_subdir in [
    # (
    #     [
    #         "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q10_ego4d_single_debug250_llava-1.5-7b-hf_likelihood_20240805180404/outputs_val.json",
    #         "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q10_ego4d_single_debug250_llava-1.5-7b-hf_coherence_20240807080723/outputs_val.json",
    #     ],
    #     [
    #         "Likelihood Ranking",
    #         "Coherence Ranking",
    #     ],
    #     "likelihood_vs_coherence"
    # ),
    # (
    #     [
    #         "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q10_ego4d_single_debug250_llava-1.5-7b-hf_coherence_20240807080723/outputs_val.json",
    #         "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q10_ego4d_single_debug250_llava-1.5-7b-hf_coherence_icl20_20240806224704/outputs_val.json",
    #     ],
    #     [
    #         "Coherence Ranking",
    #         "Coherence Ranking (+ ICL candidates)",
    #     ],
    #     "introducing_icl"
    # ),
    # (
    #     [
    #         "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q5_ego4d_single_debug250_llava-1.5-7b-hf_coherence_icl20_20240804224634/outputs_val.json",
    #         "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q5_ego4d_single_debug250_llava-1.5-7b-hf_coherence_icl20_contrastive_region1.0_20240805163803/outputs_val.json",
    #         "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q5_ego4d_single_debug250_llava-1.5-7b-hf_coherence_icl20_agla2.0_20240806115545/outputs_val.json",
    #         "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q5_ego4d_single_debug250_llava-1.5-7b-hf_coherence_icl20_spatial_blur55.0_20240807112322/outputs_val.json",
    #     ],
    #     [
    #         "No Filter",
    #         "CRG",
    #         "AGLA",
    #         "Spatial",
    #     ],
    #     "visual_filters_coherence"
    # ),
    # (
    #     [
    #         # "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q5_ego4d_single_debug250_llava-1.5-7b-hf_likelihood_20240805122212/outputs_val.json",
    #         "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q5_ego4d_single_debug250_llava-1.5-7b-hf_likelihood_icl20_20240805002323/outputs_val.json",
    #         "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q5_ego4d_single_debug250_llava-1.5-7b-hf_likelihood_icl20_contrastive_region1.0_20240806132636/outputs_val.json",
    #     ],
    #     [
    #         # "No ICL",
    #         "No Filter",
    #         "CRG",
    #     ],
    #     "visual_filters_likelihood"
    # ),    
    (
        [
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_topdown_q10_ego4d_single_debug250_llava-1.5-7b-hf_beam8-4_likelihood_20240812143901/outputs_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_topdown_q10_ego4d_single_debug250_llava-1.5-7b-hf_beam8-4_coherence_20240812143858/outputs_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q10_ego4d_single_debug250_llava-1.5-7b-hf_beam8-4_likelihood_20240812143853/outputs_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q10_ego4d_single_debug250_llava-1.5-7b-hf_beam8-4_coherence_20240812143853/outputs_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q10_ego4d_single_debug250_llava-1.5-7b-hf_beam8-4_likelihood_icl20_20240812183411/outputs_val.json",
            "/home/sstorks/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqa_mistake_detection/ego4d_single_debug250/llava-1.5-7b-hf/IterativeVQA_q10_ego4d_single_debug250_llava-1.5-7b-hf_beam8-4_coherence_icl20_20240812172402/outputs_val.json",
        ],
        [
            "Top Down Likelihood Ranking",
            "Top Down Coherence Ranking",
            "Likelihood Ranking",
            "Coherence Ranking",
            "Likelihood Ranking + ICL",
            "Coherence Ranking + ICL",
        ],
        "likelihood_vs_coherence"
    )
]:
    # Set up subdirectory
    output_dir = os.path.join(parent_output_dir, analysis_subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configure arguments here
    for results_fname in results_fnames:
        if not os.path.exists(os.path.join(os.path.join("/".join(results_fname.split("/")[:-1]), run_folder_name))):
            os.makedirs(os.path.join(os.path.join("/".join(results_fname.split("/")[:-1]), run_folder_name)))
                    
    metrics_accuracy = [
        json.load(open(fname.replace("outputs_", "metrics_accuracy_"), "r")) for fname in results_fnames
    ]
    pprint(metrics_accuracy)
    metrics_det = [{float(k): v for k, v in metrics.items() if k not in ["best_metrics", "best_threshold"]} for metrics in metrics_accuracy]


    # Analysis 0: Generate DET curves
    print("(0) Generating DET curves...")
    output_fname = f"det_comparison_{'_'.join(results_names).replace(' ', '-')}.pdf"
    save_paths = [os.path.join("/".join(fname.split("/")[:-1]), run_folder_name, output_fname) for fname in results_fnames] + [os.path.join(output_dir, output_fname)]
    generate_det_curves(metrics_det, results_names, save_paths=save_paths)

    print("Compiling pred data...")
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
    n_turns = [[] for _ in results_fnames]
    turn_success_probs = [[] for _ in results_fnames]
    turn_success_probs_noimg = [[] for _ in results_fnames]
    turn_questions = [[] for _ in results_fnames]
    turn_questions_sources = [[] for _ in results_fnames]
    for i, result_fname in enumerate(results_fnames):
        result_preds = json.load(open(result_fname, "r"))
        result_preds_noimg = None
        if os.path.exists(result_fname.replace("outputs_val.json", "/results_image_free_SuccessVQA/outputs_val.json")):
            result_preds_noimg = json.load(open(result_fname.replace("outputs_val.json", "/results_image_free_SuccessVQA/outputs_val.json"), "r"))

        for pred_idx, pred in enumerate(result_preds.values()):
            mistake = pred['mistake']
            
            mistake_prob = 1.0 - pred['success_probs'][pred['final_turn']]

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
            all_correctness[i].append(True if (mistake_prob >= metrics_accuracy[i]["best_threshold"] and mistake) or (mistake_prob < metrics_accuracy[i]["best_threshold"] and not mistake) else False)
            all_error[i].append(mistake_prob if not mistake else 1 - mistake_prob)

            n_turns[i].append(pred['final_turn'] + 1)
            turn_success_probs[i].append(pred['success_probs'])
            if result_preds_noimg is not None:
                turn_success_probs_noimg[i].append(list(result_preds_noimg.values())[pred_idx]['success_probs'])
            turn_questions[i].append(pred['questions'])
            if 'candidate_questions_sources' in pred:
                turn_questions_sources[i].append([cs[cq.index(q)] for q, cq, cs in zip(pred['questions'], pred['candidate_questions'], pred['candidate_questions_sources'])])

    # Analysis 1: save number of turns spent on each example and other stats about VQG->VQA turns
    print("(1) Running efficiency analysis...")
    lines = []
    for i in range(len(results_fnames)):
        lines.append(f"{results_names[i]} average number of turns: {np.mean(n_turns[i])}")

        all_dialog_info_gains = []
        all_turn_info_gains = []
        all_turn_mse = []

        all_dialog_info_gains_noimg = []
        all_turn_info_gains_noimg = []
        all_turn_mse_noimg = []

        all_dialog_info_diffs = []
        all_turn_info_diffs = []

        question_sources = Counter()
        for label_idx, label in enumerate(all_labels[i]):
            turn_info_gains = []
            turn_info_gains_noimg = []
            turn_info_diffs = []

            turn_mse = []
            turn_mse_noimg = []

            for turn_idx in range(n_turns[i][label_idx]):

                # Get information gain for this turn
                last_turn_success_prob = turn_success_probs[i][label_idx][turn_idx - 1] if turn_idx > 0 else None
                this_turn_success_prob = turn_success_probs[i][label_idx][turn_idx]

                last_turn_info = (1.0 - entropy(last_turn_success_prob)) if turn_idx > 0 else 0.0
                this_turn_info = 1.0 - entropy(this_turn_success_prob)
                turn_info_gain = this_turn_info - last_turn_info
                turn_info_gains.append(turn_info_gain)

                target = 0.0 if label else 1.0
                this_turn_mse = np.abs(turn_success_probs[i][label_idx][turn_idx] - target) ** 2
                turn_mse.append(this_turn_mse)

                # If possible, get information gain for this turn without an image available in success VQA step
                if len(turn_success_probs_noimg[i]) > 0:
                    last_turn_success_prob_noimg = turn_success_probs_noimg[i][label_idx][turn_idx - 1] if turn_idx > 0 else None
                    this_turn_success_prob_noimg = turn_success_probs_noimg[i][label_idx][turn_idx]
                    
                    last_turn_info_noimg = (1.0 - entropy(last_turn_success_prob_noimg)) if turn_idx > 0 else 0.0
                    this_turn_info_noimg = 1.0 - entropy(this_turn_success_prob_noimg)
                    turn_info_gain_noimg = this_turn_info_noimg - last_turn_info_noimg
                    turn_info_gains_noimg.append(turn_info_gain_noimg)

                    turn_info_diffs.append(turn_info_gain - turn_info_gain_noimg)

                    this_turn_mse_noimg = np.abs(turn_success_probs_noimg[i][label_idx][turn_idx] - target) ** 2
                    turn_mse_noimg.append(this_turn_mse_noimg)

                if len(turn_questions_sources[i]) > 0:
                    q_source = turn_questions_sources[i][label_idx][turn_idx]
                    question_sources[q_source] += 1

            all_dialog_info_gains.append(np.sum(turn_info_gains))
            all_turn_info_gains.append(np.mean(turn_info_gains))
            all_turn_mse.append(turn_mse[-1]) # Just take last prediction MSE

            if len(turn_success_probs_noimg[i]) > 0:
                all_dialog_info_gains_noimg.append(np.sum(turn_info_gains_noimg))
                all_turn_info_gains_noimg.append(np.mean(turn_info_gains_noimg))
                all_turn_mse_noimg.append(np.mean(turn_mse_noimg))

                all_dialog_info_diffs.append(np.sum(turn_info_gains) - np.sum(turn_info_gains_noimg))
                all_turn_info_diffs.append(np.mean(turn_info_gains) - np.mean(turn_info_gains_noimg))

        mean_dialog_info_gain = np.mean(all_dialog_info_gains)
        mean_turn_info_gain = np.mean(all_turn_info_gains)
        mean_mse = np.mean(all_turn_mse)
        lines.append(f"{results_names[i]} average information gained per turn: {mean_turn_info_gain}")
        lines.append(f"{results_names[i]} average information gained per dialog: {mean_dialog_info_gain}")
        lines.append(f"{results_names[i]} MSE across turns: {mean_mse}")

        if len(turn_success_probs_noimg[i]) > 0:
            mean_dialog_info_gain_noimg = np.mean(all_dialog_info_gains_noimg)
            mean_turn_info_gain_noimg = np.mean(all_turn_info_gains_noimg)
            mean_mse_noimg = np.mean(all_turn_mse_noimg)
            lines.append(f"{results_names[i]} average information gained per turn (no image): {mean_turn_info_gain_noimg}")
            lines.append(f"{results_names[i]} average information gained per dialog (no image): {mean_dialog_info_gain_noimg}")
            lines.append(f"{results_names[i]} MSE across turns (no image): {mean_mse_noimg}")

            mean_dialog_info_diff = np.mean(all_dialog_info_diffs)
            mean_turn_info_diff = np.mean(all_turn_info_diffs)
            lines.append(f"{results_names[i]} average information lost per turn by having no image: {mean_turn_info_diff}")
            lines.append(f"{results_names[i]} average information lost per dialog by having no image: {mean_dialog_info_diff}")            

        for source in question_sources:
            lines.append(f"{results_names[i]} percent of turns from source '{source}': {question_sources[source] / sum(list(question_sources.values()))}")
        lines.append("")

    output_fname = f"turn_metrics_{'_'.join(results_names).replace(' ', '-')}.txt"
    save_paths = [os.path.join("/".join(fname.split("/")[:-1]), run_folder_name, output_fname) for fname in results_fnames] + [os.path.join(output_dir, output_fname)]
    for save_path in save_paths:
        with open(save_path, 'w') as f:
            f.write("\n".join(lines))
    print("(1) Done!")

    # Analysis 2: Plot confidence and variance for model predictions on success and mistake predictions
    print("(2) Beginning confidence graph generation...")

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

    print("(2) Confidence graphs generated!")

    # Analysis 3: Correlation of confidences with mistake labels, calibration curves, etc.
    print("(3) Beginning correlation analysis of confidences...")
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

    print("(3) Done!")

    # Analysis 4: Selective prediction metrics
    print("(4) Running selective prediction metrics...")
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
        if len(all_probs[i]) > 100:
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
    print("(4) Done!")

    # Analysis 5: Correlations between tiered metrics
    print("(5) Beginning tiered metrics correlation analysis...")
    lines = []
    for i, result_fname in enumerate(results_fnames):
        coherence_metrics_path = result_fname.replace("outputs_val.json", "metrics_coherence_nli_val.json")
        coherence_metrics = None
        if os.path.exists(coherence_metrics_path):
            coherence_metrics = json.load(open(coherence_metrics_path, "r"))

        if coherence_metrics is not None:
            all_relevance = coherence_metrics['metrics_by_example']["relevance_marginal_by_example"]
            all_informativeness = coherence_metrics['metrics_by_example']["informativeness_marginal_x_relevance_marginal_by_example"]

            res = spearmanr(all_informativeness, all_error[i])
            lines.append(f"{results_names[i]} Spearman correlation between informativeness x relevance and error: {res.statistic} (p={res.pvalue})")

            res2 = pointbiserialr(all_correctness[i], all_informativeness)
            lines.append(f"{results_names[i]} point-biserial correlation between informativeness x relevance and correctness: {res2[0]} (p={res2[1]})")
                         
            lines.append("")

            output_fname = f"tiered_correlations_{'_'.join(results_names).replace(' ', '-')}.txt"
            save_paths = [os.path.join("/".join(results_fnames[i].split("/")[:-1]), run_folder_name, output_fname) for i in range(len(results_fnames))] + [os.path.join(output_dir, output_fname)]
            for save_path in save_paths:
                with open(save_path, 'w') as f:
                    f.write("\n".join(lines))

    print("(5) Done!")
