import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, auc
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from typing import Union

from travel.data.vqa import VQAResponse
from travel.data.vqg import VQGOutputs
from travel.model.nli import NLI_MODEL_PATH, run_nli, NLI_HYPOTHESIS_TEMPLATE

def mistake_detection_metrics(labels: list[bool], preds: list[bool]) -> dict[str, float]:
    """Accuracy metrics for mistake detection."""
    this_metrics = {}
    this_metrics['accuracy'] = accuracy_score(labels, preds)
    this_metrics['precision'] = precision_score(labels, preds)
    this_metrics['recall'] = recall_score(labels, preds)
    this_metrics['f1'] = f1_score(labels, preds)

    cm = confusion_matrix(labels, preds)
    TN, FP, FN, TP = cm.ravel()
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    this_metrics['false_positive_rate'] = FPR
    this_metrics['false_negative_rate'] = FNR

    return this_metrics

def effectiveness(is_mistake: bool, mistake_probs: Union[list[float], list[list[float]]]):
    """
    Verifiability metric for mistake detection which measures how well a VLM's answer likelihoods for a set of questions captured the mistake/success of a MistakeDetectionExample.
    
    :param is_mistake: Whether or not there's a mistake in the example.
    :param mistake_probs: List of mistake probabilities for each question used to detect a mistake.
    """
    return_one = False
    mistake_probs = np.array(mistake_probs)
    if len(mistake_probs.shape) == 1:
        mistake_probs = np.expand_dims(mistake_probs, axis=0)
        return_one = True

    effectiveness = []
    for this_mistake_probs in mistake_probs:
        # It only takes one question to indicate a mistake, so use the maximum mistake probability to score this questions set
        max_mistake_prob = max(this_mistake_probs)

        # For mistake examples, we want the max mistake probability to be close to 1.0
        if is_mistake:
            effectiveness.append(max_mistake_prob)

        # For non-mistake examples, we want the max mistake probability to be 0.0
        else:
            effectiveness.append(1.0 - max_mistake_prob)
        
    if not return_one:
        return effectiveness
    else:
        return effectiveness[0]

def consistency_metrics(vqg_outputs: dict[Union[str, int], VQGOutputs]):
    """
    Calculates NLI-based consistency metrics for generated questions, including relevance and informativeness of questions.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)

    all_nli_mistake_probs = torch.zeros((0, 1)).float()
    all_relevance = torch.zeros((0, 1)).float()
    all_informativeness = torch.zeros((0, 1)).float()

    procedure_descriptions = [vqg_output.procedure_description for vqg_output in vqg_outputs.values() for _ in vqg_output.questions]
    hypotheses = [NLI_HYPOTHESIS_TEMPLATE.format(procedure=procedure) for procedure in procedure_descriptions]
    premises_expected = [f"{question} {answer.name}" for vqg_output in vqg_outputs.values() for question, answer in zip(vqg_output.questions, vqg_output.answers)]
    premises_unexpected = [f"{question} {VQAResponse(1-answer.value).name}" for vqg_output in vqg_outputs.values() for question, answer in zip(vqg_output.questions, vqg_output.answers)]    

    probs_expected = run_nli(nli_tokenizer, nli_model, list(zip(premises_expected, hypotheses)))
    probs_unexpected = run_nli(nli_tokenizer, nli_model, list(zip(premises_unexpected, hypotheses)))

    # Relevance: how much mistake probability changes based on answer to question (according to NLI model)
    relevance = torch.abs(probs_expected[:, 0].unsqueeze(1) - probs_unexpected[:, 0].unsqueeze(1)).numpy()

    # Informativeness: actual success probability for expected answer to question (according to NLI model)
    informativeness = probs_expected[:, 1].unsqueeze(1).numpy()

    # Combine by multiplying to form a "consistency" metric
    consistency = relevance * informativeness

    mean_relevance, mean_informativeness, mean_consistency = round(float(np.mean(relevance)), 3), round(float(np.mean(informativeness)), 3), round(float(np.mean(consistency)), 3)
    metrics_by_output = {}
    parallel_idx = 0
    for key, vqg_output in vqg_outputs.items():
        for _ in vqg_output.questions:
            metrics_by_output[key] = {
                "relevance": round(float(relevance[parallel_idx]), 3),
                "informativeness": round(float(informativeness[parallel_idx]), 3),
                "consistency": round(float(consistency[parallel_idx]), 3),
            }
            parallel_idx += 1

    return {
        "relevance": mean_relevance,
        "informativeness": mean_informativeness,
        "consistency": mean_consistency,
        "metrics_by_output": metrics_by_output,
    }

def generate_det_curve(metrics: dict[Union[float, str], dict[str, float]], save_path: str):
    """
    Generates and saves a PDF of a Detection Error Tradeoff (DET) curve for the metrics returned by `MistakeDetectionEvaluator.evaluate_mistake_detection()`. A DET curve plots false positive rate (x-axis) versus false negative rate (y-axis) for a space of detection thresholds, and indicates an "ideal" point to set the threshold in the bottom left corner.

    :param metrics: `metrics` object returned by `evaluate_mistake_detection()`.
    :param save_path: Path to save the PDF of the DET curve.
    """
    # Some of the keys in the metrics file may not be floats (for thresholds), e.g., a "best_metrics" key is also saved here
    metrics = {k: v for k, v in metrics.items() if type(k) == float}

    # Gather FPR and FNR from metrics
    false_positive_rates = [round(metrics[threshold]['false_positive_rate'], 3) for threshold in metrics]
    false_negative_rates = [round(metrics[threshold]['false_negative_rate'], 3) for threshold in metrics]

    # # Ensure input rates are within the valid range for norm.ppf
    # false_positive_rates = np.clip(false_positive_rates, 0.0001, 0.9999)
    # false_negative_rates = np.clip(false_negative_rates, 0.0001, 0.9999)

    # # Convert FPR and FNR to normal deviate scale
    # x = norm.ppf(false_positive_rates)
    # y = norm.ppf(false_negative_rates)
    
    # Ensure all plotted values are finite by filtering out any non-finite values
    # finite_indices = np.isfinite(x) & np.isfinite(y)
    # x = x[finite_indices]
    # y = y[finite_indices]

    x = false_positive_rates
    y = false_negative_rates

    # Plot DET curve
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='magenta')  # Unique color
    
    # Label axes with normal deviate scale
    # plt.xlabel('False Positive Rate (Normal Deviate Scale)')
    # plt.ylabel('False Negative Rate (Normal Deviate Scale)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')    

    # Set grid and title
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Customize axes for better readability
    tick_vals = np.linspace(0.00, 1.0, 11)
    # ticks = norm.ppf(tick_vals)
    ticks = tick_vals
    tick_labels = [f"{round(val, 2)}" for val in tick_vals]
    plt.xticks(ticks, tick_labels)
    plt.yticks(ticks, tick_labels)

    # plt.xlim([norm.ppf(0.01), norm.ppf(0.99)])
    # plt.ylim([norm.ppf(0.01), norm.ppf(0.99)])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.savefig(save_path)

def generate_det_curves(metrics: list[dict[Union[float, str], dict[str, float]]], curve_names: list[str], save_paths: list[str]):
    """
    Generates and saves a PDF of a Detection Error Tradeoff (DET) curve for the metrics returned by `MistakeDetectionEvaluator.evaluate_mistake_detection()`.
     A DET curve plots false positive rate (x-axis) versus false negative rate (y-axis) for a space of detection thresholds, and indicates an "ideal" point 
     to set the threshold in the bottom left corner.

    :param metrics: List of `metrics` objects returned by `evaluate_mistake_detection()`.
    :param curve_names: List of names of the approach associated with each passed entry of `metrics`, e.g., ["Random", "SuccessVQA", "VQG2VQA"].
    :param save_paths: Paths to save copies of the PDF of the DET curve.
    """
    assert len(metrics) == len(curve_names), "Expected same number of metrics and curve names!"

    colors = plt.get_cmap('tab10', len(metrics))
    plt.figure(figsize=(8, 6))
    for i, (metric, name) in enumerate(zip(metrics, curve_names)):
        # Some of the keys in the metrics file may not be floats (for thresholds), e.g., a "best_metrics" key is also saved here
        metric = {k: v for k, v in metric.items() if isinstance(k, float)}

        # Gather FPR and FNR from metrics
        false_positive_rates = [round(metric[threshold]['false_positive_rate'], 3) for threshold in metric]
        false_negative_rates = [round(metric[threshold]['false_negative_rate'], 3) for threshold in metric]

        # Ensure input rates are within the valid range for norm.ppf
        # false_positive_rates = np.clip(false_positive_rates, 0.0001, 0.9999)
        # false_negative_rates = np.clip(false_negative_rates, 0.0001, 0.9999)

        # Convert FPR and FNR to normal deviate scale
        # x = norm.ppf(false_positive_rates)
        # y = norm.ppf(false_negative_rates)
        
        # Ensure all plotted values are finite by filtering out any non-finite values
        # finite_indices = np.isfinite(x) & np.isfinite(y)
        # x = x[finite_indices]
        # y = y[finite_indices]

        x = false_positive_rates
        y = false_negative_rates

        # Plot DET curve
        plt.plot(x, y, marker='o', linestyle='-', color=colors(i), label=name)

    # Label axes with normal deviate scale
    # plt.xlabel('False Positive Rate (Normal Deviate Scale)')
    # plt.ylabel('False Negative Rate (Normal Deviate Scale)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')

    # Set grid and title
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Customize axes for better readability
    tick_vals = np.linspace(0.00, 1.0, 11)
    # ticks = norm.ppf(tick_vals)
    ticks = tick_vals
    tick_labels = [f"{round(val, 2)}" for val in tick_vals]
    plt.xticks(ticks, tick_labels)
    plt.yticks(ticks, tick_labels)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # plt.xlim([norm.ppf(0.01), norm.ppf(0.99)])
    # plt.ylim([norm.ppf(0.01), norm.ppf(0.99)])

    # Add legend
    plt.legend()

    # Save to files
    for save_path in save_paths:
        if not os.path.exists("/".join(save_path.split("/")[:-1])):
            os.makedirs("/".join(save_path.split("/")[:-1]))
        plt.savefig(save_path)


def generate_roc_curves(metrics: list[dict[Union[float, str], dict[str, float]]], curve_names: list[str], save_paths: list[str]):
    """
    Generates and saves a PDF of a Receiver Operating Characteristic (ROC) curve for the metrics returned by `MistakeDetectionEvaluator.evaluate_mistake_detection()`.
     A ROC curve plots false positive rate (x-axis) versus false negative rate (y-axis) for a space of detection thresholds, and indicates an "ideal" point 
     to set the threshold in the bottom left corner. Also saves the area under the ROC curve in a text file in the same location.

    :param metrics: List of `metrics` objects returned by `evaluate_mistake_detection()`.
    :param curve_names: List of names of the approach associated with each passed entry of `metrics`, e.g., ["Random", "SuccessVQA", "VQG2VQA"].
    :param save_paths: Paths to save copies of the PDF of the DET curve.
    """
    assert len(metrics) == len(curve_names), "Expected same number of metrics and curve names!"

    colors = plt.get_cmap('tab10', len(metrics))
    plt.figure(figsize=(8, 6))
    aurocs = []
    for i, (metric, name) in enumerate(zip(metrics, curve_names)):
        # Some of the keys in the metrics file may not be floats (for thresholds), e.g., a "best_metrics" key is also saved here
        metric = {k: v for k, v in metric.items() if isinstance(k, float)}

        # Gather FPR and TPR from metrics
        false_positive_rates = [round(metric[threshold]['false_positive_rate'], 3) for threshold in metric]
        true_positive_rates = [round(1.0 - metric[threshold]['false_negative_rate'], 3) for threshold in metric]

        x = false_positive_rates
        y = true_positive_rates

        # Plot DET curve
        plt.plot(x, y, marker='o', linestyle='-', color=colors(i), label=name)

        x_y_sorted = sorted([(this_x, this_y) for this_x, this_y in zip(x, y)], key=lambda t: t[0])
        x_sorted = [t[0] for t in x_y_sorted]
        y_sorted = [t[1] for t in x_y_sorted]
        aurocs.append(auc(x_sorted, y_sorted))

    # Label axes with normal deviate scale
    # plt.xlabel('False Positive Rate (Normal Deviate Scale)')
    # plt.ylabel('False Negative Rate (Normal Deviate Scale)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')

    # Set grid and title
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Customize axes for better readability
    tick_vals = np.linspace(0.00, 1.0, 11)
    # ticks = norm.ppf(tick_vals)
    ticks = tick_vals
    tick_labels = [f"{round(val, 2)}" for val in tick_vals]
    plt.xticks(ticks, tick_labels)
    plt.yticks(ticks, tick_labels)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # plt.xlim([norm.ppf(0.01), norm.ppf(0.99)])
    # plt.ylim([norm.ppf(0.01), norm.ppf(0.99)])

    # Add legend
    plt.legend()

    # Save to files
    for save_path in save_paths:
        if not os.path.exists("/".join(save_path.split("/")[:-1])):
            os.makedirs("/".join(save_path.split("/")[:-1]))
        plt.savefig(save_path)
        with open(save_path.replace(".pdf", ".txt"), "w") as f:
            f.write("\n".join([f"{result_name}: {auc_metric}\n" for auc_metric, result_name in zip(aurocs, curve_names)]))

def calculate_abstention_metrics(mistake_probs, labels, threshold, error_cost=1):
    """Code adapted from ReCoVERR repo: https://github.com/tejas1995/ReCoVERR/blob/0cbd88de4e5782dc16092ad3dad82a33544ce827/src/VanillaSelectivePrediction.ipynb#L110"""
    num_covered, total_risk, effective_reliability, num_covered_correct, num_correct = 0, 0, 0, 0, 0

    # Convert mistake probabilities into success and mistake probabilities to get answers
    mistake_probs_binary = (1.0 - np.expand_dims(np.array(mistake_probs), 1), np.expand_dims(np.array(mistake_probs), 1))
    mistake_probs_binary = np.concatenate(mistake_probs_binary, axis=1)
    answers = np.argmax(mistake_probs_binary, axis=1)
    confidences_per_answer = mistake_probs_binary[np.arange(len(answers)), answers]

    abstained_idxs = []
    for idx, (ans, conf, label) in enumerate(zip(answers, confidences_per_answer, labels)):
        acc = 1.0 if ans == label else 0.0
        if acc == 1.0:
            num_correct += 1

        selected = True if conf >= threshold else False
        if selected:
            num_covered += 1
            if acc == 1.0:
                num_covered_correct += 1
            total_risk += 1.0 - acc
            effective_reliability += acc if acc > 0.0 else -error_cost
        else:
            abstained_idxs.append(idx)

    coverage = num_covered/len(labels)
    risk = total_risk/num_covered if num_covered > 0 else 0.0
    effective_reliability = effective_reliability/len(labels)
    selective_prediction_recall = num_covered_correct / num_correct if num_correct > 0 else 0.0
    return coverage, risk, effective_reliability, selective_prediction_recall, abstained_idxs

def plot_abstention_metrics(thresholds, coverages, risks, eff_reliabilities, sp_recalls, result_name, save_paths):
    colors = plt.get_cmap('tab10', 4)
    plt.figure(figsize=(8, 6))
    for metric, metric_name, color in [(coverages, "Coverage", colors(0)), 
                                       (risks, "Risk", colors(1)), 
                                       (eff_reliabilities, "Effective Reliability", colors(2)),
                                       (sp_recalls, "Selective Prediction Recall", colors(3))]:
        # Plot DET curve
        plt.plot(thresholds, metric, marker='o', linestyle='-', color=color, label=metric_name)

    # Label axes with normal deviate scale
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Metric Value')
    plt.title(f"{result_name} Selective Prediction Metrics")

    # Set grid and title
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Customize axes for better readability
    tick_vals = np.linspace(0.0, 1.0, 11)
    ticks = tick_vals
    tick_labels = [f"{round(val, 2)}" for val in tick_vals]
    plt.xticks(ticks, tick_labels)
    plt.yticks(ticks, tick_labels)

    plt.xlim([0.5, 1.0])
    plt.ylim([0.0, 1.0])

    # Add legend
    plt.legend()

    # Save to files
    for save_path in save_paths:
        plt.savefig(save_path)

def generate_risk_coverage_plot(coverages, risks, result_names, save_paths):
    """Plots coverage vs. risk for multiple series of data."""
    assert len(coverages) == len(risks)
    colors = plt.get_cmap('tab10', len(coverages))
    plt.figure(figsize=(8, 6))

    aucs = []
    for i, (coverage, risk, result_name) in enumerate(zip(coverages, risks, result_names)):
        plt.plot(coverage, risk, marker='.', linestyle='-', color=colors(i), label=result_name)
        
        coverage_risk_sorted = sorted([(c, r) for c, r in zip(coverage, risk)], key=lambda x: x[0])
        coverage_sorted = [t[0] for t in coverage_risk_sorted]
        risk_sorted = [t[1] for t in coverage_risk_sorted]
        aucs.append(auc(coverage_sorted, risk_sorted))
    
    # Label axes with normal deviate scale
    plt.xlabel('Coverage')
    plt.ylabel('Risk')

    # Set grid and title
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Customize axes for better readability
    tick_vals = np.linspace(0.00, 1.0, 11)
    ticks = tick_vals
    tick_labels = [f"{round(val, 2)}" for val in tick_vals]
    plt.xticks(ticks, tick_labels)
    plt.yticks(ticks, tick_labels)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    # Add legend
    plt.legend()

    # Save to files
    for save_path in save_paths:
        plt.savefig(save_path)
        with open(save_path.replace(".pdf", ".txt"), "w") as f:
            f.write("\n".join([f"{result_name}: {auc_metric}\n" for auc_metric, result_name in zip(aucs, result_names)]))
