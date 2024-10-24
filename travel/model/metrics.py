from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, auc
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from typing import Union, Optional, Any

from travel.data.mistake_detection import MistakeDetectionExample
from travel.data.utils import time_based_exponential_moving_average
from travel.data.vqa import VQAResponse, VQAOutputs, generate_iterative_vqa_success_prompt
from travel.data.vqg import VQGOutputs
from travel.model.nli import NLI_MODEL_PATH, run_nli, NLI_HYPOTHESIS_TEMPLATE
from travel.model import simple_lm_prompt

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

def generate_det_curves(metrics: list[dict[Union[float, str], dict[str, float]]], curve_names: list[str], save_paths: list[str], colors=None):
    """
    Generates and saves a PDF of a Detection Error Tradeoff (DET) curve for the metrics returned by `MistakeDetectionEvaluator.evaluate_mistake_detection()`.
     A DET curve plots false positive rate (x-axis) versus false negative rate (y-axis) for a space of detection thresholds, and indicates an "ideal" point 
     to set the threshold in the bottom left corner.

    :param metrics: List of `metrics` objects returned by `evaluate_mistake_detection()`.
    :param curve_names: List of names of the approach associated with each passed entry of `metrics`, e.g., ["Random", "SuccessVQA", "VQG2VQA"].
    :param save_paths: Paths to save copies of the PDF of the DET curve.
    """
    assert len(metrics) == len(curve_names), "Expected same number of metrics and curve names!"

    if colors is None:
        colors = plt.get_cmap('tab10', len(metrics))
        colors = [colors(i) for i in range(len(metrics))]
    
    plt.figure(figsize=(6, 5.6))  # Adjusted to make the plot nearly square

    for i, (metric, name) in enumerate(zip(metrics, curve_names)):
        # Some of the keys in the metrics file may not be floats (for thresholds), e.g., a "best_metrics" key is also saved here
        metric = {k: v for k, v in metric.items() if isinstance(k, float)}

        # Gather FPR and FNR from metrics
        false_positive_rates = [round(metric[threshold]['false_positive_rate'], 3) for threshold in metric]
        false_negative_rates = [round(metric[threshold]['false_negative_rate'], 3) for threshold in metric]

        x = false_positive_rates
        y = false_negative_rates

        # Plot DET curve with thicker lines and dots
        plt.plot(x, y, marker='o', linestyle='-', color=colors[i], label=name, linewidth=2, markersize=6)

    # Label axes with normal deviate scale, bold font, and slightly larger size
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('False Negative Rate', fontsize=14, fontweight='bold')

    # Set grid and title
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Customize axes for better readability
    tick_vals = np.linspace(0.00, 1.0, 11)
    ticks = tick_vals
    tick_labels = [f"{round(val, 2)}" for val in tick_vals]
    plt.xticks(ticks, tick_labels, fontsize=12)
    plt.yticks(ticks, tick_labels, fontsize=12)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    # Add legend with a slightly larger font size
    plt.legend(fontsize=14)

    # Remove extra space around the plot
    plt.tight_layout()

    # Save to files
    for save_path in save_paths:
        if not os.path.exists("/".join(save_path.split("/")[:-1])):
            os.makedirs("/".join(save_path.split("/")[:-1]))
        plt.savefig(save_path, bbox_inches='tight')


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


def generate_tiered_metric_curves(thresholds: list[float], accuracies: list[float], consistencies: list[float], verifiabilities: list[float], save_paths: list[str]):
    """
    Generates and saves a PDF of a Detection Error Tradeoff (DET) curve for the metrics returned by `MistakeDetectionEvaluator.evaluate_mistake_detection()`.
     A DET curve plots false positive rate (x-axis) versus false negative rate (y-axis) for a space of detection thresholds, and indicates an "ideal" point 
     to set the threshold in the bottom left corner.

    :param metrics: List of `metrics` objects returned by `evaluate_mistake_detection()`.
    :param curve_names: List of names of the approach associated with each passed entry of `metrics`, e.g., ["Random", "SuccessVQA", "VQG2VQA"].
    :param save_paths: Paths to save copies of the PDF of the DET curve.
    """
    assert len(thresholds) == len(accuracies) == len(consistencies) == len(verifiabilities), "Expected same number of all metrics!"

    colors = plt.get_cmap('tab10', 3)
    plt.figure(figsize=(8, 6))

    # Plot DET curve
    plt.plot(thresholds, accuracies, marker='.', linestyle='-', color=colors(0), label="Accuracy")
    plt.plot(thresholds, consistencies, marker='.', linestyle='-', color=colors(1), label="Consistency")
    plt.plot(thresholds, verifiabilities, marker='.', linestyle='-', color=colors(2), label="Verifiability")

    plt.xlabel('Confidence Threshold')
    plt.ylabel('Metric Value')

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

    # Add legend
    plt.legend()

    # Save to files
    for save_path in save_paths:
        if not os.path.exists("/".join(save_path.split("/")[:-1])):
            os.makedirs("/".join(save_path.split("/")[:-1]))
        plt.savefig(save_path)


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


# def calculate_relevance_informativeness(procedure, question, vlm_answer, is_mistake):
#     hypothesis = NLI_HYPOTHESIS_TEMPLATE.format(procedure=procedure)
#     prob_e = get_entailment_probability([question + " " + vlm_answer.name], [hypothesis] * 2)
#     prob_u = get_entailment_probability([question + " " + VQAResponse(1-vlm_answer.value).name], [hypothesis] * 2)
    
#     negation_based_relevance = float(np.abs(prob_e - prob_u))
#     informativeness = float(prob_e if not is_mistake else 1.0 - prob_e)

#     return negation_based_relevance, informativeness

def rephrase_question_answer(questions: list[str], answers: list[str], tokenizer, lm, generation_batch_size: int=20):
    # return f"{question} {answer.name}."
    assert all(a == "Yes" or a == "No" for a in answers), "Expected all answers to be 'Yes' or 'No', but got " + str(answers)
    examples = [
        "Question: Is there a bowl on the table?\nAnswer: Yes\nStatement: There is a bowl on the table.",
        "Question: Are the eggs cracked?\nAnswer: No\nStatement: The eggs are not cracked.",
        "Question: Does the cardboard box look open?\nAnswer: Yes\nStatement: The cardboard box looks open.",
        "Question: Are there any leaves outside of the basket?\nAnswer: No\nStatement: There are not any leaves outside of the basket.",
        "Question: Is the orange peeled?\nAnswer: Yes\nStatement: The orange is peeled.",
        "Question: Is the mug empty?\nAnswer: No\nStatement: The mug is not empty.",
        "Question: Are there hedge trimmers in the image?\nAnswer: Yes\nStatement: There are hedge trimmers in the image.",
        "Question: Has the light switch been turned on?\nAnswer: No\nStatement: The light switch has not been turned on.",
        "Question: Does the table have any cups on it?\nAnswer: Yes\nStatement: The table has cups on it.",
        "Question: Is the cabinet closed?\nAnswer: No\nStatement: The cabinet is not closed.",
    ]
    prompts = ["\n\n".join(examples) + f"\n\nQuestion: {question}\nAnswer: {answer}\nStatement: " for question, answer in zip(questions, answers)]
    rephrased_texts = simple_lm_prompt(lm, tokenizer, prompts, max_new_tokens=20, batch_size=generation_batch_size, generation_kwargs={"pad_token_id": tokenizer.eos_token_id})
    rephrased_texts = [text.split(".")[0] + "." for text in rephrased_texts]
    rephrased_texts = [text.strip() for text in rephrased_texts]
    return rephrased_texts

def entropy(binary_prob):
    if binary_prob == 0.0 or binary_prob == 1.0:
        return 0.0
    ent = binary_prob * np.log2(binary_prob)
    ent += (1.0 - binary_prob) * np.log2(1.0 - binary_prob)
    return -ent

def entropy_tensor(binary_prob):
    ent = -binary_prob
    ent[binary_prob != 0.0] *= torch.log2(binary_prob[binary_prob != 0.0])
    ent[binary_prob != 1.0] -= (1.0 - binary_prob[binary_prob != 1.0]) * torch.log2(1.0 - binary_prob[binary_prob != 1.0])
    ent[binary_prob == 0.0] = 0.0
    ent[binary_prob == 1.0] = 0.0
    return ent

def question_coherence_metrics_nli(nli_tokenizer, nli_model, lm_tokenizer, lm_model, procedures: list[str], questions: list[str], 
                                   answers: Optional[list[str]]=None, 
                                   previous_questions: Optional[list[list[str]]]=None, 
                                   previous_answers: Optional[list[list[str]]]=None, 
                                   mistake_labels: Optional[list[bool]]=None, 
                                   rephrase_batch_size=20):
    """
    Calculates coherence metrics for candidate questions about procedures in iterative VQA.
    """
    if answers is not None:
        assert all(a in ["Yes", "No"] for a in answers)
    if previous_answers is not None:
        assert all(a in ["Yes", "No"] for aa in previous_answers for a in aa)
    
    metrics = {}
    
    hypothesis_procedure = [NLI_HYPOTHESIS_TEMPLATE.format(procedure=procedure) for procedure in procedures]
    # Rephrase question with a yes and no answer as statements to compare their entailment probability of success
    rephrased_yes = rephrase_question_answer(
        questions, 
        ["Yes"] * len(questions),
        lm_tokenizer, 
        lm_model, 
        generation_batch_size=rephrase_batch_size
    )
    rephrased_no = rephrase_question_answer(
        questions, 
        ["No"] * len(questions),
        lm_tokenizer, 
        lm_model, 
        generation_batch_size=rephrase_batch_size
    )    
    metrics['rephrased_questions_yes'] = rephrased_yes
    metrics['rephrased_questions_no'] = rephrased_no
    premise_yes = rephrased_yes
    premise_no = rephrased_no
    probs_yes = run_nli(nli_tokenizer, nli_model, list(zip(premise_yes, hypothesis_procedure)))
    probs_no = run_nli(nli_tokenizer, nli_model, list(zip(premise_no, hypothesis_procedure)))
    if answers:
        probs_actual = torch.stack([probs_yes[i] if answers[i] == "Yes" else probs_no[i] for i in range(len(answers))])

    # Individual relevance: how much probability of success changes depending on the answer
    relevance = torch.abs(probs_yes[:, 0] - probs_no[:, 0])
    metrics['relevance'] = relevance.numpy()

    # Potential informativeness: at most (or for actual answer), how confident will we be that the answer to the question would indicate a success or mistake
    if not answers:
        informativeness_yes = 1.0 - entropy_tensor(probs_yes[:, 0])
        informativeness_no = 1.0 - entropy_tensor(probs_no[:, 0])
        informativeness = torch.max(torch.cat((informativeness_yes.unsqueeze(1), informativeness_no.unsqueeze(1)), dim=-1), dim=-1).values
    else:
        informativeness = 1.0 - entropy_tensor(probs_actual[:, 0])
    metrics['informativeness'] = informativeness.numpy()

    if previous_questions:
        assert len(previous_questions) == len(previous_answers), "Expected same number of questions and answers!"

        # Flatten and rephrase past questions and answers into statements, then un-flatten
        rephrased_past = rephrase_question_answer(
            [question for p_questions in previous_questions for question in p_questions],
            [answer for p_answers in previous_answers for answer in p_answers],
            lm_tokenizer,
            lm_model,
            generation_batch_size=rephrase_batch_size
        )
        parallel_idx = 0
        new_rephrased_past = []
        for p_questions in previous_questions:
            this_rephrased_past = []
            for _ in p_questions:
                this_rephrased_past.append(rephrased_past[parallel_idx])
                parallel_idx += 1
            new_rephrased_past.append(this_rephrased_past)
        rephrased_past = new_rephrased_past

        premise_past = [" ".join(past_qs_rephrased) for past_qs_rephrased in rephrased_past]
        
        premise_past_yes = [(pp + " " + py).strip() for pp, py in zip(premise_past, premise_yes)]
        premise_past_no = [(pp + " " + pn).strip() for pp, pn in zip(premise_past, premise_no)]
        # probs_past = run_nli(nli_tokenizer, nli_model, list(zip(premise_past, hypothesis_procedure)))
        probs_past_yes = run_nli(nli_tokenizer, nli_model, list(zip(premise_past_yes, hypothesis_procedure)))
        probs_past_no = run_nli(nli_tokenizer, nli_model, list(zip(premise_past_no, hypothesis_procedure)))
        if answers:
            probs_past_actual = torch.stack([probs_past_yes[i] if answers[i] == "Yes" else probs_past_no[i] for i in range(len(answers))])
        
        # Marginal relevance: how much probability of success changes depending on the answer AND information we already extracted from the image
        relevance_marginal = torch.abs(probs_past_yes[:, 0] - probs_past_no[:, 0])
        metrics['relevance_marginal'] = relevance_marginal.numpy()

        # Marginal expected informativeness: with past questions and answers, how informative could (or is) the answer to this question be toward the final decision
        if not answers:
            informativeness_yes = 1.0 - entropy_tensor(probs_past_yes[:, 0])
            informativeness_no = 1.0 - entropy_tensor(probs_past_no[:, 0])
            informativeness_marginal = torch.max(torch.cat((informativeness_yes.unsqueeze(1), informativeness_no.unsqueeze(1)), dim=-1), dim=-1).values
        else:
            informativeness_marginal = 1.0 - entropy_tensor(probs_past_actual[:, 0])
        metrics['informativeness_marginal'] = informativeness_marginal.numpy()
    else:
        metrics['relevance_marginal'] = metrics['relevance']
        metrics['informativeness_marginal'] = metrics['informativeness']

    # "Verifiability" metrics weight marginal informativeness by marginal relevance
    metrics['informativeness_x_relevance_marginal'] = metrics['informativeness'] * metrics['relevance_marginal']
    metrics['informativeness_marginal_x_relevance_marginal'] = metrics['informativeness_marginal'] * metrics['relevance_marginal']

    if answers is not None and mistake_labels is not None:
        # Calculate an alternative (not reference free) form of informativeness that is negative if leaning toward the incorrect final answer (for mistake or success)
        # (this can only be done when answers is provided, which is during final coherence evaluation rather than coherence-based reranking)
        leaning_toward_mistake = torch.tensor([1 if p < 0.5 else -1 for p in probs_past_actual[:, 1]])
        actually_is_mistake = torch.tensor([1 if l else -1 for l in mistake_labels])
        multipliers = leaning_toward_mistake * actually_is_mistake # This will be 1 if leaning the correct way, otherwise -1
        assert sum(multipliers.shape) == len(mistake_labels)
        multipliers = multipliers.numpy()

        for k in ['informativeness', 'informativeness_x_relevance_marginal', 'informativeness_marginal', 'informativeness_marginal_x_relevance_marginal']:
            metrics[k + "_ref"] = metrics[k] * multipliers

    # Convert to floats to ensure json serializable
    metrics = {
        k: [round(float(val), 6) for val in v] if type(v[0]) != str else v
        for k, v in metrics.items()
    }
    return metrics

def question_coherence_metrics_vlm(success_probs, success_probs_negated):
    """
    Calculates coherence metrics for candidate questions about procedures in iterative VQA.
    """
    metrics = {}
    
    # Generate VLM prompts
    assert len(success_probs) == len(success_probs_negated), "Expected same number of success probs and negated success probs."

    for example_success_probs, example_success_probs_negated in zip(success_probs, success_probs_negated):
        assert len(example_success_probs) == len(example_success_probs_negated), "Expected same number of success probs and negated success probs for each example."

        example_success_probs = np.array(example_success_probs)
        example_success_probs_negated = np.array(example_success_probs_negated)
    
        # NOTE: unless we were to run success VQA on each individual question without prior questions/answers, it's impossible to get individual relevance and informativeness (so just use marginal for now)

        # Marginal relevance: how much probability of success changes depending on the answer
        relevance_marginal = np.abs(example_success_probs - example_success_probs_negated)
        metrics['relevance_marginal'] = np.concatenate((metrics['relevance_marginal'], relevance_marginal), axis=0) if 'relevance_marginal' in metrics else relevance_marginal

        # Marginal informativeness: with all previous information we have, how much information does the actual answer to the question give us about success?
        informativeness_marginal = 1.0 - entropy_tensor(torch.tensor(example_success_probs)).numpy()
        metrics['informativeness_marginal'] = np.concatenate((metrics['informativeness_marginal'], informativeness_marginal), axis=0) if 'informativeness_marginal' in metrics else informativeness_marginal

        # "Verifiability" metric: weight marginal informativeness by marginal relevance
        metrics['informativeness_marginal_x_relevance_marginal'] = np.concatenate((metrics['informativeness_marginal_x_relevance_marginal'], informativeness_marginal * relevance_marginal), axis=0) if 'informativeness_marginal_x_relevance_marginal' in metrics else informativeness_marginal * relevance_marginal

    for k in metrics:
        assert metrics[k].shape[0] == len(success_probs) * len(success_probs[0]), "Didn't get correct number of VLM coherence metrics."

    # Convert to floats to ensure json serializable
    metrics = {
        k: [round(float(val), 6) for val in v]
        for k, v in metrics.items()
    }
    return metrics

def compile_accuracy_and_coherence_metrics(all_labels, all_probs, all_coherence_metrics, all_results_dicts, thresholds, unsure_range):
    # Aggregate coherence metrics by example and by turn
    coherence_metrics_by_example = defaultdict(list)
    coherence_metrics_by_turn = defaultdict(list)
    coherence_metric_names = ['relevance', 
                              'relevance_marginal', 
                              'informativeness', 
                              'informativeness_x_relevance_marginal',
                              'informativeness_marginal', 
                              'informativeness_marginal_x_relevance_marginal',
                              'informativeness_ref',
                              'informativeness_marginal_ref',
                              'informativeness_marginal_x_relevance_marginal_ref']
    for k in coherence_metric_names:
        if k in all_coherence_metrics:
            parallel_idx = 0
            for results_dict in all_results_dicts.values():
                this_metrics = []
                for question_idx in range(results_dict['final_turn'] + 1):
                    this_metrics.append(max(round(float(all_coherence_metrics[k][parallel_idx]), 6), 0.0)) # If negative, just round up to 0.0 for aggregated metrics
                    parallel_idx += 1
                coherence_metrics_by_turn[k + "_by_turn"].append(this_metrics)
                # In metric for full example, don't count informativeness for "unsure" answers - model failed to get new information
                this_metrics = [this_metrics[question_idx] if np.abs(results_dict['answer_probs'][question_idx][0] - 0.5) >= unsure_range or "informativeness" not in k else 0.0 for question_idx in range(len(this_metrics))]
                
                # We'll usually just take the mean across all turns, but we take max marginal informativeness across dialog
                if k != "informativeness_marginal" and k != "informativeness_marginal_ref":
                    example_metric = round(float(np.mean(this_metrics)), 6)
                else:
                    example_metric = round(float(np.max(this_metrics)), 6)

                coherence_metrics_by_example[k + "_by_example"].append(example_metric)
                
    # Reweight informativeness metrics by entropy of VLM's answer to questions
    for k in ['informativeness', 'informativeness_marginal', 'informativeness_x_relevance_marginal', 'informativeness_marginal_x_relevance_marginal']:
        if k in all_coherence_metrics:
            parallel_idx = 0
            for results_dict in all_results_dicts.values():
                this_metrics = []
                for question_idx in range(results_dict['final_turn'] + 1):
                    question_info = 1.0 - entropy(results_dict['answer_probs'][question_idx][0])
                    this_metrics.append(round(float(all_coherence_metrics[k][parallel_idx] * question_info), 6))
                    parallel_idx += 1
                coherence_metrics_by_example[k + "_vlm_reweight_by_example"].append(round(float(np.mean(this_metrics)), 6))
                coherence_metrics_by_turn[k + "_vlm_reweight_by_turn"].append(this_metrics)
            
    # Add an alternative "information gain" version of informativeness (weighted by relevance);
    # this metric looks at how much (relevance-weighted) information our answers accrued across the dialog
    parallel_idx = 0
    for results_dict in all_results_dicts.values():
        this_metrics = []
        max_so_far = 0.0
        total_info_gain = 0.0
        for question_idx in range(results_dict['final_turn'] + 1):
            info = max(all_coherence_metrics['informativeness_marginal_x_relevance_marginal_ref'][parallel_idx], 0.0)
            info_gain = max(info - max_so_far, 0.0)
            if info > max_so_far:
                max_so_far = info
            this_metrics.append(round(float(info_gain), 6))            
            parallel_idx += 1
        coherence_metrics_by_example["informativeness_marginal_x_relevance_marginal_ref_gain_by_example"].append(round(float(np.mean(this_metrics)), 6))
        coherence_metrics_by_turn["informativeness_marginal_x_relevance_marginal_ref_gain_by_turn"].append(this_metrics)

    # Calculate 

    # Calculate accuracy metrics
    best_metrics = None
    best_threshold = None
    accuracy_metrics_by_threshold = {}
    coherence_metrics_by_threshold = {}
    all_labels_binary = [True if l is not None else False for l in all_labels]
    for threshold in thresholds:
        preds = [1.0 - p >= threshold for p in all_probs] # Have to do 1.0 - probability since we got "success" probability from VLM
        assert len(preds) == len(all_probs) == len(all_labels), "Expected same number of preds, probs, and labels."
        this_metrics = mistake_detection_metrics(all_labels_binary, preds)
        accuracy_metrics_by_threshold[threshold] = this_metrics

        # Calculate consistency and verifiability for this example, which are conditional on correctness
        verifiability = np.mean([
            coherence_metrics_by_example['informativeness_marginal_ref_by_example'][i] * coherence_metrics_by_example['relevance_marginal_by_example'][i] if preds[i] == all_labels_binary[i] else 0.0 
            for i in range(len(preds))
        ])
        consistency = np.mean([coherence_metrics_by_example['relevance_marginal_by_example'][i] if preds[i] == all_labels_binary[i] else 0.0 for i in range(len(preds))])
        coherence_metrics_by_threshold[threshold] = {"verifiability": verifiability, "consistency": consistency,}

        if best_metrics is None or (this_metrics['false_positive_rate'] + this_metrics['false_negative_rate']) < (best_metrics['false_positive_rate'] + best_metrics['false_negative_rate']):
            best_metrics = this_metrics
            best_threshold = threshold

    accuracy_metrics_by_threshold['best_metrics'] = best_metrics
    accuracy_metrics_by_threshold['best_threshold'] = best_threshold

    coherence_metric_names += [k + "_vlm_reweight" for k in ['informativeness', 'informativeness_x_relevance_marginal', 'informativeness_marginal', 'informativeness_marginal_x_relevance_marginal']]
    coherence_metrics = {
        k: round(float(np.mean(coherence_metrics_by_example[k + "_by_example"])), 6) for k in coherence_metric_names if k + "_by_example" in coherence_metrics_by_example
    } | {
        "metrics_by_threshold": coherence_metrics_by_threshold,
        "metrics_by_example": coherence_metrics_by_example,
        "metrics_by_turn": coherence_metrics_by_turn,
    }

    return accuracy_metrics_by_threshold, coherence_metrics

# NOTE: below consistency and verifiability metrics are from legacy results and not in use/maintained
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

def mask_verbs_and_nouns(text, nlp, mask_token):
    nouns = []
    for token in nlp(text):
        if token.pos_.startswith("N") or (token.pos_.startswith("V") and token.text != "is"):
            nouns.append(token.text)
    for noun in nouns:
        text = text.replace(noun, mask_token)
    return text

def consistency_metrics_vqg(vqg_outputs: dict[Union[str, int], VQGOutputs]):
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
    nlp = spacy.load("en_core_web_lg")

    # This is just an evaluation at VQG time - base it on each individual procedure
    procedure_descriptions = [vqg_output.procedure_description for vqg_output in vqg_outputs.values() for _ in vqg_output.questions]
    hypotheses = [NLI_HYPOTHESIS_TEMPLATE.format(procedure=procedure) for procedure in procedure_descriptions]
    premises_expected = [f"{question} {answer.name}" for vqg_output in vqg_outputs.values() for question, answer in zip(vqg_output.questions, vqg_output.answers)]
    premises_unexpected = [f"{question} {VQAResponse(1-answer.value).name}" for vqg_output in vqg_outputs.values() for question, answer in zip(vqg_output.questions, vqg_output.answers)]    
    # premises_unexpected = [mask_verbs_and_nouns(question, nlp, nli_tokenizer.mask_token) + " " + nli_tokenizer.mask_token for output in vqg_outputs.values() for question, answer in zip(output.questions, output.answers)] # Create "unexpected" premises by masking out all nouns

    probs_expected = run_nli(nli_tokenizer, nli_model, list(zip(premises_expected, hypotheses)))
    probs_unexpected = run_nli(nli_tokenizer, nli_model, list(zip(premises_unexpected, hypotheses)))

    # Relevance: how much mistake probability changes based on answer to question (according to NLI model)
    relevance = torch.abs(probs_expected[:, 0].unsqueeze(1) - probs_unexpected[:, 0].unsqueeze(1)).numpy()

    # Informativeness: actual success probability for expected answer to question (according to NLI model)
    informativeness = probs_expected[:, 1].unsqueeze(1).numpy()

    # Combine by multiplying to form a "consistency" metric
    consistency = relevance * informativeness

    # Take mean informativeness for each question set
    mean_relevance, mean_informativeness, mean_consistency = round(float(np.mean(relevance)), 3), round(float(np.mean(informativeness)), 3), round(float(np.mean(consistency)), 3)

    metrics_by_output = {}
    parallel_idx = 0
    for key, vqg_output in vqg_outputs.items():
        metrics_by_output[key] = {
            "relevance": [round(float(relevance[parallel_idx]), 3), round(float(relevance[parallel_idx + 1]), 3)],
            "informativeness": [round(float(informativeness[parallel_idx]), 3), round(float(informativeness[parallel_idx + 1]), 3)],
            "consistency": [round(float(consistency[parallel_idx]), 3), round(float(consistency[parallel_idx + 1]), 3)],
        }
        parallel_idx += 2

    return {
        "relevance": mean_relevance,
        "informativeness": mean_informativeness,
        "consistency": mean_consistency,
        "metrics_by_output": metrics_by_output,
    }

def consistency_metrics_vqg2vqa(vqg_outputs: dict[Union[str, int], VQGOutputs], mistake_labels: list[bool], vqa_outputs: list[list[list[VQAOutputs]]], frame_times: list[list[float]]):
    """
    Calculates NLI-based consistency metrics for generated questions, including relevance and informativeness of questions.
    """
    # TODO: use actual mistake probabilities from VQA rather than vqa_outputs? This would allow using adjusted probabilities from mistake detection strategies
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
    nlp = spacy.load("en_core_web_lg")

    # This is an evaluation at VQA inference time - incorporate mistake detection labels in informativeness
    # TODO: should we also use predicted yes/no answers to judge this? This would require an overhaul as we'd have to average predictions over frames
    procedure_descriptions = [vqg_outputs[question_output.procedure_id].procedure_description for example_outputs in vqa_outputs for frame_outputs in example_outputs for question_output in frame_outputs]
    hypotheses = [NLI_HYPOTHESIS_TEMPLATE.format(procedure=procedure) for procedure in procedure_descriptions]
    premises_expected = [f"{question_output.question} {question_output.predicted_answer.name}" for example_outputs in vqa_outputs for frame_outputs in example_outputs for question_output in frame_outputs]
    premises_unexpected = [mask_verbs_and_nouns(question_output.question, nlp, nli_tokenizer.mask_token) + " " + nli_tokenizer.mask_token for example_outputs in vqa_outputs for frame_outputs in example_outputs for question_output in frame_outputs] # Create "unexpected" premises by masking out all nouns
    premises_negated = [f"{question_output.question} {VQAResponse(1-question_output.predicted_answer.value).name}" for example_outputs in vqa_outputs for frame_outputs in example_outputs for question_output in frame_outputs]

    probs_expected = run_nli(nli_tokenizer, nli_model, list(zip(premises_expected, hypotheses)))
    probs_unexpected = run_nli(nli_tokenizer, nli_model, list(zip(premises_unexpected, hypotheses)))

    # Relevance: how much mistake probability changes based on answer to question (according to NLI model)
    relevance = torch.abs(probs_expected[:, 0].unsqueeze(1) - probs_unexpected[:, 0].unsqueeze(1)).numpy()

    # Informativeness: actual success probability for expected answer to question (according to NLI model)
    # If mistake_labels were passed, calculate informativeness based on them;
    # if a success example, we want high entailment probability from the expected answers,
    # but if a mistake example, we want high contradiction probability from the unexpected answers
    entailment_prob_indices = torch.tensor([0 if l else 1 for li, l in enumerate(mistake_labels) for _ in range(len(vqa_outputs[li])) for _ in range(2)])
    informativeness = probs_expected[torch.arange(len([li for li, l in enumerate(mistake_labels) for _ in range(len(vqa_outputs[li])) for _ in range(2)])), entailment_prob_indices]
    informativeness = informativeness.unsqueeze(1).numpy()

    # Combine by multiplying to form a "consistency" metric
    consistency = relevance * informativeness

    # Aggregate metrics across frames in each example, using an exponential moving average to ensure earlier frames do not count as much
    mean_relevance, mean_informativeness, mean_consistency = [], [], []
    metrics_by_example = {}
    parallel_idx = 0

    mean_relevance = []
    mean_informativeness = []
    mean_consistency = []
    for example_idx, example_outputs in enumerate(vqa_outputs):
        example_relevance = []
        example_informativeness = []
        example_consistency = []
        example_frame_times = frame_times[example_idx]
        
        for frame_outputs in example_outputs:
            frame_relevance = []
            frame_informativeness = []
            frame_consistency = []
            for question_output in frame_outputs:
                frame_relevance.append(round(float(relevance[parallel_idx]), 3))
                frame_informativeness.append(round(float(informativeness[parallel_idx]), 3))
                frame_consistency.append(round(float(consistency[parallel_idx]), 3))
                parallel_idx += 1

            example_relevance.append(frame_relevance)
            example_informativeness.append(frame_informativeness)
            example_consistency.append(frame_consistency)
        
        metrics_by_example[example_outputs[0][0].example_id] = {
            "relevance": example_relevance,
            "informativeness": example_informativeness,
            "consistency": example_consistency,
        }

        mean_relevance.append(
            time_based_exponential_moving_average(
                [max(frame_relevance) for frame_relevance in example_relevance], 
                frame_times[example_idx]
            )[-1]
        )
        mean_informativeness.append(
            time_based_exponential_moving_average(
                [max(frame_informativeness) for frame_informativeness in example_informativeness], 
                frame_times[example_idx]
            )[-1]
        )
        mean_consistency.append(
            time_based_exponential_moving_average(
                [max(frame_consistency) for frame_consistency in example_consistency], 
                frame_times[example_idx]
            )[-1]
        )

    mean_relevance = np.mean(mean_relevance)
    mean_informativeness = np.mean(mean_informativeness)
    mean_consistency = np.mean(mean_consistency)
            
    return {
        "relevance": mean_relevance,
        "informativeness": mean_informativeness,
        "consistency": mean_consistency,
        "metrics_by_example": metrics_by_example,
    }

def consistency_metrics_caption(captions: list[list[str]], procedure_descriptions: list[str], mistake_labels: list[bool], example_ids: list[int], frame_times: list[list[float]]):
    """
    Calculates NLI-based consistency metrics for generated captions, including relevance and informativeness of captions.
    """
    assert len(captions) == len(procedure_descriptions) == len(mistake_labels) == len(example_ids) == len(frame_times), f"All inputs to consistency_metrics_caption must be the same length! (got {len(captions)} captions, {len(procedure_descriptions)} procedures, {len(mistake_labels)} labels, {len(example_ids)} example IDs, and {len(frame_times)} frame times lists)"

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
    nlp = spacy.load("en_core_web_lg")
    
    hypotheses = [NLI_HYPOTHESIS_TEMPLATE.format(procedure=procedure) for procedure_idx, procedure in enumerate(procedure_descriptions) for _ in range(len(captions[procedure_idx]))]
    premises_expected = [caption for frames_captions in captions for caption in frames_captions]
    premises_unexpected = [mask_verbs_and_nouns(caption, nlp, nli_tokenizer.mask_token) for frames_captions in captions for caption in frames_captions] # Create "unexpected" premises by masking out all nouns 
    # premises_unexpected = [caption.replace(" is ", " is not ") for caption in captions]

    probs_expected = run_nli(nli_tokenizer, nli_model, list(zip(premises_expected, hypotheses)))
    probs_unexpected = run_nli(nli_tokenizer, nli_model, list(zip(premises_unexpected, hypotheses)))

    # Relevance: how much mistake probability changes based on information in caption
    relevance = torch.abs(probs_expected[:, 0].unsqueeze(1) - probs_unexpected[:, 0].unsqueeze(1)).numpy()

    # Informativeness: actual success probability based on the caption 
    # (we want this to be high for success examples, low for mistake examples, so look at entailment probability for success and contradiction probability for mistakes)
    # TODO: is this comparable to VQG metric for informativeness? can we make the VQG metric more comparable?
    entailment_prob_indices = torch.tensor([0 if l else 1 for li, l in enumerate(mistake_labels) for _ in captions[li]])
    informativeness = probs_expected[torch.arange(len(hypotheses)), entailment_prob_indices].unsqueeze(1).numpy()

    # Combine by multiplying to form a "consistency" metric
    consistency = relevance * informativeness

    # Aggregate metrics across frames in each example, using an exponential moving average to ensure earlier frames do not count as much
    mean_relevance, mean_informativeness, mean_consistency = [], [], []
    metrics_by_example = {}
    parallel_idx = 0
    for example_idx, example_id in enumerate(example_ids):
        this_relevance = []
        this_informativeness = []
        this_consistency = []
        for _ in frame_times[example_idx]:
            this_relevance.append(round(float(relevance[parallel_idx]), 3))
            this_informativeness.append(round(float(informativeness[parallel_idx]), 3))
            this_consistency.append(round(float(consistency[parallel_idx]), 3))
            parallel_idx += 1
        mean_relevance.append(time_based_exponential_moving_average(this_relevance, frame_times[example_idx])[-1])
        mean_informativeness.append(time_based_exponential_moving_average(this_informativeness, frame_times[example_idx])[-1])
        mean_consistency.append(time_based_exponential_moving_average(this_consistency, frame_times[example_idx])[-1])

        metrics_by_example[example_id] = {
            "relevance": this_relevance,
            "informativeness": this_informativeness,
            "consistency": this_consistency,
        }
    mean_relevance = round(float(np.mean(mean_relevance)), 3)
    mean_informativeness = round(float(np.mean(mean_informativeness)), 3)
    mean_consistency = round(float(np.mean(mean_consistency)), 3)

    return {
        "relevance": mean_relevance,
        "informativeness": mean_informativeness,
        "consistency": mean_consistency,
        "metrics_by_example": metrics_by_example,
    }
