from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   
import numpy as np
import os
from pprint import pprint
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, auc, brier_score_loss
import torch
from tqdm import tqdm
from typing import Union, Optional, Any

from travel.model.nli import run_nli, NLI_HYPOTHESIS_TEMPLATE
from travel.model import simple_lm_prompt
from travel.model.mistake_detection import MISTAKE_DETECTION_THRESHOLDS
from travel.model.utils import expected_calibration_error

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
        plt.plot(x, y, marker='o', linestyle='-', color=colors[i], label=name, linewidth=3, markersize=7)

    # Label axes with normal deviate scale, bold font, and slightly larger size
    plt.xlabel('False Positive Rate', fontsize=22, fontweight='bold')
    plt.ylabel('False Negative Rate', fontsize=22, fontweight='bold')

    # Set grid and title
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Customize axes for better readability
    tick_vals = np.linspace(0.00, 1.0, 11)
    ticks = tick_vals
    tick_labels = [f"{round(val, 2)}" for val in tick_vals]
    plt.xticks(ticks, tick_labels, fontsize=16)
    plt.yticks(ticks, tick_labels, fontsize=16)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    # Add legend with a slightly larger font size
    plt.legend(fontsize=18)

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

def rephrase_procedure_success(procedures: list[str], tokenizer, lm, generation_batch_size: int=20):
    examples = [
        "Procedure: Soak the sponge in a soapy water with your hands.\nStatement: The sponge has been successfully soaked in soapy water with someone's hands.",
        "Procedure: Turn on a torch light.\nStatement: The torch light has been successfully turned on.",
        "Procedure: Fold the right edge of the wrapper.\nStatement: The right edge of the wrapper has been successfully folded.",
        "Procedure: Pour the water into the blue container.\nStatement: The water has been successfully poured into the blue container.",
        "Procedure: Spread the black peas on the salad with the spoon in your hand.\nStatement: The black peas have been successfully spread on the salad with the spoon in someone's hand.",
        "Procedure: Pick the scrubber from the sink.\nStatement: The scrubber has been successfully picked from the sink.",
        "Procedure: Peel the onion.\nStatement: The onion has been successfully peeled.",
        "Procedure: Put the dirt in the dust bin.\nStatement: The dirt has been successfully put in the dust bin.",
        "Procedure: Cut dough in two.\nStatement: The dough has been successfully cut in two.",
        "Procedure: Close the fridge.\nStatement: The fridge has been successfully closed.",
    ]
    prompts = ["\n\n".join(examples) + f"\n\Procedure: {procedure}\nStatement: " for procedure in procedures]
    rephrased_texts = simple_lm_prompt(lm, tokenizer, prompts, max_new_tokens=20, batch_size=generation_batch_size, generation_kwargs={"pad_token_id": tokenizer.eos_token_id})
    rephrased_texts = [text.split(".")[0] + "." for text in rephrased_texts]
    rephrased_texts = [text.strip() for text in rephrased_texts]
    print(procedures)
    print(rephrased_texts)
    print("===================")
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
                                   rephrase_batch_size=20,
                                   rephrase_success=False):
    """
    Calculates coherence metrics for candidate questions about procedures in iterative VQA.
    """
    if answers is not None:
        assert all(a in ["Yes", "No"] for a in answers)
    if previous_answers is not None:
        assert all(a in ["Yes", "No"] for aa in previous_answers for a in aa)
    
    metrics = {}
    
    if not rephrase_success:
        hypothesis_procedure = [NLI_HYPOTHESIS_TEMPLATE.format(procedure=procedure) for procedure in procedures]
    else:
        hypothesis_procedure = rephrase_procedure_success(
            procedures,
            lm_tokenizer,
            lm_model,
            generation_batch_size=rephrase_batch_size,
        )
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
                    this_metrics.append(round(float(all_coherence_metrics[k][parallel_idx]), 6)) # NOTE: these numbers can be negative
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

    n_iterations = []
    dialog_info_gain = []
    decision_errors = []
    for results_dict in all_results_dicts.values():
        this_n_iterations = results_dict['final_turn'] + 1
        n_iterations.append(this_n_iterations)

        turn_info_gains = []
        for turn_idx in range(this_n_iterations):
        
            # Get information gain for this turn
            last_turn_success_prob = results_dict['success_probs'][turn_idx - 1] if turn_idx > 0 else None
            this_turn_success_prob = results_dict['success_probs'][turn_idx]

            last_turn_info = (1.0 - entropy(last_turn_success_prob)) if turn_idx > 0 else 0.0
            this_turn_info = 1.0 - entropy(this_turn_success_prob)
            turn_info_gain = this_turn_info - last_turn_info            

            turn_info_gains.append(turn_info_gain)

        dialog_info_gain.append(np.sum(turn_info_gains))

        mistake_prob = 1.0 - results_dict['success_probs'][results_dict['final_turn']]
        decision_errors.append(mistake_prob if not results_dict['mistake'] else 1.0 - mistake_prob)

    # Get ECE with 10 bins            
    ece_probs = (1.0 - np.expand_dims(np.array(all_probs), 1), np.expand_dims(np.array(all_probs), 1))
    ece_probs = np.concatenate(ece_probs, axis=1)
    ece = expected_calibration_error(ece_probs, [1 if l else 0 for l in all_labels])

    # Get Spearman correlations between decision error and relevance/informativeness
    all_rel = coherence_metrics['metrics_by_example']['relevance_marginal_by_example']
    all_inf = coherence_metrics['metrics_by_example']['informativeness_marginal_ref_by_example']
    spearman_error_rel = spearmanr(all_rel, decision_errors)
    spearman_error_inf = spearmanr(all_inf, decision_errors)

    # Selective prediction analysis
    penalty = 1
    thresholds = np.linspace(0.0, 1.0, 101) # [0.0, 0.01, 0.02, 0.03, ..., 0.98, 0.99, 1.0]
    thresholds = [t for t in thresholds if t >= 0.5] # [0.50, 0.51, ..., 0.98, 0.99, 1.0] (only keep thresholds at least 0.5 because every class is predicted with at least 0.5 likelihood in binary classification)

    coverages, risks, eff_reliabilities, sp_recalls = [], [], [], []
    for t in tqdm(thresholds, desc="thresholds"):
        c, r, e, spr, _ = calculate_abstention_metrics(all_probs, [1 if l else 0 for l in all_labels], t, penalty)
        coverages.append(c)
        risks.append(r)
        eff_reliabilities.append(e)
        sp_recalls.append(spr)    

    coverage_risk_sorted = sorted([(c, r) for c, r in zip(coverages, risks)], key=lambda x: x[0])
    coverage_sorted = [t[0] for t in coverage_risk_sorted]
    risk_sorted = [t[1] for t in coverage_risk_sorted]
    aurc = auc(coverage_sorted, risk_sorted)

    other_metrics = {
        "n_iterations": np.mean(n_iterations),
        "dialog_info_gain": np.mean(dialog_info_gain),
        "brier_score": float(brier_score_loss([1 if l else 0 for l in all_labels], all_probs, pos_label=1)),
        "ece10": float(ece),
        "spearman_relevance_error": [float(spearman_error_rel.statistic), float(spearman_error_rel.pvalue)],
        "spearman_informativeness_error": [float(spearman_error_inf.statistic), float(spearman_error_inf.pvalue)],
        "aurc": float(aurc),
    }

    return accuracy_metrics_by_threshold, coherence_metrics, other_metrics

def generate_3d_overview_graph(coherence_metrics, all_results_dicts, dataset, save_path, graph_name="base"):
    """Generates a 3D scatter plot of decision error, example-level relevance, and example-level (reference-adjusted) informativeness for each example in results."""
    accuracy = []
    informativeness = []
    relevance = []

    example_ids = []
    verbs = []
    nouns = []
    mistake_types = []

    labels = [] # These have to be filled in later if we want them
    graph_name = "base"    
    print("Generating 3D scatter plot...")
    for (example_id, output), ex_informativeness, ex_relevance in tqdm(zip(all_results_dicts.items(), coherence_metrics['metrics_by_example']['informativeness_marginal_ref_by_example'], coherence_metrics['metrics_by_example']['relevance_marginal_by_example']), total=len(all_results_dicts)):
        target_success_prob = 0.0 if output['mistake'] else 1.0
        actual_success_prob = output['success_probs'][output['final_turn']]
            
        accuracy.append(np.abs(target_success_prob - actual_success_prob))
        relevance.append(ex_relevance)
        informativeness.append(ex_informativeness)
        
        example_ids.append(example_id)
        example = dataset.get_example_by_id(example_id, load_frames=False)
        verbs.append(example.verb_noun_pair[0].split("_")[0])
        nouns.append(example.verb_noun_pair[1].split("_")[0])
        mistake_types.append(output['mistake_type'])

    matplotlib.use('Agg')

    # Sample data
    x = accuracy
    y = relevance
    z = informativeness

    x_norm = x
    y_norm = y
    z_norm = (np.array(z) + 1.0) / 2.0

    # Combine the normalized values to get the colors
    colors = np.array([x_norm, y_norm, z_norm]).T

    fig = plt.figure(figsize=(10, 10))  # Increase the figure size
    ax = fig.add_subplot(111, projection='3d')

    # Create the scatter plot
    ax.scatter(x, y, z, c=colors, s=150, edgecolor='k', linewidth=0.25, alpha=0.5)

    # Data labels
    if len(labels) > 0:
        for i in range(len(x)):
            ax.text(x[i], y[i], z[i] + 0.03, labels[i], size=8, zorder=1, color='k')

    # Set custom z-axis tick labels
    xy_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(xy_ticks)
    ax.set_yticks(xy_ticks)
    ax.set_xticklabels([f'{tick:.1f}' for tick in xy_ticks], fontsize=12)
    ax.set_yticklabels([f'{tick:.1f}' for tick in xy_ticks], fontsize=12)

    z_ticks = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_zticks(z_ticks)
    ax.set_zticklabels([f'{tick:.1f}' for tick in z_ticks], fontsize=12)
            
    # Set axis labels
    ax.set_xlabel(f'Decision Error', labelpad=6, fontsize=20, fontweight='bold')
    ax.set_ylabel(f'Relevance', labelpad=6, fontsize=20, fontweight='bold')
    ax.set_zlabel(f'Informativeness', labelpad=6, fontsize=20, fontweight='bold')

    ax.xaxis.label.set_color('#AA0000')
    ax.yaxis.label.set_color('#00AA00')
    ax.zaxis.label.set_color('#0000AA')

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect(aspect=None, zoom=0.94)

    plt.tight_layout()

    # Display the plot
    plt.show()
    plt.savefig(os.path.join(save_path, f"3d_graph_{graph_name}.pdf"), bbox_inches='tight')
