from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pprint import pprint
from scipy.stats import norm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Union, Any
import yaml

from travel.data.mistake_detection import MistakeDetectionExample, MistakeDetectionDataset
from travel.data.utils import generate_float_series
from travel.model.vqa import VQAOutputs, VQAResponse

MISTAKE_DETECTION_THRESHOLDS = [round(threshold, 2) for threshold in generate_float_series(0.0, 1.0, 0.05)]

def mistake_detection_metrics(labels: list[bool], preds: list[bool]) -> dict[str, float]:
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


@dataclass
class MistakeDetectionOutputs:
    """Class to hold mistake detection outputs for individual frames. If using full video clips for VQA, just use a placeholder time."""
    example_id: str
    frame_times: list[float] # (# frames)
    mistake_probs: list[list[float]] # (# frames, # questions per frame)
    detection_threshold: float
    final_mistake_prediction: bool

    def __post_init__(self):
        assert len(self.frame_times) == len(self.mistake_probs), "`MistakeDetectionOutputs` expects `frame_times` and `mistake_probs` to be the same length, i.e., the number of frames used to detect the mistake."

    def to_dict(self):
        """Helper method to create a JSON-serializable version of the class instance (excluding some information)."""
        return_dict = asdict(self)
        return_dict['frame_times'] = [float(round(ft, 3)) for ft in return_dict['frame_times']]
        return_dict['mistake_probs'] = [[float(round(v, 3)) for v in l] for l in return_dict['mistake_probs']]
        return_dict['detection_threshold'] = float(round(return_dict['detection_threshold'], 3))
        return return_dict

@dataclass_json
@dataclass
class MistakeDetectionEvaluator:
    """Superclass to implement different types of evaluators based on VQAOutputs."""
    examples: MistakeDetectionDataset
    vqa_outputs: list[list[list[VQAOutputs]]] # (# examples, frames per example, questions per frame)
    
    def __post_init__(self):
        """Validates `examples` and `vqa_outputs` used to initialize the class."""

        assert len(self.examples) == len(self.vqa_outputs), "Should have same number of examples and VQAOutputs lists."
        n_outputs_per_example = len(self.vqa_outputs[0][0])
        for frame_outputs in self.vqa_outputs:
            for output in frame_outputs:
                try:
                    assert len(output) == n_outputs_per_example, f"All examples should have same number of VQAOutputs ({n_outputs_per_example})."
                except:
                    pprint(output)
                    raise
    
    def check_mistakes(self, detection_threshold: float = 0.5) -> list[MistakeDetectionOutputs]:
        """
        Given `examples` and `vqa_outputs`, determine whether there is a mistake.
        
        :param detection_threshold: Confidence threshold for evaluator to predict there's a mistake, typically based on an LM's logits.
        :return: True if there is believed to be a mistake, else False.
        """
        raise NotImplementedError("Subclass should define the strategy to check for mistakes.")
    
    def evaluate_mistake_detection(self) -> tuple[dict[float, list[MistakeDetectionOutputs]], dict[Union[float, str], dict[str, float]]]:
        """
        Calculates mistake detection evaluation metrics for this class instance's `examples` and `vqa_outputs`.

        :return: 2 `dict` objects mapping a detection threshold to a `list` of predictions or a `dict` of metrics.
        """
        labels = [example.mistake for example in self.examples]
        
        combined_preds = {}
        metrics = {}
        best_metrics = None
        best_threshold = None
        for threshold in MISTAKE_DETECTION_THRESHOLDS:
            pred_objects = self.check_mistakes(threshold)
            preds = [pred.final_mistake_prediction for pred in pred_objects]

            this_metrics = mistake_detection_metrics(labels, preds)
            combined_preds[threshold] = pred_objects
            metrics[threshold] = this_metrics

            # Save best metrics based on which threshold minimizes FPR and FNR
            if best_metrics is None or (this_metrics['false_positive_rate'] + this_metrics['false_negative_rate']) < (best_metrics['false_positive_rate'] + best_metrics['false_negative_rate']):
                best_metrics = this_metrics
                best_threshold = threshold

        metrics['best_metrics'] = best_metrics
        metrics['best_threshold'] = best_threshold
        return combined_preds, metrics

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
DETECTION_FRAMES_PROPORTION = int(config["mistake_detection_strategies"]["frames_proportion"]) # Use last N% of frames for frame-based mistake detection strategies

class HeuristicMistakeDetectionEvaluator(MistakeDetectionEvaluator):
    """Heuristic mistake detection evaluator which simply takes the average mistake probability over the passed chunk of video frames."""
        
    def check_mistakes(self, detection_threshold: float=0.5) -> list[MistakeDetectionOutputs]:
        """
        Given `examples` and `vqa_outputs`, determine whether there is a mistake. Uses a simple heuristic by averaging the mistake probability over all passed frames to determine whether there is a mistake. Typically, we should pass only a small percentage of the last frames of a clip to judge the completion of an action.
        
        :param detection_threshold: Confidence threshold for evaluator to predict there's a mistake, typically based on an LM's logits.
        :return: True if there is believed to be a mistake, else False.
        """
        # If any VQA answers don't match expected answers, there's a mistake (we can decide to make this more lenient later)
        mistake_probs = []
        for example, outputs in zip(self.examples, self.vqa_outputs):
            example_mistake_probs = []
            for frame_outputs in outputs:
                # Check all questions for this frame to decide if there's a mistake
                frame_mistake_probs = []
                for output in frame_outputs:
                    mistake_answer = VQAResponse(1-int(output.expected_answer.value))
                    frame_mistake_probs.append(output.answer_probs[mistake_answer])
                example_mistake_probs.append(frame_mistake_probs)
            mistake_probs.append(example_mistake_probs)
            
        # Heuristic: average likelihood of mistake then use a threshold to decide if there's a mistake
        agg_preds = []
        for mistake_prob, example in tqdm(zip(mistake_probs, self.examples), desc="evaluating mistake detection", total=len(self.examples)):
            if len(mistake_prob) > 0:
                example.cutoff_to_last_frames(DETECTION_FRAMES_PROPORTION) # Call this again since the example got reloaded from cache
                assert len(example.frame_times) == len(mistake_prob), "Compilation of mistake detections for example has a shape issue!"
                
                mean_mistake_prob = np.max(mistake_prob, axis=1) # Get maximum probability of a mistake for each frame (since we only need one question to indicate a mistake)
                mean_mistake_prob = np.mean(mistake_prob) # Get mean mistakeprobability over all frames
                                
                mistake_pred_final = True if mean_mistake_prob > detection_threshold else False
            else:
                # If there are no frames to predict over, this is probably because some filter was applied to remove images that don't have a target object;
                # in this case, the target object is likely not present at all in the video, suggesting an incorrect object is used instead
                mistake_prob = [[]]
                mistake_pred_final = True

            pred_object = MistakeDetectionOutputs(
                example_id=example.example_id,
                frame_times=example.frame_times,
                mistake_probs=mistake_prob,
                detection_threshold=detection_threshold,
                final_mistake_prediction=mistake_pred_final
            )

            agg_preds.append(pred_object)            
            
        return agg_preds

        
MISTAKE_DETECTION_STRATEGIES = {
    "heuristic": HeuristicMistakeDetectionEvaluator
}

# TODO: accept multiple sets of metrics as input for comparison; also add a legend where each curve has a name (e.g., SuccessVQA)
def generate_det_curve(metrics: dict[Union[float, str], dict[str, float]], save_path: str):
    """
    Generates and saves a PDF of a Detection Error Tradeoff (DET) curve for the metrics returned by `MistakeDetectionEvaluator.evaluate_mistake_detection()`. A DET curve plots false positive rate (x-axis) versus false negative rate (y-axis) for a space of detection thresholds, and indicates an "ideal" point to set the threshold in the bottom left corner.

    :param metrics: `metrics` object returned by `evaluate_mistake_detection()`.
    :param save_path: Path to save the PDF of the DET curve.
    """
    # Some of the keys in the metrics file may not be floats (for thresholds), e.g., a "best_metrics" key is also saved here
    metrics = {k: v for k, v in metrics.items() if type(k) == float}

    # Gather FPR and FNR from metrics
    false_positive_rates = [round(1.0 - metrics[threshold]['false_positive_rate'], 3) for threshold in metrics]
    false_negative_rates = [round(1.0 - metrics[threshold]['false_negative_rate'], 3) for threshold in metrics]

    # Ensure input rates are within the valid range for norm.ppf
    false_positive_rates = np.clip(false_positive_rates, 0.0001, 0.9999)
    false_negative_rates = np.clip(false_negative_rates, 0.0001, 0.9999)

    # Convert FPR and FNR to normal deviate scale
    x = norm.ppf(false_positive_rates)
    y = norm.ppf(false_negative_rates)
    
    # Ensure all plotted values are finite by filtering out any non-finite values
    finite_indices = np.isfinite(x) & np.isfinite(y)
    x = x[finite_indices]
    y = y[finite_indices]

    # Plot DET curve
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='magenta')  # Unique color
    
    # Label axes with normal deviate scale
    plt.xlabel('False Positive Rate (Normal Deviate Scale)')
    plt.ylabel('False Negative Rate (Normal Deviate Scale)')
    
    # Set grid and title
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Customize axes for better readability
    tick_vals = np.linspace(0.00, 1.0, 11)
    ticks = norm.ppf(tick_vals)
    tick_labels = [f"{round(val, 2)}" for val in tick_vals]
    plt.xticks(ticks, tick_labels)
    plt.yticks(ticks, tick_labels)

    plt.xlim([norm.ppf(0.01), norm.ppf(0.99)])
    plt.ylim([norm.ppf(0.01), norm.ppf(0.99)])

    plt.savefig(save_path)

def compile_mistake_detection_preds(dataset: MistakeDetectionDataset,
                                    vqa_outputs: list[list[list[VQAOutputs]]],
                                    mistake_detection_preds: dict[float, list[MistakeDetectionOutputs]]) -> dict[str, dict[str, Any]]:
    """
    Helper function to compile mistake detection examples with model predictions from VQA and mistake detection.

    :param dataset: Mistake detection dataset used for evaluation.
    :param vqa_outputs: Ragged list of VQA outputs; shape should correspond to (# examples, # frames, # questions per frame)
    """
    compiled_preds = {example.example_id: {"example": example.to_dict()} for example in dataset}
    assert len(dataset) == len(vqa_outputs), "Expected same number of dataset examples and VQAOutputs lists."
    for example_outputs, example in zip(vqa_outputs, dataset):
        example_id = example.example_id
        compiled_preds[example_id]["vqa"] = [[question_output.to_dict() for question_output in frame_outputs] for frame_outputs in example_outputs]
    for threshold in mistake_detection_preds:
        for pred in mistake_detection_preds[threshold]:
            example_id = pred.example_id
            if "mistake_detection" not in compiled_preds[example_id]:
                compiled_preds[example_id]["mistake_detection"] = {}
            compiled_preds[example_id]["mistake_detection"][threshold] = pred.to_dict()
    return compiled_preds
