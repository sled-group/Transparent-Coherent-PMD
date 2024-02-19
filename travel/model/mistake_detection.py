from collections import Counter
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

from travel.constants import DATA_CACHE_DIR
from travel.data.mistake_detection import MistakeDetectionExample, MistakeDetectionDataset, get_cutoff_time_by_proportion
from travel.data.utils import generate_float_series
from travel.model.vqa import VQAOutputs, VQAResponse, VQG2VQA_PROMPT_TEMPLATES
from travel.model.vqg import VQGOutputs

MISTAKE_DETECTION_THRESHOLDS = [round(threshold, 2) for threshold in generate_float_series(0.0, 1.0, 0.05)]

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
    examples: list[MistakeDetectionExample] # (# examples)
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
    
    def evaluate_mistake_detection(self) -> tuple[dict[float, ], dict[float, dict[str, float]]]:
        """
        Calculates mistake detection evaluation metrics for this class instance's `examples` and `vqa_outputs`.

        :return: 2 `dict` objects mapping a detection threshold to a `list` of predictions or a `dict` of metrics.
        """
        labels = [example.mistake for example in self.examples]
        
        combined_preds = {}
        metrics = {}
        for threshold in MISTAKE_DETECTION_THRESHOLDS:
            pred_objects = self.check_mistakes(threshold)
            preds = [pred.final_mistake_prediction for pred in pred_objects]

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

            combined_preds[threshold] = pred_objects
            metrics[threshold] = this_metrics
            
        return combined_preds, metrics
    
    # TODO: this can only handle one frame per example - smooth this over later
    def get_logits_errors(self) -> torch.FloatTensor:
        """
        For RLHF-like applications, scores the errors of VQA prediction logits (after sigmoid) compared to ground truth mistake detection labels.

        :return: FloatTensor of logit errors with shape (# examples, # questions per example, # possible question responses)
        """

        for frame_outputs in self.vqa_outputs:
            assert len(frame_outputs) == 1, "get_logits_errors expects one frame per example."
            assert frame_outputs[0].logits is not None, "get_logits_errors expects VQAOutputs to have logits"
        
        vqa_outputs_flat = [outputs[0] for outputs in self.vqa_outputs]
        
        assert len(vqa_outputs_flat[0][0].response_token_ids) == 2, "get_logits_errors() depends on VQA responses being binary (e.g., yes or no). If there are more response classes, you cannot use this method."
        target_logits = torch.stack([torch.stack([torch.nn.functional.one_hot(torch.LongTensor([output.expected_answer.value]), num_classes=len(output.response_token_ids)) for output in outputs], dim=0) for example, outputs in zip(self.examples, vqa_outputs_flat)], dim=0) # (# examples, # questions, # possible responses)
        # print("target_logits:", target_logits)
        
        # Flip target logits for mistake examples
        mistake_labels = torch.stack([torch.stack([torch.nn.functional.one_hot(torch.LongTensor([int(example.mistake)]), num_classes=2) for output in outputs], dim=0) for example, outputs in zip(self.examples, vqa_outputs_flat)], dim=0)
        # print("mistake_labels:", mistake_labels)
        target_logits = mistake_labels * (torch.ones_like(target_logits) - target_logits) + (torch.ones_like(target_logits) - mistake_labels) * target_logits
        # print("target_logits:", target_logits)
        target_logits = target_logits.squeeze(2)
        # print("target_logits:", target_logits)
        
        predicted_logits = torch.stack([torch.stack([torch.stack([output.logits[output.response_token_ids[response_type]] for response_type in VQAResponse], dim=0) for output in outputs], dim=0) for outputs in vqa_outputs_flat], dim=0) # (# examples, # questions, # possible responses)
        # print("predicted_logits:", predicted_logits)
        predicted_logits = torch.nn.functional.sigmoid(predicted_logits)
        # print("predicted_logits:", predicted_logits)
        
        assert target_logits.shape == predicted_logits.shape, f"Expected target_logits shape {target_logits.shape} to be the same as predicted_logits shape {predicted_logits.shape}."
        return (predicted_logits - target_logits).float()
        
        # NOTE: possible problem: the target_logits imply that in mistake cases, all expected answers should be violated. This may incentivize models to create duplicate questions that ultimately are incomplete.
        # Can use an argmax instead to incentivize at least one question having errors

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
HEURISTIC_TARGET_FRAMES_PROPORTION = int(config["mistake_detection_strategies"]["heuristic"]["target_frames_proportion"]) # Use last N% of frames for heuristic mistake detection

class HeuristicMistakeDetectionEvaluator(MistakeDetectionEvaluator):
    """Heuristic mistake detection evaluator which simply takes a majority vote over the last `mistake_frames_proportion` proportion of video frames."""
        
    def check_mistakes(self, detection_threshold: float=0.5) -> list[MistakeDetectionOutputs]:
        """
        Given `examples` and `vqa_outputs`, determine whether there is a mistake. Uses a simple heuristic: for the last `self.mistake_frames_proportion` percent of frames, judge whether there is a mistake based on VQA responses versus expected answers, then take the majority vote.
        
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
            
        # Heuristic: for last n% of frames (based on step duration), average likelihood of mistake then use a threshold to decide if there's a mistake
        # In the future, can prompt LLaMA again for this information?
        agg_preds = []
        for mistake_prob, example in zip(mistake_probs, self.examples):
            if len(mistake_prob) > 0:
                cutoff_time = get_cutoff_time_by_proportion(example, HEURISTIC_TARGET_FRAMES_PROPORTION)
                assert len(example.frame_times) == len(mistake_prob), "Compilation of mistake detections for example has a shape issue."
                mistake_prob_cut = [prob for prob, ft in zip(mistake_prob, example.frame_times) if ft >= cutoff_time]
                mean_mistake_prob_cut = np.mean(mistake_prob_cut)
                frame_times_cut = [ft for ft in example.frame_times if ft >= cutoff_time]
                                
                mistake_pred_final = True if mean_mistake_prob_cut > detection_threshold else False
                # mistake_pred_cut = [pred for pred, ft in zip(mistake_pred, example.frame_times) if ft >= cutoff_time]
                # mistake_pred_majority = Counter(mistake_pred_cut)
                # mistake_pred_majority, _ = mistake_pred_majority.most_common()[0]                
            else:
                mistake_prob_cut = [[]]
                mistake_pred_final = False

            pred_object = MistakeDetectionOutputs(
                example_id=example.example_id,
                frame_times=frame_times_cut,
                mistake_probs=mistake_prob_cut,
                detection_threshold=detection_threshold,
                final_mistake_prediction=mistake_pred_final
            )

            agg_preds.append(pred_object)            
            
        return agg_preds


class MistakeDetectionScorer:
    """Superclass to facilitate automated scoring of visual questions for action mistake detection in videos."""
    def __call__(self, 
                 examples: list[MistakeDetectionExample],
                 vqg_outputs: dict[int, VQGOutputs],
                 return_outputs: bool=False) -> torch.FloatTensor:
        """Score visual questions when posed on visual inputs to some multimodal language model.
        
        :param examples: List of MistakeDetectionExample objects to run through the model.
        :param vqg_outputs: Dictionary mapping unique annotated procedures (by int ID) to VQGOutputs, which include a consistent number of visual questions to verify success of the procedure.
        :return: FloatTensor of scores of shape (len(examples), # questions per example).
        """
        raise NotImplementedError("Need to use a subclass of MistakeDetectionScorer.")
        

class FrameVQAMistakeDetectionScorer(MistakeDetectionScorer):
    """Class that provides preference scores for visual questions to facilitate mistake detection on individual video frames."""
    def __init__(self, vlm_name):
        super().__init__()
        self.model_name = vlm_name
        self.processor = AutoProcessor.from_pretrained(vlm_name)
        self.vlm = AutoModelForVision2Seq.from_pretrained(vlm_name, 
                                                          cache_dir=DATA_CACHE_DIR, # TODO: add this back
                                                          load_in_8bit=True)
        self.vlm.language_model.generation_config.top_p = None
        self.vlm.language_model.generation_config.temperature = None
        self.vlm.language_model.generation_config.do_sample = False
        self.processor.tokenizer.padding_side = "left"
        
    def __call__(self, 
                 examples: list[MistakeDetectionExample],
                 vqg_outputs: dict[int, VQGOutputs],
                 return_vqa_outputs: bool=False,
                 batch_size: int=1) -> Union[torch.FloatTensor, list[VQAOutputs]]:
        """Score visual questions when posed on individual video frames to a VLM.
        
        :param examples: List of MistakeDetectionExample objects to run through the VLM.
        :param vqg_outputs: Dictionary mapping unique annotated procedures (by int ID) to VQGOutputs, which include a consistent number of visual questions to verify success of the procedure.
        :param return_vqa_outputs: Whether to return VQAOutputs from VQA inference instead of scores per example.
        :param batch_size: Batch size for VQA inference. Note that quantized LLaVA may return nan logits if greater than 1.
        :return: FloatTensor of scores of shape (len(examples), # questions per example).
        """
        # Verify every example has only a single frame, as expected for VQA mistake detection (frame selection occurs elsewhere)
        n_examples = len(examples)
        for example in examples:
            assert len(example.frames) == 1, "Before sending MistakeDetectionExample objects to FrameVQAMistakeDetectionScorer, please filter frames property to a single frame."
        
        # Verify all relevant VQGOutputs have the same number of questions
        example_vqg_outputs = [vqg_outputs[example.procedure_id] for example in examples]
        n_questions_per_frame = len(example_vqg_outputs[0].questions)
        for output in example_vqg_outputs:
            assert len(output.questions) == n_questions_per_frame, "All VQGOutputs should have the same number of generated questions."
               
        # Extract parallel frames, questions, answers, and mistake labels
        examples_questions_answers = [(example, question, answer) for example, output in zip(examples, example_vqg_outputs) for question, answer in zip(output.questions, output.answers)]
        examples_parallel = [example for example, _, _ in examples_questions_answers]
        questions = [question for _, question, _ in examples_questions_answers]
        answers = [answer for _, _, answer in examples_questions_answers]
        frames = [example.frames[0] for example in examples_parallel]
        assert len(examples_parallel) == len(questions) == len(answers) == len(frames) == n_examples * n_questions_per_frame
        mistake_labels = [example.mistake for example in examples_parallel]
             
        prompt_template = VQG2VQA_PROMPT_TEMPLATES[self.model_name]
        prompts = [prompt_template.format(question=question) for question in questions]
        
        response_tokens = {}
        for response_type in VQAResponse:
            response_tokens[response_type] = self.processor.tokenizer(response_type.name, add_special_tokens=False)['input_ids'][0]
            
        # Run VQA in batches
        logits = []
        with torch.no_grad():
            for i in tqdm(range(0, len(frames), batch_size), desc="running VQA"):
                # Prepare the batch
                batch_frames = frames[i:i+batch_size]
                batch_prompts = prompts[i:i+batch_size]            

                inputs = self.processor(text=batch_prompts, images=batch_frames, padding=True, return_tensors="pt")
                inputs = inputs.to(self.vlm.device)
                this_logits = self.vlm(**inputs).logits
                this_logits = this_logits[:, -1].detach().cpu()
                logits.append(this_logits)
            logits = torch.cat(logits, dim=0)
        
        # Gather up VQAOutputs (# examples, # questions per example)
        vqa_outputs = []
        for i, example in enumerate(examples): 
            this_vqa_outputs = []
            for j in range(n_questions_per_frame):
                parallel_idx = i * n_questions_per_frame + j
                this_vqa_outputs.append(
                    VQAOutputs(
                        example.example_id,
                        example.procedure_id,
                        frames[parallel_idx],
                        prompts[parallel_idx],
                        answers[parallel_idx],
                        response_tokens,
                        logits[parallel_idx]
                    )
                )
            vqa_outputs.append(this_vqa_outputs)
        
        if return_vqa_outputs:
            return vqa_outputs
        else:
            # In most cases, we just want scores for each generated question
            evaluator = MistakeDetectionEvaluator(examples, vqa_outputs)
            logits_errors = evaluator.get_logits_errors()
            return logits_errors
        
MISTAKE_DETECTION_STRATEGIES = {
    "heuristic": HeuristicMistakeDetectionEvaluator
}

def generate_det_curve(metrics: dict[float, dict[str, float]], save_path: str):
    """
    Generates and saves a PDF of a Detection Error Tradeoff (DET) curve for the metrics returned by `MistakeDetectionEvaluator.evaluate_mistake_detection()`. A DET curve plots false positive rate (x-axis) versus false negative rate (y-axis) for a space of detection thresholds, and indicates an "ideal" point to set the threshold in the bottom left corner.

    :param metrics: `metrics` object returned by `evaluate_mistake_detection()`.
    :param save_path: Path to save the PDF of the DET curve.
    """
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
    for example_outputs in vqa_outputs:
        example_id = example_outputs[0][0].example_id
        compiled_preds[example_id]["vqa"] = [[question_output.to_dict() for question_output in frame_outputs] for frame_outputs in example_outputs]
    for threshold in mistake_detection_preds:
        for pred in mistake_detection_preds[threshold]:
            example_id = pred.example_id
            if "mistake_detection" not in compiled_preds[example_id]:
                compiled_preds[example_id]["mistake_detection"] = {}
            compiled_preds[example_id]["mistake_detection"][threshold] = pred.to_dict()
    return compiled_preds
