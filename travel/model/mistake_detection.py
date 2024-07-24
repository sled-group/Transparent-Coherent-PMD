from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint
from scipy.special import softmax
from scipy.stats import norm
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from typing import Union, Any, Optional
import yaml

from travel.model.metrics import mistake_detection_metrics, effectiveness
from travel.data.mistake_detection import MistakeDetectionDataset
from travel.data.utils import generate_float_series, time_based_exponential_moving_average
from travel.data.vqa import VQAOutputs, VQAResponse
from travel.model.nli import NLI_MODEL_PATH, NLI_BATCH_SIZE, NLI_RELEVANCE_DELTA, NLI_REPLACE_PROBS, NLI_RERUN_ON_RELEVANT_EVIDENCE, NLI_HYPOTHESIS_TEMPLATE, run_nli

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
EMA_TAU = float(config["mistake_detection_strategies"]["ema_tau"])
DETECTION_FRAMES_PROPORTION = float(config["mistake_detection_strategies"]["frames_proportion"]) # Use last N% of frames for frame-based mistake detection strategies
MISTAKE_DETECTION_THRESHOLDS = [round(threshold, 2) for threshold in generate_float_series(0.0, 1.0, 0.05)]

@dataclass
class MistakeDetectionOutputs:
    """Class to hold mistake detection outputs for individual frames. If using full video clips for VQA, just use a placeholder time."""
    example_id: str
    frame_times: list[float] # (# frames)
    mistake_probs: list[list[float]] # (# frames, # questions per frame)
    detection_threshold: float
    final_mistake_prediction: bool
    nli_mistake_probs: Optional[list[list[float]]] = None # (# frames, # questions per frame)
    nli_relevance_probs: Optional[list[list[float]]] = None # (# frames, # questions per frame)
    nli_final_mistake_probs: Optional[list[float]] = None # (# frames); this is the NLI probability for each frame based on all relevant evidences combined into one prompt
    nli_informativeness_probs: Optional[list[list[float]]] = None # (# frames, # questions per frame)

    def __post_init__(self):
        assert len(self.frame_times) == len(self.mistake_probs), "`MistakeDetectionOutputs` expects `frame_times` and `mistake_probs` to be the same length, i.e., the number of frames used to detect the mistake."

    def to_dict(self):
        """Helper method to create a JSON-serializable version of the class instance (excluding some information)."""
        return_dict = asdict(self)
        return_dict['frame_times'] = [float(round(ft, 3)) for ft in return_dict['frame_times']]
        return_dict['mistake_probs'] = [[float(round(v, 3)) for v in l] for l in return_dict['mistake_probs']]
        return_dict['detection_threshold'] = float(round(return_dict['detection_threshold'], 3))
        if return_dict['nli_mistake_probs'] is not None:
            return_dict['nli_mistake_probs'] = [[float(round(v, 3)) for v in l] for l in return_dict['nli_mistake_probs']]
        if return_dict['nli_relevance_probs'] is not None:
            return_dict['nli_relevance_probs'] = [[float(round(v, 3)) for v in l] for l in return_dict['nli_relevance_probs']]
        if return_dict['nli_final_mistake_probs'] is not None:
            return_dict['nli_final_mistake_probs'] = [float(round(v, 3)) for v in return_dict['nli_final_mistake_probs']]            
        return return_dict

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

            # Calculate effectiveness and get the mean using an exponential moving aveage for each example
            effectiveness_metrics = [
                [round(float(e), 3) for e in effectiveness(example.mistake, pred.mistake_probs)] for pred, example in zip(pred_objects, self.examples)
            ]
            mean_effectiveness = []
            for e, example in zip(effectiveness_metrics, self.examples):
                example.cutoff_to_last_frames(DETECTION_FRAMES_PROPORTION)
                mean_effectiveness.append(time_based_exponential_moving_average(e, example.frame_times, tau=EMA_TAU))
            mean_effectiveness = round(float(np.mean(mean_effectiveness), 3))

            # Add verifiability metrics
            metrics['verifiability'] = {
                "effectiveness": effectiveness_metrics,
                "effectiveness_mean": mean_effectiveness
            }

            # Save best metrics based on which threshold minimizes FPR and FNR
            if best_metrics is None or (this_metrics['false_positive_rate'] + this_metrics['false_negative_rate']) < (best_metrics['false_positive_rate'] + best_metrics['false_negative_rate']):
                best_metrics = this_metrics
                best_threshold = threshold

        metrics['best_metrics'] = best_metrics
        metrics['best_threshold'] = best_threshold
        return combined_preds, metrics

def aggregate_mistake_probs_over_frames(mistake_prob: list[list[float]], frame_times: list[float], confidence_threshold: Optional[float]=None, verbose: bool=False) -> float:
    mistake_prob = np.array(mistake_prob)
    if verbose:
        print("Mistake probs (input):")
        pprint(mistake_prob)

    assert len(frame_times) == len(mistake_prob), f"Compilation of mistake detections for example has a shape issue! Frame times length = {len(frame_times)}; Mistake probs length = {len(mistake_prob)}"
    assert len(mistake_prob.shape) == 2, "mistake_prob passed into aggregate_mistake_probs_over_frames should only have two dimensions: (frames, questions)"

    mean_mistake_prob = mistake_prob
    if confidence_threshold is not None:
        # (this isn't in use, doesn't really help performance much)
        # Omit any responses that aren't confident enough
        mean_mistake_prob = [[p for p in frame_p if abs(p - 0.50) / 0.50 > confidence_threshold] for frame_p in mean_mistake_prob]
        if verbose:
            print("Mistake probs (after confidence thresholding):")
            pprint(mean_mistake_prob)

    # Get maximum probability of a mistake for each frame (since we only need one question to indicate a mistake);
    # if there are no answers that are confident enough, omit the frame
    keep_times = [t for frame_p, t in zip(mean_mistake_prob, frame_times) if len(frame_p) > 0]
    mean_mistake_prob = [max(frame_p) for frame_p in mean_mistake_prob if len(frame_p) > 0]
    frame_times = keep_times
    
    # For each frame, select the most confident answer to represent the 
    # select_frame = [np.argmax([abs(p - 0.50) / 0.50 for p in frame_p]) for frame_p in mean_mistake_prob]
    # mean_mistake_prob = [p[sf] for p, sf in zip(mean_mistake_prob, select_frame)]

    if verbose:
        print("Mistake probs (after max):")
        pprint(mean_mistake_prob)

    # Normalize each frame probability by relative time in video clip - if only one frame (e.g., in ego4d), this normalization coefficient would be 1

    # NOTE: due to a processing bug in some versions of Ego4D, there are rare cases with negative frame times
    if len(mean_mistake_prob) > 0:
        frame_times = [max(t, 0.0) for t in frame_times]
        if len(frame_times) > 1 and max(frame_times) - min(frame_times) > 0.0:
            mean_mistake_prob = time_based_exponential_moving_average(mean_mistake_prob, frame_times, tau=EMA_TAU) # set tau to be equal to the avg. sampling interval (2 seconds for both ego4d and captaincook4d)
            if verbose:
                print("Mistake probs (after smoothing):")
                print(mean_mistake_prob)

            # # Select last frame because it's most recent
            # mean_mistake_prob = mean_mistake_prob[-1]

            # Select the frame that has the maximum confidence * recency
            confidence = [abs(p - 0.50) / 0.50 for p in mean_mistake_prob] # Scores confidence based on distance from 50/50 probability
            time_into_clip = [(t - min(frame_times)) / (max(frame_times) - min(frame_times)) for t in frame_times]
            select_frame = int(np.argmax([p * t for p, t in zip(confidence, time_into_clip)]))
            mean_mistake_prob = mean_mistake_prob[select_frame]
        else:
            # Just do normal average if we have corrupted time info (or only one frame)
            mean_mistake_prob = np.mean(mean_mistake_prob)
    else:
        # Previous operations removed all predictions, so just say there's not a mistake
        mean_mistake_prob = 0.0 # (use a number that represents distribution of classes?)

    # mean_mistake_prob = exponential_moving_average(mean_mistake_prob, alpha=0.1)
    
    # mean_mistake_prob = [(p * ((t - min(frame_times)) / (max(frame_times) - min(frame_times))) if len(frame_times) > 1 and max(frame_times) - min(frame_times) > 0.0 else p) for p, t in zip(mean_mistake_prob, frame_times)]  
    # mistake_prob_weights = [(((t - min(frame_times)) / (max(frame_times) - min(frame_times))) if len(frame_times) > 1 and max(frame_times) - min(frame_times) > 0.0 else 1.0) for p, t in zip(mean_mistake_prob, frame_times)]
    # mistake_prob_weights = softmax(mistake_prob_weights)
    # if verbose:
    #     print("Mistake prob time weights:")
    #     pprint(mistake_prob_weights)
    # mean_mistake_prob = [p * w for p, w in zip(mean_mistake_prob, mistake_prob_weights)]

    # if verbose:
    #     print("Mistake probs (after time-weighting)")
    #     pprint(mean_mistake_prob)

    # # Get mean mistake probability over all frames
    # mean_mistake_prob = np.sum(mean_mistake_prob) 

    if verbose:
        print("Mistake prob (final moving average for best time)")
        print(mean_mistake_prob)

    return mean_mistake_prob

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
                    if output.target_object_counts is None or len(output.target_object_counts) == 0 or not max(output.target_object_counts.values()) == 0: # Check if all target objects of the question are present in this frame - if not, don't include in prediction
                        mistake_answer = VQAResponse(1-int(output.expected_answer.value))
                        frame_mistake_probs.append(output.answer_probs[mistake_answer])
                    else:
                        # Visual filter didn't see any target objects, so assume there's a mistake
                        frame_mistake_probs.append(1.0)
                example_mistake_probs.append(frame_mistake_probs)
            mistake_probs.append(example_mistake_probs)
            
        # Heuristic: average likelihood of mistake then use a threshold to decide if there's a mistake
        agg_preds = []
        for mistake_prob, example in tqdm(zip(mistake_probs, self.examples), desc=f"evaluating mistake detection at threshold {detection_threshold}", total=len(self.examples)):
            if len(mistake_prob) > 0 and len(mistake_prob[0]) > 0:
                example.cutoff_to_last_frames(DETECTION_FRAMES_PROPORTION) # Call this again since the example got reloaded from cache                
                mean_mistake_prob = aggregate_mistake_probs_over_frames(mistake_prob, example.frame_times)
                mistake_pred_final = True if mean_mistake_prob >= detection_threshold else False
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

class NLIMistakeDetectionEvaluator(MistakeDetectionEvaluator):
    """Mistake detection evaluator which uses a pre-trained natural language inference (NLI) model to judge whether answers to questions indicate mistakes."""
    def __post_init__(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
        self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)
        self.relevance_probs = None
        self.mistake_probs = None
        self.final_mistake_probs = None
        super().__post_init__()

    def run_nli(self, procedure_descriptions: list[str], premises: list[str], premises_negated: Optional[list[str]]=None) -> tuple[list[str], list[float], list[float]]:
        # TODO: use run_nli method here
        if (premises_negated and self.mistake_probs is None or self.relevance_probs is None) or (not premises_negated and self.final_mistake_probs is None):
            with torch.no_grad():
                all_mistake_probs = torch.zeros((0, 1)).float()
                if premises_negated:
                    all_relevance = torch.zeros((0, 1)).float()

                for i in tqdm(range(0, len(procedure_descriptions), NLI_BATCH_SIZE), desc=f"running NLI ({str(self.nli_model.device)})"):
                    # Prepare the batch
                    batch_procedure_descriptions = procedure_descriptions[i:i+NLI_BATCH_SIZE]
                    batch_premises = premises[i:i+NLI_BATCH_SIZE]
                    
                    hypothesis = [f'The procedure "{procedure}" has been successfully completed.' for procedure in batch_procedure_descriptions]

                    # Run premise through NLI model
                    x = self.nli_tokenizer.batch_encode_plus(list(zip(batch_premises, hypothesis)),
                                                            return_tensors='pt', 
                                                            padding="longest",
                                                            truncation='only_first')
                    logits = self.nli_model(**x.to(self.nli_model.device))[0]
                    logits = logits.cpu()
                    logits = logits[:,[0,2]] # Take logits for contradiction and entailment only
                    probs = logits.softmax(dim=1)

                    # Grab contradiction probabilities weighted by relevance and add to all_mistake_probs
                    all_mistake_probs = torch.cat([all_mistake_probs, probs[:, 0].unsqueeze(1)], dim=0)

                    # Run negated premise through NLI model
                    if premises_negated:
                        batch_premises_negated = premises_negated[i:i+NLI_BATCH_SIZE]
                        # pprint(hypothesis[:10])
                        # pprint(batch_premises[:10])
                        # pprint(batch_premises_negated[:10])
                        # print("=============")
                        x = self.nli_tokenizer.batch_encode_plus(list(zip(batch_premises_negated, hypothesis)), 
                                                                return_tensors='pt',
                                                                padding="longest",
                                                                truncation='only_first')
                        logits_negated = self.nli_model(**x.to(self.nli_model.device))[0]
                        logits_negated = logits_negated.cpu()
                        logits_negated = logits_negated[:,[0,2]] # Take logits for contradiction and entailment only
                        probs_negated = logits_negated.softmax(dim=1)
                    
                        # If probability of contradiction doesn't change enough between answer and negated answer, 
                        # then assume the question is irrelevant and assign 0 probability of contradiction 
                        # (can filter these out from prediction later)
                        relevance = probs[:, 0] - probs_negated[:, 0]
                        all_relevance = torch.cat([all_relevance, relevance.unsqueeze(1)], dim=0)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Save any probs for later runs with different thresholds
            if premises_negated:
                self.relevance_probs = list(all_relevance.squeeze(1).numpy())
                self.mistake_probs = list(all_mistake_probs.squeeze(1).numpy())
            else:
                self.final_mistake_probs = list(all_mistake_probs.squeeze(1).numpy())

        if premises_negated:
            # print("==========================================")
            # pprint(self.mistake_probs[:10])
            # print('\n\n')
            # pprint(self.relevance_probs[:10])
            return self.mistake_probs, self.relevance_probs
        else:
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # pprint(self.final_mistake_probs[:10])
            return self.final_mistake_probs

    def check_mistakes(self, detection_threshold: float=0.5) -> list[MistakeDetectionOutputs]:
        """
        Given `examples` and `vqa_outputs` class members, determine whether there is a mistake. Combines an NLI model's judgements with mistake probabilities according to VLM answers to generated questions.
        
        :param detection_threshold: Confidence threshold for evaluator to predict there's a mistake, typically based on an LM's logits.
        :return: True if there is believed to be a mistake, else False.
        """

        # If any VQA answers don't match expected answers, there's a mistake (we can decide to make this more lenient later)
        all_procedure_descriptions = []
        all_questions = []
        all_answers = []
        for example, outputs in zip(self.examples, self.vqa_outputs):
            for frame_outputs in outputs:
                # Check all questions for this frame to decide if there's a mistake
                for output in frame_outputs:
                    all_procedure_descriptions.append(example.procedure_description)
                    assert output.question is not None, "NLIMistakeDetectionEvaluator requires questions to be provided in VQAOutputs!"
                    all_questions.append(output.question)
                    all_answers.append(output.predicted_answer)
            
        negated_answers = [VQAResponse(1-answer.value) for answer in all_answers]
        premises = [f"{question} {answer.name}" for question, answer in zip(all_questions, all_answers)]
        premises_negated = [f"{question} {answer.name}" for question, answer in zip(all_questions, negated_answers)]

        nli_mistake_probs, nli_relevance = self.run_nli(all_procedure_descriptions, premises, premises_negated)
        del all_procedure_descriptions
        del all_questions
        del all_answers

        # Identify relevant evidence from each frame
        if NLI_RERUN_ON_RELEVANT_EVIDENCE:
            parallel_idx = 0
            frame_idx = 0
            all_frame_premises = []
            all_frame_premises_idxs = []
            all_frame_procedure_descriptions = []
            for example, outputs in zip(self.examples, self.vqa_outputs):
                for frame_outputs in outputs:
                    frame_relevant_premises = []

                    # Determine if there's a mistake for each frame
                    for question_output in frame_outputs:
                        if question_output.target_object_counts is None or len(question_output.target_object_counts.keys()) == 0 or not max(question_output.target_object_counts.values()) == 0: # Check if all target objects of the question are present in this frame - if not, don't include in prediction
                            # Incorporate NLI model feedback
                            if abs(nli_relevance[parallel_idx]) >= NLI_RELEVANCE_DELTA:
                                # NLI model found this question relevant
                                frame_relevant_premises.append(f"{question_output.question} {question_output.predicted_answer}")

                        parallel_idx += 1

                    all_frame_premises.append("/n".join(frame_relevant_premises))
                    all_frame_premises_idxs.append(frame_idx)
                    all_frame_procedure_descriptions.append(example.procedure_description)
                    frame_idx += 1

            nli_final_mistake_probs_by_frame = self.run_nli(all_frame_procedure_descriptions, all_frame_premises)

        # Incorporate NLI mistake probs into mistake probs
        parallel_idx = 0
        frame_idx = 0
        mistake_probs = []
        compiled_nli_mistake_probs = []
        compiled_nli_relevance_probs = []
        compiled_nli_final_mistake_probs = []
        for example, outputs in zip(self.examples, self.vqa_outputs):
            example_mistake_probs = []
            example_nli_mistake_probs = []
            example_nli_relevance_probs = []
            example_nli_final_mistake_probs = []
            for frame_outputs in outputs:
                frame_mistake_probs = []
                frame_nli_mistake_probs = []
                frame_nli_relevance_probs = []
                frame_nli_final_mistake_probs = []

                # Determine if there's a mistake for each frame
                for question_output in frame_outputs:
                    # Incorporate NLI model feedback
                    if abs(nli_relevance[parallel_idx]) < NLI_RELEVANCE_DELTA:
                        # NLI model found this question irrelevant, so 0 mistake probability
                        frame_mistake_probs.append(0.0)
                    else:
                        if question_output.target_object_counts is None or len(question_output.target_object_counts) == 0 or not max(question_output.target_object_counts.values()) == 0: # Check if all target objects of the question are present in this frame - if not, don't include in prediction
                            # Reweight mistake prob from VLM by NLI model (which accounts for bad/irrelevant generated questions)
                            mistake_answer = VQAResponse(1-int(question_output.expected_answer.value))
                            mistake_prob = question_output.answer_probs[mistake_answer]
                            if not NLI_RERUN_ON_RELEVANT_EVIDENCE:
                                # Configure whether to only use NLI probs for final mistake detection, or otherwise multiply probabilities from VQA and NLI
                                if NLI_REPLACE_PROBS:
                                    frame_mistake_probs.append(nli_mistake_probs[parallel_idx])
                                else:
                                    frame_mistake_probs.append(mistake_prob * nli_mistake_probs[parallel_idx])
                            else:
                                # If going to use a combined NLI probability for all relevant evidence from this frame, just save the VQA probability directly
                                frame_mistake_probs.append(mistake_prob)
                        else:
                            # Visual filter didn't see any target objects, so assume there's a mistake;
                            # use relevance of question here
                            frame_mistake_probs.append(abs(nli_relevance[parallel_idx]))

                    frame_nli_mistake_probs.append(nli_mistake_probs[parallel_idx])
                    frame_nli_relevance_probs.append(nli_relevance[parallel_idx])

                    parallel_idx += 1

                if NLI_RERUN_ON_RELEVANT_EVIDENCE:
                    # If we're using the NLI probability for all relevant evidence combined into one premise, then condense the frame probabilities accordingly
                    if frame_idx in all_frame_premises_idxs:
                        # If there was any relevant evidence for this frame, find it and incorporate it
                        mistake_prob = nli_final_mistake_probs_by_frame[all_frame_premises_idxs.index(frame_idx)]
                        frame_nli_final_mistake_probs.append(mistake_prob)
                        if not NLI_REPLACE_PROBS:
                            # Factor in max mistake probability for this frame from VQA step
                            mistake_prob *= max(frame_mistake_probs)
                        example_mistake_probs.append([mistake_prob])
                    else:
                        # There's no relevant evidence for this frame, so assume there's no mistake
                        example_mistake_probs.append([0.0])

                else:
                    example_mistake_probs.append(frame_mistake_probs)
                example_nli_mistake_probs.append(frame_nli_mistake_probs)
                example_nli_relevance_probs.append(frame_nli_relevance_probs)
                if len(frame_nli_final_mistake_probs) > 0:
                    example_nli_final_mistake_probs.append(frame_nli_final_mistake_probs[0])

                frame_idx += 1

            mistake_probs.append(example_mistake_probs)
            compiled_nli_mistake_probs.append(example_nli_mistake_probs)
            compiled_nli_relevance_probs.append(example_nli_relevance_probs)
            compiled_nli_final_mistake_probs.append(example_nli_final_mistake_probs)

        # From here, follow HeuristicMistakeDetectionEvaluator: just average reweighted likelihood of mistake then use a threshold to decide if there's a mistake
        agg_preds = []
        for mistake_prob, nli_mistake_prob, nli_relevance_prob, nli_final_mistake_prob, example in tqdm(zip(mistake_probs, compiled_nli_mistake_probs, compiled_nli_relevance_probs, compiled_nli_final_mistake_probs, self.examples), desc=f"evaluating mistake detection at threshold {detection_threshold}", total=len(self.examples)):
            if len(mistake_prob) > 0:
                example.cutoff_to_last_frames(DETECTION_FRAMES_PROPORTION) # Call this again since the example got reloaded from cache
                mean_mistake_prob = aggregate_mistake_probs_over_frames(mistake_prob, example.frame_times)                                
                mistake_pred_final = True if mean_mistake_prob >= detection_threshold else False
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
                final_mistake_prediction=mistake_pred_final,
                nli_mistake_probs=nli_mistake_prob,
                nli_relevance_probs=nli_relevance_prob,
                nli_final_mistake_probs=nli_final_mistake_prob
            )

            agg_preds.append(pred_object)            
            
        return agg_preds

# class ComprehensiveMistakeDetectionEvaluator(MistakeDetectionEvaluator):
#     """Mistake detection evaluator which runs a full evaluation of accuracy, consistency, and verifiability for generated questions and predicted answers, both using heuristic and NLI-based strategies to aggregate probabilities and make the final mistake detection decision."""
#     def __post_init__(self):
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             llm_int8_threshold=6.0,
#             llm_int8_has_fp16_weight=False,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#         )
#         self.nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
#         self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)

#         self.nli_mistake_probs = None
#         self.relevance_probs = None
#         self.informativeness_probs = None
#         self.compiled_probs = None
#         super().__post_init__()

#     def calculate_nli_probs(self, procedure_descriptions: list[str], premises_predicted: list[str], premises_negated: list[str], premises_expected: list[str]) -> tuple[list[str], list[float], list[float]]:
#         if self.nli_mistake_probs is None or self.relevance_probs is None or self.informativeness_probs:
#             with torch.no_grad():
#                 all_nli_mistake_probs = torch.zeros((0, 1)).float()
#                 all_relevance = torch.zeros((0, 1)).float()
#                 all_informativeness = torch.zeros((0, 1)).float()

#                 for i in tqdm(range(0, len(procedure_descriptions), NLI_BATCH_SIZE), desc=f"running NLI ({str(self.nli_model.device)})"):
#                     # Prepare the batch
#                     batch_procedure_descriptions = procedure_descriptions[i:i+NLI_BATCH_SIZE]
#                     batch_premises_predicted = premises_predicted[i:i+NLI_BATCH_SIZE]
                    
#                     hypothesis = [NLI_HYPOTHESIS_TEMPLATE.format(procedure=procedure) for procedure in batch_procedure_descriptions]

#                     # Get mistake probs for predicted answers
#                     nli_mistake_probs = run_nli(self.nli_model, self.nli_tokenizer, list(zip(batch_premises_predicted, hypothesis)))[:, 0].unsqueeze(1)
#                     all_nli_mistake_probs = torch.cat([all_nli_mistake_probs, nli_mistake_probs], dim=0)

#                     # Run negated premise through NLI model to get relevance (how much entailment probability changes based on answer to question)
#                     batch_premises_negated = premises_negated[i:i+NLI_BATCH_SIZE]
#                     probs_negated = run_nli(self.nli_model, self.nli_tokenizer, list(zip(batch_premises_negated, hypothesis)))[:, 0].unsqueeze(1)
#                     relevance = torch.abs(nli_mistake_probs - probs_negated)
#                     all_relevance = torch.cat([all_relevance, relevance], dim=0)

#                     # Get informativeness by looking how much expected answer indicates a mistake
#                     batch_premises_expected = premises_expected[i:i+NLI_BATCH_SIZE]
#                     informativeness = run_nli(self.nli_model, self.nli_tokenizer, list(zip(batch_premises_expected, hypothesis)))[:, 1].unsqueeze(1)
#                     all_informativeness = torch.cat([all_informativeness, informativeness])

#                     if torch.cuda.is_available():
#                         torch.cuda.empty_cache()

#                 # Save any probs for later runs with different thresholds
#                 self.nli_mistake_probs = list(all_nli_mistake_probs.squeeze(1).numpy())
#                 self.relevance_probs = list(all_relevance.squeeze(1).numpy())
#                 self.informativeness_probs = all_informativeness

#         return self.mistake_probs, self.relevance_probs, self.informativeness_probs

#     def check_mistakes(self, detection_threshold: float=0.5) -> list[MistakeDetectionOutputs]:
#         """
#         Given `examples` and `vqa_outputs` class members, determine whether there is a mistake. Combines an NLI model's judgements with mistake probabilities according to VLM answers to generated questions.
        
#         :param detection_threshold: Confidence threshold for evaluator to predict there's a mistake, typically based on an LM's logits.
#         :return: True if there is believed to be a mistake, else False.
#         """

#         # If any VQA answers don't match expected answers, there's a mistake (we can decide to make this more lenient later)
#         all_procedure_descriptions = []
#         all_questions = []
#         all_answers = []
#         for example, outputs in zip(self.examples, self.vqa_outputs):
#             for frame_outputs in outputs:
#                 # Check all questions for this frame to decide if there's a mistake
#                 for output in frame_outputs:
#                     all_procedure_descriptions.append(example.procedure_description)
#                     assert output.question is not None, "NLIMistakeDetectionEvaluator requires questions to be provided in VQAOutputs!"
#                     all_questions.append(output.question)
#                     all_answers.append(output.predicted_answer)
            
#         negated_answers = [VQAResponse(1-answer.value) for answer in all_answers]
#         premises = [f"{question} {answer.name}" for question, answer in zip(all_questions, all_answers)]
#         premises_negated = [f"{question} {answer.name}" for question, answer in zip(all_questions, negated_answers)]

#         nli_mistake_probs, nli_relevance, nli_informativeness = self.calculate_nli_probs(all_procedure_descriptions, premises, premises_negated)
#         del all_procedure_descriptions
#         del all_questions
#         del all_answers

#         # Calculate mistake probs
#         parallel_idx = 0
#         frame_idx = 0
        
#         if self.compiled_probs is None:
#             compiled_heuristic_mistake_probs = []
#             compiled_nli_mistake_probs = []
#             compiled_nli_relevance_probs = []
#             compiled_nli_informativeness_probs = []
#             compiled_mistake_probs = []
            
#             for example, outputs in zip(self.examples, self.vqa_outputs):
#                 example_heuristic_mistake_probs = []
#                 example_nli_mistake_probs = []
#                 example_nli_relevance_probs = []
#                 example_nli_informativeness_probs = []
#                 example_mistake_probs = []

#                 for frame_outputs in outputs:
#                     frame_heuristic_mistake_probs = []
#                     frame_nli_mistake_probs = []
#                     frame_nli_relevance_probs = []
#                     frame_nli_informativeness_probs = []
#                     frame_mistake_probs = []

#                     # Determine if there's a mistake for each frame
#                     for question_output in frame_outputs:
                        
#                         frame_heuristic_mistake_probs.append(mistake_prob)
#                         frame_nli_mistake_probs.append(nli_mistake_probs[parallel_idx])
#                         frame_nli_relevance_probs.append(nli_relevance[parallel_idx])
#                         frame_nli_informativeness_probs.append(nli_informativeness[parallel_idx])

#                         # Get mistake prob based on answers from VLM
#                         mistake_answer = VQAResponse(1-int(question_output.expected_answer.value))
#                         mistake_prob = question_output.answer_probs[mistake_answer]

#                         # Get final mistake detection probabilty by combining all information
#                         if abs(nli_relevance[parallel_idx]) < NLI_RELEVANCE_DELTA:
#                             # NLI model found this question irrelevant, so 0 mistake probability
#                             frame_mistake_probs.append(0.0)
#                         else:                     
#                             if question_output.target_object_counts is None or len(question_output.target_object_counts) == 0 or not max(question_output.target_object_counts.values()) == 0: # Check if all target objects of the question are present in this frame - if not, don't include in prediction
#                                 # Configure whether to only use NLI probs for final mistake detection, or otherwise multiply probabilities from VQA and NLI
#                                 frame_mistake_probs.append(mistake_prob * nli_mistake_probs[parallel_idx])
#                             else:
#                                 # Visual filter didn't see any target objects, so assume there's a mistake;
#                                 # use relevance of question here
#                                 frame_mistake_probs.append(abs(nli_relevance[parallel_idx]))

#                         parallel_idx += 1

#                     example_heuristic_mistake_probs.append(frame_heuristic_mistake_probs)
#                     example_nli_mistake_probs.append(frame_nli_mistake_probs)
#                     example_nli_relevance_probs.append(frame_nli_relevance_probs)
#                     example_nli_informativeness_probs.append(frame_nli_informativeness_probs)
#                     example_mistake_probs.append(frame_mistake_probs)

#                     frame_idx += 1

#                 compiled_heuristic_mistake_probs.append(example_heuristic_mistake_probs)
#                 compiled_nli_mistake_probs.append(example_nli_mistake_probs)
#                 compiled_nli_relevance_probs.append(example_nli_relevance_probs)
#                 compiled_nli_informativeness_probs.append(example_nli_informativeness_probs)
#                 compiled_mistake_probs.append(example_mistake_probs)

#                 self.compiled_probs = (
#                     compiled_heuristic_mistake_probs, 
#                     compiled_nli_mistake_probs, 
#                     compiled_nli_relevance_probs, 
#                     compiled_nli_informativeness_probs, 
#                     compiled_mistake_probs
#                 )
#             else:
#                 # Load pre-generated probs
#                 compiled_heuristic_mistake_probs, compiled_nli_mistake_probs, compiled_nli_relevance_probs, compiled_nli_informativeness_probs, compiled_mistake_probs = self.compiled_probs

#         # From here, follow HeuristicMistakeDetectionEvaluator: just average reweighted likelihood of mistake then use a threshold to decide if there's a mistake
#         agg_preds = []
#         for direct_mistake_prob, nli_mistake_prob, nli_relevance_prob, nli_informativeness_prob, mistake_prob, example in tqdm(zip(compiled_heuristic_mistake_probs, compiled_nli_mistake_probs, compiled_nli_relevance_probs, compiled_nli_informativeness_probs, compiled_mistake_probs, self.examples), desc=f"evaluating mistake detection at threshold {detection_threshold}", total=len(self.examples)):
#             if len(mistake_prob) > 0:
#                 example.cutoff_to_last_frames(DETECTION_FRAMES_PROPORTION) # Call this again since the example got reloaded from cache
#                 mean_mistake_prob = aggregate_mistake_probs_over_frames(mistake_prob, example.frame_times)                                
#                 mistake_pred_final = True if mean_mistake_prob >= detection_threshold else False
#             else:
#                 # If there are no frames to predict over, this is probably because some filter was applied to remove images that don't have a target object;
#                 # in this case, the target object is likely not present at all in the video, suggesting an incorrect object is used instead
#                 mistake_prob = [[]]
#                 mistake_pred_final = True

#             pred_object = MistakeDetectionOutputs(
#                 example_id=example.example_id,
#                 frame_times=example.frame_times,
#                 mistake_probs=mistake_prob,
#                 detection_threshold=detection_threshold,
#                 final_mistake_prediction=mistake_pred_final,
#                 nli_mistake_probs=nli_mistake_prob,
#                 nli_relevance_probs=nli_relevance_prob,
#             )

#             agg_preds.append(pred_object)            
            
#         return agg_preds
        
MISTAKE_DETECTION_STRATEGIES = {
    "heuristic": HeuristicMistakeDetectionEvaluator,
    "nli": NLIMistakeDetectionEvaluator,
    # "comprehensive": ComprehensiveMistakeDetectionEvaluator,
}


def compile_mistake_detection_preds(dataset: MistakeDetectionDataset,
                                    vqa_outputs: list[list[list[VQAOutputs]]],
                                    mistake_detection_preds: dict[float, list[MistakeDetectionOutputs]],
                                    image_base_path: Optional[str]=None) -> dict[str, dict[str, Any]]:
    """
    Helper function to compile mistake detection examples with model predictions from VQA and mistake detection.

    :param dataset: Mistake detection dataset used for evaluation.
    :param vqa_outputs: Ragged list of VQA outputs; shape should correspond to (# examples, # frames, # questions per frame)
    """
    compiled_preds = {example.example_id: {"example": example.to_dict()} for example in dataset}
    assert len(dataset) == len(vqa_outputs), "Expected same number of dataset examples and VQAOutputs lists."
    for example_outputs, example in zip(vqa_outputs, dataset):
        example_id = example.example_id
        compiled_preds[example_id]["vqa"] = [[question_output.to_dict(image_base_path=image_base_path) for question_output in frame_outputs] for frame_outputs in example_outputs]
    for threshold in mistake_detection_preds:
        for pred in mistake_detection_preds[threshold]:
            example_id = pred.example_id
            if "mistake_detection" not in compiled_preds[example_id]:
                compiled_preds[example_id]["mistake_detection"] = {}
            compiled_preds[example_id]["mistake_detection"][threshold] = pred.to_dict()
    return compiled_preds
