from collections import Counter
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from PIL import Image
from pprint import pprint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Union
import yaml

from travel.constants import DATA_CACHE_DIR
from travel.data import MistakeDetectionExample, get_cutoff_time_by_proportion
from travel.model.vqa import VQAOutputs, VQAResponse, VQA_PROMPT_TEMPLATES
from travel.model.vqg import VQGOutputs

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
    
    def check_mistake(self) -> list[bool]:
        """
        Given `examples` and `vqa_outputs`, determine whether there is a mistake.
        
        :return: True if there is believed to be a mistake, else False.
        """
        raise NotImplementedError("Subclass should define the strategy to check for mistakes.")
    
    def get_mistake_detection_metrics(self) -> dict[str, float]:
        """
        Calculates mistake detection evaluation metrics for this class instance's `examples` and `vqa_outputs`.

        :return: `dict` mapping metric name to its float value.
        """

        labels = [example.mistake for example in self.examples]
        preds = self.check_mistakes()
        
        metrics = {}
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['precision'] = precision_score(labels, preds)
        metrics['recall'] = recall_score(labels, preds)
        metrics['f1'] = f1_score(labels, preds)
        
        return metrics
    
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

# TODO: consider replacing this strategy to use last N frames - but this has to be informed by the sampling frequency of frames, which is another important variable.
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
HEURISTIC_TARGET_FRAMES_PROPORTION = int(config["mistake_detection_strategies"]["heuristic"]["target_frames_proportion"]) # Use last N% of frames for heuristic mistake detection

class HeuristicMistakeDetectionEvaluator(MistakeDetectionEvaluator):
    """Heuristic mistake detection evaluator which simply takes a majority vote over the last `mistake_frames_proportion` proportion of video frames."""
        
    def check_mistakes(self) -> list[bool]:
        """
        Given `examples` and `vqa_outputs`, determine whether there is a mistake. Uses a simple heuristic: for the last `self.mistake_frames_proportion` percent of frames, judge whether there is a mistake based on VQA responses versus expected answers, then take the majority vote.
        
        :return: True if there is believed to be a mistake, else False.
        """
        # If any VQA answers don't match expected answers, there's a mistake (we can decide to make this more lenient later)
        mistake_predictions = []
        for example, outputs in zip(self.examples, self.vqa_outputs):
            this_mistake_predictions = []
            for frame_outputs in outputs:
                predicted_mistake = False
                # Check all questions for this frame to decide if there's a mistake
                for output in frame_outputs:
                    print(output.predicted_answer, output.expected_answer)
                    pprint(output)
                    if output.predicted_answer != output.expected_answer:
                        predicted_mistake = True
                this_mistake_predictions.append(predicted_mistake)
            mistake_predictions.append(this_mistake_predictions)
            
        # Heuristic: for last n% of frames (based on step duration), take majority prediction of mistake/success
        # In the future, can prompt LLaMA again for this information?
        agg_preds = []
        for mistake_pred, example in zip(mistake_predictions, self.examples):
            if len(mistake_pred) > 0:
                cutoff_time = get_cutoff_time_by_proportion(example, HEURISTIC_TARGET_FRAMES_PROPORTION)
                mistake_pred_cut = [pred for pred, ft in zip(mistake_pred, example.frame_times) if ft >= cutoff_time]
                if len(mistake_pred_cut) == 0:
                    mistake_pred_cut = [mistake_pred[-1]]
                                
                # last_n = max(int(len(mistake_pred) * self.mistake_frames_proportion), 1) # Round up to a minimum of 1 frame
                mistake_pred_cut = Counter(mistake_pred_cut)
                mistake_pred_cut, _ = mistake_pred_cut.most_common()[0]
            else:
                mistake_pred_cut = False
            agg_preds.append(mistake_pred_cut)            
            
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
        :param batch_size: Batch size for VQA inference. Note that LLaVA may return nan logits if greater than 1.
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
             
        prompt_template = VQA_PROMPT_TEMPLATES[self.model_name]
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