import os
from pympler.tracker import SummaryTracker
import spacy
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from typing import Optional, Union

from travel.constants import DATA_CACHE_DIR
from travel.data.vqa import VQAOutputs, VQAResponse, VQG2VQA_PROMPT_TEMPLATES, get_vqa_response_token_ids
from travel.data.vqg_learning import FrameVQAMistakeDetectionExample, VQGTrainingExample
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.model.grounding import VisualFilterTypes, SpatialVisualFilter, ContrastiveRegionFilter
from travel.model.vqa import run_vqa

# NOTE: we may need to employ multiple scorers (for several VLM types)
# NOTE: we may need to implement scorers for different types of inputs, e.g., video
class FrameVQAMistakeDetectionScorer:
    """Class that provides preference scores for visual questions to facilitate mistake detection on individual video frames."""
    def __init__(self, 
                 vlm_name: str,
                 visual_filter_type: Optional[VisualFilterTypes]=None,
                 visual_filter_strength: float=1.0,
                 vlm_device: Optional[int]=None,
                 visual_filter_device: Optional[int]=None):
        """
        Initializes FrameVQAMistakeDetectionScorer.

        :param vlm_name: Name of or path to Hugging Face VLM.
        :param visual_filter_type: Class for visual filter, e.g., spatial filter.
        :param vlm_device: Index of GPU device to put VLM on. If not specified, will use the first available GPU by default.
        :param visual_filter_device: If using a visual filter, index of GPU device to put it on.
        """
        super().__init__()
        self.model_name = vlm_name
        self.processor = AutoProcessor.from_pretrained(vlm_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if vlm_device is not None:
            torch.cuda.set_device(f"cuda:{vlm_device}")
        self.vlm = AutoModelForVision2Seq.from_pretrained(vlm_name, 
                                                          quantization_config=bnb_config)
        self.vlm.language_model.generation_config.top_p = None
        self.vlm.language_model.generation_config.temperature = None
        self.vlm.language_model.generation_config.do_sample = False
        self.processor.tokenizer.padding_side = "left"

        if visual_filter_type == VisualFilterTypes.Spatial:
            self.visual_filter = SpatialVisualFilter(rephrase_questions=True, mask_strength=visual_filter_strength, device=visual_filter_device)
        elif visual_filter_type == VisualFilterTypes.Spatial_NoRephrase:
            self.visual_filter = SpatialVisualFilter(rephrase_questions=False, mask_strength=visual_filter_strength, device=visual_filter_device)
        elif visual_filter_type == VisualFilterTypes.Spatial_Blur:
            self.visual_filter = SpatialVisualFilter(rephrase_questions=False, mask_strength=visual_filter_strength, device=visual_filter_device)
        elif visual_filter_type == VisualFilterTypes.Contrastive_Region:
            self.visual_filter = ContrastiveRegionFilter(mask_strength=visual_filter_strength, device=visual_filter_device)
        else:
            self.visual_filter = None
        self.visual_filter_type = visual_filter_type
        self.nlp = spacy.load('en_core_web_lg')
        
    def get_scores(self,
                   mistake_labels: list[bool],
                   vqa_outputs: list[list[VQAOutputs]],
                   ) -> list[float]:
        """
        Scores whether VQA outputs successfully indicate mistakes/lack of mistakes. This is separate from mistake detection policies, which integrate information across multiple frames and may employ confidence thresholds, etc.

        :param mistake_labels: List of bools indicating whether there is a mistake in some mistake detection example.
        :param vqa_outputs: List of lists of VQA outputs, one for each question set which will be scored.
        :return: List of scores between 0 and 1 (with 1 being best) which indicate how preferable the question set is for the VLM.
        """
        assert len(mistake_labels) == len(vqa_outputs), "FrameVQAMistakeDetectionScorer.get_scores expected same number of mistake labels and VQA outputs!"
        scores = []
        answer_probs = []
        for mistake, example_vqa_outputs in zip(mistake_labels, vqa_outputs):
            answer_probs.append([(output.answer_probs[VQAResponse.No], output.answer_probs[VQAResponse.Yes]) for output in example_vqa_outputs])
            
            # VLM probabilities that there's a mistake for each question
            mistake_probs = [1.0 - output.answer_probs[output.expected_answer] for output in example_vqa_outputs]

            # It only takes one question to indicate a mistake, so use the maximum mistake probability to score this questions set
            max_mistake_prob = max(mistake_probs)

            # For mistake examples, we want the max mistake probability to be close to 1.0
            if mistake:
                scores.append(max_mistake_prob)

            # For non-mistake examples, we want the max mistake probability to be 0.0
            else:
                scores.append(1.0 - max_mistake_prob)

        return answer_probs, scores

    def __call__(self, 
                 examples: list[FrameVQAMistakeDetectionExample],
                 return_vqa_outputs: bool=False,
                 batch_size: int=1,
                 cache_path: Optional[str]=None,
                 memory_tracker: Optional[SummaryTracker]=None) -> Union[torch.FloatTensor, list[VQAOutputs]]:
        """
        Score visual questions when posed on individual video frames to a VLM.
        
        :param examples: List of FrameVQAMistakeDetectionExample objects to run through the VLM, each of which include a single frame, question, and expected answer for the frame.
        :param return_vqa_outputs: Whether to return VQAOutputs from VQA inference instead of scores per example.
        :param batch_size: Batch size for VQA inference.
        :param cache_path: Path to save a .pt file for logits generated so far.
        :return: FloatTensor of scores of shape (len(examples), # questions per example) and a list of VQAOutputs.
        """
        # Extract parallel frames, questions, answers, and mistake labels
        questions = [question for example in examples for question_set in example.candidate_question_sets for question in question_set.questions]
        answers = [answer for example in examples for question_set in example.candidate_question_sets for answer in question_set.answers]
        frames = [example.frame for example in examples for question_set in example.candidate_question_sets for _ in question_set.questions]
        assert len(questions) == len(answers) == len(frames), "Need same number of questions, answers, and frames to score questions on frame-based VQA!"
        mistake_labels = [example.mistake for example in examples for _ in example.candidate_question_sets] # One scoring per each question set

        # Intermediate results of detection aren't saved, so this is just a temporary hack just to check if we really need to run detection again
        logits = torch.zeros((0, self.vlm.vocab_size)).float()
        if cache_path is not None:
            assert cache_path.endswith(".pt"), "Cache path should be .pt to store logits tensor!"
            if os.path.exists(cache_path):
                try:
                    logits = torch.load(cache_path)
                except:
                    pass
                    
        # Process frames using visual attention filter
        original_frames = frames
        if self.visual_filter is not None and logits.shape[0] < len(frames):
            if memory_tracker is not None:
                print("\nMemory (before running spatial filter)")
                memory_tracker.print_diff()

            if self.visual_filter_type == VisualFilterTypes.Spatial or self.visual_filter_type == VisualFilterTypes.Spatial_NoRephrase:
                frames, questions = self.visual_filter(self.nlp, frames, questions, return_visible_target_objects=False)
            elif self.visual_filter_type == VisualFilterTypes.Contrastive_Region:
                frames = self.visual_filter(self.nlp, frames, questions)
            else:
                raise ValueError(f"Visual filter type {self.visual_filter_type} not supported for generating VQG training data.")
        
        # Then delete these pre-loaded logits
        del logits 

        prompt_template = VQG2VQA_PROMPT_TEMPLATES[type(self.vlm)]
        prompts = [prompt_template.format(question=question) for question in questions]
        
        response_tokens = get_vqa_response_token_ids(self.processor.tokenizer)
            
        if memory_tracker is not None:
            print("\nMemory (before running VQA)")
            memory_tracker.print_diff()        

        # Run VQA in batches
        logits = run_vqa(vlm=self.vlm,
                         processor=self.processor,
                         prompts=prompts,
                         frames=frames,
                         batch_size=batch_size,
                         cache_path=cache_path)

        if self.visual_filter_type == VisualFilterTypes.Contrastive_Region:
            original_logits = run_vqa(vlm=self.vlm,
                         processor=self.processor,
                         prompts=prompts,
                         frames=original_frames,
                         batch_size=batch_size,
                         cache_path=cache_path)
        del original_frames
        
        if memory_tracker is not None:
            print("\nMemory (after running VQA)")
            memory_tracker.print_diff()

        # Gather up VQAOutputs (# examples, # questions per example)
        vqa_outputs = []
        parallel_idx = 0
        for example in examples: 
            for question_set in example.candidate_question_sets:
                this_vqa_outputs = []
                for _, answer in zip(question_set.questions, question_set.answers):
                    assert answers[parallel_idx] == answer, "Parallel input examples and VQA outputs are out of sync!"
                    this_vqa_outputs.append(
                        VQAOutputs(
                            example.task_name,
                            example.example_id,
                            example.procedure_id,
                            frames[parallel_idx], # Use manipulated frame after visual filter (if any)
                            prompts[parallel_idx],
                            answers[parallel_idx],
                            response_tokens,
                            original_logits[parallel_idx] - logits[parallel_idx]
                ) if self.visual_filter_type == VisualFilterTypes.Contrastive_Region else
                        VQAOutputs(
                            example.task_name,
                            example.example_id,
                            example.procedure_id,
                            frames[parallel_idx], # Use manipulated frame after visual filter (if any)
                            prompts[parallel_idx],
                            answers[parallel_idx],
                            response_tokens,
                            logits[parallel_idx]
                        )
                    )
                    parallel_idx += 1
                vqa_outputs.append(this_vqa_outputs)
        
        answer_probs, scores = self.get_scores(mistake_labels, vqa_outputs)
        vqg_training_examples = [
            VQGTrainingExample(
                task_name=MistakeDetectionTasks.Ego4D,
                procedure_id=vqg_output.procedure_id, 
                procedure_description=vqg_output.procedure_description,
                prompt=example.prompt,
                candidate_id=candidate_id,
                questions=vqg_output.questions,
                expected_answers=vqg_output.answers,
                answer_probs=prob,
                preference_score=score
            )
            for candidate_id, (vqg_output, example, prob, score) in enumerate(zip([vqg_output for ex in examples for vqg_output in ex.candidate_question_sets], 
                                                                            [ex for ex in examples for _ in ex.candidate_question_sets],
                                                                            answer_probs, 
                                                                            scores))
        ]

        if return_vqa_outputs:
            return vqg_training_examples, vqa_outputs
        else:
            return vqg_training_examples
    