import spacy
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from typing import Optional, Union

from travel.constants import DATA_CACHE_DIR
from travel.data.vqg_learning import FrameVQAMistakeDetectionExample, VQGTrainingExample
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.model.grounding import VisualFilterTypes, SpatialVisualFilter
from travel.model.vqa import VQAOutputs, VQAResponse, VQG2VQA_PROMPT_TEMPLATES, run_vqa

# NOTE: we may need to employ multiple scorers (for several VLM types)
# NOTE: we may need to implement scorers for different types of inputs, e.g., video
class FrameVQAMistakeDetectionScorer:
    """Class that provides preference scores for visual questions to facilitate mistake detection on individual video frames."""
    def __init__(self, 
                 vlm_name: str,
                 visual_filter_type: Optional[VisualFilterTypes]=None):
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
        self.vlm = AutoModelForVision2Seq.from_pretrained(vlm_name, 
                                                          cache_dir=DATA_CACHE_DIR,
                                                          quantization_config=bnb_config)
        self.vlm.language_model.generation_config.top_p = None
        self.vlm.language_model.generation_config.temperature = None
        self.vlm.language_model.generation_config.do_sample = False
        self.processor.tokenizer.padding_side = "left"

        # TODO: may need to make sure this is on a different GPU than self.vlm
        if self.visual_filter_type == VisualFilterTypes.Spatial:
            self.visual_filter = SpatialVisualFilter() # TODO: or quantize OWL if possible?
        else:
            self.visual_filter = None
        self.nlp = spacy.load('en_core_web_sm')
        
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
                 cache_path: Optional[str]=None) -> Union[torch.FloatTensor, list[VQAOutputs]]:
        """
        Score visual questions when posed on individual video frames to a VLM.
        
        :param examples: List of FrameVQAMistakeDetectionExample objects to run through the VLM, each of which include a single frame, question, and expected answer for the frame.
        :param return_vqa_outputs: Whether to return VQAOutputs from VQA inference instead of scores per example.
        :param batch_size: Batch size for VQA inference. Note that quantized LLaVA may return nan logits if greater than 1.
        :param cache_path: Path to save a .pt file for logits generated so far.
        :return: FloatTensor of scores of shape (len(examples), # questions per example) and a list of VQAOutputs.
        """
        # Extract parallel frames, questions, answers, and mistake labels
        questions = [question for example in examples for question_set in example.candidate_question_sets for question in question_set.questions]
        answers = [answer for example in examples for question_set in example.candidate_question_sets for answer in question_set.answers]
        frames = [example.frame for example in examples for question_set in example.candidate_question_sets for _ in question_set.questions]
        assert len(questions) == len(answers) == len(frames), "Need same number of questions, answers, and frames to score questions on frame-based VQA!"
        mistake_labels = [example.mistake for example in examples for _ in example.candidate_question_sets] # One scoring per each question set
             
        # Process frames using visual attention filter
        if self.visual_filter is not None:
            frames, questions = self.visual_filter(self.nlp, frames, questions)

        prompt_template = VQG2VQA_PROMPT_TEMPLATES[type(self.vlm)]
        prompts = [prompt_template.format(question=question) for question in questions]
        
        response_tokens = {}
        for response_type in VQAResponse:
            response_tokens[response_type] = self.processor.tokenizer(response_type.name, add_special_tokens=False)['input_ids'][0]
            
        # Run VQA in batches
        logits = run_vqa(vlm=self.vlm,
                         processor=self.processor,
                         prompts=prompts,
                         frames=frames,
                         batch_size=batch_size,
                         cache_path=cache_path)
        
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
                            example.frame,
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
    