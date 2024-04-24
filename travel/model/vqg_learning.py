import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Union

from travel.constants import DATA_CACHE_DIR
from travel.data.mistake_detection import FrameVQAMistakeDetectionExample, VQGTrainingExample
from travel.model.vqa import VQAOutputs, VQAResponse, VQG2VQA_PROMPT_TEMPLATES

# NOTE: we may need to employ multiple scorers (for several VLM types)
# NOTE: we may need to implement scorers for different types of inputs, e.g., video
class FrameVQAMistakeDetectionScorer:
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
        for mistake, example_vqa_outputs in zip(mistake_labels, vqa_outputs):
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

        return scores

    def __call__(self, 
                 examples: list[FrameVQAMistakeDetectionExample],
                 return_vqa_outputs: bool=False,
                 batch_size: int=1) -> Union[torch.FloatTensor, list[VQAOutputs]]:
        """
        Score visual questions when posed on individual video frames to a VLM.
        
        :param examples: List of FrameVQAMistakeDetectionExample objects to run through the VLM, each of which include a single frame, question, and expected answer for the frame.
        :param return_vqa_outputs: Whether to return VQAOutputs from VQA inference instead of scores per example.
        :param batch_size: Batch size for VQA inference. Note that quantized LLaVA may return nan logits if greater than 1.
        :return: FloatTensor of scores of shape (len(examples), # questions per example) and a list of VQAOutputs.
        """
        # Extract parallel frames, questions, answers, and mistake labels
        questions = [question for example in examples for question in example.questions]
        answers = [answer for example in examples for answer in example.expected_answers]
        frames = [example.frame for example in examples for _ in example.questions]
        assert len(questions) == len(answers) == len(frames), "Need same number of questions, answers, and frames to score questions on frame-based VQA!"
        mistake_labels = [example.mistake for example in examples]
             
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
        parallel_idx = 0
        for example in enumerate(examples): 
            this_vqa_outputs = []
            for _ in range(example.n_questions):
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
                parallel_idx += 1
            vqa_outputs.append(this_vqa_outputs)
        
        if return_vqa_outputs:
            return vqa_outputs
        else:
            # In most cases, we just want scores for each generated question set
            scores = self.get_scores(mistake_labels, vqa_outputs)
            return [
                VQGTrainingExample(
                    task_name=example.task_name,
                    procedure_id=example.procedure_id, 
                    procedure_description=example.procedure_description,
                    questions=example.questions,
                    expected_answers=example.expected_answers,
                    preference_score=score
                )
                for example, score in zip(examples, scores)
            ]