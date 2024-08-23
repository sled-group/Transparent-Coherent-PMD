from dataclasses import dataclass, field, asdict
from enum import Enum
import os
from PIL import Image
import torch
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, InstructBlipForConditionalGeneration, Kosmos2ForConditionalGeneration, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, Phi3ForCausalLM
from typing import Optional, Union
import uuid

from travel.data.mistake_detection import MistakeDetectionTasks

COMPLETION_PROMPT_TEMPLATES = {
    Blip2ForConditionalGeneration: "A photo of",
    InstructBlipForConditionalGeneration: "A photo of",
    Kosmos2ForConditionalGeneration: "<grounding> A photo of",
    LlavaForConditionalGeneration: 'USER: <image>\nWhat is happening in this photo? ASSISTANT: This is a photo of',
    LlavaNextForConditionalGeneration: '[INST] <image>\nWhat is shown in this image? [/INST] This is a photo of',
}

SUCCESSVQA_QUESTION_TEMPLATE = 'The current goal is "{step}". Has the person successfully finished doing this?'

VQA_PROMPT_TEMPLATES = {
    Blip2ForConditionalGeneration: "Question: {question} Answer:",
    InstructBlipForConditionalGeneration: "Question: {question} Answer: ",
    Kosmos2ForConditionalGeneration: "<grounding> Question: {question} Answer: ",
    LlavaForConditionalGeneration: "USER: <image>\n{question} ASSISTANT: ",
    LlavaNextForConditionalGeneration: "[INST] <image>\n{question} [/INST]",
}

CAPTION_VQA_PROMPT_TEMPLATES = {
    LlavaForConditionalGeneration: "USER: <image>\nWhat is happening in this photo? ASSISTANT: {caption} USER: {question} ASSISTANT: ",
}

VQG2VQA2SUCCESSVQA_PROMPT_TEMPLATES = {
    LlavaForConditionalGeneration: "USER: <image>\n{question1} ASSISTANT: {answer1} USER: {question2} ASSISTANT: {answer2} USER: " + f"{SUCCESSVQA_QUESTION_TEMPLATE} ASSISTANT: "
}

# Important prompt components for iterative VQA
IMAGE_TOKENS = {
    "Salesforce/instructblip-flan-t5-xl": "",
    "Salesforce/instructblip-flan-t5-xxl": "",
    "llava-hf/llava-1.5-7b-hf": "<image>\n",
    "microsoft/Phi-3-vision-128k-instruct": "<|image_1|>\n",
    "microsoft/Phi-3.5-vision-instruct": "<|image_1|>\n",
    "microsoft/kosmos-2-patch14-224": "<grounding>",
}
USER_START_TOKENS = {
    "Salesforce/instructblip-flan-t5-xl": " ",
    "Salesforce/instructblip-flan-t5-xxl": "",
    "llava-hf/llava-1.5-7b-hf": "USER: ",
    "microsoft/Phi-3-vision-128k-instruct": "<|user|>\n",
    "microsoft/Phi-3.5-vision-instruct": "<|user|>\n",
    "microsoft/kosmos-2-patch14-224": " ",
}
USER_END_TOKENS = {
    "Salesforce/instructblip-flan-t5-xl": " ",
    "Salesforce/instructblip-flan-t5-xxl": "",
    "llava-hf/llava-1.5-7b-hf": " ",
    "microsoft/Phi-3-vision-128k-instruct": "<|end|>\n",
    "microsoft/Phi-3.5-vision-instruct": "<|end|>\n",
    "microsoft/kosmos-2-patch14-224": " ",
}
ASSISTANT_START_TOKENS = {
    "Salesforce/instructblip-flan-t5-xl": " ",
    "Salesforce/instructblip-flan-t5-xxl": "",
    "llava-hf/llava-1.5-7b-hf": "ASSISTANT: ",
    "microsoft/Phi-3-vision-128k-instruct": "<|assistant|>\n",
    "microsoft/Phi-3.5-vision-instruct": "<|assistant|>\n",
    "microsoft/kosmos-2-patch14-224": " ",
}
ASSISTANT_END_TOKENS = {
    "Salesforce/instructblip-flan-t5-xl": " ",
    "Salesforce/instructblip-flan-t5-xxl": "",
    "llava-hf/llava-1.5-7b-hf": " ",
    "microsoft/Phi-3-vision-128k-instruct": "<|end|>\n",
    "microsoft/Phi-3.5-vision-instruct": "<|end|>\n",
    "microsoft/kosmos-2-patch14-224": " ",
}

IVQA_PREAMBLE = 'This is a photo of someone working on the procedure "{procedure}". I will ask a series of different yes/no questions to gather information about the scene, then use it to determine whether the person has successfully executed the procedure. The goal is to extract as much relevant information as possible from the scene, so I will not repeat questions.'
IVQA_PREAMBLE_TOPDOWN = 'This is a photo of someone working on the procedure "{procedure}". I will ask a series of different yes/no questions to determine whether the person has successfully executed the procedure, then explain why based on information about the scene. The goal is to extract as much relevant information as possible from the scene, so I will not repeat questions.'

IVQA_SUCCESS_QUESTION = 'Based on the image and above information, has the procedure "{procedure}" been successfully executed?'
IVQA_SUCCESS_QUESTION_TOPDOWN = 'Based on the image, has the procedure "{procedure}" been successfully executed?'

class VQAResponse(int, Enum):
    No = 0
    Yes = 1

def get_vqa_response_token_ids(tokenizer):
    responses = {response: tokenizer(response.name, add_special_tokens=False)['input_ids'][0] for response in VQAResponse}
    for token_id in responses.values():
        assert type(token_id) == int, "Getting response tokens for members of VQAResponse failed."
    return responses

@dataclass
class VQAOutputs:
    """Dataclass to hold all VLM outputs from visual question answering (VQA)."""
    task_name: MistakeDetectionTasks
    example_id: str
    procedure_id: int
    frame: Union[Image.Image, str]
    prompt: str
    expected_answer: Optional[VQAResponse]
    response_token_ids: dict[VQAResponse, int]
    logits: Optional[torch.FloatTensor] # (vocab size) 
    question: Optional[str] = None
    target_object_counts: Optional[dict[str, int]] = None
    answer_probs: dict[VQAResponse, float] = field(default_factory=dict)
    predicted_answer: VQAResponse = VQAResponse["No"]

    def __post_init__(self):
        """Processes logits output from VLM into answer probabilities and final answer."""
        if len(self.answer_probs) == 0:
            for response_type in VQAResponse:
                assert response_type in self.response_token_ids, f"VLM token ID for {response_type} not provided in VQAOutputs.answer_token_ids."

            this_probs = torch.stack([self.logits[self.response_token_ids[response_type]] for response_type in VQAResponse], dim=0)
            this_probs = torch.softmax(this_probs, dim=0)
            
            self.predicted_answer = VQAResponse(torch.argmax(this_probs, dim=0).numpy())
            
            this_probs = this_probs.numpy()
            self.answer_probs = {response_type: this_probs[response_type.value] for response_type in VQAResponse}

    def to_dict(self, image_base_path: Optional[str]=None):
        """Helper method to create a JSON-serializable version of the class instance (excluding some information)."""
        return_dict = {
            k: v for k, v in asdict(self).items() if k not in ["frame", "response_token_ids", "logits"]
        }
        for response in return_dict['answer_probs']:
            return_dict['answer_probs'][response] = float(round(return_dict['answer_probs'][response], 3))
        
        # Cache frame for this example if it's not already cached somewhere
        if image_base_path is not None and type(self.frame) != str:
            self.cache_frame(image_base_path)
        return_dict["frame"] = self.frame

        return return_dict
        
    def cache_frame(self, image_base_path: str):
        assert type(self.frame) != str, "Can only cache PIL images!"

        if not os.path.exists(os.path.join(image_base_path, "frames")):
            os.makedirs(os.path.join(image_base_path, "frames"))

        # Since each VQAOutputs does not have a unique example ID, generate a unique UUID for it
        frame_uuid = str(uuid.uuid4())
        frame_path = os.path.join(image_base_path, "frames", f"frame_{self.example_id.replace('/', '-')}_{frame_uuid}.jpg")
        self.frame.save(frame_path)

        self.frame = frame_path

    def uncache_frame(self):
        assert type(self.frame) == str and self.frame.endswith(".jpg"), "Can only uncache string .jpg filenames!"
        self.frame = Image.open(self.frame)

def generate_iterative_vqa_success_prompt(questions, answers, procedure, vlm_type):
    """Generates all success VQA prompts for iterative VQA setting given the full set of questions and answers. Some of the code here is redundant with code in run_vqa_iterative.py and run_vqa_iterative_noimg.py, so if making changes here, make sure to keep those scripts concurrent with this."""
    question_success = IVQA_SUCCESS_QUESTION.format(procedure=procedure)
    success_prompt = f'{ASSISTANT_END_TOKENS[vlm_type]}{USER_START_TOKENS[vlm_type]}Q: {question_success}{USER_END_TOKENS[vlm_type]}{ASSISTANT_START_TOKENS[vlm_type]}A:'
    for question_idx in range(len(questions)):
        prompt = f'{USER_START_TOKENS[vlm_type]}{IMAGE_TOKENS[vlm_type]}{IVQA_PREAMBLE.format(procedure=procedure)}'
        for question, answer in zip(questions[:question_idx + 1], answers[:question_idx + 1]):
            prompt += f"{ASSISTANT_END_TOKENS[vlm_type] if question_idx != 0 else USER_END_TOKENS[vlm_type]}{USER_START_TOKENS[vlm_type]}Q: "
            prompt += f'{question}{USER_END_TOKENS[vlm_type]}{ASSISTANT_START_TOKENS[vlm_type]}A: {VQAResponse(answer).name}'

    return prompt + success_prompt
