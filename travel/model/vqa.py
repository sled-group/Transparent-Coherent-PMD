from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import os
from PIL import Image
from pprint import pprint
import torch
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, InstructBlipForConditionalGeneration, Kosmos2ForConditionalGeneration, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from typing import Optional, Union

from travel.constants import CACHE_FREQUENCY
from travel.data.mistake_detection import MistakeDetectionTasks

COMPLETION_PROMPT_TEMPLATES = {
    Blip2ForConditionalGeneration: "A photo of",
    InstructBlipForConditionalGeneration: "A photo of",
    Kosmos2ForConditionalGeneration: "<grounding> A photo of",
    LlavaForConditionalGeneration: 'USER: <image>\nWhat is happening in this photo? ASSISTANT: This is a photo of',
    LlavaNextForConditionalGeneration: '[INST] <image>\nWhat is shown in this image? [/INST] This is a photo of',
}

SUCCESSVQA_PROMPT_TEMPLATES = {
    Blip2ForConditionalGeneration: 'Question: The current goal is "{step}". Has the person successfully finished doing this? Answer:',
    InstructBlipForConditionalGeneration: 'Question: The current goal is "{step}". Has the person successfully finished doing this? Answer:',
    Kosmos2ForConditionalGeneration: '<grounding> Q: The current goal is "{step}". Has the person successfully finished doing this? A: ',
    LlavaForConditionalGeneration: 'USER: <image>\nThe current goal is "{step}". Has the person successfully finished doing this? ASSISTANT: ',
    LlavaNextForConditionalGeneration: '[INST] <image>\nThe current goal is "{step}". Has the person successfully finished doing this? [/INST]',
}

VQG2VQA_PROMPT_TEMPLATES = {
    Blip2ForConditionalGeneration: "Question: {question}? Answer:",
    InstructBlipForConditionalGeneration: "Question: {question}? Answer: ",
    Kosmos2ForConditionalGeneration: "<grounding> Question: {question} Answer: ",
    LlavaForConditionalGeneration: "USER: <image>\n{question} ASSISTANT: ",
    LlavaNextForConditionalGeneration: "[INST] <image>\n{question} [/INST]",
}

class VQAResponse(int, Enum):
    No = 0
    Yes = 1

def get_vqa_response_token_ids(tokenizer):
    responses = {response: tokenizer(response.name, add_special_tokens=False)['input_ids'][0] for response in VQAResponse}
    for token_id in responses.values():
        assert type(token_id) == int, "Getting response tokens for members of VQAResponse failed."
    return responses

# TODO: support saving images
@dataclass
class VQAOutputs:
    """Dataclass to hold all VLM outputs from visual question answering (VQA)."""
    task_name: MistakeDetectionTasks
    example_id: str
    procedure_id: int
    frame: Union[Image.Image, str]
    prompt: str
    expected_answer: VQAResponse
    response_token_ids: dict[VQAResponse, int]
    logits: Optional[torch.FloatTensor] # (vocab size) 
    answer_probs: dict[VQAResponse, float] = field(default_factory=list)
    predicted_answer: VQAResponse = VQAResponse["No"]

    def __post_init__(self):
        """Processes logits output from VLM into answer probabilities and final answer."""
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
            image_base_path = os.path.join(image_base_path, "frames")
            if not os.path.exists(image_base_path):
                os.makedirs(image_base_path)
            self.cache_frame(image_base_path)
            return_dict["frame"] = self.frame

        return return_dict
    
    def cache_frame(self, image_base_path: str):
        assert type(self.frame) != str, "Can only cache PIL images!"

        if not os.path.exists(os.path.join(image_base_path, "frames")):
            os.makedirs(os.path.join(image_base_path, "frames"))

        frame_path = os.path.join(image_base_path, "frames", f"frame_{self.example_id.replace('/', '-')}.jpg")

        self.frame.save(frame_path)

        self.frame = frame_path

    def uncache_frame(self):
        assert type(self.frame) == str and self.frame.endswith(".jpg"), "Can only uncache string .jpg filenames!"
        self.frame = Image.open(self.frame)

def run_vqa(vlm: PreTrainedModel, 
            processor: ProcessorMixin,
            prompts: list[str],
            frames: list[Image.Image],
            batch_size: int=1,
            cache_path: Optional[str]=None) -> torch.FloatTensor:
    """
    Runs VQA for given prompts and frames in batches with a given VLM and its processor.

    :param vlm: VLM for conditional generation from `transformers`.
    :param process: VLM processor from `transformers`, including tokenizer and image processor.
    :param prompts: List of prompts including visual questions.
    :param frames: List of images to ask visual questions about.
    :param batch_size: Batch size for running inference.
    :param cache_path: .pt file to cache incomplete logits in.
    :return: Full tensor of logits output from each question. The process of mapping this into VQAOutputs instances requires task/process-specific information, so it should be done outside of this method.
    """
    assert len(prompts) == len(frames), "Need same number of prompts and frames to run VQA!"

    # Run VQA in batches
    logits = torch.zeros((0, vlm.vocab_size)).float()
    if cache_path is not None:
        assert cache_path.endswith(".pt"), "Cache path should be .pt to store logits tensor!"
        if os.path.exists(cache_path):
            logits = torch.load(cache_path)
        else:
            if not os.path.exists("/".join(cache_path.split("/")[:-1])):
                os.makedirs("/".join(cache_path.split("/")[:-1]))

    last_save = 0
    with torch.no_grad():
        # Start at logits.shape[0] so we don't rerun any logits that were already cached (resuming logic)
        for i in tqdm(range(logits.shape[0], len(frames), batch_size), desc=f"running VQA ({str(vlm.device)})"):
            # Prepare the batch
            batch_frames = frames[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]

            # Run through VLM to get logits
            inputs = processor(text=batch_prompts, images=batch_frames, padding=True, return_tensors="pt")
            inputs = inputs.to(vlm.device)
            this_logits = vlm(**inputs).logits
            this_logits = this_logits[:, -1].detach().cpu()
            logits = torch.cat([logits, this_logits], dim=0)
            del this_logits

            # Cache logits so far
            if cache_path is not None and i - last_save >= CACHE_FREQUENCY:
                torch.save(logits, cache_path)
                last_save = i

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cache one more time
    if cache_path is not None:
        torch.save(logits, cache_path) 

    return logits

def save_vqa_outputs(vqa_outputs: list[VQAOutputs], path: str, partition: str):
    """
    Saves list of VQAOutputs.
    
    :param vqa_outputs: List of VQAOutputs.
    :param path: Path to save json file (directory).
    """
    fname = f"vqa_outputs_{partition}.json"
    if not os.path.exists(path):
        os.makedirs(path)
    json.dump([ex.to_dict(image_base_path=path) for ex in vqa_outputs], 
              open(os.path.join(path, fname), "w"),
              indent=4)    

def _shift_right(input_ids, decoder_start_token_id, pad_token_id):
    """Copy of _shift_right method from T5 to use with BLIP-2 to automatically generate decoder input IDs."""
    if decoder_start_token_id is None:
        raise ValueError(
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
            "See T5 docs for more information."
        )

    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids