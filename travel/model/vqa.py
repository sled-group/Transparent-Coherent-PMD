from dataclasses import dataclass, field, asdict
from dataclasses_json import dataclass_json
from enum import Enum
from PIL import Image
import torch

COMPLETION_PROMPT_TEMPLATES = {
    "Salesforce/blip2-flan-t5-xxl": "A photo of",
    "Salesforce/instructblip-flan-t5-xxl": "A photo of",
    "microsoft/kosmos-2-patch14-224": "<grounding> A photo of",
    "llava-hf/llava-1.5-7b-hf": 'USER: <image>\nWhat is happening in this photo? ASSISTANT: This is a photo of',
    "llava-hf/llava-v1.6-vicuna-7b-hf": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image? ASSISTANT: This is a photo of",
    "llava-hf/llava-v1.6-mistral-7b-hf": '[INST] <image>\nWhat is shown in this image? [/INST] This is a photo of',
}

SUCCESSVQA_PROMPT_TEMPLATES = {
    "Salesforce/blip2-flan-t5-xxl": 'Question: The current goal is "{step}". Has the person successfully finished doing this? Answer:',
    "Salesforce/instructblip-flan-t5-xxl": 'Question: The current goal is "{step}". Has the person successfully finished doing this? Answer:',
    "microsoft/kosmos-2-patch14-224": '<grounding> Q: The current goal is "{step}". Has the person successfully finished doing this? A: ',
    "llava-hf/llava-1.5-7b-hf": 'USER: <image>\nThe current goal is "{step}". Has the person successfully finished doing this? ASSISTANT: ',
    "llava-hf/llava-v1.6-vicuna-7b-hf": 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. USER: <image>\nThe current goal is "{step}". Has the person successfully finished doing this? ASSISTANT:',
    "llava-hf/llava-v1.6-mistral-7b-hf": '[INST] <image>\nThe current goal is "{step}". Has the person successfully finished doing this? [/INST]',
}

VQG2VQA_PROMPT_TEMPLATES = {
    "Salesforce/blip2-flan-t5-xxl": "Question: {question}? Answer:",
    "Salesforce/instructblip-flan-t5-xxl": "Question: {question}? Answer: ",
    "microsoft/kosmos-2-patch14-224": "<grounding> Question: {question} Answer: ",
    "llava-hf/llava-1.5-7b-hf": "USER: <image>\n{question} ASSISTANT: ",
    "llava-hf/llava-v1.6-vicuna-7b-hf": 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. USER: <image>\n{question} ASSISTANT:',
    "llava-hf/llava-v1.6-mistral-7b-hf": "[INST] <image>\n{question} [/INST]",
}

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
    example_id: str
    procedure_id: int
    frame: Image
    prompt: str
    expected_answer: VQAResponse
    response_token_ids: dict[VQAResponse, int]
    logits: torch.FloatTensor # (vocab size) 
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

    def to_dict(self):
        """Helper method to create a JSON-serializable version of the class instance (excluding some information)."""
        return_dict = {
            k: v for k, v in asdict(self).items() if k not in ["frame", "response_token_ids", "logits"]
        }
        for response in return_dict['answer_probs']:
            return_dict['answer_probs'][response] = float(round(return_dict['answer_probs'][response], 3))
        return return_dict
    
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