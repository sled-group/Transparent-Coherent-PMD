from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from enum import Enum
from PIL import Image
import torch

VQA_PROMPT_TEMPLATES = {
    "llava-hf/llava-1.5-7b-hf": "USER: <image>\n{question} (yes/no) ASSISTANT: "
}

class VQAResponse(Enum):
    No = 0
    Yes = 1

@dataclass_json
@dataclass
class VQAOutputs:
    """Dataclass to hold all VLM outputs from visual question answering (VQA)."""
    procedure_id: int
    frame: Image
    prompt: str
    expected_answer: VQAResponse
    response_token_ids: dict[VQAResponse, int]
    logits: torch.FloatTensor # (# questions, vocab size) 
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