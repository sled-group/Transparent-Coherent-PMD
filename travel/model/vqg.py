from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from travel.model.vqa import VQAResponse

@dataclass_json
@dataclass
class VQGOutputs:
    """Dataclass to hold all LM outputs from visual question generation (VQG)."""
    procedure_id: int
    procedure_description: str
    target_object: str
    # state_description: str
    questions: list[str]
    answers_str: list[str]
    answers: list[VQAResponse] = field(default_factory=list)
    
    def __post_init__(self):
        """Validation steps to ensure every QA-pair is valid and every question has an answer."""
        for answer in self.answers_str:
            try: 
                self.answers.append(VQAResponse[answer])
            except:
                raise ValueError(f"Unrecognized VQA answer could not be accepted by VQAResponse class: {answer}")
            
        assert len(self.questions) == len(self.answers), "VQGOutputs received mismatched number of questions and answers."
        
        for question in self.questions:
            if not question.strip().endswith("?"):
                print(f"Warning: Question '{question}' doesn't appear to be a question.")
