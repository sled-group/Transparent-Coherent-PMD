from dataclasses import dataclass, asdict
import json
import os
from PIL import Image

from travel.data.mistake_detection import MistakeDetectionTasks
from travel.model.vqa import VQAResponse
from travel.model.vqg import VQGOutputs

@dataclass
class FrameVQAMistakeDetectionExample:
    """Class to store the data for a single-frame VQA mistake detection instance, including one or more candidate sets of questions. Used for optimizing LMs for question generation."""
    task_name: MistakeDetectionTasks
    video_id: str
    procedure_id: int
    example_id: str # Mistake detection example that frame will come from
    frame: Image.Image
    frame_time: float
    procedure_description: str
    mistake: bool
    candidate_question_sets: list[VQGOutputs]

    def __post_init__(self):
        """Saves a count of questions in this question set."""
        for cqs in self.candidate_question_sets:
            assert len(cqs.questions) == len(cqs.answers), "FrameVQAMistakeDetectionExample expected same number of questions and answers for each candidate question set!"
        self.n_candidates = len(self.candidate_question_sets)

    def to_dict(self):
        """Helper method to create a JSON-serializable version of the class instance (excluding some information)."""        
        return_dict = {
            k: v for k, v in asdict(self).items() if k not in ["frame"]
        }
        return_dict['frame_time'] = float(round(return_dict['frame_time'], 3))
        return return_dict
    
def save_frameVQA_examples(frameVQA_examples: list[FrameVQAMistakeDetectionExample], path: str):
    """
    Saves list of FrameVQAMistakeDetectionExample created by `run_vqg_learning_generation.py`.
    
    :param frameVQA_examples: List of FrameVQAMistakeDetectionExample.
    :param path: Path to save json file (directory).
    """
    if not os.path.exists(path):
        os.makedirs(path)
    json.dump([example.to_dict() for example in frameVQA_examples], 
              open(os.path.join(path, "frameVQA_examples.json"), "w"),
              indent=4)    

# TODO: handle frames here for this to be doable

# def load_frameVQA_examples(path: str) -> dict[int, FrameVQAMistakeDetectionExample]:
#     """
#     Loads list of FrameVQAMistakeDetectionExample created by `run_vqg_learning_generation.py`.
    
#     :param path: Path to directory to load json file from.
#     """
#     frameVQA_examples = json.load(open(os.path.join(path, "frameVQA_examples.json"), "r"))
#     frameVQA_examples = {int(k): VQGOutputs.from_dict(v) for k, v in vqg_outputs.items()}
#     return frameVQA_examples


@dataclass
class VQGTrainingExample:
    """Class to store data to optimize LMs to generate visual questions."""
    task_name: MistakeDetectionTasks
    procedure_id: int
    procedure_description: str
    prompt: str
    candidate_id: int # Index of generated question set (unique for a given procedure ID)
    questions: list[str]
    expected_answers: list[VQAResponse]
    preference_score: float

    def to_dict(self):
        """Helper method to create a JSON-serializable version of the class instance (excluding some information)."""        
        return_dict = {
            k: v for k, v in asdict(self).items()
        }
        return_dict['preference_score'] = float(round(return_dict['preference_score'], 3))
        return return_dict    