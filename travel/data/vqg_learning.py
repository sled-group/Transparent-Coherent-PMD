from dataclasses import dataclass, asdict
import json
import os
import pickle
from PIL import Image
from typing import Optional

from travel.data.mistake_detection import MistakeDetectionTasks
from travel.model.vqa import VQAResponse
from travel.model.vqg import VQGOutputs

@dataclass
class FrameVQAMistakeDetectionExample:
    """Class to store the data for a single-frame VQA mistake detection instance, including one or more candidate sets of questions. Used for generating preference optimization data for question generation in LMs."""
    task_name: MistakeDetectionTasks
    video_id: str
    procedure_id: int
    example_id: str # Mistake detection example that frame will come from (should be unique, since this is one-to-one with MistakeDetectionExample dataset we started with)
    frame: Image.Image
    frame_time: float
    procedure_description: str
    mistake: bool
    prompt: str
    candidate_question_sets: list[VQGOutputs]

    def __post_init__(self):
        """Saves a count of questions in this question set."""
        for cqs in self.candidate_question_sets:
            assert len(cqs.questions) == len(cqs.answers), "FrameVQAMistakeDetectionExample expected same number of questions and answers for each candidate question set!"
        self.n_candidates = len(self.candidate_question_sets)

    def to_dict(self, image_base_path: Optional[str]=None):
        """
        Helper method to create a JSON-serializable version of the class instance.

        :param image_base_path: Include this to save the 'frame' key to file, and replace it with its file path. Otherwise, the 'frame' key will be discarded.
        """        
        return_dict = {
            k: v for k, v in asdict(self).items() if k not in ["frame"]
        }
        if image_base_path is not None:
            if not os.path.exists(image_base_path):
                os.makedirs(image_base_path)
            image_path = os.path.join(image_base_path, f"frame_{self.example_id}.jpg")
            self.frame.save(image_path)
            return_dict["frame"] = image_path
        return_dict['frame_time'] = float(round(return_dict['frame_time'], 9))
        return return_dict
    
    @staticmethod
    def from_dict(data: dict):
        """
        Loads an instance of FrameVQAMistakeDetectionExample from a dictionary. This adds special logic to account for loading the frame image.

        :param data: Dictionary of instance data.
        """
        assert "frame" in data, "Can't use from_dict on this class without including a `frame` image."
        data["frame"] = Image.open(data["frame"])
        data["candidate_question_sets"] = [VQGOutputs(**output) for output in data["candidate_question_sets"]]
        example = FrameVQAMistakeDetectionExample(**data)
        return example
    
def save_frameVQA_examples(frameVQA_examples: list[FrameVQAMistakeDetectionExample], path: str, partition: str):
    """
    Saves list of FrameVQAMistakeDetectionExample created by `run_vqg_learning_generation.py`.
    
    :param frameVQA_examples: List of FrameVQAMistakeDetectionExample.
    :param path: Path to save json file (directory).
    """
    fname = f"frameVQA_examples_{partition}.json"
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, "frames"))
    json.dump([example.to_dict(os.path.join(path, "frames")) for example in frameVQA_examples], 
              open(os.path.join(path, fname), "w"),
              indent=4)

def load_frameVQA_examples(path: str, partition: str) -> list[FrameVQAMistakeDetectionExample]:
    """
    Loads list of FrameVQAMistakeDetectionExample created by `run_vqg_learning_generation.py`.
    
    :param path: Path to directory to load pkl file from.
    """
    fname = f"frameVQA_examples_{partition}.json"
    frameVQA_examples = json.load(open(os.path.join(path, fname), "r"))
    frameVQA_examples = [FrameVQAMistakeDetectionExample.from_dict(d) for d in frameVQA_examples]
    return frameVQA_examples

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
    answer_probs: list[tuple[float, float]]
    preference_score: float

    def to_dict(self):
        """Helper method to create a JSON-serializable version of the class instance (excluding some information)."""        
        return_dict = {
            k: v for k, v in asdict(self).items()
        }
        return_dict['preference_score'] = float(round(return_dict['preference_score'], 9))
        return_dict['answer_probs'] = [(float(round(prob[0], 9)), float(round(prob[1], 9))) for prob in return_dict['answer_probs']]
        return return_dict    
    
def save_vqg_training_examples(examples: list[VQGTrainingExample], path: str, partition: str):
    """
    Saves list of VQGTrainingExample.
    
    :param examples: List of VQGTrainingExample generated by run_vqg_learning_vqa.py.
    :param path: Path to save json file (directory).
    """
    fname = f"vqg_training_examples_{partition}.json"
    if not os.path.exists(path):
        os.makedirs(path)
    json.dump([ex.to_dict() for ex in examples], 
              open(os.path.join(path, fname), "w"),
              indent=4)    