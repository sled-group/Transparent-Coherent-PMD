from dataclasses import dataclass, asdict
from PIL import Image

from travel.data.mistake_detection import MistakeDetectionTasks
from travel.model.vqa import VQAResponse

@dataclass
class FrameVQAMistakeDetectionExample:
    """Class to store the data for a single-frame VQA mistake detection instance, including a candidate set of questions. Used for optimizing LMs for question generation."""
    task_name: MistakeDetectionTasks # TODO: do we need all this info? some might be irrelevant or unused
    video_id: str
    procedure_id: int
    mistake_detection_example_id: str
    vqg_example_id: int
    frame: Image.Image
    frame_time: float
    procedure_description: str
    mistake: bool
    questions: list[str]
    expected_answers: list[VQAResponse]

    def __post_init__(self):
        """Saves a count of questions in this question set."""
        assert len(self.questions) == len(self.expected_answers), "FrameVQAMistakeDetectionExample expected same number of questions and answers!"
        self.n_questions = len(self.questions)

    def to_dict(self):
        """Helper method to create a JSON-serializable version of the class instance (excluding some information)."""        
        return_dict = {
            k: v for k, v in asdict(self).items() if k not in ["frame"]
        }
        return_dict['frame_time'] = float(round(return_dict['frame_time'], 3))
        return return_dict
    
@dataclass
class VQGTrainingExample:
    """Class to store data to optimize LMs to generate visual questions."""
    task_name: MistakeDetectionTasks
    procedure_id: int
    procedure_description: str
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