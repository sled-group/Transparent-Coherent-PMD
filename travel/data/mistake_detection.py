from dataclasses import dataclass, asdict
from enum import Enum
from PIL import Image
from typing import Optional, Any

class MistakeDetectionTasks(str, Enum):
    CaptainCook4D = "captaincook4d"

@dataclass
class MistakeDetectionExample:
    """Class to store the data for a video mistake detection instance."""
    task_name: MistakeDetectionTasks
    video_id: str
    procedure_id: int
    example_id: str
    frames: list[Image.Image]
    frame_times: list[float]
    procedure_description: str
    mistake: bool
    mistake_type: Optional[str] = None
    mistake_description: Optional[str] = None

    def to_dict(self):
        """Helper method to create a JSON-serializable version of the class instance (excluding some information)."""        
        return_dict = {
            k: v for k, v in asdict(self).items() if k not in ["frames"]
        }
        return_dict['frame_times'] = [float(round(ft, 3)) for ft in return_dict['frame_times']]
        return return_dict

class MistakeDetectionDataset:
    """Superclass for loading and storing a mistake detection dataset."""
    def __init__(self, data_split: str, load_videos: bool, **kwargs: dict[str, Any]):
        """
        Method to initialize and load dataset.

        :param kwargs: Task-specific arguments for dataset compilation.
        """
        self.examples: list[MistakeDetectionExample] = self.load_examples(data_split, load_videos, **kwargs)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
    
    def __iter__(self):
        return iter(self.examples)

    def load_examples(self, data_split: str, **kwargs: dict[str, Any]) -> list[MistakeDetectionExample]:
        raise NotImplementedError("Subclass should implement dataset loading procedure.")
    
def get_cutoff_time_by_proportion(example: MistakeDetectionExample, proportion: float):
    """Returns a cutoff time for the last N% of frames to support HeuristicMistakeDetectionEvaluator."""
    return max(example.frame_times) - ((max(example.frame_times) - min(example.frame_times)) * proportion)
