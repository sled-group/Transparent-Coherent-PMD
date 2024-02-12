from dataclasses import dataclass, field
from enum import Enum
from PIL import Image
from typing import Optional, Any

class MistakeDetectionTasks(Enum):
    CaptainCook4D = "captaincook4d"

@dataclass
class MistakeDetectionExample:
    """Class to store the data for a video mistake detection instance."""
    task_name: MistakeDetectionTasks
    video_id: str
    procedure_id: int
    frames: list[Image.Image]
    frame_times: list[float]
    procedure_description: str
    mistake: bool
    mistake_type: Optional[str] = None # TODO: make this an enum
    mistake_description: Optional[str] = None

# TODO: integrate with Hugging Face later?
class MistakeDetectionDataset:
    """Superclass for loading and storing a mistake detection dataset."""
    def __init__(self, **kwargs: dict[str, Any]):
        """
        Method to initialize and load dataset.

        :param kwargs: Task-specific arguments for dataset compilation.
        """
        self.examples: list[MistakeDetectionExample] = self.load_examples(**kwargs)

    def load_examples(self, **kwargs: dict[str, Any]) -> list[MistakeDetectionExample]:
        raise NotImplementedError("Subclass should implement dataset loading procedure.")