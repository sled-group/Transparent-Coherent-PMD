from dataclasses import dataclass, asdict
from enum import Enum
import json
import os
from PIL import Image
from typing import Optional, Any

class MistakeDetectionTasks(str, Enum):
    CaptainCook4D = "captaincook4d"
    Ego4D = "ego4d"
    # EpicKitchens = "epickitchens" # Can consider adding EK later if need more training data for VQG

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
    
    def to_dict(self, image_base_path: Optional[str]=None):
        """
        Helper method to create a JSON-serializable version of the class instance.

        :param image_base_path: Include this to save the 'frame' key to file, and replace it with its file path. Otherwise, the 'frame' key will be discarded.
        """        
        return_dict = {
            k: v for k, v in asdict(self).items() if k not in ["frames"]
        }
        if image_base_path is not None:
            if not os.path.exists(image_base_path):
                os.makedirs(image_base_path)
            
            for fi, frame in enumerate(self.frames):
                image_path = os.path.join(image_base_path, f"frame_{self.example_id}_{fi}.jpg")
                frame.save(image_path)
                return_dict["frame"] = image_path
        return_dict['frame_times'] = [float(round(t, 9)) for t in return_dict['frame_times']]
        return return_dict
    
    @staticmethod
    def from_dict(data: dict):
        """
        Loads an instance of MistakeDetectionExample from a dictionary. This adds special logic to account for loading the frame image.

        :param data: Dictionary of instance data.
        """
        assert "frames" in data, "Can't use from_dict on this class without including `frames` list of images."
        data["frames"] = [Image.open(fname) for fname in data["frame"]]
        example = MistakeDetectionExample(**data)
        return example

class MistakeDetectionDataset:
    """Superclass for loading and storing a mistake detection dataset."""
    def __init__(self, data_split: str, **kwargs: dict[str, Any]):
        """
        Method to initialize and load dataset.

        :param kwargs: Task-specific arguments for dataset compilation.
        """
        # Attempt to load examples from cache
        examples = self.load_cached_examples(data_split, **kwargs)

        # If we didn't get anything, load them from scratch and cache
        if examples is not None:
            self.examples: list[MistakeDetectionExample] = examples
        else:
            self.examples: list[MistakeDetectionExample] = self.load_examples(data_split, **kwargs)
        self.cache_examples(data_split, **kwargs)

        self.data_split: str = data_split

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
    
    def __iter__(self):
        return iter(self.examples)

    def load_examples(self, data_split: str, **kwargs: dict[str, Any]) -> list[MistakeDetectionExample]:
        raise NotImplementedError("Subclass should implement dataset loading procedure.")

    def get_cache_fname(self, **kwargs: dict[str, Any]):
        raise NotImplementedError("Subclass should implement logic for generating cached data filename.")
    
    def load_examples_from_file(self, path: str):
        """
        Loads cached json file of MistakeDetectionExample instances.

        :param path: File path of cached data. Can retrieve this automatically by calling self.get_cache_fname.
        """
        assert path.endswith(".json"), "Expected data to be cached in a json file!"

    def load_cached_examples(self, data_split: str, **kwargs: dict[str, Any]) -> Optional[list[MistakeDetectionExample]]:
        cache_fname = self.get_cache_fname(data_split, **kwargs)
        if os.path.exists(cache_fname):
            return self.load_examples_from_file(cache_fname)
        else:
            return None

    def save_examples_to_file(self, path: str):
        """
        Caches preprocessed dataset examples to reuse later.
        
        :param path: Path to save json file.
        """
        if not os.path.exists("/".join(path.split("/")[:-1])):
            os.makedirs("/".join(path.split("/")[:-1]))

        json.dump([v.to_dict(image_base_path=os.path.join("/".join(path.split("/")[:-1]), "frames")) for v in self.examples], 
                  open(path, "w"),
                  indent=4)
        
    def cache_examples(self, data_split: str, **kwargs: dict[str, Any]):
        cache_fname = self.get_cache_fname(data_split, **kwargs)
        self.save_examples_to_file(cache_fname)
    
def get_cutoff_time_by_proportion(example: MistakeDetectionExample, proportion: float):
    """Returns a cutoff time for the last N% of frames to support HeuristicMistakeDetectionEvaluator."""
    return max(example.frame_times) - ((max(example.frame_times) - min(example.frame_times)) * proportion)
