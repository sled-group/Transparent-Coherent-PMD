from dataclasses import dataclass, asdict
from enum import Enum
import json
import os
from PIL import Image
from typing import Optional, Any, Union

from travel.data.utils.image import FRAME_DIMENSION

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
    frames: Union[list[Image.Image], list[str]]
    frame_times: list[float]
    procedure_description: str
    mistake: bool
    mistake_type: Optional[str] = None
    mistake_description: Optional[str] = None
    
    def __post_init__(self):
        """Resizes frames to save space in caching."""
        new_sizes = [(int(FRAME_DIMENSION * (frame.width / frame.height)), FRAME_DIMENSION) if frame.width > frame.height else (FRAME_DIMENSION, int(FRAME_DIMENSION * (frame.height / frame.width))) for frame in self.frames]
        self.frames = [frame.resize(frame_size) for frame_size, frame in zip(new_sizes, self.frames)]

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
    
    def cache_frames(self, image_base_path: str):
        assert all(type(frame) == Image.Image for frame in self.frames), "Can only cache PIL images!"

        if not os.path.exists(os.path.join(image_base_path, "frames")):
            os.makedirs(os.path.join(image_base_path, "frames"))

        frame_paths = []
        for fi, frame in enumerate(self.frames):
            image_path = os.path.join(image_base_path, "frames", f"frame_{self.example_id}_{fi}.jpg")

            frame.save(image_path)
            frame_paths.append(image_path)

        self.frames = frame_paths

    def uncache_frames(self):
        assert all(type(frame) == str and frame.endswith(".jpg") for frame in self.frames), "Can only uncache string .jpg filenames!"
        self.frames = [Image.open(frame) for frame in self.frames]
        
    def is_cached(self):
        return True if type(self.frames[0]) == str else False

    def to_dict(self, image_base_path: Optional[str]=None):
        """
        Helper method to create a JSON-serializable version of the class instance.

        :param image_base_path: Include this to save the 'frame' key to file, and replace it with its file path. Otherwise, the 'frame' key will be discarded.
        """        
        # Cache frames if possible and if not cached already
        if image_base_path is not None:
            if not self.is_cached():
                self.cache_frames(os.path.join(image_base_path, "frames"))
        
        # Convert to dictionary
        return_dict = {
            k: v for k, v in asdict(self).items() if k not in ["frames"]
        }
        return_dict['frame_times'] = [float(round(t, 9)) for t in return_dict['frame_times']]
        return return_dict

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
        self.index = 0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Cache previously accessed example if possible
        if self.index > 0 and not self.examples[self.index - 1].is_cached():
            self.examples[self.index - 1].cache_frames()

        if self.index < len(self.examples):
            example = self.examples[self.index]
            # Uncache from file if needed
            if example.is_cached():
                example.uncache_frames()
            self.index += 1
            return example
        else:
            raise StopIteration   
        # TODO: need to be able to cache after not used anymore 

    def load_examples(self, data_split: str, **kwargs: dict[str, Any]) -> list[MistakeDetectionExample]:
        raise NotImplementedError("Subclass should implement dataset loading procedure.")

    def get_cache_dir(self, **kwargs: dict[str, Any]) -> str:
        raise NotImplementedError("Subclass should implement logic for generating cached data directory.")
    
    def get_cache_fname(self, **kwargs: dict[str, Any]) -> str:
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

        json.dump([v.to_dict(image_base_path="/".join(path.split("/")[:-1])) for v in self.examples], 
                  open(path, "w"),
                  indent=4)
        
    def cache_examples(self, data_split: str, **kwargs: dict[str, Any]):
        cache_fname = self.get_cache_fname(data_split, **kwargs)
        self.save_examples_to_file(cache_fname)
    
def get_cutoff_time_by_proportion(example: MistakeDetectionExample, proportion: float):
    """Returns a cutoff time for the last N% of frames to support HeuristicMistakeDetectionEvaluator."""
    return max(example.frame_times) - ((max(example.frame_times) - min(example.frame_times)) * proportion)
