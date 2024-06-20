from dataclasses import dataclass, asdict
from enum import Enum
import json
import os
from PIL import Image
from typing import Optional, Any, Union, Iterable, Generator

from travel.data.utils import split_list_into_partitions
from travel.data.utils.image import FRAME_DIMENSION

def get_cutoff_time_by_proportion(frame_times: list[float], proportion: float):
    """Returns a cutoff time for the last N% of frames to support MistakeDetectionEvaluator classes."""
    return max(frame_times) - ((max(frame_times) - min(frame_times)) * proportion)

class MistakeDetectionTasks(str, Enum):
    CaptainCook4D = "captaincook4d"
    Ego4D = "ego4d" # Ego4D following SuccessVQA format
    Ego4D_Augmented = "ego4d_augmented" # Ego4D augmented with more negative examples from mismatched verb/noun
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
        if type(self.frames[0]) != str:
            new_sizes = [(int(FRAME_DIMENSION * (frame.width / frame.height)), FRAME_DIMENSION) if frame.width > frame.height else (FRAME_DIMENSION, int(FRAME_DIMENSION * (frame.height / frame.width))) for frame in self.frames]
            # Frames have been cached before and not yet uncached - they should already be resized
            self.frames = [frame.resize(frame_size) for frame_size, frame in zip(new_sizes, self.frames)]

    @staticmethod
    def from_dict(data: dict, load_frames: bool=True):
        """
        Loads an instance of MistakeDetectionExample from a dictionary. This adds special logic to account for loading the frame image.

        :param data: Dictionary of instance data.
        """
        assert "frames" in data, "Can't use from_dict on this class without including `frames` list of images."
        if load_frames:
            data["frames"] = [Image.open(fname) for fname in data["frames"]]
        example = MistakeDetectionExample(**data)
        return example
    
    def cache_frames(self, image_base_path: str):
        assert all(type(frame) == Image.Image for frame in self.frames), "Can only cache PIL images!"

        if not os.path.exists(os.path.join(image_base_path, "frames")):
            os.makedirs(os.path.join(image_base_path, "frames"))

        frame_paths = []
        for fi, frame in enumerate(self.frames):
            image_path = os.path.join(image_base_path, "frames", f"frame_{self.example_id.replace('/', '-')}_{fi}.jpg")

            frame.save(image_path)
            frame.close()
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
                self.cache_frames(image_base_path)
        
        # Convert to dictionary
        return_dict = {
            k: v for k, v in asdict(self).items()
        }
        # If we didn't cache the images, delete them since we can't serialize them directly in json
        if image_base_path is None:
            del return_dict['frames']
        # And round off floating point values
        return_dict['frame_times'] = [float(round(t, 9)) for t in return_dict['frame_times']]
        return return_dict

    def cutoff_to_last_frames(self, proportion: float):
        """
        Cuts off example's frames and frame times to the last `proportion`% of frames based on their times.
        """
        original_length = len(self.frames)
        min_time = min(self.frame_times)
        max_time = max(self.frame_times)
        cutoff_time = get_cutoff_time_by_proportion(self.frame_times, proportion)
        self.frames = [f for f, t in zip(self.frames, self.frame_times) if t >= cutoff_time]
        self.frame_times = [t for t in self.frame_times if t >= cutoff_time]
        new_length = len(self.frames)
        print(f"{self.example_id}: cut off from {original_length} to {new_length} frames (proportion={proportion}, cutoff time={cutoff_time}, time range={min_time}-{max_time})")

class MistakeDetectionDataset:
    """Superclass for loading and storing a mistake detection dataset."""
    def __init__(self, data_split: str, **kwargs: dict[str, Any]):
        """
        Method to initialize and load dataset.

        :param kwargs: Task-specific arguments for dataset compilation.
        """
        self.data_split: str = data_split
        self.cache_dir = self.get_cache_dir(data_split, **kwargs)
        self.example_dirs: list[str] = []
        self.n_examples: int = 0
        self.data_generated: bool = False

        if not os.path.exists(self.cache_dir):
            # Loading dataset for the first time
            self.generate_examples(data_split, **kwargs)
        else:
            # Dataset directory already exists, but may or may not be fully generated
            self.load_dataset_metadata()
            if not self.data_generated:
                self.generate_examples(data_split, **kwargs)

        self.data_generated = True
        self.save_dataset_metadata()

    def __len__(self):
        return len(self.example_dirs)

    def __getitem__(self, index):
        return self.load_example_from_file(self.example_dirs[index])
    
    def get_batches(self, batch_size: int, n_workers: int=1, worker_index: int=0) -> Iterable[list[MistakeDetectionExample]]:
        assert batch_size >= 1, "Batch size must be positive!"
        assert n_workers >= 1, "Number of workers must be positive!"

        # If processing the dataset in parallel, give the appropriate partition of the dataset
        if n_workers == 1:
            example_dirs = self.example_dirs
        else:
            example_dirs = split_list_into_partitions(self.example_dirs, n_workers)[worker_index]

        if batch_size > 1:
            for i in range(0, len(example_dirs), batch_size):
                yield [self.load_example_from_file(d) for d in example_dirs[i : i + batch_size]]
        else:
            for d in example_dirs:
                yield self.load_example_from_file(d)
    
    def count_batches(self, batch_size: int, n_workers: int=1, worker_index: int=0) -> int:
        """
        Returns the number of batches in the dataset (or a partition of it in parallel situations), given a batch size, number of workers, and worker index.
        """
        assert batch_size >= 1, "Batch size must be positive!"
        assert n_workers >= 1, "Number of workers must be positive!"

        # If processing the dataset in parallel, give the appropriate partition of the dataset
        if n_workers == 1:
            example_dirs = self.example_dirs
        else:
            example_dirs = split_list_into_partitions(self.example_dirs, n_workers)[worker_index]

        if batch_size > 1:
            return list(range(0, len(example_dirs), batch_size))
        else:
            return len(example_dirs)

    def __iter__(self):
        return self.get_batches(1)
    
    def generate_examples(self, data_split: str, **kwargs: dict[str, Any]) -> list[MistakeDetectionExample]:
        raise NotImplementedError("Subclass should implement dataset loading procedure.")

    def get_cache_dir(self, **kwargs: dict[str, Any]) -> str:
        raise NotImplementedError("Subclass should implement logic for generating cached data directory.")
        
    def get_example_dir(self, example_id: str) -> str:
        """
        Gets directory name for an example by its unique ID.

        :param example_id: Unique example ID.
        :return: Directory name.
        """
        return os.path.join(self.cache_dir, example_id)

    def save_example_to_file(self, example: MistakeDetectionExample):
        """
        Caches a preprocessed dataset example in a centralized cache folder to reuse later.
        
        :param example: Example to save.
        """
        example_cache_dir = self.get_example_dir(example.example_id)
        os.makedirs(example_cache_dir), f"Tried to save an example {example.example_id} that already exists! Check to make sure example IDs are unique or there aren't untracked files in cache directory."
        json.dump(example.to_dict(image_base_path=example_cache_dir), 
                  open(os.path.join(example_cache_dir, "example.json"), "w"),
                  indent=4)
        self.example_dirs.append(example_cache_dir)
        self.n_examples += 1
        
    def load_example_from_file(self, example_dir: str, load_frames: bool=True) -> MistakeDetectionExample:
        """
        Loads an example from cache by the directory it's saved in.

        :param example_dir: Directory to load example from.
        :return: MistakeDetectionExample object.
        """
        example = json.load(open(os.path.join(example_dir, "example.json"), "r"))
        example = MistakeDetectionExample.from_dict(example, load_frames=load_frames)
        return example
    
    def save_dataset_metadata(self):
        """
        Saves dataset metadata for later.
        """
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        json.dump(self.__dict__,
                  open(os.path.join(self.cache_dir, "dataset.json"), "w"),
                  indent=4)
        
    def load_dataset_metadata(self):
        """
        Loads cached dataset metadata, including the example directories and count.
        """
        if os.path.exists(os.path.join(self.cache_dir, "dataset.json")):
            data = json.load(open(os.path.join(self.cache_dir, "dataset.json"), "r"))
            self.cache_dir = data["cache_dir"]
            self.example_dirs = data["example_dirs"]
            self.n_examples = data["n_examples"]
            self.data_generated = data["data_generated"]

    def get_all_procedures(self) -> Generator[None, list[tuple[int, str]], None]:
        """
        Quickly returns a list of all procedures in the dataset (along with their IDs). Used for VQG procedure.
        """
        already_seen = []
        for d in self.example_dirs:
            example = self.load_example_from_file(d, load_frames=False)
            if example.procedure_id not in already_seen:
                yield (example.procedure_id, example.procedure_description)
                already_seen.append(example.procedure_id) 
