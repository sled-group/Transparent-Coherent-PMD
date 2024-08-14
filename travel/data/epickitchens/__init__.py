from collections import defaultdict
import datetime
import math
import numpy as np
import os, json
import pandas as pd
from PIL import Image
from pprint import pprint
import random
import spacy
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, BitsAndBytesConfig
from typing import Optional
import yaml

from travel.constants import DATA_CACHE_DIR, CONFIG_PATH
from travel.data.mistake_detection import MistakeDetectionExample, MistakeDetectionDataset, MistakeDetectionTasks
from travel.data.utils import generate_float_series, get_subdirectories
from travel.data.utils.image import variance_of_laplacian
from travel.data.utils.video import get_video, extract_frames
from travel.model.grounding import TargetObjectCounterFilter

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
VIDEO_PATH = str(config["data"]["epickitchens"]["video_path"])
ANNOTATION_PATH_TRAIN = str(config["data"]["epickitchens"]["annotation_path_train"])
ANNOTATION_PATH_VAL = str(config["data"]["epickitchens"]["annotation_path_val"])
ANNOTATION_PATH_TEST = str(config["data"]["epickitchens"]["annotation_path_test"])
ANNOTATION_PATHS = {
    "train": ANNOTATION_PATH_TRAIN,
    "val": ANNOTATION_PATH_VAL,
    "test": ANNOTATION_PATH_TEST,
}

# TODO: some verbs need to be ignored in epic kitchens
IGNORE_VERBS = [
    
]

class EpicKitchensMistakeDetectionDataset(MistakeDetectionDataset):
    def __init__(self, 
                 data_split: str,
                 debug_n_examples_per_class: Optional[int]=None):
        """
        Method to initialize and load (single-frame) Epic Kitchens mistake detection dataset.

        :param kwargs: Task-specific arguments for dataset compilation.
        """
        super().__init__(data_split,
                         debug_n_examples_per_class=debug_n_examples_per_class)

    def get_cache_dir(self,
                      data_split: str,
                      debug_n_examples_per_class: Optional[int]=None) -> str:
        # Check if we already loaded data before
        cache_fname = f"{'epic_kitchens'}_{data_split}" 
        if debug_n_examples_per_class is not None:
            cache_fname += f"_debug{debug_n_examples_per_class}"
        cache_fname = os.path.join(DATA_CACHE_DIR, cache_fname)
        return cache_fname

    def generate_examples(self,
                          data_split: str,
                          debug_n_examples_per_class: Optional[int]=None) -> list[MistakeDetectionExample]:

        # If we're only loading a small amount of data for debugging purpose but we've loaded the full data before, just borrow the data from the full dataset
        if debug_n_examples_per_class is not None:
            full_cache_dir = self.get_cache_dir(data_split, debug_n_examples_per_class=None)
            if os.path.exists(full_cache_dir):
                full_example_dirs = json.load(open(os.path.join(full_cache_dir, "dataset.json"), "r"))["example_dirs"]
                
                positive_dirs = [d for d in full_example_dirs if d.endswith("pos")]
                random.shuffle(positive_dirs)
                positive_dirs = positive_dirs[:debug_n_examples_per_class]

                negative_dirs = [d for d in full_example_dirs if not d.endswith("pos")]
                random.shuffle(negative_dirs)
                negative_dirs = negative_dirs[:debug_n_examples_per_class]

                self.example_dirs = positive_dirs + negative_dirs
                self.n_examples += len(self.example_dirs)
                return

        # Load EK annotations and video frames
        annotations = pd.read_csv(ANNOTATION_PATHS[data_split])
        all_data = defaultdict(dict)
        data_by_verb_noun = defaultdict(list)
        video_paths = {}
        for row_idx, row in tqdm(annotations.iterrows(), total=len(annotations)):
            participant_index, video_index, clip_index = row['narration_id'].split('_')
            video_index = participant_index + "_" + str(video_index)
            clip_index = int(clip_index)
            video_path = os.path.join(VIDEO_PATH, participant_index, "videos", f"{video_index}.MP4")

            narration_text = row['narration']
            
            start_timestamp = row['start_timestamp']
            start_timestamp = datetime.strptime(start_timestamp, "%H:%M:%S.%f")
            stop_timestamp = row['stop_timestamp']
            stop_timestamp = datetime.strptime(stop_timestamp, "%H:%M:%S.%f")
            action_seconds = stop_timestamp - start_timestamp
            action_seconds = action_seconds.total_seconds()
            
            verb = row['verb']
            noun = row['noun']

            verb_class = int(row['verb_class'])
            noun_class = int(row['noun_class'])
            all_nouns_class = [int(c) for c in eval(row['all_noun_classes'])]

            video_data = {
                "video_path": video_path,
                "participant_id": participant_index,
                "video_id": video_index,
                "clip_index": clip_index,
                "verb": verb,
                "noun": noun,
                "all_nouns": eval(row["all_nouns"]),
                "verb_class": verb_class,
                "noun_class": noun_class, 
                "all_nouns_class": all_nouns_class,
                "narration_text": narration_text,
                "action_seconds": action_seconds,
                "start_timestamp": row['start_timestamp'],
                "stop_timestamp": row['stop_timestamp'],
            }
            all_data[video_index][clip_index] = video_data
            data_by_verb_noun[(verb_class, noun_class)].append(video_index)
            video_paths[video_index] = video_path

        for video_id in all_data.keys():
            try:
                video = get_video(video_paths[video_id])
            except:
                print(f"Warning: Video {video_id} could not be loaded!")
                continue
            example_cache_buffer = []

            for clip_index in sorted(list(all_data[video_index].keys())):
                example_data = all_data[video_id][clip_index]

                # Procedure ID is a unique integer representing verb and noun
                procedure_id = 100000 * example_data['verb_class'] + example_data['noun_class']

                # Positive example: use effect frame from clip
                try:
                    post_frame = extract_frames(video, [example_data['stop_timestamp']])
                    pos_example = MistakeDetectionExample(
                        task_name=MistakeDetectionTasks.EpicKitchens_Single,
                        video_id=example_data['video_id'],
                        procedure_id=procedure_id,
                        example_id=f"{example_data['video_id']}/{example_data['clip_index']}/pos",
                        frames=post_frame,
                        frame_times=[0.0],
                        procedure_description=example_data['narration_text'],
                        mistake=False,
                        mistake_type=None,
                        verb_noun_pair=(example_data['verb'], example_data['noun']),
                    )
                    example_cache_buffer.append(pos_example)
                except:
                    print(f"Warning: Could not generate a positive example for video {video_id}, clip {clip_index}!")

                # Hard negative example: use effect frame from clip
                try:
                    pre_frame = extract_frames(video, [example_data['start_timestamp']])
                    hardneg_example = MistakeDetectionExample(
                        task_name=MistakeDetectionTasks.EpicKitchens_Single,
                        video_id=example_data['video_id'],
                        procedure_id=procedure_id,
                        example_id=f"{example_data['video_id']}/{example_data['clip_index']}/hardneg",
                        frames=pre_frame,
                        frame_times=[0.0],
                        procedure_description=example_data['narration_text'],
                        mistake=True,
                        mistake_type="Action Incomplete",
                        verb_noun_pair=(example_data['verb'], example_data['noun']),
                    )
                    example_cache_buffer.append(hardneg_example)
                except:
                    print(f"Warning: Could not generate a hard negative example for video {video_id}, clip {clip_index}!")

                # Randomly select some mismatched verb and/or noun examples
                misalign_v_keys = [(v, n) for v, n in data_by_verb_noun.keys() if v != example_data['verb'] and n == example_data['noun']]
                misalign_n_keys = [(v, n) for v, n in data_by_verb_noun.keys() if v == example_data['verb'] and n != example_data['noun']]
                misalign_vn_keys = [(v, n) for v, n in data_by_verb_noun.keys() if v != example_data['verb'] and n != example_data['noun']]

                misalign_keys = [misalign_v_keys, misalign_n_keys, misalign_vn_keys]
                misalign_key_types = ["MisalignSRL_V", "MisalignSRL_ARG1", "MisalignSRL_V_ARG1"]
                for key_type, keys in zip(misalign_keys, misalign_key_types):
                    for _ in range(10):
                        try:
                            select_key = random.choice(keys)
                            other_example_data = random.choice(data_by_verb_noun[select_key])

                            misalign_frame = extract_frames(video, [example_data['start_timestamp']])
                            easyneg_example = MistakeDetectionExample(
                                task_name=MistakeDetectionTasks.EpicKitchens_Single,
                                video_id=example_data['video_id'],
                                procedure_id=procedure_id,
                                example_id=f"{example_data['video_id']}/{example_data['clip_index']}/easyneg_{key_type}",
                                frames=misalign_frame,
                                frame_times=[0.0],
                                procedure_description=other_example_data['narration_text'],
                                mistake=True,
                                mistake_type=key_type,
                                verb_noun_pair=(other_example_data['verb'], other_example_data['noun']),
                            )
                            example_cache_buffer.append(easyneg_example)
                            break
                        except:
                            continue
                    print(f"Warning: Could not generate an easy negative (type {key_type}) example for video {video_id}, clip {clip_index}!")

            # Release video capture
            video.release()

            # After getting all examples from this video, save them and move on
            for new_example in example_cache_buffer:
                self.save_example_to_file(new_example)
         
            # If we only wanted a small amount of data, stop early
            if debug_n_examples_per_class is not None and self.n_examples + len(example_cache_buffer) >= 2 * debug_n_examples_per_class:
                break
                
        # Cache any last examples in buffer
        for new_example in example_cache_buffer:
            self.save_example_to_file(new_example)
        self.save_dataset_metadata()