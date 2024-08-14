from collections import defaultdict
import math
import numpy as np
import os, json
import pandas as pd
from PIL import Image
from pprint import pprint
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

EK_ANNOTATION_PATH = f"/nfs/turbo/coe-chaijy/datasets/EPIC-KITCHENS/annotations/EPIC_100_{PARTITION}_full_sent.csv"
annotations = pd.read_csv(EK_ANNOTATION_PATH)

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
VIDEO_PATH = str(config["data"]["epic_kitchens"]["video_path"])
ANNOTATION_PATH_TRAIN = str(config["data"]["epic_kitchens"]["annotation_path_train"])
ANNOTATION_PATH_VAL = str(config["data"]["epic_kitchens"]["annotation_path_val"])
ANNOTATION_PATH_TEST = str(config["data"]["epic_kitchens"]["annotation_path_test"])
ANNOTATION_PATHS = {
    "train": EK_ANNOTATION_PATH_TRAIN,
    "val": EK_ANNOTATION_PATH_VAL,
    "test": EK_ANNOTATION_PATH_TEST,
}

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

        # Load EK annotations and video frames
        annotations = pd.read_csv(ANNOTATION_PATHS[data_split])
        all_data = []
        ids_by_verb_noun = defaultdict(dict)
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
            all_data.append(video_data)
            ids_by_verb_noun[(verb_class, noun_class)].append(video_index)

        # TODO: generate matched and mismatched examples
        # TODO: add debug count logic

        self.save_dataset_metadata()