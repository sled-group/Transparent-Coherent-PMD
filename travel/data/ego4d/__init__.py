# Most code in this file was adapted from EILEV: https://github.com/yukw777/EILEV
from collections.abc import Callable, Iterable
from fractions import Fraction
from functools import partial
import json
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os
import pandas as pd
from PIL import Image
from pprint import pprint
from pytorchvideo.data import ClipSampler
from pytorchvideo.data.clip_sampling import ClipInfo
import random
import re
import spacy
from spacy.lang.en import English
import string
import torch
from transformers import BatchEncoding, DataCollatorForSeq2Seq, PreTrainedTokenizer
from typing import Any, TypeVar, Optional
from tqdm import tqdm
import yaml

from travel.constants import DATA_CACHE_DIR, RANDOM_SEED, CONFIG_PATH
from travel.data.ego4d.constants import EGO4D_ANNOTATION_PATH, EGO4D_SPLIT_PATHS, EGO4D_VIDEO_PATH, \
                                        MISALIGNSRL_PATH
from travel.data.mistake_detection import MistakeDetectionExample, MistakeDetectionDataset
from travel.data.utils import get_subdirectories, split_list_into_partitions, ResumableParallelSequentialSampler, generate_float_series
from travel.data.utils.image import variance_of_laplacian
from travel.data.utils.text import simple_present_to_imperative
from travel.data.utils.video import get_video, extract_frames
from travel.model.grounding import TargetObjectCounterFilter

# Some verbs would not be expected in the task-oriented mistake detection setting 
# (don't really cause a meaningful/observable/fully specifiable state change toward a task goal), so filter them out
EGO4D_IGNORE_VERBS = [
    'scroll', # Scrolling on phone, tablet, etc.
    'touch', # Touching an object
    'walk', # Walking around, sometimes to a destination
    'give', # Usually handing an object to someone else
    'read', # Reading a book
    'drive_(ride,_drive)', # (Mostly) driving vehicles, sometimes driving nails or other parts into something else
    'pet', # Petting animals
    'climb', # Climbing stairs, walls, rocks, etc.
    'play', # Playing games and instruments
    'point', # Aiming hands and objects at something else
    'consume_(taste,_sip,_eat,_drink)', # Eating and drinking
    'search', # Looking for objects (typically followed by picking something up)
    'enter', # Entering rooms or buildings
    'watch', # Watching videos and movies
    'park', # Parking a vehicle
    'talk_(talk,_interact,_converse)', # Talking with others
    'sit', # (Most often) a person sitting on something
    'feed', # Feeding animals
    'cross', # People crossing over things
    'kick', # Kicking objects with foot - usually not related to any task
    "inspect_(check,_look,_examine,_view)", # Check on or look at something, e.g., look at something - doesn't imply a state change
    "adjust_(regulate,_increase/reduce,_change)", # Adjust something (slight position change)
    "turn_(spin,_rotate,_flip,_turn_over)", # Turn something (slight position change)
    "tilt", # Tilt something (slight position change)
    "operate_(use,_dial,_click-button)", # Pressing buttons or touch screen
    "shake", # Shake an object
    "swing", # Swing on swings
    "wave", # Wave hands
    "blow", # Blowing air
    "serve", # Serving food
    "stand", # Standing up
    "count", # Counting things
    "clap", # Clapping hands
    "move_(transfer,_pass,_exchange)", # Move an object
    "arrange_(straighten,_sort,_distribute,_align)", # Arrange objects (non-specific movement)
    "shuffle", # Shuffle objects, e.g., cards (non-specific movement)
] 

# Similarly, some verb noun pairs don't involve meaningful state changes (e.g., put hand)
EGO4D_IGNORE_VERB_NOUN_PAIRS = [
    ("put_(place,_leave,_drop)", "hand_(finger,_hand,_palm,_thumb)"), # Putting hand on something is the same as touching, which we also ignore
    # ("cut_(trim,_slice,_chop)", "grass"), # Cutting grass 
]

EGO4D_CORRUPTED_VIDEO_IDS = [
    "968139e2-987e-4615-a2d4-fa2e683bae8a",
    "9957a25a-8ef1-4538-a51e-f8c5ab8c2bc4"
]

C_REGEX = re.compile(r"^\#C\s+C", re.IGNORECASE)
EOS_REGEX = re.compile(r"\<\|eos\|\>$", re.IGNORECASE)
UNSURE_END_REGEX = re.compile(r"#unsure\.?$", re.IGNORECASE)
UNSURE_MIDDLE_REGEX = re.compile(r"#unsure", re.IGNORECASE)

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
FRAME_KEEP_FREQUENCY = float(config["data"]["ego4d"]["video_frame_keep_frequency"])
MIN_BRIGHTNESS = float(config["data"]["ego4d"]["video_frame_min_brightness"])

class DataCollatorForVideoSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if all("pixel_values" in feature for feature in features):
            pixel_values = torch.stack(
                [feature.pop("pixel_values") for feature in features]
            )
        else:
            # in some cases, we don't have pixel values, e.g.,
            # in-context learning evaluation
            pixel_values = None
        collated = super().__call__(features, return_tensors=return_tensors)
        if pixel_values is not None:
            collated["pixel_values"] = pixel_values
        return collated


class DataCollatorForInterleavedVideoSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        pixel_values = torch.cat(
            [feature.pop("pixel_values") for feature in features]
            if "pixel_values" in features[0].keys()
            else None,
        )
        video_input_masks = (
            [feature.pop("video_input_mask") for feature in features]
            if "video_input_mask" in features[0].keys()
            else None
        )
        collated = super().__call__(features, return_tensors=return_tensors)
        if video_input_masks is not None:
            max_input_id_len = collated["input_ids"].size(1)
            padded_video_input_masks = []
            for video_input_mask in video_input_masks:
                remainder = torch.tensor(
                    [0] * (max_input_id_len - len(video_input_mask))
                )
                if self.tokenizer.padding_side == "right":
                    padded_video_input_masks.append(
                        torch.cat([video_input_mask, remainder])
                    )
                else:
                    padded_video_input_masks.append(
                        torch.cat([remainder, video_input_mask])
                    )
            collated["video_input_mask"] = torch.stack(padded_video_input_masks)
        if pixel_values is not None:
            collated["pixel_values"] = pixel_values
        return collated


def clean_narration_text(narration_text: str) -> str:
    # strip it first
    cleaned = narration_text.strip()

    # replace "#C C" with "The camera wearer"
    cleaned = re.sub(C_REGEX, "Someone", cleaned).strip()

    # remove <|eos|>
    cleaned = re.sub(EOS_REGEX, "", cleaned).strip()

    # remove #unsure from the end
    cleaned = re.sub(UNSURE_END_REGEX, "", cleaned).strip()

    # replace #unsure in the middle with "something"
    cleaned = re.sub(UNSURE_MIDDLE_REGEX, "something", cleaned)

    if len(cleaned) == 0:
        return cleaned

    # if cleaned doesn't end with a punctuation, append a period
    if not cleaned[-1] in string.punctuation:
        cleaned += "."

    return cleaned


def generate_input_ids_and_labels(
    tokenizer: PreTrainedTokenizer, prompt: str, text: str, decoder_only_lm: bool
) -> BatchEncoding:
    """Generate input ids and labels from the given prompt and text. If
    decoder_only_lm is True, the input and label texts are the same, but label
    tokens that correspond to the prompt are masked with -100. If
    decoder_only_lm is False, the input corresponds to the prompt and the label
    to the text.

    :param tokenizer: tokenizer for tokenizing inputs and label
    :param prompt: prompt for the LLM
    :param text: text for the LLM to generate based on the prompt
    :param decoder_only_lm: whether the LLM is decoder only or not
    :returns: preprocessed results
    """
    if decoder_only_lm:
        # tokenize prompt first
        prompt_tokens = tokenizer(prompt, return_attention_mask=False).input_ids

        # tokenize the narration and append eos
        preprocessed = tokenizer(
            " " + text,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        preprocessed["input_ids"].append(tokenizer.eos_token_id)

        # join tokenized prompt and narration text
        preprocessed["input_ids"] = prompt_tokens + preprocessed["input_ids"]
        preprocessed["input_ids"] = torch.tensor(preprocessed.input_ids)

        # for decoder only LMs, labels are same as input_ids, but we mask
        # tokens for the prompt
        preprocessed["labels"] = preprocessed["input_ids"].clone()
        preprocessed["labels"][: len(prompt_tokens)] = -100
    else:
        # eos is automatically appended by the tokenizer
        # we don't use return_tensors='pt' here b/c it automatically batchifies things
        # which we don't want
        preprocessed = tokenizer(prompt, return_attention_mask=False)
        preprocessed["input_ids"] = torch.tensor(preprocessed["input_ids"])
        preprocessed["labels"] = torch.tensor(
            tokenizer(text, return_attention_mask=False).input_ids
        )

    return preprocessed


def generate_input_ids_and_labels_from_interleaved(
    tokenizer: PreTrainedTokenizer,
    prompts: list[tuple[str, int]],
    text: str | None,
    num_query_tokens: int,
    decoder_only_lm: bool,
) -> dict[str, torch.Tensor]:
    """Generate input ids and labels from the given interleaved video/text data
    point. `text_video_map` specifies which videos are the last preceding
    videos for a given text, and is used to generate `video_input_mask`.

    :param tokenizer: tokenizer for tokenizing inputs and label
    :param prompts: list of prompts, each with the number of videos
    :param text: optional text to be completed by LLM
    :param num_query_tokens: number of qformer query tokens
    :param decoder_only_lm: whether the LLM is decoder only or not
    :returns: preprocessed results including `input_ids`, `labels` and
        `video_input_mask`.
        `input_ids` is a tensor of shape (num_tokens),
        `labels` is a tensor of shape (num_tokens),
        `video_input_mask` is a tensor of shape (num_tokens)
    """
    input_ids: list[int] = []
    labels: list[int] = []
    video_input_mask: list[int] = []
    # NOTE: FLAN tokenizer treats all whitespaces the same
    newline_token_id = tokenizer("\n", add_special_tokens=False).input_ids[0]
    if decoder_only_lm:
        for i, (prompt, num_videos) in enumerate(prompts):
            # first take care of the video tokens
            for _ in range(num_videos):
                input_ids.extend(
                    [tokenizer.pad_token_id] * num_query_tokens + [newline_token_id]
                )
                labels.extend([-100] * (num_query_tokens + 1))
                video_input_mask.extend([1] * num_query_tokens + [0])
            if i == 0:
                # if first text, start with a bos token
                input_ids = [tokenizer.bos_token_id] + input_ids
                labels = [-100] + labels
                video_input_mask = [0] + video_input_mask
            if i != len(prompts) - 1:
                # if not last prompt, add newline
                prompt += "\n"
            prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            input_ids.extend(prompt_tokens)
            video_input_mask.extend([0] * len(prompt_tokens))
            labels.extend([-100] * len(prompt_tokens))
        if text is not None:
            # prepend a space to separate the text from the prompt
            text_tokens = tokenizer(
                " " + text + "\n", add_special_tokens=False
            ).input_ids + [tokenizer.eos_token_id]
            input_ids.extend(text_tokens)
            video_input_mask.extend([0] * len(text_tokens))
            labels.extend(text_tokens)
    else:
        for i, (prompt, num_videos) in enumerate(prompts):
            # first take care of the video tokens
            for _ in range(num_videos):
                input_ids.extend(
                    [tokenizer.pad_token_id] * num_query_tokens + [newline_token_id]
                )
                video_input_mask.extend([1] * num_query_tokens + [0])
            if i != len(prompts) - 1:
                # if not last prompt, add newline
                prompt += "\n"
            prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            if i == len(prompts) - 1:
                # if last prompt, add eos token
                prompt_tokens.append(tokenizer.eos_token_id)
            input_ids.extend(prompt_tokens)
            video_input_mask.extend([0] * len(prompt_tokens))
        if text is not None:
            labels.extend(tokenizer(text).input_ids)

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "video_input_mask": torch.tensor(video_input_mask),
    }


T = TypeVar("T")


def generate_chunks(list_to_chunk: list[T], chunk_size: int) -> Iterable[list[T]]:
    for i in range(0, len(list_to_chunk), chunk_size):
        yield list_to_chunk[i : i + chunk_size]


def parse_timestamp(timestamp: str) -> float:
    """Parse a timestamp of format hh:mm:ss.cc and return a float.

    :param timestamp: timestamp of format hh:mm:ss.cc
    :return: timestamp as a float
    """
    hours, minutes, seconds = timestamp.split(":")
    return float(hours) * 60 * 60 + float(minutes) * 60 + float(seconds)


class NarratedActionClipSampler(ClipSampler):
    def __init__(self, random: bool) -> None:
        """The vast majority of narrated actions are 8 seconds long, and none
        are longer.

        So let's just sample 8-second clips.

        :param random: whether to return random clips or not
        """
        super().__init__(8)
        self.random = random
        self.sample_clip_indices: list[int] | None = None

    def __call__(
        self,
        last_clip_time: float | Fraction,
        video_duration: float | Fraction,
        annotation: dict[str, Any],
    ) -> ClipInfo:
        """Draw a random clip for a narrated action.

        :param last_clip_time: unused
        :param video_duration: duration of the video
        :param annotation: narrated action data.
            See https://ego4d-data.org/docs/data/annotations-schemas/ for more details.
        """
        if self.sample_clip_indices is None:
            # pprint(annotation["narrated_actions"])
            # first time sampling from this video, so create a clip index list
            self.sample_clip_indices = list(range(len(annotation["narrated_actions"])))
            if self.random:
                # shuffle them if random
                random.shuffle(self.sample_clip_indices)

        clip_index = self.sample_clip_indices[self._current_clip_index]
        narrated_action = annotation["narrated_actions"][clip_index]
        self._current_clip_index += 1

        is_last_clip = False
        if self._current_clip_index == len(self.sample_clip_indices):
            is_last_clip = True

        # # Take pre and post frames to define the video clip
        # clip_start_sec = narrated_action['pre_frame'] / narrated_action['fps']
        # clip_end_sec = narrated_action['post_frame'] / narrated_action['fps']

        # sample a clip 8 seconds around narration_time_sec
        # if narration_time_sec is less than 4 seconds, we start from 0
        clip_start_sec = max(
            Fraction(narrated_action["narration_timestamp_sec"])
            - self._clip_duration / 2,
            0,
        )

        # add 8 seconds to clip_start_sec
        # if clip_end_sec goes over the video duration, adjust clip_start_sec
        clip_end_sec = clip_start_sec + self._clip_duration
        if clip_end_sec > video_duration:
            clip_end_sec = video_duration
            clip_start_sec = clip_end_sec - self._clip_duration

        if is_last_clip:
            self.reset()

        return ClipInfo(
            clip_start_sec,
            clip_end_sec,
            clip_index,
            0,
            is_last_clip,
        )

    def skip_to_index(self, idx: int):
        self._current_clip_index = idx

    def reset(self) -> None:
        self._current_clip_index = 0
        self.sample_clip_indices = None

def preprocess_actions(actions: list[dict]) -> list[dict]:
    # Add structured noun to action data
    for action_idx, action in enumerate(actions):
        action['structured_noun'] = get_structured_noun(action)
        
    # Record information about previous occurrences of actions
    for action_idx, action in enumerate(actions):
        structured_action = (action['structured_verb'], action['structured_noun'])
        action['previous_occurrences'] = sum([1 if structured_action == (previous_action['structured_verb'], previous_action['structured_noun']) else 0 for previous_action in actions[:action_idx - 1]])
        action['future_occurrences'] = sum([1 if structured_action == (future_action['structured_verb'], future_action['structured_noun']) else 0 for future_action in actions[action_idx + 1:]])

    # TODO: Combine repeated actions? Can look at initial data and see if this is needed
    # don't want to change number of actions here or will have to re-extract frames
    return actions

class MisalignSRLEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MisalignSRL):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
class MisalignSRL:
    def __init__(self, misalignsrl_path):
        '''
        Class to generate misaligned (negative) examples for Ego4D clips.
        '''
        self.type_name_col_name_map = {"MisalignSRL_V": "mismatch_verb", "MisalignSRL_ARG1": "mismatch_noun", "MisalignSRL_V_ARG1": "mismatch_verb_noun"} # human readable name -> column name in group_df
        
        self.misalignsrl = pd.read_parquet(misalignsrl_path)
        
    def to_dict(self):
        return {
            "misalignsrl": self.misalignsrl.to_dict(),
            "type_name_col_name_map": self.type_name_col_name_map
        }
        
    @classmethod
    def from_dict(cls, data):
        misalignsrl = pd.DataFrame.from_dict(data["misalignsrl"])
        type_name_col_name_map = data["type_name_col_name_map"]
        return cls(misalignsrl, type_name_col_name_map)
    
    def get_misaligned_samples(self, clip, random_seed, split_video_info, multi_frame=False):      
        mistake_example_meta_dict = {_: None for _ in self.type_name_col_name_map}
        
        video_uid_narration_timestamp_sec = clip["video_uid"] + "_" + str(clip["narration_timestamp_sec"])
        
        rows = self.misalignsrl[self.misalignsrl["video_uid_narration_timestamp_sec"] == video_uid_narration_timestamp_sec]
        if len(rows) == 0:
            print(f"NO MISALIGNSRL SAMPLE ANNOTATED") # there are about (50554) / 155369 = 0.3253802239 narration clips in fho_main couldn't annotate misalignsrl (e.g., some narrations doesn't have ARG1. "#C C paints on the floor with her right hand")
            return mistake_example_meta_dict
        else:
            row = rows.sample(1, random_state=random_seed)
            
        for misalignsrl_type in self.type_name_col_name_map:
            misalignsrl_index_list = row[misalignsrl_type].iloc[0]
            if len(misalignsrl_index_list) == 0: # only about 150 / 47443 = 0.003161688763 samples doesn't have misaligned V or ARG1. just skip
                continue
            
            max_sample_size = 1 
            sampled_index_list = np.random.choice(misalignsrl_index_list, min(max_sample_size, len(misalignsrl_index_list)), replace=False)
            sampled_example_list = []
            for sample_i in range(len(sampled_index_list)):
                sampled_example_list.append(self.misalignsrl.iloc[sampled_index_list[sample_i]].squeeze().to_dict())

            # Take the first sample that's within the correct data split and we can load the video for (but only try up to 5 times)
            for sample in sampled_example_list[:5]:  
                video_id = sample['video_uid']
                frame_time = sample['narration_timestamp_sec']
                
                # `effect_frame` contains the pixels for the misalignsrl sample. In this case, we need to read it given the `video_id` (another video) and `frame_time`. 
                video_path = os.path.join(EGO4D_VIDEO_PATH, video_id+".mp4")
                # mimic Ego4dFHOMainDataset.__iter__ to get the frame
                if video_path not in [tp[1] for tp in split_video_info]:
                    # the misalignsrl sample does not exists in this data split
                    continue
                try:
                    video_cap = get_video(video_path)
                except:
                    print(f"Warning: could not load video at {video_path}.")
                    continue

                # Sample single narration frame or 8 second clip around narration frame
                if not multi_frame:
                    sample['effect_frame'] = Image.fromarray(extract_frames(video_cap, [frame_time])[0])
                else:
                    # This follows SuccessVQA paper's approach for sampling positive example videos
                    post_start_time = max(frame_time - 4.0, 0.0)
                    post_end_time = min(frame_time + 4.0, sample['end_sec'])
                    post_frame_times = generate_float_series(post_start_time, post_end_time, 1 / FRAME_KEEP_FREQUENCY)
                    post_frames = extract_frames(video_cap, post_frame_times)
                    post_frame_times = [ftime for frame, ftime in zip(post_frames, post_frame_times) if frame is not None]
                    post_frames = [frame for frame in post_frames if frame is not None]

                    sample['effect_frames'] = [Image.fromarray(frame) for frame in post_frames]
                    sample['effect_frame_times'] = post_frame_times
                video_cap.release()

                # NOTE: although multiple misalignsrl samples are prepared in the index file, we only sample one for now since not sure how outer code wants to organize the structure of multiple samples for one misalignsrl_type   
                mistake_example_meta_dict[misalignsrl_type] = sample
                break
        
        return mistake_example_meta_dict
        
    def print_misalignsrl_sample_meta(self, misalignsrl_sample_dict, misalignsrl_type):
        print(f"{misalignsrl_type}: {misalignsrl_sample_dict['narration_text']} (video_uid: {misalignsrl_sample_dict['video_uid']}, narration_timestamp_sec: {misalignsrl_sample_dict['narration_timestamp_sec']}, start_sec: {misalignsrl_sample_dict['start_frame']}, end_sec: {misalignsrl_sample_dict['end_frame']})")        

    def get_clip_info_from_fho_main_index(self, video_index, big_clip_index, narration_clip_index):
        narration_clip_info = self.fho_main_json["videos"][video_index]["annotated_intervals"][big_clip_index]["narrated_actions"][narration_clip_index]
        
        return narration_clip_info
    
        # (need Peter's input.) for each misalignsrl_type, given the information of the narration clip in fho_main.json, we can load visual media (image). Such information of a narration clip should have been loaded the same way as one item (`clip`) in `ego4d` above. I.E., need to identify the index of this clip in `ego4d` so that the media can be loaded efficiently using exisiting code. 

def filter_action(action: dict[str, Any]) -> bool:
    """Return True if the given action should be used, False otherwise."""
    return (
        not action["is_rejected"]
        and action["is_valid_action"]
        and C_REGEX.match(action["narration_text"]) is not None
    )

def get_structured_noun(action: dict) -> str | None:
    if action["frames"] is None:
        return None
    for frame in action["frames"]:
        if frame["frame_type"] != "pnr_frame":
            # some actions don't have contact frames so use pnr_frame
            continue
        for box in frame["boxes"]:
            if (
                box["object_type"] == "object_of_change"
                and box["structured_noun"] is not None
            ):
                return box["structured_noun"]
    return None

class Ego4dFHOMainDataset:
    def __init__(
        self,
        annotation_path: str,
        split_path: str,
        video_dir_path: str,
        transform: Callable[[dict], Any] | None = None,
        random_clip: bool = False,
        multi_frame: bool = False,
        already_processed_videos: Optional[list[str]] = [],
        n_workers: Optional[int] = None,
        worker_index: Optional[int] = None,
    ) -> None:
        """
        :param annotation_path: path to the main annotation file, e.g., `fho_main.json`.
        :param split_path: path to video split file generated by
            `scripts/split_train_val_test.py`.
        :param video_path: path to video dir
        :param transform: optional transform function
        :param random_clip: whether to sample clips randomly
        """
        self.multi_frame = multi_frame

        with open(annotation_path) as f:
            annotations = json.load(f)

        # create a dict video_uid => video
        video_dict = {video["video_uid"]: video for video in annotations["videos"]}

        with open(split_path) as f:
            split_data = json.load(f)

        self.split = split_data["split"]

        # Count number of narrated actions for __len__
        if len(already_processed_videos) == 0:
            self.num_narrated_actions = sum(split_data["videos"].values())
        else:
            # If some videos were already processed, exclude them from count of narrated actions
            self.num_narrated_actions = sum([v for k, v in split_data["videos"].items() if k not in already_processed_videos])

        if n_workers is not None and worker_index is not None:
            # If parallelizing, only count the narrated actions for this worker
            self.num_narrated_actions = len(split_list_into_partitions(list(range(self.num_narrated_actions)), n_workers)[worker_index])
        
        self.video_info = [
                (
                    video_index, 
                    os.path.join(video_dir_path, video_uid + ".mp4"),
                    {
                        "narrated_actions": [
                            {
                                "pre_frame": action["critical_frames"]["pre_frame"] if action["critical_frames"] is not None else None,
                                "pnr_frame": action["critical_frames"]["pnr_frame"] if action["critical_frames"] is not None else None,
                                "post_frame": action["critical_frames"]["post_frame"] if action["critical_frames"] is not None else None,
                                "fps": video_dict[video_uid]["video_metadata"]["fps"],
                                "video_duration": video_dict[video_uid]["video_metadata"]["duration_sec"],
                                "narration_text": action["narration_text"],
                                "structured_verb": action["structured_verb"],
                                "structured_noun": action["structured_noun"],
                                "previous_occurrences": action["previous_occurrences"],
                                "future_occurrences": action["future_occurrences"],
                                "narration_timestamp_sec": action["narration_timestamp_sec"],
                            }
                            for interval in video_dict[video_uid]["annotated_intervals"]
                            for action in preprocess_actions(interval["narrated_actions"])
                            if filter_action(action)
                        ],
                        "video_uid": video_uid,
                        "video_metadata": video_dict[video_uid]["video_metadata"],
                    },
                )
                for video_index, video_uid in enumerate(split_data["videos"])
            ]
        
        def _extract_video_id(data: tuple[str, dict[str, Any]]) -> str:
            return data[2]['video_uid']
        self.video_sampler = partial(ResumableParallelSequentialSampler, 
                                     completed_elements=already_processed_videos,
                                     element_id_fn=_extract_video_id,
                                     n_workers=n_workers,
                                     worker_index=worker_index,
                                     return_indices=False)

    def __iter__(self) -> Iterable[dict[str, Any]]:
        for video_index, video_path, video_metadata in self.video_sampler(self.video_info):
            try:
                try:
                    video_cap = get_video(video_path)
                except:
                    print(f"Warning: could not load video at {video_path}.")
                    continue

                for clip_index, clip_info in enumerate(video_metadata['narrated_actions']):
                    if not(clip_info['post_frame'] is not None and clip_info['pre_frame'] is not None and clip_info['narration_timestamp_sec'] is not None):
                        # Missing the annotations we need to select frames
                        continue

                    pre_time = clip_info['pre_frame'] / clip_info['fps'] if clip_info['pre_frame'] is not None else None
                    pnr_time = clip_info['pnr_frame'] / clip_info['fps'] if clip_info['pnr_frame'] is not None else None
                    post_time = clip_info['post_frame'] / clip_info['fps'] if clip_info['post_frame'] is not None else None

                    if not self.multi_frame:

                        # Add extra logic to extract a few extra candidate frames around pre_time, pnr_time, and post_time and pick the least blurry ones
                        pre_frame0, pre_frame1, pre_frame2, pre_frame3, pre_frame4, pnr_frame, post_frame0, post_frame1, post_frame2, post_frame3, post_frame4 = extract_frames(video_cap, [pre_time-0.05, pre_time-0.025, pre_time, pre_time+0.025, pre_time+0.05, pnr_time, 
                                                                                                                                                                                            post_time-0.05, post_time-0.025, post_time, post_time+0.025, post_time+0.05])
                        pre_frame = max([pre_frame0, pre_frame1, pre_frame2, pre_frame3, pre_frame4], key=lambda x: variance_of_laplacian(x))
                        post_frame = max([post_frame0, post_frame1, post_frame2, post_frame3, post_frame4], key=lambda x: variance_of_laplacian(x))

                        # If video clip is too dark, omit it
                        average_brightness = np.mean([np.mean(np.asarray(frame)) for frame in [pre_frame, pnr_frame, post_frame]])
                        if average_brightness < MIN_BRIGHTNESS:
                            continue

                        yield clip_info | {"video_index": video_index,
                                        "video_uid": video_metadata["video_uid"],
                                        "clip_index": clip_index,
                                        "pre_time": pre_time,
                                        "pre_frame": pre_frame, 
                                        "pnr_time": pnr_time, 
                                        "pnr_frame": pnr_frame,
                                        "post_time": post_time,
                                        "post_frame": post_frame}
                    else:
                        # Sample a precondition and effect video clip instead (precondition clip is used to generate a negative example)
                        # Following SuccessVQA, sample positive example as an up to 8 second clip centered around narration timestamp (make sure it doesn't overlap with other clips)
                        narration_time = clip_info['narration_timestamp_sec']
                        post_start_time = max(narration_time - 4.0, pre_time)
                        post_end_time = min(narration_time + 4.0, post_time)
                        post_frame_times = generate_float_series(post_start_time, post_end_time, 1 / FRAME_KEEP_FREQUENCY)
                        post_frames = extract_frames(video_cap, post_frame_times)

                        # Following SuccessVQA, sample negative example as an 8 second clip ending at the precondition frame
                        # (but make sure it doesn't overlap with other clips)
                        pre_end_time = min(pre_time, post_start_time - 1 / FRAME_KEEP_FREQUENCY)
                        pre_end_time = max(pre_end_time, 0.0)
                        pre_start_time = max(pre_end_time - 8.0, 0.0)
                        pre_frame_times = generate_float_series(pre_start_time, pre_end_time, 1 / FRAME_KEEP_FREQUENCY)
                        if len(pre_frame_times) > 0 and pre_start_time != pre_end_time:
                            pre_frames = extract_frames(video_cap, pre_frame_times)
                        else:
                            pre_frames = []

                        # Remove any frames that failed to load
                        pre_frame_times = [ftime for frame, ftime in zip(pre_frames, pre_frame_times) if frame is not None]
                        pre_frames = [frame for frame in pre_frames if frame is not None]
                        
                        # Remove any frames that failed to load
                        post_frame_times = [ftime for frame, ftime in zip(post_frames, post_frame_times) if frame is not None]
                        post_frames = [frame for frame in post_frames if frame is not None]

                        # If video clip is too dark, omit it
                        average_brightness = np.mean([np.mean(np.asarray(frame)) for frame in pre_frames + post_frames])
                        if average_brightness < MIN_BRIGHTNESS:
                            continue

                        yield clip_info | {"video_index": video_index,
                                        "video_uid": video_metadata["video_uid"],
                                        "clip_index": clip_index,
                                        "pre_times": pre_frame_times,
                                        "pre_frames": pre_frames, 
                                        "post_times": post_frame_times,
                                        "post_frames": post_frames}
            finally:
                video_cap.release()
    
    def __len__(self) -> int:
        return self.num_narrated_actions    
    
def filter_action_for_mistake_detection(action: dict[str, Any], previous_action: dict[str, Any]=None) -> bool:
    """Return True if the given action should be used, False otherwise."""
    return (
        action["pre_frame"] is not None
        # and action["pnr_frame"] is not None
        and action["post_frame"] is not None
        and action["structured_verb"] not in EGO4D_IGNORE_VERBS # Omit clips with non-task actions
        and (action["structured_verb"], action["structured_noun"]) not in EGO4D_IGNORE_VERB_NOUN_PAIRS
        and action['structured_noun'] is not None # need at least one target object
        and "#O" not in action["narration_text"] # Omit clips involving interacting with other people
        and "something" not in action['narration_text'] # Omit clips annotated with vague object reference "something"
        and (
            previous_action is None
            or not (
                (action['structured_verb'], action['structured_noun']) == (previous_action['structured_verb'], previous_action['structured_noun']))
                or action['previous_occurrences'] > 0
            ) # Filter out clips where the same action is being performed over and over
    )

class Ego4DMistakeDetectionDataset(MistakeDetectionDataset):
    """Class to store Ego4D data in mistake detection form. Each example only has one frame for efficiency, although we may have to change this later."""
    def __init__(self, 
                 data_split: str,
                 mismatch_augmentation: bool=False,
                 multi_frame: bool=False,
                 debug_n_examples_per_class: Optional[int]=None,
                 n_workers: Optional[int]=None,
                 worker_index: Optional[int]=None):
        """
        Method to initialize and load Ego4D dataset for mistake detection, following the SuccessVQA approach (https://proceedings.mlr.press/v232/du23b.html).

        :param data_split: String name for data partition to load.
        :param mismatch_augmentation: Whether to augment negative examples by selecting alternative video clips for narrated procedures with mismatched verbs and nouns. Set this to False to be comparable to SuccessVQA approach.
        :param debug_n_examples_per_class: Load a small number of examples for each class (success and mistake).
        :param n_workers: Number of workers to parallelize dataset generation over.
        """
        # Handle Ego4D-specific initialization logic
        self.mismatch_augmentation = mismatch_augmentation
        self.multi_frame = multi_frame
        super().__init__(data_split,
                         mismatch_augmentation=mismatch_augmentation,
                         multi_frame=multi_frame,
                         debug_n_examples_per_class=debug_n_examples_per_class,
                         n_workers=n_workers,
                         worker_index=worker_index)

    @staticmethod
    def get_cache_dir(data_split: str,
                      mismatch_augmentation: bool=False,
                      multi_frame: bool=False,
                      debug_n_examples_per_class: Optional[int]=None,
                      n_workers: Optional[int]=None,
                      worker_index: Optional[int]=None) -> str:
        cache_fname = f"ego4d_{data_split}_seed{RANDOM_SEED}"
        if mismatch_augmentation:
            cache_fname += "_mismatch"
        if multi_frame:
            cache_fname += "_multiframe"
        if debug_n_examples_per_class is not None:
            cache_fname += f"_debug{debug_n_examples_per_class}"
        if n_workers is not None:
            # If parallelizing, name folder differently
            cache_fname += f"_partition{worker_index+1}of{n_workers}"
        return os.path.join(DATA_CACHE_DIR, cache_fname)
    
    def ego4d_narration_to_instruction(self, narration_text: str, nlp: English) -> str:
        instruction_text = clean_narration_text(narration_text) # Replace symbols in narration text with words
        instruction_text = simple_present_to_imperative(nlp, instruction_text)
        for original_text, replaced_text in [("in your left hand", "in your hand"),
                                                ("in your right hand", "in your hand"),
                                                ("with your left hand", "with your hand"),
                                                ("with your right hand", "with your hand"),
                                                ("with both hands", "with your hands"),
                                                ("with left hand", "with your hand"),
                                                ("with right hand", "with your hand")]:
            if not("left hand" in instruction_text and "right hand" in instruction_text):
                # In Ego4D, it was often narrated which hands were being used for various actions; since our focus is the state changes of objects in these actions, we remove unneeded mentions of this
                instruction_text = instruction_text.replace(original_text, replaced_text)
        return instruction_text

    def generate_examples(self,
                          data_split: str,
                          mismatch_augmentation: bool=False,
                          multi_frame: bool=False,
                          debug_n_examples_per_class: Optional[int]=None,
                          n_workers: Optional[int]=None,
                          worker_index: Optional[int]=None) -> list[MistakeDetectionExample]:
        
        # Resume from partial generation (Ego4D is very large so we need this to avoid wasting time in generating the mistake detection data)
        already_processed_videos = get_subdirectories(self.cache_dir) # Each subdirectory of Ego4D's cache dir should be a video ID

        if mismatch_augmentation:
            mismatch_sampler = MisalignSRL(
                MISALIGNSRL_PATH,
            )        
        else:
            # If we already generated mismatch-augmented data, just use the data from there rather than re-generating
            mismatch_cache_dir = self.get_cache_dir(data_split, True, multi_frame=multi_frame, debug_n_examples_per_class=debug_n_examples_per_class)
            if os.path.exists(mismatch_cache_dir):
                mismatch_example_dirs = json.load(open(os.path.join(mismatch_cache_dir, "dataset.json"), "r"))["example_dirs"]
                self.example_dirs = [d for d in mismatch_example_dirs if "MisalignSRL" not in d]
                self.n_examples += len(self.example_dirs)
                return

        # If we're only loading a small amount of data for debugging purpose but we've loaded the full data before, just borrow the data from the full dataset
        if debug_n_examples_per_class is not None:
            full_cache_dir = self.get_cache_dir(data_split, mismatch_augmentation=mismatch_augmentation, multi_frame=multi_frame, debug_n_examples_per_class=None)
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

        ego4d = Ego4dFHOMainDataset(
            EGO4D_ANNOTATION_PATH,
            EGO4D_SPLIT_PATHS[data_split],
            EGO4D_VIDEO_PATH,
            random_clip=True if debug_n_examples_per_class is not None else False,
            multi_frame=multi_frame,
            already_processed_videos=already_processed_videos,
            n_workers=n_workers,
            worker_index=worker_index,
        )

        print(f"({worker_index}) Found {len(already_processed_videos)} processed videos, still need to process {len(ego4d)} clips.")

        nlp = spacy.load('en_core_web_lg')
        SIMILARITY_THRESHOLD = 0.95

        # Prepare list to hold examples ready for caching
        example_cache_buffer = []
        for clip in tqdm(ego4d, desc=f"({worker_index}) generating ego4d data"):
            # Cache if we're starting a new video
            if clip['clip_index'] == 0 and len(example_cache_buffer) > 0:  
                # Cache examples in buffer
                for new_example in example_cache_buffer:
                    self.save_example_to_file(new_example)
                self.save_dataset_metadata()
                del example_cache_buffer
                example_cache_buffer: list[MistakeDetectionExample] = []

            # Check if video is one of the ones we've noticed seems to have corrupted, distorted frames
            if clip['video_uid'] in EGO4D_CORRUPTED_VIDEO_IDS:
                continue

            # Index procedure based on video and clip index (each narration is unique)
            procedure_id = 1000 * clip['video_index'] + clip['clip_index']
            clip_id = f"{clip['video_uid']}/{clip['clip_index']}"

            # Skip some actions based on conditions specified in filter_action_for_mistake_detection
            if not filter_action_for_mistake_detection(clip):
                continue

            # Convert narration text to imperative form to match the sentence structure of recipes and task instructions
            instruction_text = self.ego4d_narration_to_instruction(clip['narration_text'], nlp)
            
            # For some verbs, e.g., "move", we need to have more than one object argument in order for
            # there to be an observable state change (e.g., "move tomatoes into bowl" rather than "move tomatoes")
            if clip['structured_verb'] in ["move_(transfer,_pass,_exchange)"]:
                mentioned_objects = TargetObjectCounterFilter.parse_sentences_for_target_objects(nlp, [instruction_text])[0]
                # TODO: this still may not be enough, e.g., "reposition the water color on the table with your hand" has enough mentioned objects but not in the right places
                if len(mentioned_objects) < 2:
                    continue

            # clip['video'] shape: (C, # frames, H, W)
            if not multi_frame:
                precondition_frame_arr, effect_frame_arr = clip['pre_frame'], clip['post_frame']
                precondition_frame, effect_frame = Image.fromarray(precondition_frame_arr), Image.fromarray(effect_frame_arr)

                # Omit examples where precondition and effect frame are overly similar
                precondition_effect_similarity = dot(precondition_frame_arr.astype(float).flatten(), effect_frame_arr.astype(float).flatten()) / (norm(precondition_frame_arr.astype(float).flatten()) * norm(effect_frame_arr.astype(float).flatten()))
                if precondition_effect_similarity >= SIMILARITY_THRESHOLD:
                    continue
            else:
                precondition_frames = clip['pre_frames']
                effect_frames = clip['post_frames']

                # Omit examples where precondition and effect clips' final frames are overly similar
                if len(precondition_frames) > 0 and len(effect_frames) > 0:
                    precondition_effect_similarity = dot(precondition_frames[-1].astype(float).flatten(), effect_frames[-1].astype(float).flatten()) / (norm(precondition_frames[-1].astype(float).flatten()) * norm(effect_frames[-1].astype(float).flatten()))
                    if precondition_effect_similarity >= SIMILARITY_THRESHOLD:
                        continue

                precondition_frames = [Image.fromarray(frame) for frame in precondition_frames]
                effect_frames = [Image.fromarray(frame) for frame in effect_frames]
            
            # Generate examples from this clip
            # NOTE: example IDs intentionally have "/"s in them to ensure there's one directory per video and per clip (enables easy resuming of incomplete runs and inspection of data)

            # Generate positive example from effect frame
            frames = [effect_frame] if not multi_frame else effect_frames
            if len(frames) > 0:
                positive_example = MistakeDetectionExample(
                    task_name="ego4d",
                    video_id=clip['video_uid'],
                    procedure_id=procedure_id,
                    example_id=f"{clip_id}/pos",
                    frames=frames,
                    frame_times=[clip['post_time']] if not multi_frame else clip['post_times'],
                    procedure_description=instruction_text,
                    mistake=False,
                    verb_noun_pair=(clip["structured_verb"], clip["structured_noun"])
                )
                example_cache_buffer.append(positive_example)
            
            # Generate hard negative example from precondition frame 
            # (only do this if precondition and effect clips have enough separation and action is not super fast, as even slight annotation error can impact data quality)
            if (not multi_frame and clip['post_time'] - clip['pre_time'] >= 2.0) or (multi_frame and max(clip['post_times']) - min(clip['post_times']) >= 2.0):
                frames = [precondition_frame] if not multi_frame else precondition_frames
                if len(frames) > 0:
                    negative_example_hard = MistakeDetectionExample(
                        task_name="ego4d",
                        video_id=clip['video_uid'],
                        procedure_id=procedure_id,
                        example_id=f"{clip_id}/hardneg",
                        frames=frames,
                        frame_times=[clip['pre_time']] if not multi_frame else clip['pre_times'],
                        procedure_description=instruction_text,
                        mistake=True,
                        mistake_type="Action Incomplete",
                        verb_noun_pair=(clip["structured_verb"], clip["structured_noun"])
                    )
                    example_cache_buffer.append(negative_example_hard)
            else:
                print(f"({worker_index}) Warning: Could not generate a hard negative for clip {clip_id}!")
            
            # Generate extra negative examples by finding video clips with the same verb but not noun and vice-versa
            if mismatch_augmentation:
                mismatch_examples = mismatch_sampler.get_misaligned_samples(clip=clip, random_seed=RANDOM_SEED * procedure_id, split_video_info=ego4d.video_info, multi_frame=multi_frame)
                
                # print(f"=======MisalignSRL samples =========")
                # print(f"current clip narration: {clip['narration_text']} (video_uid: {clip['video_uid']}, narration_timestamp_sec: {clip['narration_timestamp_sec']})")
                for misalignsrl_type in mismatch_sampler.type_name_col_name_map.keys():
                    # Skip if no misaligned sample for this type is found
                    if mismatch_examples[misalignsrl_type] is None:
                        continue

                    # Omit V examples for pick & place verbs, as these verbs are often prerequisite for 
                    # other object interactions and will happen even in misaligned video
                    if misalignsrl_type == "MisalignSRL_V" and clip["structured_verb"] in ["carry",
                                                                                           "take_(pick,_grab,_get)", 
                                                                                           "put_(place,_leave,_drop)", 
                                                                                           "hold_(support,_grip,_grasp)",
                                                                                           "pull",
                                                                                           "push",
                                                                                           "remove"]:
                        continue

                    # Ensure that whatever we matched on is in line with the structured verb/noun annotation for the clip
                    # (in rare cases, e.g., "rub paint onto wall", the structured annotation, i.e., paint, will not match 
                    # the actual verb in the annotation, causing incorrect matches)
                    
                    video_id = mismatch_examples[misalignsrl_type]['video_uid']
                    frame_time = mismatch_examples[misalignsrl_type]['narration_timestamp_sec']
                    
                    # `procedure_id` is meant to be an ID for the narration.
                    procedure_id = procedure_id
                    
                    # Generate positive example from effect frame
                    frames = [mismatch_examples[misalignsrl_type]['effect_frame']] if not multi_frame else mismatch_examples[misalignsrl_type]['effect_frames']
                    if len(frames) > 0:
                        misalignsrl_example = MistakeDetectionExample(
                            task_name="ego4d",
                            video_id=video_id,
                            procedure_id=procedure_id,
                            example_id=f"{clip_id}/easyneg_{misalignsrl_type}_{video_id}_{frame_time}",
                            frames=frames,
                            frame_times=[frame_time] if not multi_frame else mismatch_examples[misalignsrl_type]['effect_frame_times'],
                            procedure_description=instruction_text,
                            mistake=True,
                            mistake_type=misalignsrl_type,
                            verb_noun_pair=(clip["structured_verb"], clip["structured_noun"])
                        )
                        example_cache_buffer.append(misalignsrl_example)
                    # print(f"Appended example:")
                    # mismatch_sampler.print_misalignsrl_sample_meta(mismatch_examples[misalignsrl_type], misalignsrl_type)
                # print(f"===================================")
            if debug_n_examples_per_class is not None and self.n_examples + len(example_cache_buffer) >= 2 * debug_n_examples_per_class:
                break

        # Cache any last examples in buffer
        for new_example in example_cache_buffer:
            self.save_example_to_file(new_example)
        self.save_dataset_metadata()
        del example_cache_buffer
        example_cache_buffer: list[MistakeDetectionExample] = []

def combine_ego4d_partitions(datasets: list[Ego4DMistakeDetectionDataset], mismatch_augmentation: bool=False, multi_frame: bool=False, debug_n_examples_per_class: int=None) -> Ego4DMistakeDetectionDataset:
    """
    Combines Ego4DMistakeDetectionDataset partitions generated from parallelization. Make sure all data partitions have been copletely generated before calling this.

    :param datasets: List of datasets.
    """
    # Check that appropriate partitions were passed in
    assert len(datasets) > 1, "Need at least 2 datasets to combine!"
    assert all(dataset.data_generated for dataset in datasets), "All dataset partitions must be finished generating before combining!"
    cache_dirs = [dataset.cache_dir for dataset in datasets]
    assert all("_partition" in cache_dir for cache_dir in cache_dirs), "You should only combine dataset partitions that were generated through the parallelization implemented in Ego4DMistakeDetectionDataset class."
    assert len(set([dataset.data_split for dataset in datasets])) == 1, "All partitions should be from the same split (e.g., train, val, test)."

    # Combine dataset metadata and save in new directory
    new_dataset_metadata = {
        "example_dirs": [],
        "n_examples": 0,
        "data_generated": True,
    }
    for dataset in datasets:
        new_dataset_metadata["example_dirs"] += dataset.example_dirs
        new_dataset_metadata["n_examples"] += dataset.n_examples
    new_cache_dir = cache_dirs[0].split("_partition")[0]
    new_dataset_metadata["cache_dir"] = new_cache_dir
    assert not os.path.exists(new_cache_dir), "Cache dir for combined dataset already exists. Please delete."
    os.makedirs(new_cache_dir)
    json.dump(new_dataset_metadata, 
              open(os.path.join(new_cache_dir, "dataset.json"), "w"))
    return Ego4DMistakeDetectionDataset(datasets[0].data_split,
                                        mismatch_augmentation=mismatch_augmentation,
                                        multi_frame=multi_frame,
                                        debug_n_examples_per_class=debug_n_examples_per_class)