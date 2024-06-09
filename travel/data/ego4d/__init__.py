# Most code in this file was adapted from EILEV: https://github.com/yukw777/EILEV
import ast
from collections.abc import Callable, Iterable
from fractions import Fraction
from functools import partial
import json
import numpy as np
import os
import pandas as pd
from pprint import pprint
from pytorchvideo.data import ClipSampler, LabeledVideoDataset
from pytorchvideo.data.clip_sampling import ClipInfo
import random
import re
import spacy
import string
import time
import torch
from torch.nn.functional import cosine_similarity
from torchvision.transforms.functional import to_pil_image
from transformers import BatchEncoding, DataCollatorForSeq2Seq, PreTrainedTokenizer
from typing import Any, TypeVar, Optional
from tqdm import tqdm

from travel.constants import DATA_CACHE_DIR, RANDOM_SEED, CACHE_FREQUENCY
from travel.data.ego4d.constants import EGO4D_ANNOTATION_PATH, EGO4D_SPLIT_PATHS, EGO4D_VIDEO_PATH, \
                                        EGO4D_MISMATCH_FHO2SRL_PATH, EGO4D_MISMATCH_NARRATIONS_PATH, EGO4D_MISMATCH_NARRATIONS_ROWS_PATH, EGO4D_MISMATCH_GROUPS_PATH, EGO4D_MISMATCH_COUNT, MISALIGNSRL_PATH
from travel.data.mistake_detection import MistakeDetectionExample, MistakeDetectionDataset
from travel.data.utils import read_large_csv, get_subdirectories, split_list_into_partitions, ResumableParallelSequentialSampler
from travel.data.utils.text import simple_present_to_imperative

# Some verbs would not be expected in the task-oriented mistake detection setting (don't really cause a meaningful state change toward a task goal), so we can filter them out
EGO4D_IGNORE_VERBS = ['scroll', # Scrolling on phone, tablet, etc.
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
                     'kick'] # Kicking objects with foot - usually not related to any task

C_REGEX = re.compile(r"^\#C\s+C", re.IGNORECASE)
EOS_REGEX = re.compile(r"\<\|eos\|\>$", re.IGNORECASE)
UNSURE_END_REGEX = re.compile(r"#unsure\.?$", re.IGNORECASE)
UNSURE_MIDDLE_REGEX = re.compile(r"#unsure", re.IGNORECASE)


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

    # TODO: this only can be used for resuming within a single clip - need to rethink how resuming is done
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
        return super().default(obj)
    
class MisalignSRL:
    def __init__(self, fho_main_path, narration_mapping_fho2srl_df_path, narration_df_path, fho_narration_df_rows_path, group_df_path, misalignsrl_path):
        '''
        fho_main_path: the path to json file (fho_main) with structure
            {"videos": 
                [{"annotated_intervals": [{"clip_uid": str, ... , 
                                            "narrated_actions": [{"narration_text": str, 
                                                                "narration_timestamp_sec"}, ...]
                                            }, ...
                ], 
                "video_metadata": {}, 
                "video_uid": str}]},
            "date": xx, 
            "description": xx, 
            "metadata": xx}
        
        narration_mapping_fho2srl_df_path: the path to csv file with columns ["video_uid", "narration_text", "narration_timestamp_sec", "srl_index", "fho_index"]
        
        narration_df_path: the path to csv file with example row:
            video_uid                 26202090-684d-4be8-b3cc-de04da827e91
            video_dur                                          3127.233333
            narration_source                              narration_pass_1
            narration_ind                                                7
            narration_time                                         40.8375
            clip_start                                           40.583986
            clip_end                                             41.090933
            clip_text                C takes his hand out of the paper bag
            tag_verb                                                  [93]
            tag_noun                                        [321, 12, 349]
            ARG0                                                         C
            V                                                        takes
            ARG1                                                  his hand
            valid_tag_noun                                           [321]
            valid_txt_noun                                        ['hand']
            valid_tag_verb                                            [93]
            valid_txt_verb                                        ['take']
            index                                                   322348
            valid_txt_verb_single                                     take
            valid_txt_noun_single                                     hand        
        
        fho_narration_df_rows_path: the path to json file with structure: a list of dict. An example dict:
                    {
                      'video_index': 1, # the index of the video in the fho_main.json
                      'interval_index': 0, # the index of the interval in the fho_main.json
                      'action_index': 3, # the index of the action in the fho_main.json                      
                      'narration_timestamp_sec': 35.4317396,
                      'video_uid': '26202090-684d-4be8-b3cc-de04da827e91',
                      'narration_text': '#C C takes a steel bowl out of the '
                                        'paper bag',}
        
        group_df_path: the path to csv/parquet file with columns ["txt_verb"(str), "txt_noun"(str), "narration_index"(list), "narration_indices"(list), "mismatch_noun"(list), "mismatch_verb"(list), "mismatch_verb_noun"(list)]. 
        '''
        
        # print("Loading fho_main_json ...")
        # start_time = time.time()
        # # file_path = "/home/yayuanli/fun/mistake_detection/fine_grained_action_mistake_detection/dataset/fho_main.json"
        # with open(fho_main_path, "r") as file:
        #     fho_main_json = json.load(file)
        # print(f"Loading fho_main.json took {time.time() - start_time} seconds.")
        
        # print("Loading narration_df ...")
        # start_time = time.time()
        # # narration_df = "/z/home/yayuanli/dat/Ego4D_Mistake/v1/egoclip_narrations_exploed_groupby_no,txt.csv"
        # narration_df = pd.read_csv(narration_df_path, index_col=0)        
        # print(f"Loading narration_df took {time.time() - start_time} seconds.")
        
        # print("Loading narration_mapping_fho2srl_df ...")  
        # start_time = time.time()
        # # 'narration_mapping_fho2srl_df.csv'
        # narration_mapping_fho2srl_df = pd.read_csv(narration_mapping_fho2srl_df_path, index_col=0)
        # narration_mapping_fho2srl_df["srl_index"] = narration_mapping_fho2srl_df["srl_index"].apply(ast.literal_eval)
        # print(f"Loading narration_mapping_fho2srl_df took {time.time() - start_time} seconds.")
        # fho_main_overlap_narration_list = narration_mapping_fho2srl_df["video_uid"].unique().tolist()
        
        # print("Loading fho_narration_df_rows ...")
        # start_time = time.time()
        # # fho_narration_df_rows.json
        # with open(fho_narration_df_rows_path, 'r') as f:
        #     fho_narration_df_rows = json.load(f)
        # print(f"Loading fho_narration_df_rows took {time.time() - start_time} seconds.")
        
        # print("Loading group_df ...")
        # start_time = time.time()
        # # f'/z/home/yayuanli/dat/Ego4D_Mistake/v1/egoclip_groups_groupby_no,txt.csv'
        # group_df = read_large_csv(group_df_path, 
        #                         #   columns_str2list=["narration_index", "narration_indices", "mismatch_noun", "mismatch_verb", "mismatch_verb_noun"], 
        #                         nrows=None, # None
        #                         )
        # print(f"Loading group_df took {time.time() - start_time} seconds.")

        # self.fho_main_json = fho_main_json
        # self.narration_mapping_fho2srl_df = narration_mapping_fho2srl_df
        # self.narration_df = narration_df
        # self.fho_narration_df_rows = fho_narration_df_rows
        # self.group_df = group_df
        # self.fho_main_overlap_narration_list = fho_main_overlap_narration_list

        self.type_name_col_name_map = {"MisalignSRL_V": "mismatch_verb", "MisalignSRL_ARG1": "mismatch_noun", "MisalignSRL_V_ARG1": "mismatch_verb_noun"} # human readable name -> column name in group_df
        
        # "/home/yayuanli/fun/mistake_detection/fine_grained_action_mistake_detection/dataset/ego4d_fho_main/misalignsrl.parquet"
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
    
    def get_misaligned_samples(self, clip):
        print(f"=======MisalignSRL samples =========")
        print(f"current clip narration: {clip['narration_text']} (video_uid: {clip['video_uid']}, narration_timestamp_sec: {clip['narration_timestamp_sec']})")
        
        mistake_example_meta_dict = {_: None for _ in self.type_name_col_name_map}
        
        video_uid_narration_timestamp_sec = clip["video_uid"] + "_" + str(clip["narration_timestamp_sec"])
        
        rows = self.misalignsrl[self.misalignsrl["video_uid_narration_timestamp_sec"] == video_uid_narration_timestamp_sec]
        if len(rows) == 0:
            print(f"NO MISALIGNSRL SAMPLE ANNOTATED") # there are about (50554) / 155369 = 0.3253802239 narration clips in fho_main couldn't annotate misalignsrl (e.g., some narrations doesn't have ARG1. "#C C paints on the floor with her right hand")
            return mistake_example_meta_dict
        else:
            row = rows.sample(1)
            
        for misalignsrl_type in self.type_name_col_name_map:
            if row[misalignsrl_type].iloc[0] == -1: # only about 150 / 47443 = 0.003161688763 samples doesn't have misaligned V or ARG1. just skip
                continue
            mistake_example_meta_dict[misalignsrl_type] = self.misalignsrl.iloc[row[misalignsrl_type]].squeeze().to_dict()
            print(f"{misalignsrl_type}: {mistake_example_meta_dict[misalignsrl_type]['narration_text']} (video_uid: {mistake_example_meta_dict[misalignsrl_type]['video_uid']}, narration_timestamp_sec: {mistake_example_meta_dict[misalignsrl_type]['narration_timestamp_sec']}, start_frame: {mistake_example_meta_dict[misalignsrl_type]['start_frame']}, end_frame: {mistake_example_meta_dict[misalignsrl_type]['end_frame']})")
        # print: 

        # for misalignsrl_type in self.type_name_col_name_map:
        #     print(f"{misalignsrl_type}: {mistake_example_meta_dict[misalignsrl_type]['narration_text']} (video_uid: {mistake_example_meta_dict[misalignsrl_type]['video_uid']}, narration_timestamp_sec: {mistake_example_meta_dict[misalignsrl_type]['narration_timestamp_sec']}, start_frame: {mistake_example_meta_dict[misalignsrl_type]['start_frame']}, end_frame: {mistake_example_meta_dict[misalignsrl_type]['end_frame']})")
            
        print(f"====================================")
        
        
        return mistake_example_meta_dict
        

    # TODO: make sure examples are only retrieved from the same split...
    def DEP_get_misaligned_samples(self, clip):
        '''
        clip: (obj?). Corresponds to one action clip in fho_main.json. The distinguishing information of this clip is `video_uid` and `narration_timestamp_sec`.
        
        return:
            mistake_example_meta_dict: {misalignsrl_type -> one action clip in fho_main.json (fho_main_json["videos"][video_index]["annotated_intervals"][big_clip_index]["narrated_actions"][narration_clip_index])}
        '''
        # for each misalignsrl_type, sample one srl_index in the group, and return the index of that narration clip in `fho_main.json`
        # mistake_example_meta_dict: misalignsrl_type -> fho_narration_df_rows. To be returned.
        mistake_example_meta_dict = {_: None for _ in self.type_name_col_name_map}
        
        video_uid = clip["video_uid"]
        narration_timestamp_sec = clip["narration_timestamp_sec"]
        
        # find the map_row (the row in `narration_mapping_fho2srl_df (pd.DataFrame)`) by matching `video_uid` and `narration_timestamp_sec`
        match_video_uid = self.narration_mapping_fho2srl_df["video_uid"] == video_uid
        # pprint(video_uid)
        # pprint(match_video_uid.value_counts())
        # pprint(self.narration_mapping_fho2srl_df["video_uid"])
        match_narration_timestamp_sec = self.narration_mapping_fho2srl_df["narration_timestamp_sec"] == narration_timestamp_sec
        # pprint(narration_timestamp_sec)
        # pprint(match_narration_timestamp_sec.value_counts())
        # pprint(self.narration_mapping_fho2srl_df["narration_timestamp_sec"])
        map_row = self.narration_mapping_fho2srl_df[match_video_uid & match_narration_timestamp_sec]
        
        # return if no map_row found. Meaning, this clip does not have misalignsrl sample in current group_df.
        if len(map_row) == 0:
            # TODO: we always return here because we can't match the video_uid or narration_timestamp_sec
            print(f"WARNING MISTAKE EXAMPLE NOT FOUND")
            return mistake_example_meta_dict
        
        # get the `fho_index` (index in `fho_narration_df_rows`) and `srl_index` (index in `narration_df`) from the map_row (a row in `narration_mapping_fho2srl_df (pd.DataFrame)`)
        fho_index = map_row["fho_index"].values[0]
        
        srl_index = random.choice(map_row["srl_index"].values[0]) # randomly pick one. one narration could be in multiple groups. E.g., "pick up a bag of clothes" could be in "pick up bag" and "pick up cloth"
        
        # find the group in `group_df` to which the `srl_narration_row` belongs
        srl_narration_row = self.narration_df.iloc[srl_index]
        match_txt_verb = self.group_df["txt_verb"] == srl_narration_row["valid_txt_verb_single"]
        match_txt_noun = self.group_df["txt_noun"] == srl_narration_row["valid_txt_noun_single"]
        target_group = self.group_df[match_txt_verb & match_txt_noun] # a row in `group_df` (pd.Series)
        
        # fill in the mistake_example_meta_dict   
        for misalignsrl_type in self.type_name_col_name_map:
            # index_list: list of int. the srl index in the srl narration df.
            index_list = target_group[self.type_name_col_name_map[misalignsrl_type]].iloc[0] 
            # TODO: this row can be optmized by saving group_df as parquet file instead of csv
            if not isinstance(index_list, list):
                index_list = ast.literal_eval(index_list)
                
            # shuffle the index_list so that different given `clip` is less likely to get the same sample for a misalignsrl_type. For example, for given `clip`s "cut carrot" and "pour sauce", the first sample for "MisalignSRL_VN" could be the same -- "pick up cap"
            index_list = np.random.permutation(index_list)
            # find fho_index (for fho_narration_df_rows)        
            for srl_index in index_list:
                # narration_mapping_fho2srl_df["srl_index"] is a column where each row is a list of int. srl_index is a int. 
                # Find the row where the srl_index is in the list of the row narration_mapping_fho2srl_df["srl_index"]
                # it may not be since current group_df is made from `egoclip.csv` (refer to: https://github.com/facebookresearch/EgoVLPv2), which contains narration clips over whole Ego4D while the fho_main.json is a subset of Ego4D (with bbox annotation).
                match_misalign_sample_in_fho_main = self.narration_mapping_fho2srl_df["srl_index"].apply(lambda x: srl_index in x)
                fho_index_row = self.narration_mapping_fho2srl_df[match_misalign_sample_in_fho_main] 
                if len(fho_index_row) > 0:
                    fho_index = fho_index_row["fho_index"].values[0]
                    mistake_example_meta_dict[misalignsrl_type] = self.fho_narration_df_rows[fho_index]
                    break

        # return mistake_example_meta_dict. For each misalignsrl_type, if None, it means no sample is found in the group. (This could be improved making group_df.parquet from fho_main.json instead of from egoclip)
        return mistake_example_meta_dict    
        

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

# TODO: add an option to not load full videos - just use index files and specific frames instead? This would be much faster.
class Ego4dFHOMainDataset(LabeledVideoDataset):
    def __init__(
        self,
        annotation_path: str,
        split_path: str,
        video_dir_path: str,
        transform: Callable[[dict], Any] | None = None,
        random_clip: bool = False,
        already_processed_videos: Optional[list[str]] = [],
        n_workers: Optional[int] = None,
        worker_index: Optional[int] = None,
        valid_video_uid_list=None
    ) -> None:
        """
        :param annotation_path: path to the main annotation file, e.g., `fho_main.json`.
        :param split_path: path to video split file generated by
            `scripts/split_train_val_test.py`.
        :param video_path: path to video dir
        :param transform: optional transform function
        :param random_clip: whether to sample clips randomly
        :param valid_video_uid_list: list of video_uids out of which the videos will NOT be used. If None, all videos will be used.
        """
        with open(annotation_path) as f:
            annotations = json.load(f)

        # create a dict video_uid => video
        video_dict = {video["video_uid"]: video for video in annotations["videos"]}

        with open(split_path) as f:
            split_data = json.load(f)

        self.split = split_data["split"]

        # Count number of narrated actions for __len__
        if len(already_processed_videos) == 0:
            self.num_narrated_actions = sum(split_data["videos"].values()) # TODO: adjust to account for parallelism and resuming
        else:
            # If some videos were already processed, exclude them from count of narrated actions
            self.num_narrated_actions = sum([v for k, v in split_data["videos"].items() if k not in already_processed_videos])

        if n_workers is not None and worker_index is not None:
            # If parallelizing, only count the narrated actions for this worker
            self.num_narrated_actions = len(split_list_into_partitions(list(range(self.num_narrated_actions)), n_workers)[worker_index])

        def _transform(item: dict) -> Any:
            """The first transform function that formats `narrated_actions` and
            `video`."""
            # format narrated_actions
            narrated_actions = item.pop("narrated_actions")
            item.update(narrated_actions[item["clip_index"]])

            # turn video tensor to torch.uint8
            item["video"] = item["video"].to(torch.uint8)
            if transform is not None:
                item = transform(item)
            return item
        
        def _extract_video_id(data: tuple[str, dict[str, Any]]) -> str:
            return data[1]['video_uid']

        
        full_video_list = [
                (
                    os.path.join(video_dir_path, video_uid + ".mp4"),
                    {
                        "narrated_actions": [
                            {
                                "pre_45": action["critical_frames"]["pre_45"] if action["critical_frames"] is not None else None,
                                "pre_30": action["critical_frames"]["pre_30"] if action["critical_frames"] is not None else None,
                                "pre_15": action["critical_frames"]["pre_15"] if action["critical_frames"] is not None else None,
                                "pre_frame": action["critical_frames"]["pre_frame"] if action["critical_frames"] is not None else None,
                                "pnr_frame": action["critical_frames"]["pnr_frame"] if action["critical_frames"] is not None else None,
                                "post_frame": action["critical_frames"]["post_frame"] if action["critical_frames"] is not None else None,
                                "fps": video_dict[video_uid]["video_metadata"]["fps"],
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
                for video_uid in split_data["videos"]
            ]
        
        init_video_list = None
        if valid_video_uid_list != None:
            # given a variable `valid_video_uid_list`, filter out the videos not in the list. The initial motivation was that when generating the misalignsrl samples, the index files of misalignmentsrl(EgoClip) and fho_main(this project) are only partially overlapped            
            init_video_list = []
            for video_tuple in full_video_list:
                if video_tuple[1]["video_uid"] in valid_video_uid_list:
                    init_video_list.append(video_tuple)
        else:
            init_video_list = full_video_list
            
        super().__init__(
            init_video_list,
            NarratedActionClipSampler(random_clip),
            video_sampler=partial(ResumableParallelSequentialSampler, 
                                  completed_elements=already_processed_videos,
                                  element_id_fn=_extract_video_id,
                                  n_workers=n_workers,
                                  worker_index=worker_index),
            transform=_transform,
            decode_audio=False,
        )

    def __len__(self) -> int:
        return self.num_narrated_actions    
    
def filter_action_for_mistake_detection(action: dict[str, Any], previous_action: dict[str, Any]=None) -> bool:
    """Return True if the given action should be used, False otherwise."""
    return (
        action["pre_frame"] is not None
        and action["pnr_frame"] is not None
        and action["post_frame"] is not None
        and action["structured_verb"] not in EGO4D_IGNORE_VERBS # Omit clips with non-task actions
        and "#O" not in action["narration_text"] # Omit clips involving interacting with other people
        and (
            previous_action is None
            or not (
                (action['structured_verb'], action['structured_noun']) == (previous_action['structured_verb'], previous_action['structured_noun']))
                or action['previous_occurrences'] > 1
            ) # Filter out clips where the same action is being performed over and over
    )


class Ego4DMistakeDetectionDataset(MistakeDetectionDataset):
    """Class to store Ego4D data in mistake detection form. Each example only has one frame for efficiency, although we may have to change this later."""
    def __init__(self, 
                 data_split: str,
                 mismatch_augmentation: bool=False,
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
        if mismatch_augmentation:
            # TODO: this takes several minutes - if we already have the data cached, we don't even need this - add logic for this
            # TODO: also, need to make sure random seeds for mismatch_sampler are appropriately configured if we're generating in parallel
            self.mismatch_sampler = MisalignSRL(
                EGO4D_ANNOTATION_PATH,
                EGO4D_MISMATCH_FHO2SRL_PATH,
                EGO4D_MISMATCH_NARRATIONS_PATH,
                EGO4D_MISMATCH_NARRATIONS_ROWS_PATH,
                EGO4D_MISMATCH_GROUPS_PATH,
                MISALIGNSRL_PATH,
            )
        else:
            self.mismatch_sampler = None
        super().__init__(data_split,
                         mismatch_augmentation=mismatch_augmentation,
                         debug_n_examples_per_class=debug_n_examples_per_class,
                         n_workers=n_workers,
                         worker_index=worker_index)

    def get_cache_dir(self, 
                      data_split: str,
                      mismatch_augmentation: bool=False,
                      debug_n_examples_per_class: Optional[int]=None,
                      n_workers: Optional[int]=None,
                      worker_index: Optional[int]=None) -> str:
        cache_fname = f"ego4d_{data_split}_seed{RANDOM_SEED}"
        if mismatch_augmentation:
            cache_fname += f"_mismatch{EGO4D_MISMATCH_COUNT}"
        if debug_n_examples_per_class is not None:
            cache_fname += f"_debug{debug_n_examples_per_class}"
        if n_workers is not None:
            # If parallelizing, name folder differently
            cache_fname += f"_partition{worker_index+1}of{n_workers}"
        return os.path.join(DATA_CACHE_DIR, cache_fname)
    
    # TODO: support parallelization by iterating through different parts of ego4D for each worker?
    def generate_examples(self,
                          data_split: str,
                          mismatch_augmentation: bool=False,
                          debug_n_examples_per_class: Optional[int]=None,
                          n_workers: Optional[int]=None,
                          worker_index: Optional[int]=None) -> list[MistakeDetectionExample]:
        
        # Resume from partial generation (Ego4D is very large so we need this to avoid wasting time in generating the mistake detection data)
        already_processed_videos = get_subdirectories(self.cache_dir) # Each subdirectory of Ego4D's cache dir should be a video ID

        ego4d = Ego4dFHOMainDataset(
            EGO4D_ANNOTATION_PATH,
            EGO4D_SPLIT_PATHS[data_split],
            EGO4D_VIDEO_PATH,
            random_clip=True if debug_n_examples_per_class is not None else False,
            already_processed_videos=already_processed_videos,
            n_workers=n_workers,
            worker_index=worker_index,
            # valid_video_uid_list=self.mismatch_sampler.fho_main_overlap_narration_list
        )

        nlp = spacy.load('en_core_web_sm')
        SIMILARITY_THRESHOLD = 0.95

        # Prepare list to hold examples ready for caching
        example_cache_buffer = []
        for clip in tqdm(ego4d, desc="generating ego4d data"):
            # Cache if we're starting a new video
            if ego4d._clip_sampler._current_clip_index == 0 and len(example_cache_buffer) > 0:  
                # Cache examples in buffer
                for new_example in example_cache_buffer:
                    self.save_example_to_file(new_example)
                self.save_dataset_metadata(MisalignSRLEncoder=MisalignSRLEncoder)
                del example_cache_buffer
                example_cache_buffer: list[MistakeDetectionExample] = []

            # Index procedure based on video and clip index (each narration is unique)
            procedure_id = 1000 * clip['video_index'] + clip['clip_index']
            clip_id = f"{clip['video_uid']}/{clip['clip_index']}"

            # Skip some actions based on conditions specified in filter_action_for_mistake_detection
            if not filter_action_for_mistake_detection(clip):
                continue
            
            # Convert narration text to imperative form to match the sentence structure of recipes and task instructions    
            instruction_text = clean_narration_text(clip['narration_text']) # Replace symbols in narration text with words
            instruction_text = simple_present_to_imperative(nlp, instruction_text)

            # clip['video'] shape: (C, # frames, H, W)
            precondition_frame_t, effect_frame_t = clip['video'][:,0], clip['video'][:,-1] # (C, H, W)
            precondition_frame, effect_frame = to_pil_image(precondition_frame_t), to_pil_image(effect_frame_t)
            
            # Omit examples where precondition and effect frame are overly similar
            precondition_effect_similarity = cosine_similarity(precondition_frame_t.flatten().float(), effect_frame_t.flatten().float(), dim=0).detach().numpy()
            if precondition_effect_similarity >= SIMILARITY_THRESHOLD:
                continue
            
            # Generate examples from this clip
            # NOTE: example IDs intentionally have "/"s in them to ensure there's one directory per video and per clip (enables easy resuming of incomplete runs, and easy inspection of data)

            # Generate positive example from effect frame
            # TODO: maybe want to get entire clip later - OR just write new code to extract single frames from video files
            positive_example = MistakeDetectionExample(
                task_name="ego4d",
                video_id=clip['video_uid'],
                procedure_id=procedure_id,
                example_id=f"{clip_id}/pos",
                frames=[effect_frame],
                frame_times=[clip['post_frame'] / clip['fps']],
                procedure_description=instruction_text,
                mistake=False,
            )
            example_cache_buffer.append(positive_example)
            
            # Generate hard negative example from precondition frame (only if this action didn't previously occur too many times)
            # if clip['previous_occurrences'] < 2:
            negative_example_hard = MistakeDetectionExample(
                task_name="ego4d",
                video_id=clip['video_uid'],
                procedure_id=procedure_id,
                example_id=f"{clip_id}/hardneg",
                frames=[precondition_frame],
                frame_times=[clip['pre_frame'] / clip['fps']],
                procedure_description=instruction_text,
                mistake=True,
                mistake_type="Action Incomplete",
            )
            example_cache_buffer.append(negative_example_hard)
            
            # Generate extra negative examples by finding video clips with the same verb but not noun and vice-versa
            if mismatch_augmentation:
                # NOTE: this doesn't do anything yet because nothing is ever returned in `mismatch_examples`
                try: 
                    mismatch_examples = self.mismatch_sampler.get_misaligned_samples(clip=clip)
                    # pprint(clip.keys())
                    # print("==========")
                    # pprint(f"{clip['narration_text']=}\n{mismatch_examples=}")
                    # see which video_uids are matched
                    # for fhouid in [_[1]["video_uid"] for _ in ego4d._labeled_videos]:
                    #     inflag = fhouid in self.mismatch_sampler.narration_mapping_fho2srl_df["video_uid"].unique().tolist()
                    #     print(f"{fhouid}, {inflag=}")
                except Exception as e:
                    # if the error is AttributeError: 'Ego4DMistakeDetectionDataset' object has no attribute 'mismatch_sampler', reinitialize the self.mismatch_sampler
                    # It is weird to have this error. It means self.mismatch_sampler disappears in the middle of the loop.
                    if "mismatch_sampler" not in dir(self):
                        self.mismatch_sampler = MisalignSRL(
                            EGO4D_ANNOTATION_PATH,
                            EGO4D_MISMATCH_FHO2SRL_PATH,
                            EGO4D_MISMATCH_NARRATIONS_PATH,
                            EGO4D_MISMATCH_NARRATIONS_ROWS_PATH,
                            EGO4D_MISMATCH_GROUPS_PATH,
                            MISALIGNSRL_PATH
                        )
            if debug_n_examples_per_class is not None and self.n_examples + len(example_cache_buffer) >= 2 * debug_n_examples_per_class:
                break

        # Cache any last examples in buffer
        for new_example in example_cache_buffer:
            self.save_example_to_file(new_example)
        self.save_dataset_metadata()
        del example_cache_buffer
        example_cache_buffer: list[MistakeDetectionExample] = []

def combine_ego4d_partitions(datasets: list[Ego4DMistakeDetectionDataset], debug_n_examples_per_class: int=None) -> Ego4DMistakeDetectionDataset:
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
    new_dataset_metadata["mismatch_sampler"] = None
    assert not os.path.exists(new_cache_dir), "Cache dir for combined dataset already exists. Please delete."
    os.makedirs(new_cache_dir)
    json.dump(new_dataset_metadata, 
              open(os.path.join(new_cache_dir, "dataset.json"), "w"))
    return Ego4DMistakeDetectionDataset(datasets[0].data_split,
                                        debug_n_examples_per_class=debug_n_examples_per_class)