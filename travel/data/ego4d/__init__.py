# Most code in this file was adapted from EILEV: https://github.com/yukw777/EILEV

import json
import os
from collections.abc import Callable, Iterable
from fractions import Fraction
from pytorchvideo.data import ClipSampler, LabeledVideoDataset
from pytorchvideo.data.clip_sampling import ClipInfo
import random
import re
import string
import torch
from transformers import BatchEncoding, DataCollatorForSeq2Seq, PreTrainedTokenizer
from typing import Any, TypeVar

# Some verbs would not be expected in the task-oriented mistake detection setting, so we can filter them out
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

        # Take pre and post frames to define the video clip
        clip_start_sec = narrated_action['pre_frame'] / narrated_action['fps']
        clip_end_sec = narrated_action['post_frame'] / narrated_action['fps']

        if is_last_clip:
            self.reset()

        return ClipInfo(
            clip_start_sec,
            clip_end_sec,
            clip_index,
            0,
            is_last_clip,
        )

    def reset(self) -> None:
        self._current_clip_index = 0
        self.sample_clip_indices = None

def filter_action(action: dict[str, Any], previous_action: dict[str, Any]=None) -> bool:
    """Return True if the given action should be used, False otherwise."""
    return (
        not action["is_rejected"]
        and action["is_valid_action"]
        and action["critical_frames"] is not None
        and action["structured_verb"] not in EGO4D_IGNORE_VERBS # Omit clips with non-task actions
        and "#O" not in action["narration_text"] # Omit clips involving interacting with other people
        and (previous_action is None or not ((action['structured_verb'], action['structured_noun']) == (previous_action['structured_verb'], previous_action['structured_noun']))) # Filter out clips where the same action is being performed over and over
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

# TODO: adapt below into a preprocessing method for actions - don't just add structures, but also combine repetitive actions
def add_structured_nouns(actions: list[dict]) -> list[dict]:
    for action_idx, action in enumerate(actions):
        action['structured_noun'] = get_structured_noun(action)
        
    for action_idx, action in enumerate(actions):
        structured_action = (action['structured_verb'], action['structured_noun'])
        action['previous_occurrences'] = sum([1 if structured_action == (previous_action['structured_verb'], previous_action['structured_noun']) else 0 for previous_action in actions[:action_idx - 1]])
        action['future_occurrences'] = sum([1 if structured_action == (future_action['structured_verb'], future_action['structured_noun']) else 0 for future_action in actions[action_idx + 1:]])
    return actions


class Ego4dFHOMainDataset(LabeledVideoDataset):
    """Class to store data from Ego4D. Some domain-specific filtering steps are performed for egocentric mistake detection."""
    def __init__(
        self,
        annotation_path: str,
        split_path: str,
        video_dir_path: str,
        transform: Callable[[dict], Any] | None = None,
        random_clip: bool = False,
    ) -> None:
        """
        :param annotation_path: path to the main annotation file, e.g., `fho_main.json`.
        :param split_path: path to video split file generated by
            `scripts/split_train_val_test.py`.
        :param video_path: path to video dir
        :param transform: optional transform function
        :param random_clip: whether to sample clips randomly
        """
        with open(annotation_path) as f:
            annotations = json.load(f)

        # create a dict video_uid => video
        video_dict = {video["video_uid"]: video for video in annotations["videos"]}

        with open(split_path) as f:
            split_data = json.load(f)

        self.split = split_data["split"]
        self.num_narrated_actions = sum(split_data["videos"].values())

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

        super().__init__(
            [
                (
                    os.path.join(video_dir_path, video_uid + ".mp4"),
                    {
                        "narrated_actions": [
                            {
                                "pre_45": action["critical_frames"]["pre_45"],
                                "pre_30": action["critical_frames"]["pre_30"],
                                "pre_15": action["critical_frames"]["pre_15"],
                                "pre_frame": action["critical_frames"]["pre_frame"],
                                "pnr_frame": action["critical_frames"]["pnr_frame"],
                                "post_frame": action["critical_frames"]["post_frame"],
                                "fps": video_dict[video_uid]["video_metadata"]["fps"],
                                "narration_text": action["narration_text"],
                                "structured_verb": action["structured_verb"],
                                "structured_noun": action["structured_noun"],
                                "previous_occurrences": action["previous_occurrences"],
                                "future_occurrences": action["future_occurrences"]
                            }
                            for interval in video_dict[video_uid]["annotated_intervals"]
                            for action_idx, action in enumerate(add_structured_nouns(interval["narrated_actions"]))
                            if filter_action(action, previous_action=interval["narrated_actions"][action_idx - 1] if action_idx > 0 else None)
                        ],
                        "video_uid": video_uid,
                        "video_metadata": video_dict[video_uid]["video_metadata"],
                    },
                )
                for video_uid in split_data["videos"]
            ],
            NarratedActionClipSampler(random_clip),
            transform=_transform,
            decode_audio=False,
        )

    def __len__(self) -> int:
        return self.num_narrated_actions