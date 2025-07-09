from collections import defaultdict
import json
import numpy as np
import os
import pickle
from PIL import Image
from pprint import pprint
import shelve
from spacy.lang.en import English
import torch
from tqdm import tqdm
import shelve
from transformers import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from typing import Optional

from travel.constants import CACHE_FREQUENCY
from travel.data.utils.image import resize_with_aspect, CACHED_FRAME_DIMENSION
from travel.model.grounding import VisualFilterTypes

def run_vqa(vlm: PreTrainedModel, 
            processor: ProcessorMixin,
            prompts: list[str],
            frames: list[Image.Image],
            batch_size: int=1,
            cache_path: Optional[str]=None,
            return_attention: bool=False,
            is_encoder_decoder: bool=False,
            ignore_frames: bool=False,) -> torch.FloatTensor:
    """
    Runs VQA for given prompts and frames in batches with a given VLM and its processor.

    :param vlm: VLM for conditional generation from `transformers`.
    :param process: VLM processor from `transformers`, including tokenizer and image processor.
    :param prompts: List of prompts including visual questions.
    :param frames: List of images to ask visual questions about.
    :param batch_size: Batch size for running inference.
    :param cache_path: .pt file to cache incomplete logits in.
    :param return_attention: Whether to return attentions for passed prompts in addition to logits.
    :return: Full tensor of logits output from each question. The process of mapping this into VQAOutputs instances requires task/process-specific information, so it should be done outside of this method.
    """
    assert len(prompts) == len(frames), "Need same number of prompts and frames to run VQA!"

    # Run VQA in batches
    if not is_encoder_decoder:
        if getattr(vlm, "vocab_size", None):
            vocab_size = vlm.vocab_size
        else:
            vocab_size = max(processor.tokenizer.vocab_size, max([k for k in processor.tokenizer.added_tokens_decoder.keys()]))
    else:
        vocab_size = 32128 # This is a hack to handle InstructBLIP based on FlanT5

    logits = torch.zeros((0, vocab_size)).float()
    if cache_path is not None:
        assert cache_path.endswith(".pt"), "Cache path should be .pt to store logits tensor!"
        if os.path.exists(cache_path):
            try:
                logits = torch.load(cache_path)
            except:
                pass
        else:
            if not os.path.exists("/".join(cache_path.split("/")[:-1])):
                os.makedirs("/".join(cache_path.split("/")[:-1]))

    last_save = 0
    with torch.inference_mode():
        # Start at logits.shape[0] so we don't rerun any logits that were already cached (resuming logic)
        for i in tqdm(range(logits.shape[0], len(frames), batch_size), desc=f"running VQA ({str(vlm.device)})"):
            # Prepare the batch
            batch_frames = frames[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]

            if is_encoder_decoder:
                batch_prompts = [p.replace("A: ", "") for p in batch_prompts]

            # Run through VLM to get logits
            if "phi3" not in str(type(vlm)):
                inputs = processor(text=batch_prompts, images=batch_frames, padding=True, return_tensors="pt")
            else:
                # Phi 3 processor does not support multiple texts and images together, have to process separately
                text_inputs = processor(batch_prompts, padding=True, return_tensors="pt")
                image_inputs = processor.image_processor(batch_frames, return_tensors="pt")
                inputs = text_inputs | image_inputs
                if 'num_img_tokens' in inputs:
                    del inputs['num_img_tokens']

            inputs = inputs.to(vlm.device)
            if is_encoder_decoder:
                # For encoder-decoder, move "A:" part of prompt to decoder input IDs
                inputs['decoder_input_ids'] = processor.tokenizer([f"{processor.tokenizer.pad_token} A: "] * len(batch_prompts), return_tensors="pt")['input_ids'].to(vlm.device) # NOTE: this only works for encoder-decoder models whose decoder_start_token is the pad token

            if not ignore_frames:
                outputs = vlm(**inputs)
            else:
                outputs = vlm.language_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            this_logits = outputs.logits
            inputs = inputs.to('cpu')
            this_logits = this_logits[:, -1].detach().cpu()
            logits = torch.cat([logits, this_logits], dim=0)
            del this_logits
            del inputs

            # Cache logits so far
            if cache_path is not None and i - last_save >= CACHE_FREQUENCY:
                torch.save(logits, cache_path)
                last_save = i

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cache one more time
    if cache_path is not None:
        torch.save(logits, cache_path) 

    return logits

def run_vqa_with_visual_filter(vlm_processor, vlm, batch_examples, batch_frames, prompts_a, new_questions, question_idx, batch_size, visual_filter=None, nlp=None, visual_filter_mode=None, frame_cache_dir=None, is_encoder_decoder=False, ignore_frames=False):
    """
    VQA and visual filter wrapper method for iterative VQA experiments.

    :param vlm_processor: VLM processor.
    :param vlm: VLM.
    :param batch_examples: Batch of MistakeDetectionExample.
    :param batch_frames: Batch of frames (PIL images).
    :param prompts_a: Full string prompts to get a yes/no answer.
    :param new_questions: Last generated questions to use with text-conditioned visual filters.
    :param question_idx: Index or identifier of the current question. This is only used to save modified frames from the visual filter.
    :param batch_size: Batch size for VQA.
    :param visual_filter: Optional training free visual filter to modify images and possibly run VQA twice.
    :param nlp: spaCy NLP pipeline.
    :param visual_filter_mode: Type of visual filter.
    :param frame_cache_dir: Directory to cache visual filter modified frames for later inspection. If not passed, will not save any frames.
    :return: Logits from VQA.
    """
    # Apply visual filter to frames for VQA
    if visual_filter:
        if visual_filter_mode == VisualFilterTypes.Contrastive_Region:
            batch_frames_filtered = visual_filter(nlp, batch_frames, new_questions)
        elif visual_filter_mode == VisualFilterTypes.Visual_Contrastive:
            batch_frames_filtered = visual_filter(batch_frames)
        elif visual_filter_mode in [VisualFilterTypes.Spatial_NoRephrase, VisualFilterTypes.Spatial_Blur]:
            batch_frames_filtered, _ = visual_filter(nlp, batch_frames, new_questions, return_visible_target_objects=False)
        elif visual_filter_mode == VisualFilterTypes.Spatial:
            batch_frames_filtered, new_questions = visual_filter(nlp, batch_frames, new_questions, return_visible_target_objects=False)
        elif visual_filter_mode == VisualFilterTypes.AGLA:
            batch_frames_filtered = visual_filter(batch_frames, new_questions)

    # Cache paths to frames (if using a visual filter, save filtered frames and cache paths to them)
    if not(visual_filter is None or frame_cache_dir is None):
        for batch_sub_idx, (frame, example) in enumerate(zip(batch_frames_filtered, batch_examples)):
            this_frame_cache_dir = os.path.join(frame_cache_dir, f"vqa_frames/{example.example_id}")
            if not os.path.exists(this_frame_cache_dir):
                os.makedirs(this_frame_cache_dir)
            frame_path = os.path.join(this_frame_cache_dir, f"frame_q{question_idx}.jpg")
            resized_frame = resize_with_aspect(frame, CACHED_FRAME_DIMENSION)
            resized_frame.save(frame_path)

    # Run VQA on base image (yes/no)
    if not (visual_filter and visual_filter_mode in [VisualFilterTypes.Spatial_NoRephrase, VisualFilterTypes.Spatial_Blur]):
        new_answers_logits = run_vqa(vlm, vlm_processor, prompts_a, batch_frames, batch_size=batch_size, is_encoder_decoder=is_encoder_decoder, ignore_frames=ignore_frames)
    else:
        # Spatial filter doesn't need original image logits, so don't get them for efficiency
        new_answers_logits = None

    # Run VQA on filtered image if needed and combine logits as proposed in approaches' papers
    if visual_filter:
        new_answers_logits_filtered = run_vqa(vlm, vlm_processor, prompts_a, batch_frames_filtered, batch_size=batch_size, is_encoder_decoder=is_encoder_decoder, ignore_frames=ignore_frames)
        new_answers_logits = visual_filter.combine_logits(new_answers_logits, new_answers_logits_filtered)

    return new_answers_logits
