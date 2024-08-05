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
from transformers import PreTrainedModel, Blip2ForConditionalGeneration, InstructBlipForConditionalGeneration, Kosmos2ForConditionalGeneration, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration
from transformers.processing_utils import ProcessorMixin
from typing import Optional, Callable

from travel.constants import CACHE_FREQUENCY, IMAGES_CHUNK_SIZE
from travel.data.mistake_detection import MistakeDetectionDataset, MistakeDetectionExample
from travel.data.utils.image import resize_with_aspect, CACHED_FRAME_DIMENSION
from travel.data.vqa import VQAResponse, VQAOutputs, get_vqa_response_token_ids, COMPLETION_PROMPT_TEMPLATES
from travel.model.grounding import VisualFilterTypes, AdaptiveVisualFilter
from travel.model.mistake_detection import DETECTION_FRAMES_PROPORTION

def run_vqa(vlm: PreTrainedModel, 
            processor: ProcessorMixin,
            prompts: list[str],
            frames: list[Image.Image],
            batch_size: int=1,
            cache_path: Optional[str]=None) -> torch.FloatTensor:
    """
    Runs VQA for given prompts and frames in batches with a given VLM and its processor.

    :param vlm: VLM for conditional generation from `transformers`.
    :param process: VLM processor from `transformers`, including tokenizer and image processor.
    :param prompts: List of prompts including visual questions.
    :param frames: List of images to ask visual questions about.
    :param batch_size: Batch size for running inference.
    :param cache_path: .pt file to cache incomplete logits in.
    :return: Full tensor of logits output from each question. The process of mapping this into VQAOutputs instances requires task/process-specific information, so it should be done outside of this method.
    """
    assert len(prompts) == len(frames), "Need same number of prompts and frames to run VQA!"

    # Run VQA in batches
    logits = torch.zeros((0, vlm.vocab_size)).float()
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
    with torch.no_grad():
        # Start at logits.shape[0] so we don't rerun any logits that were already cached (resuming logic)
        for i in tqdm(range(logits.shape[0], len(frames), batch_size), desc=f"running VQA ({str(vlm.device)})"):
            # Prepare the batch
            batch_frames = frames[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]

            # Run through VLM to get logits
            inputs = processor(text=batch_prompts, images=batch_frames, padding=True, return_tensors="pt")
            inputs = inputs.to(vlm.device)
            this_logits = vlm(**inputs).logits
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

def clean_up_captions(captions: list[str],
                    model_type: type,
                    processor: ProcessorMixin):
    """
    Remove unwanted tokens at the beginning of captions, used in run_captioning.

    :param captions: List of captions generated
    :model_type: Type of model used to generate captions
    :param  processor: VLM processor from `transformers`, including post processing methods
    """

    if (model_type == LlavaForConditionalGeneration):
        return [caption.split("ASSISTANT: ")[1].strip() for caption in captions]
    elif (model_type == LlavaNextForConditionalGeneration):
        return [caption.split("[/INST] ")[1].strip() for caption in captions]
    elif (model_type == Kosmos2ForConditionalGeneration):
        return [processor.post_process_generation(caption)[0] for caption in captions]
    elif (model_type == Blip2ForConditionalGeneration):
        return [caption.strip() for caption in captions]
    else:
        return captions


def run_captioning(vlm: PreTrainedModel,
                    processor: ProcessorMixin,
                    frames: list[Image.Image],
                    batch_size: int=1,
                    cache_path: Optional[str]=None) -> list[str]:
    """
    Get captions of given frames in batches with a given VLM and its processor.

    :param vlm: VLM for conditional generation from `transformers`.
    :param processor: VLM processor from `transformers`, including tokenizer and image processor.
    :param frames: List of images to caption.
    :param batch_size: Batch size for running inference.
    :param cache_path: Path to a file for caching captions without the file extension, ex: "/Path/to/file". In this path, there will be file.dir, file.dat, file.bak files for caching.  
    :return: List of captions corresponding to each given frame.
    """

    prompt_template = COMPLETION_PROMPT_TEMPLATES[type(vlm)]
    captions = []
    if cache_path is not None:
        # Load cached captions if there are any
        if os.path.exists(cache_path + '.dir'):
            with shelve.open(cache_path) as cache_file:
                try:
                    captions = cache_file["captions"]
                except:
                    pass

    last_save = len(captions)
    with torch.no_grad():
        # Start at len(captions) so we don't produce any captions that were already cached (resuming logic)
        for i in tqdm(range(len(captions), len(frames), batch_size), desc=f"running captioning ({str(vlm.device)})"):
            # Prepare the batch
            batch_frames = frames[i:i+batch_size]
            batch_prompts = [prompt_template for _ in range(len(batch_frames))]

            # Run through VLM to get captions
            inputs = processor(text=batch_prompts, images=batch_frames, padding=True, return_tensors="pt")
            inputs = inputs.to(vlm.device)

            outputs = vlm.generate(**inputs, max_new_tokens=128)
            outputs = processor.batch_decode(outputs, skip_special_tokens=True)
            outputs = clean_up_captions(outputs, type(vlm), processor)
            captions += outputs

            # Cache captions if needed
            if cache_path is not None and (i - last_save + 1) >= CACHE_FREQUENCY:
                with shelve.open(cache_path) as cache_file:
                    cache_file["captions"] = captions
                    cache_file.sync()
                last_save = i + 1

    # Last cache
    if cache_path is not None:
        with shelve.open(cache_path) as cache_file:
            cache_file["captions"] = captions
            cache_file.sync()

    return captions

def run_vqa_for_mistake_detection(eval_dataset: MistakeDetectionDataset,
                                  vlm: PreTrainedModel, 
                                  vlm_processor: ProcessorMixin,  
                                  generate_prompts: Callable[[MistakeDetectionExample], tuple[list[str], list[str], list[VQAResponse], list[Image.Image]]],
                                  n_prompts_per_frame: int,
                                  visual_filter_mode: Optional[VisualFilterTypes],
                                  visual_filter: Optional[AdaptiveVisualFilter],
                                  nlp: Optional[English],
                                  cache_dir: str,
                                  n_workers: int,
                                  worker_index: int,
                                  vqa_batch_size: int,
                                  cache_frames: bool=True,
                                  caption_first: bool=False,
                                  ) -> list[list[list[VQAOutputs]]]:
    """
    GPU-parallelizable method to run VQA in chunks on a MistakeDetectionDataset (with an optional AdaptiveVisualFilter).

    :param eval_dataset: MistakeDetectionDataset to run inference on.
    :param vlm: Initialized VLM for VQA.
    :param vlm_processor: VLM's processor.
    :param generate_prompts: A method that generates a list of prompts from a single MistakeDetectionExample.
    :param n_prompts_per_frame: The number of prompts expected to be generated by `generate_prompts` for each frame in each example.    
    :param visual_filter_mode: Visual filter type (if any).
    :param visual_filter: Initialized visual filter object, which may include a pre-trained object detection or phrase grounding model.
    :param nlp: NLP pipeline from spaCy for the visual filter to use.
    :param cache_dir: Directory to cache outputs in. Caches will be saved in a subdirectory for this worker.
    :param n_workers: Number of GPUs this inference is parallelized across.
    :param worker_index: Worker index for this call.
    :param vqa_batch_size: Batch size for VQA with VLM. This should be maximized for the type of GPU used.
    :param cache_frames: Whether to cache frames to disk after visual filters are applied (otherwise they will be discarded).
    :param caption_first: Whether to generate a caption for the image before answering the questions, and including captions in prompts to answer questions.
    """
    assert n_workers >= 1, "n_workers must be positive!"
    assert worker_index < n_workers and worker_index >= 0, f"Worker index should be a valid index for n_workers (n_workers)!"

    # Set up cache directory for this worker
    worker_cache_dir = os.path.join(cache_dir, f"VQA_cache_{eval_dataset.data_split}_worker{worker_index+1}of{n_workers}")
    if not os.path.exists(worker_cache_dir):
        os.makedirs(worker_cache_dir)

    # Get possible response token IDs to save in VQAOutputs
    response_token_ids = get_vqa_response_token_ids(vlm_processor.tokenizer)

    vqa_outputs = []
    all_captions_collated = []
    for chunk_idx, dataset_chunk in enumerate(tqdm(eval_dataset.get_batches(IMAGES_CHUNK_SIZE,
                                                                       n_workers=n_workers, 
                                                                       worker_index=worker_index), 
                                                   desc=f"chunks ({vlm.device})", 
                                                   total=len(eval_dataset) // IMAGES_CHUNK_SIZE)):
        prompt_cache_fname = os.path.join(worker_cache_dir, f"prompts_chunk{chunk_idx}.pkl")
        if not os.path.exists(prompt_cache_fname):
            questions = []
            prompts = []
            answers = []
            frames = []
            example_ids = []
            for example in dataset_chunk:
                example.cutoff_to_last_frames(DETECTION_FRAMES_PROPORTION)
                this_questions, this_prompts, this_answers, this_frames = generate_prompts(example, add_caption_placeholder=caption_first)
                assert len(this_questions) == len(this_prompts) == len(this_answers) == len(this_frames), "Passed `generate_prompts` method must return same number of questions, prompts, answers, and frames!"
                questions += this_questions
                prompts += this_prompts
                answers += this_answers
                frames += this_frames
                example_ids += [example.example_id] * len(this_questions)
            pickle.dump((questions, prompts, answers, frames, example_ids), open(prompt_cache_fname, "wb"))
        else:
            questions, prompts, answers, frames, example_ids = pickle.load(open(prompt_cache_fname, "rb"))
            # Still clip the example frames
            for example in dataset_chunk:
                example.cutoff_to_last_frames(DETECTION_FRAMES_PROPORTION)
        
        vqa_cache_path = os.path.join(worker_cache_dir, f"chunk{chunk_idx}.pt")

        # Intermediate results of detection aren't saved, so this is just a temporary hack just to check if we really need to run detection again
        visible_target_objects = None

        # Run visual filter if we have one, and we haven't already previously run it in full
        original_frames = frames
        if visual_filter_mode is not None and visual_filter is not None:
            if visual_filter_mode == VisualFilterTypes.Contrastive_Region:
                frames = visual_filter(nlp, frames, questions)
            elif visual_filter_mode == VisualFilterTypes.Visual_Contrastive:
                frames = visual_filter(frames)
            elif visual_filter_mode in [VisualFilterTypes.Spatial, VisualFilterTypes.Spatial_NoRephrase, VisualFilterTypes.Spatial_Blur]:
                spatial_cache_fname = os.path.join(worker_cache_dir, f"spatial_filter_outputs_chunk{chunk_idx}.pkl")
                if os.path.exists(spatial_cache_fname):
                    frames, new_questions = pickle.load(open(spatial_cache_fname, "rb"))
                else:
                    frames, new_questions = visual_filter(nlp, frames, questions, return_visible_target_objects=False)
                    pickle.dump((frames, new_questions, visible_target_objects), open(spatial_cache_fname, "wb"))
                
                # Replace rephrased questions into prompts, but save original questions for bookkeeping
                prompts = [prompt.replace(question, new_question) for prompt, question, new_question in zip(prompts, questions, new_questions)]
            elif visual_filter_mode == VisualFilterTypes.Target_Object_Counter:
                visible_target_objects = visual_filter(nlp, frames, questions, return_dict=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if caption_first:
            assert visual_filter_mode != VisualFilterTypes.Contrastive_Region, "CRG filter doesn't support conditioning with a caption yet!"

            # Generate a generic caption for each (filtered) frame first, then use that to condition VQA
            captions = run_captioning(vlm=vlm,
                                      processor=vlm_processor,
                                      frames=frames,
                                      batch_size=vqa_batch_size,
                                      cache_path=os.path.join(worker_cache_dir, f"chunk{chunk_idx}_captions"))
            prompts = [prompt.format(caption=caption) for prompt, caption in zip(prompts, captions)]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logits = run_vqa(vlm,
                         vlm_processor,
                         prompts,
                         frames,
                         batch_size=vqa_batch_size,
                         cache_path=vqa_cache_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if visual_filter_mode in [VisualFilterTypes.Contrastive_Region, VisualFilterTypes.Visual_Contrastive]:
            original_logits = run_vqa(vlm,
                                      vlm_processor,
                                      prompts,
                                      original_frames,
                                      batch_size=vqa_batch_size,
                                      cache_path=os.path.join(worker_cache_dir, f"chunk{chunk_idx}_original.pt"))
        else:
            original_logits = None

        del original_frames

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Gather up important information from VQA outputs, organized by example ID
        outputs_by_id = defaultdict(list)
        assert len(frames) == len(questions) == len(prompts) == len(answers) == len(example_ids), f"Length issue with frames ({len(frames)}), questions ({len(questions)}), prompts ({len(prompts)}), answers ({len(answers)}), or example_ids ({len(example_ids)})!"
        for output_index, (frame, question, prompt, answer, eid) in enumerate(zip(frames, questions, prompts, answers, example_ids)):
            outputs_by_id[eid].append((output_index, frame, question, prompt, answer, visible_target_objects[output_index] if visible_target_objects is not None else None))

        # Gather up VQA outputs in the correct structure for a MistakeDetectionEvaluator
        if caption_first:
            caption_idx = 0
        for example in tqdm(dataset_chunk, desc=f"gathering VQA outputs ({vlm.device})"):
            
            if caption_first:
                all_captions_collated.append(captions[caption_idx:caption_idx + len(example.frames)])
                caption_idx += len(example.frames)

            step_id = example.procedure_id
            example_vqa_outputs = []

            parallel_idx = 0
            for _ in example.frames:
                frame_vqa_outputs = []
                for _ in range(n_prompts_per_frame):
                    output_index, frame, question, prompt, answer, target_object_counts = outputs_by_id[example.example_id][parallel_idx]

                    if original_logits is not None:
                        # If we used a visual filter that involves combining logits from multiple runs, combine them
                        if visual_filter_mode == VisualFilterTypes.Contrastive_Region:
                            combined_logits = original_logits[output_index] - logits[output_index]
                        elif visual_filter_mode == VisualFilterTypes.Visual_Contrastive:
                            combined_logits = (1 + visual_filter.alpha) * original_logits[output_index] - visual_filter.alpha * logits[output_index]
                    else:
                        # Otherwise just use raw VQA logits
                        combined_logits = logits[output_index]

                    frame_vqa_outputs.append(
                        VQAOutputs(
                            example.task_name,
                            example.example_id,
                            step_id,
                            frame,
                            prompt,
                            answer,
                            response_token_ids,
                            combined_logits,
                            question=question, # Save original question for NLI mistake detection evaluator
                            target_object_counts=target_object_counts # Save count of target objects if we have it
                        )      
                    )
                    if cache_frames:
                        # Resize to a smaller size before caching to conserve disk space
                        frame_vqa_outputs[-1].frame = resize_with_aspect(frame_vqa_outputs[-1].frame, CACHED_FRAME_DIMENSION)
                        frame_vqa_outputs[-1].cache_frame(worker_cache_dir)
                    else:
                        # Just replace frame with empty string to save CPU memory and avoid saving frame to disk
                        frame_vqa_outputs[-1].frame = ""
                    parallel_idx += 1

                example_vqa_outputs.append(frame_vqa_outputs)

            vqa_outputs.append(example_vqa_outputs)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not caption_first:
        return vqa_outputs
    else:
        return vqa_outputs, all_captions_collated

def save_vqa_outputs(vqa_outputs: list[VQAOutputs], path: str, partition: str):
    """
    Saves list of VQAOutputs.
    
    :param vqa_outputs: List of VQAOutputs.
    :param path: Path to save json file (directory).
    """
    fname = f"vqa_outputs_{partition}.json"
    if not os.path.exists(path):
        os.makedirs(path)
    json.dump([ex.to_dict(image_base_path=path) for ex in vqa_outputs], 
              open(os.path.join(path, fname), "w"),
              indent=4)    

def _shift_right(input_ids, decoder_start_token_id, pad_token_id):
    """Copy of _shift_right method from T5 to use with BLIP-2 to automatically generate decoder input IDs."""
    if decoder_start_token_id is None:
        raise ValueError(
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
            "See T5 docs for more information."
        )

    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
  