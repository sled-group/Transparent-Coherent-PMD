from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import os
import pickle
from PIL import Image
from pprint import pprint
from spacy.lang.en import English
import torch
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, InstructBlipForConditionalGeneration, Kosmos2ForConditionalGeneration, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from typing import Optional, Union, Callable

from travel.constants import CACHE_FREQUENCY, IMAGES_CHUNK_SIZE
from travel.data.mistake_detection import MistakeDetectionTasks, MistakeDetectionDataset, MistakeDetectionExample
from travel.data.utils import split_list_into_partitions
from travel.model.grounding import VisualFilterTypes, AdaptiveVisualFilter
from travel.model.mistake_detection import DETECTION_FRAMES_PROPORTION

COMPLETION_PROMPT_TEMPLATES = {
    Blip2ForConditionalGeneration: "A photo of",
    InstructBlipForConditionalGeneration: "A photo of",
    Kosmos2ForConditionalGeneration: "<grounding> A photo of",
    LlavaForConditionalGeneration: 'USER: <image>\nWhat is happening in this photo? ASSISTANT: This is a photo of',
    LlavaNextForConditionalGeneration: '[INST] <image>\nWhat is shown in this image? [/INST] This is a photo of',
}

SUCCESSVQA_PROMPT_TEMPLATES = {
    Blip2ForConditionalGeneration: 'Question: The current goal is "{step}". Has the person successfully finished doing this? Answer:',
    InstructBlipForConditionalGeneration: 'Question: The current goal is "{step}". Has the person successfully finished doing this? Answer:',
    Kosmos2ForConditionalGeneration: '<grounding> Q: The current goal is "{step}". Has the person successfully finished doing this? A: ',
    LlavaForConditionalGeneration: 'USER: <image>\nThe current goal is "{step}". Has the person successfully finished doing this? ASSISTANT: ',
    LlavaNextForConditionalGeneration: '[INST] <image>\nThe current goal is "{step}". Has the person successfully finished doing this? [/INST]',
}

VQG2VQA_PROMPT_TEMPLATES = {
    Blip2ForConditionalGeneration: "Question: {question} Answer:",
    InstructBlipForConditionalGeneration: "Question: {question}? Answer: ",
    Kosmos2ForConditionalGeneration: "<grounding> Question: {question} Answer: ",
    LlavaForConditionalGeneration: "USER: <image>\n{question} ASSISTANT: ",
    LlavaNextForConditionalGeneration: "[INST] <image>\n{question} [/INST]",
}

class VQAResponse(int, Enum):
    No = 0
    Yes = 1

def get_vqa_response_token_ids(tokenizer):
    responses = {response: tokenizer(response.name, add_special_tokens=False)['input_ids'][0] for response in VQAResponse}
    for token_id in responses.values():
        assert type(token_id) == int, "Getting response tokens for members of VQAResponse failed."
    return responses

@dataclass
class VQAOutputs:
    """Dataclass to hold all VLM outputs from visual question answering (VQA)."""
    task_name: MistakeDetectionTasks
    example_id: str
    procedure_id: int
    frame: Union[Image.Image, str]
    prompt: str
    expected_answer: VQAResponse
    response_token_ids: dict[VQAResponse, int]
    logits: Optional[torch.FloatTensor] # (vocab size) 
    question: Optional[str] = None
    answer_probs: dict[VQAResponse, float] = field(default_factory=dict)
    predicted_answer: VQAResponse = VQAResponse["No"]

    def __post_init__(self):
        """Processes logits output from VLM into answer probabilities and final answer."""
        for response_type in VQAResponse:
            assert response_type in self.response_token_ids, f"VLM token ID for {response_type} not provided in VQAOutputs.answer_token_ids."

        this_probs = torch.stack([self.logits[self.response_token_ids[response_type]] for response_type in VQAResponse], dim=0)
        this_probs = torch.softmax(this_probs, dim=0)
        
        self.predicted_answer = VQAResponse(torch.argmax(this_probs, dim=0).numpy())
        
        this_probs = this_probs.numpy()
        self.answer_probs = {response_type: this_probs[response_type.value] for response_type in VQAResponse}

    def to_dict(self, image_base_path: Optional[str]=None):
        """Helper method to create a JSON-serializable version of the class instance (excluding some information)."""
        return_dict = {
            k: v for k, v in asdict(self).items() if k not in ["frame", "response_token_ids", "logits"]
        }
        for response in return_dict['answer_probs']:
            return_dict['answer_probs'][response] = float(round(return_dict['answer_probs'][response], 3))
        
        # Cache frame for this example if it's not already cached somewhere
        if image_base_path is not None and type(self.frame) != str:
            image_base_path = os.path.join(image_base_path, "frames")
            if not os.path.exists(image_base_path):
                os.makedirs(image_base_path)
            self.cache_frame(image_base_path)
            return_dict["frame"] = self.frame

        return return_dict
    
    def cache_frame(self, image_base_path: str):
        assert type(self.frame) != str, "Can only cache PIL images!"

        if not os.path.exists(os.path.join(image_base_path, "frames")):
            os.makedirs(os.path.join(image_base_path, "frames"))

        frame_path = os.path.join(image_base_path, "frames", f"frame_{self.example_id.replace('/', '-')}.jpg")

        self.frame.save(frame_path)

        self.frame = frame_path

    def uncache_frame(self):
        assert type(self.frame) == str and self.frame.endswith(".jpg"), "Can only uncache string .jpg filenames!"
        self.frame = Image.open(self.frame)

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
            logits = torch.load(cache_path)
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
            this_logits = this_logits[:, -1].detach().cpu()
            logits = torch.cat([logits, this_logits], dim=0)
            del this_logits

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
                                  ) -> list[list[list[VQAOutputs]]]:
    """
    GPU-parallelizable method to run VQA in chunks on a MistakeDetectionDataset (with an optional AdaptiveVisualFilter).

    :param eval_dataset: MistakeDetectionDataset to run inference on.
    :param vlm: Initialized VLM for VQA.
    :param vlm_processor: VLM's processor.
    :generate_prompts: A method that generates a list of prompts from a single MistakeDetectionExample.
    :n_prompts_per_frame: The number of prompts expected to be generated by `generate_prompts` for each frame in each example.    
    :visual_filter_mode: Visual filter type (if any).
    :visual_filter: Initialized visual filter object, which may include a pre-trained object detection or phrase grounding model.
    :nlp: NLP pipeline from spaCy for the visual filter to use.
    :cache_dir: Directory to cache outputs in. Caches will be saved in a subdirectory for this worker.
    :n_workers: Number of GPUs this inference is parallelized across.
    :worker_index: Worker index for this call.
    :vqa_batch_size: Batch size for VQA with VLM. This should be maximized for the type of GPU used.
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
    for chunk_idx, dataset_chunk in enumerate(eval_dataset.get_batches(IMAGES_CHUNK_SIZE,
                                                                       n_workers=n_workers, 
                                                                       worker_index=worker_index)):
        prompt_cache_fname = os.path.join(worker_cache_dir, f"prompts_chunk{chunk_idx}.pkl")
        if not os.path.exists(prompt_cache_fname):
            questions = []
            prompts = []
            answers = []
            frames = []
            example_ids = []
            for example in dataset_chunk:
                example.cutoff_to_last_frames(DETECTION_FRAMES_PROPORTION)
                this_questions, this_prompts, this_answers, this_frames = generate_prompts(example)
                assert len(this_questions) == len(this_prompts) == len(this_answers) == len(this_frames), "Passed `generate_prompts` method must return same number of questions, prompts, answers, and frames!"
                questions += this_questions
                prompts += this_prompts
                answers += this_answers
                frames += this_frames
                example_ids += [example.example_id] * len(questions)
            pickle.dump((questions, prompts, answers, frames, example_ids), open(prompt_cache_fname, "wb"))
        else:
            questions, prompts, answers, frames, example_ids = pickle.load(open(prompt_cache_fname, "rb"))
        
        vqa_cache_path = os.path.join(worker_cache_dir, f"VQA_cache_chunk{chunk_idx}.pt")

        # Intermediate results of detection aren't saved, so this is just a temporary hack just to check if we really need to run detection again
        logits = torch.zeros((0, vlm.vocab_size)).float()
        if os.path.exists(vqa_cache_path):
            logits = torch.load(vqa_cache_path)

        # Run visual filter if we have one, and we haven't already previously run it in full
        original_frames = frames
        if visual_filter_mode is not None and visual_filter is not None and logits.shape[0] < len(frames):
            if visual_filter_mode == VisualFilterTypes.Contrastive_Region:
                frames = visual_filter(nlp, frames, questions)
            elif visual_filter_mode in [VisualFilterTypes.Spatial, VisualFilterTypes.Spatial_NoRephrase]:
                frames, new_questions = visual_filter(nlp, frames, questions)
                # Replace rephrased questions into prompts
                prompts = [prompt.replace(question, new_question) for prompt, question, new_question in zip(prompts, questions, new_questions)]
                questions = new_questions

        # Then delete these pre-loaded logits
        del logits

        logits = run_vqa(vlm,
                         vlm_processor,
                         prompts,
                         frames,
                         batch_size=vqa_batch_size,
                         cache_path=os.path.join(worker_cache_dir, f"chunk{chunk_idx}.pt"))

        if visual_filter_mode is not None and VisualFilterTypes(visual_filter_mode) == VisualFilterTypes.Contrastive_Region:
            original_logits = run_vqa(vlm,
                                        vlm_processor,
                                        prompts,
                                        original_frames,
                                        batch_size=vqa_batch_size,
                                        cache_path=os.path.join(worker_cache_dir, f"chunk{chunk_idx}_crg_original.pt"))
        else:
            original_logits = None

        del original_frames

        # Gather up important information from VQA outputs, organized by example ID
        outputs_by_id = defaultdict(list)
        for output_index, (frame, question, prompt, answer, eid) in enumerate(zip(frames, questions, prompts, answers, example_ids)):
            outputs_by_id[eid].append((output_index, frame, question, prompt, answer))

        # Gather up VQA outputs in the correct structure for a MistakeDetectionEvaluator
        for example in tqdm(eval_dataset, "gathering VQA outputs"):
            # Cutoff again since the example will be reloaded from disk when we access it
            example.cutoff_to_last_frames(DETECTION_FRAMES_PROPORTION)
            step_id = example.procedure_id
            example_vqa_outputs = []

            parallel_idx = 0
            for _ in example.frames:
                frame_vqa_outputs = []
                for _ in range(n_prompts_per_frame):
                    output_index, frame, question, prompt, answer = outputs_by_id[example.example_id][parallel_idx]
                    frame_vqa_outputs.append(
                        VQAOutputs(
                            example.task_name,
                            example.example_id,
                            step_id,
                            frame,
                            prompt,
                            answer,
                            response_token_ids,
                            original_logits[output_index] - logits[output_index] if original_logits is not None else logits[output_index],        
                            question=question, # Save original question for NLI mistake detection evaluator
                        )      
                    )
                    frame_vqa_outputs[-1].cache_frame(cache_dir)
                    parallel_idx += 1

                example_vqa_outputs.append(frame_vqa_outputs)

            vqa_outputs.append(example_vqa_outputs)

    return vqa_outputs

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