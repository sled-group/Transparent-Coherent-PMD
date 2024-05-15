import dataclasses
from enum import Enum
import numpy as np
from PIL import Image
import spacy
from spacy.lang.en import English
import torch
from tqdm import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import Optional, Any
import yaml

from travel.data.mistake_detection import MistakeDetectionDataset
from travel.data.utils.image import get_preprocessed_image
from travel.model.vqg import VQGOutputs

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

OWL_THRESHOLD = config["grounding"]["owl_threshold"] # Directory to cache model outputs and other temporary data
OWLV2_PATH = config["grounding"]["owlv2_path"]

def filter_frames_by_target_objects(dataset: MistakeDetectionDataset,
                                    detector: Owlv2ForObjectDetection,
                                    detector_processor: Owlv2Processor,
                                    vqg_outputs: Optional[dict[int, VQGOutputs]]) -> MistakeDetectionDataset:
    """
    Filters the frames in MistakeDetectionExample objects within a MistakeDetectionDataset based on whether they have a target object present. The target object will be based on a dictionary of VQGOutputs if provided, otherwise it will be parsed from target recipe steps.

    :param dataset: MistakeDetectionDataset to filter frames of.
    :param detector: Initialized OWL object detector.
    :param detector_processor: OWL processor.
    :param vqg_outputs: Optional dictionary of VQGOutputs for recipe steps, which can provide target objects proposed in VQG step.
    :return: Dataset with filtered frames.
    """
    if vqg_outputs is None:
        nlp = spacy.load('en_core_web_sm')

    with torch.no_grad():
        
        all_frames = [frame for example in dataset for frame in example.frames]
        frames_before = len(all_frames)

        if vqg_outputs is not None:
            all_texts = [f"a photo of {'an' if vqg_outputs[example.procedure_id].target_object[0] in ['a','e','i','o','u'] else 'a'} {vqg_outputs[example.procedure_id].target_object}" for example in dataset for frame in example.frames]
        else:
            # TODO: implement using spaCy to parse objects from recipe step - since there may be multiple for each frame this can change data processing details
            raise NotImplementedError("Can't handle parsing objects from recipe step yet.")
            # all_texts = [ for example in dataset for frame in example.frames]
        
        batch_size = 8
        all_results = []
        # all_padded_images = []
        for i in tqdm(range(0, len(all_frames), batch_size), desc=f"detecting objects with threshold {OWL_THRESHOLD}"):
            # Prepare the batch
            batch_frames = all_frames[i:i+batch_size]
            batch_texts = all_texts[i:i+batch_size]
            
            inputs = detector_processor(text=batch_texts, images=batch_frames, return_tensors="pt").to(detector.device)
            outputs = detector(**inputs)
            inputs = inputs.to("cpu")  
            
            padded_images = [get_preprocessed_image(inputs.pixel_values[j].detach().to('cpu')) for j in range(len(batch_frames))]

            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([pi.size[::-1] for pi in padded_images])
            # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
            results = detector_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=OWL_THRESHOLD)
            all_results += results
            # all_padded_images += padded_images
            
        texts = all_texts
        results = all_results
        # padded_images = all_padded_images

        example_frame_idx = 0
        filtered_examples = []  
        filtered_out_frames = 0
        for example in tqdm(dataset, desc="filtering frames"):        
            step_id = example.procedure_id
            vqg_output = vqg_outputs[step_id]
            target_object = vqg_output.target_object

            filtered_frames = []
            filtered_frame_times = []
            for i in range(example_frame_idx, example_frame_idx + len(example.frames)):
                
                text = texts[i]
                boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

                if len(boxes) > 0:
                    filtered_frames.append(example.frames[i - example_frame_idx])
                    filtered_frame_times.append(example.frame_times[i - example_frame_idx])
                else:
                    filtered_out_frames += 1
                
            new_example = dataclasses.replace(example)
            new_example.frames = filtered_frames
            new_example.frame_times = filtered_frame_times
            filtered_examples.append(new_example)
            example_frame_idx += len(example.frames)    

    frames_after = sum([len(example.frames) for example in filtered_examples])
    print(f"Filtered out {filtered_out_frames} video frames ({frames_before} -> {frames_after}).")
    dataset.examples = filtered_examples
    return dataset
    
# TODO: debug the below 2 classes
class AdaptiveVisualFilter:
    """Parent class for adaptive attention filters that use phrase grounding models to mask/crop images based on visual questions."""

    def __init__(self):
        # Load OWL object detector for filtering frames, and filter frames
        self.detector_processor = Owlv2Processor.from_pretrained(OWLV2_PATH)
        self.detector = Owlv2ForObjectDetection.from_pretrained(OWLV2_PATH, load_in_8bit=True)

    def run_detection(self, objects: list[list[str]], frames: list[Image.Image]) -> tuple[Any, Any]:
        """
        Runs OWLv2 object detection.

        :param objects: List of lists of objects (one list of objects per frame).
        :param frames: List of images to check for objects (one image per list of objects).
        :return: Results and preprocessed padded images that can be .
        """
        assert len(objects) == len(frames), "Expected same number of object lists and frames!"
        owl_prompts = [[f"a photo of {'an' if input_obj[0] in ['a','e','i','o','u'] else 'a'} {input_obj}" for input_obj in input_objs] for input_objs, _ in zip(objects, frames)]

        inputs = self.detector_processor(text=owl_prompts, images=frames, return_tensors="pt").to(self.detector.device)
        outputs = self.detector(**inputs)
        inputs = inputs.to("cpu")

        padded_images = [get_preprocessed_image(inputs.pixel_values[j].detach().to('cpu')) for j in range(len(frames))]

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([pi.size[::-1] for pi in padded_images])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        results = self.detector_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=OWL_THRESHOLD)

        return results, padded_images
    
    def __call__(self) -> list[Image.Image]:
        raise NotImplementedError("Subclass must implement __call__!")

class SpatialVisualFilter(AdaptiveVisualFilter):
    """Visual attention filter that masks/crops an image based on spatial dependencies in a visual question."""
    def __init__(self):
        super().__init__()

    @staticmethod
    def parse_questions_for_spatial_attention_filter(nlp: English, questions: list[str]) -> list[tuple[bool, str]]:
        """
        Parses a question for spatial relations that can be visually abstracted with the spatial attention filter.

        :param nlp: spaCy pipeline. Initialize with `spacy.load("en_core_web_sm", disable=["lemmatizer"])`
        :param questions: List of yes/no questions about an image (e.g., are there any cherry tomatoes in the bowl?).
        :return: List of tuples of bools and objects indicating regions of interest.
        """
        results = []
        for question in questions:
            doc = nlp(question)
            target_noun = ""
            negation_present = False
            look_at_noun = True
            spatial_relation = False

            # Function to extract the compound noun if it exists
            def get_compound_noun(token):
                compound = " ".join([child.text for child in token.lefts if child.dep_ == "compound"])
                return compound + " " + token.text if compound else token.text

            for token in doc:
                # Detect negation
                if token.dep_ == "neg":
                    negation_present = True

                # For subjects and objects, capture the noun considering compound modifiers
                if token.dep_ in ["nsubj", "attr", "dobj", "pobj"] and token.pos_ == "NOUN":
                    target_noun = get_compound_noun(token)
                
                # Identify spatial relations based on specific dependencies
                if token.dep_ == "prep":
                    spatial_relation = True

            # Adjust the logic based on question type and negation
            # Spatial questions with negation direct attention away from the noun
            if spatial_relation:
                look_at_noun = not negation_present
            # State questions focus on the noun, negation doesn't change the focus
            else:
                look_at_noun = True

            results.append((look_at_noun, target_noun))
        return results

    def __call__(self, nlp: English, frames: list[Image.Image], questions: list[str]) -> tuple[list[Image.Image], list[str]]:
        # Parse spatial dependencies from questions
        spatial_parse_results = self.parse_questions_for_spatial_attention_filter(nlp, questions)
        detection_results, padded_images = self.run_detection([[noun] for _, noun in spatial_parse_results], frames)

        new_frames = []
        # Iterate in parallel through spatial parse results, detection results, frames, and padded frames
        for (look_at_noun, noun), detection_result, frame, frame_padded in zip(spatial_parse_results, detection_results, frames, padded_images):
            boxes, scores, labels = detection_result["boxes"], detection_result["scores"], detection_result["labels"]
            bboxes = boxes.detach().cpu().numpy() # (# boxes, 4)

            mask = np.ones((frame_padded.height, frame_padded.width), dtype=np.float64)
            
            # Mask out the areas for this noun
            # TODO: crop image to the clustered bounding box?
            for bbox in bboxes:
                # Set the area within the bounding box to 0
                # Note the order: (ymin:ymax, xmin:xmax)
                mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 0
                
            if look_at_noun:
                mask = 1 - mask
                            
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            # Apply mask and undo padding of masked/cropped image to pass to VLM later
            new_frame = np.array(frame_padded) * mask
            new_frame = Image.fromarray(new_frame.astype(np.uint8))
            frame = Image.fromarray(frame)
            new_height = new_frame.width / frame.width * frame.height
            new_frame = new_frame.crop((0, 0, new_frame.width - 1, new_height))

            if look_at_noun:
                # If we're blocking out everything but some bboxes
                min_x = np.min(bboxes[:, 0])
                min_y = np.min(bboxes[:, 1])
                max_x = np.max(bboxes[:, 2])
                max_y = np.max(bboxes[:, 3])
                new_frame = new_frame.crop((min_x, min_y, max_x, max_y))
            
            new_frames.append(new_frame)

        # TODO: remove spatial dependencies from questions
        return new_frames, questions

class VisualFilterTypes(Enum):
    Spatial = "spatial"