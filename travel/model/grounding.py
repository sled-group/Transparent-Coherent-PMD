import dataclasses
from enum import Enum
import numpy as np
from PIL import Image
from pprint import pprint
import spacy
from spacy.lang.en import English
import torch
from tqdm import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection, BitsAndBytesConfig, BatchEncoding
from typing import Optional, Any, Union
import yaml

from travel.data.mistake_detection import MistakeDetectionDataset
from travel.data.utils.image import get_preprocessed_image, BoundingBoxCluster, BoundingBox
from travel.data.utils.text import get_compound_noun
from travel.data.vqg import VQGOutputs

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

OWLV2_PATH = config["grounding"]["owlv2_path"]
OWL_THRESHOLD = config["grounding"]["owl_threshold"] 
OWL_BATCH_SIZE = config["grounding"]["owl_batch_size"]
MASK_STRENGTH = config["grounding"]["mask_strength"]
MINIMUM_CROP_SIZE = config["grounding"]["minimum_crop_size"]
MAXIMUM_BBOXES_PER_OBJECT = config["grounding"]["maximum_bboxes_per_object"]

def filter_frames_by_target_objects(dataset: MistakeDetectionDataset,
                                    detector: Owlv2ForObjectDetection,
                                    detector_processor: Owlv2Processor,
                                    vqg_outputs: Optional[dict[int, VQGOutputs]]) -> MistakeDetectionDataset:
    """
    (Not in use.) Filters the frames in MistakeDetectionExample objects within a MistakeDetectionDataset based on whether they have a target object present. The target object will be based on a dictionary of VQGOutputs if provided, otherwise it will be parsed from target recipe steps.

    :param dataset: MistakeDetectionDataset to filter frames of.
    :param detector: Initialized OWL object detector.
    :param detector_processor: OWL processor.
    :param vqg_outputs: Optional dictionary of VQGOutputs for recipe steps, which can provide target objects proposed in VQG step.
    :return: Dataset with filtered frames.
    """
    if vqg_outputs is None:
        nlp = spacy.load('en_core_web_lg')

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

class AdaptiveVisualFilter:
    """Parent class for adaptive attention filters that use phrase grounding models to mask/crop images based on visual questions."""

    def __init__(self, device: Optional[str]=None):
        # Load OWL object detector for filtering frames, and filter frames
        self.detector_processor = Owlv2Processor.from_pretrained(OWLV2_PATH)
        if device is not None:
            torch.cuda.set_device(device)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.detector = Owlv2ForObjectDetection.from_pretrained(OWLV2_PATH, quantization_config=bnb_config)

    def run_detection(self, objects: list[list[str]], frames: list[Image.Image], batch_size: int=OWL_BATCH_SIZE) -> tuple[Any, Any, Any]:
        """
        Runs OWLv2 object detection.

        :param objects: List of lists of objects (one list of objects per frame).
        :param frames: List of images to check for objects (one image per list of objects).
        :return: Object detection results and preprocessed padded images, as well as ints representing which labels are for padding.
        """
        assert len(objects) == len(frames), "Expected same number of object lists and frames!"
        # Note the awkward hack where rare objects can be None due to failure of spatial parser
        max_n_objs = max([len(this_objects) for this_objects in objects])
        owl_prompts = [[f"a photo of the {input_objs[i]}" if (i < len(input_objs) and input_objs[i] is not None) else "a photo of nothing" for i in range(max_n_objs)] for input_objs, _ in zip(objects, frames)]
        if all(len(p) == 0 for p in owl_prompts):
            print("Warning: run_detection received no prompts.")
            return [{"boxes": torch.tensor([]), "scores": torch.tensor([]), "labels": torch.tensor([])} for _ in frames], frames
        pad_labels = [[1 if (i < len(input_objs) and input_objs[i] is not None) else 0 for i in range(max_n_objs)] for input_objs, _ in zip(objects, frames)]
        # skip_indices = [all(input_obj is None for input_obj in input_objs) for input_objs, _ in zip(objects, frames)]

        # Clear CUDA cache before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        padded_images = []
        results = []
        with torch.no_grad():
            for i in tqdm(range(0, len(frames), batch_size), desc=f"running detection ({str(self.detector.device)})"):
                # Prepare the batch
                batch_prompts = owl_prompts[i:i+batch_size]
                batch_frames = frames[i:i+batch_size]

                # Run processor one by one and transfer to GPU to avoid memory spike
                inputs = [self.detector_processor(text=batch_prompts[j], images=batch_frames[j], return_tensors="pt").to(self.detector.device) for j in range(len(batch_frames))]
                inputs = BatchEncoding({k: torch.cat([inp[k] for inp in inputs], dim=0) for k in inputs[0]})
                outputs = self.detector(**inputs)
                
                this_padded_images = [get_preprocessed_image(inputs.pixel_values[j].to('cpu')) for j in range(len(batch_frames))]

                # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
                target_sizes = torch.Tensor([pi.size[::-1] for pi in this_padded_images])

                # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
                padded_images += this_padded_images
                results += self.detector_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=OWL_THRESHOLD)

                # print("================================================")
                # print("OWL processing:")
                # pprint(batch_prompts)
                # batch_pad_labels = pad_labels[i:i+batch_size]
                # pprint(batch_pad_labels)
                # print(max_n_objs)
                # pprint(self.detector_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=OWL_THRESHOLD))
                # print("================================================")

                del inputs
                del outputs

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        assert len(pad_labels) == len(results)

        # Remove padding instances from results
        for i in range(len(results)):
            keep_labels = pad_labels[i]
            keep_labels = [li for li, l in enumerate(keep_labels) if l == 1]

            detection_result = results[i]
            if len(keep_labels) > 0:
                # Keep any non-padding instances
                keep_indices = torch.stack([detection_result['labels'] == l for l in keep_labels]).sum(0).bool()
                detection_result['boxes'] = detection_result['boxes'][keep_indices]
                detection_result['scores'] = detection_result['scores'][keep_indices]
                detection_result['labels'] = detection_result['labels'][keep_indices]
                results[i] = detection_result
            else:
                # Or throw everything out and replace with empty tensors
                detection_result['boxes'] = torch.zeros([0, 4])
                detection_result['scores'] = torch.zeros([0])
                detection_result['labels'] = torch.zeros([0])
    
        # results = [detection_result for detection_result in results if detection_result['scores'] >= OWL_THRESHOLD]
        
        # Filter results for cases where no objects were passed
        return results, padded_images
    
    def __call__(self) -> list[Image.Image]:
        raise NotImplementedError("Subclass must implement __call__!")

DO_NOT_PARSE_NOUNS = [
    "image",
    "scene",
    "anyone",
    "heat",
    "temperature",
    "someone",
    "hand",
    "hands",
    "place",
]

class TargetObjectCounterFilter(AdaptiveVisualFilter):
    """
    This visual filter is used to count target objects from procedures in frames.
    """
    def __init__(self, **kwargs: dict[str, Any]):
        super().__init__(**kwargs)

    @staticmethod
    def parse_sentences_for_target_objects(nlp: English, sentences: list[str]) -> list[list[str]]:
        results = []
        for sentence in sentences:
            doc = nlp(sentence)
            nouns = []
            
            # for token in doc:
            #     print(token.text, token.dep_, token.pos_)
            # print("\n\n")

            # Grab nouns from sentence
            for token_idx, token in enumerate(doc):
                if token.dep_ in ["nsubj", "nsubjpass", "attr", "dobj", "pobj"] and token.pos_ == "NOUN":
                    
                    # Sometimes adjectives near the end of a sentence are mistakenly labeled as nsubj; 
                    # nsubj should never be at the end of the sentence in questions, so ignore if this happens
                    if token.dep_ == "nsubj" and token_idx == len(doc) - 2 and doc[token_idx + 1].pos_ == "PUNCT":
                        continue
                    
                    # If this noun is followed by "of", skip since we'll add the full noun phrase later
                    if "of" in [t.text for t in token.rights]:
                        continue

                    # Some common words should not be counted as objects
                    if token.text not in DO_NOT_PARSE_NOUNS:
                        compound_noun = get_compound_noun(token)
                        if all(n not in compound_noun for n in DO_NOT_PARSE_NOUNS):
                            # Make sure the compound noun doesn't have any "do not parse" nouns
                            nouns.append(compound_noun)

            # If we didn't find a noun with first logic, be a bit more lenient and look for anything else labeled as a noun
            if len(nouns) == 0:
                for token in doc:
                    if token.pos_ == "NOUN" and token.dep_ not in ["nsubj", "nsubjpass", "attr", "dobj", "pobj"]:
                        if token.text not in DO_NOT_PARSE_NOUNS:
                            compound_noun = get_compound_noun(token)
                            if all(n not in compound_noun for n in DO_NOT_PARSE_NOUNS):
                                # Make sure the compound noun doesn't have any "do not parse" nouns
                                nouns.append(compound_noun)

            results.append(nouns)
        return results
        
    @staticmethod
    def count_objects_in_detection_results(detection_results_single: Union[list[dict[Any, Any]], dict[Any, Any]]):
        """Counts the unique objects recognized in detection results."""
        if type(detection_results_single) == dict:
            detection_results_single = [detection_results_single]
        detection_results_single = [result["labels"].cpu().numpy() for result in detection_results_single]
        object_counts = []
        for result in detection_results_single:
            this_object_counts = {}
            for label_idx in np.unique(result):
                this_object_counts[label_idx] = result[result == label_idx].shape[0]
            object_counts.append(this_object_counts)
        return object_counts

    def __call__(self, nlp: English, frames: list[Image.Image], procedures: list[str]) -> list[int]:
        # Parse objects from questions
        object_parse_results = self.parse_sentences_for_target_objects(nlp, procedures)
        detection_results, _ = self.run_detection(object_parse_results, frames)
        
        # Get target object counts
        target_object_counts = [sum(list(result.values())) for result in TargetObjectCounterFilter.count_objects_in_detection_results(detection_results)]

        return target_object_counts

class SpatialVisualFilter(AdaptiveVisualFilter):

    """Visual attention filter that masks/crops an image based on spatial dependencies in a visual question."""
    def __init__(self, rephrase_questions: bool=True, **kwargs: dict[str, Any]):
        self.rephrase_questions = rephrase_questions
        super().__init__(**kwargs)

    @staticmethod
    def parse_questions_for_spatial_attention_filter(nlp: English, questions: list[str], rephrase_questions: bool=True) -> list[tuple[bool, Optional[str], str]]:
        """
        Parses a question for spatial relations that can be visually abstracted with the spatial attention filter.

        :param nlp: spaCy pipeline. Initialize with `spacy.load("en_core_web_lg", disable=["lemmatizer"])`
        :param questions: List of yes/no questions about an image (e.g., are there any cherry tomatoes in the bowl?).
        :return: List of tuples, each of which include a bool and object string indicating regions of interest, and a rephrased form of a question without spatial dependencies. 
        """

        spatial_preps = ["in", "on", "inside", "outside", "inside of", "outside of",
                        "off", "out", "out of", "within", "across"]
        negation_preps = ["out", "out of", "outside", "outside of", "off"]
        no_rephrase_words = ["top", "bottom", "left", "right", "each", "all", "every", "single"]
        avoid_with_on = ["temperature", "heat", "low", "medium", "high", "left", "right", "top", "bottom"]
        avoid_with_in = ["hand", "left hand", "right hand", "someone's hand", "someone's left hand", "someone's right hand"]

        results = []
        for question in questions:
            doc = nlp(question)
            target_noun = ""
            negation_present = False
            look_at_noun = True
            spatial_relation = False

            no_rephrase_word_present = False
            is_negation_prep = False

            is_avoid_on = False
            is_avoid_in = False            
            
            # Detect negation of a spatial relation
            for idx, token in enumerate(doc):
                if token.dep_ == "neg" and doc[idx+1].dep_ == "prep" and doc[idx+1].text in spatial_preps:
                    negation_present = True
                    negation_token = token.text

            # For subjects and objects, capture the noun considering compound modifiers
            for idx, token in enumerate(doc):
                if token.dep_ in ["nsubj", "nsubjpass", "attr", "dobj", "pobj"] and token.pos_ == "NOUN":
                    this_target_noun = get_compound_noun(token)
                    
                    # Check the following conditions:
                    # - We should never apply spatial filter on the nouns in avoid_with_on and avoid_with_in and DO_NOT_PARSE_NOUNS
                    # - Some common words are never objects, e.g., "anyone", "image"
                    # - Also, sometimes adjectives near the end of a sentence are mistakenly labeled as nsubj; nsubj should never be at the end of the sentence in questions, so ignore if this happens
                    # - Noun must not be followed by "of", e.g., "piece of wood"; for these noun phrases with "of", we gather them from the tail noun (e.g., "wood", not "piece")
                    if not(this_target_noun in avoid_with_on \
                           or this_target_noun in avoid_with_in \
                            or this_target_noun in DO_NOT_PARSE_NOUNS \
                            or any(n in this_target_noun for n in DO_NOT_PARSE_NOUNS) \
                            or token.dep_ == "nsubj" and idx == len(doc) - 2 and doc[idx + 1].pos_ == "PUNCT" \
                            or "of" in [t.text for t in token.rights]):
                        target_noun = this_target_noun
            
            # If we didn't find a noun with first logic, be a bit more lenient and look for anything else labeled as a noun
            if target_noun == "":
                for token in doc:
                    if token.pos_ == "NOUN" and token.dep_ not in ["nsubj", "nsubjpass", "attr", "dobj", "pobj"]:
                        if not(token.text in avoid_with_on \
                            or token.text in avoid_with_in \
                                or token.text in DO_NOT_PARSE_NOUNS \
                                or any(n in token.text for n in DO_NOT_PARSE_NOUNS)):
                            target_noun = token.text

            # Identify spatial relations based on specific dependencies
            for idx, token in enumerate(doc):
                if token.dep_ == "prep":
                    # Get preposition, for prep="of" we add the previous word
                    prep = token.text
                    if idx != 0 and prep == "of":
                        prep = doc[idx - 1].text + " of"

                    if prep in spatial_preps:
                        spatial_object_tokens = [get_compound_noun(child) for child in token.children if child.pos_ == "NOUN"]
                        is_avoid_on = prep == "on" and any(word.lower() in spatial_object_tokens for word in avoid_with_on) # TODO: consider just never rephrasing when the preposition is "on" (that seems to slightly hurt performance on small subset of ego4d)
                        is_avoid_in = prep == "in" and any(word.lower() in spatial_object_tokens for word in avoid_with_in)
                        if not is_avoid_on and not is_avoid_in:
                            spatial_relation = True
                            is_negation_prep = prep in negation_preps

                if token.text in no_rephrase_words:
                    no_rephrase_word_present = True

            # Adjust the logic based on question type and negation
            # Spatial questions with negation direct attention away from the noun
            if spatial_relation:
                look_at_noun = not negation_present if not is_negation_prep else negation_present

                # Rephrase question if needed
                if rephrase_questions and not no_rephrase_word_present:
                    for token in spatial_object_tokens:
                        question = question.replace(f"{token}?", "image?")
                        question = question.replace(f" {token} ", " image ")
                    question = question.replace(" " + prep + " ", " in ")

                    # Replace articles and possessives that don't play well with "image"
                    for determiner_phrase in [" someone's image", " an image", " a image"]:
                        question = question.replace(determiner_phrase, " the image")

                    # Remove negation
                    if negation_present:
                        question = question.replace(negation_token, "").replace("  ", " ")

            # State questions focus on the noun, negation doesn't change the focus
            else:
                look_at_noun = True

            results.append((look_at_noun, target_noun if (target_noun != "" and not is_avoid_on and not is_avoid_in) else None, question))
        return results

    def __call__(self, nlp: English, frames: list[Image.Image], questions: list[str], batch_size: int=OWL_BATCH_SIZE, return_visible_target_objects=True) -> tuple[list[Image.Image], list[str]]:

        # First, parse out all "target" objects mentioned in questions and count them in images
        if return_visible_target_objects:
            object_parse_results = TargetObjectCounterFilter.parse_sentences_for_target_objects(nlp, questions)
            counting_results, _ = self.run_detection(object_parse_results, frames)
            object_counts_old = TargetObjectCounterFilter.count_objects_in_detection_results(counting_results) # TODO: rename this back to object_counts
            object_counts = [{object_parse_results[object_count_idx][label_idx]: object_count[label_idx] if label_idx in object_count else 0 for label_idx in range(len(object_parse_results[object_count_idx]))} for object_count_idx, object_count in enumerate(object_counts_old)]

        # Parse spatial dependencies from questions and use them to detect objects
        spatial_parse_results = self.parse_questions_for_spatial_attention_filter(nlp, questions, rephrase_questions=self.rephrase_questions)
        detection_results, padded_images = self.run_detection([[noun] for _, noun, _ in spatial_parse_results], 
                                                              frames,
                                                              batch_size=batch_size)

        # for result, question, spatial, obj_parse, count_result, obj_counts, obj_counts_old in zip(detection_results, questions, spatial_parse_results, object_parse_results, counting_results, object_counts, object_counts_old):
        #     print(f"{question}\n")
        #     pprint(spatial)
        #     pprint(result)
        #     print("")
        #     pprint(obj_parse)
        #     pprint(count_result)
        #     print(obj_counts_old)
        #     pprint(obj_counts)
        #     print("\n\n")

        new_frames = []
        new_questions = []
        # Iterate in parallel through spatial parse results, detection results, frames, and padded frames
        for old_question, (look_at_noun, noun, new_question), detection_result, frame, frame_padded in zip(questions, spatial_parse_results, detection_results, frames, padded_images):
            bboxes = detection_result["boxes"]
            bboxes = bboxes.cpu().numpy() # (# boxes, 4)
            scores = detection_result["scores"].cpu().numpy()

            # print(old_question, noun)

            # Reweight the confidence of each candidate bounding box based on how close it is to the center of the image (central objects are more likely to be important)
            if bboxes.shape[0] > 0:

                # Merge together overlapping bounding boxes
                bboxes = np.array([bbox.coords for bbox in BoundingBoxCluster([BoundingBox(*bbox, score) for bbox, score in zip(bboxes, scores)]).get_merged_boxes()])

                for bbox_idx, bbox in enumerate(bboxes):
                    # old_score = scores[bbox_idx]

                    bbox_centroid = np.array(((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0))
                    image_centroid = np.array((frame_padded.width / 2.0, (frame_padded.width / frame.width * frame.height) / 2.0)) # NOTE: this assumes image is horizontal (width >= height)
                    bbox_dist_from_center = min(np.linalg.norm(image_centroid - bbox_centroid) / np.sqrt(image_centroid[0] ** 2 + image_centroid[1] ** 2), 1.0) # normalize by maximum distance
                    assert 0.0 <= bbox_dist_from_center <= 1.0, f"Bounding box distance out of range: {bbox_dist_from_center}"
                    bbox_dist_from_center_reweighted = 0.5 / (1 + np.exp(-20 * (0.5 - bbox_dist_from_center))) + 0.5 # Use a sigmoid function to re-weight the distance
                    scores[bbox_idx] *= bbox_dist_from_center_reweighted

                    # print(f"bbox {bbox} in {frame.width}x{frame.height} image: {old_score} -> {scores[bbox_idx]} (dist={bbox_dist_from_center}, {bbox_dist_from_center_reweighted} after reweight)")
                    # print("padded frame size:", frame_padded.size)
                    # print("image centroid:", image_centroid)
                    # print("bbox centroid:", bbox_centroid)
                    # print("")
                    # frame_padded.save("temp.png")

                # Remove any bboxes that are no longer above the threshold
                bboxes = np.array([bbox for bbox, score in zip(bboxes, scores) if score >= OWL_THRESHOLD])
                scores = np.array([score for score in scores if score >= OWL_THRESHOLD])
                if len(bboxes.shape) == 1:
                    # There's only one bbox left, which takes away a dim
                    bboxes = np.expand_dims(bboxes, axis=0)

            if bboxes.shape[0] > 0:
                # If we still have some bboxes

                # Select only top MAXIMUM_BBOXES_PER_OBJECT boxes
                bboxes = zip(bboxes, scores)
                bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)[:MAXIMUM_BBOXES_PER_OBJECT]
                bboxes = np.array([bbox for bbox, _ in bboxes])
                if len(bboxes.shape) == 1:
                    # There's only one bbox left, which takes away a dim
                    bboxes = np.expand_dims(bboxes, axis=0)                
                del scores

                mask = np.ones((frame_padded.height, frame_padded.width), dtype=np.float64)

                # Mask out the areas for this noun
                for bbox_idx, bbox in enumerate(bboxes):
                    # If looking specifically at this noun, make sure bounding boxes are a minimum size (configured in config.yml);
                    # also make sure no bboxes cross the buondaries of the image
                    if look_at_noun:
                        bbox_height = bbox[3] - bbox[1]
                        if bbox_height < MINIMUM_CROP_SIZE:
                            bbox[1] -= (MINIMUM_CROP_SIZE - bbox_height) / 2
                            bbox[3] += (MINIMUM_CROP_SIZE - bbox_height) / 2
                        if bbox[1] < 0:
                            bbox[1] = 0
                        if bbox[3] >= mask.shape[0]:
                            bbox[3] = mask.shape[0] - 1

                        bbox_width = bbox[2] - bbox[0]
                        if bbox_width < MINIMUM_CROP_SIZE:
                            bbox[0] -= (MINIMUM_CROP_SIZE - bbox_width) / 2
                            bbox[2] += (MINIMUM_CROP_SIZE - bbox_width) / 2
                        if bbox[0] < 0:
                            bbox[0] = 0
                        if bbox[2] >= mask.shape[1]:
                            bbox[2] = mask.shape[1] - 1     

                        bboxes[bbox_idx] = bbox                   

                    # Set the area within the bounding box to 0
                    # Note the order: (ymin:ymax, xmin:xmax)                        
                    mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 0.0
                    
                if look_at_noun:
                    mask = 1 - mask
                                
                # Apply mask strength to black parts of resulting mask
                mask = (1.0 - (1 - mask) * MASK_STRENGTH)
                
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

                # Undo padding of masked/cropped image to pass to VLM later
                new_frame = np.array(frame_padded) * mask
                new_frame = Image.fromarray(new_frame.astype(np.uint8))
                new_height = new_frame.width / frame.width * frame.height
                new_frame = new_frame.crop((0, 0, new_frame.width - 1, new_height))

                if look_at_noun and MASK_STRENGTH == 1.0:
                    # If we're blocking out everything but some bboxes, crop the image to only look at them
                    min_x = np.min(bboxes[:, 0])
                    min_y = np.min(bboxes[:, 1])
                    max_x = np.max(bboxes[:, 2])
                    max_y = np.max(bboxes[:, 3])
                    new_frame = new_frame.crop((min_x, min_y, max_x, max_y))
                
                new_frames.append(new_frame)
                new_questions.append(new_question)
            else:
                # No detection - don't modify the image or question
                new_frames.append(frame)
                new_questions.append(old_question)

        if return_visible_target_objects:
            return new_frames, new_questions, object_counts
        else:
            return new_frames, new_questions

class ContrastiveRegionFilter(AdaptiveVisualFilter):
    def __init__(self, **kwargs: dict[str, Any]):
        super().__init__(**kwargs)

    def __call__(self, nlp: English, frames: list[Image.Image], questions: list[str]) -> list[Image.Image]:
        # Parse objects from questions - reuse parsing method from TargetObjectCounterFilter
        object_parse_results = TargetObjectCounterFilter.parse_sentences_for_target_objects(nlp, questions)
        detection_results, padded_images = self.run_detection(object_parse_results, frames)
        new_frames = []

        # Iterate in parallel through spatial parse results, detection results, frames, and padded frames
        for detection_results_single, frame, frame_padded in zip(detection_results, frames, padded_images):
            mask = np.ones((frame_padded.height, frame_padded.width), dtype=np.float64)

            boxes = detection_results_single["boxes"]
            bboxes = boxes.cpu().numpy() # (# boxes, 4)

            if bboxes.shape[0] > 0:
                # Mask out the areas for this noun
                for bbox in bboxes:
                    # Make sure bbox doesn't go beyond image edges
                    if bbox[1] < 0:
                        bbox[1] = 0
                    if bbox[3] >= mask.shape[0]:
                        bbox[3] = mask.shape[0] - 1

                    if bbox[0] < 0:
                        bbox[0] = 0
                    if bbox[2] >= mask.shape[1]:
                        bbox[2] = mask.shape[1] - 1     

                    # Set the area within the bounding box to 0
                    # Note the order: (ymin:ymax, xmin:xmax)
                    mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 0.0
                                    
                # Apply mask strength to black parts of resulting mask
                mask = 1.0 - (1 - mask) * MASK_STRENGTH
                        
                # Apply mask and undo padding of masked/cropped image to pass to VLM later
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                new_frame = np.array(frame_padded) * mask
                new_frame = Image.fromarray(new_frame.astype(np.uint8))
                new_height = new_frame.width / frame.width * frame.height
                new_frame = new_frame.crop((0, 0, new_frame.width - 1, new_height))
            else:
                new_frame = frame
                    
            new_frames.append(new_frame)

        return new_frames


class VisualFilterTypes(Enum):
    Spatial = "spatial"
    Spatial_NoRephrase = "spatial_norephrase"
    Contrastive_Region = "contrastive_region"
    # Don't include target object counter here because it won't be used in the same way as other filters