import dataclasses
import spacy
from spacy.lang.en import English
import torch
from tqdm import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import Optional
import yaml

from travel.data.mistake_detection import MistakeDetectionDataset
from travel.data.utils.image import get_preprocessed_image
from travel.model.vqg import VQGOutputs

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

OWL_THRESHOLD = config["data"]["owl_threshold"] # Directory to cache model outputs and other temporary data

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
    
# TODO: this needs more testing and fine-tuning
def parse_question_for_spatial_attention_filter(nlp: English, question: str) -> tuple[bool, str]:
    """
    Parses a question for spatial relations that can be visually abstracted with the spatial attention filter.

    :param nlp: spaCy pipeline. Initialize with `spacy.load("en_core_web_sm", disable=["lemmatizer"])`
    :param question: A yes/no question about an image (e.g., are there any cherry tomatoes in the bowl?).
    :return: 
    """
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

    return (look_at_noun, target_noun)