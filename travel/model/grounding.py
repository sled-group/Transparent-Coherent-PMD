import dataclasses
import spacy
import torch
from tqdm import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import Optional

from travel.data.mistake_detection import MistakeDetectionDataset
from travel.data.utils.image import get_preprocessed_image
from travel.model.vqg import VQGOutputs

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

        if vqg_outputs is not None:
            all_texts = [f"a photo of {'an' if vqg_outputs[example.procedure_id].target_object[0] in ['a','e','i','o','u'] else 'a'} {vqg_outputs[example.procedure_id].target_object}" for example in dataset for frame in example.frames]
        else:
            # TODO: implement using spaCy to parse objects from recipe step - since there may be multiple for each frame this can change data processing details
            raise NotImplementedError("Can't handle parsing objects from recipe step yet.")
            # all_texts = [ for example in dataset for frame in example.frames]
        
        batch_size = 8
        all_results = []
        all_padded_images = []
        for i in tqdm(range(0, len(all_frames), batch_size), desc="detecting objects"):
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
            results = detector_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.2)
            all_results += results
            all_padded_images += padded_images
            
        texts = all_texts
        results = all_results
        padded_images = all_padded_images

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

        dataset.examples = filtered_examples
        return dataset
    