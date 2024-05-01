# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
import datetime
import json
import os
import pickle
import shutil

from travel.constants import RESULTS_DIR
from travel.data.vqg_learning import load_frameVQA_examples, save_vqg_training_examples
from travel.model.grounding import filter_frames_by_target_objects
from travel.model.vqa import VQG2VQA_PROMPT_TEMPLATES, save_vqa_outputs
from travel.model.vqg_learning import FrameVQAMistakeDetectionScorer

parser = argparse.ArgumentParser()
parser.add_argument("--vqg_directory", type=str, required=True, help="Directory where desired frameVQA_examples.pkl is stored.")
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=list(VQG2VQA_PROMPT_TEMPLATES.keys()), help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--detector_name", type=str, default="google/owlv2-base-patch16", help="Name or path to HuggingFace OWL model for object detection. Must be compatible with Owlv2ForObjectDetection model.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for VQA inference. For quantized models, a batch size greater than 1 can cause nans.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
args = parser.parse_args()

# Load outputs
frameVQA_examples = load_frameVQA_examples(args.vqg_directory, "train")
if "_debug" in args.vqg_directory:
    frameVQA_examples = frameVQA_examples[:20]

# TODO: introduce spatial filter - maybe need a consistent way to do this so we can reuse in vqa scripts

# Load OWL object detector for filtering frames, and filter frames
# detector_processor = Owlv2Processor.from_pretrained(args.detector_name)
# detector = Owlv2ForObjectDetection.from_pretrained(args.detector_name, load_in_8bit=True)

# Clear detector from memory
# del detector_processor
# del detector

scorer = FrameVQAMistakeDetectionScorer(args.vlm_name)

vqg_training_examples, vqa_outputs = scorer(frameVQA_examples,
                                            return_vqa_outputs=True,
                                            batch_size=args.batch_size) # TODO: this may cause nans

# Save training examples for VQG in the same folder
this_results_dir = args.vqg_directory

save_vqa_outputs([output for sub_output in vqa_outputs for output in sub_output], this_results_dir, "train")
save_vqg_training_examples(vqg_training_examples, this_results_dir, "train")
pickle.dump(vqg_training_examples, open(os.path.join(this_results_dir, "vqg_training_examples.pkl"), "wb"))

shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)
