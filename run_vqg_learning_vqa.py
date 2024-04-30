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
from travel.data.vqg_learning import load_frameVQA_examples
from travel.model.grounding import filter_frames_by_target_objects
from travel.model.vqa import VQG2VQA_PROMPT_TEMPLATES, save_vqa_outputs
from travel.model.vqg_learning import FrameVQAMistakeDetectionScorer, save_vqg_training_examples

parser = argparse.ArgumentParser()
parser.add_argument("--vqg_directory", type=str, required=True, help="Directory where desired frameVQA_examples.pkl is stored.")
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=list(VQG2VQA_PROMPT_TEMPLATES.keys()), help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--detector_name", type=str, default="google/owlv2-base-patch16", help="Name or path to HuggingFace OWL model for object detection. Must be compatible with Owlv2ForObjectDetection model.")
args = parser.parse_args()

# Load outputs
frameVQA_examples = load_frameVQA_examples(args.vqg_directory)
print("LAST EXAMPLE:")

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
                                            batch_size=8) # TODO: this may cause nans

# Save metrics, preds, DET curve, config file (which may have some parameters that vary over time), and command-line arguments
timestamp = datetime.datetime.now()
this_results_dir = f"VQG2VQA"
if args.debug:
    this_results_dir += f"_debug"
this_results_dir += f"_{args.vlm_name.split('/')[-1]}_{timestamp.strftime('%Y%m%d%H%M%S')}"
this_results_dir = os.path.join(RESULTS_DIR, "vqa_mistake_detection", this_results_dir)
os.makedirs(this_results_dir)

save_vqa_outputs(vqa_outputs, this_results_dir)
save_vqg_training_examples(vqg_training_examples, this_results_dir)
pickle.dump(vqg_training_examples, open(os.path.join(this_results_dir, "vqg_training_examples.pkl"), "rb"))

shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)
