# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
import datetime
import json
import os
import pickle
import shutil

from travel.constants import CACHE_FREQUENCY
from travel.data.vqg_learning import load_frameVQA_examples, save_vqg_training_examples
from travel.model.grounding import filter_frames_by_target_objects
from travel.model.vqa import save_vqa_outputs
from travel.model.vqg_learning import FrameVQAMistakeDetectionScorer

parser = argparse.ArgumentParser()
parser.add_argument("--vqg_directory", type=str, required=True, help="Directory where desired frameVQA_examples.json is stored.")
parser.add_argument("--partition", type=str, default="train", help="Dataset partition name to generate from.")
parser.add_argument("--vlm_name", type=str, default="/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/llava-1.5-7b-hf", help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--detector_name", type=str, default="google/owlv2-base-patch16", help="Name or path to HuggingFace OWL model for object detection. Must be compatible with Owlv2ForObjectDetection model.")
parser.add_argument("--batch_size", type=int, default=10, help="Batch size for VQA inference. For quantized models, a batch size greater than 1 can cause nans.") # TODO: can we fix nans problem
parser.add_argument("--resume_directory", type=str, help="Path to results directory for previous incomplete run of generating frameVQA examples.")
args = parser.parse_args()

# Load outputs
frameVQA_examples = load_frameVQA_examples(args.vqg_directory, args.partition)
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

# Save training examples for VQG in the same folder
if args.resume_directory is None:
    this_results_dir = os.path.join(args.vqg_directory, "VQA_data_" + args.vlm_name.split("/")[-1])
else:
    this_results_dir = args.resume_directory
cache_path = os.path.join(this_results_dir, f"VQA_cache_{args.partition}.pt")

# TODO: pass caching stuff into here and implement caching
vqg_training_examples, vqa_outputs = scorer(frameVQA_examples,
                                            return_vqa_outputs=True,
                                            batch_size=args.batch_size,
                                            cache_path=cache_path) # TODO: this may cause nans

save_vqa_outputs([output for sub_output in vqa_outputs for output in sub_output], this_results_dir, args.partition)
save_vqg_training_examples(vqg_training_examples, this_results_dir, args.partition)

shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)
