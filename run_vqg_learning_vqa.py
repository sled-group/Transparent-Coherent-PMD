# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
import datetime
import json
import os
import pickle
import shutil

from travel.data.vqg_learning import load_frameVQA_examples, save_vqg_training_examples
from travel.model.grounding import filter_frames_by_target_objects, VisualFilterTypes
from travel.model.vqa import save_vqa_outputs
from travel.model.vqg_learning import FrameVQAMistakeDetectionScorer

parser = argparse.ArgumentParser()
parser.add_argument("--vqg_directory", type=str, required=True, help="Directory where desired frameVQA_examples.json is stored.")
parser.add_argument("--vlm_name", type=str, default="/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/llava-1.5-7b-hf", help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--visual_filter_mode", type=str, required=False, choices=[t.value for t in VisualFilterTypes], help="Visual attention filter mode.")
parser.add_argument("--batch_size", type=int, default=10, help="Batch size for VQA inference. Visual filter batch size is configured in `config.yml`.")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of generating frameVQA examples.")
args = parser.parse_args()

for partition in ["train", "val"]: #, "test"]:
    # Load outputs
    frameVQA_examples = load_frameVQA_examples(args.vqg_directory, partition)
    if "_debug" in args.vqg_directory:
        frameVQA_examples = frameVQA_examples[:20]

    # Load OWL object detector for filtering frames, and filter frames
    # detector_processor = Owlv2Processor.from_pretrained(args.detector_name)
    # detector = Owlv2ForObjectDetection.from_pretrained(args.detector_name, load_in_8bit=True)

    # Clear detector from memory
    # del detector_processor
    # del detector

    scorer = FrameVQAMistakeDetectionScorer(args.vlm_name,
                                            visual_filter_type=VisualFilterTypes(args.visual_filter_mode) if args.visual_filter_mode is not None else None,)

    # Save training examples for VQG in the same folder
    if args.resume_dir is None:
        this_results_dir = os.path.join(args.vqg_directory, "VQA_data_" + args.vlm_name.split("/")[-1])
        if args.visual_filter_mode is not None:
            this_results_dir += f"_{args.visual_filter_mode}"
    else:
        this_results_dir = args.resume_dir
    cache_path = os.path.join(this_results_dir, f"VQA_cache_{partition}.pt")

    vqg_training_examples, vqa_outputs = scorer(frameVQA_examples,
                                                return_vqa_outputs=True,
                                                batch_size=args.batch_size,
                                                cache_path=cache_path)

    save_vqa_outputs([output for sub_output in vqa_outputs for output in sub_output], this_results_dir, partition)
    save_vqg_training_examples(vqg_training_examples, this_results_dir, partition)

shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)
