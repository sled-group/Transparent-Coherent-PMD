# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
import concurrent.futures
import json
import os
from PIL import Image
import shutil
import torch
from tqdm import tqdm

from travel.constants import IMAGES_CHUNK_SIZE
from travel.data.utils import split_list_into_partitions
from travel.data.vqg_learning import load_frameVQA_examples, save_vqg_training_examples, FrameVQAMistakeDetectionExample, VQGTrainingExample
from travel.model.grounding import VisualFilterTypes
from travel.model.vqa import save_vqa_outputs, VQAOutputs
from travel.model.vqg_learning import FrameVQAMistakeDetectionScorer

parser = argparse.ArgumentParser()
parser.add_argument("--vqg_directory", type=str, required=True, help="Directory where desired frameVQA_examples.json is stored.")
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--generate_partitions", nargs='+', type=str, default=["train", "val"], help="List of partitions to generate data for.")
parser.add_argument("--visual_filter_mode", type=str, required=False, choices=[t.value for t in VisualFilterTypes], help="Visual attention filter mode.")
parser.add_argument("--batch_size", type=int, default=10, help="Batch size for VQA inference. Visual filter batch size is configured in `config.yml`.")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of generating frameVQA examples.")
args = parser.parse_args()

assert not (args.visual_filter_mode is not None and torch.cuda.device_count() < 2), "Need at least 2 GPUs to run VQA scoring with spatial filter!"

for partition in args.generate_partitions:
    # Load outputs
    print("Loading data...")
    frameVQA_examples = load_frameVQA_examples(args.vqg_directory, partition, load_frames=False)
    if "_debug" in args.vqg_directory:
        frameVQA_examples = frameVQA_examples[:100]
        IMAGES_CHUNK_SIZE = 10

    # Save training examples for VQG in the same folder
    if args.resume_dir is None:
        timestamp = datetime.datetime.now()
        this_results_dir = os.path.join(args.vqg_directory, f"VQA_data_{args.vlm_name.split('/')[-1]}_{timestamp.strftime('%Y%m%d%H%M%S')}")
        if args.visual_filter_mode is not None:
            this_results_dir += f"_{args.visual_filter_mode}"
    else:
        this_results_dir = args.resume_dir
    cache_path = os.path.join(this_results_dir, "VQA_scoring_cache", f"VQA_cache_{partition}.pt")
    if not os.path.exists(os.path.join(this_results_dir, "VQA_scoring_cache")):
        os.makedirs(os.path.join(this_results_dir, "VQA_scoring_cache"))

    # If we have a lot of data, divide it into chunks to conserve memory and parallelize across GPUs if possible
    if len(frameVQA_examples) >= IMAGES_CHUNK_SIZE:
        frameVQA_examples_split = split_list_into_partitions(frameVQA_examples, len(frameVQA_examples) // IMAGES_CHUNK_SIZE)
    else:
        frameVQA_examples_split = [frameVQA_examples]
    print(f"Running VQA scoring in {len(frameVQA_examples_split)} chunk(s): " + ", ".join([str(len(s)) for s in frameVQA_examples_split]))

    # Get ready to run VQA scoring
    def run_vqa_scoring_on_chunk(scorer: FrameVQAMistakeDetectionScorer,
                                 frameVQA_examples_chunk: list[FrameVQAMistakeDetectionExample],
                                 chunk_idx: int) -> tuple[list[VQGTrainingExample], list[VQAOutputs]]:
        """Local method to run VQA scoring on a chunk of data."""

        # Load frames for this chunk
        for example in frameVQA_examples_chunk:
            example.uncache_frame()
        
        # Run VQA on this chunk
        this_vqg_training_examples, this_vqa_outputs = scorer(frameVQA_examples_chunk,
                                                              return_vqa_outputs=True,
                                                              batch_size=args.batch_size,
                                                              cache_path=cache_path.replace(".pt", f"{chunk_idx}.pt"))
        # Cache frames in VQAOutputs to conserve memory
        for outputs in this_vqa_outputs:
            for output in outputs:
                if type(output.frame) == Image.Image:
                    output.cache_frame(this_results_dir)

        return this_vqg_training_examples, this_vqa_outputs

    if torch.cuda.device_count() >= 4 if args.visual_filter_mode is not None else 2:
        # If we have enough GPUs, parallelize
        print("Setting up mistake detection scorers...")
        scorers = []
        # If we have a spatial filter, VLM and spatial filter will be put on separate GPUs in sets of 2
        # If we don't, just put a copy of the VLM on each GPU
        for i in range(0, torch.cuda.device_count(), 2 if args.visual_filter_mode is not None else 1):
            scorer = FrameVQAMistakeDetectionScorer(args.vlm_name,
                                        visual_filter_type=VisualFilterTypes(args.visual_filter_mode) if args.visual_filter_mode is not None else None,
                                        vlm_device=i,
                                        visual_filter_device=i + 1 if args.visual_filter_mode is not None else None)
            scorers.append(scorer)

        print(f"Parallelizing VQA scoring across {torch.cuda.device_count()} GPUs...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            partitions = list(executor.map(run_vqa_scoring_on_chunk, 
                                           [scorers[chunk_idx % len(scorers)] for chunk_idx in range(len(frameVQA_examples_split))],
                                           frameVQA_examples_split,
                                           list(range(len(frameVQA_examples_split)))))
        
        # Compile processed data from chunks
        vqg_training_examples = []
        vqa_outputs = []
        for this_vqg_training_examples, this_vqa_outputs in partitions:
            vqg_training_examples += this_vqg_training_examples
            vqa_outputs += this_vqa_outputs

    else:
        # Otherwise, run through the chunks sequentially
        print("Setting up mistake detection scorer...")
        vlm_device = 0 if torch.cuda.is_available() else None
        if args.visual_filter_mode is not None:
            if torch.cuda.device_count() > 1:
                visual_filter_device = 1
            else:
                visual_filter_device = 0
        else:
            visual_filter_device = None
        scorer = FrameVQAMistakeDetectionScorer(args.vlm_name,
                                                visual_filter_type=VisualFilterTypes(args.visual_filter_mode) if args.visual_filter_mode is not None else None,
                                                vlm_device=vlm_device,
                                                visual_filter_device=visual_filter_device)

        print("Running VQA scoring sequentially...")
        vqg_training_examples = []
        vqa_outputs = []
        for chunk_idx, frameVQA_examples_chunk in enumerate(tqdm(frameVQA_examples_split, desc="chunks")):
            this_vqg_training_examples, this_vqa_outputs = run_vqa_scoring_on_chunk(scorer,
                                                                                    frameVQA_examples_chunk=frameVQA_examples_chunk,
                                                                                    chunk_idx=chunk_idx)
            vqg_training_examples += this_vqg_training_examples
            vqa_outputs += this_vqa_outputs


        

    save_vqa_outputs([output for sub_output in vqa_outputs for output in sub_output], this_results_dir, partition)
    save_vqg_training_examples(vqg_training_examples, this_results_dir, partition)

shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)
