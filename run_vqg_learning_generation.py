# Need this call at the beginning of every script to set random seeds and set the HF cache
import pickle
from time import sleep
from travel import init_travel
init_travel()

import argparse
from collections import defaultdict
import concurrent.futures
from copy import deepcopy
import datetime
import json
import os
import numpy as np
import shutil
import torch
from tqdm import tqdm
from transformers import pipeline, BitsAndBytesConfig

from travel.constants import RESULTS_DIR, HF_TOKEN
from travel.data.vqg import VQG_DEMONSTRATIONS, generate_vqg_prompt_icl, VQGInputs, save_vqg_inputs, load_vqg_inputs, load_vqg_outputs, save_vqg_outputs
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.utils import split_list_into_partitions
from travel.data.vqg_learning import FrameVQAMistakeDetectionExample, save_frameVQA_examples
from travel.model.vqg import run_vqg

parser = argparse.ArgumentParser()
parser.add_argument("--lm_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Name or path to Hugging Face model for LM. Can be a fine-tuned LM for VQG.")
parser.add_argument("--partition", type=str, default="train", choices=["train", "val", "test"], help="List of partitions to generate data for.")
parser.add_argument("--n_demonstrations", type=int, default=20, choices=range(1, len(VQG_DEMONSTRATIONS) + 1), help="Number of demonstrations of VQG for in-context learning. Must be <= the number of demonstrations available in travel.model.vqg.VQG_DEMONSTRATIONS.")
parser.add_argument('--temperatures', nargs='+', type=float, default=[0.0, 0.5, 1.0, 1.0])
parser.add_argument("--top_p", type=float, default=0.9, help="top_p for language generation, i.e., top percentage of words to consider in terms of likelihood.")
parser.add_argument("--batch_size", type=int, default=20, help="Batch size for VQG.")
parser.add_argument("--output_dir", type=str, help="Directory name to output data generation results. If not provided, one will be generated.")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of generating frameVQA examples.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
parser.add_argument("--run_id", type=str, help="Unique ID for this run.")
args = parser.parse_args()

# Split up work by srun processes; if SLURM_PROCID is not accessible, just run all the work here
# NOTE: always run with the same number of parallel processes; we don't support changing the number of processes
if "SLURM_PROCID" in os.environ:
    worker_index = int(os.environ["SLURM_PROCID"])
    n_workers = int(os.environ["SLURM_NPROCS"])

else:
    worker_index = 0
    n_workers = 1

temperatures = []
counts = []
for temp in set(args.temperatures):
    n_temps = sum([1 for t in args.temperatures if t == temp])
    temperatures.append(temp)
    counts.append(n_temps)

# Set up LM
print(f"({worker_index}) Setting up LM...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model_kwargs = {"quantization_config": bnb_config}
lm = pipeline("text-generation", 
            model=args.lm_name, 
            token=HF_TOKEN,
            model_kwargs=model_kwargs)
lm.tokenizer.padding_side = "left"
lm.tokenizer.pad_token_id = lm.model.config.eos_token_id

# Set results directory
if args.resume_dir is None:
    lm_name = args.lm_name.split('/')[-1]
    this_results_dir = os.path.join(lm_name, f"VQG_data")
    if args.debug:
        this_results_dir += f"_debug{args.debug_n_examples}"
    this_results_dir += f"_{lm_name}_icl{args.n_demonstrations}_{'_'.join([str(t) for t in args.temperatures])}"
    this_results_dir = os.path.join(RESULTS_DIR, "vqg_learning", this_results_dir)
    if args.run_id is not None:
        this_results_dir += f"_{args.run_id}"
    if not os.path.exists(this_results_dir):
        os.makedirs(this_results_dir)
else:
    # Resuming from previous run
    this_results_dir = args.resume_dir
    assert os.path.exists(this_results_dir), "Specified resuming directory must already exist!"    

# Load Ego4D for mistake detection
while True:
    try:
        dataset = Ego4DMistakeDetectionDataset(data_split=args.partition,
                                                mismatch_augmentation=True,
                                                debug_n_examples_per_class=args.debug_n_examples if args.debug else None)
        print(f"({worker_index}) {len(dataset)} Ego4D mistake detection examples loaded from {args.partition} partition")
        break
    except:
        continue

# Generate or load prompts
prompts_fname = f"prompts_{args.partition}_{worker_index}.json"
if args.resume_dir is None or not os.path.exists(os.path.join(this_results_dir, prompts_fname)):
    # Starting from scratch

    # Generate prompts - one per example
    prompts = []
    all_procedures = []
    for procedure_id, procedure_description in tqdm(dataset.get_all_procedures(), desc=f"({worker_index}) generating prompts"):
        all_procedures.append((procedure_id, procedure_description))
    all_procedures_split = split_list_into_partitions(all_procedures, n_workers)
    worker_procedures = all_procedures_split[worker_index]

    for procedure_id, procedure_description in worker_procedures:
        prompt = generate_vqg_prompt_icl(procedure_description, n_demonstrations=args.n_demonstrations)
        prompts.append(
            VQGInputs(
                procedure_id=procedure_id,
                procedure_description=procedure_description,
                prompt=prompt
            )
        )
    print(f"({worker_index}) {len(prompts)} prompts generated for {args.partition} partition")

    # Save prompts
    save_vqg_inputs(prompts, os.path.join(this_results_dir, prompts_fname))
else: 
    # Load prompts
    prompts = load_vqg_inputs(os.path.join(this_results_dir, prompts_fname))
    print(f"({worker_index}) {len(prompts)} prompts loaded for {args.partition} partition")

# Load combined VQG outputs if we have any
worker_vqg_outputs_fname = f"VQG_cache_{args.partition}_worker{worker_index}.json"
if args.resume_dir is None or not os.path.exists(os.path.join(this_results_dir, worker_vqg_outputs_fname)):
    # Initialize empty dictionary of VQG outputs
    vqg_outputs = {}
else:
    # Load already generated VQG outputs from previous run
    vqg_outputs = load_vqg_outputs(os.path.join(this_results_dir, worker_vqg_outputs_fname))
    print(f"({worker_index}) {len(vqg_outputs)} pre-generated VQG outputs loaded for {args.partition} partition")

# Run prompts through LM to generate visual questions
# Try all temperatures
for temperature, temp_count in tqdm(zip(temperatures, counts), desc="temperatures"):
    for temp_trial in range(temp_count):
        this_prompt_idxs = [i for i in range(len(prompts)) if f"{temperature}_{temp_trial}_{i}" not in vqg_outputs]
        this_prompt_ids = [f"{temperature}_{temp_trial}_{i}" for i in range(len(prompts)) if f"{temperature}_{temp_trial}_{i}" not in vqg_outputs]
        this_prompts = [prompts[i] for i in this_prompt_idxs]
        print(f"({worker_index}) Loaded {len(this_prompts)} prompts to run VQG for.")
        if len(this_prompts) == 0:
            continue

        # Run VQG
        print("Running VQG...")
        if temperature == 0.0:
            lm.model.generation_config.temperature = None
            lm.model.generation_config.do_sample = False
            lm.model.generation_config.top_p = None
        else:
            lm.model.generation_config.temperature = temperature
            lm.model.generation_config.do_sample = True
            lm.model.generation_config.top_p = args.top_p
        vqg_outputs = run_vqg(lm=lm,
                              inputs=this_prompts,
                              input_ids=this_prompt_ids,
                              save_path=os.path.join(this_results_dir, worker_vqg_outputs_fname),
                              vqg_outputs=vqg_outputs,
                              omit_failed_instances=False)
print(f"{worker_index} Done generating!")

# Combine all VQG outputs from workers (only worker 0 does this)
if n_workers > 1 and worker_index == 0:
    for other_worker_index in range(n_workers):
        if other_worker_index == worker_index:
            continue
        other_worker_vqg_outputs_fname = os.path.join(this_results_dir, f"VQG_cache_{args.partition}_worker{worker_index}.json")
        print(f"({worker_index}) Waiting for worker {other_worker_index} to finish VQG...")
        while True:
            if os.path.exists(other_worker_vqg_outputs_fname):
                other_vqg_outputs = load_vqg_outputs(other_worker_vqg_outputs_fname)
                other_prompts_fname = f"prompts_{args.partition}_{worker_index}.json"
                other_prompts = load_vqg_inputs(os.path.join(this_results_dir, other_prompts_fname))
                if len(other_vqg_outputs) == len(other_prompts) * len(args.temperatures):
                    print(f"({worker_index}) Collected VQG outputs from worker {other_worker_index}.")
                    vqg_outputs |= other_vqg_outputs
                    break
            sleep(10)
            print(f"({worker_index}) Still waiting...")

if worker_index == 0:
    # Reorganize VQG outputs by procedure ID
    vqg_outputs_new = defaultdict(list)
    for key in vqg_outputs:
        key_parts = key.split("_")
        temperature, prompt_idx = float(key_parts[0]), int(key_parts[1])
        if vqg_outputs[key] is not None:
            vqg_outputs_new[vqg_outputs[key].procedure_id].append(vqg_outputs[key])
    vqg_outputs = vqg_outputs_new

    frameVQA_examples = []
    for example in dataset:
        # Recover path where the frame for this example is saved
        frame_path = [d for d in dataset.example_dirs if d.endswith(example.example_id)][0]
        frame_path = os.path.join(frame_path, "frames")
        frame_path = [os.path.join(frame_path, f) for f in os.listdir(frame_path) if f.endswith(".jpg")]
        assert len(frame_path) == 1, "Data generation script expects one frame per example!"
        frame_path = frame_path[0]
        assert os.path.exists(frame_path), "Could not recover frame path!"

        frameVQA_examples.append(
            FrameVQAMistakeDetectionExample(
                task_name=MistakeDetectionTasks.Ego4D,
                video_id=example.video_id,
                procedure_id=example.procedure_id,
                example_id=example.example_id,
                frame_path=frame_path,
                frame=None,
                frame_time=example.frame_times[0],
                procedure_description=example.procedure_description,
                mistake=example.mistake,
                prompt=generate_vqg_prompt_icl(example.procedure_description, n_demonstrations=args.n_demonstrations),
                candidate_question_sets=vqg_outputs[example.procedure_id]
            )
        )

    print(f"({worker_index}) {len(frameVQA_examples)} {args.partition} examples generated for VQG training")

    # Save generated data, config, and args
    save_frameVQA_examples(frameVQA_examples, this_results_dir, args.partition)

    shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
    json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)