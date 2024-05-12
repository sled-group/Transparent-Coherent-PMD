# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
from collections import defaultdict
import datetime
import json
import os
import numpy as np
import shutil
import torch
from tqdm import tqdm
from transformers import pipeline, BitsAndBytesConfig
from transformers.pipelines.pt_utils import KeyDataset

from travel.constants import RESULTS_DIR, HF_TOKEN, CACHE_FREQUENCY
from travel.model.vqg import VQG_DEMONSTRATIONS, generate_vqg_prompt_icl, parse_vqg_outputs, save_vqg_outputs, load_vqg_outputs
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.vqg_learning import FrameVQAMistakeDetectionExample, save_frameVQA_examples

parser = argparse.ArgumentParser()
parser.add_argument("--lm_name", type=str, default="/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Llama-3-hf/models--meta-llama--Meta-Llama-3-8B/snapshots/b6887ce03ea47d068bf8502ba6ed27f8c5c12a6b", help="Name or path to Hugging Face model for LM. Can be a fine-tuned LM for VQG.")
parser.add_argument("--n_demonstrations", type=int, default=5, choices=range(1, len(VQG_DEMONSTRATIONS) + 1), help="Number of demonstrations of VQG for in-context learning. Must be <= the number of demonstrations available in travel.model.vqg.VQG_DEMONSTRATIONS.")
parser.add_argument('--temperatures', nargs='+', type=float, default=[0.0, 0.5, 1.0])
parser.add_argument("--top_p", type=float, default=0.9, help="top_p for language generation, i.e., top percentage of words to consider in terms of likelihood.")
parser.add_argument("--output_dir", type=str, help="Directory name to output data generation results. If not provided, one will be generated.")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of generating frameVQA examples.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
args = parser.parse_args()

assert len(set(args.temperatures)) == len(args.temperatures), "Must pass a list of unique temperatures! Duplicates aren't supported yet."

# NOTE: we need to think about how to control the LMs used for question generation and answering:
# Start with: LLaMA 2-7B ( -> Vicuna-7B chat fine-tuned) -> LLaVA 1.5

# TODO: need to include Ruixuan's updates to VQG strategies here - maybe need a consistent method for VQG so we can reuse in other scripts


print("Setting up LM...")
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

print("Generation config:")
print(lm.model.generation_config)

# Set results directory
if args.resume_dir is None:
    timestamp = datetime.datetime.now()
    this_results_dir = f"VQG_data"
    if args.debug:
        this_results_dir += f"_debug"
    this_results_dir += f"_{args.lm_name.split('/')[-1]}_icl{args.n_demonstrations}_{timestamp.strftime('%Y%m%d%H%M%S')}"
    this_results_dir = os.path.join(RESULTS_DIR, "vqg_learning", this_results_dir)
    os.makedirs(this_results_dir)
else:
    # Resuming from previous run
    this_results_dir = args.resume_dir
    assert os.path.exists(this_results_dir), "Specified cache directory must already exist!"    

# Run data generation for all partitions
for partition in ["train", "val", "test"]:
    # Load Ego4D for mistake detection
    dataset = Ego4DMistakeDetectionDataset(data_split=partition,
                                           mismatch_augmentation=False,
                                           debug_n_examples_per_class=20 if args.debug else None)
    print(f"{len(dataset)} Ego4D mistake detection examples loaded from {partition} partition")

    prompts_fname = f"prompts_{partition}.json"
    vqg_outputs_fname = f"VQG_cache_{partition}.json"
    if args.resume_dir is None or not os.path.exists(os.path.join(this_results_dir, prompts_fname)) or not os.path.exists(os.path.join(this_results_dir, vqg_outputs_fname)):
        # Starting from scratch

        # Generate prompts - one per example
        prompts = []
        seen_video_clips = {}
        for example in tqdm(dataset, desc="generating prompts"):
            if (example.video_id, example.procedure_id) in seen_video_clips:
                continue
            
            prompt = generate_vqg_prompt_icl(example.procedure_description, n_demonstrations=args.n_demonstrations)
            prompts.append(
                {
                    "procedure_id": example.procedure_id,
                    "procedure_description": example.procedure_description,
                    "prompt": prompt,
                }
            )
        print(f"{len(prompts)} prompts generated for {partition} partition")

        # Save prompts
        json.dump(prompts, open(os.path.join(this_results_dir, prompts_fname), "w"))

        # Initialize empty dictionary of VQG outputs
        vqg_outputs = {}
    else: 
        # Load prompts and already generated VQG outputs from previous run
        prompts = json.load(open(os.path.join(this_results_dir, prompts_fname)))
        vqg_outputs = load_vqg_outputs(os.path.join(this_results_dir, vqg_outputs_fname))

    # Run prompts through LM to generate visual questions
    # TODO: consider making this a method to reuse in VQG script
    with torch.no_grad():
        # Try all temperatures
        for temperature in tqdm(args.temperatures, desc="temperatures"):
            if temperature == 0.0:
                lm.model.generation_config.temperature = None
                lm.model.generation_config.do_sample = False
                lm.model.generation_config.top_p = None
            else:
                lm.model.generation_config.temperature = temperature
                lm.model.generation_config.do_sample = True
                lm.model.generation_config.top_p = args.top_p

            this_prompt_idxs = [i for i in range(len(prompts)) if f"{temperature}_{i}" not in vqg_outputs]
            this_prompts = [prompts[i] for i in this_prompt_idxs]

            # Generate for each prompt
            for prompt_idx, inp, out in tqdm(zip(this_prompt_idxs,
                                                    this_prompts,
                                                    lm(KeyDataset(this_prompts, "prompt"), 
                                                    batch_size=8, 
                                                    max_new_tokens=128, 
                                                    return_full_text=False, 
                                                    truncation="do_not_truncate")),
                                                desc="running VQG",
                                                total=len(this_prompts)):

                procedure_id = int(inp['procedure_id'])
                step = inp['procedure_description']

                text = out[0]['generated_text']
                
                # Hack: sometimes output from LLaMA 2 starts with Љ and whitespace characters, and sometimes Љ replaces the first "T" in "Target object:"
                text_fixed = text.replace("Љ", "").strip() 
                if not text_fixed.startswith("Target object:") and ":" in text_fixed:
                    text_fixed = "Target object: " + ":".join(text_fixed.split(":")[1:]).strip()
                
                # Parse reported target object and questions and answers
                try:
                    try:
                        output = parse_vqg_outputs(text_fixed, procedure_id, step)
                    except:
                        print("Warning: failed to parse a VQG output.")
                        continue
                except:
                    print("Error parsing VQG outputs:")
                    print(text)
                    print('======')
                    print(text_fixed)
                    raise

                vqg_outputs[f"{temperature}_{prompt_idx}"] = output

                if prompt_idx % CACHE_FREQUENCY == 0:
                    print("Saving progress...")
                    save_vqg_outputs(vqg_outputs, os.path.join(this_results_dir, vqg_outputs_fname))

                prompt_idx += 1

    # Save progress one last time after completion
    save_vqg_outputs(vqg_outputs, os.path.join(this_results_dir, vqg_outputs_fname))

    # Reorganize VQG outputs by procedure ID
    vqg_outputs_new = defaultdict(list)
    for key in vqg_outputs:
        key_parts = key.split("_")
        temperature, prompt_idx = float(key_parts[0]), int(key_parts[1])
        vqg_outputs_new[vqg_outputs[key].procedure_id].append(vqg_outputs[key])
    vqg_outputs = vqg_outputs_new

    frameVQA_examples = []
    for example in dataset:
        frameVQA_examples.append(
            FrameVQAMistakeDetectionExample(
                task_name=MistakeDetectionTasks.Ego4D,
                video_id=example.video_id,
                procedure_id=example.procedure_id,
                example_id=example.example_id,
                frame=example.frames[0],
                frame_time=example.frame_times[0],
                procedure_description=example.procedure_description,
                mistake=example.mistake,
                prompt=generate_vqg_prompt_icl(example.procedure_description, n_demonstrations=args.n_demonstrations),
                candidate_question_sets=vqg_outputs[example.procedure_id]
            )
        )

    print(f"{len(frameVQA_examples)} {partition} examples generated for VQG training")

    # Save generated data, config, and args
    save_frameVQA_examples(frameVQA_examples, this_results_dir, partition)

# TODO: move the below 2 steps to a method
shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)