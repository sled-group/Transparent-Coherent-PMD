# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
from collections import defaultdict
import datetime
import json
import os
import numpy as np
import pickle
import shutil
import torch
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

from travel.constants import RESULTS_DIR, HF_TOKEN
from travel.model.vqg import VQG_DEMONSTRATIONS, generate_vqg_prompt_icl, VQGOutputs, save_vqg_outputs, parse_vqg_outputs
from travel.data.mistake_detection import MistakeDetectionTasks, get_cutoff_time_by_proportion
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.vqg_learning import FrameVQAMistakeDetectionExample


parser = argparse.ArgumentParser()
parser.add_argument("--lm_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Name or path to Hugging Face model for LM. Can be a fine-tuned LM for VQG.")
parser.add_argument("--n_demonstrations", type=int, default=5, choices=range(1, len(VQG_DEMONSTRATIONS) + 1), help="Number of demonstrations of VQG for in-context learning. Must be <= the number of demonstrations available in travel.model.vqg.VQG_DEMONSTRATIONS.")
parser.add_argument('--temperatures', nargs='+', type=float, default=[0.0, 0.5, 1.0])
parser.add_argument("--top_p", type=float, default=0.9, help="top_p for language generation, i.e., top percentage of words to consider in terms of likelihood.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
args = parser.parse_args()

# NOTE: we need to think about how to control the LMs used for question generation and answering:
# Start with: LLaMA 2-7B ( -> Vicuna-7B chat fine-tuned) -> LLaVA 1.5

# TODO: need to include Ruixuan's updates to VQG strategies here

# Load Ego4D for mistake detection
dataset = Ego4DMistakeDetectionDataset(data_split="train",
                                       debug_n_examples_per_class=20 if args.debug else None)
print(f"{len(dataset)} Ego4D mistake detection examples loaded")
# TODO: for some reason, dataset.examples is empty after this step

batch_size = 8
model_kwargs = {}
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Load LM
lm = pipeline("text-generation", 
              model=args.lm_name, 
              token=HF_TOKEN,
              device=device,
              model_kwargs=model_kwargs)
lm.tokenizer.padding_side = "left"
lm.tokenizer.pad_token_id = lm.model.config.eos_token_id

print("Generation config:")
print(lm.model.generation_config)

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
print(f"{len(prompts)} prompts generated")

# Run prompts through LM to generate visual questions
vqg_outputs = defaultdict(list)
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
        prompt_idx = 0

        # Generate for each prompt
        for out in tqdm(lm(KeyDataset(prompts, "prompt"), 
                            batch_size=8, 
                            max_new_tokens=128, 
                            return_full_text=False, 
                            truncation="do_not_truncate"),
                        desc="running VQG for training data generation",
                        total=len(prompts)):
            inp = prompts[prompt_idx]

            procedure_id = int(inp['procedure_id'])
            step = inp['procedure_description']

            text = out[0]['generated_text']
            
            # Hack: sometimes output from LLaMA 2 starts with Љ and whitespace characters, and sometimes Љ replaces the first "T" in "Target object:"
            text_fixed = text.replace("Љ", "").strip() 
            if not text_fixed.startswith("Target object:") and ":" in text_fixed:
                text_fixed = "Target object: " + ":".join(text_fixed.split(":")[1:]).strip()
            
            # Parse reported target object and questions and answers
            try:
                output = parse_vqg_outputs(text_fixed, procedure_id, step)
            except:
                print("Error parsing VQG outputs:")
                print(text)
                print('======')
                print(text_fixed)
                raise

            vqg_outputs[procedure_id].append(output)
            prompt_idx += 1

frameVQA_examples = []
current_example_id = 0
for example in dataset:
    for vqg_id, output in enumerate(vqg_outputs[example.procedure_id]):
        frameVQA_examples.append(
            FrameVQAMistakeDetectionExample(
                task_name=MistakeDetectionTasks.Ego4D,
                video_id=example.video_id,
                procedure_id=example.procedure_id,
                example_id=example.example_id,
                frame=example.frames[0], # NOTE: this relies on there only being one frame in the Ego4D examples
                frame_time=example.frames[0], # NOTE: this relies on there only being one frame in the Ego4D examples
                procedure_description=example.procedure_description,
                mistake=example.mistake,
                candidate_question_sets=vqg_outputs[example.procedure_id]
            )
        )

print(f"{len(frameVQA_examples)} examples generated for training VQG")

# Save generated data, config, and args
timestamp = datetime.datetime.now()
this_results_dir = f"VQG_data"
if args.debug:
    this_results_dir += f"_debug"
this_results_dir += f"_{args.lm_name.split('/')[-1]}_icl{args.n_demonstrations}_{timestamp.strftime('%Y%m%d%H%M%S')}"
this_results_dir = os.path.join(RESULTS_DIR, "vqg_learning", this_results_dir)
os.makedirs(this_results_dir)

# TODO: if too space heavy to do this, can save a json file with images in scratch? need to think about what makes sense based on estimated size
# TODO: also may need to save this in unreplicated volume instead
pickle.dump(frameVQA_examples, open(os.path.join(this_results_dir, "frameVQA_examples.pkl"), "wb"))
# TODO: we can't visualize this - maybe make methods to save/load framevqa_examples which handle loading and saving jsons

shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)