# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
import datetime
import json
import os
import shutil
import torch
from transformers import pipeline, BitsAndBytesConfig
from tqdm import tqdm

from travel.constants import RESULTS_DIR, HF_TOKEN
from travel.model.vqg import VQG_DEMONSTRATIONS, generate_vqg_prompt_icl, VQGInputs, run_vqg, load_vqg_outputs
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.data.captaincook4d.constants import RECIPE_STEPS

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="captaincook4d", choices=[task.value for task in MistakeDetectionTasks]) # TODO: support running for Ego4D's evaluation sets
parser.add_argument("--lm_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Name or path to Hugging Face model for LM. Can be a fine-tuned LM for VQG.")
parser.add_argument("--n_demonstrations", type=int, default=5, choices=range(1, len(VQG_DEMONSTRATIONS) + 1), help="Number of demonstrations of VQG for in-context learning. Must be <= the number of demonstrations available in travel.model.vqg.VQG_DEMONSTRATIONS.")
# parser.add_argument("--n_questions_to_generate", type=int, default=2, choices=range(1, len(VQG_DEMONSTRATIONS[0].questions) + 1), help="Number of questions to generate per procedure.")
parser.add_argument("--temperature", type=float, default=0.4, help="Temperature for language generation, i.e., degree of randomness to use in sampling words.")
parser.add_argument("--top_p", type=float, default=0.9, help="top_p for language generation, i.e., top percentage of words to consider in terms of likelihood.")
parser.add_argument("--batch_size", type=int, default=48, help="Batch size for VQG.")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of generating frameVQA examples.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
args = parser.parse_args()

# TODO: we may want to split the CaptainCook4D data by recipe and support that here

# Load LM
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

# Generate prompts for VQG
prompts = []
if MistakeDetectionTasks(args.task) == MistakeDetectionTasks.CaptainCook4D:
    indexed_procedures = RECIPE_STEPS
# TODO: support Ego4D and multiple possible partitions in it

# Prepare output directory
if args.resume_dir is None:
    timestamp = datetime.datetime.now()
    this_results_dir = f"VQG" # Can change this name later if we have other approaches for VQG besides prompting LM with ICL
    if args.debug:
        this_results_dir += f"_debug"
    this_results_dir += f"_{args.lm_name.split('/')[-1]}_icl{args.n_demonstrations}_{timestamp.strftime('%Y%m%d%H%M%S')}"
    this_results_dir = os.path.join(RESULTS_DIR, "vqg", this_results_dir)
    os.makedirs(this_results_dir)
else:
    this_results_dir = args.resume_dir

for procedure_id, step in indexed_procedures.items():
    prompt = generate_vqg_prompt_icl(step, n_demonstrations=args.n_demonstrations)
    prompts.append(
        VQGInputs(
            procedure_id=procedure_id,
            procedure_description=step,
            prompt=prompt    
        )
    )
    if args.debug and len(prompts) >= 10:
        break

# TODO: finish implementing resuming here and parallelization following run_vqg_learning_generation.py - may be needed if we do ego4d evaluations

# Run prompts through LM to generate visual questions (and save VQG outputs)
vqg_outputs = run_vqg(
    lm,
    prompts,
    [int(inp.procedure_id) for inp in prompts],
    batch_size=args.batch_size,
    save_path=this_results_dir,
    vqg_outputs=load_vqg_outputs(this_results_dir) # Will send in partly completed VQG outputs if we have them
)

# Save config and args
shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)