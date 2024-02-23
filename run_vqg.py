import argparse
import datetime
import json
import os
import shutil
import torch
from tqdm import tqdm

from travel.constants import MODEL_CACHE_DIR, RESULTS_DIR, HF_TOKEN
from travel.model.vqg import VQG_DEMONSTRATIONS, generate_vqg_prompt_icl, VQGOutputs, save_vqg_outputs, parse_vqg_outputs
from travel.data.mistake_detection import MistakeDetectionTasks, get_cutoff_time_by_proportion
from travel.data.captaincook4d import CaptainCook4DDataset
from travel.data.captaincook4d.constants import RECIPE_STEPS

os.environ['HF_HOME'] = MODEL_CACHE_DIR
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

# TODO: may need to reform prompts for recipe steps to include more information from the recipe - previous steps, ingredients, or recipe name?
# TODO: does there need to be a single target object for VQG?
# TODO: increase number of questions to 3?

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="captaincook4d", choices=[task.value for task in MistakeDetectionTasks])
parser.add_argument("--lm_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Name or path to Hugging Face model for LM. Can be a fine-tuned LM for VQG.")
parser.add_argument("--n_demonstrations", type=int, default=5, choices=range(1, len(VQG_DEMONSTRATIONS) + 1), help="Number of demonstrations of VQG for in-context learning. Must be <= the number of demonstrations available in travel.model.vqg.VQG_DEMONSTRATIONS.")
# parser.add_argument("--n_questions_to_generate", type=int, default=2, choices=range(1, len(VQG_DEMONSTRATIONS[0].questions) + 1), help="Number of questions to generate per procedure.")
parser.add_argument("--temperature", type=float, default=0.4, help="Temperature for language generation, i.e., degree of randomness to use in sampling words.")
parser.add_argument("--top_p", type=float, default=0.9, help="top_p for language generation, i.e., top percentage of words to consider in terms of likelihood.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
args = parser.parse_args()

# Gemma consumes more GPU memory
if "gemma" in args.lm_name:
    batch_size = 1
    model_kwargs = {"device_map": "auto"}
    device = None

else:
    batch_size = 8
    model_kwargs = []
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
lm.model.generation_config.top_p = args.top_p
if args.temperature == 0.0:
    lm.model.generation_config.temperature = None
    lm.model.generation_config.do_sample = False
else:
    lm.model.generation_config.temperature = args.temperature
    lm.model.generation_config.do_sample = True

print("Generation config:")
print(lm.model.generation_config)

prompts = []
if MistakeDetectionTasks(args.task) == MistakeDetectionTasks.CaptainCook4D:
    indexed_procedures = RECIPE_STEPS

for procedure_id, step in indexed_procedures.items():
    prompt = generate_vqg_prompt_icl(step, n_demonstrations=args.n_demonstrations)
    prompts.append({"procedure_id": procedure_id, "step": step, "prompt": prompt})
    if args.debug and len(prompts) >= 10:
        break

# Run prompts through LM to generate visual questions
vqg_outputs = {}
prompt_idx = 0
with torch.no_grad():
    for out in tqdm(lm(KeyDataset(prompts, "prompt"), 
                        batch_size=8, 
                        max_new_tokens=128, 
                        return_full_text=False, 
                        truncation="do_not_truncate"),
                    desc="running VQG",
                    total=len(prompts)):
        inp = prompts[prompt_idx]

        procedure_id = int(inp['procedure_id'])
        step = inp['step']

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

        vqg_outputs[procedure_id] = output
        prompt_idx += 1

# Save VQG outputs, config, and args
timestamp = datetime.datetime.now()
this_results_dir = f"VQG" # Can change this name later if we have other approaches for VQG besides prompting LM with ICL
if args.debug:
    this_results_dir += f"_debug"
this_results_dir += f"_{args.lm_name.split('/')[-1]}_icl{args.n_demonstrations}_{timestamp.strftime('%Y%m%d%H%M%S')}"
this_results_dir = os.path.join(RESULTS_DIR, "vqg", this_results_dir)
os.makedirs(this_results_dir)

save_vqg_outputs(vqg_outputs, this_results_dir)

shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)