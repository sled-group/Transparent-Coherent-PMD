# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
import concurrent.futures
import datetime
import json
import os
import shutil
import torch
from transformers import pipeline, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer

from travel.constants import RESULTS_DIR, HF_TOKEN
from travel.data.vqg import VQG_DEMONSTRATIONS, generate_vqg_prompt_icl, VQGInputs, save_vqg_inputs, load_vqg_inputs, load_vqg_outputs, save_vqg_outputs
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.data.captaincook4d.constants import RECIPE_STEPS
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.utils import split_list_into_partitions
from travel.model.mistake_detection import NLI_MODEL_PATH
from travel.model.vqg import run_vqg, correct_vqg_outputs_with_nli

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ego4d", choices=[task.value for task in MistakeDetectionTasks]) # TODO: support running for Ego4D's evaluation sets
parser.add_argument("--partition", type=str, required=False, choices=["val", "test"], help="Partition to run VQG on. For some tasks with a consistent set of procedures shared across partitions, this may not be required.")
parser.add_argument("--lm_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Name or path to Hugging Face model for LM. Can be a fine-tuned LM for VQG.")
parser.add_argument("--n_demonstrations", type=int, default=20, choices=range(0, len(VQG_DEMONSTRATIONS) + 1), help="Number of demonstrations of VQG for in-context learning. Must be <= the number of demonstrations available in travel.model.vqg.VQG_DEMONSTRATIONS.")
parser.add_argument("--semi_structured_prompt", action="store_true", help="Pass this argument in the zero-shot setting to prompt the LM in a semi-structured way to ensure output follows the desired format.")
parser.add_argument("--correct_with_nli", action="store_true", help="Pass this argument to correct LM's proposed answers to questions with a pre-trained NLI model.")
parser.add_argument("--temperature", type=float, default=0.4, help="Temperature for language generation, i.e., degree of randomness to use in sampling words.")
parser.add_argument("--top_p", type=float, default=0.9, help="top_p for language generation, i.e., top percentage of words to consider in terms of likelihood.")
parser.add_argument("--batch_size", type=int, default=12, help="Batch size for VQG.")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of generating visual questions.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
args = parser.parse_args()

assert not (args.partition is None and args.task == "ego4d"), f"Need to provide --partition for task {args.task}!"

# Load LM(s)
print("Setting up LM(s)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model_kwargs = {"quantization_config": bnb_config}
n_workers = 1 if torch.cuda.device_count() <= 1 else torch.cuda.device_count()
if n_workers > 1 and args.correct_with_nli:
    raise NotImplementedError("Not supporting multi-GPU parallelization with --correct_with_nli yet.")
    # TODO: convert this script to use srun instead then support correct_with_nli in parallel settings

lms = []
for worker_index in range(n_workers):
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{worker_index}")
    lm = pipeline("text-generation", 
                model=args.lm_name, 
                token=HF_TOKEN,
                model_kwargs=model_kwargs)
    lm.tokenizer.padding_side = "left"
    lm.tokenizer.pad_token_id = lm.model.config.eos_token_id
    if args.temperature == 0.0:
        lm.model.generation_config.temperature = None
        lm.model.generation_config.do_sample = False
        lm.model.generation_config.top_p = None
    else:
        lm.model.generation_config.temperature = args.temperature
        lm.model.generation_config.do_sample = True
        lm.model.generation_config.top_p = args.top_p
    lms.append(lm)

print("Generation config:")
print(lm.model.generation_config)

# Prepare output directory
if args.resume_dir is None:
    timestamp = datetime.datetime.now()
    lm_name = args.lm_name.split('/')[-1]
    task_name = args.task
    if args.debug:
        task_name += f"_debug{args.debug_n_examples}" if args.task != "captaincook4d" else "_debug"
    this_results_dir = os.path.join(task_name, lm_name, f"VQG_{task_name}")
    this_results_dir += f"_{lm_name}_icl{args.n_demonstrations}_{timestamp.strftime('%Y%m%d%H%M%S')}"
    this_results_dir = os.path.join(RESULTS_DIR, "vqg", this_results_dir)
    os.makedirs(this_results_dir)
else:
    this_results_dir = args.resume_dir

# Generate prompts for VQG
prompts_fname = f"prompts_{args.partition}.json" if args.partition is not None else "prompts.json"
if args.resume_dir is None or not os.path.exists(os.path.join(this_results_dir, prompts_fname)):
    
    # Starting from scratch - load dataset and iterate through its procedures to generate prompts
    if MistakeDetectionTasks(args.task) == MistakeDetectionTasks.CaptainCook4D:
        # TODO: we may want to split the CaptainCook4D data by recipe and support that here
        indexed_procedures = RECIPE_STEPS.items
    elif MistakeDetectionTasks(args.task) == MistakeDetectionTasks.Ego4D:
        dataset = Ego4DMistakeDetectionDataset(data_split=args.partition,
                                               mismatch_augmentation=True,
                                               multi_frame=True,
                                               debug_n_examples_per_class=args.debug_n_examples if args.debug else None)
        indexed_procedures = dataset.get_all_procedures
    elif MistakeDetectionTasks(args.task) == MistakeDetectionTasks.Ego4D_Single:
        dataset = Ego4DMistakeDetectionDataset(data_split=args.partition, 
                                               mismatch_augmentation=True,
                                               multi_frame=False,
                                               debug_n_examples_per_class=args.debug_n_examples if args.debug else None)        
        indexed_procedures = dataset.get_all_procedures

    # Generate prompts - one per example
    prompts = []
    seen_video_clips = {}
    for procedure_id, step in indexed_procedures():
        prompt = generate_vqg_prompt_icl(step, n_demonstrations=args.n_demonstrations)
        prompts.append(
            VQGInputs(
                procedure_id=procedure_id,
                procedure_description=step,
                prompt=prompt    
            )
        )
        # if args.debug and len(prompts) >= 100:
        #     break
    print(f"{len(prompts)} prompts generated for {args.partition} partition")

    # Save prompts
    save_vqg_inputs(prompts, os.path.join(this_results_dir, prompts_fname))
else: 
    # Load prompts
    prompts = load_vqg_inputs(os.path.join(this_results_dir, prompts_fname))
    print(f"{len(prompts)} prompts loaded for {args.partition} partition")

# Load combined VQG outputs if we have any
vqg_outputs_fname = f"vqg_outputs_{args.partition}.json" if args.partition is not None else "vqg_outputs.json"
if args.resume_dir is None or not os.path.exists(os.path.join(this_results_dir, vqg_outputs_fname)):
    # Initialize empty dictionary of VQG outputs
    vqg_outputs = {}
else:
    # Load already generated VQG outputs from previous run
    vqg_outputs = load_vqg_outputs(os.path.join(this_results_dir, vqg_outputs_fname))
    print(f"{len(vqg_outputs)} pre-generated VQG outputs loaded for {args.partition} partition")

# Only keep prompts that haven't already been run
prompts = [p for p in prompts if p.procedure_id not in vqg_outputs]
prompt_ids = [p.procedure_id for p in prompts]

if len(prompts) == 0 and not args.correct_with_nli:
    raise ValueError(f"The passed --resume_dir {args.resume_dir} already has all VQG outputs for {args.partition} partition!")

# Run prompts through LM to generate visual questions (and save VQG outputs)
if n_workers == 1:
    print("Running VQG sequentially...")
    if len(prompts) > 0:
        vqg_outputs = run_vqg(
            lm,
            prompts,
            prompt_ids,
            batch_size=args.batch_size,
            save_path=os.path.join(this_results_dir, vqg_outputs_fname),
            vqg_outputs=vqg_outputs # Will send in partly completed VQG outputs if we have them
        )

    # If passed --correct_with_nli, use a pre-trained NLI to correct LM's proposed answers (completely replaces them, later can just make LM only generate questions)
    if args.correct_with_nli:
        print("Correcting VQG outputs with NLI model...")
        del lm
        nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
        nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)
        vqg_outputs = correct_vqg_outputs_with_nli(vqg_outputs, nli_model, nli_tokenizer)
        del nli_model, nli_tokenizer

        # Save corrected VQG outputs
        save_vqg_outputs(vqg_outputs, os.path.join(this_results_dir, vqg_outputs_fname))

else:
    # Split up remaining prompts and prompt IDs by GPU
    prompts_split = split_list_into_partitions(prompts, n_workers)
    prompt_ids_split = split_list_into_partitions(prompt_ids, n_workers)

    # Then gather up any existing VQG outputs and exclude them from prompts and prompt IDs
    all_worker_vqg_outputs = []
    all_worker_prompt_ids = []
    all_worker_prompts = []
    worker_vqg_outputs_paths = [os.path.join(this_results_dir, vqg_outputs_fname).replace(".json", f"_{i}.json") for i in range(n_workers)]
    for i in range(n_workers):
        worker_vqg_outputs = load_vqg_outputs(worker_vqg_outputs_paths[i])
        worker_prompt_ids = [pid for pid in prompt_ids_split[i] if pid not in worker_vqg_outputs]
        worker_prompts = [prompt for pid, prompt in zip(prompt_ids_split[i], prompts_split[i]) if pid not in worker_vqg_outputs]

        all_worker_vqg_outputs.append(worker_vqg_outputs)
        all_worker_prompt_ids += worker_prompt_ids
        all_worker_prompts += worker_prompts

    # Combine VQG outputs we have so far and save
    for vqgo in all_worker_vqg_outputs:
        vqg_outputs |= vqgo
    save_vqg_outputs(vqg_outputs, os.path.join(this_results_dir, vqg_outputs_fname))

    # Redistribute prompts across workers in case one GPU ran slower than others in the previous run
    assert len(all_worker_prompt_ids) == len(all_worker_prompts), "Size issue with collecting prompts and prompt IDs from earlier run workers!"
    all_worker_prompts = split_list_into_partitions(all_worker_prompts, n_workers)
    all_worker_prompt_ids = split_list_into_partitions(all_worker_prompt_ids, n_workers)

    print(f"Loaded prompts split across {n_workers} GPUs: " + ", ".join([str(len(p)) for p in all_worker_prompts]))

    if not all(len(p) == 0 for p in all_worker_prompts):
        # Parallelize across GPUs
        print(f"Parallelizing VQG across {n_workers} GPUs...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            partitions = list(executor.map(run_vqg, 
                                           lms,
                                           all_worker_prompts,
                                           all_worker_prompt_ids,
                                           [args.batch_size for _ in range(n_workers)],
                                           worker_vqg_outputs_paths,
                                           [{} for _ in range(n_workers)]
                                          )
            )
    else:
        partitions = worker_vqg_outputs

    # Combine all VQG outputs from workers
    for p in partitions:
        vqg_outputs |= p

    # Save combined VQG outputs
    save_vqg_outputs(vqg_outputs, os.path.join(this_results_dir, vqg_outputs_fname))

print(f"{len(vqg_outputs)} VQG outputs generated!")

# Save config and args
shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)