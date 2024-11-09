from travel import init_travel
init_travel()

import argparse
from datasets import Dataset, load_from_disk
import json
import os
from peft import LoraConfig
from pprint import pprint
import random
import shutil
import time
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig           
from trl import DPOConfig, DPOTrainer
import wandb

from travel.constants import RESULTS_DIR, CONFIG_PATH, RANDOM_SEED
from travel.data.vqa import DIALOG_START_TOKENS, USER_START_TOKENS, USER_END_TOKENS, ASSISTANT_START_TOKENS, ASSISTANT_END_TOKENS, IVQA_PREAMBLE


parser = argparse.ArgumentParser()
parser.add_argument("--train_data_path", type=str, help="Path to `outputs_<partition>.json` file which will be used to train LM.")
parser.add_argument("--val_data_path", type=str, help="Path to `outputs_<partition>.json` file which will be used to validate LM.")
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=["Salesforce/instructblip-vicuna-7b", "llava-hf/llava-1.5-7b-hf"], help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs.")
parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta parameter for training.")
parser.add_argument("--lora_r", type=int, default=16, help="LoRA r (matrix dimension).")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (weight update scaling coefficient).")
parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout regularization probability.")
parser.add_argument("--unsure_range", type=int, default=0.1, help="A VQA output will be considered unsure if the probability of yes and no are within this range of 50 percent (exclusive).")
parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training.")
parser.add_argument("--eval_batch_size", type=int, default=24, help="Batch size for evaluation.")
parser.add_argument("--run_id", type=str, required=False, help="Unique ID for this run, which will be used to create the output directory (and should be shared across any parallel processes).")
parser.add_argument("--resume_dir", type=str, help="Path to output directory from previous run to resume from (starts from last checkpoint).")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--save_strategy", type=str, choices=["no", "epoch"], default="epoch", help="Save strategy for DPO (either none or epochs). For initial hyperparameter search, can use none to save space.")
args = parser.parse_args()


# Initialize DDP
dist.init_process_group(backend='gloo')

# Load local rank from torchrun if we have it (for debugging purpose)
local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
worker_index = int(os.environ["RANK"]) if "RANK" in os.environ else 0
world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

print("World size:", world_size)
print("Host address:", os.environ['MASTER_ADDR'] if 'MASTER_ADDR' in os.environ else None)
print("Host port:", os.environ['MASTER_PORT'] if 'MASTER_PORT' in os.environ else None)
print("Local rank:", local_rank)
print("Global rank:", worker_index)
print("Devices:", torch.cuda.device_count())


# Set up results directory
if args.resume_dir is None:
    vlm_name = args.vlm_name.split('/')[-1]
    this_results_dir = os.path.join("DPO", vlm_name, f"DPO_IterativeVQA")
    this_results_dir += f"_{vlm_name}"
    this_results_dir += f"_{args.run_id}"
    this_results_dir = os.path.join(RESULTS_DIR, "vqg_training", this_results_dir)
    if worker_index == 0 and not os.path.exists(this_results_dir):
        os.makedirs(this_results_dir)
else:
    this_results_dir = args.resume_dir

this_run_id = args.run_id if args.resume_dir is None else args.resume_dir.split("_")[-1]
wandb_run_name = f"DPO_IterativeVQA_{args.vlm_name.split('/')[-1]}_{this_run_id}"


# Set up models
print(f"({worker_index}) Setting up models...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
peft_config = LoraConfig(task_type="CAUSAL_LM",  # configured for causal LM
                        inference_mode=False,           # enable training - for inference, we can pre-compute the weight update matrix
                        r=args.lora_r,                           # dimension of low-rank matrices
                        lora_alpha=args.lora_alpha,                  # scaling coefficient of weight update
                        # target_modules="all-linear",
                        lora_dropout=args.lora_dropout,               # dropout regularization on LoRA weights
                        bias="none")                     # use LoRA to train "all" biases (alternatives: "none", "lora_only")

# Load VLM - some VLMs may be under AutoModelForVision2Seq, some may be under AutoModelForCausalLM
try:
    vlm = AutoModelForVision2Seq.from_pretrained(args.vlm_name, quantization_config=bnb_config, trust_remote_code=True)
except Exception as e:
    print("Encountered exception when trying to load model with AutoModelForVision2Seq:")
    pprint(e)
    
    vlm = AutoModelForCausalLM.from_pretrained(args.vlm_name, quantization_config=bnb_config, trust_remote_code=True)
vlm_processor = AutoProcessor.from_pretrained(args.vlm_name, trust_remote_code=True)
vlm_processor.tokenizer.padding_side = "left"

# We'll use VLM's LM directly to generate questions
if getattr(vlm, "language_model", None):
    lm = vlm.language_model
else:
    lm = vlm
pprint(lm.__dict__)
tokenizer = vlm_processor.tokenizer
tokenizer.pad_token_id = tokenizer.eos_token_id


# Load model outputs and process DPO data
print(f"({worker_index}) Loading DPO training and validation data...")
if args.val_data_path is None:
    args.val_data_path = args.training_data_path
datasets = {}

for p, data_path in [("train", args.train_data_path), ("val", args.val_data_path)]:
    processed_data_path = data_path.replace(".json", "_processed_dpo")
    if args.unsure_range > 0.0:
        processed_data_path += f"_ur{args.unsure_range}"
        
    n_loading_failures = 0
    while True:
        if not os.path.exists(processed_data_path):
            if worker_index == 0:
                print(f"({worker_index}) Preprocessing model outputs at {args.train_data_path} for {p} data...")
                dataset = []
                outputs = json.load(open(data_path, "r"))
                for example_id, output in tqdm(outputs.items(), desc=f"processing {p} outputs"):
                    procedure = output['procedure']
                    n_turns = output['final_turn'] + 1
                    
                    prompt = f'{DIALOG_START_TOKENS[args.vlm_name]}{USER_START_TOKENS[args.vlm_name]}{IVQA_PREAMBLE.format(procedure=procedure)}{USER_END_TOKENS[args.vlm_name]}{USER_START_TOKENS[args.vlm_name]}Q:'

                    # Generate an instance from each turn before algorithm termination
                    for turn_idx in range(n_turns):

                        # If the VLM wasn't very sure about the answer to the selected question, omit it from training
                        if args.unsure_range > 0.0 and max(output['answer_probs'][turn_idx]) - 0.5 < args.unsure_range:
                            continue

                        candidate_questions = output['candidate_questions'][turn_idx]
                        candidate_questions_scores = output['candidate_questions_scores'][turn_idx]
                        candidate_questions_scores = [cqs['informativeness_marginal_x_relevance_marginal'] for cqs in candidate_questions_scores]

                        instance_info = {"example_id": example_id, "turn": turn_idx, "prompt": prompt}

                        # Generate an instance from the top question choice and one of the worst ones (picked from the bottom half of scores)
                        candidate_questions_sorted = sorted(list(range(len(candidate_questions))), key=lambda qi: candidate_questions_scores[qi])
                        good_question = candidate_questions[candidate_questions_sorted[-1]]
                        if len(candidate_questions_sorted) > 2:
                            worst_candidate_questions = [candidate_questions[qi] for qi in candidate_questions_sorted[:len(candidate_questions_sorted) // 2]]                     
                            bad_question = random.choice(worst_candidate_questions)
                        elif len(candidate_questions_sorted) == 2:
                            bad_question = candidate_questions[candidate_questions_sorted[0]]
                        else:
                            # There was only one candidate question, so we don't have a rejected
                            continue
                            
                        instance_info |= {
                            "chosen": " " + good_question + USER_END_TOKENS[args.vlm_name],
                            "rejected": " " + bad_question + USER_END_TOKENS[args.vlm_name],
                        }
                        dataset.append(instance_info)

                        # Prepare prompt for next iteration by adding the question and answer we actually got at inference time
                        chosen_question = output['questions'][turn_idx]
                        chosen_answer = output['answers'][turn_idx]
                        prompt += f' {chosen_question}{USER_END_TOKENS[args.vlm_name]}{ASSISTANT_START_TOKENS[args.vlm_name]}A: {chosen_answer}{ASSISTANT_END_TOKENS[args.vlm_name]}{USER_START_TOKENS[args.vlm_name]}Q:'

                # Save resulting data
                dataset = Dataset.from_list(dataset)
                os.makedirs(processed_data_path)
                dataset.save_to_disk(processed_data_path)
                break
            else:
                continue
        else:
            try:
                dataset = load_from_disk(processed_data_path)
                break
            except Exception as e:
                print(f"{worker_index} Encountered error while loading data. Retrying.")
                print(e)
                if n_loading_failures < 5:
                    n_loading_failures += 1
                    time.sleep(10)
                    continue
                else:
                    raise e


    datasets[p] = dataset

# Debug mode configuration
if args.debug:
    args.n_epochs = 1

    # If just doing a quick debug run, select a few examples
    for p in datasets:
        datasets[p].shuffle(seed=RANDOM_SEED)
        datasets[p] = datasets[p].select(range(10))


# Set up DPO trainer
print(f"({worker_index}) Setting up DPO trainer...")
config_class = DPOConfig
training_args = config_class(output_dir=this_results_dir,
                             per_device_train_batch_size=args.train_batch_size,
                             per_device_eval_batch_size=args.eval_batch_size,
                             learning_rate=args.learning_rate,
                             bf16=True,
                             num_train_epochs=args.n_epochs,
                             gradient_accumulation_steps=4,
                             save_strategy=args.save_strategy,
                             save_total_limit=3,
                             save_only_model=False,
                             remove_unused_columns=False,
                             evaluation_strategy="epoch",
                             report_to="wandb",
                             logging_strategy="steps",
                             logging_steps=1 if args.debug else 10,
                             run_name=wandb_run_name,
                             ddp_backend="gloo",
                             ddp_find_unused_parameters=False,
                             warmup_ratio=0.05,
                            #  max_prompt_length=max_prompt_length,
                            #  max_length=max_total_length,
)

trainer = DPOTrainer(
    model=lm,
    args=training_args,
    beta=args.dpo_beta,
    train_dataset=datasets["train"],
    eval_dataset=datasets["val"],
    tokenizer=tokenizer,
    peft_config=peft_config,
) 

print(f"({worker_index}) Starting model training...")
trainer.train(resume_from_checkpoint=args.resume_dir is not None)

# Log hyperparams to wandb
if worker_index == 0:
    wandb.log({
        "hyperparameters/batch_size": args.train_batch_size,
        "hyperparameters/learning_rate": args.learning_rate,
        "hyperparameters/dpo_beta": args.dpo_beta,
        "hyperparameters/lora_r": args.lora_r,
        "hyperparameters/lora_alpha": args.lora_alpha,
        "hyperparameters/lora_dropout": args.lora_dropout,
        "hyperparameters/unsure_range": args.unsure_range,
    })

if worker_index == 0:
    print(f"({worker_index}) Saving best model...")
    trainer.save_model(this_results_dir)

    # Save args and config
    shutil.copy(CONFIG_PATH, os.path.join(this_results_dir, "config.yml"))
    json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)

print(f"({worker_index}) Done!")