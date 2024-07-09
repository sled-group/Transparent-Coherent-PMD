# Need this call at the beginning of every script to set random seeds and set the HF cache
import itertools
from travel import init_travel
init_travel()

import argparse
from collections import defaultdict
from datasets import Dataset
import datetime
import numpy as np
import os
from peft import LoraConfig, prepare_model_for_kbit_training
from pprint import pprint
import random
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig, SFTTrainer, SFTConfig

from travel.data.vqg import generate_vqg_prompt, generate_vqg_prompt_icl, VQG_DEMONSTRATIONS
from travel.data.vqg_learning import load_vqg_training_examples, VQGTrainingExample

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model. Method from https://github.com/DylanJamesZapzalka/eecs-545-project/blob/main/phi2-sft-code.ipynb.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

@record
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_path", type=str, required=True, help="File or directory where training vqg_training_examples.json is stored.")
    parser.add_argument("--val_data_path", type=str, required=False, help="File or directory where validation vqg_training_examples.json is stored. If not passed, will be set to the same as the training data directory.")
    parser.add_argument("--lm_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Name or path to Hugging Face model for LM. Can be a fine-tuned LM for VQG.")
    parser.add_argument("--run_id", type=str, help="Unique ID for this run. Usually a timestamp string, e.g., 20240708165001.")
    parser.add_argument("--resume_dir", type=str, help="Path to output directory from previous run to resume from (starts from last checkpoint).")
    parser.add_argument("--training_mode", type=str, default="DPO", choices=["DPO", "SFT"], help="Which mode of training to run (SFT or DPO)." )
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--n_demonstrations", type=int, default=20, choices=range(0, len(VQG_DEMONSTRATIONS) + 1), help="Number of demonstrations of VQG for in-context learning. Must be <= the number of demonstrations available in travel.model.vqg.VQG_DEMONSTRATIONS.")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta parameter for training.")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA r (matrix dimension).")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (weight update scaling coefficient).")
    parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
    parser.add_argument("--save_strategy", type=str, choices=["no", "epoch"], default="epoch", help="Save strategy for DPO (either none or epochs). For initial hyperparameter search, can use none to save space.")
    args = parser.parse_args()

    # Load local rank from torchrun if we have it (for debugging purpose)
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.val_data_path is None:
        args.val_data_path = args.training_data_path

    print("World size:", world_size)
    print("Host address:", os.environ['MASTER_ADDR'] if 'MASTER_ADDR' in os.environ else None)
    print("Host port:", os.environ['MASTER_PORT'] if 'MASTER_PORT' in os.environ else None)
    print("Local rank:", local_rank)
    print("Global rank:", global_rank)
    print("Devices:", torch.cuda.device_count())

    print(f"({global_rank}) Preparing training and validation data...")
    data = {
        "train": load_vqg_training_examples(args.training_data_path, "train"),
        "val": load_vqg_training_examples(args.val_data_path, "val")
    }
    # If a path to a .json file was provided, change it to the directory the .json file is in
    if args.training_data_path.endswith(".json"):
        args.training_data_path = "/".join(args.training_data_path.split("/")[:-1])
    if args.val_data_path.endswith(".json"):
        args.val_data_path = "/".join(args.val_data_path.split("/")[:-1])
    
    # (Save testing data for downstream Ego4D-SuccessVQA evaluation)

    # Pair examples based on preference scores
    datasets = {}
    for partition in data:
        
        # First gather all the questions and scores we collected by procedure ID
        data_paired = {}
        for example in data[partition]:
            if example.procedure_id not in data_paired:
                data_paired[example.procedure_id] = defaultdict(list)

            question_set = sorted([example.questions[0].strip().lower() + " " + str(example.expected_answers[0]), example.questions[1].strip().lower() + " " + str(example.expected_answers[1])])
            question_set = (question_set[0], question_set[1])
            data_paired[example.procedure_id][question_set].append(example)
        
        # Then average scores across duplicate question sets for each procedure
        new_data_paired = {}
        for procedure_id in data_paired:
            new_question_set_data = []
            for question_set in data_paired[procedure_id]:
                average_score = np.mean([ex.preference_score for ex in data_paired[procedure_id][question_set]])

                # Take first training example for this question set and reassign the score
                new_ex = data_paired[procedure_id][question_set][0]
                new_ex.preference_score = average_score
                new_question_set_data.append(new_ex)

            new_data_paired[procedure_id] = new_question_set_data
        data_paired = new_data_paired

        pairs = []
        for procedure_id in data_paired:

            # if partition != "train":

            # Randomly split the list into two halves and create pairs (want smaller, varied data for evaluation)
            random.shuffle(data_paired[procedure_id])
            split_point = len(data_paired[procedure_id]) // 2
            p1 = data_paired[procedure_id][:split_point]
            p2 = data_paired[procedure_id][split_point:]
            if len(p2) > len(p1):
                p1 += [random.choice(data_paired[procedure_id])]
            assert len(p1) == len(p2), "Halves of paired outputs aren't the same size!"
            pairs += [(tp1, tp2) for tp1, tp2 in zip(p1,p2)]

            # else:
            #     # Take all combinations of questions sets for this procedure ID (just want more data for training)
            #     pairs += itertools.combinations(data_paired[procedure_id], 2)

        prompt = []
        chosen = []
        rejected = []
        for ex1, ex2 in tqdm(pairs, "Pairing examples"):
            if not args.debug:
                assert ex1.procedure_description == ex2.procedure_description, f"Procedures for training pair don't match!\n\n{ex1.procedure_description}\n\n{ex2.procedure_description}"
            # prompt.append(generate_vqg_prompt(ex1.procedure_description))
            # prompt.append(ex1.prompt)
            prompt.append(generate_vqg_prompt_icl(ex1.procedure_description, args.n_demonstrations))

            gen1 = "\n".join([f"{qi+1}. {question}" for qi, question in enumerate(ex1.questions)])
            gen2 = "\n".join([f"{qi+1}. {question}" for qi, question in enumerate(ex2.questions)])

            if ex1.preference_score > ex2.preference_score:
                chosen.append(gen1)
                rejected.append(gen2)
            else:
                chosen.append(gen2)
                rejected.append(gen1)

        # Cut down data if debug mode
        if args.debug:
            prompt = prompt[:100]
            chosen = chosen[:100]
            rejected = rejected[:100]

        if args.training_mode == "DPO":
            dataset = Dataset.from_dict({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })
        elif args.training_mode == "SFT":
            # TODO: should this actually just take one best question set per prompt so we aren't training the model on competing examples?
            dataset = Dataset.from_dict({
                    "prompt": prompt + prompt,
                    "completion": chosen + rejected,
                })

        datasets[partition] = dataset

    for p in datasets:
        print(f"{p} data partition: {len(datasets[p])} examples")

    # Set up LM for training
    print(f"({global_rank}) Loading LM...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name, add_eos_token=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.lm_name, device_map="auto",
                                                 quantization_config=bnb_config, 
                                                 trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)

    if global_rank == 0:
        pprint(model.__dict__)
        print_trainable_parameters(model)
        print(f"Memory footprint: {model.get_memory_footprint() / 1e9} GB")

    if not getattr(model, "peft_config", None):
        peft_config = LoraConfig(task_type="CAUSAL_LM",  # configured for causal LM
                                inference_mode=False,           # enable training - for inference, we can pre-compute the weight update matrix
                                r=args.lora_r,                           # dimension of low-rank matrices
                                lora_alpha=args.lora_alpha,                  # scaling coefficient of weight update
                                target_modules="all-linear",
                                lora_dropout=0.1,               # dropout regularization on LoRA weights
                                bias="none")                     # use LoRA to train "all" biases (alternatives: "none", "lora_only")
    else:
        peft_config = model.peft_config['default']


    # Set up output directory, training args, and wandb
    if args.resume_dir is None:
        lm_name = args.lm_name.split('/')[-1] if "SFT" not in args.lm_name else model.peft_config['default'].base_model_name_or_path.split('/')[-1]
        output_dir_name = f"{args.training_mode}_{args.run_id}"
        if "SFT" in args.lm_name and args.training_mode == "DPO":
            output_dir_name = f"{args.lm_name.split('/')[-1]}-{output_dir_name}"
        output_dir_name = os.path.join(lm_name, output_dir_name)
        if args.debug:
            output_dir_name += "_debug"
        this_results_dir = os.path.join(args.training_data_path, output_dir_name)
        wandb_run_name = f"{output_dir_name}_lr{args.learning_rate}_{'_'.join(args.training_data_path.split('/')[-2:])}"
    else:
        # Recover original output directory and wandb run name from resume dir
        this_results_dir = args.resume_dir
        output_dir_name = "/".join(this_results_dir.split("/")[-2:])
        wandb_run_name = f"{output_dir_name}_lr{args.learning_rate}_{'_'.join(args.training_data_path.split('/')[-2:])}"

    config_class = DPOConfig if args.training_mode == "DPO" else SFTConfig
    training_args = config_class(output_dir=this_results_dir,
                                      per_device_train_batch_size=args.train_batch_size,
                                      per_device_eval_batch_size=args.eval_batch_size,
                                      learning_rate=args.learning_rate,
                                    #   optim='paged_adamw_8bit', # TODO: this might be causing error resuming training
                                      bf16=True,
                                      num_train_epochs=args.n_epochs,
                                      gradient_accumulation_steps=1,
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
                                      ddp_find_unused_parameters=False)

    if args.training_mode == "DPO":
        trainer = DPOTrainer(
            model,
            args=training_args,
            beta=args.dpo_beta,
            max_prompt_length=int(1.5 * max([len(p.split()) for p in prompt])),
            max_length=int(1.5 * max([len(g.split()) for g in chosen + rejected])),
            train_dataset=datasets["train"],
            eval_dataset=datasets["val"],
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
    elif args.training_mode == "SFT":
        trainer = SFTTrainer(
            model=model,
            train_dataset=datasets["train"],
            eval_dataset=datasets["val"],
            peft_config=peft_config,
            max_seq_length=int(1.5 * max([len(p.split()) + len(g.split()) for p, g in zip(prompt + prompt, chosen + rejected)])),
            packing=True,
            tokenizer=tokenizer,
            args=training_args,         
        )        

    print(f"({global_rank}) Starting model training...")
    trainer.train(resume_from_checkpoint=args.resume_dir is not None)

    print("Saving best model...")
    trainer.save_model(this_results_dir)

if __name__ == "__main__":
    main()