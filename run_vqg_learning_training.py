# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
from collections import defaultdict
from datasets import Dataset
import datetime
import os
from peft import LoraConfig, prepare_model_for_kbit_training
from pprint import pprint
import random
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig, SFTTrainer, SFTConfig

from travel.data.vqg import generate_vqg_prompt
from travel.data.vqg_learning import load_vqg_training_examples

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
    parser.add_argument("--training_mode", type=str, default="DPO", choices=["DPO", "SFT"], help="Which mode of training to run (SFT or DPO)." )
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter for training.")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs.")
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
        data_paired = defaultdict(list)
        for example in data[partition]:
            data_paired[example.procedure_id].append(example)
        pairs = []
        for procedure_id in data_paired:
            # Randomly split the list into two halves and create pairs
            # (we could also use all pairs but this generates way too much data)
            random.shuffle(data_paired[procedure_id])
            split_point = len(data_paired[procedure_id]) // 2
            p1 = data_paired[procedure_id][:split_point]
            p2 = data_paired[procedure_id][split_point:]
            if len(p2) > len(p1):
                p2 += random.choice(data_paired[procedure_id])
            assert len(p1) == len(p2), "Halves of paired outputs aren't the same size!"
            pairs += [(tp1, tp2) for tp1, tp2 in zip(p1,p2)]
            # pairs += itertools.combinations(data_paired[procedure_id], 2)

        prompt = []
        chosen = []
        rejected = []
        for ex1, ex2 in tqdm(pairs, "Pairing examples"):
            if not args.debug:
                assert ex1.procedure_description == ex2.procedure_description, f"Procedures for training pair don't match!\n\n{ex1.procedure_description}\n\n{ex2.procedure_description}"
            prompt.append(generate_vqg_prompt(ex1.procedure_description))

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
                                r=64,                           # dimension of low-rank matrices
                                lora_alpha=16,                  # scaling coefficient of weight update
                                target_modules="all-linear",
                                lora_dropout=0.1,               # dropout regularization on LoRA weights
                                bias="none")                     # use LoRA to train "all" biases (alternatives: "none", "lora_only")
    else:
        peft_config = model.peft_config['default']


    # Set up output directory, training args, and wandb
    timestamp = datetime.datetime.now()
    lm_name = args.lm_name.split('/')[-1] if "SFT" not in args.lm_name else model.peft_config['default'].base_model_name_or_path.split('/')[-1]
    output_dir_name = f"{args.training_mode}_{timestamp.strftime('%Y%m%d%H%M%S')}"
    if "SFT" in args.lm_name and args.training_mode == "DPO":
        output_dir_name = f"{args.lm_name.split('/')[-1]}-{output_dir_name}"
    output_dir_name = os.path.join(lm_name, output_dir_name)
    if args.debug:
        output_dir_name += "_debug"
    this_results_dir = os.path.join(args.training_data_path, output_dir_name)
    wandb_run_name = f"{output_dir_name}_lr{args.learning_rate}_{'_'.join(args.training_data_path.split('/')[-2:])}"
    config_class = DPOConfig if args.training_mode == "DPO" else SFTConfig
    training_args = config_class(output_dir=this_results_dir,
                                      per_device_train_batch_size=args.train_batch_size,
                                      per_device_eval_batch_size=args.eval_batch_size,
                                      learning_rate=args.learning_rate,
                                      optim='paged_adamw_8bit',
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
            beta=args.beta,
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
    trainer.train()

    print("Saving best model...")
    trainer.save_model(this_results_dir)

if __name__ == "__main__":
    main()