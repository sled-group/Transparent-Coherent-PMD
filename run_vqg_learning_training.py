# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
from collections import defaultdict
from datasets import Dataset
import datetime
import itertools
import os
from peft import get_peft_model, LoraConfig, TaskType
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer

from travel.data.vqg_learning import load_vqg_training_examples

parser = argparse.ArgumentParser()
parser.add_argument("--data_directory", type=str, required=True, help="Directory where desired vqg_training_examples.json is stored.")
parser.add_argument("--lm_name", type=str, default="/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Llama-3-hf/models--meta-llama--Meta-Llama-3-8B/snapshots/b6887ce03ea47d068bf8502ba6ed27f8c5c12a6b", help="Name or path to Hugging Face model for LM. Can be a fine-tuned LM for VQG.")
parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training.")
# parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
args = parser.parse_args()

print("Preparing training and validation data...")
data = {
    "train": load_vqg_training_examples(args.data_directory, "train"),
    "val": load_vqg_training_examples(args.data_directory, "val")
}

# (Save testing data for downstream Ego4D-SuccessVQA evaluation)

# Pair examples based on preference scores
datasets = {}
for partition in data:
    data_paired = defaultdict(list)
    for example in data[partition]:
        data_paired[example.procedure_id].append(example)
    pairs = []
    for procedure_id in data_paired:
        pairs += itertools.combinations(data_paired[procedure_id], 2)

    prompt = []
    chosen = []
    rejected = []
    for ex1, ex2 in tqdm(pairs, "Pairing examples"):
        assert ex1.prompt == ex2.prompt, "Prompts for training pair don't match!"
        prompt.append(ex1.prompt)

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
        prompt = prompt[:10]
        chosen = chosen[:10]
        rejected = rejected[:10]

    dataset = Dataset.from_dict(
        {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
    )
    datasets[partition] = dataset

# Set up LM for training
print("Loading LM...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(args.lm_name,
                                             quantization_config=bnb_config)
# TODO: is there a better option for PEFT on this model?
peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,  # configured for text classification
                         inference_mode=False,        # enable training - for inference, we can pre-compute the weight update matrix
                         r=8,                         # dimension of low-rank matrices
                         lora_alpha=16,               # scaling coefficient of weight update
                         lora_dropout=0.1,            # dropout regularization on LoRA weights
                         bias="all")                  # use LoRA to train "all" biases (alternatives: "none", "lora_only")
model = get_peft_model(model, peft_config)

# TODO: need to better understand how this works
timestamp = datetime.datetime.now()
output_dir_name = f"DPO_outputs_{timestamp.strftime('%Y%m%d%H%M%S')}"
training_args = TrainingArguments(output_dir=os.path.join(args.data_directory, output_dir_name),
                                  per_device_train_batch_size=args.train_batch_size,
                                  num_train_epochs=10,
                                  save_strategy="epoch",
                                  save_only_model=True,
                                  remove_unused_columns=False)
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    beta=0.1,
    max_prompt_length=2 * max([len(p.split()) for p in prompt]),
    max_length=2 * max([len(g.split()) for g in chosen + rejected]),
    train_dataset=datasets["train"],
    eval_dataset=datasets["val"],
    tokenizer=tokenizer,
)
dpo_trainer.train()