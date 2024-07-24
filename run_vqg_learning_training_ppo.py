# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

from accelerate import Accelerator
import argparse
from datasets import Dataset
import numpy as np
import os
from peft import LoraConfig
import pickle
from PIL import Image
from pprint import pprint
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
import wandb

from travel.constants import HF_TOKEN, RESULTS_DIR, DATA_CACHE_DIR, RANDOM_SEED
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.mistake_detection import NLI_HYPOTHESIS_COMPLETION_TEMPLATE, MistakeDetectionTasks
from travel.data.vqa import VQAResponse
from travel.data.vqg import generate_vqg_prompt_icl, VQG_DEMONSTRATIONS, VQGOutputs
from travel.data.vqg_learning import FrameVQAMistakeDetectionExample
from travel.model.grounding import VisualFilterTypes
from travel.model.mistake_detection import NLI_MODEL_PATH
from travel.model.ppo_trainer import PerTokenPPOTrainer as PPOTrainer
from travel.model.vqg import parse_vqg_outputs
from travel.model.vqg_learning import FrameVQAMistakeDetectionScorer

def run_nli(nli_tokenizer, nli_model, premise_hypothesis_pairs):
    with torch.no_grad():
        x = nli_tokenizer.batch_encode_plus(premise_hypothesis_pairs, 
                                                return_tensors='pt',
                                                padding="longest",
                                                truncation='only_first')
        logits = nli_model(**x.to(nli_model.device))[0]
        logits = logits.cpu()
        logits = logits[:,[0,2]] # Take logits for contradiction and entailment only
    return logits.softmax(dim=1)

def pad_and_stack(tensors, pad_value):
    # Determine the maximum length of the tensors
    max_length = max(tensor.size(0) for tensor in tensors)

    # Pad each tensor to the maximum length
    padded_tensors = []
    for tensor in tensors:
        pad_size = max_length - tensor.size(0)
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=pad_value)
        padded_tensors.append(padded_tensor)
    
    # Stack the padded tensors
    stacked_tensors = torch.stack(padded_tensors)

    return stacked_tensors

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Name or path to Hugging Face model for LM. Can be a fine-tuned LM for VQG.")
    # parser.add_argument("--temperature", type=float, default=0.4, help="Temperature for language generation, i.e., degree of randomness to use in sampling words.")
    # parser.add_argument("--top_p", type=float, default=0.9, help="top_p for language generation, i.e., top percentage of words to consider in terms of likelihood.")
    parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="Name or path to Hugging Face model for VLM.")
    parser.add_argument("--visual_filter_mode", type=str, required=False, choices=[t.value for t in VisualFilterTypes], help="Visual attention filter mode.")
    parser.add_argument("--visual_filter_strength", type=float, required=False, default=1.0, help="Float strength for masks used in visual filters. Depending on the visual filter type, this may be interpreted as a percentage darkness or a Gaussian blur kernel size.")
    parser.add_argument("--run_id", type=str, help="Unique ID for this run. Usually a timestamp string, e.g., 20240708165001.")
    parser.add_argument("--resume_dir", type=str, help="Path to output directory from previous run to resume from (starts from last checkpoint).")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--n_demonstrations", type=int, default=20, choices=range(0, len(VQG_DEMONSTRATIONS) + 1), help="Number of demonstrations of VQG for in-context learning. Must be <= the number of demonstrations available in travel.model.vqg.VQG_DEMONSTRATIONS.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r (matrix dimension).")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (weight update scaling coefficient).")
    parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
    parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
    parser.add_argument("--verbose", action="store_true", help="Pass this argument to display prompts and generations on every batch.")
    parser.add_argument("--save_strategy", type=str, choices=["no", "epoch"], default="epoch", help="Save strategy for DPO (either none or epochs). For initial hyperparameter search, can use none to save space.")
    args = parser.parse_args()

    # Load local rank from torchrun if we have it (for debugging purpose)
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    print("World size:", world_size)
    print("Host address:", os.environ['MASTER_ADDR'] if 'MASTER_ADDR' in os.environ else None)
    print("Host port:", os.environ['MASTER_PORT'] if 'MASTER_PORT' in os.environ else None)
    print("Local rank:", local_rank)
    print("Global rank:", global_rank)
    print("Devices:", torch.cuda.device_count())
    
    # Initialize DDP
    # assert torch.cuda.device_count() == 2, "PPO must be run with 2 GPUs per process!"
    if world_size > 1:
        dist.init_process_group(backend='gloo')

    # Set up LM for training
    print(f"({global_rank}) Loading LM and feedback models...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name, use_fast=True, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    peft_config = LoraConfig(task_type="CAUSAL_LM",  # configured for causal LM
                            inference_mode=False,           # enable training - for inference, we can pre-compute the weight update matrix
                            r=args.lora_r,                           # dimension of low-rank matrices
                            lora_alpha=args.lora_alpha,                  # scaling coefficient of weight update
                            # target_modules="all-linear",
                            # lora_dropout=0.1,               # dropout regularization on LoRA weights
                            bias="none")                     # use LoRA to train "all" biases (alternatives: "none", "lora_only")
    device_map = {"": Accelerator().local_process_index}
    lm = AutoModelForCausalLMWithValueHead.from_pretrained(args.lm_name, 
                                                           device_map=device_map,
                                                           peft_config=peft_config, 
                                                           quantization_config=bnb_config, 
                                                           trust_remote_code=True,
                                                           token=HF_TOKEN)
    # pprint(lm.__dict__)
    # lm.pretrained_model = prepare_model_for_kbit_training(lm.pretrained_model)
    # generation_kwargs = {
    #     "min_length": -1,
    #     "do_sample": True if args.temperature > 0.0 else False,
    #     "temperature": None if args.temperature == 0.0 else args.temperature,
    #     "top_p": None if args.temperature == 0.0 else args.top_p,
    #     "max_new_tokens": 64,
    # }
    # lm = lm.bfloat16().cuda()
    generation_kwargs = {
        "min_length": -1, # don't ignore the EOS token (see above)
        # "top_k": 0.0, # no top-k sampling
        # "top_p": 1.0, # no nucleus sampling
        # "temperature": 1.0,
        "do_sample": False, # yes, we want to sample
        "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": 40, # specify how many tokens you want to generate at most
    }    
    # TODO: play around with this more?

    # Set up online sources of feedback: NLI model and VLM (possibly with visual filter)

    # Set up NLI model for online feedback
    # torch.cuda.set_device(1)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)

    # Set up VLM-powered VQA scorer for online feedback (with optional visual filter)
    scorer = FrameVQAMistakeDetectionScorer(args.vlm_name,
                                            visual_filter_type=VisualFilterTypes(args.visual_filter_mode) if args.visual_filter_mode is not None else None,
                                            visual_filter_strength=args.visual_filter_strength,
                                            vlm_device=0,
                                            visual_filter_device=0 if args.visual_filter_mode is not None else None)

    # Set up output directory, training args, and wandb
    if args.resume_dir is None:
        lm_name = args.lm_name.split('/')[-1]
        output_dir_name = f"PPO_{args.run_id}"
        output_dir_name = os.path.join(lm_name, output_dir_name)
        output_dir_name += f"_icl{args.n_demonstrations}"
        vlm_name = args.vlm_name.split('/')[-1]
        output_dir_name += f"_{vlm_name}"
        if args.visual_filter_mode is not None:
            output_dir_name += f"_{args.visual_filter_mode}{args.visual_filter_strength}"
        if args.debug:
            output_dir_name += f"_debug{args.debug_n_examples}"
        this_results_dir = os.path.join(RESULTS_DIR, "vqg_learning/PPO", output_dir_name)
        wandb_run_name = f"{output_dir_name}_lr{args.learning_rate}"
        if not os.path.exists(this_results_dir) and global_rank == 0:
            os.makedirs(this_results_dir)        
    else:
        # Recover original output directory and wandb run name from resume dir
        this_results_dir = args.resume_dir
        assert os.path.exists(this_results_dir), "Could not find resume directory!"
        output_dir_name = "/".join(this_results_dir.split("/")[-2:])
        wandb_run_name = f"{output_dir_name}_lr{args.learning_rate}"

    if global_rank == 0:
        wandb.init(name=wandb_run_name)
        wandb.log({
            # "hyperparameters/temperature": args.temperature,
            # "hyperparameters/top_p": args.top_p,
            "hyperparameters/visual_filter_strength": args.visual_filter_strength if args.visual_filter_mode is not None else 0.0,
            "hyperparameters/batch_size": args.train_batch_size,
            "hyperparameters/learning_rate": args.learning_rate,
            "hyperparameters/n_demonstrations": args.n_demonstrations,
            "hyperparameters/lora_r": args.lora_r,
            "hyperparameters/lora_alpha": args.lora_alpha,
        })

    # Prepare training examples from Ego4D mistake detection dataset
    print(f"({global_rank}) Preparing training data...")
    dataset = Ego4DMistakeDetectionDataset(data_split="train",
                                           mismatch_augmentation=True,
                                           multi_frame=False,
                                           debug_n_examples_per_class=args.debug_n_examples if args.debug else None)
    dataset_path = os.path.join(DATA_CACHE_DIR, "ppo_training_dataset")
    if args.debug:
        dataset_path += f"_debug{args.debug_n_examples}"
    if not os.path.exists(dataset_path):
        ppo_dataset = []
        for example_dir in tqdm(dataset.example_dirs, desc="Preparing data"):
            example = dataset.load_example_from_file(example_dir)
            assert len(example.frames) == 1, "VQG PPO training is only supported for single-frame mistake detection examples."
            prompt = generate_vqg_prompt_icl(example.procedure_description, args.n_demonstrations)
            ppo_dataset.append({
                "example_id": example.example_id,
                "example_dir": example_dir,
                "procedure_id": example.procedure_id,
                "procedure_description": example.procedure_description,
                "prompt": prompt,
            })
        ppo_dataset = Dataset.from_list(ppo_dataset)
        def tokenize(sample):
            sample["query_tensors"] = tokenizer.encode(sample["prompt"], return_tensors="pt")[0]
            return sample
        ppo_dataset = ppo_dataset.map(tokenize, batched=False)    
        ppo_dataset.set_format(type="torch")
        ppo_dataset.save_to_disk(dataset_path=dataset_path) 
    else:
        ppo_dataset = Dataset.load_from_disk(dataset_path=dataset_path)
    print(f"train data partition: {len(ppo_dataset)} examples")

    print(f"({global_rank}) {len(dataset)} Ego4D mistake detection examples loaded from train partition")

    # Set up PPO trainer
    print(f"({global_rank}) Beginning PPO training...")
    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}    
    ppo_config = PPOConfig(
        model_name=args.lm_name,
        learning_rate=args.learning_rate,
        batch_size=args.train_batch_size,
        mini_batch_size=args.train_batch_size // 2,
        gradient_accumulation_steps=2,
        remove_unused_columns=False,
        optimize_cuda_cache=True,
        early_stopping=True,
        is_peft_model=True,
        seed=RANDOM_SEED,
    )
    ppo_trainer = PPOTrainer(
        model=lm,
        ref_model=None,
        config=ppo_config,
        dataset=ppo_dataset,
        tokenizer=tokenizer,
        data_collator=collator
    )
    if not ppo_trainer.is_peft_model:
        raise ValueError("PEFT model did not successfully get loaded by PPO trainer!")

    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[1] # this should be 13 for LLaMA 2
    for epoch in tqdm(range(args.n_epochs), f"({global_rank}) epoch"):
        for batch_idx, batch in enumerate(tqdm(ppo_trainer.dataloader, desc=f"({global_rank}) batch")):
            # if batch_idx == 0:
            #     keep_batch = batch
            # else:
            #     batch = keep_batch
            
            this_batch_size = len(batch["procedure_description"])

            if args.verbose:
                print("\nPrompts:")
                pprint(batch['prompt'][0])
                pprint(batch["query_tensors"][0].shape)

            # Generate and parse questions
            query_tensors = batch["query_tensors"]
            try:
                if not args.verbose:
                    response_tensors = ppo_trainer.generate(
                        query_tensors,
                        return_prompt=False,
                        generate_ref_response=False,
                        **generation_kwargs
                    )
                else:
                    response_tensors, ref_response_tensors = ppo_trainer.generate(
                        query_tensors,
                        return_prompt=False,
                        generate_ref_response=True,
                        **generation_kwargs
                    )
            except Exception as e:
                # If we have an error (e.g., nans during generation), skip to the next batch
                print("Warning: Encountered error during generation!")
                pprint(e)
                continue
            # response_tensors = lm.generate(
            #     pad_and_stack(query_tensors, tokenizer.pad_token_id).to(lm.pretrained_model.device),
            #     **generation_kwargs,
            # )
            query_lengths = [q.shape[-1] for q in query_tensors]
            response_texts = tokenizer.batch_decode(response_tensors)
            response_texts = [text.replace("Љ", "").strip() for text in response_texts] # Hack: sometimes output from LLaMA 2 starts with Љ and whitespace characters
            if args.verbose:
                print("\nResponses (model and ref):")
                pprint(response_texts[0])
                ref_response_texts = tokenizer.batch_decode(ref_response_tensors)
                pprint(ref_response_texts[0])
                pprint(response_tensors[0].shape)
            vqg_outputs = []
            bad_idxs = []
            for text_idx, text in enumerate(response_texts):
                try:
                    vqg_output = parse_vqg_outputs(
                        generated_language=text,
                        procedure_id=batch['procedure_id'][text_idx],
                        procedure_description=batch['procedure_description'][text_idx],
                    )    
                except:
                    # LM generated something we couldn't parse; make a placeholder
                    bad_idxs.append(text_idx)
                    vqg_output = VQGOutputs(
                        procedure_id = batch['procedure_id'][text_idx],
                        procedure_description=batch['procedure_description'][text_idx],
                        questions=["Is it?", "Is it?"],
                        answers_str=["Yes", "Yes"],
                    )
                    print("Warning: could not parse generated questions!")
                vqg_outputs.append(vqg_output)
            bad_idxs = torch.tensor(bad_idxs)

            # Calculate rewards using NLI model and VLM
            premise_questions_expected = [[f"{question} {answer.name}" for question, answer in zip(vqg_output.questions, vqg_output.answers)] for vqg_output in vqg_outputs]
            premise_questions_not_expected = [[f"{question} {VQAResponse(1 - answer.value).name}" for question, answer in zip(vqg_output.questions, vqg_output.answers)] for vqg_output in vqg_outputs]
            premise_questions_no = [[f"{question} No" for question in vqg_output.questions] for vqg_output in vqg_outputs]
            premise_questions_yes = [[f"{question} Yes" for question in vqg_output.questions] for vqg_output in vqg_outputs]
            hypothesis_completion = [NLI_HYPOTHESIS_COMPLETION_TEMPLATE.format(procedure=procedure) for procedure in batch['procedure_description']]

            # NLI score 1: relevance for each question 
            # (calculated by how much entailment probability of action success changes based on the answer to each question)
            probs_no = run_nli(nli_tokenizer=nli_tokenizer, 
                               nli_model=nli_model,
                               premise_hypothesis_pairs=[(premise, hypothesis) for premises, hypothesis in zip(premise_questions_no, hypothesis_completion) for premise in premises])
            probs_yes = run_nli(nli_tokenizer=nli_tokenizer, 
                               nli_model=nli_model,
                               premise_hypothesis_pairs=[(premise, hypothesis) for premises, hypothesis in zip(premise_questions_yes, hypothesis_completion) for premise in premises])
            relevance = torch.abs(probs_no[:, 0] - probs_yes[:, 0])
            relevance = relevance.view(this_batch_size, 2).cpu().float()

            # NLI score 2: mistake indication of questions 
            # (expected answer should indicate a success, unexpected answer should indicate a mistake)
            probs_expected = run_nli(nli_tokenizer=nli_tokenizer, 
                                     nli_model=nli_model,
                                     premise_hypothesis_pairs=[(premise, hypothesis) for premises, hypothesis in zip(premise_questions_expected, hypothesis_completion) for premise in premises])
            probs_not_expected = run_nli(nli_tokenizer=nli_tokenizer, 
                                         nli_model=nli_model,
                                         premise_hypothesis_pairs=[(premise, hypothesis) for premises, hypothesis in zip(premise_questions_not_expected, hypothesis_completion) for premise in premises])
            informativeness = (probs_expected[:, 1] + probs_not_expected[:, 0]) / 2.0 # TODO: think about whether there's a better way to calculate besides averaging
            informativeness = informativeness.view(this_batch_size, 2).cpu().float()

            # VLM score: mistake detection effectiveness of question sets
            # (questions should together successfully classify mistake or success in frame)
            frame_vqa_examples = [
                FrameVQAMistakeDetectionExample(
                    task_name=MistakeDetectionTasks.Ego4D,
                    video_id="",
                    procedure_id=example.procedure_id,
                    example_id=example.example_id,
                    frame_path=example.frames[0],
                    frame=Image.open(example.frames[0]),
                    frame_time=example.frame_times[0],
                    procedure_description=example.procedure_description,
                    mistake=example.mistake,
                    prompt=prompt,
                    candidate_question_sets=[vqg_output],
                ) for example, prompt, vqg_output in zip([dataset.load_example_from_file(example_dir, load_frames=False) for example_dir in batch['example_dir']], batch['prompt'], vqg_outputs)
            ]
            with torch.no_grad():
                effectiveness = torch.tensor(scorer(frame_vqa_examples, batch_size=this_batch_size * 2, return_scores_only=True)).view(this_batch_size).float()
            effectiveness = effectiveness.repeat(2, 1).permute(1, 0) # Score from VLM is shared, so assign same value to each question

            # TODO: balance positive and negative mistake detection examples
            # TODO: double check calculation of reward - seems like we're not penalizing model for making malformed outputs
            # TODO: assign rewards for every token in each question? Also assign penalty for bad responses for all tokens
            # TODO: assign effectiveness reward per question to help dig out of poorly prompt engineered questions?
            # TODO: should we have LM also generate "where to look" for mistake to bring object detector into the loop?
            # TODO: ref model generation is always generating the same thing as the main model; need to fix this (forward pass seems fine though)
            # TODO: calculate rewards for ref model and log them?
            # TODO: add another score to check whether generated questions mention objects not in procedure description?
            # TODO: add a third NLI score for whether questions contradict each other (or are redundant/duplicate)?

            assert relevance.shape == informativeness.shape == effectiveness.shape, f"Relevance, informativeness, and effectiveness shapes should be equal: {relevance.shape}, {informativeness.shape}, {effectiveness.shape}"
            reward = (relevance + informativeness + effectiveness) / 3.0

            # Find the indices to apply rewards at (at the first 2 newlines, i.e., where each question is done being generated)
            reward_indices = [(response_tensor == newline_token_id).nonzero()[:2].squeeze(1).cpu() + qt_length for qt_length, response_tensor in zip(query_lengths, response_tensors)]
            reward_indices = torch.stack([torch.tensor([-1, -1]) if ri.shape[0] != 2 else ri for ri in reward_indices]).long() # TODO: make sure this is correct
            if bad_idxs.shape[0] > 0:
                reward[bad_idxs] == -1.0
            assert reward_indices.shape == (this_batch_size, 2)

            if args.verbose:
                pprint("Rewards:")
                print("relevance =", relevance[0])
                print("informativeness =", informativeness[0])
                print("effectiveness =", effectiveness[0])
                print("combined =", reward[0, :])
                print("reward indices =", reward_indices[0, :])
                reward_indices_for_response = [reward_indices[i] - query_lengths[i] for i in range(this_batch_size)]
                tokens_at_reward_indices = [response_tensors[i][reward_indices_for_response[i].long()] if torch.min(reward_indices_for_response[i]) > 0 else torch.tensor([-1, -1]) for i in range(this_batch_size)]
                print("tokens at reward indices =", tokens_at_reward_indices[0])
                
            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, reward, reward_indices)
            # reward = [torch.mean(reward[i, :]) for i in range(reward.shape[0])]
            # stats = ppo_trainer.step(query_tensors, response_tensors, reward)
            ppo_trainer.log_stats(stats, batch, reward, columns_to_log=("prompt", "response"))
            if global_rank == 0:
                try:
                    wandb.log(stats | {"ppo/epoch": epoch, 
                                            "rewards/relevance": np.mean(relevance.cpu().numpy()),
                                            "rewards/informativeness": np.mean(informativeness.cpu().numpy()),
                                            "rewards/effectiveness": np.mean(effectiveness.cpu().numpy()),
                                            "rewards/combined": np.mean(torch.tensor(reward).cpu().numpy())})
                except Exception as e:
                    print("Warning: failed to log to wandb!")
                    pprint(e)

            #### Save model
            if epoch % 5 == 0 and global_rank == 0 and args.save_strategy == "epoch":
                if not os.path.exists(os.path.join(this_results_dir, f"epoch{epoch}")):
                    os.makedirs(os.path.join(this_results_dir, f"epoch{epoch}"))
                ppo_trainer.save_pretrained(os.path.join(this_results_dir, f"epoch{epoch}"))    

    print(f"({global_rank}) Done training!")

    #### Save model
    if global_rank == 0:
        print(f"({global_rank}) Saving model...")
        ppo_trainer.save_pretrained(this_results_dir)        
        wandb.finish()
    

if __name__ == "__main__":
    main()