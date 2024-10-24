# TODO: Finish updating this following vlm_ppo_prototyping.ipynb

from travel import init_travel
init_travel()

import argparse
from collections import defaultdict, Counter
from datasets import Dataset
import json
import numpy as np
import os
from peft import PeftModelForCausalLM, LoraConfig
import pickle
from PIL import Image
from pprint import pprint
import random
import spacy
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer, PhrasalConstraint           
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
import wandb

from travel.constants import RESULTS_DIR, IMAGES_CHUNK_SIZE, DATA_CACHE_DIR
from travel.data.captaincook4d import CaptainCook4DDataset
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.data.vqa import VQAResponse, get_vqa_response_token_ids, VQAOutputs, IMAGE_TOKENS, USER_START_TOKENS, USER_END_TOKENS, ASSISTANT_START_TOKENS, ASSISTANT_END_TOKENS, IVQA_PREAMBLE, IVQA_SUCCESS_QUESTION
from travel.data.vqg import generate_vqg_prompt_icl
from travel.model import simple_lm_prompt_beam_search, simple_vlm_prompt_beam_search, compute_completion_log_likelihoods, compute_completion_log_likelihoods_encoder_decoder, compute_completion_log_likelihoods_vlm
from travel.model.grounding import VisualFilterTypes, ContrastiveRegionFilter, VisualContrastiveFilter, SpatialVisualFilter, AGLAFilter, ImageMaskTypes
from travel.model.metrics import mistake_detection_metrics, question_coherence_metrics_nli, question_coherence_metrics_vlm, generate_det_curve, generate_tiered_metric_curves, entropy, compile_accuracy_and_coherence_metrics
from travel.model.mistake_detection import MISTAKE_DETECTION_THRESHOLDS
from travel.model.nli import NLI_MODEL_PATH, NLI_BATCH_SIZE
from travel.model.ppo_trainer import PerTokenPPOTrainer as PPOTrainer
from travel.model.vqa import run_vqa_with_visual_filter
from travel.model.vqg import cleanup_generated_question


parser = argparse.ArgumentParser()
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=["Salesforce/instructblip-vicuna-7b", "llava-hf/llava-1.5-7b-hf"], help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--task", type=str, default="ego4d_single", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs.")
parser.add_argument("--lora_r", type=int, default=16, help="LoRA r (matrix dimension).")
parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (weight update scaling coefficient).")
parser.add_argument("--max_iterations", type=int, default=10, help="Maximum number of questions to generate before making a final mistake detection decision.")
parser.add_argument("--num_beams", type=int, default=8, choices=list(range(21)), help="Number of beams in beam search.")
parser.add_argument("--num_return_sequences", type=int, default=4, choices=list(range(21)), help="Number of generation candidates to return from beam search. Recommend setting this to be less than number of beams due to generation constraints.")
parser.add_argument("--unsure_range", type=int, default=0.1, help="A VQA output will be considered unsure if the probability of yes and no are within this range of 50 percent (exclusive).")
parser.add_argument("--visual_filter_mode", type=str, required=False, choices=[t.value for t in VisualFilterTypes], help="Visual attention filter mode.")
parser.add_argument("--visual_filter_strength", type=float, required=False, default=1.0, help="Float strength for masks used in visual filters. Depending on the visual filter type, this may be interpreted as a percentage darkness or a Gaussian blur kernel size.")
parser.add_argument("--generation_batch_size", type=int, default=10, help="Batch size for question generation with LM.")
parser.add_argument("--vqa_batch_size", type=int, default=10, help="Batch size for VQA with VLM.")
parser.add_argument("--nli_batch_size", type=int, default=NLI_BATCH_SIZE, help="Batch size for scoring candidate questions with NLI model.")
parser.add_argument("--run_id", type=str, required=False, help="Unique ID for this run, which will be used to create the output directory (and should be shared across any parallel processes).")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of iterative VQA training.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
parser.add_argument("--verbose", action="store_true", help="Pass this argument to display prompts and generations on every batch.")
parser.add_argument("--save_strategy", type=str, choices=["no", "epoch"], default="epoch", help="Save strategy for DPO (either none or epochs). For initial hyperparameter search, can use none to save space.")
args = parser.parse_args()

assert torch.cuda.device_count() == 1, "Iterative VQA requires exactly 1 GPU per process; use `srun` to enable multi-GPU parallelization."

# Get parallelization details from srun if any
if "SLURM_PROCID" in os.environ and "SLURM_NPROCS" in os.environ:
    worker_index = int(os.environ["SLURM_PROCID"])
    n_workers = int(os.environ["SLURM_NPROCS"])
else:
    worker_index = 0
    n_workers = 1
# NOTE: if resuming from a previous run, must have the same number of GPUs as original run

# Set up results directory
if args.resume_dir is None:
    vlm_name = args.vlm_name.split('/')[-1]
    task_name = args.task
    if args.debug:
        task_name += f"_debug{args.debug_n_examples}" if args.task != "captaincook4d" else "_debug"
    this_results_dir = os.path.join(task_name, vlm_name, f"IterativeVQA_q{args.max_iterations}_{task_name}")
    this_results_dir += f"_{vlm_name}"
    this_results_dir += f"_beam{args.num_beams}-{args.num_return_sequences}"
    if args.visual_filter_mode is not None:
        this_results_dir += f"_{args.visual_filter_mode}{args.visual_filter_strength}"
    this_results_dir += f"_{args.run_id}"
    this_results_dir = os.path.join(RESULTS_DIR, "vqg_training", this_results_dir)
    if worker_index == 0 and not os.path.exists(this_results_dir):
        os.makedirs(this_results_dir)
else:
    this_results_dir = args.resume_dir

this_run_id = args.run_id if args.resume_dir is None else args.resume_dir.split("_")[-1]
wandb_run_name = f"PPO_lr{args.learning_rate}_bs{args.batch_size}_e{args.n_epochs}_r{args.lora_r}_alpha{args.lora_alpha}_{this_run_id}"

if worker_index == 0:
    wandb.init(name=wandb_run_name)
    wandb.log({
        "hyperparameters/visual_filter_strength": args.visual_filter_strength if args.visual_filter_mode is not None else 0.0,
        "hyperparameters/batch_size": args.train_batch_size,
        "hyperparameters/learning_rate": args.learning_rate,
        "hyperparameters/n_demonstrations": args.n_demonstrations,
        "hyperparameters/lora_r": args.lora_r,
        "hyperparameters/lora_alpha": args.lora_alpha,
    })


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
                        # lora_dropout=0.1,               # dropout regularization on LoRA weights
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
response_token_ids = get_vqa_response_token_ids(vlm_processor.tokenizer)

# We'll use VLM's LM directly to generate questions
if getattr(vlm, "language_model", None):
    lm = vlm.language_model
else:
    lm = vlm
lm = PeftModelForCausalLM(vlm.language_model, peft_config)
lm = AutoModelForCausalLMWithValueHead.from_pretrained(lm)
tokenizer = vlm_processor.tokenizer
tokenizer.pad_token_id = tokenizer.eos_token_id

# Set up visual filter if needed
visual_filter = None
nlp = None
if args.visual_filter_mode is not None:
    if VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Spatial_NoRephrase:
        visual_filter = SpatialVisualFilter(rephrase_questions=False, mask_strength=args.visual_filter_strength, mask_type=ImageMaskTypes.Darkness, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Spatial_Blur:
        visual_filter = SpatialVisualFilter(rephrase_questions=False, mask_strength=args.visual_filter_strength, mask_type=ImageMaskTypes.Blur, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')            
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Spatial:
        visual_filter = SpatialVisualFilter(rephrase_questions=True, mask_strength=args.visual_filter_strength, mask_type=ImageMaskTypes.Darkness, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')            
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Contrastive_Region:
        visual_filter = ContrastiveRegionFilter(mask_strength=args.visual_filter_strength, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Visual_Contrastive:
        visual_filter = VisualContrastiveFilter(alpha=args.visual_filter_strength, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')            
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.AGLA:
        visual_filter = AGLAFilter(alpha=args.visual_filter_strength, device=f"cuda:0")
        nlp = None
    else:
        raise NotImplementedError(f"Visual filter type {args.visual_filter_mode} is not compatible with iterative VQA!")

# Shared generation kwargs

# kwargs to force question generations to have a "?" and start with words that would typically begin a yes/no question
question_generation_constraints = [    
    PhrasalConstraint(
        [vlm_processor.tokenizer("Is it blue?", add_special_tokens=False).input_ids[-1]]
    ),
]
yes_no_q_tokens = [
    vlm_processor.tokenizer("Is it blue?", add_special_tokens=False).input_ids[0], 
    vlm_processor.tokenizer("Was it blue?", add_special_tokens=False).input_ids[0],
    vlm_processor.tokenizer("Are they blue?", add_special_tokens=False).input_ids[0], 
    vlm_processor.tokenizer("Were they blue?", add_special_tokens=False).input_ids[0],
    vlm_processor.tokenizer("Does it look blue?", add_special_tokens=False).input_ids[0],
    vlm_processor.tokenizer("Do they look blue?", add_special_tokens=False).input_ids[0],
    vlm_processor.tokenizer("Did they look blue?", add_special_tokens=False).input_ids[0],
    vlm_processor.tokenizer("Has the oven turned on?", add_special_tokens=False).input_ids[0],
    vlm_processor.tokenizer("Have the eggs boiled?", add_special_tokens=False).input_ids[0],
    vlm_processor.tokenizer("Had the eggs boiled?", add_special_tokens=False).input_ids[0],
]
begin_suppress_tokens = [t for t in list(range(vlm_processor.tokenizer.vocab_size)) if t not in yes_no_q_tokens]
bad_words_ids = [[vlm_processor.tokenizer("Yes or no?", add_special_tokens=False).input_ids[1]]]

generation_kwargs = {
    "do_sample": False,
    "num_beams": args.num_beams,
    "num_return_sequences": args.num_return_sequences,
    "constraints": question_generation_constraints,
    "begin_suppress_tokens": begin_suppress_tokens,   
    "bad_words_ids": bad_words_ids, 
    "pad_token_id": tokenizer.eos_token_id,
}

# NLI model to score coherence
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)


# Load approopriate training dataset
dataset = None
for retry in range(5):
    print(f"({worker_index}) Loading evaluation dataset (try {retry})...")
    try:
        # TODO: can think about whether we should combine multiple data sources for training
        if MistakeDetectionTasks(args.task) == MistakeDetectionTasks.Ego4D_Single:
            dataset = Ego4DMistakeDetectionDataset(data_split="train", 
                                                   mismatch_augmentation=True,
                                                   multi_frame=False,
                                                   debug_n_examples_per_class=args.debug_n_examples if args.debug else None)
        else:
            raise NotImplementedError(f"Haven't implemented usage of {args.task} dataset yet!")
        break
    except Exception as e:
        print("Encountered error during data loading:")
        pprint(e)
        time.sleep(60)
if dataset is None:
    raise ValueError("Could not load dataset after retrying!")

# Balance the data
dataset.balance_classes()


# Set up PPO trainer
wandb.init(name=wandb_run_name) # initialize wandb
def collator(data): # TODO: this will need to be updated
    return {key: [d[key] for d in data] for key in data[0]}
ppo_config = PPOConfig(
    learning_rate=args.learning_rate,
    batch_size=4,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    remove_unused_columns=False,
    optimize_cuda_cache=True,
    early_stopping=True,
    is_peft_model=True,
    seed=args.random_seed,
)
ppo_trainer = PPOTrainer(
    model=lm,
    ref_model=vlm.language_model, # TODO: don't think this makes sense
    config=ppo_config,
    dataset=dataset,
    tokenizer=vlm_processor.tokenizer,
    data_collator=collator
)


print(f"({worker_index}) Beginning iterative VQA training...")
training_progress_path = os.path.join(this_results_dir, "training_progress.pkl")
if os.path.exists(training_progress_path):
    current_epoch, current_batch_idx = pickle.load(open(training_progress_path, "rb"))
else:
    current_epoch, current_batch_idx = 0, 0
for epoch in tqdm(range(args.n_epochs), desc="epochs"):

    # If already completed this epoch, skip it
    if epoch < current_epoch:
        continue

    for batch_idx, batch_examples in tqdm(enumerate(dataset.get_batches(IMAGES_CHUNK_SIZE, 
                                                                        n_workers=n_workers, 
                                                                        worker_index=worker_index,
                                                                        load_frames=False)), 
                                                    desc=f"epoch {epoch} batches"):

        # If already ran this batch, skip it
        if batch_idx < current_batch_idx:
            continue    

        # Take first frame (expect there to only be one frame)
        batch_procedures = [example.procedure_description for example in batch_examples]
        batch_frames = [Image.open(example.frames[0]) for example in batch_examples]

        this_batch_size = len(batch_examples)

        prompts = [
            f'{USER_START_TOKENS[args.vlm_name]}{IMAGE_TOKENS[args.vlm_name]}{IVQA_PREAMBLE.format(procedure=procedure)}' 
            for procedure in batch_procedures
        ]
        if args.verbose:
            pprint(prompts[0])
        questions = [[] for _ in range(this_batch_size)]
        frames = [[] for _ in range(this_batch_size)]
        answer_probs = [[] for _ in range(this_batch_size)] 
        answers = [[] for _ in range(this_batch_size)]
        success_probs = [[] for _ in range(this_batch_size)]

        # Iteratively generate questions
        for question_idx in tqdm(range(args.max_iterations), desc="running iterative QA"):

            # Generate a question (with beam search so we have several candidates)
            prompts_q = [prompt + f"{ASSISTANT_END_TOKENS[args.vlm_name] if question_idx != 0 else USER_END_TOKENS[args.vlm_name]}{USER_START_TOKENS[args.vlm_name]}Q:" for prompt in prompts]
            new_questions, _ = simple_lm_prompt_beam_search(lm,
                                                            tokenizer,
                                                            [prompt.replace(IMAGE_TOKENS[args.vlm_name], "") for prompt in prompts_q],
                                                            max_new_tokens=20,
                                                            batch_size=max(args.generation_batch_size // (2 ** question_idx), 1),
                                                            generation_kwargs=generation_kwargs)
            new_questions = [[cleanup_generated_question(question) for question in beam_search_questions] for beam_search_questions in new_questions]                                

            # Remove duplicate candidates
            keep_idxs = [[question_idx for question_idx, question in enumerate(beam_search_outputs) if question not in beam_search_outputs[:question_idx]] for beam_search_outputs in new_questions]

            # Try to remove any candidates that we've seen before (if we've seen all the candidates before, don't remove any)
            keep_idxs_filtered = [[question_idx for question_idx, question in enumerate(beam_search_outputs) if question_idx in keep_idxs[batch_sub_idx] and question not in questions[batch_sub_idx]] for batch_sub_idx, beam_search_outputs in enumerate(new_questions)]
            keep_idxs = [keep_idxs_filtered[batch_sub_idx] if len(keep_idxs_filtered[batch_sub_idx]) > 0 else keep_idxs[batch_sub_idx] for batch_sub_idx in range(this_batch_size)]

            # Apply kept indices and grab first one in the list (which should have the highest likelihood)
            new_questions = [[new_questions[batch_sub_idx][question_idx] for question_idx in this_keep_idxs] for batch_sub_idx, this_keep_idxs in enumerate(keep_idxs)]
            new_questions = [beam_search_questions[0] for beam_search_questions in new_questions]

            # Calculate coherence metrics for generated questions
            nli_outputs = question_coherence_metrics_nli(
                nli_tokenizer, 
                nli_model,
                tokenizer,
                lm,
                batch_procedures,
                new_questions,
                previous_questions=[[q for qi, q in enumerate(batch_idx_questions) if batch_idx_answers[qi] != "Unsure"] for batch_idx_questions, batch_idx_answers in zip(questions, answers)],
                previous_answers=[[a for a in batch_idx_answers if a != "Unsure"] for batch_idx_answers in answers],
                rephrase_batch_size=args.generation_batch_size
            )
            relevance = [round(float(nli_outputs['relevance_marginal'][i]), 6) for i in range(this_batch_size)]
            informativeness = [round(float(nli_outputs['informativeness_marginal'][i]), 6) for i in range(this_batch_size)]
            relevance_x_informativeness = [round(float(nli_outputs['informativeness_marginal_x_relevance_marginal'][i]), 6) for i in range(this_batch_size)]

            # Save generated questions
            for batch_sub_idx in range(this_batch_size):
                questions[batch_sub_idx].append(new_questions[batch_sub_idx])

            # Run VQA with generated questions (and optional spatial filter)
            prompts_a = [prompt + f' {question}{USER_END_TOKENS[args.vlm_name]}{ASSISTANT_START_TOKENS[args.vlm_name]}A:' for prompt, question in zip(prompts_q, new_questions)]
            if args.verbose:
                pprint(prompts_a[0])

            # Always exclude dialog history from VQA prompt - it only distracts
            use_prompts_a = [f'{USER_START_TOKENS[args.vlm_name]}{IMAGE_TOKENS[args.vlm_name]}Q: {question}{USER_END_TOKENS[args.vlm_name]}{ASSISTANT_START_TOKENS[args.vlm_name]}A:' for prompt, question in zip(prompts_q, new_questions)]

            new_answers_logits = run_vqa_with_visual_filter(vlm_processor=vlm_processor, 
                                                            vlm=vlm, 
                                                            batch_examples=batch_examples, 
                                                            batch_frames=batch_frames, 
                                                            prompts_a=use_prompts_a, 
                                                            new_questions=new_questions, 
                                                            question_idx=question_idx,
                                                            batch_size=args.vqa_batch_size,
                                                            visual_filter=visual_filter,
                                                            nlp=nlp,
                                                            visual_filter_mode=VisualFilterTypes(args.visual_filter_mode) if visual_filter else None,
                                                            frame_cache_dir=None,
                                                            is_encoder_decoder="-t5-" in args.vlm_name.lower())

            # Gather up VQA outputs (which automatically calculates answer probabilities from logits)
            new_answers = [
                VQAOutputs(
                    task_name=MistakeDetectionTasks(args.task),
                    example_id=example.example_id,
                    procedure_id=example.procedure_id,
                    frame=example.frames[0],
                    prompt=prompt,
                    expected_answer=None,
                    response_token_ids=response_token_ids,
                    logits=logits,
                    question=question,
                ) for logits, example, prompt, question in zip(new_answers_logits, batch_examples, prompts_a, new_questions)
            ]
            new_answers_str = [output.predicted_answer.name if np.abs(output.answer_probs[VQAResponse.Yes] - 0.5) >= args.unsure_range else "Unsure" for output in new_answers]

            # Save answers and their probabilities
            for batch_sub_idx in range(this_batch_size):
                answer_probs[batch_sub_idx].append([round(float(new_answers[batch_sub_idx].answer_probs[VQAResponse(answer_idx)]), 6) for answer_idx in range(2)])
                answers[batch_sub_idx].append(new_answers_str[batch_sub_idx])

            # Update prompts with answers          
            prompts = [prompt + " " + output for prompt, output in zip(prompts_a, new_answers_str)]

            # Ask VLM probability of success
            questions_success = [
                IVQA_SUCCESS_QUESTION.format(procedure=procedure)
                for procedure in batch_procedures
            ]
            prompts_success = [
                prompt + f'{ASSISTANT_END_TOKENS[args.vlm_name]}{USER_START_TOKENS[args.vlm_name]}Q: {question}{USER_END_TOKENS[args.vlm_name]}{ASSISTANT_START_TOKENS[args.vlm_name]}A: '
                for prompt, question in zip(prompts, questions_success)
            ]
            if args.verbose:
                pprint(prompts_success[0])
            success_vqa_outputs = run_vqa_with_visual_filter(vlm_processor=vlm_processor, 
                                                             vlm=vlm, 
                                                             batch_examples=batch_examples, 
                                                             batch_frames=batch_frames, 
                                                             prompts_a=prompts_success, 
                                                             new_questions=questions_success, 
                                                             question_idx=f"{question_idx}_success",
                                                             batch_size=max(args.vqa_batch_size // (2 ** question_idx), 1),
                                                             visual_filter=visual_filter if visual_filter and VisualFilterTypes(args.visual_filter_mode) not in [VisualFilterTypes.Spatial_NoRephrase, VisualFilterTypes.Spatial_Blur] else None, # Don't use spatial filter for SuccessVQA step, since this may remove important information
                                                             nlp=nlp,
                                                             visual_filter_mode=VisualFilterTypes(args.visual_filter_mode) if visual_filter else None,
                                                             frame_cache_dir=None,
                                                             is_encoder_decoder="-t5-" in args.vlm_name.lower())
            success_vqa_outputs = [
                VQAOutputs(
                    task_name=MistakeDetectionTasks(args.task),
                    example_id=example.example_id,
                    procedure_id=example.procedure_id,
                    frame=example.frames[0],
                    prompt=prompt,
                    expected_answer=None,
                    response_token_ids=response_token_ids,
                    logits=logits,
                    question=question,
                ) for logits, example, prompt, question in zip(success_vqa_outputs, batch_examples, prompts_a, new_questions)
            ]               

            # Save success probability for this turn
            for batch_sub_idx in range(this_batch_size):
                success_probs[batch_sub_idx].append(
                    round(float(success_vqa_outputs[batch_sub_idx].answer_probs[VQAResponse.Yes]), 6)
                )

            try:
                wandb.log(
                    {
                        "rewards/relevance_mean": np.mean(relevance),
                        "rewards/informativeness_mean": np.mean(informativeness),
                        "rewards/relevance_x_informativeness_mean": np.mean(relevance_x_informativeness),
                        "rewards/success_correctness_mean": np.mean([sp[-1] for sp in success_probs]),
                    }
                )
            except Exception as e:
                print("Warning: failed to log to wandb!")
                pprint(e)

            # TODO: need to do step with PPO trainer still

            # Clear out VQA outputs now because they occupy a lot of memory
            del new_answers
            del success_vqa_outputs

        #### Save model
        if epoch % 5 == 0 and worker_index == 0 and args.save_strategy == "epoch":
            if not os.path.exists(os.path.join(this_results_dir, f"epoch{epoch}")):
                os.makedirs(os.path.join(this_results_dir, f"epoch{epoch}"))
            ppo_trainer.save_pretrained(os.path.join(this_results_dir, f"epoch{epoch}"))    

        for frame in batch_frames:
            frame.close()
        del batch_frames

print(f"({worker_index}) Done running iterative VQA training!")

#### Save model
if worker_index == 0:
    print(f"({worker_index}) Saving model...")
    ppo_trainer.save_pretrained(this_results_dir)        
    wandb.finish()

print(f"({worker_index}) Done!")