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
parser.add_argument("--print_prompts", action="store_true", help="Pass this argument to print some sample prompts during execution (for debugging purposes).")
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

# TODO: left off here in updating script
print(f"({worker_index}) Beginning iterative VQA inference...")
all_questions = []
all_candidate_questions = []
all_candidate_questions_scores = []
all_candidate_questions_sources = []
all_scores = []
all_answers = []
all_answer_probs = []
all_success_probs = []
all_success_probs_negated = []

all_example_ids = []
all_procedures = []
all_labels = []

cache_path = os.path.join(this_results_dir, f"cached_outputs{worker_index}.pkl")
is_complete = False
last_batch_idx = -1
if os.path.exists(cache_path):
    is_complete, last_batch_idx, all_questions, all_candidate_questions, all_candidate_questions_scores, all_candidate_questions_sources, all_scores, all_answers, all_answer_probs, all_success_probs, all_success_probs_negated, all_example_ids, all_procedures, all_labels = pickle.load(open(cache_path, "rb"))

batch_idx = None
if not is_complete:
    for batch_idx, batch_examples in tqdm(enumerate(dataset.get_batches(IMAGES_CHUNK_SIZE, 
                                                                        n_workers=n_workers, 
                                                                        worker_index=worker_index,
                                                                        load_frames=False)), 
                                                    desc="running iterative VQA inference"):

        # If already in cache, skip this batch
        if batch_idx <= last_batch_idx:
            continue    

        # Take first frame (expect there to only be one frame)
        batch_procedures = [example.procedure_description for example in batch_examples]
        batch_frames = [Image.open(example.frames[0]) for example in batch_examples]

        this_batch_size = len(batch_examples)

        prompts = [
            f'{USER_START_TOKENS[args.vlm_name]}{IMAGE_TOKENS[args.vlm_name]}{IVQA_PREAMBLE.format(procedure=procedure)}' 
            for procedure in batch_procedures
        ]
        if args.print_prompts:
            pprint(prompts[0])
        questions = [[] for _ in range(this_batch_size)]
        frames = [[] for _ in range(this_batch_size)]
        candidate_questions = [[] for _ in range(this_batch_size)]
        candidate_questions_scores = [[] for _ in range(this_batch_size)]
        candidate_questions_sources = [[] for _ in range(this_batch_size)]
        scores = [[] for _ in range(this_batch_size)]
        answer_probs = [[] for _ in range(this_batch_size)] 
        answers = [[] for _ in range(this_batch_size)]
        success_probs = [[] for _ in range(this_batch_size)]
        success_probs_negated = [[] for _ in range(this_batch_size)]

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
            new_questions_sources = [["vlm"] * len(beam_search_questions) for beam_search_questions in new_questions]

            # Remove duplicate candidates
            keep_idxs = [[question_idx for question_idx, question in enumerate(beam_search_outputs) if question not in beam_search_outputs[:question_idx]] for beam_search_outputs in new_questions]

            # Try to remove any candidates that we've seen before (if we've seen all the candidates before, don't remove any)
            keep_idxs_filtered = [[question_idx for question_idx, question in enumerate(beam_search_outputs) if question_idx in keep_idxs[batch_sub_idx] and question not in questions[batch_sub_idx]] for batch_sub_idx, beam_search_outputs in enumerate(new_questions)]
            keep_idxs = [keep_idxs_filtered[batch_sub_idx] if len(keep_idxs_filtered[batch_sub_idx]) > 0 else keep_idxs[batch_sub_idx] for batch_sub_idx in range(this_batch_size)]

            # Apply kept indices to new questions and their sources
            new_questions = [[new_questions[batch_sub_idx][question_idx] for question_idx in this_keep_idxs] for batch_sub_idx, this_keep_idxs in enumerate(keep_idxs)]
            new_questions_sources = [[new_questions_sources[batch_sub_idx][question_idx] for question_idx in this_keep_idxs] for batch_sub_idx, this_keep_idxs in enumerate(keep_idxs)]

            # Save all candidates from beam search
            for batch_sub_idx in range(len(candidate_questions)):
                candidate_questions[batch_sub_idx].append(new_questions[batch_sub_idx])
                candidate_questions_sources[batch_sub_idx].append(new_questions_sources[batch_sub_idx])

            # TODO: question_selection_strategy is not an arg anymore - but we do need to be able to calculate coherence metrics still for supervision
            ################################################################################################################################################

            # Select best candidate question from pool
            if args.question_selection_strategy == "likelihood":

                # First recalculate likelihood of each cleaned up question in iterative VQA context

                # (this is mostly important when we're using ICL injected questions)
                # NOTE: for some (unknown at this time) reason, LLaVA's LM likelihoods calculated during beam search (returned by output.scores) do not
                # match the likelihoods we get from a forward pass (which is unlike other models, e.g., T5). As such, we need to recompute completion
                # log likelihoods no matter what when using the "likelihood" candidate selection strategy. 
                # 
                # While this doesn't seem to affect the ordering of contextually generated candidates, the scores used here are slightly different. It's also 
                # worth noting that this step is especially important when ranking ICL-generated candidates among contextually generated candidates, as they may 
                # reasonably have higher likelihood than beam search candidates, but we need to be sure scores are calculated consistently. Further, since 
                # questions aren't "cleaned up" in initial generation, scores can be affected by this and at minimum need to be clipped at "?" token that ends
                # each question.
                # 
                # Some relevant discussion on this issue here: https://discuss.huggingface.co/t/compute-log-probabilities-of-any-sequence-provided/11710/10
                if "-t5-" not in args.vlm_name:
                    generation_scores = compute_completion_log_likelihoods(lm, tokenizer, [prompt.replace(IMAGE_TOKENS[args.vlm_name], "") for prompt in prompts_q], new_questions, batch_size=args.generation_batch_size)
                else:
                    generation_scores = compute_completion_log_likelihoods_encoder_decoder(lm, tokenizer, [prompt.replace(IMAGE_TOKENS[args.vlm_name], "") for prompt in prompts_q], new_questions, batch_size=args.generation_batch_size)

                # Select most likely question (first one in list)
                selected_questions = []
                new_scores = []
                for batch_sub_idx, (beam_search_questions, beam_search_scores) in enumerate(zip(new_questions, generation_scores)):                    
                    assert len(beam_search_questions) == len(beam_search_scores), "Expected candidate questions and their scores to have the same shape!"

                    # Save all candidate scores
                    candidate_questions_scores[batch_sub_idx].append(beam_search_scores)

                    candidate_idxs = list(range(len(beam_search_questions)))

                    # Then pick candidate with highest score
                    best_candidate = max(candidate_idxs, key=lambda x: beam_search_scores[x] == max(beam_search_scores))
                    selected_questions.append(beam_search_questions[best_candidate])
                    new_scores.append(beam_search_scores[best_candidate])

                new_questions = selected_questions

            elif args.question_selection_strategy in ["relevance", "informativeness", "coherence"]:
                # Calculate coherence metrics for each candidate question
                nli_outputs = question_coherence_metrics_nli(
                    nli_tokenizer, 
                    nli_model,
                    tokenizer,
                    lm,
                    [procedure for procedure, beam_search_questions in zip(batch_procedures, new_questions) for _ in beam_search_questions],
                    [question for beam_search_questions in new_questions for question in beam_search_questions],
                    previous_questions=[[q for qi, q in enumerate(batch_idx_questions) if batch_idx_answers[qi] != "Unsure"] for batch_idx_questions, batch_idx_answers, beam_search_questions in zip(questions, answers, new_questions) for _ in beam_search_questions],
                    previous_answers=[[a for a in batch_idx_answers if a != "Unsure"] for batch_idx_answers, beam_search_questions in zip(answers, new_questions) for _ in beam_search_questions],
                    rephrase_batch_size=args.generation_batch_size
                )

                # Select best candidate based on coherence metrics
                selected_questions = []
                new_scores = []
                parallel_idx = 0
                ranking_key_mapping = {
                    "relevance": "relevance_marginal",
                    "informativeness": "informativeness_marginal",
                    "coherence": "informativeness_marginal_x_relevance_marginal",
                }
                for batch_sub_idx, beam_search_questions in enumerate(new_questions):
                    this_nli_outputs = [{k: round(float(nli_outputs[k][i]), 3) if type(nli_outputs[k][i]) != str else nli_outputs[k][i] for k in nli_outputs} for i in range(parallel_idx, parallel_idx + len(beam_search_questions))]
                    candidate_questions_scores[batch_sub_idx].append(this_nli_outputs)
                    parallel_idx += len(beam_search_questions)

                    # Use marginal relevance (consistency) and expected informativeness (verifiability) to rank candidates
                    candidate_scores = np.array(
                        [candidate_metrics[ranking_key_mapping[args.question_selection_strategy]] for candidate_metrics in this_nli_outputs]
                    )

                    best_candidate = np.argmax(candidate_scores)
                    selected_questions.append(beam_search_questions[best_candidate])
                    new_scores.append(round(float(candidate_scores[best_candidate]), 6))
                
                new_questions = selected_questions
                    
            ################################################################################################################################################

            # Save scores for best questions
            for batch_sub_idx in range(this_batch_size):
                scores[batch_sub_idx].append(new_scores[batch_sub_idx])

            # Save generated questions
            for batch_sub_idx in range(this_batch_size):
                questions[batch_sub_idx].append(new_questions[batch_sub_idx])

            # Run VQA with generated questions (and optional spatial filter)
            prompts_a = [prompt + f' {question}{USER_END_TOKENS[args.vlm_name]}{ASSISTANT_START_TOKENS[args.vlm_name]}A:' for prompt, question in zip(prompts_q, new_questions)]
            if args.print_prompts:
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
            if args.print_prompts:
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

            # Clear out VQA outputs now because they occupy a lot of memory
            del new_answers
            del success_vqa_outputs

        # Update global lists of tracked outputs
        all_questions += questions
        all_candidate_questions += candidate_questions
        all_candidate_questions_scores += candidate_questions_scores
        all_candidate_questions_sources += candidate_questions_sources
        all_scores += scores
        all_answers += answers
        all_answer_probs += answer_probs
        all_success_probs += success_probs
        all_success_probs_negated += success_probs_negated
        all_example_ids += [example.example_id for example in batch_examples]
        all_procedures += [example.procedure_description for example in batch_examples]
        all_labels += [example.mistake_type for example in batch_examples]

        for frame in batch_frames:
            frame.close()
        del batch_frames

        # And cache tracked outputs
        pickle.dump((    
            False,
            batch_idx,
            all_questions, 
            all_candidate_questions, 
            all_candidate_questions_scores, 
            all_candidate_questions_sources,
            all_scores, 
            all_answers, 
            all_answer_probs, 
            all_success_probs,
            all_success_probs_negated,
            all_example_ids,
            all_procedures,
            all_labels,
        ), open(cache_path, "wb"))

# Verify we got correct number of outputs
all_results = [
    all_questions, 
    all_candidate_questions, 
    all_candidate_questions_scores, 
    all_candidate_questions_sources,
    all_scores, 
    all_answers, 
    all_answer_probs, 
    all_success_probs,
    all_success_probs_negated,
    all_example_ids,
    all_procedures,
    all_labels,
]
assert all(len(l) == len(all_results[0]) for l in all_results), f"Expected to get same number of all outputs! ({', '.join([str(len(l)) for l in all_results])})"

# Cache one more time to indicate the generation is finished
if batch_idx is not None:
    pickle.dump((    
        True,
        batch_idx,
        all_questions, 
        all_candidate_questions, 
        all_candidate_questions_scores, 
        all_candidate_questions_sources,
        all_scores, 
        all_answers, 
        all_answer_probs, 
        all_success_probs,
        all_success_probs_negated,
        all_example_ids,
        all_procedures,
        all_labels,
    ), open(cache_path, "wb"))

print(f"({worker_index}) Done running iterative VQA inference!")





print(f"({worker_index}) Done!")