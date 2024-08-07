from travel import init_travel
init_travel()

import argparse
from collections import defaultdict
import datetime
import json
import numpy as np
import os
import pickle
from PIL import Image
from pprint import pprint
import spacy
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer, PhrasalConstraint

from travel.constants import RESULTS_DIR, IMAGES_CHUNK_SIZE
from travel.data.captaincook4d import CaptainCook4DDataset
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.data.utils.image import resize_with_aspect, CACHED_FRAME_DIMENSION
from travel.data.vqa import VQAResponse, get_vqa_response_token_ids, VQAOutputs
from travel.data.vqg import generate_vqg_prompt_icl
from travel.model import simple_lm_prompt_beam_search, compute_completion_log_likelihoods
from travel.model.grounding import VisualFilterTypes, ContrastiveRegionFilter, VisualContrastiveFilter, SpatialVisualFilter, AGLAFilter, ImageMaskTypes
from travel.model.metrics import mistake_detection_metrics, question_coherence_metrics, generate_det_curve
from travel.model.mistake_detection import MISTAKE_DETECTION_THRESHOLDS
from travel.model.nli import NLI_MODEL_PATH, NLI_BATCH_SIZE
from travel.model.vqa import run_vqa 
from travel.model.vqg import cleanup_generated_question


parser = argparse.ArgumentParser()
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=["llava-hf/llava-1.5-7b-hf"], help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--task", type=str, default="ego4d_single", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--eval_partition", type=str, default="val", choices=["val", "test"])
parser.add_argument("--max_iterations", type=int, default=8, help="Maximum number of questions to generate before making a final mistake detection decision.")
parser.add_argument("--n_icl_demonstrations", type=int, default=0, choices=list(range(21)), help="Pass this argument to generate an extra pool of candidate questions using n in-context VQG examples (doesn't incorporate answers to previous questions).")
parser.add_argument("--question_selection_strategy", type=str, default="likelihood", choices=["likelihood", "consistency", "verifiability", "coherence"], help="Strategy to use to choose question to generate from beam search candidates.")
parser.add_argument("--early_stop_delta", type=int, default=0.1, help="If success probability changes less than this over 3 turns, stop generating questions.")
parser.add_argument("--visual_filter_mode", type=str, required=False, choices=[t.value for t in VisualFilterTypes], help="Visual attention filter mode.")
parser.add_argument("--visual_filter_strength", type=float, required=False, default=1.0, help="Float strength for masks used in visual filters. Depending on the visual filter type, this may be interpreted as a percentage darkness or a Gaussian blur kernel size.")
parser.add_argument("--generation_batch_size", type=int, default=10, help="Batch size for question generation with LM.")
parser.add_argument("--vqa_batch_size", type=int, default=10, help="Batch size for VQA with VLM.")
parser.add_argument("--nli_batch_size", type=int, default=NLI_BATCH_SIZE, help="Batch size for scoring candidate questions with NLI model.")
parser.add_argument("--run_id", type=str, required=False, help="Unique ID for this run, which will be used to create the output directory (and should be shared across any parallel processes).")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of generating frameVQA examples.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
parser.add_argument("--cache_vqa_frames", action="store_true", help="Pass this argument to cache frames in VQA outputs (e.g., to inspect visual filter resuilts). This consumes a lot of disk space for large datasets.")
args = parser.parse_args()

assert torch.cuda.device_count() == 1, "Iterative VQA requires exactly 1 GPU per process; use `srun` to enable multi-GPU parallelization."
if args.cache_vqa_frames and args.visual_filter_mode is None:
    print("Warning: --cache_vqa_frames only applies to frames modified by visual filters (configured through --visual_filter_mode and --visual_filter_strength).")

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
    this_results_dir += f"_{args.question_selection_strategy}"
    if args.n_icl_demonstrations > 0:
        this_results_dir += f"_icl{args.n_icl_demonstrations}"
    if args.visual_filter_mode is not None:
        this_results_dir += f"_{args.visual_filter_mode}{args.visual_filter_strength}"
    this_results_dir += f"_{args.run_id}"
    this_results_dir = os.path.join(RESULTS_DIR, "vqa_mistake_detection", this_results_dir)
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

# Load VLM
vlm = AutoModelForVision2Seq.from_pretrained(args.vlm_name, quantization_config=bnb_config)   
vlm_processor = AutoProcessor.from_pretrained(args.vlm_name)
vlm_processor.tokenizer.padding_side = "left"
response_token_ids = get_vqa_response_token_ids(vlm_processor.tokenizer)

# We'll use VLM's LM directly to generate questions
lm = vlm.language_model
tokenizer = vlm_processor.tokenizer
tokenizer.pad_token_id = tokenizer.eos_token_id

# Set up visual filter if needed
visual_filter = None
if args.visual_filter_mode is not None:
    if VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Spatial_NoRephrase:
        visual_filter = SpatialVisualFilter(rephrase_questions=False, mask_strength=args.visual_filter_strength, mask_type=ImageMaskTypes.Darkness, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Spatial_Blur:
        visual_filter = SpatialVisualFilter(rephrase_questions=False, mask_strength=args.visual_filter_strength, mask_type=ImageMaskTypes.Blur, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')            
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Contrastive_Region:
        visual_filter = ContrastiveRegionFilter(mask_strength=args.visual_filter_strength, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Visual_Contrastive:
        visual_filter = VisualContrastiveFilter(alpha=args.visual_filter_strength, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')            
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.AGLA:
        visual_filter = AGLAFilter(alpha=args.visual_filter_strength, beta=0.5, device=f"cuda:0")
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
]
begin_suppress_tokens = [t for t in list(range(vlm_processor.tokenizer.vocab_size)) if t not in yes_no_q_tokens]

generation_kwargs = {
    "do_sample": False,
    "num_beams": 8,
    "num_return_sequences": 4,
    "constraints": question_generation_constraints,
    "begin_suppress_tokens": begin_suppress_tokens,    
    "pad_token_id": tokenizer.eos_token_id,
}

# NLI model to score consistency and verifiability
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)


# Load approopriate evaluation dataset
# TODO: sometimes this fails due to JSON decode error, but there's no problem with the json...
dataset = None
for retry in range(5):
    print(f"({worker_index}) Loading evaluation dataset (try {retry})...")
    try:
        if MistakeDetectionTasks(args.task) == MistakeDetectionTasks.CaptainCook4D:
            dataset = CaptainCook4DDataset(data_split=args.eval_partition, debug_n_examples_per_class=args.debug_n_examples if args.debug else None)
        elif MistakeDetectionTasks(args.task) == MistakeDetectionTasks.Ego4D_Single:
            dataset = Ego4DMistakeDetectionDataset(data_split=args.eval_partition, 
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


print(f"({worker_index}) Beginning iterative VQA inference...")
all_questions = []
all_frames = []
all_candidate_questions = []
all_candidate_questions_scores = []
all_candidate_questions_sources = []
all_scores = []
all_answers = []
all_answer_probs = []
all_success_probs = []

all_example_ids = []
all_procedures = []
all_labels = []

cache_path = os.path.join(this_results_dir, f"cached_outputs{worker_index}.pkl")
is_complete = False
last_batch_idx = -1
if os.path.exists(cache_path):
    is_complete, last_batch_idx, all_questions, all_frames, all_candidate_questions, all_candidate_questions_scores, all_candidate_questions_sources, all_scores, all_answers, all_answer_probs, all_success_probs, all_example_ids, all_procedures, all_labels = pickle.load(open(cache_path, "rb"))

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

        # TODO: enable more prompt template types for other VLMs
        prompts = [
            f'USER: <image>\nThis is a photo of someone working on the procedure "{procedure}". I will ask a series of different yes/no questions about the state of the scene to determine whether the person has successfully executed the procedure. The goal is to extract as much relevant information as possible from the scene, so I will not repeat questions.' 
            for procedure in batch_procedures
        ]
        questions = [[] for _ in range(this_batch_size)]
        frames = [[] for _ in range(this_batch_size)]
        candidate_questions = [[] for _ in range(this_batch_size)]
        candidate_questions_scores = [[] for _ in range(this_batch_size)]
        candidate_questions_sources = [[] for _ in range(this_batch_size)]
        scores = [[] for _ in range(this_batch_size)]
        answer_probs = [[] for _ in range(this_batch_size)] 
        answers = [[] for _ in range(this_batch_size)]
        success_probs = [[] for _ in range(this_batch_size)]

        # Iteratively generate questions
        for question_idx in tqdm(range(args.max_iterations), desc="running iterative QA"):

            # Generate a question (with beam search so we have several candidates)
            prompts_q = [prompt + " USER: Q: " for prompt in prompts]
            new_questions, _ = simple_lm_prompt_beam_search(vlm.language_model,
                                                            vlm_processor.tokenizer,
                                                            [prompt.replace("<image>\n", "") for prompt in prompts_q],
                                                            max_new_tokens=20,
                                                            batch_size=max(args.generation_batch_size // (2 ** question_idx), 1),
                                                            generation_kwargs=generation_kwargs)
            new_questions = [[cleanup_generated_question(question) for question in beam_search_questions] for beam_search_questions in new_questions]                                
            new_questions_sources = [["vlm"] * len(beam_search_questions) for beam_search_questions in new_questions]

            # Optionally inject more candidates from original VQG ICL code
            if args.n_icl_demonstrations > 0:
                icl_prompts = [generate_vqg_prompt_icl(procedure, args.n_icl_demonstrations, include_answers=False) for procedure in batch_procedures] # Create ICL prompt
                icl_prompts = [
                    prompt + '\n'.join([str(pqi+1) + ' ' + pq for pqi, pq in enumerate(previous_questions[-2:])]) + ("\n" if len(previous_questions) > 0 else "") + f"{len(previous_questions) + 1}. " 
                    for prompt, previous_questions in zip(icl_prompts, questions)
                ] # Add some previous questions if possible (take last 2 that were asked)
                icl_new_questions, _ = simple_lm_prompt_beam_search(vlm.language_model,
                                                                    vlm_processor.tokenizer,
                                                                    icl_prompts,
                                                                    max_new_tokens=20,
                                                                    batch_size=max(args.generation_batch_size // args.n_icl_demonstrations, 1),
                                                                    generation_kwargs=generation_kwargs)
                
                icl_new_questions = [[cleanup_generated_question(question) for question in beam_search_questions] for beam_search_questions in icl_new_questions]
                
                for batch_sub_idx in range(this_batch_size):
                    new_questions[batch_sub_idx] += icl_new_questions[batch_sub_idx]
                    new_questions_sources[batch_sub_idx] += ["icl"] * len(icl_new_questions[batch_sub_idx])

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

            # Select best candidate question from pool
            if args.question_selection_strategy == "likelihood":

                # First recalculate likelihood of each cleaned up question in iterative VQA context
                # (this is mostly important when we're using ICL injected questions)
                # TODO: why not just use beam search transition scores? These don't seem to match likelihood of forward pass with model, but need to understand why this is
                generation_scores = compute_completion_log_likelihoods(lm, tokenizer, prompts_q, new_questions, batch_size=max(args.generation_batch_size // max(args.n_icl_demonstrations, 1), 1))

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

            elif args.question_selection_strategy in ["consistency", "verifiability", "coherence"]:
                # Calculate coherence metrics for each candidate question
                nli_outputs = question_coherence_metrics(
                    nli_tokenizer, 
                    nli_model,
                    tokenizer,
                    lm,
                    [procedure for procedure, beam_search_questions in zip(batch_procedures, new_questions) for _ in beam_search_questions],
                    [question for beam_search_questions in new_questions for question in beam_search_questions],
                    previous_questions=[batch_idx_questions for batch_idx_questions, beam_search_questions in zip(questions, new_questions) for _ in beam_search_questions],
                    previous_answers=[batch_idx_answers for batch_idx_answers, beam_search_questions in zip(answers, new_questions) for _ in beam_search_questions],
                    rephrase_batch_size=args.generation_batch_size
                )

                # Select best candidate based on coherence metrics
                selected_questions = []
                new_scores = []
                parallel_idx = 0
                for batch_sub_idx, beam_search_questions in enumerate(new_questions):
                    this_nli_outputs = [{k: round(float(nli_outputs[k][i]), 3) for k in nli_outputs} for i in range(parallel_idx, parallel_idx + len(beam_search_questions))]
                    candidate_questions_scores[batch_sub_idx].append(this_nli_outputs)
                    parallel_idx += len(beam_search_questions)

                    # Use marginal relevance (consistency) and/or expected informativeness (verifiability)
                    if args.question_selection_strategy == "consistency":
                        candidate_scores = np.array(
                            [candidate_metrics['relevance_marginal'] for candidate_metrics in this_nli_outputs]
                        )
                    elif args.question_selection_strategy == "verifiability":
                        candidate_scores = np.array(
                            [candidate_metrics['informativeness_marginal'] for candidate_metrics in this_nli_outputs]
                        )
                    elif args.question_selection_strategy == "coherence":
                        candidate_scores = np.array(
                            [candidate_metrics['relevance_marginal'] * candidate_metrics['informativeness_marginal'] for candidate_metrics in this_nli_outputs]
                        )

                    best_candidate = np.argmax(candidate_scores)
                    selected_questions.append(beam_search_questions[best_candidate])
                    new_scores.append(round(float(candidate_scores[best_candidate]), 6))
                
                new_questions = selected_questions
                    
            # Save scores for best questions
            for batch_sub_idx in range(this_batch_size):
                scores[batch_sub_idx].append(new_scores[batch_sub_idx])

            # Save generated questions
            for batch_sub_idx in range(this_batch_size):
                questions[batch_sub_idx].append(new_questions[batch_sub_idx])

            # Apply visual filter to frames
            if visual_filter:
                if VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Contrastive_Region:
                    batch_frames_filtered = visual_filter(nlp, batch_frames, new_questions)
                elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Visual_Contrastive:
                    batch_frames_filtered = visual_filter(batch_frames)
                elif VisualFilterTypes(args.visual_filter_mode) in [VisualFilterTypes.Spatial_NoRephrase, VisualFilterTypes.Spatial_Blur]:
                    batch_frames_filtered, _ = visual_filter(nlp, batch_frames, new_questions, return_visible_target_objects=False)
                elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.AGLA:
                    batch_frames_filtered = visual_filter(batch_frames, new_questions)

            # Cache paths to frames (if using a visual filter, save filtered frames and cache paths to them)
            if args.visual_filter_mode is None or not args.cache_vqa_frames:
                for batch_sub_idx in range(this_batch_size):
                    frames[batch_sub_idx].append(batch_examples[batch_sub_idx].frames[0])
            else:
                for batch_sub_idx, (frame, example) in enumerate(zip(batch_frames_filtered, batch_examples)):
                    frame_cache_dir = os.path.join(this_results_dir, f"vqa_frames/{example.example_id}")
                    if not os.path.exists(frame_cache_dir):
                        os.makedirs(frame_cache_dir)
                    frame_path = os.path.join(frame_cache_dir, f"frame_q{question_idx}.jpg")
                    resized_frame = resize_with_aspect(frame, CACHED_FRAME_DIMENSION)
                    resized_frame.save(frame_path)
                    frames[batch_sub_idx].append(frame_path)

            # Run VQA on base image (yes/no)
            prompts_a = [prompt + f'{question} ASSISTANT: A: ' for prompt, question in zip(prompts_q, new_questions)]
            if not (visual_filter and VisualFilterTypes(args.visual_filter_mode) in [VisualFilterTypes.Spatial_NoRephrase, VisualFilterTypes.Spatial_Blur]):
                new_answers_logits = run_vqa(vlm, vlm_processor, prompts_a, batch_frames, batch_size=max(args.vqa_batch_size // (2 ** question_idx), 1))
            else:
                # Spatial filter doesn't need original image logits, so don't get them for efficiency
                new_answers_logits = None

            # Run VQA on filtered image if needed and combine logits as proposed in approaches' papers
            # TODO: write a method in each visual filter class to do this
            if visual_filter:
                new_answers_logits_filtered = run_vqa(vlm, vlm_processor, prompts_a, batch_frames_filtered, batch_size=max(args.vqa_batch_size // (2 ** question_idx), 1))
                new_answers_logits = visual_filter.combine_logits(new_answers_logits, new_answers_logits_filtered)

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

            # Save answers and their probabilities
            for batch_sub_idx in range(this_batch_size):
                answer_probs[batch_sub_idx].append([round(float(new_answers[batch_sub_idx].answer_probs[VQAResponse(answer_idx)]), 6) for answer_idx in range(2)])
                answers[batch_sub_idx].append(new_answers[batch_sub_idx].predicted_answer)

            # Update prompts with answers
            prompts = [prompt + output.predicted_answer.name for prompt, output in zip(prompts_a, new_answers)]

            # Ask VLM probability of success
            prompt_success = [
                prompt + f' USER: Q: Based on the above information, has the procedure "{procedure}" been successfully executed? ASSISTANT: A:'
                for prompt, procedure in zip(prompts, batch_procedures)
            ]
            # TODO: should SuccessVQA stepalso incorporate visual filter?
            success_vqa_outputs = run_vqa(
                vlm, 
                vlm_processor, 
                prompt_success, 
                batch_frames, 
                batch_size=args.vqa_batch_size
            )
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
        all_frames += frames
        all_candidate_questions += candidate_questions
        all_candidate_questions_scores += candidate_questions_scores
        all_candidate_questions_sources += candidate_questions_sources
        all_scores += scores
        all_answers += answers
        all_answer_probs += answer_probs
        all_success_probs += success_probs
        all_example_ids += [example.example_id for example in batch_examples]
        all_procedures += [example.procedure_description for example in batch_examples]
        all_labels += [example.mistake for example in batch_examples]

        for frame in batch_frames:
            frame.close()
        del batch_frames

        # And cache tracked outputs
        pickle.dump((    
            False,
            batch_idx,
            all_questions, 
            all_frames,
            all_candidate_questions, 
            all_candidate_questions_scores, 
            all_candidate_questions_sources,
            all_scores, 
            all_answers, 
            all_answer_probs, 
            all_success_probs,
            all_example_ids,
            all_procedures,
            all_labels,
        ), open(cache_path, "wb"))

# Verify we got correct number of outputs
all_results = [
    all_questions, 
    all_frames,
    all_candidate_questions, 
    all_candidate_questions_scores, 
    all_candidate_questions_sources,
    all_scores, 
    all_answers, 
    all_answer_probs, 
    all_success_probs,
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
        all_frames,
        all_candidate_questions, 
        all_candidate_questions_scores, 
        all_candidate_questions_sources,
        all_scores, 
        all_answers, 
        all_answer_probs, 
        all_success_probs,
        all_example_ids,
        all_procedures,
        all_labels,
    ), open(cache_path, "wb"))

print(f"({worker_index}) Done running iterative VQA inference!")


# Gather up results across processes and evaluate
if worker_index == 0:
    print(f"({worker_index}) Gathering all results...")
    for other_worker_index in range(1, n_workers):
        print(f"({worker_index}) Gathering results from worker {other_worker_index}...")
        delay_per_try = 10
        delay_so_far = 0
        max_delay = 7200 # TODO: change this to a lower number
        while True:
            other_cache_path = os.path.join(this_results_dir, f"cached_outputs{other_worker_index}.pkl")
            if os.path.exists(other_cache_path):
                is_complete, \
                _, \
                other_questions, \
                other_frames, \
                other_candidate_questions, \
                other_candidate_questions_scores, \
                other_candidate_questions_sources, \
                other_scores, \
                other_answers, \
                other_answer_probs, \
                other_success_probs, \
                other_example_ids, \
                other_procedures, \
                other_labels = pickle.load(open(other_cache_path, "rb"))
                if is_complete:
                    # Add other process results to our results
                    all_questions += other_questions
                    all_frames += other_frames
                    all_candidate_questions += other_candidate_questions
                    all_candidate_questions_scores += other_candidate_questions_scores
                    all_candidate_questions_sources += other_candidate_questions_sources
                    all_scores += other_scores
                    all_answers += other_answers
                    all_answer_probs += other_answer_probs
                    all_success_probs += other_success_probs
                    all_example_ids += other_example_ids
                    all_procedures += other_procedures
                    all_labels += other_labels
                    print(f"({worker_index}) Collected results from worker {other_worker_index}.")
                    break

            # Decide whether to try again
            if delay_so_far >= max_delay:
                raise TimeoutError(f"Waited for {max_delay} seconds for results from worker {other_worker_index}. Process may have failed.")
            print(f"({worker_index}) Still waiting for results from worker {other_worker_index} ({delay_so_far} sec.)!")
            time.sleep(delay_per_try)
            delay_so_far += delay_per_try

    # Collect key information from results rollouts and final success probabilities
    all_results_dicts = {}
    all_probs = []
    for questions, frames, candidate_questions, candidate_questions_scores, candidate_questions_sources, scores, answers, answer_probs, success_probs, example_id, procedure, label \
        in tqdm(zip(all_questions,
                    all_frames,
                    all_candidate_questions,
                    all_candidate_questions_scores,
                    all_candidate_questions_sources,
                    all_scores,
                    all_answers,
                    all_answer_probs,
                    all_success_probs,
                    all_example_ids,
                    all_procedures,
                    all_labels), desc="compiling results"):
        
        final_success_prob = None
        for success_prob_idx, success_prob in enumerate(success_probs):
            # Early stopping mechanism: 
            # if success score doesn't change enough over 3 turns, stop incorporating questions
            # (we still run inference across all questions for efficiency and simplicity, but later can make a proper demo script)
            final_success_prob = success_prob
            if success_prob_idx >= 2 and success_prob_idx < len(success_probs) - 1:
                if np.abs(success_probs[success_prob_idx-1] - success_probs[success_prob_idx-2]) < args.early_stop_delta and np.abs(success_probs[success_prob_idx] - success_probs[success_prob_idx-1]) < args.early_stop_delta:
                    break
        all_probs.append(round(final_success_prob, 6))   

        results_dict = {
            "procedure": procedure,
            "mistake": label,
            "questions": questions,
            "frame_paths": frames,
            "answers": [a.value for a in answers],
            "answer_probs": answer_probs,
            "scores": scores,
            "success_probs": success_probs,
            "final_turn": success_prob_idx,
            "final_success_prob": final_success_prob,
            "candidate_questions": candidate_questions,
            "candidate_questions_scores": candidate_questions_scores,
            "candidate_questions_sources": candidate_questions_sources,
        }
        all_results_dicts[example_id] = results_dict

    json.dump(all_results_dicts, 
            open(os.path.join(this_results_dir, f"outputs_{args.eval_partition}.json"), "w"),
            indent=4)


    print(f"({worker_index}) Evaluating outputs...")
    metrics = {}

    # Calculate accuracy metrics
    best_metrics = None
    best_threshold = None
    accuracy_metrics = {}
    for threshold in MISTAKE_DETECTION_THRESHOLDS:
        preds = [1.0 - p >= threshold for p in all_probs]
        assert len(preds) == len(all_probs) == len(all_labels), "Expected same number of preds, probs, and labels."
        this_metrics = mistake_detection_metrics(all_labels, preds)
        accuracy_metrics[threshold] = this_metrics

        if best_metrics is None or (this_metrics['false_positive_rate'] + this_metrics['false_negative_rate']) < (best_metrics['false_positive_rate'] + best_metrics['false_negative_rate']):
            best_metrics = this_metrics
            best_threshold = threshold

    accuracy_metrics['best_metrics'] = best_metrics
    accuracy_metrics['best_threshold'] = best_threshold

    json.dump(accuracy_metrics, 
            open(os.path.join(this_results_dir, f"metrics_accuracy_{args.eval_partition}.json"), "w"),
            indent=4)

    # Generate DET curve
    # TODO: add more thresholds? e.g., every 0.01
    generate_det_curve(accuracy_metrics, os.path.join(this_results_dir, f"det_accuracy_{args.eval_partition}.pdf"))

    # Calculate coherence metrics of final rollouts
    all_chosen_questions = [question for results_dict in all_results_dicts.values() for question in results_dict['questions'][:results_dict['final_turn'] + 1]]
    all_previous_questions = [results_dict['questions'][:question_idx] for results_dict in all_results_dicts.values() for question_idx in range(results_dict['final_turn'] + 1)]

    all_predicted_answers = [VQAResponse(answer) for results_dict in all_results_dicts.values() for answer in results_dict['answers'][:results_dict['final_turn'] + 1]]
    all_previous_answers = [[VQAResponse(a) for a in results_dict['answers'][:question_idx]] for results_dict in all_results_dicts.values() for question_idx in range(results_dict['final_turn'] + 1)]

    all_metrics = question_coherence_metrics(nli_tokenizer,
                                            nli_model,
                                            tokenizer,
                                            lm,                                         
                                            [procedure for results_dict, procedure in zip(all_results_dicts.values(), all_procedures) for _ in range(results_dict['final_turn'] + 1)],
                                            all_chosen_questions,
                                            answers=all_predicted_answers, # TODO: this returns an informativeness metric which says how much different success probability is for a yes or no answer - should we just look at probability of entailment if it's a success example, and contradiction if it's a mistake?
                                            previous_questions=all_previous_questions,
                                            previous_answers=all_previous_answers,
                                            rephrase_batch_size=args.generation_batch_size)

    parallel_idx = 0
    coherence_metrics_by_example = defaultdict(list)
    coherence_metric_names = ['relevance', 'informativeness', 'relevance_marginal', 'informativeness_marginal']
    for results_dict in all_results_dicts.values():
        for k in coherence_metric_names:
            if k in all_metrics:
                this_metrics = []
                for question_idx in range(results_dict['final_turn'] + 1):
                    this_metrics.append(all_metrics[k][parallel_idx])
                coherence_metrics_by_example[k].append(round(float(np.mean(this_metrics)), 6))
        parallel_idx += 1

    coherence_metrics = {
        k: round(float(np.mean(coherence_metrics_by_example[k])), 6) for k in coherence_metric_names if k in coherence_metrics_by_example
    } | {
        "metrics_by_example": coherence_metrics_by_example,
    }
    json.dump(coherence_metrics, 
            open(os.path.join(this_results_dir, f"metrics_coherence_{args.eval_partition}.json"), "w"),
            indent=4)

    
    print(f"({worker_index}) Done!")