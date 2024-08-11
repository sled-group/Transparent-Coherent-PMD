from travel import init_travel
init_travel()

import argparse
from collections import defaultdict, Counter
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
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer, PhrasalConstraint           

from travel.constants import RESULTS_DIR, IMAGES_CHUNK_SIZE
from travel.data.captaincook4d import CaptainCook4DDataset
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.data.vqa import VQAResponse, get_vqa_response_token_ids, VQAOutputs, IMAGE_TOKENS, USER_START_TOKENS, USER_END_TOKENS, ASSISTANT_START_TOKENS, ASSISTANT_END_TOKENS, IVQA_PREAMBLE, IVQA_SUCCESS_QUESTION
from travel.data.vqg import generate_vqg_prompt_icl
from travel.model import simple_lm_prompt_beam_search, simple_vlm_prompt_beam_search, compute_completion_log_likelihoods, compute_completion_log_likelihoods_vlm
from travel.model.grounding import VisualFilterTypes, ContrastiveRegionFilter, VisualContrastiveFilter, SpatialVisualFilter, AGLAFilter, ImageMaskTypes
from travel.model.metrics import mistake_detection_metrics, question_coherence_metrics_nli, question_coherence_metrics_vlm, generate_det_curve, generate_tiered_metric_curves
from travel.model.mistake_detection import MISTAKE_DETECTION_THRESHOLDS
from travel.model.nli import NLI_MODEL_PATH, NLI_BATCH_SIZE
from travel.model.vqa import run_vqa_with_visual_filter
from travel.model.vqg import cleanup_generated_question


parser = argparse.ArgumentParser()
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=["llava-hf/llava-1.5-7b-hf"], help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--task", type=str, default="ego4d_single", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--eval_partition", type=str, default="val", choices=["val", "test"])
parser.add_argument("--max_iterations", type=int, default=8, help="Maximum number of questions to generate before making a final mistake detection decision.")
parser.add_argument("--num_beams", type=int, default=8, choices=list(range(21)), help="Number of beams in beam search.")
parser.add_argument("--num_return_sequences", type=int, default=4, choices=list(range(21)), help="Number of generation candidates to return from beam search. Recommend setting this to be less than number of beams due to generation constraints.")
parser.add_argument("--n_icl_demonstrations", type=int, default=0, choices=list(range(21)), help="Pass this argument to generate an extra pool of candidate questions using n in-context VQG examples (doesn't incorporate answers to previous questions).")
parser.add_argument("--condition_questions_with_frames", action="store_true", help="Pass this argument to pass frame into VLM while generating questions (usually off by default since this hurts performance).")
parser.add_argument("--question_selection_strategy", type=str, default="likelihood", choices=["likelihood", "coherence"], help="Strategy to use to choose question to generate from beam search candidates.")
parser.add_argument("--visual_filter_mode", type=str, required=False, choices=[t.value for t in VisualFilterTypes], help="Visual attention filter mode.")
parser.add_argument("--visual_filter_strength", type=float, required=False, default=1.0, help="Float strength for masks used in visual filters. Depending on the visual filter type, this may be interpreted as a percentage darkness or a Gaussian blur kernel size.")
parser.add_argument("--generation_batch_size", type=int, default=10, help="Batch size for question generation with LM.")
parser.add_argument("--vqa_batch_size", type=int, default=10, help="Batch size for VQA with VLM.")
parser.add_argument("--nli_batch_size", type=int, default=NLI_BATCH_SIZE, help="Batch size for scoring candidate questions with NLI model.")
parser.add_argument("--run_id", type=str, required=False, help="Unique ID for this run, which will be used to create the output directory (and should be shared across any parallel processes).")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of iterative VQA.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
parser.add_argument("--cache_vqa_frames", action="store_true", help="Pass this argument to cache frames in VQA outputs (e.g., to inspect visual filter resuilts). This consumes a lot of disk space for large datasets.")
parser.add_argument("--print_prompts", action="store_true", help="Pass this argument to print some sample prompts during execution (for debugging purposes).")
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
    this_results_dir = os.path.join(task_name, vlm_name, f"IterativeVQA_topdown_q{args.max_iterations}_{task_name}")
    this_results_dir += f"_{vlm_name}"
    this_results_dir += f"_beam{args.num_beams}-{args.num_return_sequences}"
    if args.condition_questions_with_frames:
        this_results_dir += f"_cqframe"
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

# Load VLM - some VLMs may be under AutoModelForVision2Seq, some may be under AutoModelForCausalLM
try:
    vlm = AutoModelForVision2Seq.from_pretrained(args.vlm_name, quantization_config=bnb_config)   
except:
    vlm = AutoModelForCausalLM.from_pretrained(args.vlm_name, quantization_config=bnb_config)
vlm_processor = AutoProcessor.from_pretrained(args.vlm_name)
vlm_processor.tokenizer.padding_side = "left"
response_token_ids = get_vqa_response_token_ids(vlm_processor.tokenizer)

# We'll use VLM's LM directly to generate questions
lm = vlm.language_model
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

generation_kwargs = {
    "do_sample": False,
    "num_beams": args.num_beams,
    "num_return_sequences": args.num_return_sequences,
    "constraints": question_generation_constraints,
    "begin_suppress_tokens": begin_suppress_tokens,    
    "pad_token_id": tokenizer.eos_token_id,
}

# NLI model to score consistency and verifiability
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)


# Load approopriate evaluation dataset
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
            f'{USER_START_TOKENS[type(vlm)]}{IMAGE_TOKENS[type(vlm)]}{IVQA_PREAMBLE.format(procedure=procedure)}' 
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

        # Ask VLM probability of success
        questions_success = [
            IVQA_SUCCESS_QUESTION.format(procedure=procedure)
            for procedure in batch_procedures
        ]
        prompts_success = [
            prompt + f'{ASSISTANT_END_TOKENS[type(vlm)]}{USER_START_TOKENS[type(vlm)]}Q: {question}{USER_END_TOKENS[type(vlm)]}{ASSISTANT_START_TOKENS[type(vlm)]}A:'
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
                                                            question_idx=f"success",
                                                            batch_size=args.vqa_batch_size,
                                                            visual_filter=visual_filter if visual_filter and VisualFilterTypes(args.visual_filter_mode) not in [VisualFilterTypes.Spatial_NoRephrase, VisualFilterTypes.Spatial_Blur] else None, # Don't use spatial filter for SuccessVQA step, since this may remove important information
                                                            nlp=nlp,
                                                            visual_filter_mode=VisualFilterTypes(args.visual_filter_mode) if visual_filter else None,
                                                            frame_cache_dir=this_results_dir if args.cache_vqa_frames else None)
        success_vqa_outputs = [
            VQAOutputs(
                task_name=MistakeDetectionTasks(args.task),
                example_id=example.example_id,
                procedure_id=example.procedure_id,
                frame=example.frames[0],
                prompt="",
                expected_answer=None,
                response_token_ids=response_token_ids,
                logits=logits,
                question="",
            ) for logits, example in zip(success_vqa_outputs, batch_examples)
        ]

        # Inject success question and answer into prompts
        prompts = [f"{prompt} {output.predicted_answer.name}" for prompt, output in zip(prompts_success, success_vqa_outputs)]            
        success_vqa_outputs = [output.answer_probs[VQAResponse.Yes] for output in success_vqa_outputs]

        # Iteratively generate questions
        for question_idx in tqdm(range(args.max_iterations), desc="running iterative QA"):

            # Save success probability for this turn (just save another copy of them)
            for batch_sub_idx in range(this_batch_size):
                success_probs[batch_sub_idx].append(
                    round(float(success_vqa_outputs[batch_sub_idx]), 6)
                )

            # Generate a question (with beam search so we have several candidates)
            prompts_q = [prompt + f"{ASSISTANT_END_TOKENS[type(vlm)]}{USER_START_TOKENS[type(vlm)]}Q:" for prompt in prompts]
            if not args.condition_questions_with_frames:
                new_questions, _ = simple_lm_prompt_beam_search(vlm.language_model,
                                                                vlm_processor.tokenizer,
                                                                [prompt.replace(IMAGE_TOKENS[type(vlm)], "") for prompt in prompts_q],
                                                                max_new_tokens=20,
                                                                batch_size=max(args.generation_batch_size // (2 ** question_idx), 1),
                                                                generation_kwargs=generation_kwargs)
            else:
                new_questions = simple_vlm_prompt_beam_search(vlm,
                                                              vlm_processor,
                                                              prompts_q,
                                                              batch_frames,
                                                              IMAGE_TOKENS[type(vlm)],
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
                if not args.condition_questions_with_frames:
                    generation_scores = compute_completion_log_likelihoods(lm, tokenizer, [prompt.replace(IMAGE_TOKENS[type(vlm)], "") for prompt in prompts_q], new_questions, batch_size=max(args.generation_batch_size // max(args.n_icl_demonstrations, 1), 1))
                else:
                    generation_scores = compute_completion_log_likelihoods_vlm(vlm, vlm_processor, prompts_q, batch_frames, new_questions, batch_size=max(args.generation_batch_size // max(args.n_icl_demonstrations, 1), 1))

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

            elif args.question_selection_strategy == "coherence":
                # Calculate coherence metrics for each candidate question
                nli_outputs = question_coherence_metrics_nli(
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

                    # Use marginal relevance (consistency) and expected informativeness (verifiability) to rank candidates
                    candidate_scores = np.array(
                        [candidate_metrics['informativeness_marginal_x_relevance_marginal'] for candidate_metrics in this_nli_outputs]
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

            # Run VQA with generated questions (and optional spatial filter)
            prompts_a = [prompt + f' {question}{USER_END_TOKENS[type(vlm)]}{ASSISTANT_START_TOKENS[type(vlm)]}A:' for prompt, question in zip(prompts_q, new_questions)]
            if args.print_prompts:
                pprint(prompts_a[0])
            new_answers_logits = run_vqa_with_visual_filter(vlm_processor=vlm_processor, 
                                                            vlm=vlm, 
                                                            batch_examples=batch_examples, 
                                                            batch_frames=batch_frames, 
                                                            prompts_a=prompts_a, 
                                                            new_questions=new_questions, 
                                                            question_idx=question_idx,
                                                            batch_size=max(args.vqa_batch_size // (2 ** question_idx), 1),
                                                            visual_filter=visual_filter,
                                                            nlp=nlp,
                                                            visual_filter_mode=VisualFilterTypes(args.visual_filter_mode) if visual_filter else None,
                                                            frame_cache_dir=this_results_dir if args.cache_vqa_frames else None)

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
            prompts = [prompt + " " + output.predicted_answer.name for prompt, output in zip(prompts_a, new_answers)]

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


# Gather up results across processes and evaluate
if worker_index == 0:
    print(f"({worker_index}) Gathering all results...")
    for other_worker_index in range(1, n_workers):
        print(f"({worker_index}) Gathering results from worker {other_worker_index}...")
        delay_per_try = 10
        delay_so_far = 0
        max_delay = 7200 if args.resume_dir is not None else 1800 # Allow a longer delay in case some processes are already finished in resumed run
        while True:
            other_cache_path = os.path.join(this_results_dir, f"cached_outputs{other_worker_index}.pkl")
            if os.path.exists(other_cache_path):
                is_complete, \
                _, \
                other_questions, \
                other_candidate_questions, \
                other_candidate_questions_scores, \
                other_candidate_questions_sources, \
                other_scores, \
                other_answers, \
                other_answer_probs, \
                other_success_probs, \
                other_success_probs_negated, \
                other_example_ids, \
                other_procedures, \
                other_labels = pickle.load(open(other_cache_path, "rb"))
                if is_complete:
                    # Add other process results to our results
                    all_questions += other_questions
                    all_candidate_questions += other_candidate_questions
                    all_candidate_questions_scores += other_candidate_questions_scores
                    all_candidate_questions_sources += other_candidate_questions_sources
                    all_scores += other_scores
                    all_answers += other_answers
                    all_answer_probs += other_answer_probs
                    all_success_probs += other_success_probs
                    all_success_probs_negated += other_success_probs_negated
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
    for questions, candidate_questions, candidate_questions_scores, candidate_questions_sources, scores, answers, answer_probs, success_probs, success_probs_negated, example_id, procedure, label \
        in tqdm(zip(all_questions,
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
                    all_labels), desc="compiling results"):
        
        final_success_prob = None
        for success_prob_idx, success_prob in enumerate(success_probs):
            # Don't use early stopping
            final_success_prob = success_prob

        all_probs.append(round(final_success_prob, 6))   

        results_dict = {
            "procedure": procedure,
            "mistake": True if label is not None else False,
            "mistake_type": label,
            "questions": questions,
            "frame_dir": os.path.join(this_results_dir, f"vqa_frames/{example_id}"),
            "answers": [a.value for a in answers],
            "answer_probs": answer_probs,
            "scores": scores,
            "success_probs": success_probs,
            "success_probs_negated": success_probs_negated,
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

    # Calculate coherence metrics of final rollouts
    all_chosen_questions = [question for results_dict in all_results_dicts.values() for question in results_dict['questions'][:results_dict['final_turn'] + 1]]
    all_previous_questions = [results_dict['questions'][:question_idx] for results_dict in all_results_dicts.values() for question_idx in range(results_dict['final_turn'] + 1)]

    all_predicted_answers = [VQAResponse(answer) for results_dict in all_results_dicts.values() for answer in results_dict['answers'][:results_dict['final_turn'] + 1]]
    all_previous_answers = [[VQAResponse(a) for a in results_dict['answers'][:question_idx]] for results_dict in all_results_dicts.values() for question_idx in range(results_dict['final_turn'] + 1)]

    all_coherence_metrics = question_coherence_metrics_nli(nli_tokenizer,
                                            nli_model,
                                            tokenizer,
                                            lm,                                         
                                            [procedure for results_dict, procedure in zip(all_results_dicts.values(), all_procedures) for _ in range(results_dict['final_turn'] + 1)],
                                            all_chosen_questions,
                                            answers=all_predicted_answers,
                                            previous_questions=all_previous_questions,
                                            previous_answers=all_previous_answers,
                                            rephrase_batch_size=args.generation_batch_size)

    # Aggregate coherence metrics by example and by turn
    parallel_idx = 0
    coherence_metrics_by_example = defaultdict(list)
    coherence_metrics_by_turn = defaultdict(list)
    coherence_metric_names = ['relevance', 'informativeness', 'relevance_marginal', 'informativeness_marginal', 'informativeness_marginal_x_relevance_marginal']
    for results_dict in all_results_dicts.values():
        for k in coherence_metric_names:
            if k in all_coherence_metrics:
                this_metrics = []
                for question_idx in range(results_dict['final_turn'] + 1):
                    this_metrics.append(round(float(all_coherence_metrics[k][parallel_idx]), 6))
                coherence_metrics_by_example[k + "_by_example"].append(round(float(np.mean(this_metrics)), 6))
                coherence_metrics_by_turn[k + "_by_turn"].append(this_metrics)
        parallel_idx += 1

    # Calculate accuracy metrics
    best_metrics = None
    best_threshold = None
    accuracy_metrics_by_threshold = {}
    coherence_metrics_by_threshold = {}
    all_labels_binary = [True if l is not None else False for l in all_labels]
    for threshold in MISTAKE_DETECTION_THRESHOLDS:
        preds = [1.0 - p >= threshold for p in all_probs] # Have to do 1.0 - probability since we got "success" probability from VLM
        assert len(preds) == len(all_probs) == len(all_labels), "Expected same number of preds, probs, and labels."
        this_metrics = mistake_detection_metrics(all_labels_binary, preds)
        accuracy_metrics_by_threshold[threshold] = this_metrics

        # Calculate consistency and verifiability for this example, which are conditional on correctness
        verifiability = np.mean([coherence_metrics_by_example['informativeness_marginal_x_relevance_marginal_by_example'][i] if preds[i] == all_labels_binary[i] else 0.0 for i in range(len(preds))])
        consistency = np.mean([coherence_metrics_by_example['relevance_marginal_by_example'][i] if preds[i] == all_labels_binary[i] else 0.0 for i in range(len(preds))])
        coherence_metrics_by_threshold[threshold] = {"verifiability": verifiability, "consistency": consistency,}

        if best_metrics is None or (this_metrics['false_positive_rate'] + this_metrics['false_negative_rate']) < (best_metrics['false_positive_rate'] + best_metrics['false_negative_rate']):
            best_metrics = this_metrics
            best_threshold = threshold

    accuracy_metrics_by_threshold['best_metrics'] = best_metrics
    accuracy_metrics_by_threshold['best_threshold'] = best_threshold

    # Save accuracy and coherence metrics
    json.dump(accuracy_metrics_by_threshold, 
            open(os.path.join(this_results_dir, f"metrics_accuracy_{args.eval_partition}.json"), "w"),
            indent=4)
    
    coherence_metrics = {
        k: round(float(np.mean(coherence_metrics_by_example[k + "_by_example"])), 6) for k in coherence_metric_names if k + "_by_example" in coherence_metrics_by_example
    } | {
        "metrics_by_threshold": coherence_metrics_by_threshold,
        "metrics_by_example": coherence_metrics_by_example,
        "metrics_by_turn": coherence_metrics_by_turn,
    }
    json.dump(coherence_metrics, 
            open(os.path.join(this_results_dir, f"metrics_coherence_nli_{args.eval_partition}.json"), "w"),
            indent=4)

    # Generate DET curves for accuracy
    generate_det_curve(accuracy_metrics_by_threshold, os.path.join(this_results_dir, f"det_accuracy_{args.eval_partition}.pdf"))

    # Generate curves for all metrics by threshold
    generate_tiered_metric_curves(MISTAKE_DETECTION_THRESHOLDS, 
                                  [accuracy_metrics_by_threshold[t]['accuracy'] for t in MISTAKE_DETECTION_THRESHOLDS],
                                  [coherence_metrics_by_threshold[t]['consistency'] for t in MISTAKE_DETECTION_THRESHOLDS], 
                                  [coherence_metrics_by_threshold[t]['verifiability'] for t in MISTAKE_DETECTION_THRESHOLDS],
                                  [os.path.join(this_results_dir, f"graph_tiered_metrics_{args.coherence_evaluation_strategy}_{args.eval_partition}.pdf")])
    
    print(f"({worker_index}) Done!")