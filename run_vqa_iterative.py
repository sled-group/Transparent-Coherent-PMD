from travel import init_travel
init_travel()

import argparse
from collections import defaultdict
import concurrent.futures
from copy import deepcopy
import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pickle
from PIL import Image
from pprint import pprint
import spacy
import torch
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer, PhrasalConstraint

from travel.constants import RESULTS_DIR, IMAGES_CHUNK_SIZE
from travel.data.captaincook4d import CaptainCook4DDataset
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.mistake_detection import MistakeDetectionExample, get_cutoff_time_by_proportion, MistakeDetectionTasks
from travel.data.utils import split_list_into_partitions
from travel.data.vqa import VQA_PROMPT_TEMPLATES, VQAResponse, SUCCESSVQA_QUESTION_TEMPLATE, CAPTION_VQA_PROMPT_TEMPLATES, VQG2VQA2SUCCESSVQA_PROMPT_TEMPLATES, get_vqa_response_token_ids, VQAOutputs
from travel.data.vqg import VQGOutputs
from travel.model import simple_lm_prompt_beam_search
from travel.model.grounding import VisualFilterTypes, ContrastiveRegionFilter, TargetObjectCounterFilter, VisualContrastiveFilter
from travel.model.metrics import generate_det_curve, mistake_detection_metrics, question_coherence_metrics
from travel.model.mistake_detection import aggregate_mistake_probs_over_frames, DETECTION_FRAMES_PROPORTION, MISTAKE_DETECTION_STRATEGIES, compile_mistake_detection_preds, MISTAKE_DETECTION_THRESHOLDS
from travel.model.nli import NLI_HYPOTHESIS_TEMPLATE, NLI_MODEL_PATH, NLI_BATCH_SIZE
from travel.model.vqa import run_vqa, rephrase_question_answer
from travel.model.vqg import cleanup_generated_question

parser = argparse.ArgumentParser()
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=["llava-hf/llava-1.5-7b-hf"], help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--task", type=str, default="captaincook4d", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--eval_partition", type=str, default="val", choices=["val", "test"])
parser.add_argument("--max_iterations", type=int, default=8, help="Maximum number of questions to generate before making a final mistake detection decision.")
parser.add_argument("--question_selection_strategy", type=str, default="likelihood", choices=["likelihood", "consistency", "verifiability", "coherence"], help="Strategy to use to choose question to generate from beam search candidates.")
parser.add_argument("--early_stop_delta", type=int, default=0.1, help="If success probability changes less than this over 3 turns, stop generating questions.")
parser.add_argument("--visual_filter_mode", type=str, required=False, choices=[t.value for t in VisualFilterTypes], help="Visual attention filter mode.")
parser.add_argument("--visual_filter_strength", type=float, required=False, default=1.0, help="Float strength for masks used in visual filters. Depending on the visual filter type, this may be interpreted as a percentage darkness or a Gaussian blur kernel size.")
parser.add_argument("--generation_batch_size", type=int, default=10, help="Batch size for question generation with LM.")
parser.add_argument("--vqa_batch_size", type=int, default=10, help="Batch size for VQA with VLM.")
parser.add_argument("--nli_batch_size", type=int, default=NLI_BATCH_SIZE, help="Batch size for scoring candidate questions with NLI model.")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of generating frameVQA examples.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
parser.add_argument("--cache_vqa_frames", action="store_true", help="Pass this argument to cache frames in VQA outputs (e.g., to inspect visual filter resuilts). This consumes a lot of disk space for large datasets.")
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
    timestamp = datetime.datetime.now()
    vlm_name = args.vlm_name.split('/')[-1]
    task_name = args.task
    if args.debug:
        task_name += f"_debug{args.debug_n_examples}" if args.task != "captaincook4d" else "_debug"
    this_results_dir = os.path.join(task_name, vlm_name, f"IterativeVQA_q{args.max_iterations}_{task_name}")
    this_results_dir += f"_{vlm_name}"
    if args.visual_filter_mode is not None:
        this_results_dir += f"_{args.visual_filter_mode}{args.visual_filter_strength}"
    this_results_dir += f"_{timestamp.strftime('%Y%m%d%H%M%S')}"
    this_results_dir = os.path.join(RESULTS_DIR, "vqa_mistake_detection", this_results_dir)
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
vlm = AutoModelForVision2Seq.from_pretrained(args.vlm_name, 
                                            quantization_config=bnb_config)   
vlm_processor = AutoProcessor.from_pretrained(args.vlm_name)
vlm_processor.tokenizer.padding_side = "left"
response_token_ids = get_vqa_response_token_ids(vlm_processor.tokenizer)

# We'll use VLM's LM directly to generate questions
lm = vlm.language_model
tokenizer = vlm_processor.tokenizer

# Set up visual filter if needed
visual_filter = None
visual_filter = None
if args.visual_filter_mode is not None:
    if VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Contrastive_Region:
        visual_filter = ContrastiveRegionFilter(mask_strength=args.visual_filter_strength, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')
    if VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Visual_Contrastive:
        visual_filter = VisualContrastiveFilter(alpha=args.visual_filter_strength, device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')            
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Target_Object_Counter:
        visual_filter = TargetObjectCounterFilter(device=f"cuda:0")
        nlp = spacy.load('en_core_web_lg')      
    else:
        raise NotImplementedError(f"Visual filter type {args.visual_filter_mode} is not compatible with SuccessVQA!")
    # TODO: add AGLA as an option here?

# Shared generation kwargs

# kwargs to force question generations to have a "?" and start with "Is" or "Are"
question_generation_constraints = [    
    PhrasalConstraint(
        [vlm_processor.tokenizer("Is it blue?", add_special_tokens=False).input_ids[-1]]
    ),
]
yes_no_q_tokens = [
    vlm_processor.tokenizer("Is it blue?", add_special_tokens=False).input_ids[0], 
    vlm_processor.tokenizer("Are they blue?", add_special_tokens=False).input_ids[0],
    vlm_processor.tokenizer("Does it look blue?", add_special_tokens=False).input_ids[0],
    vlm_processor.tokenizer("Do they look blue?", add_special_tokens=False).input_ids[0],
]
begin_suppress_tokens = [t for t in list(range(vlm_processor.tokenizer.vocab_size)) if t not in yes_no_q_tokens]

generation_kwargs = {
    "do_sample": False,
    "num_beams": 8,
    "num_beam_groups": 2,
    "diversity_penality": 1.0,
    "num_return_sequences": 8,
    "constraints": question_generation_constraints,
    "begin_suppress_tokens": begin_suppress_tokens,    
}

# NLI model to score consistency and verifiability
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)


# Load approopriate evaluation dataset
print(f"({worker_index}) Loading evaluation dataset...")
if MistakeDetectionTasks(args.task) == MistakeDetectionTasks.CaptainCook4D:
    dataset = CaptainCook4DDataset(data_split=args.eval_partition, debug_n_examples_per_class=args.debug_n_examples if args.debug else None)
elif MistakeDetectionTasks(args.task) == MistakeDetectionTasks.Ego4D_Single:
    dataset = Ego4DMistakeDetectionDataset(data_split=args.eval_partition, 
                                           mismatch_augmentation=True,
                                           multi_frame=False,
                                           debug_n_examples_per_class=args.debug_n_examples if args.debug else None)
else:
    raise NotImplementedError(f"Haven't implemented usage of {args.task} dataset yet!")


print(f"({worker_index}) Beginning iterative VQA inference...")
all_questions = []
all_candidate_questions = []
all_candidate_questions_scores = []
all_scores = []
all_vqa_outputs = []
all_answers = []
all_success_probs = []

all_example_ids = []
all_procedures = []
all_labels = []

cache_path = os.path.join(this_results_dir, f"cached_outputs{worker_index}.pkl")
if os.path.exists(cache_path):
    all_questions, all_candidate_questions, all_candidate_questions_scores, all_scores, all_vqa_outputs, all_answers, all_success_probs, all_example_ids, all_procedures, all_labels = pickle.load(open(cache_path, "rb"))

for batch_idx, batch_examples in tqdm(enumerate(dataset.get_batches(IMAGES_CHUNK_SIZE, 
                                                                    n_workers=n_workers, 
                                                                    worker_index=worker_index,
                                                                    load_frames=False), 
                                                desc="running iterative VQA inference")):

    # If already in cache, skip this batch
    if len(all_questions) > batch_idx:
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
    candidate_questions = [[] for _ in range(this_batch_size)]
    candidate_questions_scores = [[] for _ in range(this_batch_size)]
    scores = [[] for _ in range(this_batch_size)]
    vqa_outputs = [[] for _ in range(this_batch_size)] 
    answers = [[] for _ in range(this_batch_size)]
    success_probs = [[] for _ in range(this_batch_size)]

    # Iteratively generate questions
    for question_idx in tqdm(range(args.max_iterations), desc="generating questions"):

        # Generate a question
        prompts_q = [prompt + " USER: Q: " for prompt in prompts]
        new_questions, generation_scores = simple_lm_prompt_beam_search(vlm.language_model,
                                                            vlm_processor.tokenizer,
                                                            [prompt.replace("<image>\n", "") for prompt in prompts_q],
                                                            max_new_tokens=20,
                                                            batch_size=args.generation_batch_size,
                                                            generation_kwargs=generation_kwargs)
                            
        new_questions = [[cleanup_generated_question(question) for question in beam_search_questions] for beam_search_questions in new_questions]
        for batch_sub_idx in range(len(candidate_questions)):
            candidate_questions[batch_sub_idx].append(new_questions[batch_sub_idx])

        # TODO: optionally inject more candidates from original VQG ICL code

        if args.question_selection_strategy == "likelihood":
            # Select most likely question (first one in list)
            # TODO: this won't be the case once we add more candidates from VQG, so update later to incorporate likelihoods (need to return them from simple_prompt_lm)
            new_questions = [beam_search_questions[0] for beam_search_questions in new_questions]
            new_scores = [gs[0] for gs in generation_scores]

        elif args.question_selection_strategy in ["consistency", "verifiability", "coherence"]:
            # Calculate coherence metrics for each candidate question
            nli_outputs = question_coherence_metrics(
                nli_tokenizer, 
                nli_model,
                [procedure for procedure, beam_search_questions in zip(batch_procedures, new_questions) for question in beam_search_questions],
                [question for beam_search_questions in new_questions for question in beam_search_questions],
                [questions],
                [answers],
            )
            parallel_idx = 0

            # Select best candidate based on coherence metrics
            selected_questions = []
            new_scores = []
            for batch_sub_idx, beam_search_questions in enumerate(new_questions):
                this_nli_outputs = nli_outputs[parallel_idx:parallel_idx + len(beam_search_questions)]
                candidate_questions_scores[batch_sub_idx].append(this_nli_outputs)
                parallel_idx += len(beam_search_questions)

                # Use marginal relevance (consistency) and/or expected informativeness (verifiability)
                if args.question_selection_strategy == "consistency":
                    candidate_scores = np.array(
                        [candidate_metrics['relevance_marginal' if question_idx > 0 else 'relevance'] for candidate_metrics in this_nli_outputs]
                    )
                elif args.question_selection_strategy == "verifiability":
                    candidate_scores = np.array(
                        [candidate_metrics['informativeness_marginal' if question_idx > 0 else 'informativeness'] for candidate_metrics in this_nli_outputs]
                    )
                elif args.question_selection_strategy == "coherence":
                    candidate_scores = np.array(
                        [candidate_metrics['relevance_marginal' if question_idx > 0 else 'relevance'] * candidate_metrics['informativeness_marginal' if question_idx > 0 else 'informativeness'] for candidate_metrics in this_nli_outputs]
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

        # TODO: apply visual filter and add logic to save frames for each example

        # Predict an answer (yes/no)
        prompts_a = [prompt + f'{question} ASSISTANT: A (yes/no): ' for prompt, question in zip(prompts_q, new_questions)]
        new_answers_logits = run_vqa(vlm, vlm_processor, prompts_a, batch_frames, batch_size=args.vqa_batch_size)
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

        # Save answers
        for batch_sub_idx in range(this_batch_size):
            vqa_outputs[batch_sub_idx].append(new_answers[batch_sub_idx])
            answers[batch_sub_idx].append(new_answers[batch_sub_idx].predicted_answer)

        # Update prompts with answers
        prompts = [prompt + pred.name for prompt, (pred, _) in zip(prompts_a, new_answers)]

        # Ask VLM probability of success
        prompt_success = [
            prompt + f' USER: Q: Based on the above information, is the procedure "{procedure}" 100% finished?'
            for prompt, procedure in zip(prompts, batch_procedures)
        ]
        success_vqa_outputs = run_vqa(
            vlm, 
            vlm_processor, 
            prompt_success, 
            batch_frames, 
            batch_size=args.vqa_batch_size
        )

        # Save success probability for this turn
        for batch_sub_idx in range(this_batch_size):
            success_probs[batch_sub_idx].append(
                success_vqa_outputs[batch_sub_idx].answer_probs[VQAResponse.Yes]
            )

    # Update global lists of tracked outputs
    all_questions += questions
    all_candidate_questions += candidate_questions
    all_candidate_questions_scores += candidate_questions_scores
    all_scores += scores
    all_vqa_outputs += vqa_outputs
    all_answers += answers
    all_success_probs += success_probs

    all_example_ids += [example.example_id for example in batch_examples]
    all_procedures += [example.procedure_description for example in batch_examples]
    all_labels += [example.mistake for example in batch_examples]

    # And cache tracked outputs
    pickle.dump((    
        all_questions, 
        all_candidate_questions, 
        all_candidate_questions_scores, 
        all_scores, 
        all_vqa_outputs, 
        all_answers, 
        all_success_probs,
        all_example_ids,
        all_procedures,
        all_labels,
    ), open(cache_path, "wb"))


print(f"({worker_index}) Gathering and scoring results...")

all_results_dicts = {}
all_probs = []
for questions, candidate_questions, candidate_questions_scores, scores, vqa_outputs, answers, success_probs, example_id, procedure, label \
    in tqdm(zip(all_questions,
                all_candidate_questions,
                all_candidate_questions_scores,
                all_scores,
                all_vqa_outputs,
                all_answers,
                all_success_probs,
                all_example_ids,
                all_procedures,
                all_labels), desc="evaluating"):
    
    final_success_prob = None
    for success_prob_idx, success_prob in enumerate(all_success_probs):
        # Early stopping mechanism: 
        # if success score doesn't change enough over 3 turns, stop incorporating questions
        final_success_prob = success_prob
        if success_prob_idx >= 2:
            score_change = 0.0
            for i in range(success_prob_idx - 2, success_prob_idx):
                score_change += scores[i+1] - scores[i]
            if score_change < args.early_stop_delta:
                break
    all_probs.append(round(final_success_prob, 6))

    results_dict = {
        "procedure": procedure,
        "mistake": label,
        "questions": questions,
        "answers": [a.name for a in answers],
        "answer_probs": [[float(output.answer_probs[VQAResponse(a)]) for a in range(2)] for output in vqa_outputs],
        "frame": [output.frame for output in vqa_outputs][0],
        "scores": scores,
        "success_probs": success_probs,
        "final_turn": success_prob_idx,
        "final_success_prob": final_success_prob,
        "candidate_questions": candidate_questions,
        "candidate_questions_scores": [[float(score) for score in this_scores] for this_scores in candidate_questions_scores],
    }
    all_results_dicts[example_id] = results_dict

json.dump(all_results_dicts, 
          open(os.path.join(this_results_dir, f"outputs_{args.eval_partition}.json"), "w"),
          indent=4)

# Calculate accuracy metrics
best_metrics = None
best_threshold = None
accuracy_metrics = {}
for threshold in MISTAKE_DETECTION_THRESHOLDS:
    preds = [1.0 - p >= threshold for p in all_probs]
    this_metrics = mistake_detection_metrics(all_labels, preds)
    accuracy_metrics[threshold] = this_metrics

    if best_metrics is None or (this_metrics['false_positive_rate'] + this_metrics['false_negative_rate']) < (best_metrics['false_positive_rate'] + best_metrics['false_negative_rate']):
        best_metrics = this_metrics
        best_threshold = threshold

accuracy_metrics['best_metrics'] = best_metrics
accuracy_metrics['best_threshold'] = best_threshold

json.dump(accuracy_metrics, 
          open(os.path.join(this_results_dir, f"metrics_accuracy_{args.eval_partition}.json"), "wb"),
          indent=4)

# Calculate coherence metrics of final rollouts
# TODO: implement this