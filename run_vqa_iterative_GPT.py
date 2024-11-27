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
import shutil
import spacy
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer, PhrasalConstraint           

from travel.constants import RESULTS_DIR, IMAGES_CHUNK_SIZE, HF_TOKEN, CONFIG_PATH
from travel.data.captaincook4d import CaptainCook4DDataset
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.data.vqa import VQAResponse, get_vqa_response_token_ids, VQAOutputs, DIALOG_START_TOKENS, IMAGE_TOKENS, USER_START_TOKENS, USER_END_TOKENS, ASSISTANT_START_TOKENS, ASSISTANT_END_TOKENS, IVQA_PREAMBLE, IVQA_SUCCESS_QUESTION
from travel.data.vqg import generate_vqg_prompt_icl
from travel.model import simple_lm_prompt_beam_search, simple_vlm_prompt_beam_search, compute_completion_log_likelihoods, compute_completion_log_likelihoods_encoder_decoder, compute_completion_log_likelihoods_vlm
from travel.model.grounding import VisualFilterTypes, ContrastiveRegionFilter, VisualContrastiveFilter, SpatialVisualFilter, AGLAFilter, ImageMaskTypes
from travel.model.metrics import mistake_detection_metrics, question_coherence_metrics_nli, question_coherence_metrics_vlm, generate_det_curve, generate_tiered_metric_curves, entropy, compile_accuracy_and_coherence_metrics
from travel.model.mistake_detection import MISTAKE_DETECTION_THRESHOLDS
from travel.model.nli import NLI_MODEL_PATH, NLI_BATCH_SIZE
from travel.model.vqa import run_vqa_with_visual_filter
from travel.model.vqg import cleanup_generated_question
from travel.model.api import GPT

parser = argparse.ArgumentParser()

parser.add_argument("--vlm_name", type=str, default="gpt-4o-mini", help="Name for GPT model to use.")
parser.add_argument("--api_key", type=str, required=True, help="API key to send a request to GPT.")
parser.add_argument("--endpoint", type=str, required=False, help="Endpoint URL to send a request to OpenAI.")
parser.add_argument("--task", type=str, default="ego4d_single", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--eval_partition", type=str, default="val", choices=["train", "val", "test"])
parser.add_argument("--max_iterations", type=int, default=10, help="Maximum number of questions to generate before making a final mistake detection decision.")
parser.add_argument("--exclude_history_from_vqa", action="store_true", help="Pass this argument to exclude the dialog history from VQA, and instead directly ask only questions.")
parser.add_argument("--coherence_evaluation_strategy", type=str, default="nli", choices=["vlm", "nli"], help="Strategy to use to perform final coherence evaluation of dialog.")
parser.add_argument("--early_stop_delta", type=float, default=0.1, help="If success probability changes less than this over 3 turns, stop generating questions.")
parser.add_argument("--confident_range", type=float, default=0.05, help="If success probability is within this from 0.0 or 1.0, stop early due to high confidence.")
parser.add_argument("--unsure_range", type=float, default=0.1, help="A VQA output will be considered unsure if the probability of yes and no are within this range of 50 percent (exclusive).")
parser.add_argument("--visual_filter_mode", type=str, required=False, choices=[t.value for t in VisualFilterTypes], help="Visual attention filter mode.")
parser.add_argument("--visual_filter_strength", type=float, required=False, default=1.0, help="Float strength for masks used in visual filters. Depending on the visual filter type, this may be interpreted as a percentage darkness or a Gaussian blur kernel size.")
parser.add_argument("--generation_batch_size", type=int, default=10, help="Batch size for question generation with LM.")
parser.add_argument("--vqa_batch_size", type=int, default=10, help="Batch size for VQA with VLM.")
parser.add_argument("--nli_batch_size", type=int, default=NLI_BATCH_SIZE, help="Batch size for scoring candidate questions with NLI model.")
parser.add_argument("--run_id", type=str, required=False, help="Unique ID for this run, which will be used to create the output directory (and should be shared across any parallel processes).")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of iterative VQA.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
parser.add_argument("--get_negated_success_probs", action="store_true", help="Pass this argument to calculate success probabilities for negated answers to questions.")
parser.add_argument("--run_allturns_metrics", action="store_true", help="Pass this argument run an additional set of metrics without early stopping.")
parser.add_argument("--cache_vqa_frames", action="store_true", help="Pass this argument to cache frames in VQA outputs (e.g., to inspect visual filter resuilts). This consumes a lot of disk space for large datasets.")
parser.add_argument("--print_prompts", action="store_true", help="Pass this argument to print some sample prompts during execution (for debugging purposes).")
args = parser.parse_args()

if args.cache_vqa_frames and args.visual_filter_mode is None:
    print("Warning: --cache_vqa_frames only applies to frames modified by visual filters (configured through --visual_filter_mode and --visual_filter_strength).")
if args.run_allturns_metrics and args.coherence_evaluation_strategy != "nli":
    print("Warning: --run_allturns_metrics only works with NLI-based coherence evaluation. Will not run all-turns metrics.")

if not args.endpoint or not args.api_key:
    raise ValueError("For GPT usage, you need to pass your endpoint URL and API key.")

lm = GPT(api_key=args.api_key,
          endpoint=args.endpoint,
          model_name=args.vlm_name)

# Set up results directory
if args.resume_dir is None:
    vlm_name = args.vlm_name.split('/')[-1]
    task_name = args.task
    if args.debug:
        task_name += f"_debug{args.debug_n_examples}" if args.task != "captaincook4d" else "_debug"
    this_results_dir = os.path.join(task_name, vlm_name, f"IterativeVQA_q{args.max_iterations}_{task_name}")
    this_results_dir += f"_{vlm_name}" 
    if args.exclude_history_from_vqa:
        this_results_dir += "_nohistory"
    if args.visual_filter_mode is not None:
        this_results_dir += f"_{args.visual_filter_mode}{args.visual_filter_strength}"
    this_results_dir += f"_{args.run_id}"
    this_results_dir = os.path.join(RESULTS_DIR, "vqa_mistake_detection", this_results_dir)
    if not os.path.exists(this_results_dir):
        os.makedirs(this_results_dir)
else:
    this_results_dir = args.resume_dir

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

response_token_ids = lm.get_vqa_response_token_ids()

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

# NLI model to score consistency and verifiability
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)

# Load approopriate evaluation dataset
dataset = None
for retry in range(5):
    print(f"Loading evaluation dataset (try {retry})...")
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


print(f"Beginning iterative VQA inference...")
all_questions = []
all_answers = []
all_answer_probs = []
all_success_probs = []
all_success_probs_negated = []

all_example_ids = []
all_procedures = []
all_labels = []

cache_path = os.path.join(this_results_dir, f"cached_outputs.pkl")
is_complete = False
last_batch_idx = -1
if os.path.exists(cache_path):
    is_complete, last_batch_idx, all_questions, all_answers, all_answer_probs, all_success_probs, all_success_probs_negated, all_example_ids, all_procedures, all_labels = pickle.load(open(cache_path, "rb"))

batch_idx = None
if not is_complete:
    for batch_idx, batch_examples in tqdm(enumerate(dataset.get_batches(IMAGES_CHUNK_SIZE, 
                                                                        n_workers=1, 
                                                                        worker_index=0,
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
            f'{IVQA_PREAMBLE.format(procedure=procedure)} Generate an appropriate yes/no question.'  
            for procedure in batch_procedures
        ]
        if args.print_prompts:
            pprint(prompts[0])
        questions = [[] for _ in range(this_batch_size)]
        frames = [[] for _ in range(this_batch_size)]
        answer_probs = [[] for _ in range(this_batch_size)] 
        answers = [[] for _ in range(this_batch_size)]
        success_probs = [[] for _ in range(this_batch_size)]
        success_probs_negated = [[] for _ in range(this_batch_size)]

        # Iteratively generate questions
        for question_idx in tqdm(range(args.max_iterations), desc="running iterative QA"):

            # Generate questions
            prompts_q = [prompt + f"{"\nQ:" if question_idx != 0 else " Q:"}" for prompt in prompts]
            new_questions = lm.generate_questions(prompts_q, max_tokens=20, temperature=0)

            new_questions = [cleanup_generated_question(question) for question in new_questions]

            # Save generated questions
            for batch_sub_idx in range(this_batch_size):
                questions[batch_sub_idx].append(new_questions[batch_sub_idx])

            # Run VQA with generated questions (and optional spatial filter)
            prompts_a = [prompt.replace(' Generate an appropriate yes/no question.', "") + f' {question}\nA:' for prompt, question in zip(prompts_q, new_questions)]
            if args.print_prompts:
                pprint(prompts_a[0])

            # Effective prompt for VQA depends on whether we want to exclude dialog history from prompt
            if not args.exclude_history_from_vqa:
                use_prompts_a = prompts_a
            else:
                use_prompts_a = [f'Q: {question}\nA:' for question in new_questions]
            
            new_answer_probs = lm.run_GPT_vqa(prompts=use_prompts_a,
                                              frames=batch_frames,
                                              temperature=0)

            # Gather up VQA outputs
            new_answers = [
                VQAOutputs(
                    task_name=MistakeDetectionTasks(args.task),
                    example_id=example.example_id,
                    procedure_id=example.procedure_id,
                    frame=example.frames[0],
                    prompt=prompt,
                    expected_answer=None,
                    response_token_ids=response_token_ids,
                    question=question,
                    answer_probs=probs,
                    predicted_answer=VQAResponse.Yes if probs[VQAResponse.Yes] > 0.5 else VQAResponse.No
                ) for probs, example, prompt, question in zip(new_answer_probs, batch_examples, prompts_a, new_questions)
            ]
            new_answers_str = [output.predicted_answer.name if np.abs(output.answer_probs[VQAResponse.Yes] - 0.5) >= args.unsure_range else "Unsure" for output in new_answers]

            # Save answers and their probabilities
            for batch_sub_idx in range(this_batch_size):
                answer_probs[batch_sub_idx].append([round(float(new_answers[batch_sub_idx].answer_probs[VQAResponse(answer_idx)]), 6) for answer_idx in range(2)])
                answers[batch_sub_idx].append(new_answers_str[batch_sub_idx])

            # Update prompts with answers
            if args.coherence_evaluation_strategy == "vlm" or args.get_negated_success_probs:
                # Save negated version of new prompt if we're using VLM-based coherence evaluation
                new_answers_str_negated = [VQAResponse(1 - output.predicted_answer.value).name if np.abs(output.answer_probs[VQAResponse.Yes] - 0.5) >= args.unsure_range else "Unsure" for output in new_answers]
                prompts_negated = [prompt + ' ' + output for prompt, output in zip(prompts_a, new_answers_str_negated)] 
            
            
            prompts = [prompt + " " + output for prompt, output in zip(prompts_a, new_answers_str)]

            # Ask VLM probability of success
            questions_success = [
                IVQA_SUCCESS_QUESTION.format(procedure=procedure)
                for procedure in batch_procedures
            ]
            prompts_success = [
                prompt + f'\nQ: {question}\nA: '
                for prompt, question in zip(prompts, questions_success)
            ]
            if args.print_prompts:
                pprint(prompts_success[0])
    
            success_vqa_probs = lm.run_GPT_vqa(prompts=prompts_success,
                                               frames=batch_frames,
                                               temperature=0)

            success_vqa_outputs = [
                VQAOutputs(
                    task_name=MistakeDetectionTasks(args.task),
                    example_id=example.example_id,
                    procedure_id=example.procedure_id,
                    frame=example.frames[0],
                    prompt=prompt,
                    expected_answer=None,
                    response_token_ids=response_token_ids,
                    question=question,
                    answer_probs=probs,
                    predicted_answer=VQAResponse.Yes if probs[VQAResponse.Yes] > 0.5 else VQAResponse.No
                ) for probs, example, prompt, question in zip(success_vqa_probs, batch_examples, prompts_a, new_questions)
            ]               

            # Save success probability for this turn
            for batch_sub_idx in range(this_batch_size):
                success_probs[batch_sub_idx].append(
                    round(float(success_vqa_outputs[batch_sub_idx].answer_probs[VQAResponse.Yes]), 6)
                )

            # Clear out VQA outputs now because they occupy a lot of memory
            del new_answers
            del success_vqa_outputs

            # If using VLM-based coherence evaluation, also need to get success probability for negated answers
            if args.coherence_evaluation_strategy == "vlm" or args.get_negated_success_probs:
                prompts_success_negated = [
                    prompt + f'\nQ: {question}\nA:'
                    for prompt, question in zip(prompts_negated, questions_success)
                ]
                success_vqa_probs_negated = lm.run_GPT_vqa(prompts=prompts_success_negated,
                                                           frames=batch_frames,
                                                           temperature=0)

                success_vqa_outputs_negated = [
                    VQAOutputs(
                        task_name=MistakeDetectionTasks(args.task),
                        example_id=example.example_id,
                        procedure_id=example.procedure_id,
                        frame=example.frames[0],
                        prompt=prompt,
                        expected_answer=None,
                        response_token_ids=response_token_ids,
                        question=question,
                        answer_probs=probs,
                        predicted_answer=VQAResponse.Yes if probs[VQAResponse.Yes] > 0.5 else VQAResponse.No
                    ) for probs, example, prompt, question in zip(success_vqa_probs_negated, batch_examples, prompts_a, new_questions)
                ]

                # Save success probability for negated question answers for this turn
                for batch_sub_idx in range(this_batch_size):
                    success_probs_negated[batch_sub_idx].append(
                        round(float(success_vqa_outputs_negated[batch_sub_idx].answer_probs[VQAResponse.Yes]), 6)
                    )

                # Delete VQAOutputs now since they occupy a lot of memory
                del success_vqa_outputs_negated


        # Update global lists of tracked outputs
        all_questions += questions
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
        all_answers, 
        all_answer_probs, 
        all_success_probs,
        all_success_probs_negated,
        all_example_ids,
        all_procedures,
        all_labels,
    ), open(cache_path, "wb"))

print(f"Done running iterative VQA inference!")


# Evaluate
# Collect key information from results rollouts and final success probabilities
all_results_dicts = {}
all_probs = []
for questions, answers, answer_probs, success_probs, success_probs_negated, example_id, procedure, label \
    in tqdm(zip(all_questions,
                all_answers,
                all_answer_probs,
                all_success_probs,
                all_success_probs_negated,
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
        # OR if success score is within confident_delta of 0.0 or 1.0 (i.e., highly confident), stop
        if success_prob < args.confident_range or 1.0 - success_prob < args.confident_range:
            break           
    all_probs.append(round(final_success_prob, 6))   

    results_dict = {
        "procedure": procedure,
        "mistake": True if label is not None else False,
        "mistake_type": label,
        "questions": questions,
        "frame_dir": os.path.join(this_results_dir, f"vqa_frames/{example_id}") if args.cache_vqa_frames else dataset.get_example_dir(example_id),
        "answers": answers,
        "answer_probs": answer_probs,
        "success_probs": success_probs,
        "success_probs_negated": success_probs_negated,
        "final_turn": success_prob_idx,
        "final_success_prob": final_success_prob
    }
    all_results_dicts[example_id] = results_dict

json.dump(all_results_dicts, 
        open(os.path.join(this_results_dir, f"outputs_{args.eval_partition}.json"), "w"),
        indent=4)


print(f"Evaluating outputs...")
metrics = {}

if args.coherence_evaluation_strategy == "nli":
    # Calculate coherence metrics of final rollouts
    all_chosen_questions = [question for results_dict in all_results_dicts.values() for question in results_dict['questions'][:results_dict['final_turn'] + 1]]
    all_previous_questions = [[q for qi, q in enumerate(results_dict['questions'][:question_idx]) if results_dict['answers'][qi] != "Unsure"] for results_dict in all_results_dicts.values() for question_idx in range(results_dict['final_turn'] + 1)]

    label_answer_mapping = {0: "No", 1: "Yes"}
    all_predicted_answers = [label_answer_mapping[np.argmax(answer_probs)] for results_dict in all_results_dicts.values() for answer_probs in results_dict['answer_probs'][:results_dict['final_turn'] + 1]]
    all_previous_answers = [[a for a in results_dict['answers'][:question_idx] if a != "Unsure"] for results_dict in all_results_dicts.values() for question_idx in range(results_dict['final_turn'] + 1)]

    all_coherence_metrics = question_coherence_metrics_nli(nli_tokenizer,
                                            nli_model,
                                            tokenizer,
                                            lm,                                         
                                            [procedure for results_dict, procedure in zip(all_results_dicts.values(), all_procedures) for _ in range(results_dict['final_turn'] + 1)],
                                            all_chosen_questions,
                                            answers=all_predicted_answers,
                                            previous_questions=all_previous_questions,
                                            previous_answers=all_previous_answers,
                                            mistake_labels=[results_dict['mistake'] for results_dict in all_results_dicts.values() for _ in range(results_dict['final_turn'] + 1)],
                                            rephrase_batch_size=args.generation_batch_size)

    if args.run_allturns_metrics:
        # Calculate alternative metrics based on all iterations
        all_chosen_questions = [question for results_dict in all_results_dicts.values() for question in range(len(results_dict['questions']))]
        all_previous_questions = [[q for qi, q in enumerate(results_dict['questions'][:question_idx]) if results_dict['answers'][qi] != "Unsure"] for results_dict in all_results_dicts.values() for question_idx in range(len(results_dict['questions']))]

        label_answer_mapping = {0: "No", 1: "Yes"}
        all_predicted_answers = [label_answer_mapping[np.argmax(answer_probs)] for results_dict in all_results_dicts.values() for answer_probs in range(len(results_dict['answer_probs']))]
        all_previous_answers = [[a for a in results_dict['answers'][:question_idx] if a != "Unsure"] for results_dict in all_results_dicts.values() for question_idx in range(len(results_dict['answers']))]

        all_coherence_metrics_allturns = lm.question_coherence_metrics_nli_GPT(nli_tokenizer,
                                                                               nli_model,
                                                                               [procedure for results_dict, procedure in zip(all_results_dicts.values(), all_procedures) for _ in range(args.max_iterations)],
                                                                               all_chosen_questions,
                                                                               answers=all_predicted_answers,
                                                                               previous_questions=all_previous_questions,
                                                                               previous_answers=all_previous_answers,
                                                                               mistake_labels=[results_dict['mistake'] for results_dict in all_results_dicts.values() for _ in range(args.max_iterations)])

elif args.coherence_evaluation_strategy == "vlm":
    all_coherence_metrics = question_coherence_metrics_vlm(all_success_probs, all_success_probs_negated)
else:
    raise NotImplementedError(f"Coherence evaluation strategy {args.coherence_evaluation_strategy} not supported yet.")

# Get accuracy and coherence metrics
accuracy_metrics_by_threshold, coherence_metrics = compile_accuracy_and_coherence_metrics(all_labels, all_probs, all_coherence_metrics, all_results_dicts, MISTAKE_DETECTION_THRESHOLDS, args.unsure_range)
coherence_metrics_by_threshold = coherence_metrics['metrics_by_threshold']

# Save accuracy and coherence metrics
json.dump(accuracy_metrics_by_threshold, 
        open(os.path.join(this_results_dir, f"metrics_accuracy_{args.eval_partition}.json"), "w"),
        indent=4)

json.dump(coherence_metrics, 
        open(os.path.join(this_results_dir, f"metrics_coherence_{args.coherence_evaluation_strategy}_{args.eval_partition}.json"), "w"),
        indent=4)

json.dump(all_coherence_metrics, 
        open(os.path.join(this_results_dir, f"metrics_coherence_raw_{args.coherence_evaluation_strategy}_{args.eval_partition}.json"), "w"),
        indent=4)            

# Generate DET curves for accuracy
generate_det_curve(accuracy_metrics_by_threshold, os.path.join(this_results_dir, f"det_accuracy_{args.eval_partition}.pdf"))

# Get all-turns form of metrics and save them
if args.coherence_evaluation_strategy == 'nli' and args.run_allturns_metrics:
    all_probs_allturns = [results_dict['success_probs'][-1] for results_dict in all_results_dicts.values()]
    accuracy_metrics_by_threshold_allturns, coherence_metrics_allturns = compile_accuracy_and_coherence_metrics(all_labels, all_probs_allturns, all_coherence_metrics_allturns, all_results_dicts, MISTAKE_DETECTION_THRESHOLDS, args.unsure_range)

    allturns_results_dir = os.path.join(this_results_dir, "allturns")
    if not os.path.exists(allturns_results_dir):
        os.makedirs(allturns_results_dir)
    json.dump(accuracy_metrics_by_threshold_allturns, 
            open(os.path.join(allturns_results_dir, f"metrics_accuracy_{args.eval_partition}.json"), "w"),
            indent=4)
    
    json.dump(coherence_metrics_allturns, 
            open(os.path.join(allturns_results_dir, f"metrics_coherence_{args.coherence_evaluation_strategy}_{args.eval_partition}.json"), "w"),
            indent=4)
    
    json.dump({k: v | {"final_turn": args.max_iterations - 1} for k, v in all_results_dicts.items()}, 
            open(os.path.join(allturns_results_dir, f"outputs_{args.eval_partition}.json"), "w"),
            indent=4)

# Generate curves for all metrics by threshold
generate_tiered_metric_curves(MISTAKE_DETECTION_THRESHOLDS, 
                                [accuracy_metrics_by_threshold[t]['accuracy'] for t in MISTAKE_DETECTION_THRESHOLDS],
                                [coherence_metrics_by_threshold[t]['consistency'] for t in MISTAKE_DETECTION_THRESHOLDS], 
                                [coherence_metrics_by_threshold[t]['verifiability'] for t in MISTAKE_DETECTION_THRESHOLDS],
                                [os.path.join(this_results_dir, f"graph_tiered_metrics_{args.coherence_evaluation_strategy}_{args.eval_partition}.pdf")])

# Save args and config
shutil.copy(CONFIG_PATH, os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)

print(f"Done!")

