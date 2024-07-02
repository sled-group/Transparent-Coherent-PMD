# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
import concurrent.futures
import datetime
import json
import os
from PIL import Image
from pprint import pprint
import shutil
import spacy
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

from travel.constants import DATA_CACHE_DIR, RESULTS_DIR
from travel.data.captaincook4d import CaptainCook4DDataset
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.mistake_detection import MistakeDetectionTasks, MistakeDetectionExample
from travel.data.vqa import VQAResponse, SUCCESSVQA_PROMPT_TEMPLATES, get_vqa_response_token_ids
from travel.model.grounding import VisualFilterTypes, ContrastiveRegionFilter
from travel.model.mistake_detection import MISTAKE_DETECTION_STRATEGIES, generate_det_curve, compile_mistake_detection_preds
from travel.model.vqa import run_vqa_for_mistake_detection

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ego4d", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=list(SUCCESSVQA_PROMPT_TEMPLATES.keys()), help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--eval_partitions", nargs='+', type=str, default=["val", "test"])
parser.add_argument("--mistake_detection_strategy", type=str, default="heuristic", choices=list(MISTAKE_DETECTION_STRATEGIES.keys()))
parser.add_argument("--visual_filter_mode", type=str, required=False, choices=[t.value for t in VisualFilterTypes], help="Visual attention filter mode.")
parser.add_argument("--visual_filter_strength", type=float, required=False, default=1.0, help="Float strength for masks used in visual filters.")
parser.add_argument("--batch_size", type=int, default=10, help="Batch size for VQA inference.")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of generating frameVQA examples.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
parser.add_argument("--cache_vqa_frames", action="store_true", help="Pass this argument to cache frames in VQA outputs (e.g., to inspect visual filter resuilts). This consumes a lot of disk space for large datasets.")
args = parser.parse_args()

# Load VLM(s), processors, visual filters, etc. - if multiple GPUs available, use them
print("Setting up VLMs and visual filters...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
n_workers = 1 if torch.cuda.device_count() <= 1 else torch.cuda.device_count()
vlms = []
vlm_processors = []
visual_filters = []
nlps = []
for worker_index in range(n_workers):
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{worker_index}")
    vlm = AutoModelForVision2Seq.from_pretrained(args.vlm_name, 
                                                quantization_config=bnb_config)
    vlm.language_model.generation_config.temperature = None
    vlm.language_model.generation_config.top_p = None
    vlm.language_model.generation_config.do_sample = False
    vlms.append(vlm)
        
    vlm_processor = AutoProcessor.from_pretrained(args.vlm_name)
    vlm_processor.tokenizer.padding_side = "left"
    vlm_processors.append(vlm_processor)

    if args.visual_filter_mode is not None:
        if VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Contrastive_Region:
            visual_filter = ContrastiveRegionFilter(mask_strength=args.visual_filter_strength, device=f"cuda:{worker_index}")
            nlp = spacy.load('en_core_web_lg')
        else:
            raise NotImplementedError(f"Visual filter type {args.visual_filter_mode} is not compatible with SuccessVQA!")
        
        visual_filters.append(visual_filter)
        nlps.append(nlp)
    else:
        visual_filters.append(None)
        nlps.append(None)

prompt_template = SUCCESSVQA_PROMPT_TEMPLATES[type(vlm)]
response_token_ids = get_vqa_response_token_ids(vlm_processors[0].tokenizer)

# Configure results directory
if args.resume_dir is None:
    timestamp = datetime.datetime.now()
    this_results_dir = os.path.join(args.task, f"SuccessVQA_{args.task}")
    if args.debug:
        this_results_dir += f"_debug{args.debug_n_examples}"        
    this_results_dir += f"_{args.vlm_name.split('/')[-1]}"
    if args.visual_filter_mode is not None:
        this_results_dir += f"_{args.visual_filter_mode}{args.visual_filter_strength}"
    this_results_dir += f"_{timestamp.strftime('%Y%m%d%H%M%S')}"
    this_results_dir = os.path.join(RESULTS_DIR, "vqa_mistake_detection", this_results_dir)
    os.makedirs(this_results_dir)
else:
    this_results_dir = args.resume_dir

def generate_prompts(example: MistakeDetectionExample) -> tuple[list[str], list[str], list[VQAResponse], list[Image.Image]]:
    questions = []
    prompts = []
    answers = []
    frames = []
    prompt = prompt_template.format(step=example.procedure_description)
    question = prompt
    expected_answer = VQAResponse["Yes"]

    for frame in example.frames:
        questions.append(question)
        prompts.append(prompt)
        answers.append(expected_answer)
        frames.append(frame)

    return questions, prompts, answers, frames

for eval_partition in args.eval_partitions:
    print(f"Running VQA on {eval_partition}...")

    # Load mistake detection dataset
    if MistakeDetectionTasks(args.task) == MistakeDetectionTasks.CaptainCook4D:
        eval_datasets = [CaptainCook4DDataset(data_split=eval_partition, debug_n_examples_per_class=args.debug_n_examples if args.debug else None) for _ in range(n_workers)]
    elif MistakeDetectionTasks(args.task) == MistakeDetectionTasks.Ego4D:
        eval_datasets = [Ego4DMistakeDetectionDataset(data_split=eval_partition, 
                                                      mismatch_augmentation=True,
                                                      multi_frame=True,
                                                      debug_n_examples_per_class=args.debug_n_examples if args.debug else None) for _ in range(n_workers)]
    elif MistakeDetectionTasks(args.task) == MistakeDetectionTasks.Ego4D_Single:
        eval_datasets = [Ego4DMistakeDetectionDataset(data_split=eval_partition, 
                                                      mismatch_augmentation=True,
                                                      multi_frame=False,
                                                      debug_n_examples_per_class=args.debug_n_examples if args.debug else None) for _ in range(n_workers)]
    else:
        raise NotImplementedError(f"Haven't implemented usage of {args.task} dataset yet!")                                        
        
    if torch.cuda.device_count() >= 2: 
        print(f"Running VQA in parallel across {torch.cuda.device_count()} GPUs...")
        with concurrent.futures.ThreadPoolExecutor() as executor:   
            partitions = list(executor.map(run_vqa_for_mistake_detection, 
                                           eval_datasets,
                                           vlms,
                                           vlm_processors,
                                           [generate_prompts] * n_workers,
                                           [1] * n_workers,
                                           [None if args.visual_filter_mode is None else VisualFilterTypes(args.visual_filter_mode)] * n_workers,
                                           visual_filters,
                                           nlps,
                                           [this_results_dir] * n_workers,
                                           [n_workers] * n_workers,
                                           list(range(n_workers)),
                                           [args.batch_size] * n_workers,
                                           [args.cache_vqa_frames] * n_workers)
                             )
        # Compile processed data from partitions
        vqa_outputs = []
        for this_vqa_outputs in partitions:
            vqa_outputs += this_vqa_outputs        
    else:
        print("Running VQA sequentially...")
        vqa_outputs = run_vqa_for_mistake_detection(eval_dataset=eval_datasets[0],
                                                    vlm=vlms[0],
                                                    vlm_processor=vlm_processors[0],
                                                    generate_prompts=generate_prompts,
                                                    n_prompts_per_frame=1,
                                                    visual_filter_mode=None if args.visual_filter_mode is None else VisualFilterTypes(args.visual_filter_mode),
                                                    visual_filter=visual_filters[0],
                                                    nlp=nlps[0],
                                                    cache_dir=this_results_dir,
                                                    n_workers=1,
                                                    worker_index=0,
                                                    vqa_batch_size=args.batch_size,
                                                    cache_frames=args.cache_vqa_frames)
    
    print("Evaluating and saving results...")

    # Free up memory in case we need to load another model during mistake detection
    del vlms
    del vlm_processors
    if args.visual_filter_mode is not None:
        del visual_filters

    evaluator = MISTAKE_DETECTION_STRATEGIES[args.mistake_detection_strategy](eval_datasets[0], vqa_outputs)
    mistake_detection_preds, metrics = evaluator.evaluate_mistake_detection()
    print(f"Mistake Detection Metrics ({eval_partition}, Detection Threshold={metrics['best_threshold']}):")
    pprint(metrics[metrics['best_threshold']])

    # Compile preds per mistake detection example
    preds = compile_mistake_detection_preds(eval_datasets[0], vqa_outputs, mistake_detection_preds, image_base_path=this_results_dir)

    # Save metrics, preds, DET curve, config file (which may have some parameters that vary over time), and command-line arguments
    metrics_filename = f"metrics_{args.mistake_detection_strategy}_{eval_partition}.json"
    json.dump(metrics, open(os.path.join(this_results_dir, metrics_filename), "w"), indent=4)

    preds_filename = f"preds_{args.mistake_detection_strategy}_{eval_partition}.json"
    json.dump(preds, open(os.path.join(this_results_dir, preds_filename), "w"), indent=4)

    det_filename = f"det_{args.mistake_detection_strategy}_{eval_partition}.pdf"
    generate_det_curve(metrics, os.path.join(this_results_dir, det_filename))

shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)