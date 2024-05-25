# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
from collections import defaultdict
import datetime
import json
import os
from pprint import pprint
import shutil
import spacy
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

from travel.constants import DATA_CACHE_DIR, RESULTS_DIR
from travel.model.grounding import VisualFilterTypes, SpatialVisualFilter, ContrastiveRegionFilter
from travel.model.mistake_detection import MISTAKE_DETECTION_STRATEGIES, DETECTION_FRAMES_PROPORTION, generate_det_curve, compile_mistake_detection_preds
from travel.model.vqa import VQAOutputs, VQAResponse, SUCCESSVQA_PROMPT_TEMPLATES, get_vqa_response_token_ids, run_vqa
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.data.captaincook4d import CaptainCook4DDataset

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="captaincook4d", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=list(SUCCESSVQA_PROMPT_TEMPLATES.keys()), help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--eval_partitions", nargs='+', type=str, default=["val", "test"])
parser.add_argument("--mistake_detection_strategy", type=str, default="heuristic", choices=list(MISTAKE_DETECTION_STRATEGIES.keys()))
parser.add_argument("--visual_filter_mode", type=str, required=False, choices=[t.value for t in VisualFilterTypes], help="Visual attention filter mode.")
parser.add_argument("--batch_size", type=int, default=10, help="Batch size for VQA inference.")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of generating frameVQA examples.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
args = parser.parse_args()

# Load VLM
vlm_processor = AutoProcessor.from_pretrained(args.vlm_name)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
vlm = AutoModelForVision2Seq.from_pretrained(args.vlm_name, 
                                             cache_dir=DATA_CACHE_DIR,
                                             quantization_config=bnb_config)
vlm_processor.tokenizer.padding_side = "left"
vlm.language_model.generation_config.temperature = None
vlm.language_model.generation_config.top_p = None
vlm.language_model.generation_config.do_sample = False

if args.visual_filter_mode is not None:
    if VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Spatial:
        raise NotImplementedError("Spatial filter not implemented.")
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Spatial_NoRephrase:
        raise NotImplementedError("Spatial filter not implemented.")
    elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Contrastive_Region:
        visual_filter = ContrastiveRegionFilter(device="cuda:1" if torch.cuda.device_count() >= 2 else None)
        nlp = spacy.load('en_core_web_sm')
    else:
        raise NotImplementedError(f"Visual filter type {args.visual_filter_mode} is not compatible with VQG2VQA!")

prompt_template = SUCCESSVQA_PROMPT_TEMPLATES[type(vlm)]
response_token_ids = get_vqa_response_token_ids(vlm_processor.tokenizer)

# Configure results directory
if args.resume_dir is None:
    timestamp = datetime.datetime.now()
    this_results_dir = f"SuccessVQA"
    if args.debug:
        this_results_dir += f"_debug"
    this_results_dir += f"_{args.vlm_name.split('/')[-1]}_{timestamp.strftime('%Y%m%d%H%M%S')}"
    this_results_dir = os.path.join(RESULTS_DIR, "vqa_mistake_detection", this_results_dir)
    os.makedirs(this_results_dir)
else:
    this_results_dir = args.resume_dir

for eval_partition in args.eval_partitions:

    # Load mistake detection dataset
    if args.task == "captaincook4d":
        eval_dataset = CaptainCook4DDataset(data_split=eval_partition, debug_n_examples_per_class=20 if args.debug else None)
    # TODO: integrate ego4d here
    else:
        raise NotImplementedError(f"Haven't implemented usage of {args.task} dataset yet!")                                        

    # Generate SuccessVQA prompts
    # TODO: should we apply "contrastive region guidance" as a normalization for this baseline?
    frames = []
    prompts = []
    answers = []
    example_ids = []
    for example in tqdm(eval_dataset, "generating prompts"):
        step_id = example.procedure_id
        step = example.procedure_description
        
        # TODO: consider doing more prompt engineering to improve performance
        prompt = prompt_template.format(step=step)
        expected_answer = VQAResponse["Yes"]

        example.cutoff_to_last_frames(DETECTION_FRAMES_PROPORTION)
        for frame in example.frames:
            frames.append(frame)
            prompts.append(prompt)  
            answers.append(expected_answer)
            example_ids.append(example.example_id)

    # Run visual filter if we have one
    if args.visual_filter_mode is not None:
        if VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Contrastive_Region:
            original_frames = frames
            frames = visual_filter(nlp, frames, questions)
        else:
            frames, questions = visual_filter(nlp, frames, questions)

    prompts = []
    for question in questions:
        prompts.append(prompt_template.format(question=question.strip()))
        
    # Run SuccessVQA inference
    logits = run_vqa(vlm,
                    vlm_processor,
                    prompts,
                    frames,
                    batch_size=args.batch_size,
                    cache_path=os.path.join(this_results_dir, f"VQA_cache_{eval_partition}.pt"))
    
    if VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Contrastive_Region:
            original_logits = run_vqa(vlm,
                     vlm_processor,
                     prompts,
                     original_frames,
                     batch_size=args.batch_size,
                     cache_path=os.path.join(this_results_dir, f"VQA_cache_{eval_partition}_crg_original.pt"))

    outputs_by_id = defaultdict(list)
    for output_index, (frame, prompt, answer, eid) in enumerate(zip(frames, prompts, answers, example_ids)):
        outputs_by_id[eid].append((output_index, frame, prompt, answer))

    vqa_outputs = []
    for example in tqdm(eval_dataset, "gathering VQA outputs"):
        this_vqa_outputs = []
        for output_index, frame, prompt, answer in outputs_by_id[example.example_id]:
            this_vqa_outputs.append(
                [VQAOutputs(
                    example.task_name,
                    example.example_id,
                    step_id,
                    frame,
                    prompt,
                    answer,
                    response_token_ids,
                    original_logits[output_index] - logits[output_index],        
            ) if VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Contrastive_Region else
                VQAOutputs(
                    example.task_name,
                    example.example_id,
                    step_id,
                    frame,
                    prompt,
                    answer,
                    response_token_ids,
                    logits[output_index],        
                )]
            )
            
        vqa_outputs.append(this_vqa_outputs)

    evaluator = MISTAKE_DETECTION_STRATEGIES[args.mistake_detection_strategy](eval_dataset, vqa_outputs)
    mistake_detection_preds, metrics = evaluator.evaluate_mistake_detection()
    print(f"Mistake Detection Metrics ({eval_partition}, Detection Threshold=0.5):")
    pprint(metrics[0.5])

    # Compile preds per mistake detection example
    preds = compile_mistake_detection_preds(eval_dataset, vqa_outputs, mistake_detection_preds, image_base_path=this_results_dir)

    # Save metrics, preds, DET curve, config file (which may have some parameters that vary over time), and command-line arguments
    metrics_filename = f"metrics_{args.mistake_detection_strategy}_{eval_partition}.json"
    json.dump(metrics, open(os.path.join(this_results_dir, metrics_filename), "w"), indent=4)

    preds_filename = f"preds_{eval_partition}.json"
    json.dump(preds, open(os.path.join(this_results_dir, preds_filename), "w"), indent=4)

    det_filename = f"det_{args.mistake_detection_strategy}_{eval_partition}.pdf"
    generate_det_curve(metrics, os.path.join(this_results_dir, det_filename))

shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)