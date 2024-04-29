from travel.constants import MODEL_CACHE_DIR
import os
os.environ['HF_HOME'] = MODEL_CACHE_DIR

import argparse
import datetime
import json
import pickle
from pprint import pprint
import shutil
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

from travel.constants import DATA_CACHE_DIR, RESULTS_DIR
from travel.model.mistake_detection import MISTAKE_DETECTION_STRATEGIES, HEURISTIC_TARGET_FRAMES_PROPORTION, generate_det_curve, compile_mistake_detection_preds
from travel.model.vqa import VQAOutputs, VQAResponse, SUCCESSVQA_PROMPT_TEMPLATES, get_vqa_response_token_ids
from travel.data.mistake_detection import MistakeDetectionTasks, get_cutoff_time_by_proportion
from travel.data.captaincook4d import CaptainCook4DDataset

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="captaincook4d", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--eval_split", type=str, choices=["val", "test"])
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=list(SUCCESSVQA_PROMPT_TEMPLATES.keys()), help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--mistake_detection_strategy", type=str, default="heuristic", choices=list(MISTAKE_DETECTION_STRATEGIES.keys()))
parser.add_argument("--reset_cache", action="store_true", help="Pass this argument to not save cached VQA outputs for the VLM, and generate new ones.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
args = parser.parse_args()

# Load mistake detection dataset
eval_dataset = CaptainCook4DDataset(data_split=args.eval_split,
                                    debug_n_examples_per_class=20 if args.debug else None)

# Load VLM
vlm_processor = AutoProcessor.from_pretrained(args.vlm_name)
vlm = AutoModelForVision2Seq.from_pretrained(args.vlm_name, cache_dir=DATA_CACHE_DIR, load_in_8bit=True) # NOTE: when loading in 8bit, batched inference may output nans
vlm_processor.tokenizer.padding_side = "left"
vlm.language_model.generation_config.temperature = None
vlm.language_model.generation_config.top_p = None
vlm.language_model.generation_config.do_sample = False

prompt_template = SUCCESSVQA_PROMPT_TEMPLATES[args.vlm_name]
response_token_ids = get_vqa_response_token_ids(vlm_processor.tokenizer)

# Load cached VLM outputs
vqa_cache_fname = os.path.join(DATA_CACHE_DIR, f"vqa_successvqa_{args.vlm_name.replace('/','_')}.pkl")
if os.path.exists(vqa_cache_fname):
    vqa_cache = pickle.load(open(vqa_cache_fname, "rb"))
else:
    vqa_cache = {}

# Some mistake detection strategies are only applied to a specific proportion of frames; if so, we can skip running inference on these frames to save time
if args.mistake_detection_strategy == "heuristic":
    target_frames_proportion = HEURISTIC_TARGET_FRAMES_PROPORTION
else:
    target_frames_proportion = None

# Run SuccessVQA inference
# TODO: should we apply "contrastive region guidance" as a normalization for this baseline?
vqa_outputs = []
for example in tqdm(eval_dataset, "running inference on clips"):
    this_vqa_outputs = []
    
    step_id = example.procedure_id
    step = example.procedure_description
    
    # TODO: consider doing more prompt engineering to improve performance
    prompt = prompt_template.format(step=step)
    expected_answer = VQAResponse["Yes"]
    
    if target_frames_proportion is not None:
        cutoff_time = get_cutoff_time_by_proportion(example, target_frames_proportion)
    else:
        cutoff_time = None

    for frame, frame_time in zip(example.frames, example.frame_times):
        if cutoff_time is not None and frame_time < cutoff_time:
            # Don't run inference on this frame
            this_vqa_outputs.append([VQAOutputs(
                example.task_name,
                example.example_id,
                step_id,
                frame,
                prompt,
                expected_answer,
                response_token_ids,
                torch.zeros((vlm_processor.tokenizer.vocab_size)).float() # Placeholder zero logits since we didn't prompt the VLM
            )])
            continue

        # Forward pass
        with torch.no_grad():
            prompt_id = (example.example_id, frame_time, prompt)
            if prompt_id in vqa_cache:
                # Check if we already cached VLM outputs for this prompt
                logits = vqa_cache[prompt_id]
            else:
                # If we didn't, prompt the VLM
                inputs = vlm_processor(text=prompt, images=frame, return_tensors="pt").to(vlm.device)
                logits = vlm(**inputs).logits[0] # (seq length, vocab size)
                logits = logits[-1].detach().cpu() # (vocab size)
                vqa_cache[prompt_id] = logits

        this_vqa_outputs.append(
            [VQAOutputs(
                example.task_name,
                example.example_id,
                step_id,
                frame,
                prompt,
                expected_answer,
                response_token_ids,
                logits,        
            )]
        )
        
    vqa_outputs.append(this_vqa_outputs)

# Save VQA cache
pickle.dump(vqa_cache, open(vqa_cache_fname, "wb"))

evaluator = MISTAKE_DETECTION_STRATEGIES[args.mistake_detection_strategy](eval_dataset.examples, vqa_outputs)
mistake_detection_preds, metrics = evaluator.evaluate_mistake_detection()
print("Mistake Detection Metrics (Detection Threshold=0.5):")
pprint(metrics[0.5])

# Compile preds per mistake detection example
preds = compile_mistake_detection_preds(eval_dataset, vqa_outputs, mistake_detection_preds)

# Save metrics, preds, DET curve, config file (which may have some parameters that vary over time), and command-line arguments
timestamp = datetime.datetime.now()
this_results_dir = f"SuccessVQA"
if args.debug:
    this_results_dir += f"_debug"
this_results_dir += f"_{args.vlm_name.split('/')[-1]}_{timestamp.strftime('%Y%m%d%H%M%S')}"
this_results_dir = os.path.join(RESULTS_DIR, "vqa_mistake_detection", this_results_dir)
os.makedirs(this_results_dir)

metrics_filename = f"metrics_{args.mistake_detection_strategy}_{args.eval_split}.json"
json.dump(metrics, open(os.path.join(this_results_dir, metrics_filename), "w"), indent=4)

preds_filename = f"preds_{args.eval_split}.json"
json.dump(preds, open(os.path.join(this_results_dir, preds_filename), "w"), indent=4)

det_filename = f"det_{args.mistake_detection_strategy}_{args.eval_split}.pdf"
generate_det_curve(metrics, os.path.join(this_results_dir, det_filename))

shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)