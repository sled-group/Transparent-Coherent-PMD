import argparse
import datetime
import json
import os
from pprint import pprint
import shutil
import torch
from tqdm import tqdm

from travel.constants import DATA_CACHE_DIR, MODEL_CACHE_DIR, RESULTS_DIR
from travel.model.mistake_detection import MISTAKE_DETECTION_STRATEGIES, HEURISTIC_TARGET_FRAMES_PROPORTION, generate_det_curve
from travel.model.vqa import VQAOutputs, VQAResponse, SUCCESSVQA_PROMPT_TEMPLATES, get_vqa_response_token_ids
from travel.data.mistake_detection import MistakeDetectionTasks, get_cutoff_time_by_proportion
from travel.data.captaincook4d import CaptainCook4DDataset

os.environ['HF_HOME'] = MODEL_CACHE_DIR
from transformers import AutoProcessor, AutoModelForVision2Seq

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="captaincook4d", choices=[task.value for task in MistakeDetectionTasks])
parser.add_argument("--eval_partition", type=str, choices=["val", "test"])
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=list(SUCCESSVQA_PROMPT_TEMPLATES.keys()), help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--mistake_detection_strategy", type=str, default="heuristic", choices=list(MISTAKE_DETECTION_STRATEGIES.keys()))
args = parser.parse_args()

# Load mistake detection dataset
eval_dataset = CaptainCook4DDataset(data_split=args.eval_partition,
                                    debug_n_examples_per_class=20)

# Some mistake detection strategies are only applied to a specific proportion of frames; if so, we can skip running inference on these frames to save time
if args.mistake_detection_strategy == "heuristic":
    target_frames_proportion = HEURISTIC_TARGET_FRAMES_PROPORTION
else:
    target_frames_proportion = None

# Load VLM
vlm_processor = AutoProcessor.from_pretrained(args.vlm_name)
vlm = AutoModelForVision2Seq.from_pretrained(args.vlm_name, cache_dir=DATA_CACHE_DIR, load_in_8bit=True) # NOTE: when loading in 8bit, batched inference may output nans
vlm_processor.tokenizer.padding_side = "left"

prompt_template = SUCCESSVQA_PROMPT_TEMPLATES[args.vlm_name]
response_token_ids = get_vqa_response_token_ids(vlm_processor.tokenizer)

# TODO: perform inference in batches?
# TODO: cache VQA outputs for models?
vqa_outputs = []
for example in tqdm(eval_dataset, "running inference on clips"):
    this_vqa_outputs = []
    
    step_id = example.procedure_id
    step = example.procedure_description
    
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
            inputs = vlm_processor(text=prompt, images=frame, return_tensors="pt").to(vlm.device)
            logits = vlm(**inputs).logits[0] # (seq length, vocab size)
            logits = logits[-1].detach().cpu() # (vocab size)

            this_vqa_outputs.append(
                [VQAOutputs(
                    step_id,
                    frame,
                    prompt,
                    expected_answer,
                    response_token_ids,
                    logits,        
                )]
            )        
        
    vqa_outputs.append(this_vqa_outputs)

# TODO: improve heuristic evaluator to be softer and incorporate answer probabilities
# TODO: save VQAOutputs - what is the best way to do this?
evaluator = MISTAKE_DETECTION_STRATEGIES[args.mistake_detection_strategy](eval_dataset.examples, vqa_outputs)
metrics = evaluator.get_mistake_detection_metrics()
print("Mistake Detection Metrics (Detection Threshold=0.5):")
print(metrics.keys())
pprint(metrics[0.5])

# Save results, config file (which may have some parameters that vary over time), and command-line arguments
timestamp = datetime.datetime.now()
this_results_dir = f"SuccessVQA_{args.vlm_name.split('/')[-1]}_{timestamp.strftime('%Y%m%d%H%M%S')}"
this_results_dir = os.path.join(RESULTS_DIR, this_results_dir)
os.makedirs(this_results_dir)

metrics_filename = f"metrics_{args.mistake_detection_strategy}_{args.eval_partition}.json"
json.dump(metrics, open(os.path.join(this_results_dir, metrics_filename), "w"))

# TODO: improve appearance of DET curve and set consistent gridlines/axes ticks
det_filename = f"det_{args.mistake_detection_strategy}_{args.eval_partition}.pdf"
generate_det_curve(metrics, os.path.join(this_results_dir, det_filename))

shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
with open(os.path.join(this_results_dir, "args.json"), 'w') as f:
    json.dump(args.__dict__, f, indent=2)