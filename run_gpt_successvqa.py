from travel import init_travel
init_travel()

import argparse
import datetime
import json
import os
from PIL import Image
from pprint import pprint
import shutil

from travel.constants import RESULTS_DIR
from travel.data.captaincook4d import CaptainCook4DDataset
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.mistake_detection import MistakeDetectionTasks, MistakeDetectionExample
from travel.data.vqa import VQAResponse, SUCCESSVQA_PROMPT_TEMPLATES
from travel.model.mistake_detection import MISTAKE_DETECTION_STRATEGIES, generate_det_curve, compile_mistake_detection_preds, NLI_RERUN_ON_RELEVANT_EVIDENCE
from travel.model.api  import GPT

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ego4d", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--eval_partitions", nargs='+', type=str, default=["val", "test"])
parser.add_argument("--mistake_detection_strategy", type=str, default="heuristic", choices=list(MISTAKE_DETECTION_STRATEGIES.keys()))
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of generating frameVQA examples.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
parser.add_argument("--api_key", type=str, required=True, help="API key to send a request to GPT.")
parser.add_argument("--endpoint", type=str, required=False, help="Endpoint URL to send a request to OpenAI.")
args = parser.parse_args()


if not args.endpoint or not args.api_key:
    raise ValueError("For GPT4 usage, you need to pass your endpoint URL and API key.")

vlm = GPT(api_key=args.api_key,
          endpoint=args.endpoint)
prompt_template = SUCCESSVQA_PROMPT_TEMPLATES['GPT']

# Configure results directory
if args.resume_dir is None:
    timestamp = datetime.datetime.now()
    vlm_name = 'GPT-4'
    task_name = args.task
    if args.debug:
        task_name += f"_debug{args.debug_n_examples}" if args.task != "captaincook4d" else "_debug"
    this_results_dir = os.path.join(task_name, vlm_name, f"SuccessVQA_{task_name}")
    this_results_dir += f"_{vlm_name}"
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
        eval_datasets = [CaptainCook4DDataset(data_split=eval_partition, debug_n_examples_per_class=args.debug_n_examples if args.debug else None)]
    elif MistakeDetectionTasks(args.task) == MistakeDetectionTasks.Ego4D:
        eval_datasets = [Ego4DMistakeDetectionDataset(data_split=eval_partition, 
                                                      mismatch_augmentation=True,
                                                      multi_frame=True,
                                                      debug_n_examples_per_class=args.debug_n_examples if args.debug else None)]
    elif MistakeDetectionTasks(args.task) == MistakeDetectionTasks.Ego4D_Single:
        eval_datasets = [Ego4DMistakeDetectionDataset(data_split=eval_partition, 
                                                      mismatch_augmentation=True,
                                                      multi_frame=False,
                                                      debug_n_examples_per_class=args.debug_n_examples if args.debug else None)]
    else:
        raise NotImplementedError(f"Haven't implemented usage of {args.task} dataset yet!")                                        
        
    print("Running VQA sequentially...")
    vqa_outputs = vlm.run_vqa(eval_dataset=eval_datasets[0],
                              generate_prompts=generate_prompts,
                              cache_dir=this_results_dir)
    
    print("Evaluating and saving results...")
    
    evaluator = MISTAKE_DETECTION_STRATEGIES[args.mistake_detection_strategy](eval_datasets[0], vqa_outputs)
    mistake_detection_preds, metrics = evaluator.evaluate_mistake_detection()
    print(f"Mistake Detection Metrics ({eval_partition}, Detection Threshold={metrics['best_threshold']}):")
    pprint(metrics[metrics['best_threshold']])

    # Compile preds per mistake detection example
    preds = compile_mistake_detection_preds(eval_datasets[0], vqa_outputs, mistake_detection_preds, image_base_path=this_results_dir)

    # Save metrics, preds, DET curve, config file (which may have some parameters that vary over time), and command-line arguments
    metrics_filename = f"metrics_{args.mistake_detection_strategy}{'rerun' if args.mistake_detection_strategy == 'nli' and NLI_RERUN_ON_RELEVANT_EVIDENCE else ''}_{eval_partition}.json"
    json.dump(metrics, open(os.path.join(this_results_dir, metrics_filename), "w"), indent=4)

    preds_filename = f"preds_{args.mistake_detection_strategy}{'rerun' if args.mistake_detection_strategy == 'nli' and NLI_RERUN_ON_RELEVANT_EVIDENCE else ''}_{eval_partition}.json"
    json.dump(preds, open(os.path.join(this_results_dir, preds_filename), "w"), indent=4)

    det_filename = f"det_{args.mistake_detection_strategy}{'rerun' if args.mistake_detection_strategy == 'nli' and NLI_RERUN_ON_RELEVANT_EVIDENCE else ''}_{eval_partition}.pdf"
    generate_det_curve(metrics, os.path.join(this_results_dir, det_filename))

shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)