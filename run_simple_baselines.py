# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
from collections import Counter
import datetime
import json
import os
from pprint import pprint
from random import randint
import shutil
from tqdm import tqdm

from travel.constants import RESULTS_DIR
from travel.model.mistake_detection import mistake_detection_metrics
from travel.data.mistake_detection import MistakeDetectionTasks
from travel.data.captaincook4d import CaptainCook4DDataset

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="captaincook4d", choices=[task.value for task in MistakeDetectionTasks], help="Target mistake detection task.")
parser.add_argument("--eval_partitions", nargs='+', type=str, default=["val", "test"])
args = parser.parse_args()

for eval_partition in args.eval_partitions:
    # Load mistake detection dataset
    if args.task == "captaincook4d":
        eval_dataset = CaptainCook4DDataset(data_split=eval_partition)
    # TODO: integrate ego4d here
    else:
        raise NotImplementedError(f"Haven't implemented usage of {args.task} dataset yet!")                                        
    labels = [example.mistake for example in eval_dataset]
    majority_class = Counter(labels).most_common()[0][0]

    # Run simple baseline inference
    random_preds = []
    majority_preds = []
    for example in tqdm(eval_dataset, "running inference on clips"):
        random_preds.append(bool(randint(0,1)))
        majority_preds.append(majority_class)

    metrics_random = mistake_detection_metrics(labels, random_preds)
    metrics_majority = mistake_detection_metrics(labels, majority_preds)

    print("Mistake Detection Metrics (random):")
    pprint(metrics_random)

    print("\nMistake Detection Metrics (majority)")
    pprint(metrics_majority)

    # Save metrics, preds, DET curve, config file (which may have some parameters that vary over time), and command-line arguments
    timestamp = datetime.datetime.now()
    this_results_dir = f"simple_baselines"
    this_results_dir += f"_{timestamp.strftime('%Y%m%d%H%M%S')}"
    this_results_dir = os.path.join(RESULTS_DIR, "vqa_mistake_detection", this_results_dir)
    os.makedirs(this_results_dir)

    metrics_filename = f"metrics_random_{args.eval_split}.json"
    json.dump(metrics_random, open(os.path.join(this_results_dir, metrics_filename), "w"), indent=4)

    metrics_filename = f"metrics_majority_{args.eval_split}.json"
    json.dump(metrics_majority, open(os.path.join(this_results_dir, metrics_filename), "w"), indent=4)

shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)