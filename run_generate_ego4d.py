# Need this call at the beginning of every script to set random seeds and set the HF cache
from time import sleep
from travel import init_travel
init_travel()

import argparse
import concurrent.futures
import json
import os

from travel.data.ego4d import Ego4DMistakeDetectionDataset, combine_ego4d_partitions

parser = argparse.ArgumentParser()
parser.add_argument("--partition", type=str, default="train", help="Dataset partition name to generate from.")
parser.add_argument("--mismatch_augmentation", action="store_true", help="Pass this argument to generate extra negative examples for training purposes.")
parser.add_argument("--multi_frame", action="store_true", help="Pass this argument to sample multiple frames from 8-second clips in Ego4D rather than single frames.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
args = parser.parse_args()

def generate_ego4d_partition(n_workers: int,
                             worker_index: int,
                             debug: bool,) -> Ego4DMistakeDetectionDataset:
    dataset_partition = Ego4DMistakeDetectionDataset(data_split=args.partition,
                                                     mismatch_augmentation=args.mismatch_augmentation,
                                                     multi_frame=args.multi_frame,
                                                     debug_n_examples_per_class=args.debug_n_examples if debug else None,
                                                     n_workers=n_workers,
                                                     worker_index=worker_index)
    return dataset_partition

# Split up work by srun processes; if SLURM_PROCID is not accessible, just run all the work here
# NOTE: always run with the same number of parallel processes; we don't support changing the number of processes
if "SLURM_PROCID" in os.environ and "SLURM_NPROCS" in os.environ:
    worker_index = int(os.environ["SLURM_PROCID"])
    n_workers = int(os.environ["SLURM_NPROCS"])
else:
    worker_index = 0
    n_workers = 1

# Load Ego4D for mistake detection
print(f"({worker_index}) Loading dataset across {n_workers} workers...")
dataset_segment = Ego4DMistakeDetectionDataset(data_split=args.partition,
                                               mismatch_augmentation=args.mismatch_augmentation,
                                               multi_frame=args.multi_frame,
                                               debug_n_examples_per_class=args.debug_n_examples if args.debug else None,
                                               n_workers=n_workers,
                                               worker_index=worker_index)

# Combine all dataset pieces from workers (only worker 0 does this)
if n_workers > 1 and worker_index == 0:
    print(f"({worker_index}) Starting to gather up all dataset segments from workers.")
    other_dataset_segments = []
    for other_worker_index in range(n_workers):
        if other_worker_index == worker_index:
            continue

        print(f"({worker_index}) Waiting for worker {other_worker_index} to finish generating...")
        while True:
            other_worker_dataset_metadata_fname = os.path.join(dataset_segment.get_cache_dir(args.partition,
                                                                                             mismatch_augmentation=args.mismatch_augmentation,
                                                                                             multi_frame=args.multi_frame,
                                                                                             debug_n_examples_per_class=args.debug_n_examples if args.debug else None,
                                                                                             n_workers=n_workers,
                                                                                             worker_index=other_worker_index),
                                                               "dataset.json")
            if os.path.exists(other_worker_dataset_metadata_fname):
                other_worker_dataset_metadata = json.load(open(other_worker_dataset_metadata_fname, "r"))

                if other_worker_dataset_metadata["data_generated"] == True:
                    other_dataset_segment = Ego4DMistakeDetectionDataset(data_split=args.partition,
                                                                         mismatch_augmentation=args.mismatch_augmentation,
                                                                         multi_frame=args.multi_frame,
                                                                         debug_n_examples_per_class=args.debug_n_examples if args.debug else None,
                                                                         n_workers=n_workers,
                                                                         worker_index=other_worker_index)
                    other_dataset_segments.append(other_dataset_segment)
                    print(f"({worker_index}) Collected dataset segment from worker {other_worker_index}.")
                    break
            sleep(10)
            print(f"({worker_index}) Still waiting...")

    combined_dataset = combine_ego4d_partitions([dataset_segment] + other_dataset_segments,
                                                mismatch_augmentation=args.mismatch_augmentation,
                                                multi_frame=args.multi_frame,
                                                debug_n_examples_per_class=args.debug_n_examples if args.debug else None)
print(f"({worker_index}) Done!")