# Need this call at the beginning of every script to set random seeds and set the HF cache
import argparse
import concurrent.futures
import requests

from travel import init_travel
init_travel()

from travel.data.ego4d import Ego4DMistakeDetectionDataset, combine_ego4d_partitions

parser = argparse.ArgumentParser()
parser.add_argument("--partition", type=str, default="train", help="Dataset partition name to generate from.")
parser.add_argument("--n_workers", type=int, default=1, help="Number of workers to parallelize dataset generation.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
args = parser.parse_args()

# TODO: Lower memory consumption for MistakeDetectionDataset - list of dirs can be a dataframe?

def generate_ego4d_partition(n_workers: int,
                             worker_index: int,
                             debug: bool,) -> Ego4DMistakeDetectionDataset:
    dataset_partition = Ego4DMistakeDetectionDataset(data_split=args.partition,
                                                     mismatch_augmentation=False,
                                                     debug_n_examples_per_class=20 if debug else None,
                                                     n_workers=n_workers,
                                                     worker_index=worker_index)
    return dataset_partition

# Load Ego4D for mistake detection
print("Loading dataset...")
if args.n_workers == 1:
    # Generate in one thread
    dataset = Ego4DMistakeDetectionDataset(data_split=args.partition,
                                           mismatch_augmentation=False,
                                           debug_n_examples_per_class=20 if args.debug else None)
else:
    # Generate in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        partitions = list(executor.map(generate_ego4d_partition, 
                                       [args.n_workers] * args.n_workers, 
                                       list(range(args.n_workers)),
                                       [args.debug] * args.n_workers))
    combined_dataset = combine_ego4d_partitions(partitions,
                                                debug_n_examples_per_class=20 if args.debug else None)
print("Done!")