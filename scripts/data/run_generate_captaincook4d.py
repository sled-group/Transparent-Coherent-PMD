# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
from collections import Counter
from pprint import pprint

from travel.data.captaincook4d import CaptainCook4DDataset

parser = argparse.ArgumentParser()
parser.add_argument("--partition", type=str, default="val", help="Dataset partition name to generate from.")
parser.add_argument("--multi_frame", action="store_true", help="Pass this argument to sample multiple frames rather than a single frame.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=20, help="Configure the number of examples per class to generate for debugging purposes.")
args = parser.parse_args()

print("Generating CaptainCook4D dataset...")
dataset = CaptainCook4DDataset(data_split=args.partition, 
                               multi_frame=args.multi_frame,
                               debug_n_examples_per_class=args.debug_n_examples if args.debug else None) 

print("Done!")

print("\nDistribution of mistake types:")
mistake_types = []
for example in dataset:
    if example.mistake:
        mistake_types.append(example.mistake_type)
mistake_dist = Counter(mistake_types)
pprint(mistake_dist.most_common())

