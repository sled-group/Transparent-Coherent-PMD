# Need this call at the beginning of every script to set random seeds and set the HF cache
import argparse

from travel import init_travel
init_travel()

from travel.data.ego4d import Ego4DMistakeDetectionDataset

parser = argparse.ArgumentParser()
parser.add_argument("--partition", type=str, default="train", help="Dataset partition name to generate from.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
args = parser.parse_args()

# TODO: can we parallelize the dataset generation?

# Load Ego4D for mistake detection
print("Loading dataset...")
dataset = Ego4DMistakeDetectionDataset(data_split=args.partition,
                                       mismatch_augmentation=False,
                                       debug_n_examples_per_class=20 if args.debug else None) # TODO: this doesn't work yet - can't seem to retrieve anything
print("Done!")