# This script manually removes any duplicates (by example ID) from an Ego4D dataset.json file - it does not remove the actual source data which is duplicated.
# Some duplicates were observed across different "partitions" of the training data generated in parallel, so the purpose of this script is to remove those.
# Duplicates weren't found in any other partitions of data.
from travel import init_travel
init_travel()

import json
import os
from collections import defaultdict
from tqdm import tqdm

DATASET_PATH = "/path/to/ego4d_train_seed222_mismatch/dataset.json"
d = json.load(open(DATASET_PATH, "r"))
seen_so_far = []
seen_so_far_ids = []

print("Looking for duplicates...")
duplicate_ids = []
for dd in tqdm(d['example_dirs']):
    if dd not in seen_so_far:
        seen_so_far.append(dd)
    else:
        seen_so_far.append(dd)
        # print(f"Warning: Directory {dd} seen multiple times.")

    example_id = "/".join(dd.split("/")[-3:])
    if example_id not in seen_so_far_ids:
        seen_so_far_ids.append(example_id)
    else:
        seen_so_far_ids.append(example_id)
        # print(f"Warning: ID {example_id} seen multiple times.")
        duplicate_ids.append(example_id)

print("Removing duplicates...")
if len(duplicate_ids) > 0:

    duplicates = defaultdict(list)
    for dd, eid in zip(seen_so_far, seen_so_far_ids):
        if eid in duplicate_ids:
            duplicates[eid].append(dd)

    new_example_dirs = d['example_dirs']
    for eid in duplicates:
        print("Duplicate:", eid)
        for dd in duplicates[eid]:
            print(dd)

        delete_dup = duplicates[eid][1:]

        # Delete all but the first instance of the duplicated directory
        for dd in delete_dup:
            new_example_dirs = [ddd for ddd in new_example_dirs if ddd != dd]

        print("\n")

    d['example_dirs'] = new_example_dirs
    json.dump(d, open(DATASET_PATH, "w"), indent=4)
    print("Removed duplicates from", DATASET_PATH)
else:
    print("No duplicates found!")


