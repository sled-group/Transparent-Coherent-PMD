# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import os

from travel.data.vqg_learning import save_vqg_training_examples, load_vqg_training_examples

TARGET_DIR = "/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/saved_results_222/vqg_learning/VQG_data_Llama-2-7b-hf_icl20_0.0_0.5_1.0_1.0/llava-1.5-7b-hf/VQA_data_llava-1.5-7b-hf_spatial1.0_20240706025209"
PARTITION = "train"

all_examples = []
for d in os.listdir(os.path.join(TARGET_DIR, f"VQA_scoring_cache_{PARTITION}")):
    if os.path.isdir(os.path.join(TARGET_DIR, f"VQA_scoring_cache_{PARTITION}", d)):
        this_examples = load_vqg_training_examples(os.path.join(TARGET_DIR, f"VQA_scoring_cache_{PARTITION}", d), PARTITION)
        all_examples += this_examples
        print(f"Got examples from {d}.")

save_vqg_training_examples(all_examples, TARGET_DIR, PARTITION)
print("Saved combined examples!")
        
    