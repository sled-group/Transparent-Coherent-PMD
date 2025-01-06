import subprocess
import itertools
import os
from datetime import datetime
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Submit Slurm jobs for fine-tuning a language model with different --dpo_beta and --learning_rate combinations.")
parser.add_argument("--account_name", type=str, help="Slurm billing account name.", default="chaijy2")
parser.add_argument("--eval_partition", type=str, default="val", choices=["train", "val", "test"])
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=["Salesforce/instructblip-vicuna-7b", "llava-hf/llava-1.5-7b-hf"], help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--hf_hub_revision", type=str, default=None, help="Optional revision ID for VLM in Hugging Face Hub.")
parser.add_argument("--vqg_adapter_path", type=str, help="Name or path to adapter of VLM's LM to be used for VQG. This is for fine-tuned VQG models. Adapter base model should match the model used by the VLM specified in `vlm_name`.")
parser.add_argument("--question_selection_strategy", type=str, default="likelihood", choices=["likelihood", "relevance", "informativeness", "coherence"], help="Strategy to use to choose question to generate from beam search candidates.")
parser.add_argument("--n_icl_demonstrations", type=int, default=0, choices=list(range(21)), help="Pass this argument to generate an extra pool of candidate questions using n in-context VQG examples (doesn't incorporate answers to previous questions).")
parser.add_argument("--early_stop_delta", nargs='+', type=float, default=[0.05, 0.1, 0.2, 0.4], help="List of early_stop_delta values to consider, separated by spaces. If success probability changes less than this over 3 turns, stop generating questions.")
parser.add_argument("--confident_range", nargs='+', type=float, default=[0.025, 0.05, 0.1, 0.2], help="List of confident_range values to consider, separated by spaces. If success probability is within this from 0.0 or 1.0, stop early due to high confidence.")
parser.add_argument("--length_penalty", type=float, default=1.0, help="Exponential length penalty for generation (> 0.0 promotes long sequences, < 0.0 promotes short sequences).")
parser.add_argument("--restrict_q_words", action="store_true", help="Pass this argument to restrict first words of generated questions to 'is', 'are', 'do', and 'does'.")
parser.add_argument("--timestamp", type=str, default=None, help="This argument is a string that will replace the timestamp string used to identify results.")
parser.add_argument("--timestamp_suffix", type=str, default=None, help="This argument will be concatenated to the subdirectory where DPO results are saved.")

# Parse arguments
args = parser.parse_args()

# Create the directory to store Slurm output files if it doesn't exist
log_dir = "slurm_logs"
os.makedirs(log_dir, exist_ok=True)

# Create the directory to store Slurm scripts if it doesn't exist
output_dir = "slurm_scripts"
os.makedirs(output_dir, exist_ok=True)

# Generate a common timestamp for all jobs
if args.timestamp is None:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
else:
    timestamp = args.timestamp

# Template for the Slurm script
slurm_script_template = """#!/bin/bash
#SBATCH --job-name vqa_mistake_detection_iterativevqa
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10g
#SBATCH --time=48:00:00
#SBATCH --account={account_name}
#SBATCH --partition=spgpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=~/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/slurm_logs/iterativevqa-%j.out

cd ~/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl
bash prepare_great_lakes.sh

RDZV_ID=$RANDOM
export HOST_NODE=$(srun --nodes=1 --ntasks=1 hostname)

nvidia-smi
lspci | grep -i nvidia

srun --cpus-per-task 4 poetry run python run_vqa_iterative.py --run_id "{run_id}" --max_iterations 10 --question_selection_strategy {question_selection_strategy} --generation_batch_size 20 --exclude_history_from_vqa --vqa_batch_size 20 --n_icl_demonstrations {n_icl_demonstrations} \
     --debug --debug_n_examples {debug_n_examples} --eval_partition {eval_partition} \
     --vlm_name "{vlm_name}" \
     --max_iterations 10 --early_stop_delta {early_stop_delta} --confident_range {confident_range} --length_penalty "{length_penalty}"
"""

# Define a unique run_id for this job
if args.timestamp_suffix is None:
    run_id = f"{timestamp}"
else:
    run_id = f"{timestamp}_{args.timestamp_suffix}"

# Fill in the template with specific values for this job
slurm_script = slurm_script_template.format(account_name=args.account_name,
                                            run_id=run_id, 
                                            question_selection_strategy=args.question_selection_strategy,
                                            n_icl_demonstrations=args.n_icl_demonstrations,
                                            debug_n_examples=args.debug_n_examples,
                                            eval_partition=args.eval_partition,
                                            vlm_name=args.vlm_name,
                                            early_stop_delta=" ".join([str(esd) for esd in args.early_stop_delta]),
                                            confident_range=" ".join([str(cr) for cr in args.confident_range]),
                                            length_penalty=args.length_penalty)

if args.hf_hub_revision is not None:
    slurm_script += f' --hf_hub_revision "{args.hf_hub_revision}"'
if args.vqg_adapter_path is not None:
    slurm_script += f' --vqg_adapter_path "{args.vqg_adapter_path}"'
if args.restrict_q_words:
    slurm_script += " --restrict_q_words"

# Write the script to the output directory
script_filename = os.path.join(output_dir, f"slurm_job_{run_id.replace('/','_')}.sh")
with open(script_filename, "w") as f:
    f.write(slurm_script)

# Submit the job using sbatch
try:
    subprocess.run(["sbatch", script_filename], check=True)
    print(f"Submitted job!")
except subprocess.CalledProcessError as e:
    print(f"Failed to submit job. Error: {e}")
