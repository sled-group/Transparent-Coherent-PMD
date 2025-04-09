import subprocess
import itertools
import os
from datetime import datetime
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Submit Slurm jobs for fine-tuning a language model with different --dpo_beta and --learning_rate combinations.")
parser.add_argument("--account_name", type=str, help="Slurm billing account name.", default="chaijy2")
parser.add_argument("--n_workers", type=int, help="Number of processes to parallelize training across.", default=4)
parser.add_argument("--train_data_path", type=str, help="Path to `outputs_<partition>.json` file which will be used to train LM.")
parser.add_argument("--val_data_path", type=str, help="Path to `outputs_<partition>.json` file which will be used to validate LM.")
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--dpo_beta", nargs='+', type=float, default=[0.05, 0.1, 0.5],
                    help="List of --dpo_beta values to use, separated by spaces. Default: [0.05, 0.1, 0.5]")
parser.add_argument("--learning_rate", nargs='+', type=float, default=[1e-6, 2.5e-6, 5e-6, 7.5e-6, 1e-5],
                    help="List of --learning_rate values to use, separated by spaces. Default: [1e-6, 2.5e-6, 5e-6, 7.5e-6, 1e-5]")
parser.add_argument("--unsure_range", type=float, default=0.1, help="A VQA output will be considered unsure if the probability of yes and no are within this range of 50 percent (exclusive).")
parser.add_argument("--top_half", action='store_true', help="Pass this to select good questions from the top half of scored questions (rather than always using top 1).")
parser.add_argument("--timestamp", type=str, default=None, help="This argument is a string that will replace the timestamp string used to identify results.")
parser.add_argument("--timestamp_suffix", type=str, default=None, help="This argument will be concatenated to the subdirectory where DPO results are saved.")

# Parse arguments
args = parser.parse_args()
dpo_betas = args.dpo_beta
learning_rates = args.learning_rate

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
#SBATCH --partition=spgpu
#SBATCH --time=24:00:00
#SBATCH --job-name=vqg_training_sft_dpo
#SBATCH --account={account_name}
#SBATCH --nodes={n_proc}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nfs/turbo/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl/slurm_logs/dpo-%j.out

cd ~/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl
bash prepare_great_lakes.sh

RDZV_ID=$RANDOM
export HOST_NODE=$(srun --nodes=1 --ntasks=1 hostname)

nvidia-smi
lspci | grep -i nvidia

# DPO              
srun --cpus-per-task 4 poetry run torchrun --nnodes={n_proc} --nproc_per_node=1 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint="$HOST_NODE:25703" train_vqa_iterative_dpo.py \
     --train_data_path "{train_data_path}" \
     --val_data_path "{val_data_path}" \
     --vlm_name "{vlm_name}" \
     --run_id "{run_id}" --n_epochs 10 --learning_rate {learning_rate} --dpo_beta {dpo_beta} --unsure_range {unsure_range}
"""[:-1]

# Iterate over all combinations of dpo_beta and learning_rate
first_job = True
for dpo_beta, learning_rate in itertools.product(dpo_betas, learning_rates):
    # Define a unique run_id for this job
    if args.timestamp_suffix is None:
        run_id = f"{timestamp}/beta{dpo_beta}_lr{learning_rate}"
    else:
        run_id = f"{timestamp}_{args.timestamp_suffix}/beta{dpo_beta}_lr{learning_rate}"
    
    # Fill in the template with specific values for this job
    slurm_script = slurm_script_template.format(account_name=args.account_name,
                                                run_id=run_id, 
                                                dpo_beta=dpo_beta, 
                                                learning_rate=learning_rate, 
                                                train_data_path=args.train_data_path,
                                                val_data_path=args.val_data_path,
                                                vlm_name=args.vlm_name,
                                                unsure_range=args.unsure_range,
                                                n_proc=args.n_workers)
    
    if args.top_half:
        slurm_script += " --top_half"
    
    # Only the first submitted job will be responsible for preprocessing data
    if not first_job:
        slurm_script += " --wait_for_data"
    else:
        first_job = True
    
    # Write the script to the output directory
    script_filename = os.path.join(output_dir, f"slurm_job_dpo_{run_id.replace('/','_')}.sh")
    with open(script_filename, "w") as f:
        f.write(slurm_script)
    
    # Submit the job using sbatch
    try:
        subprocess.run(["sbatch", script_filename], check=True)
        print(f"Submitted job with dpo_beta={dpo_beta} and learning_rate={learning_rate}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job with dpo_beta={dpo_beta} and learning_rate={learning_rate}. Error: {e}")
