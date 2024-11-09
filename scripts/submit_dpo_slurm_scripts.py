import subprocess
import itertools
import os
from datetime import datetime
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Submit Slurm jobs for fine-tuning a language model with different --dpo_beta and --learning_rate combinations.")
parser.add_argument("--train_data_path", type=str, help="Path to `outputs_<partition>.json` file which will be used to train LM.")
parser.add_argument("--val_data_path", type=str, help="Path to `outputs_<partition>.json` file which will be used to validate LM.")
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", choices=["Salesforce/instructblip-vicuna-7b", "llava-hf/llava-1.5-7b-hf"], help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--dpo_beta", nargs='+', type=float, default=[0.05, 0.1, 0.5],
                    help="List of --dpo_beta values to use, separated by spaces. Default: [0.05, 0.1, 0.5]")
parser.add_argument("--learning_rate", nargs='+', type=float, default=[1e-6, 2.5e-6, 5e-6, 7.5e-6, 1e-5],
                    help="List of --learning_rate values to use, separated by spaces. Default: [1e-6, 2.5e-6, 5e-6, 7.5e-6, 1e-5]")


# Parse arguments
args = parser.parse_args()
dpo_betas = args.dpo_beta
learning_rates = args.learning_rate

# Create the directory to store Slurm scripts if it doesn't exist
output_dir = "dpo_slurm_scripts"
os.makedirs(output_dir, exist_ok=True)

# Generate a common timestamp for all jobs
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

# Template for the Slurm script
slurm_script_template = """#!/bin/bash
#SBATCH --partition=spgpu
#SBATCH --time=72:00:00
#SBATCH --job-name=vqg_training_sft_dpo
#SBATCH --account=chaijy2
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10g
#SBATCH --mail-type=BEGIN,END,FAIL

cd ~/coe-chaijy/sstorks/simulation_informed_pcr4nlu/TRAVEl
bash prepare_great_lakes.sh

RDZV_ID=$RANDOM
export HOST_NODE=$(srun --nodes=1 --ntasks=1 hostname)

nvidia-smi
lspci | grep -i nvidia

# DPO              
srun --cpus-per-task 4 poetry run torchrun --nnodes=4 --nproc_per_node=1 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint="$HOST_NODE:25703" train_vqa_iterative_dpo.py \
     --train_data_path "{train_data_path}" \
     --val_data_path "{val_data_path}" \
     --vlm_name "{vlm_name}" \
     --run_id "{run_id}" --n_epochs 10 --learning_rate {learning_rate} --dpo_beta {dpo_beta}
"""

# Iterate over all combinations of dpo_beta and learning_rate
for dpo_beta, learning_rate in itertools.product(dpo_betas, learning_rates):
    # Define a unique run_id for this job
    run_id = f"{timestamp}_beta{dpo_beta}_lr{learning_rate}"
    
    # Fill in the template with specific values for this job
    slurm_script = slurm_script_template.format(run_id=run_id, 
                                                dpo_beta=dpo_beta, 
                                                learning_rate=learning_rate, 
                                                train_data_path=args.train_data_path,
                                                val_data_path=args.val_data_path,
                                                vlm_name=args.vlm_name)
    
    # Write the script to the output directory
    script_filename = os.path.join(output_dir, f"slurm_job_{run_id}.sh")
    with open(script_filename, "w") as f:
        f.write(slurm_script)
    
    # Submit the job using sbatch
    try:
        subprocess.run(["sbatch", script_filename], check=True)
        print(f"Submitted job with dpo_beta={dpo_beta} and learning_rate={learning_rate}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job with dpo_beta={dpo_beta} and learning_rate={learning_rate}. Error: {e}")
