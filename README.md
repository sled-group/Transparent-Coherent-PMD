# Coherent Reasoning for Procedural Mistake Detection

This is the code and data release for the paper "[Coherent Reasoning for Procedural Mistake Detection](https://arxiv.org/abs/2412.11927)," originally code-named `TRAVEl` (Tiered Reasoning for Action-Video Equivalence) during development.

## Setup

Clone the repo:

```
git clone git@github.com:shanestorks/TRAVEl.git
```

Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) if needed, and ensure a CUDA installation is available which is compatible with both `bitsandbytes` and `torch`. 

In some environments, you may have issues getting Poetry to use the correct Python interpreter (also during installation of Poetry itself). You can consider creating a `conda` environment before installing Poetry and dependencies for this project:

```
module load python3.10-anaconda/2023.03
conda create --name python310 python=3.10.9
conda activate python310
```

Before running any other `poetry` commands, run the following:

```
poetry env use ~/.conda/envs/python310/bin/python
```

Then set up the virtual environment:

```
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install
poetry shell
```

Note that the same CUDA installation must be available whenever you activate the environment using `poetry shell` or `poetry run`.

### Jupyter Notebook Support

For Jupyter notebook support, run this when the `travel` environment is activated (i.e., after running `poetry shell`):

```
python -m ipykernel install --user --name=travel
```

Also ensure that the same CUDA installation is available in your notebook environment; for example, in a Great Lakes JupyterLab session, you can add `load cuda/12.1.1` under "Module commands". Then use the `travel` kernel in Jupyter.

### Slurm Script Support

In a Slurm script, ensure you prepend commands with `poetry run` to activate the virtual environment.

## Preparing Ego4D-PMD Data

TODO: add explanation for generating ego4d data. Also add a note that CaptainCook4D support is available but not maintained, since we didn't use it in the final paper

## Running Experiments

### Configuration

Make a copy of `sample_config.yml` and name it `config.yml`. Configure the arguments in `config.yml` as needed (especially the Hugging Face token `hf_token` and cache directories). If you ever need multiple config files in the same environment (e.g., to run multiple experiments with different settings at the same time), you can set the environment variable `TRAVEl_config_path` to an appropriate specialized `config_xxx.yml` file before running a script. Keep the random seed as 222 to replicate results presented in the paper.

TODO: need to overhaul all remaining commands

### Iterative VQA

The below commands can reproduce the iterative VQA results presented in Shane's dissertation. These are example commands for LLaVA evaluated on the validation data subset. You can configure the VLM type using the choices specified for `--vlm_name` in `run_vqa_iterative.py`. You can evaluate on the testing data subset by adding `--eval_partition test` and changing `--debug_n_examples` to `1000`.

#### Base Job Script

All commands are run on the Great Lakes Slurm cluster using srun. A job script typically starts like this, followed by one of the later commands:

```
#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name vqa_mistake_detection
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10g
#SBATCH --time=24:00:00
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --mail-type=BEGIN,END,FAIL

cd ~/path/to/TRAVEl
bash prepare_great_lakes.sh
```

#### Likelihood-Based Ranking

Without in-context learning:

```
srun --cpus-per-task 4 poetry run python run_vqa_iterative.py --run_id "$timestamp" --max_iterations 10 --question_selection_strategy likelihood --generation_batch_size 20 --vqa_batch_size 20 --debug --debug_n_examples 250 --coherence_evaluation_strategy nli --exclude_history_from_vqa --vlm_name "llava-hf/llava-1.5-7b-hf"
```

With in-context learning:

```
srun --cpus-per-task 4 poetry run python run_vqa_iterative.py --run_id "$timestamp" --max_iterations 10 --question_selection_strategy likelihood --generation_batch_size 20 --vqa_batch_size 20 --debug --debug_n_examples 250 --n_icl_demonstrations 20 --coherence_evaluation_strategy nli --exclude_history_from_vqa --vlm_name "llava-hf/llava-1.5-7b-hf"
```

#### Coherence-Based Ranking

Without in-context learning:

```
srun --cpus-per-task 4 poetry run python run_vqa_iterative.py --run_id "$timestamp" --max_iterations 10 --question_selection_strategy coherence --generation_batch_size 20 --vqa_batch_size 20 --debug --debug_n_examples 250 --coherence_evaluation_strategy nli --get_negated_success_probs --exclude_history_from_vqa --vlm_name "llava-hf/llava-1.5-7b-hf"
```

With in-context learning:

```
srun --cpus-per-task 4 poetry run python run_vqa_iterative.py --run_id "$timestamp" --max_iterations 10 --question_selection_strategy coherence --generation_batch_size 20 --vqa_batch_size 20 --debug --debug_n_examples 250 --n_icl_demonstrations 20 --coherence_evaluation_strategy nli --get_negated_success_probs --exclude_history_from_vqa --vlm_name "llava-hf/llava-1.5-7b-hf"
```

#### Visual Strategies

To add visual strategies, add one of the following groups of arguments to your command from above.

Contrastive Region Guidance (CRG):

```
--visual_filter_mode contrastive_region --visual_filter_strength 1.0
```

Visual Contrastive Decoding (VCD):

```
--visual_filter_mode visual_contrastive --visual_filter_strength 1.0
```

Assembling Global and Local Attention (AGLA):

```
--visual_filter_mode agla --visual_filter_strength 2.0
```

Proposed "spatial" filter (using Gaussian blur):

```
--visual_filter_mode spatial_blur --visual_filter_strength 55.0
```

### Analysis of Results

There are several analysis scripts and notebooks which must be manually configured to point to results you want to analyze:

1. `graph_classification_curves.py` graphs DET and ROC curves.
2. `analyze_IterativeVQA.py` runs several kinds of analysis on the iterative VQA results.
3. `notebooks/generate_overview_graphs.ipynb` can be used to generate 3D graphs of error and coherence metrics.
4. `notebooks/visualize_results_IterativeVQA.ipynb` can be used to extract sample mistake detection outputs from the iterative VQA.

# Citation

TODO: add citation