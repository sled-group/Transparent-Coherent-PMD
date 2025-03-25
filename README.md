# Coherent Reasoning for Procedural Mistake Detection

This is the code and data release for the paper "[Coherent Reasoning for Procedural Mistake Detection](https://arxiv.org/abs/2412.11927)," originally code-named `TRAVEl` (Tiered Reasoning for Action-Video Equivalence) during development.

## Setup

Clone the repo:

```
git clone git@github.com:shanestorks/TRAVEl.git
```

Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) if needed, and ensure a CUDA installation is available which is compatible with both `bitsandbytes` and `torch`. We used CUDA 12.1.1.

If needed, before running any other `poetry` commands, run the following to make sure Poetry has access to an instance of Python 3.10:

```
poetry env use ~/.conda/envs/python310/bin/python
```

Then set up the virtual environment:

```
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install
poetry shell
```

Note that you should run the above `poetry env use` command before activating the environment using `poetry shell` or `poetry run`. Also, the same CUDA installation must be available whenever you activate the environment.

### Jupyter Notebook Support

For Jupyter notebook support, run this when the `travel` environment is activated (i.e., after running `poetry shell`):

```
python -m ipykernel install --user --name=travel
```

Ensure that the same CUDA installation is available in your notebook environment.

## Configuration

Make a copy of `sample_config.yml` and name it `config.yml`. Configure the arguments in `config.yml` as needed (especially the Hugging Face token `hf_token` and cache directories). Note that for the results presented in the paper, we configured `random_seed` to 222.

If you ever need multiple config files in the same environment (e.g., to run multiple experiments with different settings at the same time), you can set the environment variable `TRAVEl_config_path` to an appropriate specialized `config_xxx.yml` file before running a script. Keep the random seed as 222 to replicate results presented in the paper.

## Preparing Ego4D-PMD Data

The below steps cover how to regenerate the Ego4D-PMD data. Note that while this repo contains some code for using [CaptainCook4D](https://github.com/CaptainCook4D), this code is not maintained. 

### Gathering Required Artifacts

Download the Ego4D v2 data from [here](https://visualize.ego4d-data.org/), configuring `data.ego4d.video_path` accordingly.

Next, download the `.parquet` file for mismatching Ego4D examples.

```
mkdir ego4d_mismatch_srl_files
cd ego4d_mismatch_srl_files
wget https://prism.eecs.umich.edu/yayuanli/z_web/dataset/Ego4D_Mistake/misalignsrl_more_samples_same_split_combinedwords_wohand_objectstatesafe.parquet
```

Configure `data.ego4d.misalignsrl_path` to the full path to the downloaded `.parquet` file.

### Generating Data

First, generate the full Ego4D-PMD data:

```
python run_generate_ego4d.py --partition train --mismatch_augmentation
python run_generate_ego4d.py --partition val --mismatch_augmentation
python run_generate_ego4d.py --partition test --mismatch_augmentation
```

This can take several CPU days. We recommend running these commands as Slurm jobs using Slurm's `srun` command. Note that progress is intermittently saved.

Then generate our randomly sampled subsets (this is much quicker):

```
python run_generate_ego4d.py --partition train --mismatch_augmentation --debug --debug_n_examples 5000
python run_generate_ego4d.py --partition val --mismatch_augmentation --debug --debug_n_examples 250
python run_generate_ego4d.py --partition test --mismatch_augmentation --debug --debug_n_examples 1000
```

## Running Experiments

# TODO: overhaul below commands to match paper experiments

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

## Citation

TODO: add citation

