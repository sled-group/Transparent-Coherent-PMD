# Explainable Procedural Mistake Detection

This is the code and data release for the paper "[Explainable Procedural Mistake Detection](https://arxiv.org/abs/2412.11927)," originally code-named `TRAVEl` (Tiered Reasoning for Action-Video Equivalence) during development.

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
python scripts/data/run_generate_ego4d.py --partition train --mismatch_augmentation
python scripts/data/run_generate_ego4d.py --partition val --mismatch_augmentation
python scripts/data/run_generate_ego4d.py --partition test --mismatch_augmentation
```

When generating the training data, a small number of duplicate records may be generated. Configure `scripts/utils/remove_duplicates_in_ego4d.py` then run it to remove them.

This can take several CPU days. We recommend running these commands as Slurm jobs using Slurm's `srun` command to enable data parallelism. Note that progress is intermittently saved.

Then generate our randomly sampled subsets (this is much quicker):

```
python scripts/data/run_generate_ego4d.py --partition train --mismatch_augmentation --debug --debug_n_examples 5000
python scripts/data/run_generate_ego4d.py --partition val --mismatch_augmentation --debug --debug_n_examples 250
python scripts/data/run_generate_ego4d.py --partition test --mismatch_augmentation --debug --debug_n_examples 1000
```

## Running Experiments

### Open-Source VLM Inference and Evaluation

Use the `scripts/run_vqa_iterative.py` script to run inference with VLMs as shown below. This script supports data parallelism with Slurm's `srun` command.

```
python scripts/run_vqa_iterative.py --run_id "<unique ID>" --max_iterations 10 --question_selection_strategy <coherence|likelihood> --exclude_history_from_vqa \
     --eval_partition val --debug --debug_n_examples 250 \
     --vlm_name <vlm_name_or_path> --hf_hub_revision <revision_id>
```

Several notes about configuring the command:

* For the validation set, use the above `--eval_partition` and `--debug_n_examples` arguments. For testing, use `--eval_partition test` and `--debug_n_examples 1000`, and additionally provide the arguments `--early_stop_delta` and `--confident_range` as tuned in `stopping_criteria_tuning/tuned_stopping_criteria.json` in the output directory from the corresponding validation run. For inference on the training data (i.e., to generate training data for coherence-based fine-tuning), use `eval_partition train` and `--debug_n_examples 5000`.

* To evaluate the VLMs studied in the paper, configure `--vlm_name` and `--hf_hub_revision` as follows:
     * InstructBLIP: `--vlm_name "Salesforce/instructblip-vicuna-7b" --hf_hub_revision "52ba0cb2c44d96b2fcceed4e84141dc40d2b6a92"`
     * LLaVA 1.5: `--vlm_name "llava-hf/llava-1.5-7b-hf" --hf_hub_revision "12e054b30e8e061f423c7264bc97d4248232e965"`
     * Llama 3: `--vlm_name "meta-llama/Llama-3.2-11B-Vision-Instruct" --hf_hub_revision "cee5b78e6faed15d5f2e6d8a654fd5b247c0d5ca"`

* Configure `--question_selection_strategy` for `likelihood`- or `coherence`-based question selection.

* For experiments with in-context learning, add `--n_icl_demonstrations 20`.

* For experiments utilizing a trained VQG adapter, add `vqg_adapter_path <path/to/vqg/adapter/output/director>`.

* For experiments with length penalty, add --length_penalty "-1.0".

* Note that any `batch_size` args for the script can be maximized for your GPU memory; the default arguments were configured for NVIDIA A40 GPUs (48GB VRAM).

If your environment uses Slurm, feel free to use `scripts/slurm/submit_iterativevqa_slurm_script.py` to submit jobs (or create a Bash script to easily submit several jobs). 

After completing, you can view `metrics_table_<partition>.json`, which includes the metrics in the paper's result's tables. Please note that for testing, you should not use the `accuracy` metric there, and instead identify the accuracy value in `metrics_accuracy_<partition>.json` for the mistake confidence threshold selected on the validation data. You can also see a 3D scatter plot of evaluation metrics at `3d_graph_base.pdf`.

### Coherence-Based VLM Fine-Tuning

Before running training, you'll need to run the above inference script on the training and validation data, and note the output directories for each run. To reproduce the main results in the paper, use arguments `--question_selection_strategy coherence` and `--n_icl_demonstrations 20`. To reproduce the additional fine-tuning results in the appendices, remove `--n_icl_demonstrations 20` for no in-context learning, or add `--length_penalty "-1.0"` to include a length penalty.

Use the `scripts/train_vqa_iterative_dpo.py` script to run coherence-based fine-tuning. We use `srun` and `torchrun` to enable GPU parallelism, but the script should also run on a single GPU.

```
srun --cpus-per-task 4 poetry run torchrun --nnodes=4 --nproc_per_node=1 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint="$HOST_NODE:25703" scripts/train_vqa_iterative_dpo.py \
     --train_data_path "/path/to/training/outputs/outputs_train.json" \
     --val_data_path "/path/to/val/outputs/outputs_val.json" \
     --run_id "<unique ID>" --n_epochs 10 --learning_rate <lr> --dpo_beta <beta>
```

Batch size arguments can again be maximized for your environment. The script `scripts/slurm/submit_dpo_slurm_scripts.py` can be used to initiate hyperparameter tuning through Slurm.

### DET Curves

To generate DET curves and other various metrics for multiple compared results, configure `analysis_config.yml` to point to your desired configurations of output directories, then run `scripts/analyze_IterativeVQA.py`.

## Citation

```
@misc{storks2024explainableproceduralmistakedetection,
      title={Explainable Procedural Mistake Detection}, 
      author={Shane Storks and Itamar Bar-Yossef and Yayuan Li and Zheyuan Zhang and Jason J. Corso and Joyce Chai},
      year={2024},
      eprint={2412.11927},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.11927}, 
}
```
