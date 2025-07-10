# Transparent and Coherent Procedural Mistake Detection

This is the code and data release for the paper "[Transparent and Coherent Procedural Mistake Detection](https://arxiv.org/abs/2412.11927)."

## Setup

Clone the repo:

```
git clone git@github.com:shanestorks/TRAVEl.git
```

This project uses a Poetry virtual environment. This project was originally code-named `TRAVEl` (Tiered Reasoning for Action-Video Equivalence) during development, thus the virtual environment is named `travel`.

[Install Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) if needed, and ensure a CUDA installation is available which is compatible with both `bitsandbytes` and `torch`. We used CUDA 12.1.1.

If needed, before running any other `poetry` commands, run a command like the following to make sure Poetry has access to an instance of Python 3.10:

```bash
poetry env use ~/.conda/envs/python310/bin/python
```

Then set up the virtual environment:

```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install
poetry shell
```

Note that you should run the above `poetry env use` command before activating the environment using `poetry shell` or `poetry run`. Also, the same CUDA installation must be available whenever you activate the environment.

### Jupyter Notebook Support

For Jupyter notebook support, run this when the `travel` environment is activated (i.e., after running `poetry shell`):

```bash
python -m ipykernel install --user --name=travel
```

Ensure that the same CUDA installation is available in your notebook environment.

## Configuration

Make a copy of `sample_config.yml` and name it `config.yml`. Configure the arguments in `config.yml` as needed (especially the Hugging Face token `hf_token` and cache directories). Note that for the results presented in the paper, we configured `random_seed` to **222**.

If you ever need multiple config files in the same environment (e.g., to run multiple experiments with different settings at the same time), you can set the environment variable `TRAVEl_config_path` to an appropriate specialized `config_xxx.yml` file before running a script. Keep the random seed as 222 to replicate results presented in the paper.

## Preparing Ego4D-PMD Data

The below steps cover how to regenerate the Ego4D-PMD data. Note that while this repo contains some code for using [CaptainCook4D](https://github.com/CaptainCook4D) that we thought might be useful to share, it is not maintained and may not work off the shelf.

### Gathering Required Artifacts

Download the Ego4D v2 data from [here](https://visualize.ego4d-data.org/), configuring `data.ego4d.video_path` accordingly.

Next, download the `.parquet` file for mismatching Ego4D examples.

```bash
mkdir ego4d_mismatch_srl_files
cd ego4d_mismatch_srl_files
wget https://prism.eecs.umich.edu/yayuanli/z_web/dataset/Ego4D_Mistake/misalignsrl_more_samples_same_split_combinedwords_wohand_objectstatesafe.parquet
```

Configure `data.ego4d.misalignsrl_path` in `config.yml` to the full path to the downloaded `.parquet` file.

### Generating Data

First, generate the full Ego4D-PMD data:

```bash
python scripts/data/run_generate_ego4d.py --partition train --mismatch_augmentation
python scripts/data/run_generate_ego4d.py --partition val --mismatch_augmentation
python scripts/data/run_generate_ego4d.py --partition test --mismatch_augmentation
```

When generating the training data, a small number of duplicate records may be generated. Configure `scripts/utils/remove_duplicates_in_ego4d.py` then run it to remove them.

This can take several CPU days. We recommend running these commands as Slurm jobs using Slurm's `srun` command to enable data parallelism. Note that progress is intermittently saved.

Then generate our randomly sampled subsets (this is much quicker):

```bash
python scripts/data/run_generate_ego4d.py --partition train --mismatch_augmentation --debug --debug_n_examples 5000
python scripts/data/run_generate_ego4d.py --partition val --mismatch_augmentation --debug --debug_n_examples 250
python scripts/data/run_generate_ego4d.py --partition test --mismatch_augmentation --debug --debug_n_examples 1000
```

Note: We noticed later that some videos were missing or corrupted on our system, which may cause slight variations in the videos that end up in your generated version of Ego4D-PMD. If you run into this issue while trying to reproduce results, contact Shane Storks and he'll share our generated data subsets directly with you.

## Running Experiments

### Self-Dialog Inference and Evaluation

Use the `scripts/run_vqa_iterative.py` script to run self-dialog inference with VLMs as shown below. This script supports data parallelism with Slurm's `srun` command.

```bash
python scripts/run_vqa_iterative.py --run_id <unique ID> --max_iterations 10 --question_selection_strategy <coherence|likelihood> --exclude_history_from_vqa \
     --eval_partition val --debug --debug_n_examples 250 \
     --vlm_name <vlm_name_or_path> --hf_hub_revision <revision_id>
```

Several notes about configuring the command:

* For the validation set, use the above `--eval_partition` and `--debug_n_examples` arguments. For testing, use `--eval_partition test` and `--debug_n_examples 1000`, and additionally provide the arguments `--early_stop_delta` and `--confident_range` as tuned in `stopping_criteria_tuning/tuned_stopping_criteria.json` in the output directory from the corresponding validation run. For inference on the training data (i.e., to generate training data for coherence-based fine-tuning), use `eval_partition train` and `--debug_n_examples 5000`.

* To evaluate the VLMs studied in the paper, configure `--vlm_name` and `--hf_hub_revision` as follows:
     * InstructBLIP: `--vlm_name "Salesforce/instructblip-vicuna-7b" --hf_hub_revision "52ba0cb2c44d96b2fcceed4e84141dc40d2b6a92"`
     * LLaVA 1.5: `--vlm_name "llava-hf/llava-1.5-7b-hf" --hf_hub_revision "8c85e9a4d626b7b908448be32c1ba5ad79b95e76"`
     * Llama 3: `--vlm_name "meta-llama/Llama-3.2-11B-Vision-Instruct" --hf_hub_revision "cee5b78e6faed15d5f2e6d8a654fd5b247c0d5ca"`

* Configure `--question_selection_strategy` for `likelihood`- or `coherence`-based question selection.

* For experiments with in-context learning, add `--n_icl_demonstrations 20`.

* For experiments utilizing a trained VQG adapter, add `vqg_adapter_path <path/to/vqg/adapter/output/director>`.

* For experiments with length penalty, add --length_penalty "-1.0".

* Note that any `batch_size` args for the script can be maximized for your GPU memory; the default arguments were configured for NVIDIA A40 GPUs (48GB VRAM).

If your environment uses Slurm, feel free to use `scripts/slurm/submit_iterativevqa_slurm_script.py` to submit jobs (or create a Bash script to easily submit several jobs at once). 

After completing, you can view `metrics_table_<partition>.json`, which includes the metrics in the paper's result's tables. Please note that for testing, you should not use the `accuracy` metric there, and instead identify the accuracy value in `metrics_accuracy_<partition>.json` for the mistake confidence threshold selected on the validation data. You can also see a 3D scatter plot of evaluation metrics at `3d_graph_base.pdf`.

#### GPT-4o

To evaluate GPT-4o, use `scripts/run_vqa_iterative_GPT.py`:

```bash
python scripts/run_vqa_iterative_GPT.py --api_key <your_api_key> --endpoint <endpoint_url> --run_id <unique_ID> --max_iterations 10 --eval_partition val  --debug --debug_n_examples 250 --exclude_history_from_vqa  --no_early_stopping
```

Configure `--api_key` and `--endpoint` accordingly for your method of accessing GPT-4o. The `--eval_partition` and `--debug_n_examples` arguments should be configured like the above command for open VLMs. Note that the above was run using Azure OpenAI Studio, and may require adjustments for using the OpenAI API.

The first run of this script will generate GPT-4o questions for all 10 possible iterations without any early stopping criteria to avoid waste from repeated runs. To tune stopping criteria, run `scripts/stopping_criteria_hyperparameter_search.py`:

```bash
python scripts/stopping_criteria_hyperparameter_search.py --this_results_dir "/path/to/gpt-4o/output/directory" --load_coherence_metrics
```

After running this, the results after stopping criteria tuning should appear in the `stopping_criteria_tuning` subdirectory of your provided output directory.

### Coherence-Based Fine-Tuning

Before running training, you'll need to generate data for DPO training and validation. Run the above inference script on the training and validation data, and note the output directories for each run. To reproduce the main results in the paper, use arguments `--question_selection_strategy coherence` and `--n_icl_demonstrations 20`. To reproduce the additional fine-tuning results in the appendices, remove `--n_icl_demonstrations 20` for no in-context learning, or add `--length_penalty "-1.0"` to include a length penalty.

Use the `scripts/train_vqa_iterative_dpo.py` script to run coherence-based fine-tuning. We use `srun` and `torchrun` to enable GPU parallelism, but the script should also run on a single GPU.

```bash
srun --cpus-per-task 4 poetry run torchrun --nnodes=4 --nproc_per_node=1 --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint="$HOST_NODE:25703" scripts/train_vqa_iterative_dpo.py \
     --train_data_path "/path/to/training/outputs/outputs_train.json" \
     --val_data_path "/path/to/val/outputs/outputs_val.json" \
     --run_id <unique ID> --n_epochs 10 --learning_rate <lr> --dpo_beta <beta>
```

Batch size arguments can again be maximized for your environment. For reproducibility, we used a batch size of 4 and used `srun` to train models in parallel on 4 NVIDIA A40 GPUs each. See the paper appendix for more details about hyperparameter search. The script `scripts/slurm/submit_dpo_slurm_scripts.py` can be used to initiate hyperparameter search through Slurm.

### Rationale-Free Inference and Evaluation

In the appendix, we include a rationale-free inference approach as a reference point. To reproduce this result, use the following command for open-source VLMs:

```bash
python scripts/run_vqa_successvqa.py --vlm_name "llava-hf/llava-1.5-7b-hf" --eval_partition val --debug --debug_n_examples 250 --run_id <unique ID>
```

The `--vlm_name`, `--eval_partition`, and `--debug_n_examples` arguments in this command should be configured like the above command for the self-dialog approach. For GPT-4o:

```bash
python scripts/run_vqa_successvqa_GPT.py --api_key <your_api_key> --endpoint <endpoint_url> --eval_partition val --debug --debug_n_examples 250 --run_id <unique_ID>
```

Configure `--api_key` and `--endpoint` accordingly for your method of accessing GPT-4o. The `--eval_partition` and `--debug_n_examples` arguments should be configured like the above command for the self-dialog approach.

### Diversity-Based Ranking

In the appendix, we include results for a diversity-based ranking approach applied to open-source VLMs. To reproduce this result, use the following command:

```bash
python run_vqa_iterative_diversity.py --run_id <unique ID> --max_iterations 10 --question_selection_strategy <coherence|likelihood> --exclude_history_from_vqa \
     --eval_partition val --debug --debug_n_examples 250 \
     --vlm_name <vlm_name_or_path> --hf_hub_revision <revision_id>
```

You can configure this script similarly to the main `run_vqa_iterative.py` script (as described above).

### DET Curves and Other Analysis

To generate DET curves and other various metrics for multiple compared results, configure `analysis_config.yml` to point to your desired configurations of output directories, then run `scripts/analyze_IterativeVQA.py`.

To reproduce the timing results reported in the appendix, you can use the Jupyter notebook at `notebooks/timing_demo.ipynb`.

To visualize data from Ego4D-PMD and generated VQG training data, you can use the Jupyter notebook at `notebooks/visualize_data.ipynb`.

## Citation

If you use any part of this work, please cite us:

```
@misc{storks2025coherentproceduralmistakedetection,
      title={Transparent and Coherent Procedural Mistake Detection}, 
      author={Shane Storks and Itamar Bar-Yossef and Yayuan Li and Zheyuan Zhang and Jason J. Corso and Joyce Chai},
      year={2025},
      eprint={2412.11927},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.11927}, 
}
```
