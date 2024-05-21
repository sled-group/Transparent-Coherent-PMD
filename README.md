# TRAVEl

Development code for work on creating **T**iered **R**easoning for **A**ction-**V**ideo **E**quiva**l**ence (**TRAVEl**) for action mistake detection perceptually-enabled task guidance (PTG).

## Setup

Clone the repo:

```
git clone git@github.com:shanestorks/TRAVEl.git
```

**Reconfigure `cache.model_cache_dir` in `config.yml` to point to a directory you own. If you're using Great Lakes, you should be able to replace `sstorks` with your own uniqname to use `scratch` storage.**

Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) if needed, and ensure a CUDA installation is available which is compatible with both `bitsandbytes` and `torch`. On Great Lakes, you will need to load an appropriate module before installing the dependencies. For example:

```
module load cuda/11.7.1
```

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

Note that the same CUDA installation must be available whenever you activate the environment using `poetry shell`. The script `prepare_great_lakes.sh` provides the set of commands to run to prepare the Great Lakes environment after initial setup; run it from this repo.

### Jupyter Notebook Support

For Jupyter notebook support, run this when the `travel` environment is activated (i.e., after running `poetry shell`):

```
python -m ipykernel install --user --name=travel
```

Also ensure that the same CUDA installation is available in your notebook environment; for example, in a Great Lakes JupyterLab session, you can add `load cuda/11.7.1` under "Module commands". Then use the `travel` kernel in Jupyter.

### Slurm Script Support

In a Slurm script, ensure you prepend commands with `poetry run` to activate the virtual environment.

## Running Experiments

Before running experiments, be sure to configure the directories in `config.yml` accordingly.

### Configuration

Make a copy of `config_sample.yml` and name it `config.yml`. Configure the arguments in `config.yml` as needed (especially the cache directories).

### SuccessVQA Baseline

Baseline that simply asks VLMs whether some procedure was successfully performed. Check `run_vqa_successvqa.py` for more command-line arguments.

```
python run_vqa_successvqa.py --eval_split <val|test>
```

### VQG2VQA

First, run visual question generation (VQG) to generate questions for each recipe step:

```
python run_vqg.py --temperature <temperature> --top_p <top_p>
```

Then, based on the outputs generated from the VQG script, run VQA mistake detection:

```
python run_vqa_vqg2vqa.py --eval_split <val|test> --vqg_directory "path/to/vqg/outputs"
```

Check `run_vqa_vqg2vqa.py` for more configurable command-line arguments.

### Learning VQG

#### Generating Training Data from Ego4D
```
python run_vqg_learning_generation.py
python run_vqg_learning_vqa.py --vqg_directory /saved_results/path/to/generated/questions/from/previous/step/
```

Add `--visual_filter_mode spatial` to the second command if you want to use the spatial attention filter.

#### Training VQG from Generated Data

With only 1 GPU:
```
python run_vqg_learning_training.py --data_directory <path/to/generated/training/data/directory>
```

With multiple GPUs:
```
python -m torch.distributed.launch --nproc_per_node=2 run_vqg_learning_training.py --data_directory <path/to/generated/training/data/directory>
```

Evaluate the trained pipeline with `run_vqa_vqg2vqa.py` following the above.