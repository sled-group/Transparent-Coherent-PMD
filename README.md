# TRAVEl

Development code for work on creating **T**iered **R**easoning for **A**ction-**V**ideo **E**quiva**l**ence (**TRAVEl**) for action mistake detection perceptually-enabled task guidance (PTG).

## Setup

Clone the repo:

```
git clone git@github.com:shanestorks/TRAVEl.git
```

**Reconfigure `cache.model_cache_dir` in `config.yml` to point to a directory you own. If you're using Great Lakes, you should be able to replace `sstorks` with your own uniqname to use `scratch` storage.**

In some environments, you may have issues getting Poetry to use the correct Python interpreter. In this case, you can specify which one to use by:

```
poetry env use /path/to/desired/python
```

If using Great Lakes, you will need to create a `conda` environment with Python 3.10 and activate it before running the above:

```
module load python3.10-anaconda/2023.03
conda create --name python310 python=3.10
```

Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) if needed, and ensure a CUDA installation is available which is compatible with both `bitsandbytes` and `torch`. On Great Lakes, you will need to load an appropriate module before installing the dependencies. For example:

```
module load cuda/11.7.1
```

Then set up the virtual environment:

```
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install
poetry shell
```

Note that the same CUDA installation must be available whenever you activate the environment using `poetry shell`.

### Jupyter Notebook Support

For Jupyter notebook support, run this when the `travel` environment is activated (i.e., after running `poetry shell`):

```
python -m ipykernel install --user --name=travel
```

Also ensure that the same CUDA installation is available in your notebook environment; for example, in a Great Lakes JupyterLab session, you can add `load cuda/11.7.1` under "Module commands". Then use the `travel` kernel in Jupyter.

### Slurm Script Support

In a Slurm script, ensure you prepend commands with `poetry run` to activate the virtual environment.

## Running Experiments

### Configuration

You can configure some arguments and hyperparameters in `config.yml`, including video frame sampling frequency and directories where preprocessed data and results are stored.

### SuccessVQA

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
