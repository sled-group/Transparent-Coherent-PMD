# TRAVEl

Development code for work on creating **T**iered **R**easoning for **A**ction-**V**ideo **E**quiva**l**ence (**TRAVEl**) for action mistake detection perceptually-enabled task guidance (PTG).

## Setup

Clone the repo:

```
git clone git@github.com:shanestorks/TRAVEl.git
```

In some environments, you may have issues getting Poetry to use the correct Python interpreter. In this case, you can specify which one to use by:
```
poetry env use /path/to/desired/python
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

For Jupyter notebook support, run this when the `travel` environment is activated (i.e., after running `poetry shell`):
```
python -m ipykernel install --user --name=travel
```

Also ensure that the same CUDA installation is available in your notebook environment; for example, in a Great Lakes JupyterLab session, you can add `load cuda/11.7.1` under "Module commands". Then use the `travel` kernel in Jupyter.