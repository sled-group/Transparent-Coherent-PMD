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

Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) if needed, and set up the virtual environment:

```
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install
poetry shell
```

After setup, activate the environment by running `poetry shell`. For Jupyter notebook support, run this when the `travel` environment is activated:
```
python -m ipykernel install --user --name=travel
```

Then use the `travel` kernel in Jupyter.
