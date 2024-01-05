# TRAVEl-Benchmark

Development code for work on creating the **T**iered **R**easoning for **A**ction-**V**ideo **E**quiva**l**ence (**TRAVEl**) benchmark.

## Setup

Clone the repo and initialize submodules:

```
git clone git@github.com:shanestorks/TRAVEl-Benchmark.git
git submodule init
git submodule update
```

Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) if needed, and set up the virtual environment:

```
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install
poetry shell
```

For Jupyter notebook support, run this when the `travel` environment is activated:
```
python -m ipykernel install --user --name=travel
```

Then use the `travel` kernel in Jupyter.