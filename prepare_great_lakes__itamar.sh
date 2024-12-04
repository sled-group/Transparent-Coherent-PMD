module load cuda/12.1.1
module load python3.10-anaconda/2023.03
poetry env use ~/.conda/envs/python310/bin/python
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry shell