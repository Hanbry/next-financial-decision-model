# Next Financial Decision Model

This project utilizes reinforcement learning to develop a trading agent capable of operating in a financial environment. The implementation leverages the `tf-agents` library from TensorFlow for efficient and effective reinforcement learning algorithms.

## Project Structure

- `agents/`: Contains the implementation of the PPO agent.
- `checkpoints/`: Contains the saved model checkpoints.
- `data/`: Contains the financial data used for training and evaluation.
- `environments/`: Contains the implementation of the financial environment.
- `experiments/`: Contains the scripts to run experiments.
- `models/`: Contains the model definitions.

## Branches

The `main` branch is configured to run on Apple's M-series chips.
The `cuda12-version` branch is configured to run on devices with CUDA 12.3.
Both branches are up to date with the latest developments.

## Setup

```sh
pip install pipenv
pipenv install
```
## Train 
### Run and continue from last checkpoint

```sh
pipenv run python3 main.py
```

### Run and clean all previous checkpoints

```sh
pipenv run python3 main.py --clean_run
```