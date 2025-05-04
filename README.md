# AI Methods Coursework

This [repository](https://github.com/mskobun/aimethods) contains 4 portfolio optimization algorithms our group has implemented for COMP2024 module at the University of Nottingham Malaysia.

The algorithms are:
- Particle swarm optimization (`pso.py`)
- Genetic algorithms (`genetic.py`)
- Temporal convolutional neural network (`tcn` folder)
- Long short-term memory (LSTM) neural network, combined with a mean-variance optimizer (`lstm_pyopt.py`)

In addition, a data downloader (`dataloader.py`) and a backtesting framework (`backtest.py`) is included.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency managment. If you have `uv` installed:
```sh
# Install dependencies
uv sync
# Activate the virtual environment
source .venv/bin/activate
```

A `requirements.txt` file is also provided.

## Running

To run the backtest or view backtest results:
```sh
python backtest.py
```

The resulting plots will be shown on your screen as well as saved to the `results` folder.