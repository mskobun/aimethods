# AI Methods Coursework

This [repository](https://github.com/mskobun/aimethods) contains 4 portfolio optimization algorithms our group has implemented for COMP2024 module at the University of Nottingham Malaysia.

The algorithms are:
- Particle swarm optimization (`pso.py`)
- Genetic algorithms (`genetic.py`)
- Temporal convolutional neural network (`tcn` folder)
- Long short-term memory (LSTM) neural network, combined with a mean-variance optimizer (`lstm_pyopt.py`)

In addition, a data downloader (`dataloader.py`) and a backtesting framework (`backtest.py`) is included.

## Datasets

There are two datasets included in `data`:
1. `return_df.csv` - Daily S&P 100 simple returns from 05.01.2010 to 30.12.2024.
2. `backtesting.csv` - Daily S&P 100 simple return from 05.01.2010 to 30.12.2020.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. If you have `uv` installed:
```sh
# Install dependencies
uv sync
# Activate the virtual environment (Linux/MacOS)
source .venv/bin/activate
# Activate the virtual environment (Windows)
.venv\Scripts\activate
```

A `requirements.txt` file is also provided.

## Running

To run the backtest or view backtest results:
```sh
python backtest.py
```

The resulting plots will be shown on your screen as well as saved to the `results` folder.