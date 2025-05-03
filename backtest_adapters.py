import numpy as np
import pandas as pd
from genetic import GA
from pso import PSO
from tcn.make_dataset import make_DL_dataset
import torch
import json
import sys
import os

# Hack to make imports inside tcn work, without having to rewrite inside imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/tcn")
from tcn.train.train import Trainer


class GABacktest:
    def __init__(
        self,
        eps=0.01,
        delta=0.99,
        cardinality=10,
        pop_size=100,
        num_gens=200,
        trial_count=5,
    ):
        self.pop_size = pop_size
        self.num_gens = num_gens
        self.trial_count = trial_count
        self.eps = eps
        self.delta = delta
        self.cardinality = cardinality
        self.eps_floor = np.array([eps] * 101)
        self.delta_ceil = np.array([delta] * 101)

    def train(self):
        pass

    def get_weights(self, returns_arr: np.ndarray):
        df = pd.DataFrame(returns_arr)
        Genetic_Algorithm = GA(
            df,
            eps=self.eps_floor,
            delta=self.delta_ceil,
            cardinality=self.cardinality,
        )

        # Sample parameters
        pop_size = 500
        num_gens = 500
        trial_count = 5

        crsvr = "uniform_crossover"

        global_best_score = 0
        best_wts = []

        for i in range(0, trial_count):
            scores, wts, times, runtime = Genetic_Algorithm.solveGA(
                pop_size=pop_size,
                tournament_size=7,
                crossover_str=crsvr,
                crossover_rate=0.9,
                mutation_rate=0.2,
                num_generations=num_gens,
                elite_pct=0.1,
                patience_pct=0.2,
                seed=i + 1,
            )
            best_wts.append(wts)
            # scores : record of scores over the generations
            # wts : the record of best wts
            best_solution = Genetic_Algorithm.optimal_wts
            best_score = Genetic_Algorithm.sharpe(best_solution)
            if best_score > global_best_score:
                global_best_solution = best_solution
                global_best_score = best_score
        return global_best_solution


class PSOBacktest:
    def __init__(self, risk_free_rate: float = 0.0524 / 365, num_iterations: int = 500):
        self.risk_free_rate = risk_free_rate
        self.num_iterations = num_iterations

    def train(self):
        pass

    def get_weights(self, returns: np.ndarray):
        returns_df = pd.DataFrame(returns)
        pso = PSO(returns_df, risk_free_rate=self.risk_free_rate, quiet=True)
        weights, _, _ = pso.run(self.num_iterations)
        return weights


class TCNBacktest:
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_len=60,
        pred_len=20,
        n_stock=101,
        max_assets=10,
        min_weight=0.01,
        max_weight=0.9,
    ):
        self.max_assets = max_assets
        self.min_weight = min_weight
        self.max_weight = max_weight
        train_data, times_train = make_DL_dataset(
            train_df[: int(len(train_df))], train_len + pred_len, n_stock
        )
        test_data, times_test = make_DL_dataset(
            test_df[: int(len(test_df))], train_len + pred_len, n_stock
        )

        self.worker = Trainer(json.load(open("tcn/config/tcn_backtest.json", "r")))
        x_tr = np.array([x[:train_len] for x in train_data])
        y_tr = np.array([x[-pred_len:] for x in train_data])
        x_te = np.array([x[:train_len] for x in test_data])
        y_te = np.array([x[-pred_len:] for x in test_data])

        self.worker.set_backtest_data(x_tr, y_tr, x_te, y_te, times_test)

    def train(self):
        self.worker.train(visualize=False)
        self.model = self.worker.model

    def get_weights(self, returns: np.ndarray):
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()
        with torch.no_grad():
            # Scale returns by 10 and reshape for TCN
            returns = torch.from_numpy(returns * 10).float()
            # Add batch dimension
            returns = returns.unsqueeze(0)
            weights = self.model(returns)

            # Get the raw weights as numpy array
            raw_weights = weights.cpu().numpy().flatten()

            # Apply max weight constraint - cap weights at max threshold
            raw_weights = np.minimum(raw_weights, self.max_weight)

            # Get indices of top k weights
            topk_indices = np.argsort(raw_weights)[-self.max_assets :]
            # Create mask with zeros except at topk indices
            mask = np.zeros_like(raw_weights)
            mask[topk_indices] = 1

            # Apply min weight constraint to topk assets, set to min_weight if below threshold
            raw_weights = raw_weights * mask
            raw_weights[topk_indices] = np.maximum(
                raw_weights[topk_indices], self.min_weight
            )

            # Normalize weights to sum to 1
            if np.sum(raw_weights) > 0:
                raw_weights = raw_weights / np.sum(raw_weights)

            # Reshape back to original shape
            weights = torch.from_numpy(raw_weights).reshape(weights.shape)

        return weights.cpu().numpy()
