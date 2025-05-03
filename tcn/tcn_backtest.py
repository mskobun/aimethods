import sys
import os

# Hack to make importing from outside the directory work, without having to rewrite inside import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import pandas as pd
import numpy as np
import torch
from make_dataset import make_DL_dataset
from train.train import Trainer


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
