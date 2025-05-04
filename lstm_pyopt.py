import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
from pypfopt import (
    EfficientFrontier,
    objective_functions,
)

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ReturnDataset(Dataset):
    def __init__(self, return_df, seq_len=60):
        """
        return_df: pandas DataFrame of returns
                  shape: (time, num_assets)
        seq_len:  window size (default: 60)
        """
        self.tickers = return_df.columns.tolist()  # Store ticker names
        returns = return_df.values.astype(np.float32)
        print(
            f"Dataset shape: {returns.shape}, Mean: {returns.mean():.6f}, Std: {returns.std():.6f}"
        )

        num_samples = len(returns) - seq_len - 1
        self.X = []
        self.Y = []

        for i in range(num_samples):
            window = returns[i : i + seq_len]
            self.X.append(window)
            self.Y.append(returns[i + seq_len])

        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32).to(DEVICE)
        self.Y = torch.tensor(np.stack(self.Y), dtype=torch.float32).to(DEVICE)
        print(f"Created dataset with {len(self.X)} samples")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class LSTMReturnPredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        output_dim,
    ):
        super(LSTMReturnPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(input_dim)])
        self.to(DEVICE)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.norm(out[:, -1])
        return torch.cat([head(h) for head in self.heads], dim=1)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=500,
    lr=1e-3,
    weight_decay=1e-5,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    patience = 20
    patience_counter = 0

    # For plotting
    train_losses = []
    val_losses = []
    best_model_state = None

    print(f"\nStarting training with lr={lr}, weight_decay={weight_decay}")
    print(f"Model architecture:\n{model}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            predicted_returns = model(X_batch)
            loss = criterion(predicted_returns, Y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / num_batches
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                predicted_returns = model(X_batch)
                batch_loss = criterion(predicted_returns, Y_batch)
                val_loss += batch_loss.item()
                val_batches += 1

        val_loss = val_loss / val_batches
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch + 2} epochs")
                break

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            f"\nRestored best model weights with validation loss: {best_val_loss:.6f}"
        )

    return model


class LSTMPyOptBacktest:
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        seq_len: int,
        future_days: int,
        params: dict = None,
    ):
        # Create dataloaders with fixed worker seeds
        def worker_init_fn(worker_id):
            np.random.seed(SEED + worker_id)

        self.dim = train_df.shape[1]
        self.tickers = train_df.columns.tolist()
        self.train_df = train_df
        self.test_df = test_df
        self.seq_len = seq_len
        self.future_days = future_days

        # Default parameters, overridden by any provided params
        self.params = {
            "lr": 0.001,
            "hidden_dim": 64,
            "num_layers": 1,
            "weight_decay": 1e-5,
            "max_weight": 0.1,  # 10% maximum weight constraint
            "min_weight": 0.01,
            "k_assets": 10,
            "risk_free_rate": 0.0524,  # 5.24% annual risk-free rate
        }

        if params:
            self.params.update(params)

        # Z-score normalization
        self.mu = train_df.mean(0)  # per-ticker mean
        self.sigma = train_df.std(0).replace(0, 1e-6)  # avoid divide-by-zero

        # Apply z-score normalization
        train_df_norm = (train_df - self.mu) / self.sigma
        test_df_norm = (test_df - self.mu) / self.sigma

        self.train_dataset = ReturnDataset(train_df_norm, seq_len)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=True,
            worker_init_fn=worker_init_fn,
        )

        self.test_dataset = ReturnDataset(test_df_norm, seq_len)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=64,
            shuffle=False,
            worker_init_fn=worker_init_fn,
        )

        self.model = None

    def train(self):
        # Create and train the LSTM model to predict returns
        self.model = LSTMReturnPredictor(
            input_dim=self.dim,
            hidden_dim=self.params["hidden_dim"],
            num_layers=self.params["num_layers"],
            output_dim=self.dim,
        )

        train_model(
            self.model,
            self.train_loader,
            self.test_loader,
            epochs=500,
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )

    def predict_price(self, returns: np.ndarray):
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Input returns has shape (seq_len, num_assets)
        # Z-score normalize the input
        returns_norm = (returns - self.mu.values) / self.sigma.values

        # Need to add batch dimension
        X = torch.tensor(returns_norm, dtype=torch.float32).to(DEVICE)
        X = X.unsqueeze(0)  # Add batch dimension (1, seq_len, num_assets)

        # Get return predictions
        self.model.eval()
        with torch.no_grad():
            predicted_returns = self.model(X)
            # Un-normalize: first scale back, then un-z-score
            predicted_returns = (
                predicted_returns.cpu().numpy()[0] * self.sigma.values
            ) + self.mu.values
        return predicted_returns

    def get_weights(self, returns: np.ndarray):
        """
        Use the LSTM model to predict next day returns,
        then use PyPortfolioOpt to optimize the portfolio.

        Args:
            returns: numpy array of shape (seq_len, num_assets)

        Returns:
            weights: numpy array of shape (num_assets,)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Z-score normalize the input
        returns_norm = (returns - self.mu.values) / self.sigma.values

        # Input returns has shape (seq_len, num_assets)
        # Need to add batch dimension
        X = torch.tensor(returns_norm, dtype=torch.float32).to(DEVICE)
        X = X.unsqueeze(0)  # Add batch dimension (1, seq_len, num_assets)

        # Get return predictions
        self.model.eval()
        with torch.no_grad():
            predicted_returns = self.model(X)
            predicted_returns = predicted_returns.cpu().numpy()[
                0
            ]  # Shape: (num_assets,)
            # Un-normalize: first scale back, then un-z-score
            predicted_returns = (predicted_returns * self.sigma.values) + self.mu.values
        # Convert daily returns to annualized returns
        annual_returns = (1 + predicted_returns) ** 252 - 1
        mu = pd.Series(annual_returns, index=self.tickers)

        # Create DataFrame from returns array with proper column names
        returns_df = pd.DataFrame(returns, columns=self.tickers)
        cov_matrix = returns_df.cov()

        try:
            # Create the portfolio optimization problem with weight bounds
            ef = EfficientFrontier(mu, cov_matrix)

            ef.add_constraint(lambda x: x <= self.params["max_weight"])
            # Use L2 regularization to promote a more diversified portfolio
            # Adjust gamma to control diversification (higher = more diversification)
            if self.params["k_assets"]:
                gamma = 1.0 / self.params["k_assets"]
                ef.add_objective(objective_functions.L2_reg, gamma=gamma)

            warnings.filterwarnings("ignore", module="pypfopt")
            # Optimize for maximum Sharpe ratio
            ef.max_sharpe(risk_free_rate=self.params["risk_free_rate"])

            # Clean weights - PyPortfolioOpt has utilities to clean small weights
            cleaned_weights = ef.clean_weights()

            # First enforce the cardinality constraint - keep only the top k assets
            weight_series = pd.Series(cleaned_weights)
            top_k_assets = weight_series.nlargest(self.params["k_assets"]).index

            # Reset weights dictionary with only top k assets and enforce min weight
            filtered_weights = {}
            for ticker in self.tickers:
                if ticker in top_k_assets:
                    # Apply the minimum weight constraint only to weights that will be non-zero
                    filtered_weights[ticker] = max(
                        cleaned_weights[ticker], self.params["min_weight"]
                    )
                else:
                    filtered_weights[ticker] = 0

            # Normalize the weights to sum to 1
            weight_sum = sum(filtered_weights.values())
            normalized_weights = {
                ticker: weight / weight_sum
                for ticker, weight in filtered_weights.items()
            }

            # Convert to numpy array in the same order as the input
            weight_array = np.array(
                [normalized_weights[ticker] for ticker in self.tickers]
            )

            return weight_array

        except Exception:
            # Fallback to equal weights if optimization fails
            n_assets = returns.shape[1]
            return np.ones(n_assets) / n_assets


if __name__ == "__main__":
    import backtest

    print("Running LSTM_PyOpt backtest")
    backtest.run_backtest(selected_algorithms=["LSTM_PyOpt"])
