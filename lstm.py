import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import optuna
from optuna.trial import Trial
import random

# Set ALL random seeds for complete reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Determine device and set appropriate seeds
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"\nUsing device: {DEVICE}")


class LSTMBacktest:
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        seq_len: int,
        future_days: int,
        params: dict,
        training_plot_filename: str = None,
    ):
        # Create dataloaders with fixed worker seeds
        def worker_init_fn(worker_id):
            np.random.seed(SEED + worker_id)

        self.training_plot_filename = training_plot_filename
        self.dim = train_df.shape[1]
        self.train_df = train_df
        self.train_dataset = PriceDataset(train_df, seq_len, future_days)
        self.test_df = test_df
        self.params = {
            "lr": 0.0001288635099180348,
            "hidden_dim": 64,
            "num_layers": 1,
            "dropout": 0.2,
            "weight_decay": 5.6136677829138486e-05,
            "max_weight": 0.9,
            "min_weight": 0.01,
            "k_assets": 10,
            "risk_free_rate": 0.0524,  # 5.24% annual risk-free rate
        } | params
        self.train_dataset = PriceDataset(train_df, seq_len, future_days)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=False,
            worker_init_fn=worker_init_fn,
        )

        self.test_dataset = PriceDataset(test_df, seq_len, future_days)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=64,
            shuffle=False,
            worker_init_fn=worker_init_fn,
        )

        self.model = None

    def train(self, plot=True, plot_filename=None):
        # Train final model with best parameters
        self.model = LSTMPortfolio(
            input_dim=self.dim,
            hidden_dim=self.params["hidden_dim"],
            num_layers=self.params["num_layers"],
            output_dim=self.dim,
            dropout=self.params["dropout"],
            max_weight=self.params["max_weight"],
            min_weight=self.params["min_weight"],
            k_assets=self.params["k_assets"],
            risk_free_rate=self.params["risk_free_rate"],
        )

        train(
            self.model,
            self.train_loader,
            self.test_loader,
            epochs=500,
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
            plot_filename=self.training_plot_filename,
        )

    def get_weights(self, returns: np.ndarray):
        # Input returns is already (seq_len, num_assets)
        # Just need to add batch dimension
        X = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        X = X.unsqueeze(0)  # Add batch dimension (1, seq_len, num_assets)

        if self.model is None:
            raise ValueError("Model not trained")
        self.model.eval()
        with torch.no_grad():
            weights = self.model(X)
        return weights.cpu().numpy()


# ===== Dataset class =====


class PriceDataset(Dataset):
    def __init__(self, return_df, seq_len, future_days=1):
        """
        return_df: pandas DataFrame of returns
                  shape: (time, num_assets)
        seq_len:  window size
        future_days: number of days before selling the assets
        """
        self.tickers = return_df.columns.tolist()  # Store only the ticker names
        returns = return_df.values.astype(np.float32)
        print(
            f"Dataset shape: {returns.shape}, Mean: {returns.mean():.6f}, Std: {returns.std():.6f}"
        )

        num_samples = len(returns) - seq_len - future_days - 1
        self.X = []
        self.Y = []

        for i in range(num_samples):
            window = returns[i : i + seq_len]
            self.X.append(window)
            self.Y.append(returns[i + seq_len + future_days - 1])

        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32).to(DEVICE)
        self.Y = torch.tensor(np.stack(self.Y), dtype=torch.float32).to(DEVICE)
        print(f"Created dataset with {len(self.X)} samples")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ===== Model =====


class LSTMPortfolio(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        output_dim,
        dropout=0.1,
        max_weight=0.10,
        min_weight=0.01,  # 1% minimum weight
        k_assets=None,
        risk_free_rate=0.0524,  # 5.24% annual risk-free rate
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_dim: Number of output features (number of assets)
            dropout: Dropout rate
            max_weight: Maximum allowed weight per asset (default: 0.10 or 10%)
            min_weight: Minimum allowed weight per asset (default: 0.01 or 1%)
            k_assets: Maximum number of assets to invest in (if None, no constraint)
            risk_free_rate: Annual risk-free rate (default: 0.0524 or 5.24%)
        """
        super(LSTMPortfolio, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.k_assets = k_assets
        self.risk_free_rate = risk_free_rate
        self.to(DEVICE)  # Move model to appropriate device

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        weights = self.fc(last_hidden)

        # Apply softmax to get initial portfolio weights
        weights = F.softmax(weights, dim=1)  # portfolio weights

        if self.k_assets is not None:
            # For each sample in the batch, keep only the top k assets
            batch_size = weights.shape[0]
            topk_values, _ = torch.topk(weights, self.k_assets, dim=1)
            # Get the smallest value among top k for each sample
            threshold = topk_values[:, -1].unsqueeze(1)
            # Create a mask for weights >= threshold
            mask = weights >= threshold
            # Zero out weights not in top k
            weights = weights * mask.float()

            # Apply minimum weight to selected assets
            weights = torch.where(
                mask,
                torch.maximum(weights, torch.tensor(self.min_weight).to(DEVICE)),
                torch.tensor(0.0).to(DEVICE),
            )

            # Renormalize to sum to 1
            weights = weights / weights.sum(dim=1, keepdim=True)

        return weights


# ===== Loss =====


def portfolio_loss(model, weights, future_returns):
    portfolio_returns = (weights * future_returns).sum(
        dim=1
    )  # returns for each sample in batch

    # Annualize Sharpe ratio (assuming daily data)
    mean_return = portfolio_returns.mean() * 252  # annualize mean
    std_return = portfolio_returns.std() * (252**0.5) + 1e-6  # annualize std
    sharpe = (
        mean_return - model.risk_free_rate
    ) / std_return  # use model's risk-free rate

    # Penalize weights outside [min_weight, max_weight] range
    max_weight_penalty = torch.mean(torch.relu(weights - model.max_weight)) * 1000.0

    # For min weight, only penalize non-zero weights that are below min_weight
    non_zero_mask = weights > 0
    min_weight_penalty = (
        torch.mean(torch.relu(model.min_weight - weights) * non_zero_mask.float())
        * 1000.0
    )

    total_penalty = max_weight_penalty + min_weight_penalty

    return (
        -sharpe + total_penalty,
        sharpe.item(),
    )  # return both loss and sharpe for monitoring


def evaluate(model, dataloader, plot=False, plot_filename=None):
    model.eval()
    all_weights = []
    all_returns = []
    dates = []  # Store dates from the dataset

    with torch.no_grad():
        for batch_idx, (X_batch, Y_batch) in enumerate(dataloader):
            weights = model(X_batch)
            all_weights.append(weights)
            all_returns.append(Y_batch)

    weights = torch.cat(all_weights)
    returns = torch.cat(all_returns)
    weights_np = weights.cpu().numpy()

    portfolio_returns = (weights * returns).sum(dim=1)
    mean_return = portfolio_returns.mean() * 252
    std_return = portfolio_returns.std() * (252**0.5) + 1e-6
    sharpe = (mean_return - model.risk_free_rate) / std_return

    # Calculate average weights and find max
    avg_weights = weights.mean(dim=0).cpu().numpy()
    max_weight_val = avg_weights.max()
    max_weight_idx = avg_weights.argmax()

    # Get ticker names if available
    tickers = getattr(dataloader.dataset, "tickers", None)

    # Print average weights
    print("\nAverage Portfolio Weights:")
    # Sort by weight descending
    sorted_idx = avg_weights.argsort()[::-1]
    for idx in sorted_idx:
        if avg_weights[idx] > 0.001:  # Only print weights > 0.1%
            if tickers:
                print(f"{tickers[idx]}: {avg_weights[idx]:.2%}")
            else:
                print(f"Asset {idx}: {avg_weights[idx]:.2%}")

    return sharpe.item(), max_weight_val, max_weight_idx


# ===== Train loop =====


def train(
    model,
    train_loader,
    val_loader,
    epochs=1000,
    lr=1e-3,
    weight_decay=2e-5,
    plot_filename=None,
):
    print(train_loader.dataset.X[0])
    torch.manual_seed(SEED)  # Reset seed before training
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_sharpe = -float("inf")
    patience = 20
    patience_counter = 0

    # Store validation Sharpe ratios for plotting
    val_sharpes = []
    # Store best model state
    best_model_state = None

    print(f"\nStarting training with lr={lr}, weight_decay={weight_decay}")
    print(f"Model architecture:\n{model}")
    print(f"Weight constraints: min={model.min_weight:.2%}, max={model.max_weight:.2%}")
    print(f"Risk-free rate: {model.risk_free_rate:.2%}")
    if model.k_assets is not None:
        print(f"Maximum number of assets: {model.k_assets}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_sharpe = 0
        num_batches = 1
        max_weight_epoch = 0

        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            weights = model(X_batch)
            loss, batch_sharpe = portfolio_loss(model, weights, Y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_sharpe += batch_sharpe
            num_batches += 1
            max_weight_epoch = max(max_weight_epoch, weights.max().item())

            if batch_idx == 0:  # Print first batch stats
                print(
                    f"Epoch {epoch + 1} Batch 0: Shape X={X_batch.shape}, Y={Y_batch.shape}, Device={X_batch.device}"
                )

        train_sharpe /= num_batches
        val_sharpe, max_weight, max_weight_idx = evaluate(model, val_loader, plot=False)
        val_sharpes.append(val_sharpe)

        print(
            f"Epoch {epoch + 1}: Train Sharpe = {train_sharpe:.2f}, "
            f"Val Sharpe = {val_sharpe:.2f}, Max Weight = {max_weight:.2f} (Asset {max_weight_idx}), "
            f"Loss = {total_loss / num_batches:.4f}"
        )

        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
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
            f"\nRestored best model weights with validation Sharpe ratio: {best_val_sharpe:.2f}"
        )

    # Plot validation Sharpe ratio over epochs if requested
    if plot_filename:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(val_sharpes) + 1), val_sharpes, "b-", label="Validation Sharpe"
        )
        # Calculate best Sharpe so far at each epoch
        best_sharpes = [max(val_sharpes[: i + 1]) for i in range(len(val_sharpes))]
        plt.plot(
            range(1, len(best_sharpes) + 1),
            best_sharpes,
            "r--",
            label="Best Sharpe So Far",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Sharpe Ratio")
        plt.title("Validation Sharpe Ratio over Training")
        plt.grid(True)
        plt.legend()
        plt.savefig(plot_filename)
        plt.close()


def objective(trial: Trial, train_loader, val_loader):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)  # Try 1-3 LSTM layers
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    max_weight = trial.suggest_float(
        "max_weight", 0.05, 0.20
    )  # Allow 5% to 20% per asset
    min_weight = trial.suggest_float(
        "min_weight", 0.01, 0.10
    )  # Allow 1% to 10% per asset
    k_assets = trial.suggest_int(
        "k_assets", 5, 20
    )  # Allow portfolio size from 5 to 20 assets
    risk_free_rate = trial.suggest_float("risk_free_rate", 0.02, 0.08)  # 2% to 8% range

    # Create model with suggested hyperparameters
    model = LSTMPortfolio(
        input_dim=train_loader.dataset.X.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=train_loader.dataset.X.shape[2],
        dropout=dropout,
        max_weight=max_weight,
        min_weight=min_weight,
        k_assets=k_assets,
        risk_free_rate=risk_free_rate,
    )

    # Train with suggested hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_sharpe = -float("inf")
    patience = 20
    patience_counter = 0

    for epoch in range(500):  # Max epochs
        model.train()
        train_sharpe = 0
        num_batches = 0
        max_weight_this_epoch = 0.0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            weights = model(X_batch)

            # Track maximum weight this epoch
            max_weight_this_epoch = max(max_weight_this_epoch, weights.max().item())

            # Calculate loss
            portfolio_returns = (weights * Y_batch).sum(dim=1)
            mean_return = portfolio_returns.mean() * 252
            std_return = portfolio_returns.std() * (252**0.5) + 1e-6
            sharpe = (mean_return - model.risk_free_rate) / std_return

            weight_penalty = (
                torch.mean(torch.relu(weights - model.max_weight)) * 10000.0
            )

            loss = -sharpe + weight_penalty
            loss.backward()
            optimizer.step()

            train_sharpe += sharpe.item()
            num_batches += 1

        train_sharpe /= num_batches
        val_sharpe, max_weight, max_weight_idx = evaluate(model, val_loader)

        # Log progress every epoch
        print(
            f"Trial {trial.number} Epoch {epoch}: Train Sharpe = {train_sharpe:.2f}, "
            f"Val Sharpe = {val_sharpe:.2f}, Max Weight = {max_weight:.3f}, "
            f"K assets = {model.k_assets}"
        )

        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Trial {trial.number} Early stopping after {epoch + 1} epochs")
                break

        # Report intermediate value for pruning based on validation performance
        trial.report(val_sharpe, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned by Optuna due to poor performance")
            raise optuna.TrialPruned()

    # Final check - if weights are still too high after all training, reject the trial
    if max_weight > model.max_weight * 1.1:  # Allow 10% margin over max_weight
        print(
            f"Trial {trial.number} rejected due to final high weights: {max_weight:.3f}"
        )
        return -float("inf")

    print(
        f"\nTrial {trial.number} completed with best val Sharpe: {best_val_sharpe:.2f}"
    )
    return best_val_sharpe


def find_best_parameters(train_loader, val_loader, input_dim, output_dim):
    """
    Find the best hyperparameters using Optuna optimization.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        input_dim: Input dimension for the model
        output_dim: Output dimension for the model

    Returns:
        dict: Best hyperparameters found by Optuna
    """
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )

    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader),
        n_trials=50,
        timeout=7200,
    )

    print("\nBest trial:")
    trial = study.best_trial
    print("  Value (Best Val Sharpe): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params


def train_best_model(
    best_params, train_loader, test_loader, val_loader, input_dim, output_dim
):
    """
    Train the final model using the best parameters on combined training data.

    Args:
        best_params: Dictionary of best hyperparameters
        combined_train_loader: DataLoader for combined training data
        test_loader: DataLoader for test data
        input_dim: Input dimension for the model
        output_dim: Output dimension for the model

    Returns:
        model: Trained model
        float: Test Sharpe ratio
        float: Maximum weight
        int: Index of asset with maximum weight
    """
    torch.manual_seed(SEED)  # Reset seed before training
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    best_model = LSTMPortfolio(
        input_dim=input_dim,
        hidden_dim=best_params["hidden_dim"],
        num_layers=best_params["num_layers"],
        output_dim=output_dim,
        dropout=best_params["dropout"],
        max_weight=best_params["max_weight"],
        min_weight=best_params["min_weight"],
        k_assets=best_params["k_assets"],
        risk_free_rate=best_params["risk_free_rate"],
    )

    print("\nTraining final model on combined train+val data...")
    train(
        best_model,
        combined_train_loader,
        test_loader,
        epochs=500,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )

    # Final evaluation with plot
    test_sharpe, max_weight, max_weight_idx = evaluate(
        best_model, val_loader, plot=True, plot_filename="portfolio_weights.png"
    )

    print(f"\nFinal Test Set Performance:")
    print(f"Sharpe Ratio: {test_sharpe:.2f}")
    print(f"Maximum Weight: {max_weight:.2f} (Asset {max_weight_idx})")
    print(f"Weight evolution plot saved as 'portfolio_weights.png'")

    return best_model, test_sharpe, max_weight, max_weight_idx


# ===== Example run =====

if __name__ == "__main__":
    seq_len = 60
    future_days = 20
    df = pd.read_csv("data/return_df.csv", index_col=0)
    print(f"\nLoaded data with shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Sample of returns:\n{df.iloc[:5, :5]}\n")

    # Three-way split (60% train, 20% validation, 20% test)
    train_size = int(0.6 * len(df))
    val_size = int(0.2 * len(df))

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size : train_size + val_size]
    test_df = df.iloc[train_size + val_size :]

    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Create datasets
    train_dataset = PriceDataset(train_df, seq_len, future_days)
    val_dataset = PriceDataset(val_df, seq_len, future_days)
    test_dataset = PriceDataset(test_df, seq_len, future_days)

    # Create dataloaders with fixed worker seeds
    def worker_init_fn(worker_id):
        np.random.seed(SEED + worker_id)

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=False, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, worker_init_fn=worker_init_fn
    )

    # Find best hyperparameters
    # best_params = find_best_parameters(
    #     train_loader, val_loader, input_dim=df.shape[1], output_dim=df.shape[1]
    # )

    best_params = {
        "lr": 0.0001288635099180348,
        "hidden_dim": 64,
        "num_layers": 1,
        "dropout": 0.2,
        "weight_decay": 5.6136677829138486e-05,
        "max_weight": 0.9,
        "min_weight": 0.01,
        "k_assets": 10,
        "risk_free_rate": 0.0524,  # 5.24% annual risk-free rate
    }

    # Create combined train+val dataset for final training
    combined_train_df = pd.concat([train_df, val_df])
    combined_train_dataset = PriceDataset(combined_train_df, seq_len, future_days)
    combined_train_loader = DataLoader(
        combined_train_dataset,
        batch_size=64,
        shuffle=False,
        worker_init_fn=worker_init_fn,
    )

    # Train final model with best parameters
    best_model, test_sharpe, max_weight, max_weight_idx = train_best_model(
        best_params,
        train_loader,
        test_loader,
        val_loader,
        input_dim=df.shape[1],
        output_dim=df.shape[1],
    )
