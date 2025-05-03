import pandas as pd
import numpy as np
from backtest_adapters import GABacktest, PSOBacktest
from tcn.tcn_backtest import TCNBacktest
from lstm_pyopt import LSTMPyOptBacktest
import time
from datetime import datetime
import pickle
import os
import matplotlib.pyplot as plt
import torch


class ReturnDataset:
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

        self.X = np.stack(self.X)
        self.Y = np.stack(self.Y)
        print(f"Created dataset with {len(self.X)} samples")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class BacktestResults:
    def __init__(self, algorithm_name, start_date, end_date):
        self.algorithm_name = algorithm_name
        self.start_date = start_date
        self.end_date = end_date
        self.daily_returns = []
        self.cumulative_returns = []
        self.weights_history = None  # Initialize as None
        self.dates = []
        self.final_cumulative_return = 1.0  # Start at 100%
        self.annualized_return = 0.0
        self.sharpe_ratio = 0.0
        self.training_time = 0.0
        self.backtest_time = 0.0

    def add_result(self, date, weights, portfolio_return):
        self.daily_returns.append(portfolio_return)
        if self.weights_history is None:
            self.weights_history = weights.reshape(
                1, -1
            )  # Initialize with first weights
        else:
            self.weights_history = np.vstack([self.weights_history, weights])
        self.dates.append(date)
        self.final_cumulative_return *= 1 + portfolio_return
        self.cumulative_returns.append(self.final_cumulative_return)

    def calculate_stats(self, benchmark_returns, risk_free_rate=0.0524):
        returns_array = np.array(self.daily_returns)
        num_days = len(returns_array)

        # Calculate annualized return
        self.annualized_return = (self.final_cumulative_return ** (252 / num_days)) - 1

        # Calculate Sharpe ratio with proper annualization
        daily_mean = np.mean(returns_array)
        daily_std = np.std(returns_array) + 1e-6
        annualized_mean = daily_mean * 252
        annualized_std = daily_std * np.sqrt(252)
        self.sharpe_ratio = (annualized_mean - risk_free_rate) / annualized_std
        self.tracking_error = np.std(returns_array - benchmark_returns) * np.sqrt(252)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def plot_cumulative_returns(self, ax=None, show=True, save_path=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Convert dates to datetime if they're strings
        if isinstance(self.dates[0], str):
            dates = pd.to_datetime(self.dates)
        else:
            dates = self.dates

        ax.plot(dates, self.cumulative_returns, label=self.algorithm_name)
        ax.set_title("Cumulative Returns Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.grid(True)
        ax.legend()

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        return ax

    def plot_weights(self, tickers=None, show=True, save_path=None):
        weights = self.weights_history

        # Convert dates to datetime if they're strings
        if isinstance(self.dates[0], str):
            dates = pd.to_datetime(self.dates)
        else:
            dates = self.dates

        # Calculate mean weights and sort
        mean_weights = np.mean(weights, axis=0)
        sorted_indices = np.argsort(mean_weights)[::-1]  # Sort in descending order

        # Plot in order of mean weight
        for i in sorted_indices:
            if max(weights[:, i]) > 0.01:
                plt.plot(dates, weights[:, i], label=f"{tickers[i]}")

        plt.title(f"{self.algorithm_name} - Asset Weights")
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        return plt.gcf(), plt.gca()

    def __str__(self):
        return (
            f"\n{self.algorithm_name} Results ({self.start_date} to {self.end_date}):\n"
            f"  Cumulative Return: {self.final_cumulative_return:.2%}\n"
            f"  Annualized Return: {self.annualized_return:.2%}\n"
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"  Tracking Error: {self.tracking_error:.2%}\n"
            f"  Training Time: {self.training_time:.1f} seconds\n"
            f"  Backtest Time: {self.backtest_time:.1f} seconds"
        )


class EWPBacktest:
    def __init__(self):
        pass

    def train(self):
        pass

    def get_weights(self, X: np.ndarray):
        return np.ones(X.shape[1]) / X.shape[1]


RISK_FREE_RATE = 0.0524


def main():
    print(f"\nStarting backtest at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()

    data = pd.read_csv("data/return_df.csv", index_col="Date")
    data.index = pd.to_datetime(data.index)  # Convert index to datetime

    # Filter data to 2010-2020 period
    data = data["2010":"2020"]
    print(
        f"Filtered data to period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
    )

    # Data for training (LSTM only)
    train_len = int(0.6 * len(data))
    train_data = data[:train_len]

    # Data for validation (LSTM only)
    val_len = int(0.2 * len(data))
    val_data = data[train_len : train_len + val_len]

    # Data for training (TCN only)
    val_and_train_data = pd.concat([val_data, train_data])

    # Data for benchmarking (All algorithms)
    bench_len = int(0.2 * len(data))
    bench_data = data[train_len + val_len :]

    seq_len = 60
    future_days = 20
    bench_dataset = ReturnDataset(bench_data, seq_len)

    algorithms = {
        #        "GA": GABacktest(),
        # "PSO": PSOBacktest(),
        "TCN": TCNBacktest(train_data, val_data, seq_len, future_days),
        # "LSTM_PyOpt": LSTMPyOptBacktest(
        #     train_data,
        #     val_data,
        #     seq_len,
        #     future_days,
        #     params={
        #         "lr": 0.001,
        #         "hidden_dim": 64,
        #         "num_layers": 1,
        #         "dropout": 0.2,
        #         "weight_decay": 1e-5,
        #         "max_weight": 0.9,
        #         "min_weight": 0.01,
        #         "k_assets": 10,
        #         "risk_free_rate": RISK_FREE_RATE,
        #     },
        #     training_plot_filename="lstm_pyopt_training.png",
        # ),
        # "EWP": EWPBacktest(),
    }

    print("Starting a backtest with the following algorithms:")
    for algorithm in algorithms:
        print(f"  - {algorithm}")

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Initialize results objects
    results = {}
    for name in algorithms:
        results[name] = BacktestResults(
            name,
            bench_data.index[0].strftime("%Y-%m-%d"),
            bench_data.index[-1].strftime("%Y-%m-%d"),
        )

    # Train models
    for name, algorithm in algorithms.items():
        print(f"Training {name}...")
        train_start = time.time()
        algorithm.train()
        results[name].training_time = time.time() - train_start

    total_batches = len(bench_dataset)
    print(f"\nStarting backtest with {total_batches} batches to process")

    for batch_idx, (X, Y) in enumerate(bench_dataset):
        if batch_idx % 2 == 0:
            elapsed_time = time.time() - start_time
            progress = (batch_idx + 1) / total_batches * 100
            print(
                f"\nProcessing batch {batch_idx + 1}/{total_batches} ({progress:.1f}%)"
            )
            print(f"Time elapsed: {elapsed_time:.1f} seconds")
            print(
                f"Estimated time remaining: {(elapsed_time / (batch_idx + 1) * (total_batches - batch_idx - 1)):.1f} seconds"
            )

            # Print current cumulative returns
            for name, result in results.items():
                print(
                    f"{name} current cumulative return: {result.final_cumulative_return:.2%}"
                )

        current_date = bench_data.index[batch_idx + seq_len]
        for name, algorithm in algorithms.items():
            weights = algorithm.get_weights(X)
            portfolio_return = (weights * Y).sum()
            results[name].add_result(current_date, weights, portfolio_return)

    # Calculate final statistics
    for result in results.values():
        result.backtest_time = time.time() - start_time - result.training_time
        result.calculate_stats(
            # Use Equal Weight Portfolio as benchmark
            benchmark_returns=results["EWP"].daily_returns,
            risk_free_rate=RISK_FREE_RATE,
        )
        print(result)

        # Save results
        result.save(f"results/{result.algorithm_name}_results.pkl")

    total_time = time.time() - start_time
    print(f"\nBacktest completed in {total_time:.1f} seconds")


def plot_results(save_path=None):
    results = []
    for file in os.listdir("results"):
        if file.endswith("_results.pkl"):
            with open(f"results/{file}", "rb") as f:
                results.append(pickle.load(f))

    if not results:
        print("No results found in results directory")
        return

    # Plot cumulative returns
    fig, ax = plt.subplots(figsize=(12, 6))
    for result in results:
        print(result)
        result.plot_cumulative_returns(ax=ax, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    # Plot weights for each result
    for result in results:
        # Get tickers from the data
        data = pd.read_csv("data/return_df.csv")
        tickers = data.columns[1:].tolist()  # Skip the Date column

        result.plot_weights(
            tickers, save_path=f"results/{result.algorithm_name}_weights.png"
        )


def plot_predicted_vs_actual_returns(
    model_name="LSTM_PyOpt", ticker="AAPL", save_path=None
):
    """
    Plot predicted vs actual returns for a specific ticker.

    Args:
        model_name: Name of the model to use (default "LSTM_PyOpt")
        ticker: Ticker symbol to plot (default "AAPL")
        save_path: Path to save the plot (default None)
    """
    # Load data
    data = pd.read_csv("data/return_df.csv", index_col="Date")
    data.index = pd.to_datetime(data.index)

    # # Filter data to 2010-2020 period
    # data = data["2010":"2020"]

    # Check if ticker exists in the data
    if ticker not in data.columns:
        print(
            f"Ticker {ticker} not found in data. Available tickers: {data.columns.tolist()}"
        )
        return

    # Find ticker index
    ticker_idx = data.columns.get_loc(ticker)

    # Data split
    train_len = int(0.6 * len(data))
    val_len = int(0.2 * len(data))
    train_data = data[:train_len]
    val_data = data[train_len : train_len + val_len]
    test_data = data[train_len + val_len :]

    # Load the algorithms dictionary from main
    seq_len = 60
    future_days = 10

    # Create model with the same configuration as in main()
    if model_name == "LSTM_PyOpt":
        from lstm_pyopt import LSTMPyOptBacktest

        model = LSTMPyOptBacktest(
            train_data,
            val_data,
            seq_len,
            future_days,
            params={
                "lr": 0.005,
                "hidden_dim": 128,
                "num_layers": 3,
                "dropout": 0.2,
                "weight_decay": 1e-5,
                "max_weight": 0.9,
                "min_weight": 0.01,
                "k_assets": 10,
                "risk_free_rate": 0.0524,
            },
            training_plot_filename=None,
        )
    else:
        print(f"Model {model_name} not supported for prediction plotting")
        return

    # Train the model
    print(f"Training {model_name} model...")
    model.train()

    # Generate predictions
    actual_returns = []
    predicted_returns = []
    dates = []

    print(f"Generating predictions for {ticker}...")
    for i in range(len(test_data) - seq_len):
        # Get window of returns
        window = test_data.iloc[i : i + seq_len].values

        # Get actual return for the ticker on the next day
        actual_return = test_data.iloc[i + seq_len, ticker_idx]

        # Predict returns using the model
        prediction = model.predict_price(window)

        # Store the prediction for the specific ticker
        predicted_return = prediction[ticker_idx]

        # Store data for plotting
        actual_returns.append(actual_return)
        predicted_returns.append(predicted_return)
        dates.append(test_data.index[i + seq_len])

    # Create DataFrame for plotting
    results_df = pd.DataFrame(
        {"Date": dates, "Actual": actual_returns, "Predicted": predicted_returns}
    )

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(
        results_df["Date"], results_df["Actual"], label="Actual Returns", color="blue"
    )
    plt.plot(
        results_df["Date"],
        results_df["Predicted"],
        label="Predicted Returns",
        color="red",
    )
    plt.title(f"{ticker} - Actual vs Predicted Returns")
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.legend()
    plt.grid(True)

    # Format x-axis dates
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Calculate metrics
    mse = np.mean((np.array(actual_returns) - np.array(predicted_returns)) ** 2)
    mae = np.mean(np.abs(np.array(actual_returns) - np.array(predicted_returns)))
    correlation = np.corrcoef(actual_returns, predicted_returns)[0, 1]

    # Add metrics to the plot
    plt.figtext(
        0.15, 0.85, f"MSE: {mse:.6f}\nMAE: {mae:.6f}\nCorrelation: {correlation:.4f}"
    )

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    return results_df


if __name__ == "__main__":
    # Choose one:
    # 1. Run the main backtest
    main()
    plot_results()

    # 2. Plot predicted vs actual returns for AAPL
    # plot_predicted_vs_actual_returns(save_path="results/aapl_predicted_vs_actual.png")
