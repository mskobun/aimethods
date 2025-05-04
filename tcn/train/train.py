import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model.tcn import TCN
from model.transformer import Transformer
from model.sam import SAM
from model.loss import max_sharpe
from train.utils import save_model, load_model


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if self.config["USE_CUDA"] else "cpu"
        model_name = self.config["MODEL"]
        if model_name.lower() == "tcn":
            hidden_size, level = 5, 3
            num_channels = [hidden_size] * (level - 1) + [
                self.config["TCN"]["n_timestep"]
            ]
            self.model = TCN(
                self.config["TCN"]["n_feature"],
                self.config["TCN"]["n_output"],
                num_channels,
                self.config["TCN"]["kernel_size"],
                self.config["TCN"]["n_dropout"],
                self.config["TCN"]["n_timestep"],
                self.config["LB"],
                self.config["UB"],
                self.config["K"],
            ).to(self.device)
        elif model_name.lower() == "transformer":
            self.model = Transformer(
                self.config["TRANSFORMER"]["n_feature"],
                self.config["TRANSFORMER"]["n_timestep"],
                self.config["TRANSFORMER"]["n_layer"],
                self.config["TRANSFORMER"]["n_head"],
                self.config["TRANSFORMER"]["n_dropout"],
                self.config["TRANSFORMER"]["n_output"],
                self.config["LB"],
                self.config["UB"],
            ).to(self.device)
        base_optimizer = torch.optim.SGD
        if self.config["OPTIMISER"] == "SAM":
            self.optimizer = SAM(
                self.model.parameters(),
                base_optimizer,
                lr=self.config["LR"],
                momentum=self.config["MOMENTUM"],
            )
        if self.config["OPTIMISER"] == "ADAM":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["LR"] * 5.0,
                weight_decay=self.config.get("WEIGHT_DECAY", 1e-4),
            )
        self.criterion = max_sharpe

        if (1.0 / self.config["K"]) < self.config["LB"]:
            raise Exception("K is not high enough to be compatible with the LB.")

    def _dataload(self):
        with open("data/dataset.pkl", "rb") as f:
            train_x_raw, train_y_raw, test_x_raw, test_y_raw = pickle.load(f)

        with open("data/date.pkl", "rb") as f:
            test_date = pickle.load(f)

        self.train_x_raw = train_x_raw
        self.train_y_raw = train_y_raw
        self.test_x_raw = test_x_raw
        self.test_y_raw = test_y_raw
        self.test_date = test_date

    def _scale_data(self, scale=10):
        self.train_x = torch.from_numpy(self.train_x_raw.astype("float32") * scale)
        self.train_y = torch.from_numpy(self.train_y_raw.astype("float32") * scale)
        self.test_x = torch.from_numpy(self.test_x_raw.astype("float32") * scale)
        self.test_y = torch.from_numpy(self.test_y_raw.astype("float32") * scale)

    def _set_parameter(self):
        self.LEN_TRAIN = self.train_x.shape[1]
        self.LEN_PRED = self.train_y.shape[1]
        self.N_STOCK = self.config["N_FEAT"]

    def _shuffle_data(self):
        randomized = np.arange(len(self.train_x))
        np.random.shuffle(randomized)
        self.train_x = self.train_x[randomized]
        self.train_y = self.train_y[randomized]

    def set_data(self):
        self._dataload()
        self._scale_data()
        self._set_parameter()
        self._shuffle_data()

    def set_backtest_data(self, x_tr, y_tr, x_te, y_te, times_test):
        self.train_x_raw = x_tr
        self.train_y_raw = y_tr
        self.test_x_raw = x_te
        self.test_y_raw = y_te
        self.test_date = times_test
        self._scale_data()
        self._set_parameter()
        self._shuffle_data()

    def dataloader(self, x, y):
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config["BATCH"],
            shuffle=False,
            drop_last=True,
        )

        return loader

    def train(self, visualize=True):
        train_loader = self.dataloader(self.train_x, self.train_y)
        test_loader = self.dataloader(self.test_x, self.test_y)

        valid_loss = []
        train_loss = []

        early_stop_count = 0
        early_stop_th = self.config["EARLY_STOP"]

        for epoch in range(self.config["EPOCHS"]):
            print(f"Epoch {epoch + 1}/{self.config['EPOCHS']}")
            print("-" * 10)
            for phase in ["train", "valid"]:
                if phase == "train":
                    self.model.train()
                    dataloader = train_loader
                else:
                    self.model.eval()
                    dataloader = test_loader

                running_loss = 0.0

                for idx, data in enumerate(dataloader):
                    x, y = data
                    x = x.to(self.device)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        out = self.model(x)
                        loss = self.criterion(
                            y,
                            out,
                            rf=self.config["RF"],
                            device=self.device,
                            lb=self.config["LB"],
                            ub=self.config["UB"],
                            K=self.config["K"],
                        )
                        if phase == "train":
                            loss.backward()

                            if self.config["OPTIMISER"] == "SAM":
                                self.optimizer.first_step(zero_grad=True)
                                self.criterion(
                                    y,
                                    self.model(x),
                                    rf=self.config["RF"],
                                    device=self.device,
                                    lb=self.config["LB"],
                                    ub=self.config["UB"],
                                    K=self.config["K"],
                                ).backward()
                                self.optimizer.second_step(zero_grad=True)

                            if self.config["OPTIMISER"] == "ADAM":
                                self.optimizer.step()

                    running_loss += loss.item() / len(dataloader)
                if phase == "train":
                    train_loss.append(running_loss)
                else:
                    valid_loss.append(running_loss)
                    print(f"running loss: {running_loss}")
                    print(f"valid loss: {min(valid_loss)}")
                    if running_loss <= min(valid_loss):
                        early_stop_count = 0
                        save_model(
                            self.model,
                            self.config.get(
                                "MODEL_PATH", "result/best_model_weight_hb.pt"
                            ),
                        )
                        print(f"Improved! Epoch {epoch + 1}, loss: {running_loss}")

                    else:
                        early_stop_count += 1

            if early_stop_count == early_stop_th:
                break

        if visualize:
            self._visualize_training(train_loss, valid_loss)

        return self.model, train_loss, valid_loss

    def _visualize_training(self, train_loss, valid_loss):
        plt.plot(train_loss, label="train")
        plt.plot(valid_loss, label="valid")
        plt.legend()
        plt.show()

    def backtest(self, visualize=True):
        self.model = load_model(
            self.model,
            self.config.get("MODEL_PATH", "result/best_model_weight_hb.pt"),
            use_cuda=self.config["USE_CUDA"],
        )

        myPortfolio, equalPortfolio = [10000], [10000]
        EWPWeights = np.ones(self.N_STOCK) / self.N_STOCK
        myWeights = []
        daily_returns_my = []
        daily_returns_ewp = []

        # Risk-free rate (annual)
        annual_rf_rate = self.config["RF"]
        daily_rf_rate = annual_rf_rate / 252

        for i in range(0, self.test_x.shape[0], self.LEN_PRED):
            x = self.test_x[i][np.newaxis, :, :]
            out = self.model(x.float().to(self.device))[0]
            myWeights.append(out.detach().cpu().numpy())

            # Calculate returns properly for the rebalancing period
            end_idx = min(i + self.LEN_PRED, len(self.test_y_raw))

            # Calculate returns for each day in the period
            period_returns_my = []
            period_returns_ewp = []

            for j in range(i, end_idx):
                # Check the structure of test_y_raw
                if len(self.test_y_raw[j].shape) > 1:
                    # If test_y_raw[j] is 2D with shape (21, 101), we need to handle this
                    # Option 1: Use the first day's returns
                    daily_return = self.test_y_raw[j][
                        0
                    ]  # Take first row if it's multiple days

                    # Option 2: If these are actually returns for different assets
                    # Reshape if needed - uncomment the appropriate line
                    # daily_return = self.test_y_raw[j].reshape(-1)  # Flatten if it's a different structure
                    # daily_return = self.test_y_raw[j].mean(axis=0)  # Average if these are multiple predictions
                else:
                    # If test_y_raw[j] is already 1D (shape (101,)), use as is
                    daily_return = self.test_y_raw[j]

                # Now daily_return should be shape (101,)
                # Calculate my portfolio's daily return
                my_daily_return = np.dot(out.detach().cpu().numpy(), daily_return)
                period_returns_my.append(my_daily_return)
                daily_returns_my.append(my_daily_return)

                # Calculate equal weight portfolio daily return
                ewp_daily_return = np.dot(EWPWeights, daily_return)
                period_returns_ewp.append(ewp_daily_return)
                daily_returns_ewp.append(ewp_daily_return)

            # Compound the returns for the period properly
            my_period_return = np.prod(1 + np.array(period_returns_my)) - 1
            ewp_period_return = np.prod(1 + np.array(period_returns_ewp)) - 1

            # Update portfolio values
            myPortfolio.append(myPortfolio[-1] * (1 + my_period_return))
            equalPortfolio.append(equalPortfolio[-1] * (1 + ewp_period_return))

        # Create performance dataframe
        idx = np.arange(0, len(self.test_date), self.LEN_PRED)
        performance = pd.DataFrame(
            {"EWP": equalPortfolio, "MyPortfolio": myPortfolio},
            index=np.array(self.test_date)[idx],
        )
        performance.to_csv("result/backtest.csv")

        if visualize:
            self._visualize_backtest(performance)
            self._visualize_weights(performance, myWeights)

        # Calculate period-to-period returns
        result = performance.copy()
        result["EWP_Return"] = result["EWP"] / result["EWP"].shift(1) - 1
        result["My_Return"] = result["MyPortfolio"] / result["MyPortfolio"].shift(1) - 1
        result = result.dropna()

        # Calculate MDD
        print("MDD")
        mdd_df = result[["EWP", "MyPortfolio"]].apply(self._get_mdd)
        print(mdd_df)

        # Print detailed weight information
        print("\nDetailed weights analysis:")
        ticker = pd.read_csv("data/aim_dataset.csv", index_col=0).columns
        weights_array = np.array(myWeights)

        print("Final Weights:")
        final = weights_array[-1]
        non_zero_indices = np.where(final > 0.001)[0]
        if len(non_zero_indices) > 0:
            for idx in non_zero_indices:
                print(f"  {ticker[idx]}: {final[idx]:.4f}")
            print(f"  Sum of weights: {final.sum():.4f}")

        # Convert daily returns to numpy arrays for calculations
        daily_returns_my = np.array(daily_returns_my)
        daily_returns_ewp = np.array(daily_returns_ewp)

        # Calculate performance metrics using daily returns
        # (more accurate than using rebalancing period returns)

        # 1. Excess returns over risk-free rate
        excess_returns_my = daily_returns_my - daily_rf_rate
        excess_returns_ewp = daily_returns_ewp - daily_rf_rate

        # 2. Annualized return
        annual_return_my = np.mean(daily_returns_my) * 252
        annual_return_ewp = np.mean(daily_returns_ewp) * 252

        # 3. Annualized volatility
        annual_vol_my = np.std(daily_returns_my) * np.sqrt(252)
        annual_vol_ewp = np.std(daily_returns_ewp) * np.sqrt(252)

        # 4. Sharpe ratio
        sharpe_my = (
            np.mean(excess_returns_my) * 252 / (np.std(daily_returns_my) * np.sqrt(252))
        )
        sharpe_ewp = (
            np.mean(excess_returns_ewp)
            * 252
            / (np.std(daily_returns_ewp) * np.sqrt(252))
        )

        # Print portfolio performance metrics
        print("\nPerformance Metrics (Daily Returns):")
        print("-" * 40)
        print("Annualized Return of Portfolios")
        print(f"Equal Weight: {annual_return_ewp:.4f}")
        print(f"My Portfolio: {annual_return_my:.4f}")
        print("-" * 40)

        print("Annualized Volatility of Portfolios")
        print(f"Equal Weight: {annual_vol_ewp:.4f}")
        print(f"My Portfolio: {annual_vol_my:.4f}")
        print("-" * 40)

        print(f"Annualized Sharpe Ratio of Portfolios (Rf = {annual_rf_rate:.2f}%)")
        print(f"Equal Weight: {sharpe_ewp:.4f}")
        print(f"My Portfolio: {sharpe_my:.4f}")
        print("-" * 40)

        return result, myWeights

    def _visualize_backtest(self, performance):
        performance.plot(figsize=(14, 7), fontsize=10)
        plt.legend(fontsize=10)
        plt.savefig("result/performance.png")
        plt.show()

    def _visualize_weights(self, performance, weights):
        weights = np.array(weights)
        ticker = pd.read_csv("data/aim_dataset.csv", index_col=0).columns
        n = self.N_STOCK
        plt.figure(figsize=(15, 10))
        for i in range(n):
            plt.plot(weights[:, i], label=ticker[i])
        plt.title("Weights")
        plt.xticks(
            np.arange(0, len(list(performance.index[1:]))),
            list(performance.index[1:]),
            rotation="vertical",
        )
        plt.legend()
        plt.savefig("result/weights.png")
        plt.show()

    def _get_mdd(self, x):
        arr_v = np.array(x)
        peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
        peak_upper = np.argmax(arr_v[:peak_lower])
        return (
            x.index[peak_upper],
            x.index[peak_lower],
            (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper],
        )
