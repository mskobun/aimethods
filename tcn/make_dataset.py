import json
import pickle
import numpy as np
import pandas as pd

def make_DL_dataset(data, data_len, n_stock):
    times = []
    dataset = np.array(data.iloc[:data_len, :]).reshape((1, -1, n_stock))
    times.append(data.iloc[:data_len, :].index)

    for i in range(1, len(data) - data_len + 1):
        addition = np.array(data.iloc[i : data_len + i, :]).reshape(1, -1, n_stock)
        dataset = np.concatenate((dataset, addition))
        times.append(data.iloc[i : data_len + i, :].index)
    return dataset, times


def data_split(data, train_len, pred_len, tr_ratio, n_stock):
    return_train, times_train = make_DL_dataset(
        data[: int(len(data) * tr_ratio)], train_len + pred_len, n_stock
    )
    return_test, times_test = make_DL_dataset(
        data[int(len(data) * tr_ratio) :], train_len + pred_len, n_stock
    )

    x_tr = np.array([x[:train_len] for x in return_train])
    y_tr = np.array([x[-pred_len:] for x in return_train])
    times_tr = np.unique(
        np.array([x[-pred_len:] for x in times_train]).flatten()
    ).tolist()

    x_te = np.array([x[:train_len] for x in return_test])
    y_te = np.array([x[-pred_len:] for x in return_test])
    times_te = np.unique(
        np.array([x[-pred_len:] for x in times_test]).flatten()
    ).tolist()

    return x_tr, y_tr, x_te, y_te, times_tr, times_te

def main_aim():
    config = json.load(open("config/config.json", "r", encoding="utf8"))
    stock_df = pd.read_csv("data/aim_dataset.csv", index_col="Date")
    
    # Use the actual number of columns in the dataframe
    n_stock = len(stock_df.columns)
    
    # The aim_dataset.csv already contains simple returns, so we use it directly
    x_tr, y_tr, x_te, y_te, times_tr, times_te = data_split(
        stock_df,
        config["TRAIN_LEN"],
        config["PRED_LEN"],
        config["TRAIN_RATIO"],
        n_stock  # Use the actual number of columns
    )

    print(f"Data ranges - X_train: [{x_tr.min():.6f}, {x_tr.max():.6f}], Y_train: [{y_tr.min():.6f}, {y_tr.max():.6f}]")
    
    # Save test dates to pickle file
    with open("data/date.pkl", "wb") as f:
        pickle.dump(times_te, f)
    
    # Save dataset to pickle file
    with open("data/dataset.pkl", "wb") as f:
        pickle.dump([x_tr, y_tr, x_te, y_te], f)

if __name__ == "__main__":
    main_aim()












