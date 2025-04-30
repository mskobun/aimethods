# Downloads data for all S&P 100 constituents (snp100.csv)
# regardless of how much historical data is available.
# Adapted from: https://github.com/hobinkwak/Portfolio-Optimization-Deep-Learning

import os
import json
import pandas as pd
import yfinance as yf


def get_stock_data(code, start, end):
    data = yf.download(code, start=start, end=end)["Close"].dropna()
    df = pd.DataFrame(data)
    df.columns = ["Close"]
    df["Close"] = df["Close"].interpolate(method="linear")
    return df


def stock_download(
    dic,
    start,
    end,
    download_dir="data/stocks/",
):
    os.makedirs(download_dir, exist_ok=True)
    stock_dict = {}
    for symbol in dic:
        try:
            symbol = symbol if symbol != "BRK.B" else "BRK-B"
            data = get_stock_data(symbol, start, end)
            if data is not None and len(data) > 0:
                data.to_csv(download_dir + f"{symbol}.csv")
                stock_dict[symbol] = dic[symbol]
                print(f"Successfully downloaded {symbol}")
            else:
                print(f"No data for {symbol}")
        except Exception as e:
            print(f"Error downloading {symbol}: {str(e)}")
    return stock_dict


def get_return_df(stock_dic, in_path="data/stocks/", out_path="data/"):
    for i, ticker in enumerate(stock_dic):
        stock = in_path + f"{ticker}.csv"
        stock_df = pd.read_csv(stock, index_col="Date")[["Close"]]
        if i == 0:
            return_df = stock_df / stock_df.shift(1) - 1
            return_df.columns = [ticker]
        else:
            return_df[ticker] = stock_df / stock_df.shift(1) - 1

    # Drop the first row which will be all NaNs due to the shift operation
    return_df = return_df.iloc[1:]

    # Fill companies which haven't existed yet with the mean of the returns for that day
    return_df = return_df.fillna(return_df.mean(axis=1), axis=0)

    # Any remaining NaNs likely mean *all* stocks were NaN on that day (e.g., market holiday)
    # We can fill these with 0, assuming no return on those days.
    return_df = return_df.fillna(0)

    return_df.to_csv(out_path + "return_df.csv")
    return return_df


if __name__ == "__main__":
    snp100 = pd.read_csv("data/constituents.csv")
    snp100.loc[snp100.Symbol == "BRK.B", "Symbol"] = "BRK-B"
    snp100 = {tup[2]: tup[1] for tup in snp100.values.tolist()}
    stock_pair = stock_download(
        snp100, start="2010-01-01", end="2024-12-31", download_dir="data/stocks/"
    )
    sp100 = yf.download("^OEX", "2010-01-01", "2024-12-31")
    sp100.to_csv("data/snp100_index.csv")
    json.dump(stock_pair, open("data/stock.json", "w", encoding="UTF-8"))
    stock_dict_sp = json.load(open("data/stock.json", "r", encoding="UTF8"))
    return_df = get_return_df(stock_dict_sp)
