import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tickers = {
    "BLUESTARCO": "BLUESTARCO.NS",
    "CPSEETF": "CPSEETF.NS",
    "DIVOPPBEES": "DIVOPPBEES.NS",
    "KIRLOSBROS": "KIRLOSBROS.NS",
    "MIDCAPETF": "MIDCAPETF.NS",
    "SUZLON": "SUZLON.NS",
    "SWIGGY": "SWIGGY.NS"
}

data = yf.download(
    list(tickers.values()),
    period="6mo",
    interval="1d",
    auto_adjust=True,
    progress=False
)

prices = data["Close"]
prices.columns = tickers.keys()
prices = prices.dropna()

#CLOSED PRICES
print(prices.head())
print(prices.tail())
print(prices.shape)


#RETURNS
returns = prices.pct_change().dropna()

print(returns.head())
print(returns.shape)

#MEAN RETURNS
mean_returns = returns.mean()
print(mean_returns)

#CENTER DATA
centered_returns = returns - mean_returns
print(centered_returns.head())

#COVARIANCE MATRIX
cov_matrix = centered_returns.cov()
print(cov_matrix)


#EIGEN VALUE AND VECTOR
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

idx = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues:")
print(eigenvalues)

print("Eigenvectors")
print(eigenvectors)

pc1_loadings = pd.Series(
    eigenvectors[:, 0],
    index=prices.columns
).sort_values(key=np.abs, ascending=False)

print(pc1_loadings)

#PC SCORES

pc_scores = centered_returns @ eigenvectors
pc_scores.columns = [f"PC{i+1}" for i in range(pc_scores.shape[1])]
print(pc_scores.head())



# Download NIFTY data
nifty = yf.download(
    "^NSEI",
    period="6mo",
    interval="1d",
    auto_adjust=True,
    progress=False
)


nifty_close = nifty["Close"].squeeze()


nifty_returns = nifty_close.pct_change().dropna()


common_dates = pc_scores.index.intersection(nifty_returns.index)


pc1_series = pc_scores.loc[common_dates, "PC1"].squeeze()
nifty_returns = nifty_returns.loc[common_dates].squeeze()


correlation = pc1_series.corr(nifty_returns)

print("Correlation between PC1 and NIFTY:", correlation)

