import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define tickers (same as yours)
tickers = {
    "BLUESTARCO": "BLUESTARCO.NS",
    "CPSEETF": "CPSEETF.NS",
    "DIVOPPBEES": "DIVOPPBEES.NS",
    "KIRLOSBROS": "KIRLOSBROS.NS",
    "MIDCAPETF": "MIDCAPETF.NS",
    "SUZLON": "SUZLON.NS",
    "SWIGGY": "SWIGGY.NS"
}

# Step 1: Download adjusted closing prices for 6 months
# Math: We need time-series data for returns. Adjusted close accounts for dividends/splits.
data = yf.download(
    list(tickers.values()),
    period="6mo",
    interval="1d",
    auto_adjust=True,
    progress=False
)
prices = data["Close"]
prices.columns = list(tickers.keys())  
prices = prices.dropna()  

print("Step 1: Adjusted Closing Prices (first 5 rows):")
print(prices.head())
print("\nLast 5 rows:")
print(prices.tail())
print(f"\nShape: {prices.shape} (rows = trading days, columns = stocks)")

# Step 2: Compute daily returns
# Math: return_t = (price_t - price_{t-1}) / price_{t-1} for each stock.
returns = prices.pct_change().dropna()  

print("\nStep 2: Daily Returns (first 5 rows):")
print(returns.head())
print(f"\nShape: {returns.shape}")

# Step 3: Compute mean returns for each stock
# Math: mean_return_j = (1/T) * sum_{t=1 to T} return_{t,j} for stock j.
mean_returns = returns.mean()
print("\nStep 3: Mean Daily Returns for each stock:")
print(mean_returns)

# Step 4: Center the returns (subtract mean for each column)
# Math: centered_return_{t,j} = return_{t,j} - mean_return_j
# This shifts the "cloud" of data points to origin, so covariance is about spread, not location.
centered_returns = returns - mean_returns

print("\nStep 4: Centered Returns (mean-subtracted, first 5 rows):")
print(centered_returns.head())

# Step 5: Compute covariance matrix
# Math: cov_{j,k} = (1/(T-1)) * sum_{t=1 to T} (centered_return_{t,j} * centered_return_{t,k})
# Diagonals: variance (risk) of each stock. Off-diagonals: covariance (shared risk).
# Note: pandas.cov() uses unbiased estimator (divides by T-1).
cov = centered_returns.cov()  # Equivalent to (1/(T-1)) * centered_returns.T @ centered_returns

print("\nStep 5: Covariance Matrix (7x7 symmetric matrix):")
print(cov)

# Optional: Plot correlation matrix (derived from cov, but normalized)
corr = returns.corr()
print("\nCorrelation Matrix (for reference, cov normalized by std devs):")
print(corr)

plt.figure(figsize=(8,6))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Matrix of Portfolio Returns")
plt.tight_layout()
plt.show()

# Step 6: Eigen decomposition of covariance matrix
# Math: Solve cov * v = lambda * v for eigenvectors v and eigenvalues lambda.
# Eigenvectors: directions of max variance (PCs). Eigenvalues: amount of variance along each.
# We use np.linalg.eigh (for symmetric matrices, returns sorted ascending).
eigvals, eigvecs = np.linalg.eigh(cov)

# Sort descending (largest eigenvalue first)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

print("\nStep 6: Eigenvalues (variance explained by each PC, descending):")
print(eigvals)

print("\nEigenvectors (columns are PC1 to PC7 directions):")
print(pd.DataFrame(eigvecs, index=cov.index, columns=[f"PC{i+1}" for i in range(len(eigvals))]))

# Step 7: Explained variance ratios
# Math: explained_var_i = eigval_i / sum(eigvals)
# This shows % of total variance captured by each PC.
explained_var = eigvals / eigvals.sum()
cum_explained_var = np.cumsum(explained_var)  # Cumulative for seeing how many PCs needed

print("\nStep 7: Explained Variance Ratios (fraction of total variance):")
print(explained_var)
print("\nCumulative Explained Variance:")
print(cum_explained_var)

# Interpretation print (based on your question)
print("\nInterpretation of Explained Variance:")
print("If PC1 is much larger (e.g., >70%), most risk is shared (e.g., market factor)—diversification among these stocks is limited; hedge the common factor instead.")
print("If variance is distributed (e.g., top 3 PCs ~30% each), risks are more stock-specific—diversification works well to reduce portfolio risk.")
print("Here, check the numbers: PC1 captures {:.2%}, so [insert your conclusion based on run].".format(explained_var[0]))

plt.bar(range(1, len(explained_var) + 1), explained_var)
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.title("Explained Variance by PCA Factors")
plt.show()

# Step 8: Compute PC scores (transformed data)
# Math: scores = centered_returns @ eigvecs  (matrix multiplication)
# Each row: a day's returns projected onto the PCs.
# Columns: PC1 score, PC2 score, etc. for each day.
pc_scores = centered_returns @ eigvecs
pc_scores.columns = [f"PC{i+1}_Score" for i in range(len(eigvals))]

print("\nStep 8: PC Scores (first 5 rows):")
print(pc_scores.head())
print(f"\nShape: {pc_scores.shape} (same as centered_returns)")

# Optional: Compare PC scores (e.g., plot PC1 vs PC2 for visualization of data in PC space)
plt.figure(figsize=(8,6))
plt.scatter(pc_scores.iloc[:, 0], pc_scores.iloc[:, 1])
plt.xlabel("PC1 Scores")
plt.ylabel("PC2 Scores")
plt.title("Data Projected onto PC1 and PC2")
plt.show()

# Bonus: To hedge/diversify, you could use PCs—e.g., if PC1 is market, regress portfolio on it and hedge accordingly.