import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# STEP 0: RAW RETURNS DATA
# Rows = days
# Columns = stocks (AAPL, MSFT, GOOGL)
# ---------------------------------------------------

returns = np.array([
    [0.01,  0.008,  0.012],
    [-0.005, -0.004, -0.006],
    [0.004,  0.003,  0.005],
    [0.006,  0.005,  0.007],
    [-0.002, -0.001, -0.003],
    [0.003,  0.002,  0.004]
])

print("RAW RETURNS (rows = days, columns = stocks):\n")
print(returns)

# ---------------------------------------------------
# STEP 1: MEAN OF EACH STOCK
# Mean return = average of each column
# This represents the 'center' of the data cloud
# ---------------------------------------------------

mean_returns = returns.mean(axis=0)

print("\nMEAN RETURNS (per stock):")
print(mean_returns)

# ---------------------------------------------------
# STEP 2: MEAN-CENTERING
# Subtract mean from each return
# This gives deviation from average on each day
#
# Mathematically:
# deviation = return - mean
#
# PCA ALWAYS works on centered data
# ---------------------------------------------------

R = returns - mean_returns

print("\nMEAN-CENTERED RETURNS (deviations):\n")
print(R)

# ---------------------------------------------------
# STEP 3: COVARIANCE MATRIX
#
# Covariance between stock i and j:
#   = average of (deviation_i * deviation_j)
#
# Diagonal  -> variance of each stock
# Off-diag  -> how two stocks move together
# ---------------------------------------------------

cov = np.cov(R, rowvar=False)

print("\nCOVARIANCE MATRIX:\n")
print(cov)

# ---------------------------------------------------
# STEP 4: EIGENVALUES & EIGENVECTORS
#
# Eigenvalues:
#   - How much variance exists in a direction
#
# Eigenvectors:
#   - Which combination of stocks defines that direction
#
# Solves:
#   cov * v = lambda * v
# ---------------------------------------------------

eigenvalues, eigenvectors = np.linalg.eig(cov)

print("\nEIGENVALUES (variance along each principal direction):\n")
print(eigenvalues)

print("\nEIGENVECTORS (principal directions / factor loadings):\n")
print(eigenvectors)

# ---------------------------------------------------
# STEP 5: VISUALIZE EIGENVALUES
#
# Each bar = how much variance is explained
# by that principal component
#
# Tall bar -> important factor
# Small bar -> noise / minor effect
# ---------------------------------------------------

plt.figure()
plt.bar(range(len(eigenvalues)), eigenvalues)
plt.title("Eigenvalues (Variance Explained by Each Component)")
plt.xlabel("Principal Component Index")
plt.ylabel("Variance")
plt.show()
