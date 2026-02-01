import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PART 3: COVARIANCE MATRIX - COMPLETE CODE
# ============================================================

returns = np.array([
    [0.01,   0.008,  0.012],
    [-0.005, -0.004, -0.006],
    [0.004,  0.003,  0.005],
    [0.006,  0.005,  0.007],
    [-0.002, -0.001, -0.003],
    [0.003,  0.002,  0.004]
])

mean_returns = returns.mean(axis=0)
R = returns - mean_returns

print("="*60)
print("COVARIANCE MATRIX CALCULATION")
print("="*60)

# Manual calculation
n = R.shape[0]
cov_matrix = (R.T @ R) / (n - 1)

print("Covariance Matrix:")
print(cov_matrix)

# Correlation matrix
std_devs = np.sqrt(np.diag(cov_matrix))
correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Diagnostics
print("\n" + "="*60)
print("DIAGNOSTICS")
print("="*60)
print(f"Symmetric? {np.allclose(cov_matrix, cov_matrix.T)}")
eigenvalues_check = np.linalg.eigvals(cov_matrix)
print(f"Positive semi-definite? {np.all(eigenvalues_check >= -1e-10)}")
print(f"Eigenvalues: {eigenvalues_check}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(cov_matrix, annot=True, fmt='.2e', cmap='coolwarm',
            center=0, square=True,
            xticklabels=['Stock 1', 'Stock 2', 'Stock 3'],
            yticklabels=['Stock 1', 'Stock 2', 'Stock 3'],
            ax=axes[0], cbar_kws={'label': 'Covariance'})
axes[0].set_title('COVARIANCE MATRIX', fontsize=13, fontweight='bold')

sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, square=True,
            xticklabels=['Stock 1', 'Stock 2', 'Stock 3'],
            yticklabels=['Stock 1', 'Stock 2', 'Stock 3'],
            ax=axes[1], cbar_kws={'label': 'Correlation'})
axes[1].set_title('CORRELATION MATRIX', fontsize=13, fontweight='bold')

axes[2].scatter(R[:, 0], R[:, 1], s=150, c='blue', alpha=0.6, 
                edgecolors='black', linewidths=2)
axes[2].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[2].axvline(0, color='red', linestyle='--', alpha=0.5)
axes[2].set_xlabel('Stock 1 (deviation)', fontsize=11)
axes[2].set_ylabel('Stock 2 (deviation)', fontsize=11)
axes[2].set_title('Stock 1 vs Stock 2', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ PART 3 COMPLETE!")
print("Next: Eigendecomposition - finding the principal axes")