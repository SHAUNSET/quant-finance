import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# PART 4: EIGENDECOMPOSITION - COMPLETE CODE
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
cov_matrix = (R.T @ R) / (R.shape[0] - 1)

print("="*60)
print("EIGENDECOMPOSITION OF COVARIANCE MATRIX")
print("="*60)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Variance explained
total_variance = eigenvalues.sum()
variance_explained = eigenvalues / total_variance
cumulative_variance = np.cumsum(variance_explained)

print("\nEigenvalues:")
for i, val in enumerate(eigenvalues):
    print(f"  λ{i+1}: {val:.6e} ({variance_explained[i]*100:.2f}%)")

print("\nEigenvectors (loadings):")
print(eigenvectors)

print("\nCumulative variance:")
for i in range(len(eigenvalues)):
    print(f"  PC1-PC{i+1}: {cumulative_variance[i]*100:.2f}%")

# Verify properties
print("\n" + "="*60)
print("VERIFICATION")
print("="*60)

# Orthonormality
VtV = eigenvectors.T @ eigenvectors
print(f"Orthonormality error: {np.linalg.norm(VtV - np.eye(3)):.10e}")

# Reconstruction
reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
print(f"Reconstruction error: {np.linalg.norm(cov_matrix - reconstructed):.10e}")

print("\n✅ PART 4 COMPLETE!")
print(f"\nKEY FINDING: {variance_explained[0]*100:.1f}% of variance in 1 factor!")
print("Next: Project data onto principal components")