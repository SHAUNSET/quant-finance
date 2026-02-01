import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PART 5: PROJECTION & FACTOR RETURNS - COMPLETE CODE
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

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("="*60)
print("PROJECT DATA ONTO PRINCIPAL COMPONENTS")
print("="*60)

# Project
PC_scores = R @ eigenvectors

print("\nPC Scores:")
print(PC_scores)

print("\nPC1 (Market Factor):")
for i in range(len(PC_scores)):
    print(f"  Day {i+1}: {PC_scores[i, 0]:+.6f}")

# Verify reconstruction
R_reconstructed = PC_scores @ eigenvectors.T
reconstruction_error = np.linalg.norm(R - R_reconstructed)
print(f"\nReconstruction error: {reconstruction_error:.10e}")

# Approximate with PC1 only
R_approx = PC_scores[:, 0:1] @ eigenvectors[:, 0:1].T
approx_quality = eigenvalues[0] / eigenvalues.sum()
print(f"\nPC1 alone captures: {approx_quality*100:.2f}%")

print("\n✅ PART 5 COMPLETE!")
print("\n🎉 YOU NOW UNDERSTAND PCA FROM GROUND UP!")
