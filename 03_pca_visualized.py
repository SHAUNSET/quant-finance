import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# 1. RAW RETURNS (rows = days, columns = stocks)
# ============================================================

returns = np.array([
    [0.01,  0.008,  0.012],
    [-0.005, -0.004, -0.006],
    [0.004, 0.003, 0.005],
    [0.006, 0.005, 0.007],
    [-0.002, -0.001, -0.003],
    [0.003, 0.002, 0.004]
])

print("\nRAW RETURNS:\n", returns)

# --- Plot raw return cloud ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(returns[:,0], returns[:,1], returns[:,2], s=100, c='blue', alpha=0.6)
ax.set_title("RAW RETURNS CLOUD", fontsize=14)
ax.set_xlabel("Stock 1")
ax.set_ylabel("Stock 2")
ax.set_zlabel("Stock 3")
plt.show()


# ============================================================
# 2. MEAN CENTERING (remove average return)
# ============================================================

mean_returns = returns.mean(axis=0)
R = returns - mean_returns

print("\nMEAN RETURNS:\n", mean_returns)
print("\nMEAN-CENTERED RETURNS:\n", R)

# --- Plot centered cloud ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(R[:,0], R[:,1], R[:,2], s=100, c='green', alpha=0.6)
ax.set_title("MEAN-CENTERED RETURNS CLOUD", fontsize=14)
ax.set_xlabel("Stock 1 Deviation")
ax.set_ylabel("Stock 2 Deviation")
ax.set_zlabel("Stock 3 Deviation")
plt.show()


# ============================================================
# 3. COVARIANCE MATRIX
# Cov(i,j) = average of (Ri * Rj)
# ============================================================

# Manual calculation since R is already centered
cov = (R.T @ R) / (R.shape[0] - 1)
print("\nCOVARIANCE MATRIX:\n", cov)

# Verify with np.cov on original data
cov_verify = np.cov(returns.T)
print("\nVERIFICATION (should match):\n", cov_verify)
print("\nMax difference:", np.max(np.abs(cov - cov_verify)))


# ============================================================
# 4. EIGEN DECOMPOSITION
# ============================================================

eigenvalues, eigenvectors = np.linalg.eig(cov)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nEIGENVALUES (variance along directions):\n", eigenvalues)
print("\nEIGENVECTORS (columns = directions):\n", eigenvectors)


# ============================================================
# 5. VISUALIZE EIGENVECTORS ON DATA CLOUD
# ============================================================

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(R[:,0], R[:,1], R[:,2], s=150, c='blue', alpha=0.6, label='Data points', edgecolors='black')

colors = ['red', 'green', 'orange']
labels = ['PC1', 'PC2', 'PC3']

# Draw eigenvectors as lines through origin (more visible than arrows)
line_length = 0.006  # Adjust this to match your data scale

for i in range(3):
    # Direction scaled by sqrt(eigenvalue)
    direction = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * line_length
    
    # Draw line in both directions from origin
    ax.plot([-direction[0], direction[0]], 
            [-direction[1], direction[1]], 
            [-direction[2], direction[2]], 
            color=colors[i], 
            linewidth=5, 
            label=f'{labels[i]} ({eigenvalues[i]/eigenvalues.sum():.1%} var)',
            alpha=0.9)
    
    # Add arrowheads at the ends
    ax.quiver(0, 0, 0,
              direction[0], direction[1], direction[2],
              color=colors[i], arrow_length_ratio=0.3, linewidth=3, alpha=0.7)
    ax.quiver(0, 0, 0,
              -direction[0], -direction[1], -direction[2],
              color=colors[i], arrow_length_ratio=0.3, linewidth=3, alpha=0.7)

ax.set_title("EIGENVECTORS = PRINCIPAL AXES", fontsize=16, fontweight='bold')
ax.set_xlabel("Stock 1", fontsize=12)
ax.set_ylabel("Stock 2", fontsize=12)
ax.set_zlabel("Stock 3", fontsize=12)
ax.legend(loc='upper left', fontsize=11)

# Equal aspect ratio to see directions clearly
max_range = np.max(np.abs(R)) * 1.3
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

# Add grid
ax.grid(True, alpha=0.3)

plt.show()


# ============================================================
# 6. VARIANCE EXPLAINED
# ============================================================

variance_explained = eigenvalues / eigenvalues.sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
ax1.bar([1, 2, 3], variance_explained, color=['red', 'green', 'orange'], alpha=0.7, edgecolor='black')
ax1.set_title("VARIANCE EXPLAINED BY EACH PC", fontsize=14, fontweight='bold')
ax1.set_xlabel("Principal Component", fontsize=12)
ax1.set_ylabel("Proportion of Variance", fontsize=12)
ax1.set_xticks([1, 2, 3])
ax1.set_ylim([0, 1.1])
ax1.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(variance_explained):
    ax1.text(i+1, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Cumulative variance
cumulative = np.cumsum(variance_explained)
ax2.plot([1, 2, 3], cumulative, 'o-', linewidth=3, markersize=10, color='darkblue')
ax2.fill_between([1, 2, 3], 0, cumulative, alpha=0.3, color='blue')
ax2.set_title("CUMULATIVE VARIANCE EXPLAINED", fontsize=14, fontweight='bold')
ax2.set_xlabel("Number of Components", fontsize=12)
ax2.set_ylabel("Cumulative Variance", fontsize=12)
ax2.set_xticks([1, 2, 3])
ax2.set_ylim([0, 1.1])
ax2.grid(True, alpha=0.3)
for i, v in enumerate(cumulative):
    ax2.text(i+1, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print("\nVARIANCE EXPLAINED:")
for i, v in enumerate(variance_explained):
    print(f"  PC{i+1}: {v:.2%}")
print(f"  Cumulative PC1: {cumulative[0]:.2%}")
print(f"  Cumulative PC1+PC2: {cumulative[1]:.2%}")
print(f"  Cumulative PC1+PC2+PC3: {cumulative[2]:.2%}")


# ============================================================
# 7. PROJECT DATA ON FIRST PRINCIPAL COMPONENT (FACTOR RETURNS)
# ============================================================

pc1 = eigenvectors[:, 0]  # Already sorted, so index 0 is largest
factor_returns = R @ pc1

print("\nFIRST PRINCIPAL COMPONENT (FACTOR LOADINGS):\n", pc1)
print("\nFACTOR RETURNS (projection onto PC1):\n", factor_returns)

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(factor_returns)+1), factor_returns, marker="o", linewidth=2, 
         markersize=10, color='darkred', label='Factor Returns')
plt.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
plt.fill_between(range(1, len(factor_returns)+1), 0, factor_returns, alpha=0.3, color='red')
plt.title("FACTOR RETURNS (PC1) - The 'Market Factor'", fontsize=14, fontweight='bold')
plt.xlabel("Day", fontsize=12)
plt.ylabel("Factor Value", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xticks(range(1, len(factor_returns)+1))
plt.tight_layout()
plt.show()


# ============================================================
# 8. BONUS: PROJECT ALL DATA ONTO PRINCIPAL COMPONENTS
# ============================================================

# Transform all data to PC space
all_pcs = R @ eigenvectors

print("\nALL PRINCIPAL COMPONENT SCORES:")
print("(Each row = day, each column = PC score)\n", all_pcs)

# Plot first 2 PCs
plt.figure(figsize=(10, 8))
plt.scatter(all_pcs[:, 0], all_pcs[:, 1], s=150, c='purple', alpha=0.6, edgecolors='black')
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(0, color='k', linestyle='--', alpha=0.3)
plt.xlabel(f'PC1 ({variance_explained[0]:.1%} variance)', fontsize=12)
plt.ylabel(f'PC2 ({variance_explained[1]:.1%} variance)', fontsize=12)
plt.title('DATA PROJECTED ONTO FIRST 2 PRINCIPAL COMPONENTS', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Label each point with day number
for i, (x, y) in enumerate(zip(all_pcs[:, 0], all_pcs[:, 1])):
    plt.annotate(f'Day {i+1}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)

plt.tight_layout()
plt.show()


print("\n" + "="*60)
print("PCA COMPLETE!")
print("="*60)
print(f"The first PC explains {variance_explained[0]:.1%} of total variance")
print(f"Factor loadings (weights): Stock1={pc1[0]:.3f}, Stock2={pc1[1]:.3f}, Stock3={pc1[2]:.3f}")
print(f"\nInterpretation: All stocks move together (positive loadings)")
print(f"This PC1 represents a 'market factor' - when it's positive, all stocks tend to go up")
print("="*60)