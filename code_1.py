import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# PART 1: RAW RETURNS - FOUNDATION OF PCA
# ============================================================

# Sample data: 6 days, 3 stocks
returns = np.array([
    [0.01,   0.008,  0.012],   # Day 1
    [-0.005, -0.004, -0.006],  # Day 2
    [0.004,  0.003,  0.005],   # Day 3
    [0.006,  0.005,  0.007],   # Day 4
    [-0.002, -0.001, -0.003],  # Day 5
    [0.003,  0.002,  0.004]    # Day 6
])

print("="*60)
print("RAW RETURNS MATRIX")
print("="*60)
print("Shape:", returns.shape)
print("(Rows=Days, Columns=Stocks)")
print("\n", returns)

# Diagnostic checks
print("\n" + "="*60)
print("DATA QUALITY CHECKS")
print("="*60)
print(f"Missing values: {np.isnan(returns).sum()}")
print(f"Min return: {returns.min():.4f} ({returns.min()*100:.2f}%)")
print(f"Max return: {returns.max():.4f} ({returns.max()*100:.2f}%)")

stock_variances = returns.var(axis=0)
print(f"\nStock variances: {stock_variances}")

print("\nPer-Stock Statistics:")
for i in range(returns.shape[1]):
    print(f"Stock {i+1}: Mean={returns[:,i].mean():.4f}, Std={returns[:,i].std():.4f}")
print("="*60)

# Visualize
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    returns[:, 0], returns[:, 1], returns[:, 2],
    s=200, c='blue', alpha=0.7, edgecolors='black', linewidths=2
)

for i in range(len(returns)):
    ax.text(
        returns[i, 0], returns[i, 1], returns[i, 2], 
        f'  Day {i+1}', fontsize=10
    )

ax.set_xlabel('Stock 1 Return', fontsize=12, fontweight='bold')
ax.set_ylabel('Stock 2 Return', fontsize=12, fontweight='bold')
ax.set_zlabel('Stock 3 Return', fontsize=12, fontweight='bold')
ax.set_title('RAW RETURNS CLOUD\n(Each point = one day)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

