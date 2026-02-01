import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# PART 2: MEAN CENTERING - COMPLETE CODE
# ============================================================

returns = np.array([
    [0.01,   0.008,  0.012],
    [-0.005, -0.004, -0.006],
    [0.004,  0.003,  0.005],
    [0.006,  0.005,  0.007],
    [-0.002, -0.001, -0.003],
    [0.003,  0.002,  0.004]
])

print("="*60)
print("STEP 1: CALCULATE MEAN RETURNS")
print("="*60)

mean_returns = returns.mean(axis=0)
print("Mean returns per stock:")
for i in range(len(mean_returns)):
    print(f"Stock {i+1}: {mean_returns[i]:.6f} ({mean_returns[i]*100:.4f}%)")

print("\n" + "="*60)
print("STEP 2: CENTER THE DATA")
print("="*60)

R = returns - mean_returns
print("Centered returns:")
print(R)

print("\n" + "="*60)
print("STEP 3: VERIFY CENTERING")
print("="*60)

centered_means = R.mean(axis=0)
print("Mean of centered data:")
for i in range(len(centered_means)):
    print(f"Stock {i+1}: {centered_means[i]:.10f}")

if np.allclose(centered_means, 0):
    print("\n✅ Centering successful!")

# Verify variance is preserved
print("\nVariance check:")
print(f"Original:  {returns.var(axis=0)}")
print(f"Centered:  {R.var(axis=0)}")
print(f"Preserved? {np.allclose(returns.var(axis=0), R.var(axis=0))}")

# Visualize
fig = plt.figure(figsize=(16, 7))

# Before centering
ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(returns[:, 0], returns[:, 1], returns[:, 2],
            s=200, c='blue', alpha=0.7, edgecolors='black', linewidths=2)
ax1.scatter([mean_returns[0]], [mean_returns[1]], [mean_returns[2]],
            s=300, c='red', marker='X', edgecolors='black', linewidths=2)
for i in range(len(returns)):
    ax1.text(returns[i, 0], returns[i, 1], returns[i, 2], f'  D{i+1}', fontsize=9)
ax1.set_title('BEFORE CENTERING', fontsize=13, fontweight='bold')
ax1.set_xlabel('Stock 1', fontsize=11)
ax1.set_ylabel('Stock 2', fontsize=11)
ax1.set_zlabel('Stock 3', fontsize=11)
ax1.grid(True, alpha=0.3)

# After centering
ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(R[:, 0], R[:, 1], R[:, 2],
            s=200, c='green', alpha=0.7, edgecolors='black', linewidths=2)
ax2.scatter([0], [0], [0],
            s=300, c='red', marker='X', edgecolors='black', linewidths=2)
for i in range(len(R)):
    ax2.text(R[i, 0], R[i, 1], R[i, 2], f'  D{i+1}', fontsize=9)
ax2.set_title('AFTER CENTERING', fontsize=13, fontweight='bold')
ax2.set_xlabel('Stock 1 (deviation)', fontsize=11)
ax2.set_ylabel('Stock 2 (deviation)', fontsize=11)
ax2.set_zlabel('Stock 3 (deviation)', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ PART 2 COMPLETE!")
print("Next: Covariance matrix - measuring how stocks move together")