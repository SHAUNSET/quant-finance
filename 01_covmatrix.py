import numpy as np
import matplotlib.pyplot as plt

returns = np.array([
    [0.01, 0.008, 0.012],
    [-0.005, -0.004, -0.006],
    [0.004, 0.003, 0.005],
    [0.006, 0.005, 0.007],
    [-0.002, -0.001, -0.003],
    [0.003, 0.002, 0.004]
])

cov_matrix = np.cov(returns, rowvar=False)

print(cov_matrix)

plt.figure()
plt.imshow(cov_matrix)
plt.colorbar()
plt.title("Covariance Matrix")
plt.xticks([0,1,2], ["AAPL","MSFT","GOOGL"])
plt.yticks([0,1,2], ["AAPL","MSFT","GOOGL"])
plt.savefig('cov_matrix.png')
plt.show()
