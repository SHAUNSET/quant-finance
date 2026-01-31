import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "AAPL": [0.01, -0.005, 0.004, 0.006, -0.002, 0.003],
    "MSFT": [0.008, -0.004, 0.003, 0.005, -0.001, 0.002],
    "GOOGL": [0.012, -0.006, 0.005, 0.007, -0.003, 0.004]
}

returns_df = pd.DataFrame(data)

print(returns_df)

R = returns_df.values   # numpy matrix
print("Shape:", R.shape)

mean_returns = R.mean(axis=0)
print("Mean returns:", mean_returns)



plt.figure()
for col in returns_df.columns:
    plt.plot(returns_df[col], label=col)

plt.title("Daily Stock Returns")
plt.xlabel("Time (Days)")
plt.ylabel("Return")
plt.legend()
plt.show()
