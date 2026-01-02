import numpy as np
import matplotlib.pyplot as plt

W0 = 100

R = np.array([0.10, 0.15])  # Asset A = 10%, Asset B = 15%
sigma = np.array([0.05, 0.10])  # Asset A = 5%, Asset B = 10%
corr = 0.2

cov_matrix = np.array([
    [sigma[0]**2, sigma[0]*sigma[1]*corr],
    [sigma[0]*sigma[1]*corr, sigma[1]**2]
])

def utility(W, lam):
    if lam == 1:
        return np.log(W)           
    else:
        return (W**(1 - lam)) / (1 - lam)  

weights = np.linspace(0, 1, 11)

def plot_utility(lam):
    portfolio_utilities = []

    for wA in weights:
        w = np.array([wA, 1 - wA])  
        Rp = np.dot(w, R)
        sigma_p2 = w.T @ cov_matrix @ w
        Wf = W0 * (1 + Rp)
        U = utility(Wf, lam) - 0.5 * lam * sigma_p2
        portfolio_utilities.append(U)
    
    plt.plot(weights, portfolio_utilities, marker='o')
    plt.xlabel("Weight in Asset A")
    plt.ylabel("Risk-adjusted Utility")
    plt.title(f"Portfolio Utility vs Allocation (lambda = {lam})")
    plt.grid(True)
    plt.show()

plot_utility(0.9)
