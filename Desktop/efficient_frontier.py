import numpy as np
import matplotlib.pyplot as plt


R = np.array([0.10, 0.15])  
sigma = np.array([0.05, 0.10])  

corr = 0.2

cov_matrix = np.array([[sigma[0]**2, sigma[0]*sigma[1]*corr],
                       [sigma[0]*sigma[1]*corr, sigma[1]**2]])

weights = np.linspace(0 , 1 , 11)

portfolio_returns = []
portfolio_risks = []

for w1 in weights :
    w = np.array([w1 , 1-w1])

    Rp = np.dot(w,R)

    sigma_p2 = w.T @ cov_matrix @ w
    sigma_p = np.sqrt(sigma_p2)

    portfolio_returns.append(Rp)
    portfolio_risks.append(sigma_p)


plt.plot(portfolio_risks , portfolio_returns, marker = 'o')
plt.xlabel("Portfolio Risks")
plt.ylabel("Portfolio Returns")
plt.title("Efficient Frontier")
plt.grid(True)
plt.show()


