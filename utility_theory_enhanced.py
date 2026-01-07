import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class CleanUtilityTheory:
    def __init__(self):
        self.assets = ['Stocks', 'Bonds', 'Real Estate', 'Gold']
        self.returns = np.array([0.12, 0.06, 0.08, 0.04])
        self.risks = np.array([0.15, 0.05, 0.10, 0.08])
        
        self.corr_matrix = np.array([
            [1.00, -0.20, 0.30, 0.10],
            [-0.20, 1.00, 0.00, 0.20],
            [0.30, 0.00, 1.00, 0.40],
            [0.10, 0.20, 0.40, 1.00]
        ])
        
        self.cov_matrix = self._make_cov_matrix()
        self.W0 = 100
        self.lam = 2.0
        
        self.create_plots()
    
    def _make_cov_matrix(self):
        cov = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                cov[i, j] = self.corr_matrix[i, j] * self.risks[i] * self.risks[j]
        return cov
    
    def utility(self, W, lam):
        if lam == 1:
            return np.log(W)
        return (W**(1 - lam)) / (1 - lam)
    
    def portfolio_stats(self, weights):
        Rp = np.dot(weights, self.returns)
        sigma_p = np.sqrt(weights.T @ self.cov_matrix @ weights)
        return Rp, sigma_p
    
    def calculate_utility(self, weights, lam):
        Rp, sigma_p = self.portfolio_stats(weights)
        W_final = self.W0 * (1 + Rp)
        U = self.utility(W_final, lam) - 0.5 * lam * sigma_p**2
        return U, Rp, sigma_p
    
    def find_optimal_portfolio(self, lam, n_trials=5000):
        best_U = -np.inf
        best_weights = None
        
        for _ in range(n_trials):
            weights = np.random.random(4)
            weights = weights / weights.sum()
            U, _, _ = self.calculate_utility(weights, lam)
            
            if U > best_U:
                best_U = U
                best_weights = weights
        
        return best_weights, best_U
    
    def create_plots(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.ax4 = ax4
        
        slider_ax = plt.axes([0.25, 0.01, 0.5, 0.03])
        self.slider = Slider(slider_ax, 'Risk Aversion (λ)', 0.5, 5.0, 
                           valinit=self.lam, valstep=0.1)
        
        self.slider.on_changed(self.update_plots)
        
        self.update_plots(self.lam)
        plt.tight_layout()
        plt.show()
    
    def update_plots(self, val):
        self.lam = val
        
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Plot 1: Utility vs Stock-Bond mix
        stock_weights = np.linspace(0, 1, 100)
        utilities = []
        
        for w_stock in stock_weights:
            w_bond = 1 - w_stock
            weights = np.array([w_stock, w_bond, 0, 0])
            U, _, _ = self.calculate_utility(weights, self.lam)
            utilities.append(U)
        
        self.ax1.plot(stock_weights * 100, utilities, 'b-', linewidth=2)
        self.ax1.set_xlabel('Weight in Stocks (%)')
        self.ax1.set_ylabel('Utility')
        self.ax1.set_title(f'Utility Curve (λ = {self.lam:.1f})')
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Risk-Return tradeoff
        returns = []
        risks = []
        
        for _ in range(1000):
            weights = np.random.random(4)
            weights = weights / weights.sum()
            Rp, sigma_p = self.portfolio_stats(weights)
            returns.append(Rp)
            risks.append(sigma_p)
        
        self.ax2.scatter(risks, returns, alpha=0.5, s=10)
        self.ax2.set_xlabel('Risk')
        self.ax2.set_ylabel('Return')
        self.ax2.set_title('Risk-Return Tradeoff')
        self.ax2.grid(True, alpha=0.3)
        
        # Plot 3: Optimal portfolio
        opt_weights, opt_utility = self.find_optimal_portfolio(self.lam)
        colors = ['red', 'blue', 'green', 'orange']
        
        bars = self.ax3.bar(self.assets, opt_weights * 100, color=colors)
        self.ax3.set_ylabel('Weight (%)')
        self.ax3.set_title(f'Optimal Portfolio (U = {opt_utility:.3f})')
        self.ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Lambda sensitivity
        lambdas = np.linspace(0.5, 5, 20)
        max_utilities = []
        
        for lam in lambdas:
            _, U_opt = self.find_optimal_portfolio(lam, n_trials=2000)
            max_utilities.append(U_opt)
        
        self.ax4.plot(lambdas, max_utilities, 'g-')
        self.ax4.scatter(self.lam, max_utilities[10], color='red', s=50)
        self.ax4.set_xlabel('Risk Aversion (λ)')
        self.ax4.set_ylabel('Max Utility')
        self.ax4.set_title('Utility vs Risk Aversion')
        self.ax4.grid(True, alpha=0.3)

if __name__ == "__main__":
    CleanUtilityTheory()