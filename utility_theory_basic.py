# SIMPLER DASHBOARD - EASIER TO UNDERSTAND
import numpy as np
import matplotlib.pyplot as plt

class SimpleUtilityExplorer:
    def __init__(self):
        self.assets = ['Stocks', 'Bonds']
        self.returns = np.array([0.12, 0.06])
        self.risks = np.array([0.15, 0.05])
        self.corr = -0.2  
        
        # Covariance
        self.cov = np.array([
            [self.risks[0]**2, self.risks[0]*self.risks[1]*self.corr],
            [self.risks[0]*self.risks[1]*self.corr, self.risks[1]**2]
        ])
        
        self.W0 = 100
        self.lam = 2.0  
    
    def utility(self, W, lam):
        if lam == 1:
            return np.log(W)
        return (W**(1 - lam)) / (1 - lam)
    
    def plot_simple_dashboard(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        self.plot_utility_curve(axes[0, 0])
        
        self.plot_risk_return(axes[0, 1])
        
        self.plot_optimal_allocation(axes[1, 0])
        
        self.plot_lambda_effect(axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    def plot_utility_curve(self, ax):
        """Graph 1: Utility vs Stock Weight"""
        stock_weights = np.linspace(0, 1, 100)
        utilities = []
        
        for w_stock in stock_weights:
            w = np.array([w_stock, 1 - w_stock])
            Rp = np.dot(w, self.returns)
            sigma_p2 = w.T @ self.cov @ w
            W_final = self.W0 * (1 + Rp)
            U = self.utility(W_final, self.lam) - 0.5 * self.lam * sigma_p2
            utilities.append(U)
        
        ax.plot(stock_weights * 100, utilities, 'b-', linewidth=2)
        

        optimal_idx = np.argmax(utilities)
        ax.scatter(stock_weights[optimal_idx] * 100, utilities[optimal_idx], 
                  color='red', s=100, zorder=5)
        
        ax.set_xlabel('Weight in Stocks (%)')
        ax.set_ylabel('Utility')
        ax.set_title(f'Utility Curve (λ = {self.lam})')
        ax.grid(True, alpha=0.3)
    
    def plot_risk_return(self, ax):
        """Graph 2: Risk-Return tradeoff"""
        stock_weights = np.linspace(0, 1, 50)
        risks = []
        returns = []
        
        for w_stock in stock_weights:
            w = np.array([w_stock, 1 - w_stock])
            Rp = np.dot(w, self.returns)
            sigma_p = np.sqrt(w.T @ self.cov @ w)
            returns.append(Rp)
            risks.append(sigma_p)
        
        ax.plot(risks, returns, 'g-', linewidth=2)
        
        # Mark individual assets
        ax.scatter(self.risks[0], self.returns[0], color='red', s=100, label='Stocks')
        ax.scatter(self.risks[1], self.returns[1], color='blue', s=100, label='Bonds')
        
        ax.set_xlabel('Portfolio Risk')
        ax.set_ylabel('Portfolio Return')
        ax.set_title('Risk-Return Tradeoff')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_optimal_allocation(self, ax):
        """Graph 3: Optimal portfolio pie chart"""
        # Find optimal weights by testing many points
        test_weights = np.linspace(0, 1, 1000)
        best_U = -np.inf
        best_w = 0.5
        
        for w_stock in test_weights:
            w = np.array([w_stock, 1 - w_stock])
            Rp = np.dot(w, self.returns)
            sigma_p2 = w.T @ self.cov @ w
            W_final = self.W0 * (1 + Rp)
            U = self.utility(W_final, self.lam) - 0.5 * self.lam * sigma_p2
            
            if U > best_U:
                best_U = U
                best_w = w_stock
        
        sizes = [best_w * 100, (1 - best_w) * 100]
        colors = ['lightcoral', 'lightskyblue']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=self.assets, colors=colors,
                                         autopct='%1.1f%%', startangle=90)
        
        ax.set_title(f'Optimal Allocation\nUtility: {best_U:.3f}')
    
    def plot_lambda_effect(self, ax):
        """Graph 4: How λ affects max utility"""
        lambdas = np.linspace(0.5, 5, 20)
        max_utilities = []
        
        for lam in lambdas:
            # For each λ, find max utility
            test_weights = np.linspace(0, 1, 1000)
            best_U = -np.inf
            
            for w_stock in test_weights:
                w = np.array([w_stock, 1 - w_stock])
                Rp = np.dot(w, self.returns)
                sigma_p2 = w.T @ self.cov @ w
                W_final = self.W0 * (1 + Rp)
                
                if lam == 1:
                    U = np.log(W_final) - 0.5 * lam * sigma_p2
                else:
                    U = (W_final**(1 - lam))/(1 - lam) - 0.5 * lam * sigma_p2
                
                if U > best_U:
                    best_U = U
            
            max_utilities.append(best_U)
        
        ax.plot(lambdas, max_utilities, 'r-', linewidth=2)
        ax.scatter(self.lam, max_utilities[7], color='black', s=100, zorder=5)
        
        ax.set_xlabel('Risk Aversion (λ)')
        ax.set_ylabel('Maximum Utility')
        ax.set_title('Effect of Risk Aversion')
        ax.grid(True, alpha=0.3)


explorer = SimpleUtilityExplorer()
explorer.plot_simple_dashboard()

