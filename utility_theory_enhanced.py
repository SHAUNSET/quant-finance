import numpy as np
import matplotlib.pyplot as plt

class UtilityTheoryExplorer:
    def __init__(self, num_assets=4):
        self.num_assets = num_assets
        self.asset_names = ['Stocks', 'Bonds', 'Real Estate', 'Gold']
        self.W0 = 100
        
        self.returns = np.array([0.12, 0.06, 0.08, 0.04])
        self.volatilities = np.array([0.15, 0.05, 0.10, 0.08])
        
        self.corr_matrix = np.array([
            [1.0, -0.2, 0.3, 0.1],
            [-0.2, 1.0, 0.0, 0.2],
            [0.3, 0.0, 1.0, 0.4],
            [0.1, 0.2, 0.4, 1.0]
        ])
        
        self.cov_matrix = self._make_cov_matrix()
        self.current_lam = 2.0
    
    def _make_cov_matrix(self):
        cov = np.zeros((self.num_assets, self.num_assets))
        for i in range(self.num_assets):
            for j in range(self.num_assets):
                cov[i, j] = self.corr_matrix[i, j] * self.volatilities[i] * self.volatilities[j]
        return cov
    
    def utility(self, W, lam):
        if lam == 1:
            return np.log(W)
        else:
            return (W**(1 - lam)) / (1 - lam)
    
    def portfolio_stats(self, weights):
        Rp = np.dot(weights, self.returns)
        sigma_p2 = weights.T @ self.cov_matrix @ weights
        sigma_p = np.sqrt(sigma_p2)
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
            weights = np.random.random(self.num_assets)
            weights = weights / weights.sum()
            
            U, _, _ = self.calculate_utility(weights, lam)
            
            if U > best_U:
                best_U = U
                best_weights = weights
        
        return best_weights, best_U
    
    def plot_utility_analysis(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        self._plot_utility_surface(axes[0, 0])
        self._plot_risk_return_tradeoff(axes[0, 1])
        self._plot_optimal_composition(axes[1, 0])
        self._plot_utility_vs_lambda(axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    def _plot_utility_surface(self, ax):
        n_points = 30
        w1_range = np.linspace(0, 1, n_points)
        w2_range = np.linspace(0, 1, n_points)
        W1, W2 = np.meshgrid(w1_range, w2_range)
        
        U_surface = np.zeros_like(W1)
        for i in range(n_points):
            for j in range(n_points):
                w1 = W1[i, j]
                w2 = W2[i, j]
                w3 = 1 - w1 - w2
                
                if w3 >= 0:
                    weights = np.array([w1, w2, w3, 0])
                    U, _, _ = self.calculate_utility(weights, self.current_lam)
                    U_surface[i, j] = U
                else:
                    U_surface[i, j] = np.nan
        
        contour = ax.contourf(W1, W2, U_surface, levels=20, cmap='viridis', alpha=0.8)
        ax.contour(W1, W2, U_surface, levels=10, colors='black', linewidths=0.5, alpha=0.7)
        
        opt_weights, _ = self.find_optimal_portfolio(self.current_lam)
        ax.scatter(opt_weights[0], opt_weights[1], color='red', s=200, marker='*', zorder=5)
        
        ax.set_xlabel('Weight in Stocks')
        ax.set_ylabel('Weight in Bonds')
        ax.set_title(f'Utility Surface (λ = {self.current_lam})')
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(contour, ax=ax, label='Utility')
    
    def _plot_risk_return_tradeoff(self, ax):
        n_portfolios = 1000
        returns = []
        risks = []
        utilities = []
        
        for _ in range(n_portfolios):
            weights = np.random.random(self.num_assets)
            weights = weights / weights.sum()
            U, Rp, sigma_p = self.calculate_utility(weights, self.current_lam)
            returns.append(Rp)
            risks.append(sigma_p)
            utilities.append(U)
        
        scatter = ax.scatter(risks, returns, c=utilities, cmap='plasma', alpha=0.7, s=30)
        ax.set_xlabel('Portfolio Risk')
        ax.set_ylabel('Portfolio Return')
        ax.set_title('Risk-Return Tradeoff')
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='Utility')
        
        for i in range(self.num_assets):
            ax.scatter(self.volatilities[i], self.returns[i], color='red', s=100, marker='o', zorder=5)
    
    def _plot_optimal_composition(self, ax):
        opt_weights, opt_utility = self.find_optimal_portfolio(self.current_lam)
        
        colors = plt.cm.Set3(np.arange(self.num_assets) / self.num_assets)
        wedges, texts, autotexts = ax.pie(opt_weights * 100, 
                                         labels=self.asset_names,
                                         autopct='%1.1f%%',
                                         colors=colors,
                                         startangle=90)
        
        for text in texts + autotexts:
            text.set_fontsize(10)
        
        ax.set_title(f'Optimal Portfolio\nUtility: {opt_utility:.3f}')
    
    def _plot_utility_vs_lambda(self, ax):
        lambdas = np.linspace(0.5, 5, 20)
        optimal_utilities = []
        
        for lam in lambdas:
            _, U_opt = self.find_optimal_portfolio(lam, n_trials=2000)
            optimal_utilities.append(U_opt)
        
        ax.plot(lambdas, optimal_utilities, 'g-', linewidth=2)
        ax.scatter(self.current_lam, optimal_utilities[10], color='red', s=100, zorder=5)
        ax.axvline(x=self.current_lam, color='r', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Risk Aversion (λ)')
        ax.set_ylabel('Maximum Utility')
        ax.set_title('Utility Sensitivity to Risk Aversion')
        ax.grid(True, alpha=0.3)
    
    def plot_interactive_utility(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(bottom=0.2)
        
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = plt.Slider(ax_slider, 'Risk Aversion (λ)', 0.5, 5.0, valinit=2.0, valstep=0.1)
        
        def update(val):
            self.current_lam = slider.val
            
            axes[0, 0].clear()
            axes[0, 1].clear()
            axes[1, 0].clear()
            axes[1, 1].clear()
            
            self._plot_utility_surface(axes[0, 0])
            self._plot_risk_return_tradeoff(axes[0, 1])
            self._plot_optimal_composition(axes[1, 0])
            self._plot_utility_vs_lambda(axes[1, 1])
            
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        
        self._plot_utility_surface(axes[0, 0])
        self._plot_risk_return_tradeoff(axes[0, 1])
        self._plot_optimal_composition(axes[1, 0])
        self._plot_utility_vs_lambda(axes[1, 1])
        
        plt.show()

if __name__ == "__main__":
    print("Enhanced Utility Theory Explorer")
    print("=" * 50)
    
    explorer = UtilityTheoryExplorer(num_assets=4)
    
    print("\nAssets:")
    for i in range(explorer.num_assets):
        print(f"{explorer.asset_names[i]:15} | Return: {explorer.returns[i]:.2%} | Risk: {explorer.volatilities[i]:.2%}")
    
    print("\n1. Static Analysis (4 plots)")
    print("2. Interactive Analysis (with slider)")
    
    choice = input("\nChoose (1 or 2): ")
    
    if choice == '1':
        explorer.plot_utility_analysis()
    else:
        explorer.plot_interactive_utility()
    
    print("\nAnalysis complete!")