import numpy as np
import matplotlib.pyplot as plt

class EfficientFrontier:
    def __init__(self, num_assets=5):
        self.num_assets = num_assets
        self.asset_names = ['Tech', 'Healthcare', 'Energy', 'Finance', 'Consumer']
        
        np.random.seed(42)
        self.returns = np.array([0.12, 0.15, 0.10, 0.08, 0.13])
        self.volatilities = np.array([0.15, 0.18, 0.12, 0.08, 0.10])
        
        self.corr_matrix = self.make_corr_matrix()
        self.cov_matrix = self.make_cov_matrix()
        
        self.make_portfolios()
        self.find_special_portfolios()
    
    def make_corr_matrix(self):
        corr = np.eye(self.num_assets)
        
        corr[0, 1] = corr[1, 0] = 0.3
        corr[0, 2] = corr[2, 0] = 0.1
        corr[0, 3] = corr[3, 0] = 0.4
        corr[0, 4] = corr[4, 0] = 0.2
        
        corr[1, 2] = corr[2, 1] = 0.4
        corr[1, 3] = corr[3, 1] = 0.2
        corr[1, 4] = corr[4, 1] = 0.3
        
        corr[2, 3] = corr[3, 2] = 0.1
        corr[2, 4] = corr[4, 2] = 0.2
        
        corr[3, 4] = corr[4, 3] = 0.5
        
        return corr
    
    def make_cov_matrix(self):
        cov = np.zeros((self.num_assets, self.num_assets))
        for i in range(self.num_assets):
            for j in range(self.num_assets):
                cov[i, j] = self.corr_matrix[i, j] * self.volatilities[i] * self.volatilities[j]
        return cov
    
    def make_portfolios(self, n_portfolios=2000):
        self.portfolio_returns = []
        self.portfolio_risks = []
        self.portfolio_weights = []
        
        for _ in range(n_portfolios):
            weights = np.random.random(self.num_assets)
            weights = weights / weights.sum()
            
            port_return = np.dot(weights, self.returns)
            port_risk = np.sqrt(weights.T @ self.cov_matrix @ weights)
            
            self.portfolio_returns.append(port_return)
            self.portfolio_risks.append(port_risk)
            self.portfolio_weights.append(weights)
        
        self.portfolio_returns = np.array(self.portfolio_returns)
        self.portfolio_risks = np.array(self.portfolio_risks)
        self.portfolio_weights = np.array(self.portfolio_weights)
    
    def find_efficient_frontier(self):
        sorted_idx = np.argsort(self.portfolio_risks)
        sorted_risks = self.portfolio_risks[sorted_idx]
        sorted_returns = self.portfolio_returns[sorted_idx]
        sorted_weights = self.portfolio_weights[sorted_idx]
        
        self.efficient_risks = []
        self.efficient_returns = []
        self.efficient_weights = []
        
        max_return = -np.inf
        
        for i in range(len(sorted_risks)):
            if sorted_returns[i] > max_return:
                max_return = sorted_returns[i]
                self.efficient_risks.append(sorted_risks[i])
                self.efficient_returns.append(sorted_returns[i])
                self.efficient_weights.append(sorted_weights[i])
    
    def find_special_portfolios(self):
        self.mvp_index = np.argmin(self.portfolio_risks)
        self.mvp_return = self.portfolio_returns[self.mvp_index]
        self.mvp_risk = self.portfolio_risks[self.mvp_index]
        self.mvp_weights = self.portfolio_weights[self.mvp_index]
        
        self.max_return_index = np.argmax(self.portfolio_returns)
        self.max_return = self.portfolio_returns[self.max_return_index]
        self.max_return_risk = self.portfolio_risks[self.max_return_index]
        self.max_return_weights = self.portfolio_weights[self.max_return_index]
        
        self.find_efficient_frontier()
    
    def plot_frontier(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        scatter = self.ax1.scatter(self.portfolio_risks, self.portfolio_returns,
                                  c=self.portfolio_returns/self.portfolio_risks,
                                  cmap='viridis', alpha=0.5, s=20)
        
        self.ax1.plot(self.efficient_risks, self.efficient_returns, 
                     'r-', linewidth=3)
        
        for i in range(self.num_assets):
            self.ax1.scatter(self.volatilities[i], self.returns[i],
                           color='red', s=150, marker='o', 
                           edgecolors='black', linewidth=2)
            self.ax1.annotate(self.asset_names[i],
                            xy=(self.volatilities[i], self.returns[i]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=10, fontweight='bold')
        
        self.ax1.scatter(self.mvp_risk, self.mvp_return,
                        color='blue', s=150, marker='D',
                        edgecolors='black', linewidth=2)
        
        self.ax1.scatter(self.max_return_risk, self.max_return,
                        color='green', s=150, marker='^',
                        edgecolors='black', linewidth=2)
        
        self.ax1.set_xlabel('Portfolio Risk')
        self.ax1.set_ylabel('Portfolio Return')
        self.ax1.set_title('Efficient Frontier')
        self.ax1.legend(['Efficient Frontier', 'MVP', 'Max Return'])
        self.ax1.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=self.ax1, label='Return/Risk Ratio')
        
        self.current_portfolio = self.max_return_weights
        self.plot_composition(self.ax2, self.current_portfolio)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.tight_layout()
        plt.show()
    
    def plot_composition(self, ax, weights):
        ax.clear()
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_assets))
        bars = ax.bar(range(self.num_assets), weights * 100, color=colors)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Assets')
        ax.set_ylabel('Weight (%)')
        ax.set_title('Portfolio Composition')
        ax.set_xticks(range(self.num_assets))
        ax.set_xticklabels(self.asset_names, rotation=45)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
    
    def on_click(self, event):
        if event.inaxes != self.ax1:
            return
        
        distances = []
        for i in range(len(self.efficient_risks)):
            dist = np.sqrt(
                (self.efficient_risks[i] - event.xdata)**2 +
                (self.efficient_returns[i] - event.ydata)**2
            )
            distances.append(dist)
        
        closest_idx = np.argmin(distances)
        
        self.current_portfolio = self.efficient_weights[closest_idx]
        self.plot_composition(self.ax2, self.current_portfolio)
        
        risk = self.efficient_risks[closest_idx]
        ret = self.efficient_returns[closest_idx]
        self.ax2.set_title(f'Portfolio Composition\nRisk: {risk:.2%}, Return: {ret:.2%}')
        
        if hasattr(self, 'clicked_point'):
            self.clicked_point.remove()
        
        self.clicked_point, = self.ax1.plot(
            self.efficient_risks[closest_idx],
            self.efficient_returns[closest_idx],
            'ko', markersize=12, markeredgewidth=2,
            markerfacecolor='none'
        )
        
        self.fig.canvas.draw()

if __name__ == "__main__":
    frontier = EfficientFrontier(num_assets=5)
    frontier.plot_frontier()