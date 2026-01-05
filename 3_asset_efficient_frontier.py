import numpy as np
import matplotlib.pyplot as plt

class InteractiveEfficientFrontier:
    def __init__(self):

        self.R = np.array([0.08, 0.12, 0.15])
        self.sigma = np.array([0.05, 0.10, 0.12])
        self.names = ['Tech', 'Healthcare', 'Energy']
        

        corr = np.array([
            [1.0, 0.3, 0.1],
            [0.3, 1.0, 0.4],
            [0.1, 0.4, 1.0]
        ])
        

        self.cov_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                self.cov_matrix[i, j] = self.sigma[i] * self.sigma[j] * corr[i, j]
        
  
        self.generate_portfolios()
        self.find_efficient_frontier()
    
    def generate_portfolios(self):
        """Generate random portfolios"""
        np.random.seed(42)  
        
        n_portfolios = 1000
        self.portfolio_returns = []
        self.portfolio_risks = []
        self.portfolio_weights = []
        
        for _ in range(n_portfolios):

            w = np.random.random(3)
            w = w / np.sum(w)  
            

            Rp = np.dot(w, self.R)
            

            sigma_p = np.sqrt(w.T @ self.cov_matrix @ w)
            
            self.portfolio_returns.append(Rp)
            self.portfolio_risks.append(sigma_p)
            self.portfolio_weights.append(w)
        

        self.portfolio_returns = np.array(self.portfolio_returns)
        self.portfolio_risks = np.array(self.portfolio_risks)
        self.portfolio_weights = np.array(self.portfolio_weights)
    
    def find_efficient_frontier(self):
        """Find efficient frontier portfolios"""

        sorted_indices = np.argsort(self.portfolio_risks)
        sorted_risks = self.portfolio_risks[sorted_indices]
        sorted_returns = self.portfolio_returns[sorted_indices]
        sorted_weights = self.portfolio_weights[sorted_indices]
        

        self.efficient_risks = []
        self.efficient_returns = []
        self.efficient_weights = []
        
        max_return_so_far = -np.inf
        
        for i in range(len(sorted_risks)):
            if sorted_returns[i] > max_return_so_far:
                max_return_so_far = sorted_returns[i]
                self.efficient_risks.append(sorted_risks[i])
                self.efficient_returns.append(sorted_returns[i])
                self.efficient_weights.append(sorted_weights[i])
    
    def plot(self):
        """Create interactive plot"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Plot all portfolios
        self.ax.scatter(self.portfolio_risks, self.portfolio_returns, 
                       alpha=0.3, s=10, color='lightblue', 
                       label='Possible Portfolios')
        
        # Plot efficient frontier
        self.ax.plot(self.efficient_risks, self.efficient_returns, 
                    'r-', linewidth=3, label='Efficient Frontier')
        
        # Plot individual assets
        for i in range(3):
            self.ax.scatter(self.sigma[i], self.R[i], color='green', 
                          s=150, marker='o', edgecolors='black', linewidth=2)
            self.ax.annotate(self.names[i], 
                           xy=(self.sigma[i], self.R[i]),
                           xytext=(5, 5), textcoords='offset points')
        
        # Plot MVP
        mvp_idx = np.argmin(self.portfolio_risks)
        self.mvp_point, = self.ax.plot(
            self.portfolio_risks[mvp_idx], self.portfolio_returns[mvp_idx],
            'r*', markersize=15, label='Minimum Variance Portfolio'
        )
        
        # Set up click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add annotation for clicked point
        self.annotation = self.ax.annotate(
            '', xy=(0, 0), xytext=(20, 20),
            textcoords='offset points',
            bbox=dict(boxstyle='round', fc='yellow', alpha=0.8),
            arrowprops=dict(arrowstyle='->')
        )
        self.annotation.set_visible(False)
        
        self.ax.set_xlabel('Portfolio Risk (Standard Deviation)')
        self.ax.set_ylabel('Portfolio Return')
        self.ax.set_title('Interactive Efficient Frontier\nClick on any point!')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def on_click(self, event):
        """Handle click events"""
        if event.inaxes != self.ax:
            return
        
        # Find the closest portfolio
        x_click, y_click = event.xdata, event.ydata
        
        # Calculate distances to all efficient frontier points
        distances = np.sqrt(
            (np.array(self.efficient_risks) - x_click)**2 + 
            (np.array(self.efficient_returns) - y_click)**2
        )
        
        closest_idx = np.argmin(distances)
        
        # Update annotation
        weights = self.efficient_weights[closest_idx]
        weight_text = '\n'.join([f'{self.names[i]}: {weights[i]:.1%}' 
                                for i in range(3)])
        
        self.annotation.xy = (self.efficient_risks[closest_idx], 
                             self.efficient_returns[closest_idx])
        self.annotation.set_text(
            f'Portfolio Details:\n'
            f'Risk: {self.efficient_risks[closest_idx]:.3f}\n'
            f'Return: {self.efficient_returns[closest_idx]:.3f}\n\n'
            f'Weights:\n{weight_text}'
        )
        self.annotation.set_visible(True)
        
        # Draw a marker at clicked point
        if hasattr(self, 'clicked_point'):
            self.clicked_point.remove()
        
        self.clicked_point, = self.ax.plot(
            self.efficient_risks[closest_idx], 
            self.efficient_returns[closest_idx],
            'bo', markersize=10, alpha=0.7
        )
        
        self.fig.canvas.draw()


if __name__ == "__main__":
    print("=== 3-ASSET EFFICIENT FRONTIER ===")
    print("Assets: Tech (8%, 5%), Healthcare (12%, 10%), Energy (15%, 12%)")
    print("Click on the efficient frontier (red line) to see portfolio weights!\n")
    
    frontier = InteractiveEfficientFrontier()
    frontier.plot()

