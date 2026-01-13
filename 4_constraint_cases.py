"""
PORTFOLIO THEORY: 4 CONSTRAINT CASES
Simple, clear implementation with visual intuition
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class FourCasesPortfolio:
    def __init__(self):
        # 3 real assets with different characteristics
        self.assets = ['Tech (AAPL)', 'Healthcare (JNJ)', 'Energy (XOM)']
        self.returns = np.array([0.15, 0.08, 0.12])    # Expected returns
        self.vols = np.array([0.25, 0.15, 0.20])       # Volatilities
        
        # Realistic correlations
        self.corr = np.array([
            [1.0, 0.1, 0.3],   # Tech-Healthcare low, Tech-Energy moderate
            [0.1, 1.0, 0.15],  # Healthcare-Energy low
            [0.3, 0.15, 1.0]   # Energy-Tech moderate
        ])
        
        # Create covariance matrix
        self.cov = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                self.cov[i, j] = self.corr[i, j] * self.vols[i] * self.vols[j]
        
        self.rf = 0.04  # 4% risk-free rate
    
    def portfolio_stats(self, weights):
        """Calculate return and risk for given weights"""
        Rp = np.dot(weights, self.returns)
        σp = np.sqrt(weights.T @ self.cov @ weights)
        return Rp, σp
    
    def case1_full_freedom(self):
        """CASE 1: Can short & borrow/lend (Hedge Fund)"""
        print("\n📈 CASE 1: HEDGE FUND MODE")
        print("-"*40)
        print("✓ Short selling: YES (weights can be negative)")
        print("✓ Borrow/Lend: YES (use risk-free asset)")
        
        # Find tangency portfolio (max Sharpe)
        def neg_sharpe(w):
            Rp, σp = self.portfolio_stats(w)
            return - (Rp - self.rf) / σp if σp > 0 else 1e6
        
        result = minimize(neg_sharpe, [0.33, 0.33, 0.34],
                         constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                         bounds=[(-2, 2), (-2, 2), (-2, 2)])
        
        if result.success:
            w_tangency = result.x
            Rp_t, σp_t = self.portfolio_stats(w_tangency)
            sharpe = (Rp_t - self.rf) / σp_t
            
            print(f"\nTangency Portfolio (Max Sharpe = {sharpe:.3f}):")
            for i, (asset, w) in enumerate(zip(self.assets, w_tangency)):
                print(f"  {asset}: {w:+.1%}")
            
            # CML: Return = rf + sharpe × risk
            print(f"\nCapital Market Line: Return = {self.rf:.1%} + {sharpe:.3f} × Risk")
            
            return {'type': 'cml', 'sharpe': sharpe, 'tangency': (σp_t, Rp_t)}
        return None
    
    def case2_no_short_but_borrow(self):
        """CASE 2: No shorting but can borrow/lend (Mutual Fund)"""
        print("\n🏦 CASE 2: MUTUAL FUND MODE")
        print("-"*40)
        print("✓ Short selling: NO (weights ≥ 0)")
        print("✓ Borrow/Lend: YES")
        
        def neg_sharpe(w):
            Rp, σp = self.portfolio_stats(w)
            return - (Rp - self.rf) / σp if σp > 0 else 1e6
        
        result = minimize(neg_sharpe, [0.33, 0.33, 0.34],
                         constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                         bounds=[(0, 1), (0, 1), (0, 1)])
        
        if result.success:
            w_tangency = result.x
            Rp_t, σp_t = self.portfolio_stats(w_tangency)
            sharpe = (Rp_t - self.rf) / σp_t
            
            print(f"\nTangency Portfolio (Max Sharpe = {sharpe:.3f}):")
            for i, (asset, w) in enumerate(zip(self.assets, w_tangency)):
                print(f"  {asset}: {w:.1%}")
            
            print(f"\nCapital Market Line: Return = {self.rf:.1%} + {sharpe:.3f} × Risk")
            
            return {'type': 'cml', 'sharpe': sharpe, 'tangency': (σp_t, Rp_t)}
        return None
    
    def case3_short_but_no_rf(self):
        """CASE 3: Can short but no risk-free (Hedge Fund with 100% invested)"""
        print("\n⚡ CASE 3: HEDGE FUND (No Cash)")
        print("-"*40)
        print("✓ Short selling: YES")
        print("✓ Borrow/Lend: NO (must stay 100% invested)")
        
        # Generate efficient frontier with shorting
        frontier = []
        for target in np.linspace(0.05, 0.20, 15):
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, self.returns) - target}
            ]
            
            result = minimize(lambda w: w.T @ self.cov @ w, [0.33, 0.33, 0.34],
                             constraints=constraints, bounds=[(-1, 1), (-1, 1), (-1, 1)])
            
            if result.success:
                σp = np.sqrt(result.fun)
                frontier.append((σp, target, result.x))
        
        # Find max Sharpe portfolio (relative to 0%)
        def neg_sharpe_no_rf(w):
            Rp, σp = self.portfolio_stats(w)
            return - Rp / σp if σp > 0 else 1e6
        
        result = minimize(neg_sharpe_no_rf, [0.33, 0.33, 0.34],
                         constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                         bounds=[(-1, 1), (-1, 1), (-1, 1)])
        
        if result.success:
            w_max_sharpe = result.x
            Rp_ms, σp_ms = self.portfolio_stats(w_max_sharpe)
            sharpe_ms = Rp_ms / σp_ms
            
            print(f"\nMaximum Sharpe Portfolio (Sharpe = {sharpe_ms:.3f}):")
            for i, (asset, w) in enumerate(zip(self.assets, w_max_sharpe)):
                action = "SHORT" if w < 0 else "LONG"
                print(f"  {asset}: {w:+.1%} ({action})")
            
            return {'type': 'frontier', 'points': frontier, 'max_sharpe': (σp_ms, Rp_ms)}
        return {'type': 'frontier', 'points': frontier}
    
    def case4_no_short_no_rf(self):
        """CASE 4: No shorting, no risk-free (Restricted Investor)"""
        print("\n🔒 CASE 4: RESTRICTED INVESTOR")
        print("-"*40)
        print("✓ Short selling: NO")
        print("✓ Borrow/Lend: NO")
        print("✗ Only long positions, fully invested")
        
        frontier = []
        for target in np.linspace(0.06, 0.16, 10):
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, self.returns) - target}
            ]
            
            result = minimize(lambda w: w.T @ self.cov @ w, [0.33, 0.33, 0.34],
                             constraints=constraints, bounds=[(0, 1), (0, 1), (0, 1)])
            
            if result.success:
                σp = np.sqrt(result.fun)
                frontier.append((σp, target, result.x))
        
        print(f"\nAvailable Return Range: {frontier[0][1]:.1%} to {frontier[-1][1]:.1%}")
        print(f"Available Risk Range: {frontier[0][0]:.1%} to {frontier[-1][0]:.1%}")
        
        return {'type': 'frontier', 'points': frontier}
    
    def plot_all_cases(self):
        """Plot all 4 cases for visual comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        cases = [
            ("Case 1: Short + Borrow/Lend", self.case1_full_freedom, 'green'),
            ("Case 2: No Short + Borrow/Lend", self.case2_no_short_but_borrow, 'blue'),
            ("Case 3: Short + No Risk-Free", self.case3_short_but_no_rf, 'orange'),
            ("Case 4: No Short + No Risk-Free", self.case4_no_short_no_rf, 'red')
        ]
        
        for idx, (title, case_func, color) in enumerate(cases):
            ax = axes[idx//2, idx%2]
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Risk (σ)')
            ax.set_ylabel('Return (R)')
            ax.grid(True, alpha=0.3)
            
            # Run the case
            result = case_func()
            
            if result is None:
                continue
                
            if result['type'] == 'cml':
                # Plot Capital Market Line
                sharpe = result['sharpe']
                σ_t, R_t = result['tangency']
                
                # Plot CML line
                σ_range = np.linspace(0, σ_t * 2, 100)
                R_cml = self.rf + sharpe * σ_range
                ax.plot(σ_range, R_cml, '--', color=color, alpha=0.7, label=f'CML (Sharpe={sharpe:.2f})')
                
                # Mark tangency portfolio
                ax.scatter(σ_t, R_t, color=color, s=100, marker='*', label='Tangency')
                
                # Mark risk-free asset
                ax.scatter(0, self.rf, color='black', s=80, marker='^', label='Risk-Free')
                
                ax.legend(loc='upper left', fontsize=9)
                
            else:  # frontier
                # Plot efficient frontier
                points = result['points']
                if points:
                    σ_vals = [p[0] for p in points]
                    R_vals = [p[1] for p in points]
                    
                    ax.plot(σ_vals, R_vals, '-o', color=color, alpha=0.7, markersize=4)
                
                # Mark individual assets
                for i in range(3):
                    ax.scatter(self.vols[i], self.returns[i], 
                              color='gray', s=60, alpha=0.5)
                    ax.annotate(self.assets[i].split()[0], 
                               (self.vols[i], self.returns[i]),
                               fontsize=8, alpha=0.7)
                
                # Mark max Sharpe if available
                if 'max_sharpe' in result:
                    σ_ms, R_ms = result['max_sharpe']
                    ax.scatter(σ_ms, R_ms, color='purple', s=100, 
                              marker='s', label='Max Sharpe')
                    ax.legend(loc='upper left', fontsize=9)
        
        plt.suptitle('Portfolio Theory: 4 Constraint Cases', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

# Run the analysis
if __name__ == "__main__":
    # Create instance first
    portfolio = FourCasesPortfolio()
    
    print("="*70)
    print("PORTFOLIO THEORY: 4 CONSTRAINT CASES VISUALIZED")
    print("="*70)
    print("\nAssets Available:")
    for i in range(3):
        print(f"  {portfolio.assets[i]}: Return={portfolio.returns[i]:.1%}, Risk={portfolio.vols[i]:.1%}")
    
    print(f"\nRisk-Free Rate: {portfolio.rf:.1%}")
    print("="*70)
    
    portfolio.plot_all_cases()
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print("\nCASE 1 (Green): Short + Borrow/Lend")
    print("  • Highest possible returns via leverage & shorting")
    print("  • Straight Capital Market Line (CML)")
    print("  • Best risk-return tradeoff")
    
    print("\nCASE 2 (Blue): No Short + Borrow/Lend")
    print("  • Can only go long (no negative bets)")
    print("  • Still get CML but lower Sharpe")
    print("  • Typical for mutual funds")
    
    print("\nCASE 3 (Orange): Short + No Risk-Free")
    print("  • Can short but must stay 100% invested")
    print("  • Stuck on curved efficient frontier")
    print("  • No leverage via risk-free")
    
    print("\nCASE 4 (Red): No Short + No Risk-Free")
    print("  • Most restrictive: only long, fully invested")
    print("  • Smallest efficient frontier")
    print("  • Typical for some retirement accounts")
    print("="*70)