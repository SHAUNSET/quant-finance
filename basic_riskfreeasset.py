import numpy as np
import matplotlib.pyplot as plt

Rf = 0.02  # Risk-free rate = 2%
Rp = 0.10  # Risky portfolio return = 10%
sigma_p = 0.15  # Risky portfolio risk = 15%

# w_rf = weight in risk-free (can be negative for borrowing!)
w_rf_values = np.linspace(1.5, -0.5, 100)  # From 150% to -50%

portfolio_returns = []
portfolio_risks = []
labels = []

for w_rf in w_rf_values:
    w_risky = 1 - w_rf  # Weight in risky portfolio
    
    # Portfolio return: mix of risk-free and risky
    R_portfolio = w_rf * Rf + w_risky * Rp
    
    # Portfolio risk: only risky part contributes!
    sigma_portfolio = abs(w_risky) * sigma_p  # abs() because borrowing increases risk
    
    portfolio_returns.append(R_portfolio)
    portfolio_risks.append(sigma_portfolio)
    
    # Mark special points
    if abs(w_rf - 1.0) < 0.01:  # 100% risk-free
        labels.append(("100% Risk-Free", sigma_portfolio, R_portfolio))
    elif abs(w_rf - 0.0) < 0.01:  # 100% risky
        labels.append(("100% Risky", sigma_portfolio, R_portfolio))
    elif abs(w_rf + 0.5) < 0.01:  # Borrowing 50%
        labels.append(("Borrow 50%", sigma_portfolio, R_portfolio))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(portfolio_risks, portfolio_returns, 'b-', linewidth=3)

# Mark special points
for label, risk, ret in labels:
    plt.scatter(risk, ret, s=150, zorder=5)
    plt.annotate(label, xy=(risk, ret), xytext=(10, 10), 
                 textcoords='offset points', fontweight='bold')

plt.xlabel('Portfolio Risk (σ)', fontsize=12)
plt.ylabel('Portfolio Return (E[R])', fontsize=12)
plt.title('Mixing Risk-Free with Risky Portfolio = STRAIGHT LINE!', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

print("="*60)
print("KEY INSIGHTS:")
print("="*60)
print("\n1. When w_rf = 1.0 (100% risk-free):")
print(f"   Return = {Rf:.1%}, Risk = 0%")
print("\n2. When w_rf = 0.0 (100% risky):")
print(f"   Return = {Rp:.1%}, Risk = {sigma_p:.1%}")
print("\n3. When w_rf = -0.5 (borrow 50% to invest 150% in risky):")
print(f"   Return = {1.5*Rp - 0.5*Rf:.1%}, Risk = {1.5*sigma_p:.1%}")
print("\n4. IMPORTANT: The line is STRAIGHT because:")
print("   Risk = |w_risky| × σ_p  (linear!)")
print("   Return = w_rf×Rf + w_risky×Rp  (linear!)")
print("="*60)