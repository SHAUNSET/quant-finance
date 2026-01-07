import numpy as np
import matplotlib.pyplot as plt

print("="*100)
print("INVESTMENT CLUB: From Individual Portfolios to One Optimal Strategy")
print("="*100)

##############################################################################
# CHAPTER 1: THE INITIAL SITUATION
##############################################################################

print("\n" + "="*80)
print("CHAPTER 1: THREE FRIENDS, THREE DIFFERENT STRATEGIES")
print("="*80)

# Each friend has ₹100,000 to invest
initial_wealth = 100000

# Available risky assets
print("\n📊 AVAILABLE RISKY ASSETS:")
print("-"*50)
print("1. BOND FUND (Conservative):")
print(f"   Expected Return = 6.0% per year")
print(f"   Risk (Standard Deviation) = 5.0%")
print("\n2. STOCK FUND (Aggressive):")
print(f"   Expected Return = 12.0% per year")
print(f"   Risk (Standard Deviation) = 15.0%")
print("\n3. Correlation between Bond & Stock funds = 0.3")

# Define variables
R_bond = 0.06     # 6% return
σ_bond = 0.05     # 5% risk
R_stock = 0.12    # 12% return  
σ_stock = 0.15    # 15% risk
ρ = 0.3           # Correlation

print("\n" + "-"*80)
print("THEIR INDIVIDUAL PORTFOLIO FORMULAS:")
print("-"*80)

print("\nFor any portfolio with:")
print("  w_bond = weight in bonds (0 to 1)")
print("  w_stock = weight in stocks (0 to 1)")
print("  where w_bond + w_stock = 1")
print("\nFORMULA 1: PORTFOLIO RETURN")
print("  R_portfolio = w_bond × R_bond + w_stock × R_stock")
print(f"              = w_bond × {R_bond:.1%} + w_stock × {R_stock:.1%}")

print("\nFORMULA 2: PORTFOLIO RISK (Standard Deviation)")
print("  σ_portfolio = √[w_bond² × σ_bond² + w_stock² × σ_stock² + 2 × w_bond × w_stock × ρ × σ_bond × σ_stock]")
print(f"              = √[w_bond² × ({σ_bond:.1%})² + w_stock² × ({σ_stock:.1%})² + 2 × w_bond × w_stock × {ρ} × {σ_bond:.1%} × {σ_stock:.1%}]")

print("\n" + "-"*80)
print("THEIR INITIAL PORTFOLIOS:")
print("-"*80)

# Each friend's chosen weights
friends = {
    "Rahul (Conservative)": {"w_bond": 0.80, "w_stock": 0.20},
    "Priya (Moderate)": {"w_bond": 0.50, "w_stock": 0.50},
    "Rohan (Aggressive)": {"w_bond": 0.20, "w_stock": 0.80}
}

for name, weights in friends.items():
    w_bond = weights["w_bond"]
    w_stock = weights["w_stock"]
    
    # Calculate return
    R_portfolio = w_bond * R_bond + w_stock * R_stock
    
    # Calculate risk
    σ2_portfolio = (w_bond**2 * σ_bond**2 + 
                    w_stock**2 * σ_stock**2 + 
                    2 * w_bond * w_stock * ρ * σ_bond * σ_stock)
    σ_portfolio = np.sqrt(σ2_portfolio)
    
    print(f"\n{name}:")
    print(f"  Allocation: {w_bond:.0%} bonds + {w_stock:.0%} stocks")
    print(f"  Return = {w_bond:.2f} × {R_bond:.1%} + {w_stock:.2f} × {R_stock:.1%} = {R_portfolio:.2%}")
    print(f"  Risk = √[{w_bond**2:.4f}×{σ_bond**2:.4f} + {w_stock**2:.4f}×{σ_stock**2:.4f} + 2×{w_bond:.2f}×{w_stock:.2f}×{ρ}×{σ_bond:.3f}×{σ_stock:.3f}]")
    print(f"       = √[{w_bond**2 * σ_bond**2:.6f} + {w_stock**2 * σ_stock**2:.6f} + {2 * w_bond * w_stock * ρ * σ_bond * σ_stock:.6f}]")
    print(f"       = √[{σ2_portfolio:.6f}] = {σ_portfolio:.3%}")

print("\n" + "="*80)
print("PROBLEM: Each friend is researching and monitoring DIFFERENT portfolios!")
print("="*80)

##############################################################################
# CHAPTER 2: INTRODUCTION OF RISK-FREE ASSET
##############################################################################

print("\n\n" + "="*80)
print("CHAPTER 2: ENTER THE RISK-FREE ASSET")
print("="*80)

Rf = 0.03  # Government bonds at 3% risk-free rate

print(f"\n💵 NEW OPTION: Risk-Free Government Bonds")
print(f"   Guaranteed Return = {Rf:.1%}")
print(f"   Risk = 0% (perfectly safe)")

print("\n" + "-"*80)
print("NEW POSSIBILITY: Mix ANY risky portfolio with risk-free asset")
print("-"*80)

print("\nFORMULA 3: PORTFOLIO WITH RISK-FREE ASSET")
print("  Let w_rf = weight in risk-free asset (can be negative for borrowing)")
print("  Let w_risky = weight in risky portfolio = 1 - w_rf")
print("  where w_rf + w_risky = 1")
print("\n  Return of mix = w_rf × Rf + w_risky × R_risky")
print(f"                = w_rf × {Rf:.1%} + (1 - w_rf) × R_risky")

print("\nFORMULA 4: RISK OF MIX (only risky part contributes)")
print("  σ_mix = |w_risky| × σ_risky")
print("        = |1 - w_rf| × σ_risky")

print("\n" + "-"*80)
print("KEY INSIGHT: Creates a STRAIGHT LINE in risk-return space!")
print("-"*80)
print("  Return = Rf + [(R_risky - Rf) / σ_risky] × Risk")
print("  This is a line with:")
print(f"  - Intercept = Risk-free rate = {Rf:.1%}")
print("  - Slope = Sharpe Ratio of the risky portfolio")

##############################################################################
# CHAPTER 3: FINDING THE BEST RISKY PORTFOLIO (TANGENCY)
##############################################################################

print("\n\n" + "="*80)
print("CHAPTER 3: FINDING THE TANGENCY PORTFOLIO")
print("="*80)

print("\nFORMULA 5: SHARPE RATIO")
print("  Sharpe = (Portfolio Return - Risk-Free Rate) / Portfolio Risk")
print("         = (R_portfolio - Rf) / σ_portfolio")
print(f"  Higher Sharpe = Better risk-adjusted returns")

print("\n" + "-"*80)
print("CALCULATING SHARPE RATIOS FOR EACH FRIEND'S PORTFOLIO:")
print("-"*80)

for name, weights in friends.items():
    w_bond = weights["w_bond"]
    w_stock = weights["w_stock"]
    
    # Calculate portfolio stats
    R_portfolio = w_bond * R_bond + w_stock * R_stock
    σ2_portfolio = (w_bond**2 * σ_bond**2 + 
                    w_stock**2 * σ_stock**2 + 
                    2 * w_bond * w_stock * ρ * σ_bond * σ_stock)
    σ_portfolio = np.sqrt(σ2_portfolio)
    
    sharpe = (R_portfolio - Rf) / σ_portfolio
    
    print(f"\n{name}:")
    print(f"  Return = {R_portfolio:.3f}, Risk = {σ_portfolio:.3f}")
    print(f"  Sharpe = ({R_portfolio:.3f} - {Rf:.3f}) / {σ_portfolio:.3f} = {sharpe:.4f}")

print("\n" + "-"*80)
print("BUT WAIT! Is Priya's portfolio REALLY the best?")
print("Maybe there's a BETTER mix of bonds and stocks!")
print("-"*80)

##############################################################################
# CHAPTER 4: MATHEMATICAL OPTIMIZATION FOR TANGENCY PORTFOLIO
##############################################################################

print("\n\n" + "="*80)
print("CHAPTER 4: OPTIMIZING FOR MAXIMUM SHARPE RATIO")
print("="*80)

print("\nWe need to find w_bond that MAXIMIZES:")
print("  Sharpe(w_bond) = [w_bond×R_bond + (1-w_bond)×R_stock - Rf] /")
print("                   √[w_bond²×σ_bond² + (1-w_bond)²×σ_stock² + 2×w_bond×(1-w_bond)×ρ×σ_bond×σ_stock]")

print("\nMATHEMATICAL SOLUTION (taking derivative = 0):")
print("The optimal weight in bonds is:")
print("  w_bond* = [σ_stock² - ρ×σ_bond×σ_stock] / [σ_bond² + σ_stock² - 2×ρ×σ_bond×σ_stock]")
print("          × (R_bond - Rf) / (R_stock - Rf)")

print("\nCALCULATION:")
print("-"*40)

# Calculate optimal weight using formula
numerator = σ_stock**2 - ρ * σ_bond * σ_stock
denominator = σ_bond**2 + σ_stock**2 - 2 * ρ * σ_bond * σ_stock
factor = (R_bond - Rf) / (R_stock - Rf)

w_bond_optimal = (numerator / denominator) * factor
w_stock_optimal = 1 - w_bond_optimal

print(f"Step 1: Calculate factor = (R_bond - Rf) / (R_stock - Rf)")
print(f"        = ({R_bond:.3f} - {Rf:.3f}) / ({R_stock:.3f} - {Rf:.3f})")
print(f"        = {R_bond - Rf:.3f} / {R_stock - Rf:.3f} = {factor:.4f}")

print(f"\nStep 2: Calculate numerator = σ_stock² - ρ×σ_bond×σ_stock")
print(f"        = {σ_stock**2:.4f} - {ρ:.1f}×{σ_bond:.3f}×{σ_stock:.3f}")
print(f"        = {σ_stock**2:.4f} - {ρ * σ_bond * σ_stock:.4f} = {numerator:.4f}")

print(f"\nStep 3: Calculate denominator = σ_bond² + σ_stock² - 2×ρ×σ_bond×σ_stock")
print(f"        = {σ_bond**2:.4f} + {σ_stock**2:.4f} - 2×{ρ:.1f}×{σ_bond:.3f}×{σ_stock:.3f}")
print(f"        = {σ_bond**2:.4f} + {σ_stock**2:.4f} - {2 * ρ * σ_bond * σ_stock:.4f} = {denominator:.4f}")

print(f"\nStep 4: Calculate w_bond* = (numerator/denominator) × factor")
print(f"        = ({numerator:.4f}/{denominator:.4f}) × {factor:.4f}")
print(f"        = {numerator/denominator:.4f} × {factor:.4f} = {w_bond_optimal:.4f}")

print(f"\nStep 5: Calculate w_stock* = 1 - w_bond*")
print(f"        = 1 - {w_bond_optimal:.4f} = {w_stock_optimal:.4f}")

# Calculate tangency portfolio stats
R_tangency = w_bond_optimal * R_bond + w_stock_optimal * R_stock
σ2_tangency = (w_bond_optimal**2 * σ_bond**2 + 
               w_stock_optimal**2 * σ_stock**2 + 
               2 * w_bond_optimal * w_stock_optimal * ρ * σ_bond * σ_stock)
σ_tangency = np.sqrt(σ2_tangency)
sharpe_tangency = (R_tangency - Rf) / σ_tangency

print("\n" + "="*80)
print("🎯 TANGENCY PORTFOLIO FOUND!")
print("="*80)
print(f"\nOptimal Allocation: {w_bond_optimal:.1%} bonds + {w_stock_optimal:.1%} stocks")
print(f"Expected Return: {R_tangency:.2%}")
print(f"Risk: {σ_tangency:.2%}")
print(f"\nSHARPE RATIO: {sharpe_tangency:.4f}")
print(f"(Higher than ANY individual friend's portfolio!)")

##############################################################################
# CHAPTER 5: THE CAPITAL MARKET LINE (CML)
##############################################################################

print("\n\n" + "="*80)
print("CHAPTER 5: THE CAPITAL MARKET LINE (CML)")
print("="*80)

print("\nFORMULA 6: CAPITAL MARKET LINE EQUATION")
print("  R_CML = Rf + Sharpe_tangency × σ")
print(f"        = {Rf:.3f} + {sharpe_tangency:.4f} × σ")

print("\n" + "-"*80)
print("INTERPRETATION:")
print("-"*80)
print("For ANY desired risk level σ, the CML tells you:")
print("1. The MAXIMUM possible return you can achieve")
print("2. How to achieve it: Mix risk-free asset with tangency portfolio")
print("\nSpecifically, to achieve risk σ:")
print("  Weight in tangency portfolio = σ / σ_tangency")
print("  Weight in risk-free asset = 1 - (σ / σ_tangency)")

print("\n" + "-"*80)
print("HOW EACH FRIEND NOW INVESTs:")
print("-"*80)

# Each friend's desired risk level
risk_targets = {
    "Rahul (Conservative)": 0.05,
    "Priya (Moderate)": 0.08,
    "Rohan (Aggressive)": 0.12
}

for name, σ_target in risk_targets.items():
    # Calculate optimal allocation
    w_tangency = σ_target / σ_tangency
    w_rf = 1 - w_tangency
    
    # Calculate expected return
    R_expected = w_rf * Rf + w_tangency * R_tangency
    
    print(f"\n{name}:")
    print(f"  Desired Risk: {σ_target:.1%}")
    print(f"  Weight in Tangency: {σ_target:.3f} / {σ_tangency:.3f} = {w_tangency:.3f}")
    print(f"  Weight in Risk-Free: 1 - {w_tangency:.3f} = {w_rf:.3f}")
    
    if w_rf < 0:
        print(f"  💰 Action: BORROW {abs(w_rf):.1%} at {Rf:.1%} interest")
    else:
        print(f"  💰 Action: LEND {w_rf:.1%} at {Rf:.1%} interest")
    
    print(f"  Expected Return: {w_rf:.3f}×{Rf:.3f} + {w_tangency:.3f}×{R_tangency:.3f} = {R_expected:.3%}")
    
    # Calculate final wealth after 1 year
    final_wealth = initial_wealth * (1 + R_expected)
    print(f"  Final Wealth after 1 year: ₹{initial_wealth:,.0f} → ₹{final_wealth:,.0f}")

print("\n" + "="*80)
print("THE ONE FUND THEOREM IN ACTION!")
print("="*80)
print("\n✅ All 3 friends invest in the SAME risky portfolio (Tangency)")
print("✅ They only differ in how much they mix it with risk-free")
print("✅ No more researching individual portfolios!")
print("✅ Everyone achieves MAXIMUM possible return for their risk tolerance!")

##############################################################################
# VISUALIZATION
##############################################################################

print("\n\n" + "="*80)
print("VISUAL SUMMARY")
print("="*80)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Individual Portfolios
ax1.set_title("BEFORE: Individual Portfolios", fontsize=14, fontweight='bold')
colors = {'Rahul': 'blue', 'Priya': 'green', 'Rohan': 'red'}

for name, weights in friends.items():
    key = name.split()[0]
    w_bond = weights["w_bond"]
    w_stock = weights["w_stock"]
    
    R_portfolio = w_bond * R_bond + w_stock * R_stock
    σ2_portfolio = (w_bond**2 * σ_bond**2 + 
                    w_stock**2 * σ_stock**2 + 
                    2 * w_bond * w_stock * ρ * σ_bond * σ_stock)
    σ_portfolio = np.sqrt(σ2_portfolio)
    
    ax1.scatter(σ_portfolio, R_portfolio, s=200, color=colors[key], zorder=5)
    ax1.annotate(f"{key}\n({w_bond:.0%}B,{w_stock:.0%}S)", 
                xy=(σ_portfolio, R_portfolio),
                xytext=(10, 10), textcoords='offset points',
                fontweight='bold')

# Plot individual assets
ax1.scatter(σ_bond, R_bond, s=150, color='black', marker='s', label='Bond Fund', zorder=5)
ax1.scatter(σ_stock, R_stock, s=150, color='black', marker='^', label='Stock Fund', zorder=5)

ax1.set_xlabel('Portfolio Risk (σ)', fontsize=12)
ax1.set_ylabel('Portfolio Return (R)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Capital Market Line
ax2.set_title("AFTER: All on Capital Market Line", fontsize=14, fontweight='bold')

# Plot CML
σ_range = np.linspace(0, 0.15, 100)
R_cml = Rf + sharpe_tangency * σ_range
ax2.plot(σ_range, R_cml, 'black', linewidth=3, label=f'CML: R = {Rf:.1%} + {sharpe_tangency:.3f}×σ')

# Mark tangency portfolio
ax2.scatter(σ_tangency, R_tangency, s=250, color='purple', 
           marker='*', label='Tangency Portfolio', zorder=6)

# Mark risk-free asset
ax2.scatter(0, Rf, s=200, color='orange', marker='^', 
           label=f'Risk-Free ({Rf:.1%})', zorder=6)

# Mark each friend on CML
for name, σ_target in risk_targets.items():
    key = name.split()[0]
    R_target = Rf + sharpe_tangency * σ_target
    ax2.scatter(σ_target, R_target, s=150, color=colors[key], zorder=5)
    ax2.annotate(key, xy=(σ_target, R_target),
                xytext=(10, 10), textcoords='offset points',
                fontweight='bold')

ax2.set_xlabel('Portfolio Risk (σ)', fontsize=12)
ax2.set_ylabel('Portfolio Return (R)', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

