# Quantitative Finance Portfolio

Modern Portfolio Theory implementation from scratch - efficient frontiers, utility optimization, and capital market theory.

## Quick Start

```bash
pip install numpy matplotlib
python run_all.py
```

## Files

| File | Description |
|------|-------------|
| `efficient_frontier.py` | 2-asset portfolio basics |
| `utility_theory.py` | Risk aversion & utility functions |
| `3_asset_efficient_frontier.py` | Multi-asset optimization |
| `efficient_frontier_enhanced.py` | 5-asset professional implementation |
| `utility_theory_basic.py` | Simple utility theory |
| `utility_theory_enhanced.py` | 4-asset utility optimization with 3D viz |
| `basic_riskfreeasset.py` | Risk-free asset & CAL |
| `risk_free_story_friends.py` | Capital Market Line & One Fund Theorem |

## Core Concepts

**Mathematics:**
```
Portfolio Return:     Rp = w₁R₁ + w₂R₂
Portfolio Risk:       σp = √(wᵀΣw)
Sharpe Ratio:         S = (Rp - Rf)/σp
Utility Function:     U = E[R] - 0.5λσ²
Capital Market Line:  E[R] = Rf + S × σ
```

**Topics:**
- Efficient frontier construction
- Covariance matrices & correlation
- Monte Carlo portfolio generation
- Minimum Variance Portfolio (MVP)
- Risk-free asset integration
- Tangency portfolio (max Sharpe)
- Separation Theorem / One Fund Theorem

## Features

- Interactive visualizations (click portfolios to see weights)
- Risk aversion sliders
- 3D utility surfaces
- Color-coded Sharpe ratio plots
- Dual-panel dashboards

## Theory

Based on:
- Modern Portfolio Theory (Markowitz, 1952)
- CAPM (Sharpe, 1964)
- Separation Theorem (Tobin, 1958)

---

**Status:** Learning project | **Last Updated:** January 7, 2025
