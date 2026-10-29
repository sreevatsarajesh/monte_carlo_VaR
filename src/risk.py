from src.risk import calculate_var
from src.portfolio import calculate_portfolio_pnl
from config import PORTFOLIO_VALUE
import numpy as np
def calculate_var(pnl, confidence_level=0.95):
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        raise ValueError("P&L array cannot be empty.")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    losses = -pnl
    return max(0.0, float(np.quantile(losses, confidence_level)))
def calculate_expected_shortfall(pnl, confidence_level=0.95):
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        raise ValueError("P&L array cannot be empty.")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    losses = -pnl
    var = np.quantile(losses, confidence_level)
    tail_losses = losses[losses >= var]
    return max(0.0, float(tail_losses.mean()))
concentrated_returns = simulated_returns[:, 0]
diversified_returns = portfolio_returns
concentrated_pnl = calculate_portfolio_pnl(concentrated_returns)
diversified_pnl = calculate_portfolio_pnl(diversified_returns)
concentrated_var = calculate_var(concentrated_pnl, 0.95)
diversified_var = calculate_var(diversified_pnl, 0.95)
print("\n--- Diversification Experiment ---")
print(f"Asset A only — 95% VaR: ₹{concentrated_var:,.2f}")
print(f"Diversified — 95% VaR: ₹{diversified_var:,.2f}")
reduction = (concentrated_var - diversified_var) / concentrated_var * 100
print(f"VaR reduction: {reduction:.2f}%")