import numpy as np
from config import PORTFOLIO_VALUE
from src.simulation import simulate_asset_returns
from src.portfolio import calculate_portfolio_returns, calculate_portfolio_pnl
from src.risk import calculate_var, calculate_expected_shortfall
from src.visualization import plot_portfolio_risk

simulated_returns = simulate_asset_returns()
portfolio_returns = calculate_portfolio_returns(simulated_returns)
pnl = calculate_portfolio_pnl(portfolio_returns)

var_95 = calculate_var(pnl, 0.95)
var_99 = calculate_var(pnl, 0.99)
es_95 = calculate_expected_shortfall(pnl, 0.95)

print(f"Simulations: {len(pnl):,}")
print(f"Portfolio value: ₹{PORTFOLIO_VALUE:,.0f}")
print(f"Mean daily return: {portfolio_returns.mean():.4%}")
print(f"Daily volatility: {portfolio_returns.std(ddof=1):.4%}")
print(f"95% VaR: ₹{var_95:,.2f}")
print(f"99% VaR: ₹{var_99:,.2f}")
print(f"95% Expected Shortfall: ₹{es_95:,.2f}")

plot_portfolio_risk(pnl, var_95, var_99, es_95)

concentrated_pnl = calculate_portfolio_pnl(simulated_returns[:, 0])
concentrated_var = calculate_var(concentrated_pnl, 0.95)

print("\n--- Diversification Experiment ---")
print(f"Asset A only — 95% VaR: ₹{concentrated_var:,.2f}")
print(f"Diversified — 95% VaR: ₹{var_95:,.2f}")

reduction = (concentrated_var - var_95) / concentrated_var * 100
print(f"VaR reduction: {reduction:.2f}%")
