from src.simulation import simulate_asset_returns
from src.portfolio import calculate_portfolio_returns, calculate_portfolio_pnl
from src.risk import calculate_var, calculate_expected_shortfall
from src.visualization import plot_portfolio_risk
simulated_returns = simulate_asset_returns()
portfolio_returns = calculate_portfolio_returns(simulated_returns)
pnl = calculate_portfolio_pnl(portfolio_returns)
plot_portfolio_risk(
    pnl,
    calculate_var(pnl, 0.95),
    calculate_var(pnl, 0.99),
    calculate_expected_shortfall(pnl, 0.95),
)
print(f"Simulations: {len(pnl):,}")
print(f"Portfolio value: ₹1,000,000")
print(f"Mean daily return: {portfolio_returns.mean():.4%}")
print(f"Daily volatility: {portfolio_returns.std(ddof=1):.4%}")
print(f"95% VaR: ₹{calculate_var(pnl, 0.95):,.2f}")
print(f"99% VaR: ₹{calculate_var(pnl, 0.99):,.2f}")
print(f"95% Expected Shortfall: ₹{calculate_expected_shortfall(pnl, 0.95):,.2f}")
