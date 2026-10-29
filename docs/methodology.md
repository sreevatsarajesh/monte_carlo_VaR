# Methodology

## Objective
Estimate one-day portfolio loss thresholds with Monte Carlo simulation and compare the risk of a concentrated position with a diversified portfolio.

## Inputs
The project uses three illustrative assets, a ₹1,000,000 portfolio, weights of 50%, 30%, and 20%, assumed annual returns, annual volatilities, and a correlation matrix. These are demonstration parameters, not calibrated market forecasts.

## Return simulation
Annual expected returns are divided by 252 to obtain daily means. Annual volatilities are divided by the square root of 252 to obtain daily volatilities. The daily covariance matrix is:

`cov[i, j] = daily_volatility[i] * daily_volatility[j] * correlation[i, j]`

Each scenario is drawn from a multivariate normal distribution with these daily means and covariance. This is a normal approximation for daily simple returns; it is not a full Geometric Brownian Motion price-path simulation.

## Portfolio return and P&L
For asset weights `w` and simulated asset return vector `r`, portfolio return is `r_p = sum(w_i * r_i)`. P&L is `portfolio_value * r_p`. Weights are long-only and sum to one.

## Value at Risk
For confidence level `c`, loss is defined as `L = -P&L`. VaR is the empirical `c` quantile of simulated losses, floored at zero in this implementation. For example, 95% VaR is a model-estimated threshold exceeded by approximately 5% of simulated losses.

## Expected Shortfall
Expected Shortfall is calculated as the average simulated loss at or beyond the empirical VaR threshold. This is an empirical tail estimate and can differ slightly from alternative quantile/tail conventions.

## Diversification comparison
The model compares the 95% VaR of a position entirely in Asset A against the configured 50/30/20 portfolio, using the same simulated asset-return scenarios. The result depends on the illustrative assumptions and is not a universal claim about diversification.

## Limitations
- The assumptions are illustrative and not estimated from historical market data.
- Normal returns may underrepresent fat tails and extreme events.
- The model uses a one-day horizon and fixed weights.
- It excludes transaction costs, liquidity, taxes, changing correlations, and rebalancing.
- VaR and Expected Shortfall are estimates, not guarantees or investment advice.
