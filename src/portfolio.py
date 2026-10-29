import numpy as np

from config import PORTFOLIO_VALUE, WEIGHTS


def calculate_portfolio_returns(simulated_returns, weights=WEIGHTS):
    simulated_returns = np.asarray(simulated_returns, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if simulated_returns.ndim != 2:
        raise ValueError("simulated_returns must be a 2D array.")
    if simulated_returns.shape[1] != len(weights):
        raise ValueError("Number of assets must match number of weights.")
    if weights.size == 0 or not np.isfinite(weights).all():
        raise ValueError("Weights must be non-empty and finite.")
    if np.any(weights < 0):
        raise ValueError("Weights cannot be negative in this long-only model.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    return simulated_returns @ weights


def calculate_portfolio_pnl(portfolio_returns, portfolio_value=PORTFOLIO_VALUE):
    returns = np.asarray(portfolio_returns, dtype=float)
    if not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise ValueError("Portfolio value must be positive and finite.")
    return portfolio_value * returns
