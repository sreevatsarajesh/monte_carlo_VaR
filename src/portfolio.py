import numpy as np
from config import PORTFOLIO_VALUE, WEIGHTS
def calculate_portfolio_returns(simulated_returns):
    weights = np.asarray(WEIGHTS,dtype=float)
    if simulated_returns.ndim != 2:
        raise ValueError("Returns must be a 2D array")
    if simulated_returns.shape[1] != len(weights):
        raise ValueError("no. of assets should match no. of weights")
    if not np.isclose(weights.sum(),1.0):
        raise ValueError("weights must sum to 1")
    return simulated_returns @ weights
def calculate_portfolio_pnl(portfolio_returns):
    return PORTFOLIO_VALUE * np.asarray(portfolio_returns)
