import numpy as np
from config import (
    ANNUAL_RETURNS,
    ANNUAL_VOLATILITIES,
    CORRELATION_MATRIX,
    N_SIMULATIONS,
    RANDOM_SEED,
    TRADING_DAYS,
)
def simulate_asset_returns():
    rng = np.random.default_rng(RANDOM_SEED)
    annual_returns = np.asarray(ANNUAL_RETURNS, dtype=float)
    annual_volatilities = np.asarray(ANNUAL_VOLATILITIES, dtype=float)
    correlation = np.asarray(CORRELATION_MATRIX, dtype=float)
    daily_means = annual_returns / TRADING_DAYS
    daily_volatilities = annual_volatilities / np.sqrt(TRADING_DAYS)
    covariance = np.outer(daily_volatilities, daily_volatilities) * correlation
    simulated_returns = rng.multivariate_normal(
        mean=daily_means,
        cov=covariance,
        size=N_SIMULATIONS,
    )
    return simulated_returns
