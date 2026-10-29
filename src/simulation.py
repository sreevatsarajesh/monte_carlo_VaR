import numpy as np

from config import (
    ANNUAL_RETURNS,
    ANNUAL_VOLATILITIES,
    CORRELATION_MATRIX,
    N_SIMULATIONS,
    RANDOM_SEED,
    TRADING_DAYS,
)


def simulate_asset_returns(n_simulations=N_SIMULATIONS, seed=RANDOM_SEED):
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    annual_returns = np.asarray(ANNUAL_RETURNS, dtype=float)
    annual_volatilities = np.asarray(ANNUAL_VOLATILITIES, dtype=float)
    correlation = np.asarray(CORRELATION_MATRIX, dtype=float)

    n_assets = len(annual_returns)
    if annual_volatilities.shape != (n_assets,):
        raise ValueError("Returns and volatilities must have matching lengths.")
    if correlation.shape != (n_assets, n_assets):
        raise ValueError("Correlation matrix shape does not match asset count.")
    if not np.allclose(correlation, correlation.T):
        raise ValueError("Correlation matrix must be symmetric.")
    if not np.allclose(np.diag(correlation), 1.0):
        raise ValueError("Correlation matrix diagonal must be all ones.")
    if np.any(annual_volatilities < 0):
        raise ValueError("Volatilities cannot be negative.")

    daily_means = annual_returns / TRADING_DAYS
    daily_volatilities = annual_volatilities / np.sqrt(TRADING_DAYS)
    covariance = np.outer(daily_volatilities, daily_volatilities) * correlation

    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(
        mean=daily_means,
        cov=covariance,
        size=n_simulations,
        check_valid="raise",
    )
