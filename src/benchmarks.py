from statistics import NormalDist
from time import perf_counter
import numpy as np
from config import (
    ANNUAL_RETURNS,
    ANNUAL_VOLATILITIES,
    CORRELATION_MATRIX,
    PORTFOLIO_VALUE,
    RANDOM_SEED,
    TRADING_DAYS,
    WEIGHTS,
)
from src.portfolio import calculate_portfolio_returns, calculate_portfolio_pnl
from src.risk import calculate_var
from src.simulation import simulate_asset_returns
def calculate_parametric_var(
    confidence_level=0.95,
    portfolio_value=PORTFOLIO_VALUE,
    weights=WEIGHTS,
    annual_returns=ANNUAL_RETURNS,
    annual_volatilities=ANNUAL_VOLATILITIES,
    correlation_matrix=CORRELATION_MATRIX,
    trading_days=TRADING_DAYS,
):
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive.")
    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")
    weights = np.asarray(weights, dtype=float)
    annual_returns = np.asarray(annual_returns, dtype=float)
    annual_volatilities = np.asarray(annual_volatilities, dtype=float)
    correlation = np.asarray(correlation_matrix, dtype=float)
    n = len(weights)
    if n == 0 or annual_returns.shape != (n,) or annual_volatilities.shape != (n,):
        raise ValueError("Weights, returns, and volatilities must have matching nonzero lengths.")
    if correlation.shape != (n, n):
        raise ValueError("Correlation matrix dimensions do not match the assets.")
    if not all(
        np.all(np.isfinite(array))
        for array in (weights, annual_returns, annual_volatilities, correlation)
    ):
        raise ValueError("Inputs must contain only finite values.")
    if np.any(annual_volatilities < 0):
        raise ValueError("Volatilities cannot be negative.")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must be nonnegative and sum to 1.")
    if not np.allclose(correlation, correlation.T, atol=1e-8):
        raise ValueError("Correlation matrix must be symmetric.")
    if not np.allclose(np.diag(correlation), 1.0, atol=1e-8):
        raise ValueError("Correlation matrix must have a diagonal of 1.")
    if np.any(np.abs(correlation) > 1):
        raise ValueError("Correlations must lie between -1 and 1.")
    daily_means = annual_returns / trading_days
    daily_volatilities = annual_volatilities / np.sqrt(trading_days)
    covariance = (
        np.outer(daily_volatilities, daily_volatilities) * correlation
    )
    portfolio_mean = float(weights @ daily_means)
    portfolio_variance = float(weights @ covariance @ weights)
    portfolio_std = float(np.sqrt(max(portfolio_variance, 0.0)))
    z_score = NormalDist().inv_cdf(confidence_level)
    estimated_loss = portfolio_value * (
        z_score * portfolio_std - portfolio_mean
    )
    return max(0.0, float(estimated_loss))
def calculate_historical_var(
    historical_portfolio_returns,
    confidence_level=0.95,
    portfolio_value=PORTFOLIO_VALUE,
):
    returns = np.asarray(historical_portfolio_returns, dtype=float)
    if returns.ndim != 1 or returns.size == 0:
        raise ValueError("Historical returns must be a nonempty 1D array.")
    if not np.all(np.isfinite(returns)):
        raise ValueError("Historical returns must be finite.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive.")
    historical_losses = -returns * portfolio_value
    return max(
        0.0,
        float(np.quantile(historical_losses, confidence_level)),
    )
def run_convergence_experiment(
    simulation_counts=(1000, 5000, 10000, 50000),
    confidence_level=0.95,
    seed=RANDOM_SEED,
):
    results = []
    for count in simulation_counts:
        if not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError("Simulation counts must be positive integers.")
        start = perf_counter()
        simulated_returns = simulate_asset_returns(
            n_simulations=int(count),
            seed=int(seed),
        )
        portfolio_returns = calculate_portfolio_returns(simulated_returns)
        pnl = calculate_portfolio_pnl(portfolio_returns)
        var = calculate_var(pnl, confidence_level)
        elapsed = perf_counter() - start
        results.append({
            "simulations": int(count),
            "confidence_level": float(confidence_level),
            "var": float(var),
            "runtime_seconds": float(elapsed),
        })
    return results
