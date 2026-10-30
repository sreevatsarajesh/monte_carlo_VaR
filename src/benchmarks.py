
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
from src.simulation import simulate_asset_returns
from src.portfolio import calculate_portfolio_returns, calculate_portfolio_pnl
from src.risk import calculate_var


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

    if portfolio_value <= 0 or trading_days <= 0:
        raise ValueError("Portfolio value and trading days must be positive.")

    weights = np.asarray(weights, dtype=float)
    annual_returns = np.asarray(annual_returns, dtype=float)
    annual_volatilities = np.asarray(annual_volatilities, dtype=float)
    correlation = np.asarray(correlation_matrix, dtype=float)

    n = len(weights)

    if n == 0 or annual_returns.shape != (n,) or annual_volatilities.shape != (n,):
        raise ValueError("Weights, returns and volatilities must have matching lengths.")

    if correlation.shape != (n, n):
        raise ValueError("Correlation matrix dimensions do not match assets.")

    if not all(
        np.isfinite(x).all()
        for x in (weights, annual_returns, annual_volatilities, correlation)
    ):
        raise ValueError("Inputs must contain only finite values.")

    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must be nonnegative and sum to 1.")

    if np.any(annual_volatilities < 0):
        raise ValueError("Volatilities cannot be negative.")

    if not np.allclose(correlation, correlation.T):
        raise ValueError("Correlation matrix must be symmetric.")

    if not np.allclose(np.diag(correlation), 1.0):
        raise ValueError("Correlation matrix diagonal must equal 1.")

    if np.any(np.abs(correlation) > 1):
        raise ValueError("Correlations must lie between -1 and 1.")

    daily_means = annual_returns / trading_days
    daily_volatilities = annual_volatilities / np.sqrt(trading_days)

    covariance = np.outer(daily_volatilities, daily_volatilities) * correlation

    portfolio_mean = float(weights @ daily_means)
    portfolio_variance = float(weights @ covariance @ weights)
    portfolio_std = np.sqrt(max(portfolio_variance, 0.0))

    z_score = NormalDist().inv_cdf(confidence_level)

    var = portfolio_value * (
        z_score * portfolio_std - portfolio_mean
    )

    return max(0.0, float(var))


def calculate_historical_var(
    asset_returns,
    confidence_level=0.95,
    portfolio_value=PORTFOLIO_VALUE,
    weights=WEIGHTS,
):
    values = np.asarray(asset_returns, dtype=float)

    if values.ndim not in (1, 2) or values.shape[0] < 2:
        raise ValueError("Returns must be a 1D or 2D array with at least two observations.")

    if not np.isfinite(values).all():
        raise ValueError("Returns must contain only finite values.")

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    if portfolio_value <= 0:
        raise ValueError("Portfolio value must be positive.")

    if values.ndim == 1:
        portfolio_returns = values
    else:
        weights = np.asarray(weights, dtype=float)

        if values.shape[1] != len(weights):
            raise ValueError("Asset-return columns must match portfolio weights.")

        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Weights must be nonnegative and sum to 1.")

        portfolio_returns = values @ weights

    losses = -portfolio_returns * portfolio_value

    return max(
        0.0,
        float(np.quantile(losses, confidence_level)),
    )


def run_convergence_experiment(
    counts=(1000, 5000, 10000, 50000),
    confidence_level=0.95,
    seed=RANDOM_SEED,
):
    results = []

    for count in counts:
        if not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError("Simulation counts must be positive integers.")

        start = perf_counter()

        simulated = simulate_asset_returns(
            n_simulations=int(count),
            seed=int(seed),
        )

        portfolio_returns = calculate_portfolio_returns(simulated)
        pnl = calculate_portfolio_pnl(portfolio_returns)

        results.append({
            "simulations": int(count),
            "var": float(calculate_var(pnl, confidence_level)),
            "runtime_seconds": float(perf_counter() - start),
        })

    return results


def benchmark_engine(count=50000, seed=RANDOM_SEED):
    if count <= 0:
        raise ValueError("Simulation count must be positive.")

    start = perf_counter()

    simulated = simulate_asset_returns(
        n_simulations=int(count),
        seed=int(seed),
    )

    returns = calculate_portfolio_returns(simulated)
    pnl = calculate_portfolio_pnl(returns)

    elapsed = perf_counter() - start

    return {
        "simulation_count": int(count),
        "runtime_seconds": float(elapsed),
        "mean_portfolio_return": float(np.mean(returns)),
        "daily_volatility": float(np.std(returns, ddof=1)),
        "var_95": float(calculate_var(pnl, 0.95)),
        "var_99": float(calculate_var(pnl, 0.99)),
        "seed": int(seed),
    }
