"""
Unified one-day Monte Carlo showdown benchmark.

Copy this file into the ROOT of BOTH project repositories and run:
    python showdown_benchmark.py

Optional:
    python showdown_benchmark.py --counts 10000 50000 100000 250000 --repeats 7

The benchmark deliberately measures a ONE-DAY, THREE-ASSET workload.
It adapts to the two project APIs but calculates all reported risk metrics
with the same independent definitions so the outputs are comparable.

Important:
- This is a one-day benchmark, not a comparison of full-year path generation.
- The two engines use different stochastic models (normal simple returns
  versus one-step GBM). Their sample outputs need not match exactly.
- Runtime includes the engine call and portfolio-return construction.
- tracemalloc peak is Python-traced allocation, not total process RSS.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np


# Ensure imports resolve from the repository root when this file is run by path.
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Frozen common benchmark specification.
INITIAL_CAPITAL = 1_000_000.0
INITIAL_PRICES = np.array([100.0, 120.0, 80.0], dtype=float)
WEIGHTS = np.array([0.50, 0.30, 0.20], dtype=float)
ANNUAL_DRIFT = np.array([0.10, 0.08, 0.12], dtype=float)
ANNUAL_VOLATILITY = np.array([0.20, 0.15, 0.25], dtype=float)
CORRELATION = np.array(
    [
        [1.00, 0.60, 0.30],
        [0.60, 1.00, 0.40],
        [0.30, 0.40, 1.00],
    ],
    dtype=float,
)
TRADING_DAYS = 252
SEED = 42
CONFIDENCE_LEVELS = (0.95, 0.99)
TOL = 1e-10


def detect_engine():
    """Load whichever project API exists in the current repository."""
    try:
        from src.simulation import simulate_asset_returns
        from config import (
            ANNUAL_RETURNS,
            ANNUAL_VOLATILITIES,
            CORRELATION_MATRIX,
            TRADING_DAYS as PROJECT_TRADING_DAYS,
            WEIGHTS as PROJECT_WEIGHTS,
            PORTFOLIO_VALUE,
        )

        checks = {
            "annual_returns": np.allclose(ANNUAL_RETURNS, ANNUAL_DRIFT, atol=TOL),
            "annual_volatilities": np.allclose(
                ANNUAL_VOLATILITIES, ANNUAL_VOLATILITY, atol=TOL
            ),
            "correlation": np.allclose(CORRELATION_MATRIX, CORRELATION, atol=TOL),
            "weights": np.allclose(PROJECT_WEIGHTS, WEIGHTS, atol=TOL),
            "trading_days": PROJECT_TRADING_DAYS == TRADING_DAYS,
            "portfolio_value": np.isclose(PORTFOLIO_VALUE, INITIAL_CAPITAL),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                "Your config.py does not match the frozen showdown inputs: "
                + ", ".join(failed)
                + ". Update config.py to the values printed in this script "
                  "or ask for an adapter that accepts explicit parameters."
            )

        def run(count, seed):
            asset_returns = simulate_asset_returns(
                n_simulations=count,
                seed=seed,
            )
            if asset_returns.shape != (count, len(WEIGHTS)):
                raise RuntimeError(
                    f"Unexpected return array shape: {asset_returns.shape}"
                )
            return asset_returns, asset_returns @ WEIGHTS

        return "normal_simple_returns", run

    except ModuleNotFoundError as exc:
        # If src is absent, try the friend's top-level GBM API.
        if exc.name not in {"src", "src.simulation"}:
            raise

    from simulation import simulate_gbm

    def run(count, seed):
        price_paths = simulate_gbm(
            initial_prices=INITIAL_PRICES,
            drift=ANNUAL_DRIFT,
            volatility=ANNUAL_VOLATILITY,
            correlation=CORRELATION,
            time_horizon=1.0 / TRADING_DAYS,
            steps=1,
            n_simulations=count,
            seed=seed,
        )
        if price_paths.shape != (count, 2, len(WEIGHTS)):
            raise RuntimeError(f"Unexpected price-path shape: {price_paths.shape}")

        asset_returns = price_paths[:, -1, :] / INITIAL_PRICES - 1.0
        portfolio_returns = asset_returns @ WEIGHTS
        return price_paths, portfolio_returns

    return "gbm_one_day", run


def risk_summary(portfolio_returns):
    """Common metrics: losses are positive; ES averages the worst 1-alpha tail."""
    returns = np.asarray(portfolio_returns, dtype=float)
    if returns.ndim != 1 or returns.size < 2 or not np.isfinite(returns).all():
        raise ValueError("Engine produced invalid portfolio returns.")

    pnl = INITIAL_CAPITAL * returns
    losses = -pnl
    result = {
        "mean_portfolio_return": float(np.mean(returns)),
        "daily_volatility_sample": float(np.std(returns, ddof=1)),
        "probability_of_loss": float(np.mean(returns < 0.0)),
        "mean_pnl": float(np.mean(pnl)),
    }

    ordered_losses = np.sort(losses)
    n = len(ordered_losses)
    for alpha in CONFIDENCE_LEVELS:
        var = float(np.quantile(ordered_losses, alpha, method="linear"))
        # Fractional-tail ES: average exactly the worst (1-alpha) fraction.
        tail_mass = (1.0 - alpha) * n
        full = int(np.floor(tail_mass))
        fraction = tail_mass - full
        worst = ordered_losses[::-1]
        tail_sum = float(np.sum(worst[:full]))
        if fraction > 0:
            tail_sum += fraction * float(worst[full])
        es = tail_sum / tail_mass
        label = str(int(alpha * 100))
        result[f"var_{label}_currency"] = max(0.0, var)
        result[f"es_{label}_currency"] = max(0.0, es)

    return result


def one_run(engine_run, count, seed):
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    raw_output, portfolio_returns = engine_run(count, seed)
    elapsed = time.perf_counter() - started
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summary = risk_summary(portfolio_returns)
    summary.update(
        {
            "runtime_seconds": float(elapsed),
            "simulations_per_second": float(count / elapsed) if elapsed else None,
            "peak_traced_memory_mb": float(peak_traced / (1024**2)),
            "returned_array_bytes_mb": float(raw_output.nbytes / (1024**2)),
            "output_shape": list(raw_output.shape),
        }
    )
    del raw_output, portfolio_returns
    return summary


def main():
    parser = argparse.ArgumentParser(description="Unified one-day quant showdown benchmark")
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=[10_000, 50_000, 100_000, 250_000],
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default="showdown_results.json")
    args = parser.parse_args()

    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    if not args.counts or any(n < 2 for n in args.counts):
        parser.error("every simulation count must be at least 2")

    engine_name, engine_run = detect_engine()

    print("=" * 68)
    print("UNIFIED QUANT SHOWDOWN — ONE-DAY / THREE-ASSET BENCHMARK")
    print("=" * 68)
    print(f"Detected engine:       {engine_name}")
    print(f"Python:                {platform.python_version()}")
    print(f"NumPy:                 {np.__version__}")
    print(f"Platform:              {platform.platform()}")
    print(f"Processor:             {platform.processor() or 'Not reported'}")
    print(f"Capital:               {INITIAL_CAPITAL:,.2f}")
    print(f"Weights:               {WEIGHTS.tolist()}")
    print(f"Annual drift:          {ANNUAL_DRIFT.tolist()}")
    print(f"Annual volatility:     {ANNUAL_VOLATILITY.tolist()}")
    print(f"Trading days:          {TRADING_DAYS}")
    print(f"Seed:                  {args.seed}")
    print(f"Warm-up:               1 run per size")
    print(f"Measured repetitions:  {args.repeats}")
    print("-" * 68)

    results = []
    for count in args.counts:
        print(f"\n{count:,} simulations")
        # Warm-up excluded from measured results.
        one_run(engine_run, count, args.seed)

        runs = [one_run(engine_run, count, args.seed) for _ in range(args.repeats)]
        median_runtime = statistics.median(r["runtime_seconds"] for r in runs)
        median_memory = statistics.median(
            r["peak_traced_memory_mb"] for r in runs
        )
        median_output = statistics.median(
            r["returned_array_bytes_mb"] for r in runs
        )

        row = {
            "simulation_count": count,
            "repeats": args.repeats,
            "seed": args.seed,
            "median_runtime_seconds": median_runtime,
            "median_simulations_per_second": (
                count / median_runtime if median_runtime else None
            ),
            "median_peak_traced_memory_mb": median_memory,
            "returned_array_bytes_mb": median_output,
            "median_mean_portfolio_return": statistics.median(
                r["mean_portfolio_return"] for r in runs
            ),
            "median_daily_volatility_sample": statistics.median(
                r["daily_volatility_sample"] for r in runs
            ),
            "median_probability_of_loss": statistics.median(
                r["probability_of_loss"] for r in runs
            ),
            "median_var_95_currency": statistics.median(
                r["var_95_currency"] for r in runs
            ),
            "median_es_95_currency": statistics.median(
                r["es_95_currency"] for r in runs
            ),
            "median_var_99_currency": statistics.median(
                r["var_99_currency"] for r in runs
            ),
            "median_es_99_currency": statistics.median(
                r["es_99_currency"] for r in runs
            ),
            "run_details": runs,
        }
        results.append(row)
        print(
            f"  median runtime: {median_runtime:.6f}s | "
            f"throughput: {count / median_runtime:,.0f} sims/s | "
            f"peak traced: {median_memory:.2f} MB"
        )
        print(
            f"  VaR95: {row['median_var_95_currency']:,.2f} | "
            f"ES95: {row['median_es_95_currency']:,.2f} | "
            f"VaR99: {row['median_var_99_currency']:,.2f} | "
            f"ES99: {row['median_es_99_currency']:,.2f}"
        )

    payload = {
        "benchmark": "Unified one-day, three-asset Monte Carlo showdown",
        "engine": engine_name,
        "configuration": {
            "initial_capital": INITIAL_CAPITAL,
            "initial_prices": INITIAL_PRICES.tolist(),
            "weights": WEIGHTS.tolist(),
            "annual_drift": ANNUAL_DRIFT.tolist(),
            "annual_volatility": ANNUAL_VOLATILITY.tolist(),
            "correlation": CORRELATION.tolist(),
            "trading_days": TRADING_DAYS,
            "horizon_years": 1.0 / TRADING_DAYS,
            "steps": 1,
            "confidence_levels": list(CONFIDENCE_LEVELS),
            "seed": args.seed,
            "warmup_runs": 1,
            "measured_repeats": args.repeats,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "results": results,
        "notes": [
            "Normal simple-return and one-step GBM models are different distributions.",
            "Risk metrics use common independent calculations and positive-loss convention.",
            "Runtime includes simulation and portfolio-return construction.",
            "tracemalloc reports traced allocations, not total process RSS.",
            "Returned array sizes differ because the engines return different structures.",
        ],
    }

    output_path = Path(args.output).resolve()
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + "=" * 68)
    print(f"Saved results: {output_path}")
    print("Share this JSON together with the console output.")
    print("=" * 68)


if __name__ == "__main__":
    main()
