
import csv
from pathlib import Path

from config import (
    ANNUAL_RETURNS,
    ANNUAL_VOLATILITIES,
    CORRELATION_MATRIX,
    PORTFOLIO_VALUE,
    RANDOM_SEED,
    WEIGHTS,
)
from src.benchmarks import (
    calculate_parametric_var,
    run_convergence_experiment,
)
from src.portfolio import calculate_portfolio_returns, calculate_portfolio_pnl
from src.risk import calculate_var
from src.simulation import simulate_asset_returns


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    simulated = simulate_asset_returns(
        n_simulations=50000,
        seed=RANDOM_SEED,
    )
    portfolio_returns = calculate_portfolio_returns(simulated)
    pnl = calculate_portfolio_pnl(portfolio_returns)

    rows = []

    for confidence in (0.95, 0.99):
        rows.append({
            "experiment": "model_comparison",
            "simulations": len(pnl),
            "confidence_level": confidence,
            "model": "Monte Carlo",
            "var": calculate_var(pnl, confidence),
            "runtime_seconds": "",
            "seed": RANDOM_SEED,
        })
        rows.append({
            "experiment": "model_comparison",
            "simulations": len(pnl),
            "confidence_level": confidence,
            "model": "Parametric Normal",
            "var": calculate_parametric_var(
                confidence_level=confidence,
                portfolio_value=PORTFOLIO_VALUE,
                weights=WEIGHTS,
                annual_returns=ANNUAL_RETURNS,
                annual_volatilities=ANNUAL_VOLATILITIES,
                correlation_matrix=CORRELATION_MATRIX,
            ),
            "runtime_seconds": "",
            "seed": RANDOM_SEED,
        })

    convergence = run_convergence_experiment(
        simulation_counts=(1000, 5000, 10000, 50000),
        confidence_level=0.95,
        seed=RANDOM_SEED,
    )

    for result in convergence:
        rows.append({
            "experiment": "convergence",
            "simulations": result["simulations"],
            "confidence_level": result["confidence_level"],
            "model": "Monte Carlo",
            "var": result["var"],
            "runtime_seconds": result["runtime_seconds"],
            "seed": RANDOM_SEED,
        })

    output_path = results_dir / "quant_benchmark.csv"

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "simulations",
                "confidence_level",
                "model",
                "var",
                "runtime_seconds",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nMODEL COMPARISON")
    print("-" * 65)

    for row in rows:
        if row["experiment"] == "model_comparison":
            print(
                f'{row["model"]:20s} '
                f'{row["confidence_level"]:.0%} VaR: '
                f'₹{row["var"]:,.2f}'
            )

    print("\nCONVERGENCE EXPERIMENT")
    print("-" * 65)

    for result in convergence:
        print(
            f'{result["simulations"]:>6,} scenarios | '
            f'VaR ₹{result["var"]:>12,.2f} | '
            f'{result["runtime_seconds"]:.4f}s'
        )

    print(f"\nSaved benchmark report to {output_path}")


if __name__ == "__main__":
    main()
