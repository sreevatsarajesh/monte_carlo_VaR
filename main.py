from pathlib import Path

import pandas as pd

from config import (
    ASSET_NAMES,
    CONFIDENCE_LEVELS,
    N_SIMULATIONS,
    PORTFOLIO_VALUE,
    WEIGHTS,
)
from src.metrics import summarize_pnl, summarize_returns
from src.portfolio import calculate_portfolio_pnl, calculate_portfolio_returns
from src.risk import calculate_expected_shortfall, calculate_var
from src.simulation import simulate_asset_returns
from src.visualization import plot_portfolio_risk


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    simulated_returns = simulate_asset_returns()
    portfolio_returns = calculate_portfolio_returns(simulated_returns)
    pnl = calculate_portfolio_pnl(portfolio_returns)

    var_95 = calculate_var(pnl, 0.95)
    var_99 = calculate_var(pnl, 0.99)
    es_95 = calculate_expected_shortfall(pnl, 0.95)

    return_summary = summarize_returns(portfolio_returns)
    pnl_summary = summarize_pnl(pnl)

    print("\n=== Monte Carlo Portfolio Risk Simulator ===")
    print(f"Simulations: {N_SIMULATIONS:,}")
    print(f"Portfolio value: ₹{PORTFOLIO_VALUE:,.0f}")
    print(f"Weights: {dict(zip(ASSET_NAMES, WEIGHTS))}")
    print(f"Mean daily return: {return_summary['mean']:.4%}")
    print(f"Daily volatility: {return_summary['volatility']:.4%}")
    print(f"Probability of a loss: {pnl_summary['loss_probability']:.2%}")
    print(f"95% VaR: ₹{var_95:,.2f}")
    print(f"99% VaR: ₹{var_99:,.2f}")
    print(f"95% Expected Shortfall: ₹{es_95:,.2f}")

    concentrated_pnl = calculate_portfolio_pnl(simulated_returns[:, 0])
    concentrated_var = calculate_var(concentrated_pnl, 0.95)
    reduction = (
        (concentrated_var - var_95) / concentrated_var * 100
        if concentrated_var > 0
        else 0.0
    )

    print("\n--- Diversification Experiment ---")
    print(f"Asset A only — 95% VaR: ₹{concentrated_var:,.2f}")
    print(f"Diversified — 95% VaR: ₹{var_95:,.2f}")
    print(f"VaR reduction: {reduction:.2f}%")

    plot_portfolio_risk(
        pnl=pnl,
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        output_path=results_dir / "portfolio_risk.png",
    )

    asset_returns_df = pd.DataFrame(simulated_returns, columns=ASSET_NAMES)
    asset_returns_df.to_csv(results_dir / "simulated_asset_returns.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {"metric": "mean_daily_return", "value": return_summary["mean"]},
            {"metric": "daily_volatility", "value": return_summary["volatility"]},
            {"metric": "95%_VaR_INR", "value": var_95},
            {"metric": "99%_VaR_INR", "value": var_99},
            {"metric": "95%_Expected_Shortfall_INR", "value": es_95},
            {"metric": "asset_A_only_95%_VaR_INR", "value": concentrated_var},
            {"metric": "VaR_reduction_percent", "value": reduction},
        ]
    )
    summary_df.to_csv(results_dir / "risk_summary.csv", index=False)
    print("\nSaved charts and CSV outputs in results/.")


if __name__ == "__main__":
    main()
