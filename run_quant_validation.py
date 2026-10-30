import json
from pathlib import Path
import numpy as np
from config import RANDOM_SEED
from src.simulation import simulate_asset_returns
from src.validation import validate_simulation, quantile_rank_uncertainty
from src.benchmarks import calculate_parametric_var, run_convergence_experiment, benchmark_engine
from src.portfolio import calculate_portfolio_returns, calculate_portfolio_pnl
from src.risk import calculate_var

def safe(x):
    if isinstance(x, dict): return {k: safe(v) for k, v in x.items()}
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.integer, np.floating)): return x.item()
    return x

def main():
    scenarios = simulate_asset_returns(n_simulations=100000, seed=RANDOM_SEED)
    report = validate_simulation(scenarios)
    returns = calculate_portfolio_returns(scenarios)
    pnl = calculate_portfolio_pnl(returns)
    report.update({
        "model": "Multivariate normal one-day simple returns (not GBM)",
        "seed": RANDOM_SEED,
        "var_comparison": {
            "monte_carlo_95": calculate_var(pnl, .95),
            "parametric_95": calculate_parametric_var(.95),
            "monte_carlo_99": calculate_var(pnl, .99),
            "parametric_99": calculate_parametric_var(.99)
        },
        "quantile_rank_standard_error_95": quantile_rank_uncertainty(len(pnl), .95),
        "quantile_rank_standard_error_99": quantile_rank_uncertainty(len(pnl), .99),
        "convergence": run_convergence_experiment(),
        "performance": benchmark_engine()
    })
    Path("results").mkdir(exist_ok=True)
    Path("results/quant_validation_report.json").write_text(json.dumps(safe(report), indent=2))
    print(json.dumps(safe(report), indent=2))
    print("Saved results/quant_validation_report.json")

if __name__ == "__main__":
    main()
