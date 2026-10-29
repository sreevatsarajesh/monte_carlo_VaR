from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_portfolio_risk(pnl, var_95, var_99, es_95, output_path="results/portfolio_risk.png"):
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        raise ValueError("P&L cannot be empty.")

    losses = -pnl
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(pnl, bins=80, edgecolor="black", alpha=0.75)
    axes[0].axvline(0, linestyle="--", label="Break-even")
    axes[0].set_title("Simulated Portfolio P&L")
    axes[0].set_xlabel("Profit / Loss (₹)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].hist(losses, bins=80, edgecolor="black", alpha=0.75)
    axes[1].axvline(var_95, linestyle="--", label=f"95% VaR: ₹{var_95:,.0f}")
    axes[1].axvline(var_99, linestyle="--", label=f"99% VaR: ₹{var_99:,.0f}")
    axes[1].axvline(es_95, linestyle=":", label=f"95% ES: ₹{es_95:,.0f}")
    axes[1].set_title("Portfolio Loss Distribution")
    axes[1].set_xlabel("Loss (₹)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
