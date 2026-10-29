import numpy as np
import matplotlib.pyplot as plt
def plot_portfolio_risk(pnl, var_95, var_99, es_95):
    pnl = np.asarray(pnl)
    losses = -pnl
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
    plt.tight_layout()
    plt.savefig("results/portfolio_risk.png", dpi=150)
    plt.show()
