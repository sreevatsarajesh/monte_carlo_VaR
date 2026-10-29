import numpy as np


def summarize_returns(returns):
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0 or returns.ndim != 1:
        raise ValueError("Returns must be a non-empty one-dimensional array.")
    if not np.isfinite(returns).all():
        raise ValueError("Returns must be finite.")

    return {
        "mean": float(np.mean(returns)),
        "volatility": float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0,
        "minimum": float(np.min(returns)),
        "maximum": float(np.max(returns)),
        "median": float(np.median(returns)),
    }


def summarize_pnl(pnl):
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0 or pnl.ndim != 1:
        raise ValueError("P&L must be a non-empty one-dimensional array.")
    if not np.isfinite(pnl).all():
        raise ValueError("P&L must be finite.")

    return {
        "mean_pnl": float(np.mean(pnl)),
        "profit_probability": float(np.mean(pnl > 0)),
        "loss_probability": float(np.mean(pnl < 0)),
        "worst_loss": float(max(0.0, -np.min(pnl))),
        "best_profit": float(max(0.0, np.max(pnl))),
    }
