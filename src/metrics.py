import numpy as np
def summarize_returns(returns):
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        raise ValueError("Returns cannot be empty.")
    return {
        "mean": float(np.mean(returns)),
        "volatility": float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0,
        "minimum": float(np.min(returns)),
        "maximum": float(np.max(returns)),
        "median": float(np.median(returns)),
    }
def summarize_pnl(pnl):
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        raise ValueError("P&L cannot be empty.")
    return {
        "mean_pnl": float(np.mean(pnl)),
        "profit_probability": float(np.mean(pnl > 0)),
        "loss_probability": float(np.mean(pnl < 0)),
        "worst_loss": float(max(0.0, -np.min(pnl))),
        "best_profit": float(max(0.0, np.max(pnl))),
    }
