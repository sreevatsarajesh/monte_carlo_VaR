import numpy as np


def _validate_pnl(pnl, confidence_level):
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        raise ValueError("P&L array cannot be empty.")
    if pnl.ndim != 1:
        raise ValueError("P&L must be a one-dimensional array.")
    if not np.isfinite(pnl).all():
        raise ValueError("P&L values must be finite.")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    return pnl


def calculate_var(pnl, confidence_level=0.95):
    pnl = _validate_pnl(pnl, confidence_level)
    losses = -pnl
    return max(0.0, float(np.quantile(losses, confidence_level)))


def calculate_expected_shortfall(pnl, confidence_level=0.95):
    pnl = _validate_pnl(pnl, confidence_level)
    losses = -pnl
    var = np.quantile(losses, confidence_level)
    tail_losses = losses[losses >= var]
    return max(0.0, float(tail_losses.mean()))
