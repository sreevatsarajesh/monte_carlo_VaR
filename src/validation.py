import numpy as np
from config import ANNUAL_RETURNS, ANNUAL_VOLATILITIES, CORRELATION_MATRIX, TRADING_DAYS

def theoretical_daily_moments():
    mu = np.asarray(ANNUAL_RETURNS, float) / TRADING_DAYS
    vol = np.asarray(ANNUAL_VOLATILITIES, float) / np.sqrt(TRADING_DAYS)
    corr = np.asarray(CORRELATION_MATRIX, float)
    return mu, np.outer(vol, vol) * corr

def validate_simulation(values):
    x = np.asarray(values, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite 2D array with at least two scenarios.")
    mu, cov = theoretical_daily_moments()
    if x.shape[1] != len(mu):
        raise ValueError("Asset count does not match config.")
    sample_mu = x.mean(axis=0)
    sample_cov = np.cov(x, rowvar=False, ddof=1)
    sample_corr = np.corrcoef(x, rowvar=False)
    mean_se = np.sqrt(np.diag(cov) / len(x))
    z = (sample_mu - mu) / np.maximum(mean_se, 1e-15)
    return {
        "n": len(x),
        "theoretical_mean": mu,
        "sample_mean": sample_mu,
        "mean_z_scores": z,
        "theoretical_covariance": cov,
        "sample_covariance": sample_cov,
        "theoretical_correlation": np.asarray(CORRELATION_MATRIX, float),
        "sample_correlation": sample_corr,
        "variance_relative_error": np.diag(sample_cov) / np.diag(cov) - 1,
        "max_abs_mean_z_score": float(np.max(np.abs(z))),
        "max_abs_correlation_error": float(np.max(np.abs(sample_corr - CORRELATION_MATRIX))),
    }

def quantile_rank_uncertainty(n, confidence):
    if n < 2 or not 0 < confidence < 1:
        raise ValueError("Require n >= 2 and confidence between 0 and 1.")
    return float(np.sqrt(confidence * (1 - confidence) / n))
