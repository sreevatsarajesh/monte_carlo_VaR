import numpy as np
import pytest
from src.simulation import simulate_asset_returns
from src.validation import validate_simulation, theoretical_daily_moments, quantile_rank_uncertainty

def test_mean_within_five_sampling_standard_errors():
    report = validate_simulation(simulate_asset_returns(100000, seed=2026))
    assert report["max_abs_mean_z_score"] < 5

def test_variances_close_to_theory():
    report = validate_simulation(simulate_asset_returns(100000, seed=2026))
    assert np.all(np.abs(report["variance_relative_error"]) < .04)

def test_correlations_close_to_config():
    report = validate_simulation(simulate_asset_returns(100000, seed=2026))
    assert report["max_abs_correlation_error"] < .025

def test_seed_reproducibility():
    np.testing.assert_array_equal(simulate_asset_returns(100, seed=7), simulate_asset_returns(100, seed=7))

def test_covariance_symmetric():
    _, cov = theoretical_daily_moments()
    np.testing.assert_allclose(cov, cov.T)

def test_invalid_validation_input():
    with pytest.raises(ValueError):
        validate_simulation([1, 2, 3])

def test_quantile_rank_uncertainty_decreases_with_sample_size():
    assert quantile_rank_uncertainty(10000, .95) < quantile_rank_uncertainty(1000, .95)
