import numpy as np
import pytest
from src.benchmarks import (
    calculate_historical_var,
    calculate_parametric_var,
    run_convergence_experiment,
)
def test_parametric_var_matches_known_standard_normal_case():
    result = calculate_parametric_var(
        confidence_level=0.95,
        portfolio_value=1_000_000,
        weights=[1.0],
        annual_returns=[0.0],
        annual_volatilities=[0.20],
        correlation_matrix=[[1.0]],
        trading_days=252,
    )
    expected = 1_000_000 * 1.6448536269514722 * (0.20 / np.sqrt(252))
    assert result == pytest.approx(expected)
def test_parametric_var_increases_with_confidence():
    var_95 = calculate_parametric_var(confidence_level=0.95)
    var_99 = calculate_parametric_var(confidence_level=0.99)
    assert var_99 >= var_95
def test_parametric_var_rejects_invalid_weights():
    with pytest.raises(ValueError):
        calculate_parametric_var(weights=[0.4, 0.4, 0.4])
def test_historical_var_known_example():
    returns = np.array([-0.10, -0.05, 0.00, 0.02, 0.04])
    result = calculate_historical_var(
        returns,
        confidence_level=0.80,
        portfolio_value=1000,
    )
    assert result == pytest.approx(40.0)
def test_historical_var_rejects_empty_input():
    with pytest.raises(ValueError):
        calculate_historical_var([])
def test_historical_var_rejects_nonfinite_input():
    with pytest.raises(ValueError):
        calculate_historical_var([0.01, np.nan, -0.02])
def test_convergence_experiment_is_reproducible():
    first = run_convergence_experiment([1000, 2000], seed=123)
    second = run_convergence_experiment([1000, 2000], seed=123)
    assert [x["var"] for x in first] == pytest.approx(
        [x["var"] for x in second]
    )
    assert all(x["runtime_seconds"] >= 0 for x in first)
def test_convergence_experiment_rejects_invalid_count():
    with pytest.raises(ValueError):
        run_convergence_experiment([0])