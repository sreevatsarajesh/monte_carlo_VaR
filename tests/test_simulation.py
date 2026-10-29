import numpy as np
import pytest

from src.simulation import simulate_asset_returns


def test_simulation_shape_and_finite_values():
    returns = simulate_asset_returns(n_simulations=500, seed=7)
    assert returns.shape == (500, 3)
    assert np.isfinite(returns).all()


def test_simulation_is_reproducible_for_same_seed():
    first = simulate_asset_returns(n_simulations=100, seed=123)
    second = simulate_asset_returns(n_simulations=100, seed=123)
    np.testing.assert_array_equal(first, second)


def test_simulation_rejects_nonpositive_count():
    with pytest.raises(ValueError):
        simulate_asset_returns(n_simulations=0)
