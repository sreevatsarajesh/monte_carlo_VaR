import numpy as np
import pytest

from src.portfolio import calculate_portfolio_pnl, calculate_portfolio_returns


def test_weighted_portfolio_returns():
    asset_returns = np.array([[0.10, 0.00, -0.05]])
    result = calculate_portfolio_returns(asset_returns, weights=[0.5, 0.3, 0.2])
    assert result[0] == pytest.approx(0.04)


def test_portfolio_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        calculate_portfolio_returns(np.zeros((2, 3)), weights=[0.5, 0.3, 0.3])


def test_pnl_uses_portfolio_value():
    pnl = calculate_portfolio_pnl(np.array([0.01, -0.02]), portfolio_value=100_000)
    np.testing.assert_allclose(pnl, [1000, -2000])
