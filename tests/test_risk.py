import numpy as np
import pytest
from src.risk import calculate_var, calculate_expected_shortfall
def test_var_calculation():
    pnl = np.array([100, 50, 0, -50, -100, -150, -200, -250, -300, -400])
    assert calculate_var(pnl, 0.90) == pytest.approx(310.0)
def test_expected_shortfall_calculation():
    pnl = np.array([100, 50, 0, -50, -100, -150, -200, -250, -300, -400])
    assert calculate_expected_shortfall(pnl, 0.90) == pytest.approx(400.0)
def test_var_increases_with_confidence():
    pnl = np.array([100, 50, 0, -50, -100, -150, -200, -250, -300, -400])
    assert calculate_var(pnl, 0.99) >= calculate_var(pnl, 0.95)
def test_empty_pnl_raises_error():
    with pytest.raises(ValueError):
        calculate_var(np.array([]))
