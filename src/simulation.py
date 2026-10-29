import numpy as np 
from config import(
    ANNUAL_RETURNS,
    ANNUAL_VOLATILITIES,
    CORRELATION_MATRIX,
    N_SIMULATIONS,
    RANDOM_SEED,
    TRADING_DAYS,
)
def simulate_asset_returns():
    rng = np.random.default_rng(RANDOM_SEED)
    annual_returns = np.asarray(ANNUAL_RETURNS,dtype=float)
    annual_volatilites = np.asarray(ANNUAL_VOLATILITIES,dtype=float)
    correlation = np.asarray(CORRELATION_MATRIX,dtype=float)
    