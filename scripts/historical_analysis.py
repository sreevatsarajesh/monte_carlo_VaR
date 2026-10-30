import argparse
import numpy as np
import pandas as pd
from config import ASSET_NAMES
from src.benchmarks import calculate_historical_var

p = argparse.ArgumentParser()
p.add_argument("csv_path")
p.add_argument("--date-column", default=None)
p.add_argument("--confidence", type=float, default=.95)
a = p.parse_args()
df = pd.read_csv(a.csv_path)
missing = [c for c in ASSET_NAMES if c not in df.columns]
if missing:
    raise ValueError(f"CSV missing configured asset columns: {missing}")
if a.date_column and a.date_column in df:
    df[a.date_column] = pd.to_datetime(df[a.date_column], errors="raise")
    df = df.sort_values(a.date_column)
values = df[ASSET_NAMES].apply(pd.to_numeric, errors="raise").to_numpy(float)
if len(values) < 2 or not np.isfinite(values).all():
    raise ValueError("Need at least two rows of finite decimal daily returns.")
print(f"Observations: {len(values)}")
print(f"Historical VaR ({a.confidence:.0%}): ₹{calculate_historical_var(values, a.confidence):,.2f}")
print("Input values must be actual decimal daily simple returns, not simulated data.")
