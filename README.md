# QUANT//RISK — validation addendum

Run:

```bash
python -m pytest -q
python run_quant_validation.py
python run_benchmarks.py --count 50000 --seed 42
```

Validation output: `results/quant_validation_report.json`. Benchmark output: `results/benchmark.json`.

For historical VaR, prepare a CSV with columns matching `ASSET_NAMES`, containing actual decimal daily simple returns:

```bash
python scripts/historical_analysis.py returns.csv --confidence 0.95
```

See `docs/validation_and_benchmarking.md` for equations, assumptions, limitations, and fair comparison protocol. The current engine models multivariate normal one-day simple returns, not GBM.
