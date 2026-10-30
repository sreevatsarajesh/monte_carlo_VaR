# QUANT//RISK validation and benchmarking

## Model definition

The current engine samples one-day simple asset returns from a multivariate normal distribution:

\[
r \sim \mathcal{N}(\mu_d,\Sigma_d)
\]

with \(\mu_{d,i}=\mu_{a,i}/252\), daily volatility \(\sigma_{d,i}=\sigma_{a,i}/\sqrt{252}\), and \(\Sigma_d=DRD\), where \(D\) is the diagonal daily-volatility matrix and \(R\) is the configured correlation matrix.

**This is not GBM.** GBM simulates lognormal prices using \(S_{t+\Delta t}=S_t\exp[(\mu-\sigma^2/2)\Delta t+\sigma\sqrt{\Delta t}Z]\). Do not describe this engine as GBM unless it is changed and validated separately.

## Parametric VaR

For weights \(w\), portfolio mean is \(w^\top\mu_d\) and variance is \(w^\top\Sigma_dw\). Under the normal-return assumption:

\[
VaR_c=\max(0,V[z_c\sqrt{w^\top\Sigma_dw}-w^\top\mu_d])
\]

Parametric VaR is an analytical benchmark under the same assumptions as Monte Carlo. Similarity checks implementation consistency, not real-world predictive accuracy.

## Validation

Run `python -m pytest -q` and `python run_quant_validation.py`. Tests compare sample means, variances, and correlations with configured theoretical moments using fixed seeds and finite-sample tolerances.

The reported mean z-score uses standard error \(\sigma/\sqrt{N}\). The quantile-rank standard error is approximately \(\sqrt{p(1-p)/N}\). This is uncertainty in percentile rank, not a currency confidence interval for VaR.

## Historical VaR

Provide a CSV with one column per configured asset name. Each cell must be a decimal daily simple return (e.g. `0.01` means +1%). Optional date sorting:

`python scripts/historical_analysis.py returns.csv --date-column Date --confidence 0.95`

Historical VaR is an empirical quantile, not out-of-sample backtesting. Backtesting requires forecasts made before each realized return and chronological breach analysis.

## Fair showdown protocol

Run in both repositories on the same machine:

`python run_benchmarks.py --count 50000 --seed 42 --output results/benchmark.json`

Then compare:

`python scripts/compare_benchmarks.py results/benchmark.json results/roommate_benchmark.json`

Match portfolio value, weights, return/volatility/correlation assumptions, horizon, confidence levels, simulation count, seed, hardware, Python and dependency versions. Different RNGs need not produce identical paths; compare statistical behavior instead. Runtime comparisons require the same machine and environment.

## Limitations

Static parameters and correlations; normal simple returns; no jumps, fat tails, or volatility clustering; one-day horizon; no bundled live market data; historical VaR only when real data is supplied; no out-of-sample backtest claim. Outputs are research estimates, not investment advice.
