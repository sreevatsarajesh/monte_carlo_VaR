# Monte Carlo Portfolio Risk Simulator

A Python-based quantitative finance mini-project that simulates
10,000 portfolio return scenarios to estimate Value at Risk (VaR)
and Expected Shortfall.

## Results

- 95% VaR: ₹16,102.41
- 99% VaR: ₹23,261.08
- 95% Expected Shortfall: ₹20,420.39
- Observed 95% VaR reduction from diversification: 22.19%

## Visualizations

![Portfolio Risk Distribution](results/portfolio_risk.png)

## Run locally

```bash
python -m pip install -r requirements.txt
python main.py