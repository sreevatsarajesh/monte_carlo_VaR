
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.simulation import simulate_asset_returns
from src.portfolio import calculate_portfolio_returns, calculate_portfolio_pnl
from src.risk import calculate_var, calculate_expected_shortfall

st.set_page_config(
    page_title="Monte Carlo | Risk Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp {
    background: #080c14;
    color: #e8edf5;
}
[data-testid="stSidebar"] {
    background: #0d1320;
    border-right: 1px solid #202a3a;
}
[data-testid="stMetric"] {
    background: #101827;
    border: 1px solid #263247;
    padding: 18px;
    border-radius: 12px;
}
[data-testid="stMetricLabel"] {
    color: #91a0b8;
}
[data-testid="stMetricValue"] {
    color: #f1f5fb;
    font-family: 'DM Mono', monospace;
}
h1, h2, h3 {
    letter-spacing: -0.5px;
}
.eyebrow {
    color: #45e0b1;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
}
.subtitle {
    color: #8e9bb0;
    margin-top: -12px;
}
.panel {
    background: #101827;
    border: 1px solid #263247;
    border-radius: 12px;
    padding: 18px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="eyebrow">QUANT SYSTEMS / 01</p>', unsafe_allow_html=True)
st.sidebar.title("Risk Terminal")
st.sidebar.caption("MONTE CARLO SIMULATION")
st.sidebar.divider()

st.sidebar.subheader("Portfolio configuration")

portfolio_value = st.sidebar.number_input(
    "Portfolio value (₹)",
    min_value=1000,
    max_value=1_000_000_000,
    value=int(config.PORTFOLIO_VALUE),
    step=10000,
)

n_simulations = st.sidebar.select_slider(
    "Simulation scenarios",
    options=[1000, 5000, 10000, 25000, 50000],
    value=10000 if 10000 in [1000, 5000, 10000, 25000, 50000] else 5000,
)

seed = st.sidebar.number_input("Random seed", min_value=0, value=int(config.RANDOM_SEED), step=1)

st.sidebar.subheader("Asset allocation")

asset_names = list(config.ASSET_NAMES)
default_weights = list(config.WEIGHTS)
weights = []

for i, asset in enumerate(asset_names):
    weights.append(
        st.sidebar.slider(
            asset,
            min_value=0.0,
            max_value=1.0,
            value=float(default_weights[i]),
            step=0.05,
            key=f"weight_{i}",
        )
    )

weight_total = sum(weights)
if weight_total <= 0:
    st.sidebar.error("Allocate a positive weight to at least one asset.")
    st.stop()

if abs(weight_total - 1.0) > 1e-6:
    st.sidebar.warning(f"Weights total {weight_total:.0%}. They will be normalized automatically.")
    weights = [w / weight_total for w in weights]

st.sidebar.caption("Illustrative assumptions · Not investment advice")

try:
    with st.spinner("Running Monte Carlo scenarios..."):
        simulated_returns = simulate_asset_returns(
            n_simulations=int(n_simulations),
            seed=int(seed),
        )

        portfolio_returns = calculate_portfolio_returns(
            simulated_returns,
            weights=weights,
        )

        pnl = calculate_portfolio_pnl(
            portfolio_returns,
            portfolio_value=float(portfolio_value),
        )

        var_95 = calculate_var(pnl, 0.95)
        var_99 = calculate_var(pnl, 0.99)
        es_95 = calculate_expected_shortfall(pnl, 0.95)
        es_99 = calculate_expected_shortfall(pnl, 0.99)

except Exception as exc:
    st.error(f"Simulation failed: {exc}")
    st.stop()

losses = -pnl
daily_volatility = float(np.std(portfolio_returns, ddof=1))
mean_return = float(np.mean(portfolio_returns))
probability_loss = float(np.mean(pnl < 0))
worst_loss = max(0.0, float(np.max(losses)))
best_gain = max(0.0, float(np.max(pnl)))

st.markdown('<p class="eyebrow">MONTE CARLO ENGINE / LIVE ANALYTICS</p>', unsafe_allow_html=True)
st.title("Portfolio Risk Dashboard")
st.markdown(
    '<p class="subtitle">Scenario-based market risk analytics · One-day horizon · '
    'Multivariate normal returns</p>',
    unsafe_allow_html=True,
)

st.divider()

k1, k2, k3, k4 = st.columns(4)
k1.metric("95% Value at Risk", f"₹{var_95:,.0f}", help="Estimated loss threshold at 95% confidence.")
k2.metric("99% Value at Risk", f"₹{var_99:,.0f}", help="Estimated loss threshold at 99% confidence.")
k3.metric("95% Expected Shortfall", f"₹{es_95:,.0f}", help="Average loss in the simulated tail beyond the 95% VaR threshold.")
k4.metric("Daily volatility", f"{daily_volatility:.2%}")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Portfolio value", f"₹{portfolio_value:,.0f}")
k6.metric("Mean daily return", f"{mean_return:.3%}")
k7.metric("Scenarios simulated", f"{int(n_simulations):,}")
k8.metric("Probability of loss", f"{probability_loss:.1%}")

st.divider()

left, right = st.columns([1.65, 1])

with left:
    st.subheader("Portfolio P&L distribution")
    st.caption("Each bar represents simulated one-day profit and loss.")

    fig = px.histogram(
        x=pnl,
        nbins=90,
        labels={"x": "Profit / loss (₹)", "count": "Scenario count"},
        color_discrete_sequence=["#45e0b1"],
        opacity=0.88,
    )

    fig.add_vline(
        x=-var_95,
        line_dash="dash",
        line_color="#f6bd60",
        annotation_text="95% VaR",
        annotation_position="top left",
    )
    fig.add_vline(
        x=-var_99,
        line_dash="dash",
        line_color="#ff6b6b",
        annotation_text="99% VaR",
        annotation_position="top right",
    )
    fig.add_vline(x=0, line_color="#91a0b8", line_width=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080c14",
        plot_bgcolor="#101827",
        height=420,
        margin=dict(l=10, r=10, t=35, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Asset allocation")
    st.caption("Normalized portfolio weights used in the simulation.")

    allocation = pd.DataFrame({
        "Asset": asset_names,
        "Weight": weights,
    })

    fig_alloc = px.pie(
        allocation,
        names="Asset",
        values="Weight",
        hole=0.64,
        color_discrete_sequence=["#45e0b1", "#6c8cff", "#f6bd60", "#c792ea"],
    )
    fig_alloc.update_traces(
        textposition="outside",
        textinfo="label+percent",
    )
    fig_alloc.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080c14",
        plot_bgcolor="#101827",
        height=420,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12),
    )
    st.plotly_chart(fig_alloc, use_container_width=True)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Risk metrics")
    risk_table = pd.DataFrame({
        "Metric": [
            "95% Value at Risk",
            "99% Value at Risk",
            "95% Expected Shortfall",
            "99% Expected Shortfall",
            "Worst simulated loss",
            "Best simulated gain",
        ],
        "Value": [
            var_95,
            var_99,
            es_95,
            es_99,
            worst_loss,
            best_gain,
        ],
    })
    risk_table["Value"] = risk_table["Value"].map(lambda x: f"₹{x:,.2f}")
    st.dataframe(risk_table, hide_index=True, use_container_width=True)

with right:
    st.subheader("Asset-level assumptions")
    asset_table = pd.DataFrame({
        "Asset": asset_names,
        "Weight": weights,
        "Annual return assumption": config.ANNUAL_RETURNS,
        "Annual volatility": config.ANNUAL_VOLATILITIES,
    })
    asset_table["Weight"] = asset_table["Weight"].map(lambda x: f"{x:.1%}")
    asset_table["Annual return assumption"] = asset_table["Annual return assumption"].map(lambda x: f"{x:.1%}")
    asset_table["Annual volatility"] = asset_table["Annual volatility"].map(lambda x: f"{x:.1%}")
    st.dataframe(asset_table, hide_index=True, use_container_width=True)

st.divider()

with st.expander("Methodology and model limitations"):
    st.markdown("""
    **Simulation model**
    - Draws correlated one-day asset-return scenarios from a multivariate normal distribution.
    - Converts annual expected returns to daily means by dividing by 252.
    - Converts annual volatility to daily volatility by dividing by √252.
    - Uses the correlation matrix specified in `config.py`.
    - Uses a fixed random seed for reproducibility.

    **Risk definitions**
    - **VaR:** the selected quantile of simulated losses, floored at zero.
    - **Expected Shortfall:** average simulated loss among scenarios at or beyond the VaR threshold, floored at zero.

    **Limitations**
    - Assumes normally distributed returns and static correlations.
    - Uses illustrative inputs, not live market data.
    - One-day estimates can understate risks from jumps, fat tails, and changing market regimes.
    - Historical backtesting is not included yet.
    """)

st.caption(
    "MONTE CARLO RISK TERMINAL · Research prototype · "
    "Results depend on assumptions and simulation seed."
)
