
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.benchmarks import (
    calculate_historical_var,
    calculate_parametric_var,
    run_convergence_experiment,
)
from src.portfolio import calculate_portfolio_returns, calculate_portfolio_pnl
from src.risk import calculate_var
from src.simulation import simulate_asset_returns

st.set_page_config(
    page_title="Quant Validation",
    page_icon="🧪",
    layout="wide",
)

st.title("Quant Model Validation")
st.caption(
    "Analytical consistency, simulation convergence, and optional "
    "historical risk comparison."
)

st.info(
    "Model comparisons use the assumptions in config.py and its default "
    "portfolio weights. Historical VaR requires real daily asset returns."
)

confidence = st.selectbox("VaR confidence level", [0.95, 0.99], index=0)
scenario_count = st.selectbox(
    "Scenarios for model comparison",
    [5000, 10000, 25000, 50000],
    index=3,
)

simulated = simulate_asset_returns(
    n_simulations=int(scenario_count),
    seed=int(config.RANDOM_SEED),
)
portfolio_returns = calculate_portfolio_returns(simulated)
pnl = calculate_portfolio_pnl(portfolio_returns)

mc_var = calculate_var(pnl, confidence)
parametric_var = calculate_parametric_var(
    confidence_level=confidence,
    portfolio_value=config.PORTFOLIO_VALUE,
    weights=config.WEIGHTS,
    annual_returns=config.ANNUAL_RETURNS,
    annual_volatilities=config.ANNUAL_VOLATILITIES,
    correlation_matrix=config.CORRELATION_MATRIX,
    trading_days=config.TRADING_DAYS,
)

left, right = st.columns(2)
left.metric("Monte Carlo VaR", f"₹{mc_var:,.2f}")
right.metric("Parametric Normal VaR", f"₹{parametric_var:,.2f}")

difference = abs(mc_var - parametric_var)
relative_difference = (
    100 * difference / parametric_var if parametric_var > 0 else np.nan
)

st.metric(
    "Absolute difference",
    f"₹{difference:,.2f}",
    help="Difference between the simulated estimate and analytical benchmark.",
)

if np.isfinite(relative_difference):
    st.caption(f"Difference relative to parametric VaR: {relative_difference:.2f}%")

comparison = pd.DataFrame({
    "Model": ["Monte Carlo", "Parametric Normal"],
    "VaR": [mc_var, parametric_var],
})

fig = px.bar(
    comparison,
    x="Model",
    y="VaR",
    color="Model",
    title=f"{confidence:.0%} VaR comparison",
    text_auto=",.0f",
)
fig.update_layout(
    template="plotly_dark",
    showlegend=False,
    yaxis_title="Estimated loss (₹)",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Simulation convergence")
st.write(
    "Each run uses the same random seed. Smaller runs use prefixes of the "
    "larger scenario sequence, allowing us to inspect how the estimate changes."
)

if st.button("Run convergence experiment"):
    with st.spinner("Running scenario-count benchmarks..."):
        convergence = run_convergence_experiment(
            simulation_counts=(1000, 5000, 10000, 50000),
            confidence_level=confidence,
            seed=config.RANDOM_SEED,
        )

    convergence_df = pd.DataFrame(convergence)
    st.dataframe(convergence_df, hide_index=True, use_container_width=True)

    fig = px.line(
        convergence_df,
        x="simulations",
        y="var",
        markers=True,
        title=f"{confidence:.0%} VaR versus simulation count",
        labels={
            "simulations": "Number of scenarios",
            "var": "Estimated VaR (₹)",
        },
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download convergence results",
        convergence_df.to_csv(index=False),
        file_name="convergence_results.csv",
        mime="text/csv",
    )

st.divider()
st.subheader("Optional: historical VaR")
st.write(
    "Upload a CSV containing daily simple returns for each configured asset. "
    "Returns must be decimals, such as 0.01 for +1% and -0.02 for -2%. "
    "Use actual historical observations, not Monte Carlo-generated returns."
)

uploaded = st.file_uploader(
    "Upload historical asset returns CSV",
    type=["csv"],
)

if uploaded is not None:
    try:
        historical = pd.read_csv(uploaded)
        missing = [
            name for name in config.ASSET_NAMES
            if name not in historical.columns
        ]

        if missing:
            st.error(
                "Missing required asset columns: " + ", ".join(missing)
            )
        else:
            asset_returns = historical[config.ASSET_NAMES].apply(
                pd.to_numeric,
                errors="raise",
            )

            if asset_returns.empty:
                st.error("The CSV contains no observations.")
            elif not np.isfinite(asset_returns.to_numpy()).all():
                st.error("Returns contain missing or non-finite values.")
            else:
                weights = np.asarray(config.WEIGHTS, dtype=float)
                weights = weights / weights.sum()

                historical_portfolio_returns = (
                    asset_returns.to_numpy() @ weights
                )

                historical_var = calculate_historical_var(
                    historical_portfolio_returns,
                    confidence_level=confidence,
                    portfolio_value=config.PORTFOLIO_VALUE,
                )

                st.metric("Historical VaR", f"₹{historical_var:,.2f}")
                st.caption(
                    f"Based on {len(asset_returns):,} uploaded observations. "
                    "The result depends on the selected historical period."
                )

                comparison_rows = [
                    {"Model": "Monte Carlo", "VaR": mc_var},
                    {"Model": "Parametric Normal", "VaR": parametric_var},
                    {"Model": "Historical", "VaR": historical_var},
                ]

                st.dataframe(
                    pd.DataFrame(comparison_rows),
                    hide_index=True,
                    use_container_width=True,
                )

                st.download_button(
                    "Download historical observations",
                    historical.to_csv(index=False),
                    file_name="historical_returns_used.csv",
                    mime="text/csv",
                )

    except Exception as exc:
        st.error(f"Could not process this CSV: {exc}")

with st.expander("Assumptions and limitations"):
    st.markdown("""
    - Monte Carlo scenarios use the configured multivariate normal model.
    - Parametric VaR uses the analytical variance implied by configured
      asset volatilities, correlations, weights, and annual mean returns.
    - Historical VaR uses the empirical quantile of portfolio losses from
      uploaded daily returns.
    - Historical and model-based VaR can differ because they rely on
      different distributions and periods.
    - This page does not perform out-of-sample backtesting.
    - These are research estimates, not investment recommendations.
    """)
