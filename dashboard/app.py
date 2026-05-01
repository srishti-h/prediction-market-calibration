"""
Streamlit dashboard for prediction market calibration analysis.

Run with:
    streamlit run dashboard/app.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.ingest import get_conn, load_analysis_df, db_summary
from src.pipeline.features import build_features
from src.analysis.calibration import (
    expected_calibration_error, brier_score, reliability_data,
    calibration_by_group, temporal_calibration, log_loss,
)
from src.analysis.bias import (
    favorite_longshot_stats, logistic_calibration_curve,
    overconfidence_by_bin, volume_bias_correlation,
)
from src.analysis.efficiency import (
    category_efficiency, early_vs_late_predictiveness, price_drift_by_resolution,
)

DB_PATH = os.getenv("DB_PATH", "data/markets.duckdb")

st.set_page_config(
    page_title="Prediction Market Calibration",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_conn():
    return get_conn(DB_PATH)


@st.cache_data(ttl=300)
def load_data(hours: int, category_filter: list, source_filter: list):
    conn = load_conn()
    df = load_analysis_df(conn, hours_to_close=hours)
    if category_filter:
        df = df[df["category"].isin(category_filter)]
    if source_filter:
        df = df[df["source"].isin(source_filter)]
    return build_features(df)


@st.cache_data(ttl=300)
def get_summary():
    conn = load_conn()
    return db_summary(conn)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Filters")

    summary = get_summary()
    categories = [r["category"] for r in summary["categories"]]
    sources = [r["source"] for r in summary["sources"]]

    selected_hours = st.selectbox(
        "Price snapshot (hours before close)",
        [168, 72, 24, 1],
        index=2,
        help="Which snapshot to use for analysis. T-24h = price 24 hours before market resolved.",
    )
    selected_categories = st.multiselect("Categories", categories, default=categories)
    selected_sources = st.multiselect("Sources", sources, default=sources)
    n_bins = st.slider("Calibration bins", 5, 20, 10)

    st.markdown("---")
    st.markdown(f"**Total markets in DB:** {summary['markets']:,}")
    st.markdown(f"**Total price records:** {summary['prices']:,}")


# ── Load Data ────────────────────────────────────────────────────────────────
df = load_data(selected_hours, selected_categories, selected_sources)

if df.empty:
    st.error("No data found. Run `python scripts/collect_data.py` first.")
    st.stop()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Prediction Market Calibration Study")
st.markdown(
    f"Analyzing **{len(df):,} resolved markets** from {', '.join(df['source'].unique())} "
    f"using T-{selected_hours}h price snapshots."
)

# ── Top Metrics ───────────────────────────────────────────────────────────────
ece = expected_calibration_error(df, n_bins)
bs = brier_score(df)
ll = log_loss(df)
yes_rate = df["outcome"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected Calibration Error", f"{ece:.4f}", help="Weighted avg |predicted - actual|. Lower = better.")
col2.metric("Brier Score", f"{bs:.4f}", help="Mean squared error. Random guess = 0.25.")
col3.metric("Log Loss", f"{ll:.4f}")
col4.metric("YES Resolution Rate", f"{yes_rate:.1%}", help="Fraction of markets that resolved YES.")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Calibration", "Bias Analysis", "Efficiency", "Raw Data"])

# ════════════════════════════════════════════════════════════
# TAB 1: CALIBRATION
# ════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Reliability Diagram")
        rel = reliability_data(df, n_bins)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color="#adb5bd", dash="dash", width=1.5),
            name="Perfect calibration",
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([rel["mean_pred"], rel["mean_pred"][::-1]]),
            y=pd.concat([rel["ci_high"], rel["ci_low"][::-1]]),
            fill="toself",
            fillcolor="rgba(0,102,204,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        ))
        fig.add_trace(go.Scatter(
            x=rel["mean_pred"],
            y=rel["actual_rate"],
            mode="markers+lines",
            marker=dict(size=rel["count"] / rel["count"].max() * 20 + 6, color="#0066cc"),
            line=dict(color="#0066cc", width=2),
            name="Market prices",
            text=[f"n={n:,}<br>pred={p:.2f}<br>actual={a:.2f}"
                  for n, p, a in zip(rel["count"], rel["mean_pred"], rel["actual_rate"])],
            hovertemplate="%{text}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Predicted probability",
            yaxis_title="Actual resolution rate",
            xaxis=dict(range=[-0.02, 1.02]),
            yaxis=dict(range=[-0.02, 1.02]),
            legend=dict(x=0.02, y=0.98),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Calibration by Category")
        cat_cal = calibration_by_group(df, "category", n_bins)
        if not cat_cal.empty:
            fig2 = px.bar(
                cat_cal.sort_values("ece"),
                x="ece", y="category",
                orientation="h",
                color="ece",
                color_continuous_scale="Blues_r",
                text=cat_cal.sort_values("ece")["n"].apply(lambda x: f"n={x:,}"),
                labels={"ece": "ECE", "category": "Category"},
            )
            fig2.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Calibration Over Time (hours before close)")
    conn = load_conn()
    temporal = temporal_calibration(conn, n_bins=n_bins)
    if not temporal.empty:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=temporal["hours_to_close"],
            y=temporal["ece"],
            mode="markers+lines",
            marker=dict(size=10, color="#0066cc"),
            line=dict(width=2.5, color="#0066cc"),
            text=[f"T-{h}h: ECE={e:.3f} (n={n:,})"
                  for h, e, n in zip(temporal["hours_to_close"], temporal["ece"], temporal["n"])],
            hovertemplate="%{text}<extra></extra>",
        ))
        fig3.update_layout(
            xaxis=dict(title="Hours before close", autorange="reversed"),
            yaxis_title="Expected Calibration Error",
            height=350,
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Markets become better calibrated as new information arrives closer to resolution.")


# ════════════════════════════════════════════════════════════
# TAB 2: BIAS
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Favorite-Longshot Bias")
    fls = favorite_longshot_stats(df)

    col1, col2 = st.columns([2, 1])
    with col1:
        colors = {"longshot": "#e63946", "contested": "#457b9d", "favorite": "#2a9d8f"}
        fig_fls = go.Figure()
        for _, row in fls.iterrows():
            fig_fls.add_trace(go.Bar(
                x=[row["tier"]],
                y=[row["bias"]],
                name=row["tier"],
                marker_color=colors.get(row["tier"], "#457b9d"),
                text=f"{row['bias']:+.3f}{'*' if row['significant'] else ''}<br>n={row['n']:,}",
                textposition="outside",
            ))
        fig_fls.add_hline(y=0, line_color="black", line_width=1)
        fig_fls.update_layout(
            yaxis_title="Bias (actual − predicted)",
            showlegend=False,
            height=380,
            title="Positive = market underestimates probability (* p < 0.05)",
        )
        st.plotly_chart(fig_fls, use_container_width=True)

    with col2:
        st.markdown("#### Key Statistics")
        for _, row in fls.iterrows():
            st.markdown(f"**{row['tier'].capitalize()}**")
            st.markdown(f"- Predicted: {row['mean_predicted']:.1%}")
            st.markdown(f"- Actual: {row['actual_rate']:.1%}")
            st.markdown(f"- Bias: {row['bias']:+.3f} {'✓ sig.' if row['significant'] else ''}")
            st.markdown("")

    st.subheader("Overconfidence by Bin")
    oconf = overconfidence_by_bin(df)
    if not oconf.empty:
        fig_oc = go.Figure()
        fig_oc.add_bar(
            x=oconf["bin"],
            y=oconf["bias"],
            marker_color=["#2a9d8f" if b > 0 else "#e63946" for b in oconf["bias"]],
            text=[f"{b:+.3f}" for b in oconf["bias"]],
            textposition="outside",
        )
        fig_oc.add_hline(y=0, line_color="black", line_width=1)
        fig_oc.update_layout(
            xaxis_title="Implied probability bin",
            yaxis_title="Bias (actual − predicted)",
            height=350,
        )
        st.plotly_chart(fig_oc, use_container_width=True)

    st.subheader("Logistic Calibration Fit")
    try:
        logistic = logistic_calibration_curve(df)
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Slope", f"{logistic['slope']:.3f}", help="Perfect = 1.0. <1 = overconfident.")
        lc2.metric("Intercept", f"{logistic['intercept']:.3f}", help="Perfect = 0.0.")
        lc3.metric("Interpretation", logistic["interpretation"])
    except Exception as e:
        st.warning(f"Logistic fit failed (need more data): {e}")

    vol_bias = volume_bias_correlation(df)
    st.subheader("Liquidity vs. Calibration Error")
    st.metric("Pearson r (log volume vs. |bias|)", f"{vol_bias['pearson_r']:.4f}", help=vol_bias["interpretation"])
    st.caption(vol_bias["interpretation"])


# ════════════════════════════════════════════════════════════
# TAB 3: EFFICIENCY
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Predictive Power by Horizon (AUC)")
    auc = early_vs_late_predictiveness(conn)
    if auc:
        auc_df = pd.DataFrame(list(auc.items()), columns=["hours_to_close", "auc"]).sort_values("hours_to_close", ascending=False)
        fig_auc = go.Figure()
        fig_auc.add_trace(go.Scatter(
            x=auc_df["hours_to_close"],
            y=auc_df["auc"],
            mode="markers+lines",
            marker=dict(size=10, color="#0066cc"),
            line=dict(width=2.5, color="#0066cc"),
            text=[f"T-{h}h AUC={a:.3f}" for h, a in zip(auc_df["hours_to_close"], auc_df["auc"])],
            hovertemplate="%{text}<extra></extra>",
        ))
        fig_auc.add_hline(y=0.5, line_dash="dash", line_color="#adb5bd", annotation_text="Random")
        fig_auc.update_layout(
            xaxis=dict(title="Hours before close", autorange="reversed"),
            yaxis_title="AUC (binary classifier for resolution)",
            height=350,
        )
        st.plotly_chart(fig_auc, use_container_width=True)
        st.caption("Higher AUC at T-1h vs T-168h shows markets incorporate new information over time.")

    st.subheader("Price Drift by Resolution Outcome")
    drift = price_drift_by_resolution(conn)
    if not drift.empty:
        fig_drift = go.Figure()
        for resolution, sub in drift.groupby("resolution"):
            sub = sub.sort_values("hours_to_close", ascending=False)
            color = "#2a9d8f" if resolution == "YES" else "#e63946"
            fig_drift.add_trace(go.Scatter(
                x=sub["hours_to_close"],
                y=sub["mean_price"],
                mode="lines+markers",
                name=f"Resolved {resolution}",
                line=dict(color=color, width=2),
                marker=dict(size=7),
            ))
        fig_drift.update_layout(
            xaxis=dict(title="Hours before close", autorange="reversed"),
            yaxis_title="Mean YES price",
            height=380,
        )
        st.plotly_chart(fig_drift, use_container_width=True)

    st.subheader("Category Efficiency")
    cat_eff = category_efficiency(conn)
    if not cat_eff.empty:
        st.dataframe(cat_eff.style.background_gradient(subset=["ece"], cmap="RdYlGn_r"), use_container_width=True)
        st.caption("Sports markets tend to be most efficient (lowest ECE) due to informed bettors.")


# ════════════════════════════════════════════════════════════
# TAB 4: RAW DATA
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Filtered Dataset")
    display_cols = ["question", "category", "source", "predicted_prob", "outcome", "resolution", "volume", "confidence_tier"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[display_cols].sort_values("volume", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=500,
    )
    st.caption(f"{len(df):,} markets shown")

    if st.button("Export to CSV"):
        csv = df[display_cols].to_csv(index=False)
        st.download_button("Download CSV", csv, "markets.csv", "text/csv")
