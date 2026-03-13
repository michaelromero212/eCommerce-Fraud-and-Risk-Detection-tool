"""
dashboards/fraud_dashboard.py
------------------------------
EFRiskEngine — Interactive Fraud & Risk Dashboard (Streamlit)

Launch with:
    streamlit run dashboards/fraud_dashboard.py

Tabs:
  1. Overview        — KPI cards and dataset summary
  2. Time Series     — Fraud events and risk scores over time
  3. Geo & Device    — Risk breakdown by country and device type
  4. Flagged Records — Sortable tables of flagged transactions and high-risk users
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root
# ---------------------------------------------------------------------------
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DASHBOARD_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data_pipeline import run_etl  # noqa: E402
from src.risk_engine import run_risk_engine  # noqa: E402
from src.reporting import summary_stats  # noqa: E402
from src.ai_assist import get_provider, analyze_user_behavior  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EFRiskEngine — Fraud Dashboard",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark premium theme (complements config.toml)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  /* Metric cards */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
  }
  [data-testid="metric-container"] label { color: #475569 !important; font-size: 1.05rem !important; font-weight: 600;}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ea580c !important; font-size: 2.2rem !important; font-weight: 800;
  }

  /* Tabs text contrast */
  .stTabs [role="tab"] { color: #64748b; font-weight: 600; }
  .stTabs [role="tab"][aria-selected="true"] {
    color: #ea580c; border-bottom: 2px solid #ea580c;
  }

  /* DataFrame Header Contrast */
  [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0; }
  
  /* Text Readability for headers */
  h1, h2, h3 { color: #ea580c !important; font-weight: 800 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/shield.png",
        width=60,
    )
    st.title("EFRiskEngine")
    st.caption("eCommerce Fraud & Risk Detection")
    st.markdown("---")

    risk_threshold = st.slider(
        "Minimum Risk Score to Flag",
        min_value=0, max_value=100, value=25, step=5,
        help="Only transactions at or above this score appear in flagged views.",
    )

    show_all = st.checkbox("Show all transactions (unfiltered)", value=False)

    st.markdown("---")
    if st.button("🔄 Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Built with Streamlit & Plotly")


# ---------------------------------------------------------------------------
# Data loading with caching
# ---------------------------------------------------------------------------
from plotly.subplots import make_subplots

@st.cache_data(ttl=300, show_spinner="Running ETL pipeline…")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, normalise, and score data; cache for 5 minutes."""
    users, txns = run_etl()
    scored_txns, user_summaries = run_risk_engine(txns, users)
    return users, txns, scored_txns, user_summaries


try:
    users_df, txns_df, scored_txns, user_summaries = load_data()
except FileNotFoundError:
    st.error(
        "⚠️ Sample data not found. "
        "Run `python generate_sample_data.py` from the project root first."
    )
    st.stop()

stats = summary_stats(scored_txns, user_summaries)

# Apply threshold filter
filtered_txns = scored_txns if show_all else scored_txns[scored_txns["risk_score"] >= risk_threshold]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🛡 EFRiskEngine — Fraud & Risk Dashboard")
st.caption(f"Dataset: {stats['total_transactions']} transactions | {len(users_df)} users")
st.markdown("---")

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Time Series",
    "🌍 Geo & Device",
    "🚨 Flagged Records",
    "🤖 AI Insights",
])


# ─────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Key Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Transactions", f"{stats['total_transactions']:,}")
    col2.metric(
        "Flagged (Medium+)",
        f"{stats['flagged_transactions']:,}",
        delta=f"{stats['flag_rate_pct']}%",
        delta_color="inverse",
    )
    col3.metric("Critical Transactions", f"{stats['critical_transactions']:,}")
    col4.metric("High-Risk Users", f"{stats['high_risk_users']:,}")
    col5.metric("Avg Risk Score", f"{stats['avg_risk_score']:.1f} / 100")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Risk Score Distribution")
        fig_hist = px.histogram(
            scored_txns,
            x="risk_score",
            nbins=20,
            color_discrete_sequence=["#f97316"],
            labels={"risk_score": "Risk Score"},
            title="Transaction Risk Score Distribution",
        )
        fig_hist.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font_color="#334155",
            title_font_color="#ea580c",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("Shows the frequency of different risk scores. The distribution highlights the long-tail nature of fraudulent transactions.")

    with col_b:
        st.subheader("Risk Label Breakdown")
        label_counts = scored_txns["risk_label"].value_counts().reset_index()
        label_counts.columns = ["Risk Label", "Count"]
        colour_map = {
            "Critical": "#ef4444",
            "High": "#f97316",
            "Medium": "#eab308",
            "Low": "#22c55e",
        }
        fig_pie = px.pie(
            label_counts,
            names="Risk Label",
            values="Count",
            color="Risk Label",
            color_discrete_map=colour_map,
            hole=0.45,
            title="Transactions by Risk Label",
        )
        fig_pie.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font_color="#334155",
            title_font_color="#ea580c",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("Proportional breakdown of risk severities across all processed transactions.")


# ─────────────────────────────────────────────────────────────────
# TAB 2 — TIME SERIES
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Fraud Events Over Time")

    # Aggregate daily
    ts_df = scored_txns.copy()
    ts_df["date"] = pd.to_datetime(ts_df["timestamp"]).dt.date
    daily = (
        ts_df.groupby("date")
        .agg(
            total=("risk_score", "count"),
            flagged=("risk_score", lambda x: (x >= risk_threshold).sum()),
            avg_risk=("risk_score", "mean"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])

    # Create figure with secondary y-axis
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add transaction volume bars
    fig_ts.add_trace(
        go.Bar(
            x=daily["date"], y=daily["total"],
            name="Total Transactions",
            marker_color="#cbd5e1",
            opacity=0.6
        ),
        secondary_y=False,
    )
    
    # Add flagged transaction bars on top
    fig_ts.add_trace(
        go.Bar(
            x=daily["date"], y=daily["flagged"],
            name=f"Flagged (≥{risk_threshold})",
            marker_color="#f97316",
        ),
        secondary_y=False,
    )

    # Add Average Risk Score line on secondary y-axis
    fig_ts.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["avg_risk"],
            name="Avg Risk Score",
            line=dict(color="#8b5cf6", width=3, shape="spline"),
            mode="lines+markers"
        ),
        secondary_y=True,
    )

    fig_ts.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font_color="#334155",
        title="Transaction Volume vs. Average Risk Score",
        title_font_color="#ea580c",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(showgrid=False),
        hovermode="x unified",
        barmode='overlay'
    )
    
    # Set y-axes titles
    fig_ts.update_yaxes(title_text="Transaction Volume", secondary_y=False, gridcolor="#e2e8f0")
    fig_ts.update_yaxes(title_text="Average Risk Score", secondary_y=True, showgrid=False)

    st.plotly_chart(fig_ts, use_container_width=True)
    st.caption("Tracks daily transaction volume against the average risk score. Spikes in the risk line independent of volume often signify concentrated, systemic fraud attacks occurring on those specific days.")


# ─────────────────────────────────────────────────────────────────
# TAB 3 — GEO & DEVICE
# ─────────────────────────────────────────────────────────────────
with tab3:
    col_geo, col_dev = st.columns(2)

    with col_geo:
        st.subheader("Risk by Transaction Country")
        if "transaction_country" in filtered_txns.columns:
            country_risk = (
                filtered_txns.groupby("transaction_country")["risk_score"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "avg_risk", "count": "txn_count"})
            )
            fig_country = px.choropleth(
                country_risk,
                locations="transaction_country",
                locationmode="country names" if len(country_risk["transaction_country"].iloc[0]) > 2 else "ISO-3", # Basic check, EFRisk uses ISO-2 which Plotly handles well via locationmode='country names' (requires mapping if purely ISO-2, but we will let Plotly auto-resolve or explicitly map)
                color="avg_risk",
                hover_name="transaction_country",
                hover_data={"txn_count": True, "avg_risk": ":.1f", "transaction_country": False},
                color_continuous_scale="Oranges",
                title="Global Risk Heatmap",
                labels={"avg_risk": "Avg Risk Score"}
            )
            # Fix ISO-2 to Plotly's default expected formats natively
            if len(country_risk["transaction_country"].iloc[0]) == 2:
                # EFRisk generates 2-letter ISO codes (US, GB, etc.)
                import pycountry
                def get_iso3(iso2):
                    try:
                        return pycountry.countries.get(alpha_2=iso2).alpha_3
                    except:
                        return iso2
                country_risk['iso_3'] = country_risk['transaction_country'].apply(get_iso3)
                
                fig_country = px.choropleth(
                    country_risk,
                    locations="iso_3",
                    color="avg_risk",
                    hover_name="transaction_country",
                    hover_data={"txn_count": True, "avg_risk": ":.1f", "iso_3": False},
                    color_continuous_scale="Oranges",
                    title="Global Risk Heatmap",
                    labels={"avg_risk": "Avg Risk Score"}
                )

            fig_country.update_layout(
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font_color="#334155",
                title_font_color="#ea580c",
                geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular')
            )
            st.plotly_chart(fig_country, use_container_width=True)
            st.caption("Global heatmap of average risk scores. Darker regions indicate higher concentrations of flagged transactions or geolocation anomalies.")
        else:
            st.info("Country data not available.")

    with col_dev:
        st.subheader("Risk by Device Type")
        if "device_type" in filtered_txns.columns:
            fig_dev = px.histogram(
                filtered_txns,
                x="device_type",
                color="risk_label",
                barmode="stack",
                color_discrete_map={
                    "Critical": "#ef4444",
                    "High": "#f97316",
                    "Medium": "#eab308",
                    "Low": "#22c55e",
                },
                labels={"device_type": "Device Type", "risk_label": "Risk Tier"},
                title="Risk Severity Breakdown by Device",
            )
            fig_dev.update_layout(
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font_color="#334155",
                title_font_color="#ea580c",
                yaxis_title="Transaction Volume"
            )
            st.plotly_chart(fig_dev, use_container_width=True)
            st.caption("A stacked histogram cleanly visualizing both the total transaction volume and the proportion of high-risk activity across different device types.")
        else:
            st.info("Device type data not available.")

    st.subheader("Payment Method vs Risk")
    if "payment_method" in filtered_txns.columns:
        pm_risk = (
            filtered_txns.groupby("payment_method")["risk_score"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_risk", "count": "volume"})
        )
        # Adding a constant root for the treemap to render cleanly
        pm_risk["root"] = "All Payment Methods"
        
        fig_pm = px.treemap(
            pm_risk,
            path=["root", "payment_method"],
            values="volume",
            color="avg_risk",
            color_continuous_scale="Reds",
            title="Payment Method Risk Treemap",
            labels={"avg_risk": "Avg Risk Score", "volume": "Txn Volume", "root": ""},
        )
        fig_pm.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font_color="#334155",
            title_font_color="#ea580c",
            margin=dict(t=50, l=25, r=25, b=25)
        )
        st.plotly_chart(fig_pm, use_container_width=True)
        st.caption("Treemap sizing represents the total volume of transactions per payment method, while the color intensity indicates the average risk score. This isolates high-volume, high-risk payment vectors.")


# ─────────────────────────────────────────────────────────────────
# TAB 4 — FLAGGED RECORDS
# ─────────────────────────────────────────────────────────────────
with tab4:
    st.subheader(f"🚨 Flagged Transactions (risk ≥ {risk_threshold})")

    display_txns = filtered_txns.sort_values("risk_score", ascending=False).copy()
    display_txns["timestamp"] = display_txns["timestamp"].astype(str)

    # Colour-code by risk label
    def highlight_risk(row):
        colours = {
            "Critical": "background-color: rgba(239,68,68,0.25)",
            "High":     "background-color: rgba(249,115,22,0.20)",
            "Medium":   "background-color: rgba(234,179,8,0.15)",
            "Low":      "",
        }
        return [colours.get(row.get("risk_label", ""), "")] * len(row)

    cols_to_show = [c for c in [
        "transaction_id", "user_id", "timestamp", "purchase_amount",
        "payment_method", "transaction_country", "ip_asn", "device_type",
        "risk_score", "risk_label", "reasons"
    ] if c in display_txns.columns]

    st.dataframe(
        display_txns[cols_to_show].style.apply(highlight_risk, axis=1),
        use_container_width=True,
        height=400,
    )

    st.markdown("---")
    st.subheader("⚠️ High-Risk Users")
    high_risk_users = user_summaries[user_summaries["user_risk_score"] >= risk_threshold].copy()

    user_cols = [c for c in [
        "user_id", "username", "country", "failed_login_count",
        "recent_password_reset_hours", "account_age_days", "flagged_txn_count",
        "avg_txn_risk", "user_risk_score", "user_risk_label"
    ] if c in high_risk_users.columns]

    st.dataframe(
        high_risk_users[user_cols],
        use_container_width=True,
        height=350,
    )

    st.markdown("---")
    st.subheader("📥 Export Flagged Data")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_txns = display_txns[cols_to_show].to_csv(index=False)
        st.download_button(
            "⬇ Download Flagged Transactions (CSV)",
            data=csv_txns,
            file_name="flagged_transactions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        csv_users = high_risk_users[user_cols].to_csv(index=False)
        st.download_button(
            "⬇ Download High-Risk Users (CSV)",
            data=csv_users,
            file_name="high_risk_users.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────
# TAB 5 — AI INSIGHTS
# ─────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("🤖 AI-Powered Risk Analyst")
    st.markdown(
        "Leverage the configured LLM ("
        f"`{os.environ.get('LLM_PROVIDER', 'gemini')}`"
        ") to generate qualitative explanations for why a specific user was flagged."
    )

    # Get high risk users to populate the dropdown
    hr_users = user_summaries[user_summaries["user_risk_score"] >= risk_threshold].copy()
    
    if hr_users.empty:
        st.info("No high-risk users found above the current threshold.")
    else:
        # Create a dropdown mapping visually friendly strings to the underlying User ID
        options = hr_users.apply(
            lambda x: f"{x['user_id']} — Risk: {x['user_risk_score']:g} ({x['flagged_txn_count']} flagged txns)", 
            axis=1
        ).tolist()
        
        selected_option = st.selectbox("Select a High-Risk User to Analyse", options)
        
        if st.button("Generate AI Explanation", type="primary"):
            # Extract User ID from the start of the selected string
            selected_uid = selected_option.split(" —")[0]
            
            with st.spinner("Initialising LLM Provider..."):
                try:
                    provider = get_provider()
                except Exception as e:
                    st.error(f"Failed to load LLM provider: {e}")
                    provider = None
            
            if provider:
                with st.spinner(f"Requesting analysis from {provider.name}..."):
                    try:
                        # Extract the required data rows
                        user_row = users_df[users_df["user_id"] == selected_uid].iloc[0]
                        user_txns = scored_txns[scored_txns["user_id"] == selected_uid]
                        rules_score = int(hr_users[hr_users["user_id"] == selected_uid].iloc[0]["user_risk_score"])
                        
                        # Run the actual AI Analysis function
                        result = analyze_user_behavior(user_row, user_txns, provider, rules_score)
                        
                        # Render result in a clean chat-like message
                        st.success(f"Analysis complete using **{provider.name}**")
                        with st.chat_message("ai", avatar="🤖"):
                            st.markdown(f"### Report for User `{selected_uid}`")
                            st.write(result["ai_analysis"])
                            
                    except Exception as e:
                        st.error(f"Error during AI analysis generation: {e}")

