import streamlit as st
import pandas as pd
import altair as alt
from prophet import Prophet
from io import BytesIO
import numpy as np

st.set_page_config(page_title="Utility Dashboard", layout="wide")

# Session flag for landing page
if "show_dashboard" not in st.session_state:
    st.session_state["show_dashboard"] = False

# -----------------------------
# GLOBAL STYLES
# -----------------------------
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        h1, h2, h3 {
            font-weight: 600;
            color: #1F3B4D;
        }
        .header-container {
            background: linear-gradient(90deg, #1F618D, #2980B9);
            padding: 15px 25px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.12);
            margin-bottom: 20px;
            color: #ffffff;
        }
        .metric-card {
            background-color: #ffffff;
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .section-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 25px;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# DATA LOADER
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("gridforge_1.1.xlsx", sheet_name="Property")

    # Clean dates
    df["Billing Date"] = pd.to_datetime(df["Billing Date"], errors="coerce")

    # Month + Year
    df["Year"] = df["Billing Date"].dt.year
    df["Month"] = df["Billing Date"].dt.month_name().str[:3]

    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    df["Month"] = pd.Categorical(df["Month"], categories=month_order, ordered=True)

    # Cost per unit
    df["Cost_per_Unit"] = df["$ Amount"] / df["Usage"]

    return df, month_order

with st.spinner("Loading latest data…"):
    df, month_order = load_data()

last_updated = df["Billing Date"].max()

# -----------------------------
# LANDING PAGE
# -----------------------------
if not st.session_state["show_dashboard"]:
    st.markdown(
        """
        <div style="display:flex;justify-content:center;align-items:center;height:80vh;">
          <div style="background-color:#ffffff;padding:40px 50px;border-radius:16px;
                      box-shadow:0 4px 18px rgba(0,0,0,0.12);max-width:650px;width:100%;">
            <h1 style="text-align:center;margin-bottom:0.5rem;">Utility Performance Dashboard</h1>
            <p style="text-align:center;color:#555;font-size:1.05rem;margin-bottom:1.5rem;">
              Portfolio-wide visibility into usage, cost, anomalies, and efficiency—built for operators, not analysts.
            </p>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Properties", df["Property Name"].nunique())
    with c2:
        st.metric("Total Utilities", df["Utility"].nunique())
    with c3:
        st.metric("Years of History", df["Year"].nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Start Analysis", use_container_width=True, type="primary"):
        st.session_state["show_dashboard"] = True
        st.experimental_rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()

# -----------------------------
# DASHBOARD HEADER
# -----------------------------
st.markdown(
    f"""
    <div class="header-container">
        <h2>Portfolio Utility Dashboard</h2>
        <span>Last Updated: {last_updated.date()}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# FILTERS
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    selected_property = st.selectbox(
        "Select Property",
        ["All"] + sorted(df["Property Name"].unique())
    )

with col2:
    selected_utility = st.selectbox(
        "Select Utility",
        ["All"] + sorted(df["Utility"].unique())
    )

with col3:
    selected_year = st.selectbox(
        "Select Year",
        ["All"] + sorted(df["Year"].unique())
    )

# Apply filters
filtered = df.copy()

if selected_property != "All":
    filtered = filtered[filtered["Property Name"] == selected_property]

if selected_utility != "All":
    filtered = filtered[filtered["Utility"] == selected_utility]

if selected_year != "All":
    filtered = filtered[filtered["Year"] == selected_year]

# -----------------------------
# METRICS
# -----------------------------
m1, m2, m3, m4 = st.columns(4)

m1.metric("Total Spend", f"${filtered['$ Amount'].sum():,.0f}")
m2.metric("Total Usage", f"{filtered['Usage'].sum():,.0f}")
m3.metric("Avg Cost/Unit", f"${filtered['Cost_per_Unit'].mean():.2f}")
m4.metric("Bills Count", f"{len(filtered):,}")

# -----------------------------
# COST TREND
# -----------------------------
st.markdown("### Monthly Cost Trend")

cost_trend = (
    filtered.groupby(["Year", "Month"], as_index=False)["$ Amount"].sum()
)

chart = (
    alt.Chart(cost_trend)
    .mark_line(point=True)
    .encode(
        x=alt.X("Month", sort=month_order),
        y="$ Amount",
        color="Year:N"
    )
    .properties(height=350)
)

st.altair_chart(chart, use_container_width=True)

# -----------------------------
# USAGE TREND
# -----------------------------
st.markdown("### Monthly Usage Trend")

usage_trend = (
    filtered.groupby(["Year", "Month"], as_index=False)["Usage"].sum()
)

chart2 = (
    alt.Chart(usage_trend)
    .mark_line(point=True)
    .encode(
        x=alt.X("Month", sort=month_order),
        y="Usage",
        color="Year:N"
    )
    .properties(height=350)
)

st.altair_chart(chart2, use_container_width=True)

# -----------------------------
# FORECASTING
# -----------------------------
st.markdown("### Forecasting (Prophet)")

forecast_df = (
    filtered.groupby("Billing Date", as_index=False)["$ Amount"].sum()
    .rename(columns={"Billing Date": "ds", "$ Amount": "y"})
)

if len(forecast_df) > 3:
    model = Prophet()
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    forecast_chart = (
        alt.Chart(forecast)
        .mark_line()
        .encode(
            x="ds:T",
            y="yhat:Q"
        )
        .properties(height=350)
    )

    st.altair_chart(forecast_chart, use_container_width=True)
else:
    st.info("Not enough data for forecasting.")

# -----------------------------
# PROPERTY BREAKDOWN
# -----------------------------
st.markdown("### Spend by Property")

prop_breakdown = (
    df.groupby("Property Name", as_index=False)["$ Amount"].sum()
    .sort_values("$ Amount", ascending=False)
)

bar = (
    alt.Chart(prop_breakdown)
    .mark_bar()
    .encode(
        x="Property Name:N",
        y="$ Amount:Q",
        tooltip=["Property Name", "$ Amount"]
    )
    .properties(height=400)
)

st.altair_chart(bar, use_container_width=True)
