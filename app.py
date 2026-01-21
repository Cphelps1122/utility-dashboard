import streamlit as st
import pandas as pd
import altair as alt
from prophet import Prophet
from io import StringIO

st.set_page_config(page_title="Utility Dashboard", layout="wide")

# -----------------------------
# DATA LOADER
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Database with pivot tables.xlsx", sheet_name="Sheet1")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month_name().str[:3]

    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    df["Month"] = pd.Categorical(df["Month"], categories=month_order, ordered=True)

    df["Cost_per_Unit"] = df["$ Amt"] / df["# units"]

    return df, month_order

df, month_order = load_data()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("Filters")

prop_options = sorted(df["Prop Name"].unique())
utility_options = sorted(df["Utility"].unique())
year_options = sorted(df["Year"].unique())

selected_prop = st.sidebar.selectbox("Property", prop_options)
selected_utility = st.sidebar.selectbox("Utility", utility_options)
selected_year = st.sidebar.selectbox("Year", year_options)

filtered = df[
    (df["Prop Name"] == selected_prop) &
    (df["Utility"] == selected_utility) &
    (df["Year"] == selected_year)
].copy()

st.title("Utility Dashboard")
st.caption(f"{selected_prop} — {selected_utility} — {selected_year}")

# -----------------------------
# TOP METRICS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Usage", f"{filtered['Usage'].sum():,}")
col2.metric("Total Cost ($)", f"{filtered['$ Amt'].sum():,.0f}")
col3.metric("Avg Monthly Cost ($)", f"{filtered['$ Amt'].mean():,.2f}")
col4.metric("Avg Cost per Unit ($)", f"{filtered['Cost_per_Unit'].mean():,.2f}")

# -----------------------------
# TABS
# -----------------------------
tab_usage, tab_cost, tab_yoy, tab_summary, tab_forecast, tab_map_mix = st.tabs(
    ["📈 Usage Trend", "💵 Cost Trend", "📊 Year-over-Year", "🏨 Property Summary", "🔮 Forecast", "🗺️ Map & Mix"]
)

# -----------------------------
# TAB 1: USAGE TREND
# -----------------------------
with tab_usage:
    st.subheader("Monthly Usage Trend")

    usage_chart = (
        alt.Chart(filtered)
        .mark_line(point=True)
        .encode(
            x=alt.X("Month", sort=month_order),
            y="Usage",
            tooltip=["Month", "Usage", "$ Amt"]
        )
        .properties(height=350)
    )
    st.altair_chart(usage_chart, use_container_width=True)

    st.subheader("Data Table")
    st.dataframe(filtered.sort_values("Date"), use_container_width=True)

    csv_buf = StringIO()
    filtered.to_csv(csv_buf, index=False)
    st.download_button(
        "Download filtered data as CSV",
        data=csv_buf.getvalue(),
        file_name=f"{selected_prop}_{selected_utility}_{selected_year}_usage.csv",
        mime="text/csv"
    )

# -----------------------------
# TAB 2: COST TREND
# -----------------------------
with tab_cost:
    st.subheader("Monthly Cost Trend")

    cost_chart = (
        alt.Chart(filtered)
        .mark_line(point=True, color="orange")
        .encode(
            x=alt.X("Month", sort=month_order),
            y="$ Amt",
            tooltip=["Month", "$ Amt", "Usage"]
        )
        .properties(height=350)
    )
    st.altair_chart(cost_chart, use_container_width=True)

    st.subheader("Cost per Unit")
    cpu_chart = (
        alt.Chart(filtered)
        .mark_line(point=True, color="green")
        .encode(
            x=alt.X("Month", sort=month_order),
            y="Cost_per_Unit",
            tooltip=["Month", "Cost_per_Unit"]
        )
        .properties(height=350)
    )
    st.altair_chart(cpu_chart, use_container_width=True)

# -----------------------------
# TAB 3: YEAR-OVER-YEAR
# -----------------------------
with tab_yoy:
    st.subheader("Year-over-Year Usage Comparison")

    yoy_filtered = df[
        (df["Prop Name"] == selected_prop) &
        (df["Utility"] == selected_utility)
    ].copy()

    pivot = yoy_filtered.pivot_table(
        index="Month",
        columns="Year",
        values="Usage",
        aggfunc="sum"
    ).reset_index()

    # FIX: Convert year columns to strings so Altair doesn't break
    pivot.columns = pivot.columns.map(lambda c: str(c) if isinstance(c, int) else c)

    if len(pivot.columns) > 2:
        chart = (
            alt.Chart(pivot)
            .transform_fold(
                list(pivot.columns[1:]),
                as_=["Year", "Usage"]
            )
            .mark_line(point=True)
            .encode(
                x=alt.X("Month", sort=month_order),
                y="Usage:Q",
                color="Year:N",
                tooltip=["Month", "Year", "Usage"]
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough years of data for YOY comparison.")

    st.subheader("YOY Table")
    st.dataframe(pivot, use_container_width=True)

# -----------------------------
# TAB 4: PROPERTY SUMMARY
# -----------------------------
with tab_summary:
    st.subheader("Summary by Property and Utility")

    summary = df.groupby(["Prop Name", "Utility"]).agg(
        Total_Usage=("Usage", "sum"),
        Total_Cost=("$ Amt", "sum"),
        Avg_Monthly_Cost=("$ Amt", "mean"),
        Avg_Cost_per_Unit=("Cost_per_Unit", "mean")
    ).reset_index()

    st.dataframe(summary, use_container_width=True)

    st.subheader("Totals by Property")
    prop_totals = summary.groupby("Prop Name").agg(
        Total_Usage=("Total_Usage", "sum"),
        Total_Cost=("Total_Cost", "sum")
    ).reset_index()
    st.dataframe(prop_totals, use_container_width=True)

    csv_buf2 = StringIO()
    summary.to_csv(csv_buf2, index=False)
    st.download_button(
        "Download summary as CSV",
        data=csv_buf2.getvalue(),
        file_name="property_summary.csv",
        mime="text/csv"
    )

# -----------------------------
# TAB 5: FORECAST (PROPHET)
# -----------------------------
with tab_forecast:
    st.subheader("Usage Forecast (Prophet)")

    series = df[
        (df["Prop Name"] == selected_prop) &
        (df["Utility"] == selected_utility)
    ].copy()

    monthly = series.groupby(pd.Grouper(key="Date", freq="MS")).agg(
        Usage=("Usage", "sum")
    ).reset_index()

    if len(monthly) >= 6:
        forecast_periods = st.slider("Forecast months ahead", 3, 24, 12)

        prophet_df = monthly.rename(columns={"Date": "ds", "Usage": "y"})
        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=forecast_periods, freq="MS")
        forecast = model.predict(future)

        forecast_chart_df = forecast[["ds", "yhat"]].merge(
            prophet_df, on="ds", how="left"
        )

        base = alt.Chart(forecast_chart_df).encode(
            x="ds:T"
        )

        actual_line = base.mark_line(point=True, color="blue").encode(
            y="y:Q",
            tooltip=["ds", "y"]
        )

        forecast_line = base.mark_line(color="red").encode(
            y="yhat:Q",
            tooltip=["ds", "yhat"]
        )

        st.altair_chart(actual_line + forecast_line, use_container_width=True)
        st.caption("Blue = actual, Red = forecast (yhat)")
        st.dataframe(forecast.tail(forecast_periods), use_container_width=True)
    else:
        st.info("Need at least 6 months of history to run a meaningful forecast.")

# -----------------------------
# TAB 6: MAP & UTILITY MIX
# -----------------------------
with tab_map_mix:
    st.subheader("Property Map & Utility Mix")

    coords = {
        ("Twin Falls", "ID"): (42.5629, -114.4609),
        ("Boise", "ID"): (43.6150, -116.2023),
        ("Bloomington", "IL"): (40.4842, -88.9937),
        ("Southaven", "MS"): (34.9889, -90.0126),
        ("Auburn", "AL"): (32.6099, -85.4808),
        ("Waco", "TX"): (31.5493, -97.1467),
        ("Beaverton", "OR"): (45.4871, -122.8037),
        ("Germantown", "TN"): (35.0868, -89.8101),
        ("Olive Branch", "MS"): (34.9618, -89.8295),
        ("West Jordan", "UT"): (40.6097, -111.9391),
        ("Gainesville", "FL"): (29.6516, -82.3248),
        ("Columbia", "MD"): (39.2037, -76.8610),
        ("Destin", "FL"): (30.3935, -86.4958),
        ("Athens", "GA"): (33.9519, -83.3576),
        ("Manhattan", "KS"): (39.1836, -96.5717),
        ("Carmel", "IN"): (39.9784, -86.1180),
    }

    props = df.groupby(["Prop Name", "City", "State", "# units"]).agg(
        Total_Cost=("$ Amt", "sum")
    ).reset_index()

    props["lat"] = props.apply(
        lambda r: coords.get((r["City"], r["State"]), (None, None))[0], axis=1
    )
    props["lon"] = props.apply(
        lambda r: coords.get((r["City"], r["State"]), (None, None))[1], axis=1
    )

    map_df = props.dropna(subset=["lat", "lon"])

    st.map(map_df[["lat", "lon"]])

    st.subheader("Utility Cost Mix (Selected Property)")

    mix = df[df["Prop Name"] == selected_prop].groupby("Utility").agg(
        Total_Cost=("$ Amt", "sum")
    ).reset_index()

    pie_chart = (
        alt.Chart(mix)
        .mark_arc()
        .encode(
            theta="Total_Cost",
            color="Utility",
            tooltip=["Utility", "Total_Cost"]
        )
        .properties(height=350)
    )
    st.altair_chart(pie_chart, use_container_width=True)

    st.dataframe(mix, use_container_width=True)