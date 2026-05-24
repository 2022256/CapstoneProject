"""
Irish Hospital Demand Forecaster — Streamlit Dashboard
=======================================================
CCT College Dublin | Capstone Project 2026
Andrei Chistiakov | Ivan Popov

Run: python -m streamlit run app.py
Data: NTPF open data — https://www.ntpf.ie/home/waiting-list-data
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Irish Hospital Demand Forecaster",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Irish Hospital Demand Forecaster")
st.caption("Predictive analytics for HSE capacity planning · NTPF open data · GDPR compliant")

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    model_type = st.selectbox("ML Model", ["XGBoost", "Random Forest", "Gradient Boosting"])
    horizon    = st.slider("Forecast Horizon (months)", 1, 12, 6)
    show_ci    = st.checkbox("Show confidence intervals", value=True)
    st.divider()
    st.markdown("**Data source:** [NTPF Open Data](https://www.ntpf.ie/home/waiting-list-data)")
    st.markdown("Aggregate · Anonymous · GDPR compliant")
    st.divider()
    st.markdown("**Data limitation**")
    st.caption(
        "OpenData CSVs are a partial extract. "
        "Full NTPF dashboard shows IPDC 115K + OP 650K = 765K (Mar 2026). "
        "This model trains on the available IPDC CSV series."
    )

# ── Load legacy IPDC data (2014–2020) ────────────────────────────────
def load_legacy(data_dir):
    frames = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.startswith("IPDC-Waiting-List") or not fname.endswith(".csv"):
            continue
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path, encoding="latin-1")
            df.columns = (df.columns
                          .str.replace("ï»¿", "", regex=False)
                          .str.strip().str.lower().str.replace(" ", "_"))
            date_col = next((c for c in df.columns if "archive" in c or "date" in c), None)
            if not date_col:
                continue
            df["archive_date"] = pd.to_datetime(df[date_col], errors="coerce")
            count_col = next((c for c in df.columns if c == "count"), None)
            if count_col:
                df["row_total"] = pd.to_numeric(df[count_col], errors="coerce").fillna(0)
            else:
                exclude = {"archive_date", date_col, "group", "hospital_hipe",
                           "hospital", "specialty_hipe", "specialty",
                           "case_type", "adult/child", "age_categorisation",
                           "time_bands", "adult_child"}
                num_cols = [c for c in df.columns
                            if c not in exclude
                            and pd.api.types.is_numeric_dtype(df[c])]
                if not num_cols:
                    continue
                df["row_total"] = df[num_cols].apply(
                    pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
            frames.append(df[["archive_date", "row_total"]].dropna())
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["year_month", "waiting_total"])

    combined = pd.concat(frames, ignore_index=True)
    monthly = (combined
               .groupby(pd.Grouper(key="archive_date", freq="MS"))["row_total"]
               .sum().reset_index()
               .rename(columns={"archive_date": "year_month", "row_total": "waiting_total"}))
    return monthly[monthly["year_month"] < "2021-01-01"].reset_index(drop=True)


# ── Load OpenData IPDC (2021–2026) ────────────────────────────────────
def load_opendata(data_dir):
    frames = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.startswith("OpenData_IPDC") or not fname.endswith(".csv"):
            continue
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path, encoding="latin-1")
            df.columns = (df.columns
                          .str.replace("ï»¿", "", regex=False)
                          .str.strip().str.lower().str.replace(" ", "_"))
            date_col  = next((c for c in df.columns if "archive" in c), None)
            count_col = next((c for c in df.columns if c in ("count", "total")), None)
            if not date_col or not count_col:
                continue
            df["archive_date"] = pd.to_datetime(df[date_col], errors="coerce")
            df["row_total"]    = pd.to_numeric(df[count_col], errors="coerce").fillna(0)
            frames.append(df[["archive_date", "row_total"]].dropna())
        except Exception:
            continue

    combined = pd.concat(frames, ignore_index=True)
    monthly = (combined
               .groupby(pd.Grouper(key="archive_date", freq="MS"))["row_total"]
               .sum().reset_index()
               .rename(columns={"archive_date": "year_month", "row_total": "waiting_total"}))
    return monthly[monthly["waiting_total"] > 10000].reset_index(drop=True)


# ── Load case type split (OpenData only — has case_type column) ───────
def load_split(data_dir):
    frames = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.startswith("OpenData_IPDC") or not fname.endswith(".csv"):
            continue
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path, encoding="latin-1")
            df.columns = (df.columns
                          .str.replace("ï»¿", "", regex=False)
                          .str.strip().str.lower().str.replace(" ", "_"))
            date_col  = next((c for c in df.columns if "archive" in c), None)
            count_col = next((c for c in df.columns if c in ("count", "total")), None)
            if not date_col or not count_col:
                continue
            df["archive_date"] = pd.to_datetime(df[date_col], errors="coerce")
            df["row_total"]    = pd.to_numeric(df[count_col], errors="coerce").fillna(0)
            df["category"]     = df["case_type"].str.strip() if "case_type" in df.columns else "All"
            frames.append(df[["archive_date", "row_total", "category"]].dropna())
        except Exception:
            continue

    combined = pd.concat(frames, ignore_index=True)
    split = (combined
             .groupby([pd.Grouper(key="archive_date", freq="MS"), "category"])["row_total"]
             .sum().reset_index()
             .rename(columns={"archive_date": "year_month", "row_total": "waiting_total"}))
    return split[split["waiting_total"] > 1000].reset_index(drop=True)


@st.cache_data
def load_all_data():
    legacy   = load_legacy("data")
    opendata = load_opendata("data")
    split    = load_split("data")

    # Merge: legacy pre-2021, opendata from 2021
    monthly = pd.concat([legacy, opendata], ignore_index=True)
    monthly = monthly.sort_values("year_month").reset_index(drop=True)
    return monthly, split


monthly, split = load_all_data()

# ── Feature engineering ───────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    df["month"]           = df["year_month"].dt.month
    df["month_sin"]       = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]       = np.cos(2 * np.pi * df["month"] / 12)
    df["year"]            = df["year_month"].dt.year
    df["quarter"]         = df["year_month"].dt.quarter
    df["trend"]           = range(len(df))
    df["lag_1m"]          = df["waiting_total"].shift(1)
    df["lag_3m"]          = df["waiting_total"].shift(3)
    df["lag_6m"]          = df["waiting_total"].shift(6)
    df["rolling_3m_mean"] = df["waiting_total"].rolling(3).mean()
    df["rolling_6m_mean"] = df["waiting_total"].rolling(6).mean()
    df["rolling_3m_std"]  = df["waiting_total"].rolling(3).std()
    return df.dropna().reset_index(drop=True)

FEATURE_COLS = [
    "month_sin", "month_cos", "year", "quarter", "trend",
    "lag_1m", "lag_3m", "lag_6m",
    "rolling_3m_mean", "rolling_6m_mean", "rolling_3m_std"
]

df_features = add_features(monthly)

# ── Train ─────────────────────────────────────────────────────────────
@st.cache_resource
def train(model_type):
    X = df_features[FEATURE_COLS].values
    y = df_features["waiting_total"].values
    if model_type == "XGBoost":
        m = xgb.XGBRegressor(n_estimators=300, max_depth=4,
                              learning_rate=0.05, random_state=42, verbosity=0)
    elif model_type == "Random Forest":
        m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        m = GradientBoostingRegressor(n_estimators=200, random_state=42)
    m.fit(X[:-6], y[:-6])
    mae = mean_absolute_error(y[-6:], m.predict(X[-6:]))
    m.fit(X, y)
    return m, mae

with st.spinner(f"Training {model_type} on {len(df_features)} months (2014–2026)..."):
    model, mae = train(model_type)

# ── Recursive forecast ────────────────────────────────────────────────
def make_forecast(horizon):
    lag_buffer = list(df_features["waiting_total"].values)
    last = df_features.iloc[-1]
    rows = []
    for step in range(1, horizon + 1):
        nd = df_features["year_month"].iloc[-1] + pd.DateOffset(months=step)
        feat = {
            "month_sin":       np.sin(2 * np.pi * nd.month / 12),
            "month_cos":       np.cos(2 * np.pi * nd.month / 12),
            "year":            nd.year,
            "quarter":         nd.quarter,
            "trend":           last["trend"] + step,
            "lag_1m":          lag_buffer[-1],
            "lag_3m":          lag_buffer[-3],
            "lag_6m":          lag_buffer[-6],
            "rolling_3m_mean": np.mean(lag_buffer[-3:]),
            "rolling_6m_mean": np.mean(lag_buffer[-6:]),
            "rolling_3m_std":  np.std(lag_buffer[-3:]),
        }
        yp = float(model.predict(np.array([[feat[c] for c in FEATURE_COLS]]))[0])
        ci = np.std(lag_buffer[-6:]) * (1 + 0.1 * step)
        rows.append({"date": nd, "forecast": round(yp),
                     "lower": round(max(0, yp - ci)), "upper": round(yp + ci)})
        lag_buffer.append(yp)
    return pd.DataFrame(rows)

fc = make_forecast(horizon)

# ── KPI cards ─────────────────────────────────────────────────────────
latest = int(monthly["waiting_total"].iloc[-1])
fc_end = int(fc["forecast"].iloc[-1])
pct    = (fc_end - latest) / latest * 100
n_months = len(df_features)
yr_start = df_features["year_month"].iloc[0].year

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Waiting List", f"{latest:,}", help="Latest NTPF snapshot (IPDC partial)")
c2.metric(f"Forecast ({horizon}m)", f"{fc_end:,}", f"{pct:+.1f}%")
c3.metric("Model CV MAE", f"{mae:,.0f} patients")
c4.metric("Training Months", f"{n_months}  ({yr_start}–2026)")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Inpatient vs Outpatient", "🔍 Feature Importance"])

with tab1:
    fig = go.Figure()
    # Legacy shading
    legacy_part = monthly[monthly["year_month"] < "2021-01-01"]
    opendata_part = monthly[monthly["year_month"] >= "2021-01-01"]

    fig.add_trace(go.Scatter(
        x=legacy_part["year_month"], y=legacy_part["waiting_total"],
        name="Historical — Legacy (2014–2020)",
        line=dict(color="#f59e0b", width=2),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.08)"))
    fig.add_trace(go.Scatter(
        x=opendata_part["year_month"], y=opendata_part["waiting_total"],
        name="Historical — OpenData (2021–2026)",
        line=dict(color="#2171b5", width=2),
        fill="tozeroy", fillcolor="rgba(33,113,181,0.1)"))
    if show_ci:
        fig.add_trace(go.Scatter(
            x=pd.concat([fc["date"], fc["date"][::-1]]),
            y=pd.concat([fc["upper"], fc["lower"][::-1]]),
            fill="toself", fillcolor="rgba(215,48,39,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence interval"))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["forecast"],
        name=f"Forecast ({model_type})",
        line=dict(color="#d73027", width=2.5, dash="dash")))
    fig.add_hline(y=75000, line_dash="dot", line_color="green",
                  annotation_text="Sláintecare indicative IPDC target",
                  annotation_position="bottom right")
    fig.update_layout(
        title=f"IPDC Waiting List — {horizon}-Month Forecast  |  Trained on {n_months} months ({yr_start}–2026)",
        yaxis_title="Patients Waiting", yaxis_tickformat=",",
        hovermode="x unified", height=480,
        legend=dict(orientation="h", y=1.06))
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "⚠️ **Data note:** These figures reflect the NTPF OpenData CSV series (IPDC subset). "
        "The full national waiting list (IPDC 115K + Outpatient 650K = 765K as of Mar 2026) "
        "is available via the NTPF interactive dashboard at ntpf.ie."
    )

    with st.expander("📋 Forecast table"):
        st.dataframe(
            fc.rename(columns={"date": "Month", "forecast": "Forecast",
                                "lower": "Lower CI", "upper": "Upper CI"})
              .set_index("Month"), use_container_width=True)

with tab2:
    cats    = sorted(split["category"].unique())
    colours = ["#2171b5", "#d73027", "#2ca02c", "#ff7f0e", "#9467bd"]
    fig2    = go.Figure()
    for i, cat in enumerate(cats):
        s = split[split["category"] == cat].sort_values("year_month")
        fig2.add_trace(go.Scatter(
            x=s["year_month"], y=s["waiting_total"], name=cat,
            line=dict(width=2, color=colours[i % len(colours)])))
    fig2.update_layout(
        title="Waiting List by Case Type — OpenData Series (2021–2026)",
        yaxis_title="Patients Waiting", yaxis_tickformat=",",
        hovermode="x unified", height=460,
        legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig2, use_container_width=True)

    summary = (split.groupby("category")["waiting_total"]
               .agg(["mean", "max", "min"])
               .rename(columns={"mean": "Avg Monthly", "max": "Peak", "min": "Lowest"})
               .map(lambda x: f"{x:,.0f}"))
    st.subheader("Summary by Case Type")
    st.dataframe(summary, use_container_width=True)

with tab3:
    fi = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig3 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                  title=f"Feature Importance — {model_type}  (trained on {n_months} months)",
                  color="Importance", color_continuous_scale="Blues")
    fig3.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=440)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        f"**Key finding:** `rolling_3m_mean` dominates at "
        f"{fi['Importance'].iloc[0]*100:.1f}% importance — recent momentum is the "
        "strongest predictor. Seasonality features rank low, confirming the IPDC "
        "waiting list has no strong seasonal pattern.  \n"
        "**CI note:** Confidence bands are heuristic (rolling std × horizon penalty), "
        "not formal statistical prediction intervals."
    )

st.caption(
    "CCT College Dublin · Capstone Project 2026 · Andrei Chistiakov · Ivan Popov  |  "
    "⚠️ IPDC CSV subset only — full national total requires NTPF API access"
)
