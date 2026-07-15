"""
House Price Prediction — Streamlit Web App
Author: Tetali Lakshmi Nagalahari
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .main-header {
    background: linear-gradient(135deg, #0d0f14 0%, #181c26 100%);
    padding: 2.5rem 2rem 2rem; border-radius: 16px;
    margin-bottom: 2rem; border: 1px solid #252a38; text-align: center;
  }
  .main-header h1 { color: #e8eaf2; font-size: 2.4rem; font-weight: 700; margin: 0; }
  .main-header p  { color: #8b91a8; font-size: 1rem; margin: 0.5rem 0 0; }
  .accent { color: #6c9aff; }

  .metric-card {
    background: #13161e; border: 1px solid #252a38;
    border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
  }
  .metric-card .val { font-size: 2rem; font-weight: 700; color: #34d399; }
  .metric-card .lbl { font-size: 0.8rem; color: #8b91a8; text-transform: uppercase; letter-spacing: 0.08em; }

  .predict-result {
    background: linear-gradient(135deg, #13161e, #181c26);
    border: 2px solid #6c9aff; border-radius: 16px;
    padding: 2rem; text-align: center; margin-top: 1.5rem;
  }
  .predict-result .price { font-size: 2.8rem; font-weight: 700; color: #6c9aff; }
  .predict-result .sub   { color: #8b91a8; font-size: 0.9rem; margin-top: 0.4rem; }

  .section-title {
    font-size: 1.3rem; font-weight: 600; color: #e8eaf2;
    border-left: 3px solid #6c9aff; padding-left: 12px; margin: 1.5rem 0 1rem;
  }
  div[data-testid="stSidebar"] { background: #13161e; }
</style>
""", unsafe_allow_html=True)


# ─── LOAD & TRAIN (cached) ─────────────────────────────────────
@st.cache_resource(show_spinner="Training models, please wait...")
def load_and_train():
    df = pd.read_csv("housing_data.csv")
    df.rename(columns={"MedHouseVal": "Price"}, inplace=True)

    df["RoomsPerHousehold"]      = df["AveRooms"]   / df["HouseAge"].replace(0, 1)
    df["BedroomsPerRoom"]        = df["AveBedrms"]  / df["AveRooms"].replace(0, 1)
    df["PopulationPerHousehold"] = df["Population"] / df["AveOccup"].replace(0, 1)

    features = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude",
        "RoomsPerHousehold", "BedroomsPerRoom", "PopulationPerHousehold"
    ]
    X, y = df[features], df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model_defs = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42),
    }
    results = {}
    for name, m in model_defs.items():
        m.fit(X_tr, y_train)
        yp = m.predict(X_te)
        results[name] = {
            "model": m, "y_pred": yp,
            "R2":   round(r2_score(y_test, yp), 4),
            "MAE":  round(mean_absolute_error(y_test, yp), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, yp)), 4),
        }
    return df, features, scaler, results, X_test, y_test

df, features, scaler, results, X_test, y_test = load_and_train()

# ─── SIDEBAR: PREDICT ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Predict a Price")
    st.markdown("---")
    model_choice = st.selectbox("Model", ["Random Forest","Gradient Boosting","Linear Regression"])

    st.markdown("### 📐 House Features")
    med_inc    = st.slider("Median Income (×$10k)",            0.5,  15.0,  5.0, 0.1)
    house_age  = st.slider("House Age (years)",                1,    52,    20)
    ave_rooms  = st.slider("Avg Rooms / Household",            1.0,  14.0,  5.0, 0.1)
    ave_bedrms = st.slider("Avg Bedrooms / Household",         0.5,  5.0,   1.0, 0.1)
    population = st.slider("Block Population",                 5,    35000, 1500)
    ave_occup  = st.slider("Avg Occupants / Household",        1.0,  10.0,  3.0, 0.1)

    st.markdown("### 🌐 Location")
    latitude   = st.slider("Latitude",   32.5,  42.0,  37.5, 0.01)
    longitude  = st.slider("Longitude", -124.5, -114.0,-120.0, 0.01)

    rph = ave_rooms  / max(house_age, 1)
    bpr = ave_bedrms / max(ave_rooms, 1)
    pph = population / max(ave_occup, 1)

    inp     = np.array([[med_inc, house_age, ave_rooms, ave_bedrms,
                         population, ave_occup, latitude, longitude,
                         rph, bpr, pph]])
    inp_sc  = scaler.transform(inp)
    pred    = results[model_choice]["model"].predict(inp_sc)[0]

    st.markdown("---")
    st.markdown(f"""
    <div class="predict-result">
        <div style="font-size:0.8rem;color:#8b91a8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">Estimated Price</div>
        <div class="price">${pred * 100_000:,.0f}</div>
        <div class="sub">Model: {model_choice}</div>
    </div>""", unsafe_allow_html=True)

# ─── MAIN PAGE ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏠 House Price <span class="accent">Prediction</span></h1>
  <p>California Housing Dataset · Linear Regression · Random Forest · Gradient Boosting</p>
  <p style="font-size:0.82rem;margin-top:4px;">
    By <strong style="color:#e8eaf2;">Tetali Lakshmi Nagalahari</strong>
  </p>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Performance", "🔮 Predictions Analysis",
    "📈 Data Insights", "🔍 Feature Importance"
])

# ── TAB 1: Model Performance ───────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Model Metrics at a Glance</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:0.95rem;font-weight:600;color:#e8eaf2;margin-bottom:10px;">{name}</div>
              <div class="val">{res['R2']}</div>
              <div class="lbl">R² Score</div>
              <hr style="border-color:#252a38;margin:10px 0;">
              <div style="display:flex;justify-content:space-around;">
                <div>
                  <div style="font-size:1.1rem;font-weight:600;color:#a78bfa;">{res['MAE']}</div>
                  <div class="lbl">MAE</div>
                </div>
                <div>
                  <div style="font-size:1.1rem;font-weight:600;color:#6c9aff;">{res['RMSE']}</div>
                  <div class="lbl">RMSE</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Performance Bar Charts</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#0d0f14")
    COLORS = ["#6c9aff","#a78bfa","#34d399"]
    mnames = list(results.keys())

    for ax, (metric, key) in zip(axes, [("R² Score","R2"),("MAE","MAE"),("RMSE","RMSE")]):
        vals = [results[m][key] for m in mnames]
        bars = ax.bar([m.replace(" ","\n") for m in mnames], vals, color=COLORS, edgecolor="#252a38", width=0.5)
        ax.set_title(metric, color="#e8eaf2", fontsize=12, fontweight="bold", pad=10)
        ax.set_facecolor("#13161e")
        ax.tick_params(colors="#8b91a8", labelsize=9)
        for sp in ax.spines.values(): sp.set_color("#252a38")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f"{val:.3f}", ha="center", va="bottom", color="#e8eaf2", fontsize=9, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── TAB 2: Predictions Analysis ───────────────────────────────
with tab2:
    sel = st.selectbox("Choose model:", list(results.keys()), key="pred_sel")
    yp  = results[sel]["y_pred"]
    res = y_test.values - yp

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d0f14")

    axes[0].scatter(y_test, yp, alpha=0.25, s=10, color="#6c9aff")
    lims = [min(y_test.min(), yp.min()), max(y_test.max(), yp.max())]
    axes[0].plot(lims, lims, "r--", lw=2, label="Perfect Fit")
    axes[0].set_xlabel("Actual Price ($100k)", color="#8b91a8")
    axes[0].set_ylabel("Predicted Price ($100k)", color="#8b91a8")
    axes[0].set_title(f"Actual vs Predicted — R²={results[sel]['R2']}", color="#e8eaf2", fontweight="bold")
    axes[0].set_facecolor("#13161e"); axes[0].tick_params(colors="#8b91a8")
    for sp in axes[0].spines.values(): sp.set_color("#252a38")
    axes[0].legend(facecolor="#13161e", labelcolor="#e8eaf2")

    axes[1].hist(res, bins=60, color="#a78bfa", edgecolor="#252a38", alpha=0.85)
    axes[1].axvline(0, color="#34d399", linestyle="--", lw=2)
    axes[1].set_xlabel("Residual", color="#8b91a8"); axes[1].set_ylabel("Count", color="#8b91a8")
    axes[1].set_title("Residuals Distribution", color="#e8eaf2", fontweight="bold")
    axes[1].set_facecolor("#13161e"); axes[1].tick_params(colors="#8b91a8")
    for sp in axes[1].spines.values(): sp.set_color("#252a38")

    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Sample Predictions (first 15 test rows)</div>', unsafe_allow_html=True)
    sample_df = pd.DataFrame({
        "Actual ($100k)":    y_test.values[:15].round(3),
        "Predicted ($100k)": yp[:15].round(3),
        "Error":             (y_test.values[:15] - yp[:15]).round(3),
        "Abs Error":         np.abs(y_test.values[:15] - yp[:15]).round(3),
    })
    st.dataframe(sample_df.style.background_gradient(subset=["Abs Error"], cmap="RdYlGn_r"),
                 use_container_width=True)

# ── TAB 3: Data Insights ──────────────────────────────────────
with tab3:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", f"{len(df):,}")
    c2.metric("Features",      "11")
    c3.metric("Min Price",     f"${df['Price'].min()*1e5:,.0f}")
    c4.metric("Max Price",     f"${df['Price'].max()*1e5:,.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d0f14")

    axes[0].hist(df["Price"]*1e5, bins=60, color="#6c9aff", edgecolor="#252a38", alpha=0.85)
    axes[0].set_xlabel("House Price ($)", color="#8b91a8"); axes[0].set_ylabel("Count", color="#8b91a8")
    axes[0].set_title("Price Distribution", color="#e8eaf2", fontweight="bold")
    axes[0].set_facecolor("#13161e"); axes[0].tick_params(colors="#8b91a8")
    for sp in axes[0].spines.values(): sp.set_color("#252a38")

    axes[1].scatter(df["MedInc"], df["Price"], alpha=0.12, s=6, color="#34d399")
    axes[1].set_xlabel("Median Income (×$10k)", color="#8b91a8"); axes[1].set_ylabel("Price ($100k)", color="#8b91a8")
    axes[1].set_title("Income vs Price", color="#e8eaf2", fontweight="bold")
    axes[1].set_facecolor("#13161e"); axes[1].tick_params(colors="#8b91a8")
    for sp in axes[1].spines.values(): sp.set_color("#252a38")

    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#0d0f14"); ax.set_facecolor("#13161e")
    sns.heatmap(df[features+["Price"]].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", ax=ax, linewidths=0.5, annot_kws={"size":8}, linecolor="#252a38")
    ax.set_title("Feature Correlation Matrix", color="#e8eaf2", fontweight="bold", fontsize=13)
    ax.tick_params(colors="#8b91a8")
    st.pyplot(fig); plt.close()

# ── TAB 4: Feature Importance ─────────────────────────────────
with tab4:
    st.info("Available for tree-based models (Random Forest & Gradient Boosting).")
    sel2 = st.selectbox("Model:", ["Random Forest","Gradient Boosting"], key="fi_sel")
    m_fi = results[sel2]["model"]

    if hasattr(m_fi, "feature_importances_"):
        imp = pd.Series(m_fi.feature_importances_, index=features).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("#0d0f14"); ax.set_facecolor("#13161e")
        colors_g = plt.cm.Blues(np.linspace(0.3, 0.9, len(imp)))
        bars = ax.barh(imp.index, imp.values, color=colors_g, edgecolor="#252a38")
        ax.set_xlabel("Importance Score", color="#8b91a8", fontsize=11)
        ax.set_title(f"Feature Importance — {sel2}", color="#e8eaf2", fontweight="bold", fontsize=13)
        ax.tick_params(colors="#8b91a8")
        for sp in ax.spines.values(): sp.set_color("#252a38")
        for bar, val in zip(bars, imp.values):
            ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
                    f"{val:.3f}", va="center", color="#e8eaf2", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        imp_df = pd.DataFrame({
            "Rank": range(1, len(imp)+1),
            "Feature": imp.index[::-1],
            "Importance Score": imp.values[::-1].round(4),
        })
        st.dataframe(imp_df, use_container_width=True)
