"""
House Price Prediction — Model Training Script
Author: Tetali Lakshmi Nagalahari
Description: Trains Linear Regression, Random Forest, and Gradient Boosting
             models on California Housing data and saves the best model + plots.
Run: python train_model.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  HOUSE PRICE PREDICTION — MODEL TRAINING")
print("=" * 60)

print("\n📦 Loading California Housing dataset...")
df = pd.read_csv("housing_data.csv")
df.rename(columns={"MedHouseVal": "Price"}, inplace=True)

print(f"   Shape       : {df.shape}")
print(f"   Features    : {list(df.columns[:-1])}")
print(f"   Target      : Price (Median house value ×$100k)")
print(f"   Missing vals: {df.isnull().sum().sum()}")
print(f"\n{df.describe().round(3)}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n🔧 Feature Engineering...")
df["RoomsPerHousehold"]      = df["AveRooms"]   / df["HouseAge"].replace(0, 1)
df["BedroomsPerRoom"]        = df["AveBedrms"]  / df["AveRooms"].replace(0, 1)
df["PopulationPerHousehold"] = df["Population"] / df["AveOccup"].replace(0, 1)

features = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
    "RoomsPerHousehold", "BedroomsPerRoom", "PopulationPerHousehold"
]

X = df[features]
y = df["Price"]
print(f"   Total features after engineering: {X.shape[1]}")

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT & SCALING
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"\n📊 Split → Train: {len(X_train)}  Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42),
}

results = {}
print("\n🤖 Training Models...\n")
for name, model in models.items():
    print(f"   ▸ {name}...")
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv   = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="r2").mean()
    results[name] = {"model": model, "y_pred": y_pred,
                     "R2": round(r2,4), "MAE": round(mae,4),
                     "RMSE": round(rmse,4), "CV_R2": round(cv,4)}
    print(f"     R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  CV_R²={cv:.4f}")

# ─────────────────────────────────────────────
# 5. RESULTS SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)
summary = pd.DataFrame({k: {"R²":v["R2"],"MAE":v["MAE"],"RMSE":v["RMSE"],"CV R²":v["CV_R2"]}
                         for k,v in results.items()}).T
print(summary.to_string())
best_name = max(results, key=lambda k: results[k]["R2"])
print(f"\n🏆 Best Model: {best_name}  (R² = {results[best_name]['R2']})")

# ─────────────────────────────────────────────
# 6. SAVE ARTIFACTS
# ─────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
with open("model/best_model.pkl","wb") as f: pickle.dump(results[best_name]["model"], f)
with open("model/scaler.pkl",    "wb") as f: pickle.dump(scaler, f)
with open("model/features.pkl",  "wb") as f: pickle.dump(features, f)
print("\n✅ Artifacts saved to model/")

# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)
COLORS = ["#6c9aff", "#a78bfa", "#34d399"]
plt.style.use("seaborn-v0_8-darkgrid")

# Plot 1: Model Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")
for ax, (metric, key) in zip(axes, [("R² Score","R2"),("MAE","MAE"),("RMSE","RMSE")]):
    vals = [results[m][key] for m in results]
    bars = ax.bar([m.replace(" ","\n") for m in results], vals, color=COLORS, edgecolor="white")
    ax.set_title(metric, fontsize=13, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout(); plt.savefig("plots/model_comparison.png", dpi=150, bbox_inches="tight"); plt.close()

# Plot 2: Actual vs Predicted
y_pred_best = results[best_name]["y_pred"]
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(y_test, y_pred_best, alpha=0.3, s=12, color="#6c9aff")
lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
ax.plot(lims, lims, "r--", lw=2, label="Perfect Fit")
ax.set_xlabel("Actual Price ($100k)"); ax.set_ylabel("Predicted Price ($100k)")
ax.set_title(f"Actual vs Predicted — {best_name}\nR² = {results[best_name]['R2']}", fontsize=13, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png", dpi=150, bbox_inches="tight"); plt.close()

# Plot 3: Feature Importance
best_m = results[best_name]["model"]
if hasattr(best_m, "feature_importances_"):
    imp = pd.Series(best_m.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9,6))
    ax.barh(imp.index, imp.values, color="#6c9aff", edgecolor="white")
    ax.set_xlabel("Importance Score"); ax.set_title(f"Feature Importance — {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig("plots/feature_importance.png", dpi=150, bbox_inches="tight"); plt.close()

# Plot 4: Residuals
residuals = y_test.values - y_pred_best
fig, axes = plt.subplots(1, 2, figsize=(13,5))
axes[0].hist(residuals, bins=50, color="#a78bfa", edgecolor="white", alpha=0.85)
axes[0].axvline(0, color="red", linestyle="--", lw=1.5)
axes[0].set_title("Residuals Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Residual (Actual − Predicted)")
axes[1].scatter(y_pred_best, residuals, alpha=0.3, s=12, color="#34d399")
axes[1].axhline(0, color="red", linestyle="--", lw=1.5)
axes[1].set_title("Residuals vs Predicted", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Predicted Price"); axes[1].set_ylabel("Residual")
plt.tight_layout(); plt.savefig("plots/residuals.png", dpi=150, bbox_inches="tight"); plt.close()

# Plot 5: Correlation Heatmap
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df[features+["Price"]].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", ax=ax, linewidths=0.5, annot_kws={"size":8})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig("plots/correlation_heatmap.png", dpi=150, bbox_inches="tight"); plt.close()

print("📊 All 5 plots saved to plots/")
print("\n✅ Done! Run:  streamlit run app.py")
