# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


# ------------------ Utilities ------------------
def calculate_aic(y_true, y_pred, k, eps=1e-12):
    """AIC = n * ln(RSS/n) + 2k"""
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    rss = max(rss, eps)  # avoid log(0)
    return n * np.log(rss / n) + 2 * k


def fit_model(df, target, features):
    """Fit linear regression, return model, aic."""
    y = df[target].values
    X = df[features].values if features else np.zeros((len(y), 0))
    model = LinearRegression().fit(X, y) if features else None
    y_pred = model.predict(X) if features else np.mean(y) * np.ones_like(y)
    k = len(features) + 1 if features else 1
    aic = calculate_aic(y, y_pred, k)
    return model, aic, y_pred


def forward_selection(df, target, predictors):
    """Forward stepwise AIC."""
    remaining = predictors.copy()
    selected = []
    records = []
    while remaining:
        best_feat, best_aic, best_model, best_pred = None, np.inf, None, None
        for feat in remaining:
            candidate = selected + [feat]
            model, aic, y_pred = fit_model(df, target, candidate)
            if aic < best_aic:
                best_feat, best_aic, best_model, best_pred = feat, aic, model, y_pred
        if best_feat is None:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        records.append({
            "step": len(records)+1,
            "added_feature": best_feat,
            "removed_feature": None,
            "selected_features": selected.copy(),
            "aic": best_aic,
            "model": best_model,
            "y_pred": best_pred
        })
    return records


def backward_elimination(df, target, predictors):
    """Backward stepwise AIC."""
    selected = predictors.copy()
    records = []
    while len(selected) > 0:
        model, aic, y_pred = fit_model(df, target, selected)
        best_aic, best_model, best_pred, worst_feat = aic, model, y_pred, None
        for feat in selected:
            candidate = [f for f in selected if f != feat]
            model, aic, y_pred = fit_model(df, target, candidate)
            if aic < best_aic:
                best_aic, best_model, best_pred, worst_feat = aic, model, y_pred, feat
        if worst_feat is None:
            break
        selected.remove(worst_feat)
        records.append({
            "step": len(records)+1,
            "added_feature": None,
            "removed_feature": worst_feat,
            "selected_features": selected.copy(),
            "aic": best_aic,
            "model": best_model,
            "y_pred": best_pred
        })
    return records


def bidirectional_selection(df, target, predictors):
    """Bidirectional (stepwise) AIC selection."""
    selected = []
    remaining = predictors.copy()
    records = []
    improved = True

    while improved:
        improved = False
        y = df[target].values

        # ---- Forward step ----
        best_feat, best_aic, best_model, best_pred = None, np.inf, None, None
        for feat in remaining:
            candidate = selected + [feat]
            model, aic, y_pred = fit_model(df, target, candidate)
            if aic < best_aic:
                best_feat, best_aic, best_model, best_pred = feat, aic, model, y_pred

        if best_feat and (not records or best_aic < records[-1]["aic"]):
            selected.append(best_feat)
            remaining.remove(best_feat)
            records.append({
                "step": len(records)+1,
                "added_feature": best_feat,
                "removed_feature": None,
                "selected_features": selected.copy(),
                "aic": best_aic,
                "model": best_model,
                "y_pred": best_pred
            })
            improved = True

        # ---- Backward step ----
        worst_feat, best_aic, best_model, best_pred = None, np.inf, None, None
        for feat in selected:
            candidate = [f for f in selected if f != feat]
            model, aic, y_pred = fit_model(df, target, candidate)
            if aic < best_aic:
                worst_feat, best_aic, best_model, best_pred = feat, aic, model, y_pred

        if worst_feat and best_aic < records[-1]["aic"]:
            selected.remove(worst_feat)
            records.append({
                "step": len(records)+1,
                "added_feature": None,
                "removed_feature": worst_feat,
                "selected_features": selected.copy(),
                "aic": best_aic,
                "model": best_model,
                "y_pred": best_pred
            })
            improved = True

    return records


# ------------------ Streamlit UI ------------------
st.set_page_config(layout="wide", page_title="Stepwise AIC Feature Selection")
st.title("AIC-based Feature Selection")

uploaded_file = st.file_uploader("Upload Excel dataset (.xlsx)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully.")
    st.write("### Dataset Preview", df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        target_col = st.selectbox("Select target variable", options=numeric_cols, index=len(numeric_cols)-1)
        predictors = st.multiselect("Select predictor variables", [c for c in numeric_cols if c != target_col],
                                    default=[c for c in numeric_cols if c != target_col])
        method = st.selectbox("Selection method", ["Forward", "Backward", "Bidirectional"])
        run_btn = st.button("Run Selection")

        if run_btn and predictors:
            df_clean = df[[target_col] + predictors].dropna()
            if method == "Forward":
                records = forward_selection(df_clean, target_col, predictors)
            elif method == "Backward":
                records = backward_elimination(df_clean, target_col, predictors)
            else:
                records = bidirectional_selection(df_clean, target_col, predictors)

            if not records:
                st.error("No selection steps performed.")
            else:
                # Results table
                table = pd.DataFrame([{
                    "Step": r["step"],
                    "Added": r["added_feature"],
                    "Removed": r["removed_feature"],
                    "Selected": ", ".join(r["selected_features"]),
                    "AIC": r["aic"]
                } for r in records])
                st.write("### Stepwise Results")
                st.dataframe(table)

                # Plot AIC
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(table["Step"], table["AIC"], marker="o")
                for i, row in table.iterrows():
                    label = row["Added"] if pd.notna(row["Added"]) else f"-{row['Removed']}"
                    ax.annotate(label, (row["Step"], row["AIC"]), xytext=(0, 6),
                                textcoords="offset points", ha="center", fontsize=8)
                ax.set_xlabel("Step")
                ax.set_ylabel("AIC")
                ax.set_title(f"AIC vs Steps ({method})")
                ax.grid(True)
                st.pyplot(fig)

                # Best step
                best_idx = int(np.argmin(table["AIC"]))
                best_row = records[best_idx]
                st.write("### Best Subset")
                st.write(f"Step {best_row['step']} | Features: {', '.join(best_row['selected_features'])} | AIC = {best_row['aic']:.3f}")

else:
    st.info("Upload an Excel file to get started.")
