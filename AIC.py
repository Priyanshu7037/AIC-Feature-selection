# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from io import BytesIO

# ------------------ Utilities ------------------
def calculate_aic(y_true, y_pred, k, eps=1e-12):
    """Full Gaussian log-likelihood AIC"""
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    rss = max(rss, eps)  # avoid log(0)
    logL = -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1)
    aic = 2 * k - 2 * logL
    return aic

def fit_model(df, target, features):
    """Fit linear regression, return model, aic."""
    y = df[target].values
    X = df[features].values if features else np.zeros((len(y), 0))
    model = LinearRegression().fit(X, y) if features else None
    y_pred = model.predict(X) if features else np.mean(y) * np.ones_like(y)
    k = len(features) + 1 if features else 1
    aic = calculate_aic(y, y_pred, k)
    return model, aic, y_pred

# Stepwise selection functions
def forward_selection(df, target, predictors):
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
    selected = []
    remaining = predictors.copy()
    records = []
    improved = True
    while improved:
        improved = False
        # Forward step
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
        # Backward step
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

# ------------------ Demo Dataset ------------------
def generate_demo_dataset(n_samples=100, n_features=20, n_true=5, random_state=42):
    """
    Generate synthetic dataset for demo:
    - n_true variables actually affect the target.
    - Rest are pure noise.
    """
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    
    # Only first n_true features are true predictors
    coefs = np.zeros(n_features)
    coefs[:n_true] = np.random.randn(n_true) * 2  # stronger signal for true variables
    
    y = X @ coefs + 0.5 * np.random.randn(n_samples)  # add noise
    columns = [f"X{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["Target"] = y
    return df


def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

# ------------------ Streamlit UI ------------------
st.set_page_config(layout="wide", page_title="Stepwise AIC Feature Selection")
st.title("AIC-based Feature Selection")

uploaded_file = st.file_uploader("Upload Excel dataset (.xlsx)", type=["xlsx"])
use_demo = False

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully.")
elif st.checkbox("Use built-in demo dataset"):
    df = generate_demo_dataset()
    st.info("Using built-in demo dataset with 20 input variables.")
    use_demo = True
else:
    st.info("Upload an Excel file or select the demo dataset to get started.")
    df = None

if df is not None:
    st.write("### Dataset Preview", df.head())

    if use_demo:
        st.download_button(
            label="Download Demo Dataset as Excel",
            data=to_excel(df),
            file_name="demo_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

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
                table = pd.DataFrame([{
                    "Step": r["step"],
                    "Added": r["added_feature"],
                    "Removed": r["removed_feature"],
                    "Selected": ", ".join(r["selected_features"]),
                    "AIC": r["aic"]
                } for r in records])
                st.write("### Stepwise Results")
                st.dataframe(table)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(table["Step"], table["AIC"], marker="o")
                for i, row in table.iterrows():
                    label = row["Added"] if pd.notna(row["Added"]) else f"-{row['Removed']}"
                    ax.annotate(label, (row["Step"], row["AIC"]), xytext=(0, 6),
                                textcoords="offset points", ha="center", fontsize=8)
                    
                ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                ax.set_xlabel("Step")
                ax.set_ylabel("AIC")
                ax.set_title(f"AIC vs Steps ({method})")
                ax.grid(True)
                st.pyplot(fig)

                best_idx = int(np.argmin(table["AIC"]))
                best_row = records[best_idx]
                st.write("### Best Subset")
                st.write(f"Step {best_row['step']} | Features: {', '.join(best_row['selected_features'])} | AIC = {best_row['aic']:.3f}")


