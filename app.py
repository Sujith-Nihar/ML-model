# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Depression Detection Dashboard", layout="wide")

# ===================== Title & Summary =====================
st.title("Depression Detection Dashboard")

with st.expander("📚 Project Summary", expanded=True):
    st.markdown("""
    Classify social media posts as **Depressed** or **Not Depressed** using:
    - Visual features: **Brightness, Saturation, Hue**
    - Emotion scores: **Happiness, Sadness, Fear, Disgust, Anger, Surprise**
    - **PHQ-9** binary indicators (0/1)

    Models: **Logistic Regression** (with scaling) and **Random Forest** (class_weight='balanced').
    This app safely coerces features to numeric to avoid dtype errors.
    """)

# ===================== Sidebar Controls =====================
st.sidebar.header("🔧 Controls")
uploaded = st.sidebar.file_uploader("Upload merged dataset (.xlsx)", type=["xlsx"])
default_path = st.sidebar.text_input(
    "…or provide a local path (optional)", 
    "/Users/sujiththota/Downloads/Python/Research/ML_DATA/new_data/merged_dataset.xlsx"
)
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Logistic Regression"])
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

# Optional: decision threshold for converting probabilities to labels (for LogReg)
threshold = st.sidebar.slider("Decision threshold (for probability→label)", 0.05, 0.95, 0.50, 0.01)

# ===================== Load Data =====================
def load_df(uploaded_file, fallback_path):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    try:
        return pd.read_excel(fallback_path)
    except Exception:
        return None

df = load_df(uploaded, default_path)
if df is None:
    st.warning("Please upload an Excel file or provide a valid local path.")
    st.stop()

st.subheader("🗂️ Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# ===================== Columns Setup =====================
# Columns to drop if present (text/meta fields you don't want in X)
drop_columns = [
    'Media Name', 'Profile Name', 'Simple Description', 'Embedded Text',
    'Caption', 'Important Note', 'Diagnosed Date', 'Media Type'
]
drop_columns = [c for c in drop_columns if c in df.columns]

# Expected numeric feature set (order is just a hint, not required)
expected_numeric = [
    'Brightness Value','Saturation Value','Hue value',
    'Happiness','Sadness','Fear','Disgust','Anger','Surprise',
    'Loss of Interest Binary','Feeling depressed Binary','Sleeping Disorder Binary',
    'Lack of Energy Binary','Eating Disorder Binary','Low Self-Esteem Binary',
    'Concentration difficulty Binary','Psychomotor changes Binary','Self harm risk Binary'
]

# ===================== Target (y) =====================
if 'Depressed post' not in df.columns:
    st.error("Column **'Depressed post'** is missing. Please include it with values 'Yes'/'No'.")
    st.stop()

y = (
    df['Depressed post']
    .astype(str).str.strip().str.lower()
    .map({'yes': 1, 'no': 0})
)

if y.isna().any():
    st.warning("Some rows in **'Depressed post'** were not Yes/No and will be dropped.")
    valid_mask = y.notna()
    df = df.loc[valid_mask].copy()
    y = y.loc[valid_mask]

y = y.astype(int)

# ===================== Features (X) =====================
# Start from all columns except target + unwanted text/meta columns
X = df.drop(columns=drop_columns + ['Depressed post'], errors='ignore')

# If PHQ-9 columns are strings like Yes/No accidentally, normalize them first
phq9_cols = [
    'Loss of Interest Binary','Feeling depressed Binary','Sleeping Disorder Binary',
    'Lack of Energy Binary','Eating Disorder Binary','Low Self-Esteem Binary',
    'Concentration difficulty Binary','Psychomotor changes Binary','Self harm risk Binary'
]
for col in phq9_cols:
    if col in X.columns and X[col].dtype == object:
        X[col] = (
            X[col].astype(str).str.strip().str.lower()
                 .map({'yes': 1, 'no': 0})
        )

# Force ALL feature columns to numeric (junk → NaN)
X = X.apply(pd.to_numeric, errors='coerce')

# Optional: place expected columns first if they exist, then append the rest
ordered_present = [c for c in expected_numeric if c in X.columns]
rest = [c for c in X.columns if c not in ordered_present]
X = X[ordered_present + rest]

# Impute missing values with column means (numeric_only avoids dtype issues)
X = X.fillna(X.mean(numeric_only=True))

# If after coercion/imputation you have no columns, stop
if X.shape[1] == 0:
    st.error("No numeric features available after cleaning. Please verify your data.")
    st.stop()

# ===================== Train/Test Split =====================
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
except ValueError:
    # Stratify fails if only one class
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    st.warning("Stratified split not possible (only one class present). Using random split.")

# ===================== Train Models =====================
if model_choice == "Random Forest":
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Probabilities for thresholding & ROC
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    # Convert probs to labels using chosen threshold
    if y_proba is not None:
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

else:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight='balanced',
        random_state=random_state,
        max_iter=2000
    )
    model.fit(X_train_s, y_train)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

# ===================== Metrics =====================
st.subheader("📝 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
col4.metric("F1", f"{f1_score(y_test, y_pred, zero_division=0):.3f}")
if (y_proba is not None) and (len(np.unique(y_test)) > 1):
    try:
        col5.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
    except Exception:
        col5.metric("ROC-AUC", "N/A")

st.markdown("**Classification Report**")
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

# ===================== Visualizations =====================
st.subheader("📈 Visualizations")

# Confusion Matrix
st.markdown("**Confusion Matrix**")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm, clear_figure=True)

# ROC Curve (if probs available)
if y_proba is not None and len(np.unique(y_test)) > 1:
    fpr, tpr, thr = roc_curve(y_test, y_proba)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC ≈ {roc_auc_score(y_test, y_proba):.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc, clear_figure=True)

# Feature Importance / Coefficients
if model_choice == "Random Forest" and hasattr(model, "feature_importances_"):
    st.markdown("**Random Forest – Feature Importances**")
    fi = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}) \
           .sort_values("Importance", ascending=False)
    fig_fi, ax_fi = plt.subplots(figsize=(7, min(0.35 * len(fi), 10)))
    sns.barplot(data=fi, x="Importance", y="Feature", ax=ax_fi)
    st.pyplot(fig_fi, clear_figure=True)

if model_choice == "Logistic Regression":
    st.markdown("**Logistic Regression – Coefficients (standardized)**")
    # Coeffs correspond to standardized features (X_test_s)
    coefs = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_[0]}) \
              .sort_values("Coefficient", ascending=False)
    fig_cf, ax_cf = plt.subplots(figsize=(7, min(0.35 * len(coefs), 10)))
    sns.barplot(data=coefs, x="Coefficient", y="Feature", ax=ax_cf)
    st.pyplot(fig_cf, clear_figure=True)

# ===================== Predictions & Download =====================
st.subheader("📥 Predictions")
pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
if y_proba is not None:
    pred_df["Pred_Prob"] = y_proba

st.dataframe(pred_df.head(50), use_container_width=True)

to_save = pd.concat([X_test.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    to_save.to_excel(writer, index=False, sheet_name="predictions")

st.download_button(
    label="Download predictions (.xlsx)",
    data=buffer.getvalue(),
    file_name="predictions.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.markdown("---")
st.caption("© 2025 Depression Detection Project | Built with ❤️ using Streamlit")
