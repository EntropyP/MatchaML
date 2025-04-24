import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import numpy as np
from utils import session_handler

st.title("🧠 Classification Models")

df = session_handler.get_uploaded_data()
if df is None:
    st.warning("Please upload and prepare data first.")
    st.stop()

st.subheader("Select Features and Target")
columns = df.columns.tolist()
x_cols = st.multiselect("Select features (X)", options=columns)
y_col = st.selectbox("Select target variable (y)", options=columns)

if not x_cols or not y_col:
    st.stop()

# Feature power manipulation
st.subheader("Feature Power Manipulation")
feature_powers = {}
X_transformed = pd.DataFrame()
for col in x_cols:
    power = st.slider(f"Select power for {col}", 1, 5, 1)
    feature_powers[col] = power
    X_transformed[f"{col}^({power})"] = df[col] ** power

X = X_transformed
y_raw = df[y_col]

# Ensure y is categorical
y = None
if not pd.api.types.is_numeric_dtype(y_raw):
    y = pd.factorize(y_raw)[0]
else:
    unique_values = y_raw.nunique()
    if unique_values <= 10:
        y = y_raw.astype(int)
    else:
        quantiles = y_raw.quantile([1/3, 2/3])
        q1, q2 = quantiles.iloc[0], quantiles.iloc[1]
        st.info(f"Target variable is continuous. It has been automatically binned into three categories:\n\n- Low (≤ {q1:.2f})\n- Medium (> {q1:.2f} and ≤ {q2:.2f})\n- High (> {q2:.2f})")
        y = pd.qcut(y_raw, q=3, labels=False)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select K for KNN
st.subheader("KNN Hyperparameter")
k_neighbors = st.slider("Number of Neighbors (K)", min_value=1, max_value=20, value=5)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, verbosity=0),
    f"KNN (k={k_neighbors})": KNeighborsClassifier(n_neighbors=k_neighbors)
}

st.subheader("Train and Evaluate Models")
model_results = {}
col1, col2, col3, col4 = st.columns(4)

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    model_results[name] = {
        "model": model,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": recall,
        "F1": f1,
        "y_pred": y_pred
    }

    with [col1, col2, col3, col4][i]:
        st.markdown(f"#### {name}")
        st.metric("Accuracy", f"{acc:.4f}")
        st.metric("Precision", f"{prec:.4f}")
        st.metric("Recall", f"{recall:.4f}")
        st.metric("F1 Score", f"{f1:.4f}")

st.markdown("---")

# Model export section
st.subheader("Export Trained Model")
selected_model_name = st.selectbox("Select model to export", list(model_results.keys()))
if st.button("Export Selected Model"):
    model = model_results[selected_model_name]["model"]
    filename = f"models/{selected_model_name.replace(' ', '_').lower()}_classifier.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, filename)
    st.success(f"Model saved as {filename}")
    with open(filename, "rb") as f:
        st.download_button("Download Model", f, file_name=filename)