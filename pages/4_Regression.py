import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import numpy as np
from utils import session_handler

st.title("📈 Regression Models")

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
y = df[y_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0)
}

st.subheader("Train and Evaluate Models")
model_results = {}
col1, col2, col3 = st.columns(3)

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    model_results[name] = {
        "model": model,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    with [col1, col2, col3][i]:
        st.markdown(f"#### {name}")
        st.metric("MAE", f"{mae:.4f}")
        st.metric("MSE", f"{mse:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("R²", f"{r2:.4f}")

# Show equation for Linear Regression
if "Linear Regression" in model_results:
    lr_model = model_results["Linear Regression"]["model"]
    coeffs = lr_model.coef_
    intercept = lr_model.intercept_
    equation = f"{y_col} = " + " + ".join([f"({coeff:.4f} × {col})" for coeff, col in zip(coeffs, X.columns)]) + f" + ({intercept:.4f})"
    st.subheader("Linear Regression Equation")
    st.code(equation)

st.markdown("---")

# Model export section
st.subheader("Export Trained Model")
selected_model_name = st.selectbox("Select model to export", list(model_results.keys()))
if st.button("Export Selected Model"):
    model = model_results[selected_model_name]["model"]
    filename = f"models/{selected_model_name.replace(' ', '_').lower()}_regressor.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, filename)
    st.success(f"Model saved as {filename}")
    with open(filename, "rb") as f:
        st.download_button("Download Model", f, file_name=filename)
