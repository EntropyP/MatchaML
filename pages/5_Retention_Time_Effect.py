import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from utils import session_handler

st.title("⏱️ Retention Time Effect Analysis")

df = session_handler.get_uploaded_data()
if df is None:
    st.warning("Please upload and prepare data first.")
    st.stop()

st.subheader("Define Inputs")
columns = df.columns.tolist()
x_cols = st.multiselect("Select independent variable(s) (X)", columns)
y_col = st.selectbox("Select target variable (y)", columns)

if not x_cols or not y_col:
    st.stop()

col_start, col_end, col_step = st.columns(3)
with col_start:
    lag_start = st.number_input("Start lag", min_value=1, max_value=100, value=1)
with col_end:
    lag_end = st.number_input("End lag", min_value=lag_start, max_value=100, value=10)
with col_step:
    lag_step = st.number_input("Lag step", min_value=1, max_value=lag_end - lag_start + 1, value=1)

results = []

if st.button("Run Analysis"):
    for lag in range(int(lag_start), int(lag_end) + 1, int(lag_step)):
        df_shifted = df[x_cols + [y_col]].copy()
        for col in x_cols:
            df_shifted[col] = df_shifted[col].shift(lag)
        df_shifted.dropna(inplace=True)

        X = df_shifted[x_cols]
        y = df_shifted[y_col]

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        r2 = r2_score(y, y_pred)

        results.append({
            "Lag": lag,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2
        })

    results_df = pd.DataFrame(results)

    st.subheader("Evaluation Metrics vs Lag")
    st.line_chart(results_df.set_index("Lag")[['MAE', 'RMSE', 'R²']])

    best_lag = results_df.sort_values("R²", ascending=False).iloc[0]
    st.success(f"Best lag based on R²: Lag = {int(best_lag['Lag'])} with R² = {best_lag['R²']:.4f}")

    st.subheader("Detailed Results")
    st.dataframe(results_df)

    # Option to export
    st.download_button("Download Results as CSV", results_df.to_csv(index=False), file_name="retention_time_effect.csv")