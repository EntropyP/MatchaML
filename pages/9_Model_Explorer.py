# pages/9_Model_Explorer.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("🔍 Model Explorer")

st.subheader("1. Load Trained Model")

# Try to reuse previously loaded model from session
if 'loaded_model' not in st.session_state:
    st.session_state.loaded_model = None

model_file = st.file_uploader("Upload a trained model (.pkl)", type=["pkl"])

if model_file is not None:
    try:
        model = joblib.load(model_file)
        st.session_state.loaded_model = model
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

model = st.session_state.loaded_model

if model:
    st.subheader("2. Adjust X Input Features")

    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        st.warning("Feature names not found in model. Please enter them manually.")
        num_features = st.number_input("Number of input features", min_value=1, step=1)
        feature_names = [st.text_input(f"Feature {i+1} name") for i in range(int(num_features))]

    input_data = {}
    st.markdown("Use sliders to define input values:")
    for feat in feature_names:
        input_data[feat] = st.slider(f"{feat}", min_value=-100.0, max_value=100.0, value=0.0, step=0.1)

    X_input = pd.DataFrame([input_data])

    st.subheader("3. Predicted Output")
    try:
        y_pred = model.predict(X_input)
        st.success(f"Predicted y: {y_pred[0]:.4f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
