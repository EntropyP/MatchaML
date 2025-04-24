import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from utils import session_handler

st.title("🧹 Data Preparation")

# Load uploaded data
df = session_handler.get_uploaded_data()
if df is None:
    st.warning("Please upload a dataset first in the File Upload module.")
    st.stop()

st.subheader("Original Data Preview")
st.dataframe(df.head())

st.markdown("---")
st.subheader("Step 1: Handle Missing Values")

missing_option = st.selectbox("Choose missing data strategy", ["None", "Drop rows", "Impute with mean", "Impute with median", "Forward fill"])

if missing_option == "Drop rows":
    df = df.dropna()
elif missing_option == "Impute with mean":
    imputer = SimpleImputer(strategy='mean')
    df[df.select_dtypes(include=np.number).columns] = imputer.fit_transform(df.select_dtypes(include=np.number))
elif missing_option == "Impute with median":
    imputer = SimpleImputer(strategy='median')
    df[df.select_dtypes(include=np.number).columns] = imputer.fit_transform(df.select_dtypes(include=np.number))
elif missing_option == "Forward fill":
    df = df.fillna(method='ffill')

st.success("Missing value handling applied.")

st.markdown("---")
st.subheader("Step 2: Feature Scaling")

scaling_option = st.selectbox("Choose scaling method", ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])

if scaling_option != "None":
    scaler = None
    if scaling_option == "StandardScaler":
        scaler = StandardScaler()
    elif scaling_option == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaling_option == "RobustScaler":
        scaler = RobustScaler()

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    st.success(f"{scaling_option} applied to numeric columns.")

st.markdown("---")
st.subheader("Step 3: Drop Unwanted Columns")
cols_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
if st.button("Drop Selected Columns") and cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    st.success("Selected columns dropped.")

st.markdown("---")
st.subheader("Step 4: Encode Categorical Columns")
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

if categorical_cols:
    selected_cat_cols = st.multiselect("Select categorical columns to one-hot encode", categorical_cols)
    if selected_cat_cols:
        for col in selected_cat_cols:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            col_index = df.columns.get_loc(col)
            df = df.drop(columns=[col])
            for i, dummy_col in enumerate(dummies.columns):
                df.insert(col_index + i, dummy_col, dummies[dummy_col])
        st.success(f"One-hot encoding applied to: {', '.join(selected_cat_cols)}")
else:
    st.info("No categorical columns detected.")

st.markdown("---")
st.subheader("Final Prepared Data")
st.dataframe(df.head())

# Save prepared data
session_handler.save_uploaded_data(df)

st.success("Data preparation complete. You can now proceed to the next module.")
