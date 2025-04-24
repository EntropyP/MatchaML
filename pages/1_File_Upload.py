import streamlit as st
import pandas as pd
import io
from utils import session_handler

st.title("📁 File Upload")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        session_handler.save_uploaded_data(df)
        st.success("File uploaded and saved successfully!")

        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error reading file: {e}")

st.markdown("---")
st.markdown("Proceed to the sidebar to start data preparation.")
