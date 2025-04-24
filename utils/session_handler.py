import streamlit as st
import pandas as pd

def initialize_session():
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None

def save_uploaded_data(df: pd.DataFrame):
    st.session_state.uploaded_data = df

def get_uploaded_data():
    return st.session_state.uploaded_data
