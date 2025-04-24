import streamlit as st
from utils import session_handler

st.set_page_config(page_title="Data Science App", layout="wide")

st.title("📊 Modular Data Science App")
st.markdown("Use the sidebar to navigate between modules.")

# Ensure session state is initialized
session_handler.initialize_session()

