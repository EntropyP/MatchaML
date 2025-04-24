import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils import session_handler

st.title("📊 Data Visualization")

# Load data
df = session_handler.get_uploaded_data()
if df is None:
    st.warning("Please upload and prepare data first in earlier modules.")
    st.stop()

columns = df.columns.tolist()

st.subheader("1. Select Target Variable (y)")
y_col = st.selectbox("Choose y variable", options=columns)

st.markdown("---")
st.subheader("2. Box Plot for All Variables")

# Prepare data for boxplot
numeric_cols = df.select_dtypes(include='number').columns.tolist()
if y_col not in numeric_cols:
    st.warning("Selected y is not numeric. Boxplot may be incomplete.")

melted_df = df[numeric_cols].melt(var_name="Variable", value_name="Value")
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.boxplot(x="Variable", y="Value", data=melted_df, ax=ax1)
ax1.tick_params(axis='x', rotation=90)
st.pyplot(fig1)

st.markdown("---")
st.subheader("3. Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
st.pyplot(fig2)

st.markdown("---")
st.subheader("4. Scatter Plot")
x_cols = st.multiselect("Select features for X", options=[col for col in numeric_cols if col != y_col])

if x_cols:
    for x_col in x_cols:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
        ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
        st.pyplot(fig)