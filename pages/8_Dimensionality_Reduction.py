import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from utils import session_handler

st.title("📉 Dimensionality Reduction")

df = session_handler.get_uploaded_data()
if df is None:
    st.warning("Please upload and prepare data first.")
    st.stop()

st.subheader("Select Features for Dimensionality Reduction")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
selected_features = st.multiselect("Select numeric columns", options=numeric_cols)

if not selected_features:
    st.stop()

X = df[selected_features]

st.subheader("Choose Reduction Technique")
method = st.radio("Reduction Method", ["PCA", "t-SNE"])

max_components = min(len(selected_features), 5)

if method == "t-SNE":
    max_components = min(max_components, 3)  # t-SNE usually supports only 2 or 3

n_components = st.slider("Number of Components", min_value=2, max_value=max_components, value=2)

run = st.button("Run Dimensionality Reduction")

if run:
    try:
        if method == "PCA":
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X)
            st.write("Explained Variance Ratio:", reducer.explained_variance_ratio_)
        elif method == "t-SNE":
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
            X_reduced = reducer.fit_transform(X)

        reduced_df = pd.DataFrame(X_reduced, columns=[f"Component {i+1}" for i in range(n_components)])

        if n_components == 2:
            fig, ax = plt.subplots()
            sns.scatterplot(data=reduced_df, x="Component 1", y="Component 2", ax=ax)
            st.pyplot(fig)
        elif n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced_df.iloc[:, 0], reduced_df.iloc[:, 1], reduced_df.iloc[:, 2], c='blue', s=60)
            st.pyplot(fig)

        st.subheader("Reduced Data Preview")
        st.dataframe(reduced_df.head())

        # Download reduced data
        st.download_button("Download Reduced Data", reduced_df.to_csv(index=False), file_name="reduced_data.csv")

    except Exception as e:
        st.error(f"Dimensionality reduction failed: {e}")