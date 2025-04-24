import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from utils import session_handler

st.title("🔍 Clustering Models")

df = session_handler.get_uploaded_data()
if df is None:
    st.warning("Please upload and prepare data first.")
    st.stop()

st.subheader("Select Features for Clustering")
features = st.multiselect("Select numerical features to cluster", df.select_dtypes(include=np.number).columns.tolist())
if not features:
    st.stop()

X = df[features]

st.subheader("Select Clustering Model")
model_type = st.selectbox("Clustering algorithm", ["KMeans", "DBSCAN", "Agglomerative Clustering"])

# Model specific parameters
if model_type == "KMeans":
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    model = KMeans(n_clusters=n_clusters, random_state=42)
elif model_type == "DBSCAN":
    eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
    min_samples = st.slider("Minimum Samples", 1, 10, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
elif model_type == "Agglomerative Clustering":
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    model = AgglomerativeClustering(n_clusters=n_clusters)

# User-defined cluster column name
st.subheader("Customize Cluster Output")
cluster_column_name = st.text_input("Cluster column name", value="Cluster")

if st.button("Run Clustering"):
    labels = model.fit_predict(X)
    df[cluster_column_name] = labels
    st.success(f"Clustering completed. Results added to dataset as '{cluster_column_name}'.")

    # Evaluation (only for KMeans or Agglomerative where more than 1 cluster)
    if len(set(labels)) > 1 and model_type != "DBSCAN":
        score = silhouette_score(X, labels)
        st.write(f"**Silhouette Score**: {score:.4f}")

    # Dimensionality reduction for plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    cluster_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    cluster_df[cluster_column_name] = df[cluster_column_name]

    fig, ax = plt.subplots()
    sns.scatterplot(data=cluster_df, x="PC1", y="PC2", hue=cluster_column_name, palette="tab10", ax=ax)
    st.pyplot(fig)

    st.dataframe(df.head())

    # Option to export
    st.download_button("Download Clustered Data", df.to_csv(index=False), file_name="clustered_data.csv")
