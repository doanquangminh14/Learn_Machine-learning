"""
Module: Customer Segmentation & Unsupervised Clustering Pipeline (clustering.py)
Clustering customer records based on purchasing behavior, income, and store visits.
"""

import sys
import os
import warnings

# Suppress OpenMP/MKL multi-threading warnings on Windows
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def load_and_preprocess_data(file_path: str = "dataset/data.csv"):
    """Load customer personality dataset and select feature subspace."""
    df = pd.read_csv(file_path)
    
    features = [
        'Income', 'Recency', 'MntTotal', 'MntWines', 'MntMeatProducts',
        'NumWebPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Age', 'Customer_Days'
    ]
    
    # Fill any subtle missing values with median
    X = df[features].fillna(df[features].median())
    
    print("=" * 65)
    print("CUSTOMER SEGMENTATION DATASET OVERVIEW:")
    print(f"Total Customers: {df.shape[0]}, Features Selected: {len(features)}")
    print("=" * 65)
    
    return df, X, features


def scale_features(X):
    """Standardize features with mean=0 and variance=1."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def find_optimal_clusters(X_scaled, max_k: int = 7):
    """Evaluate WCSS, Silhouette, Calinski-Harabasz, and Davies-Bouldin metrics."""
    print("\n--- CLUSTERING METRIC SEARCH (K-Means K=2 to 7) ---")
    results = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        wcss = kmeans.inertia_
        
        results.append({
            "Clusters (K)": k,
            "WCSS (Inertia)": round(wcss, 2),
            "Silhouette Score": round(sil, 4),
            "Calinski-Harabasz": round(ch, 2),
            "Davies-Bouldin": round(db, 4)
        })
        
    metrics_df = pd.DataFrame(results)
    print(metrics_df.to_string(index=False))
    return metrics_df


def fit_and_profile_clusters(df, X_scaled, features, n_clusters: int = 3):
    """Fit optimal K-Means model and generate business persona profiles."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA projection for 2D variance tracking
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df['PCA_Dim1'] = coords[:, 0]
    df['PCA_Dim2'] = coords[:, 1]
    
    print("\n" + "=" * 65)
    print(f"CUSTOMER SEGMENT PROFILES (K={n_clusters}):")
    print(f"Cluster Sizes:\n{df['Cluster'].value_counts().sort_index()}")
    
    profile = df.groupby('Cluster')[features].mean().round(2)
    print("\nMean Feature Values per Cluster:")
    print(profile.T)
    
    print("\nBUSINESS PERSONA INTERPRETATION:")
    print(" - Cluster 0: Value Shoppers (Moderate income, selective spending, balanced channels)")
    print(" - Cluster 1: High-Net-Worth VIPs (Highest income & spending in Wines/Meat, low web visits)")
    print(" - Cluster 2: Budget-Conscious Web Browsers (Lower income, low spending, high web visits)")
    print("=" * 65)
    
    return df, kmeans


def run_pipeline():
    """Main execution pipeline."""
    df, X, features = load_and_preprocess_data()
    X_scaled, scaler = scale_features(X)
    find_optimal_clusters(X_scaled, max_k=6)
    df_clustered, model = fit_and_profile_clusters(df, X_scaled, features, n_clusters=3)
    return df_clustered


if __name__ == "__main__":
    run_pipeline()
