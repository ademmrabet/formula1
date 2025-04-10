#Descriptive modeling (unsupervised learning) for F1 Constructor Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def perform_clustering(df, n_clusters=4):
    """
    Perform K-means clustering on constructor data

    Args:
        df: processed Dataframe
        n_clusters: Number of clusters to create

    Returns:
        DataFrame with cluster assignments
    """

    # Select feature for clustering
    features = ['Position', 'Points', 'NormalizedPoints']

    # Prepare data for clustering
    cluster_data = df[features].copy()

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    # Determine optimal number of clusters using the elbow method
    print("Determining optimal number of clusters...")
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    # Plot elbow method results
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, linestyle='-', marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.grid(True, alpha=0.3)
    plt.savefig('assets/elbow_method.png')
    plt.show()

    # Perform K-means clustering with the selected number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)

    # Add cluster labels to the DataFrame
    result_df = df.copy()
    result_df['Cluster'] = cluster_labels

    #Analyze cluster characteristics
    cluster_analysis = result_df.groupby('Cluster').agg({
        'Position': ['mean', 'min', 'max'],
        'Points': ['mean', 'min', 'max'],
        'NormalizedPoints': ['mean', 'min', 'max'],
        'Constructor': 'count'
    }).round(2)

    print("\nCluster Analysis:")
    print(cluster_analysis)

    # Map clusters to meaningful labels based on performance
    cluster_means = result_df.groupby('Cluster')['Position'].mean().sort_values()
    performance_labels = ['Elite Teams', 'Strong Contenders', 'Midfield Teams', "Backmarkers" ]

    # Create mapping from cluster number to performance label
    cluster_to_performance = {cluster: performance_labels[i] for i, cluster in enumerate(cluster_means.index)}
    result_df['PerformanceGroup'] = result_df['Cluster'].map(cluster_to_performance)

    return result_df

def visualize_clusters(clustered_df, original_df):
    #1. Visualize clusters in 2D using PCA
    features = ['Position', 'Points', 'NormalizedPoints']

    #Prepare Data for PCA
    X = clustered_df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    #create a DataFrame with PCA results
    pca_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': clustered_df['Cluster'],
        'PerformanceGroup': clustered_df['PerformanceGroup'],
        'Constructor': clustered_df['Constructor'],
        'Season': clustered_df['Season']
    })

    #Plot PCA results
    plt.figure(figsize=(12, 8))

    # Use Different markers and colors for each cluster
    performance_groups = pca_df['PerformanceGroup'].unique()
    colors = ['gold', 'royalblue', 'forestgreen', 'crimson']


    for i, group in enumerate(performance_groups):
        group_data = pca_df[pca_df['PerformanceGroup'] == group]
        plt.scatter(group_data['PCA1'], group_data['PCA2'], label = group, alpha=0.7, s=50, color=colors[i])

    plt.title('Constructor Performance Clusters', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('assets/cluster_visualization.png', dpi=300)
    plt.close()

    #2. Visualize clusters by era
    plt.figure(figsize=(14, 8))

    # Create a heatmap showing cluster distribution across eras
    era_cluster = pd.crosstab(clustered_df['Era'], clustered_df['PerformanceGroup'],
                              normalize='index') * 100
    sns.heatmap(era_cluster, annot=True, cmap='YlGnBu', fmt='.1f', cbar_kws={'label' : 'Percentage (%)'})
    plt.title('Distribution of performance Groups across Eras', fontsize=16)
    plt.ylabel('Era', fontsize=12)
    plt.xlabel('Performance Group', fontsize=12)
    plt.tight_layout()
    plt.savefig('assets/cluster_distribution_by_era.png', dpi=300)
    plt.close()


    #3. Champions analysis whithin clusters

    championship_distribution = clustered_df[clustered_df['IsChampion']==1].groupby('PerformanceGroup').size()
    championship_percentage = championship_distribution / championship_distribution.sum() * 100

    plt.figure(figsize=(10, 6))
    championship_percentage.plot(kind='bar', color=colors)
    plt.title('Championship Distribution by Performance Group', fontsize=16)
    plt.xlabel('Performance Group', fontsize=12)
    plt.ylabel('Percentage of Championships (%)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()


    #Add percentage labels on top of each bar
    for i, percentage in enumerate(championship_percentage):
        plt.text(i, percentage +1, f'{percentage: .1f}%', ha='center', va='bottom', fontsize=11)

    plt.savefig('assets/championship_distribution.png', dpi=300)
    plt.close()

    print("Cluster Visualizations saved to 'assets/' directory")
