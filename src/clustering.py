"""
Clustering Module for Urban Traffic Flow Analysis

This module performs K-Means clustering to group junctions based on:
- Traffic volume patterns
- Peak hour characteristics
- Temporal congestion patterns

Clusters represent different congestion levels (Low, Medium, High)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, Tuple, List
import joblib


class TrafficClusterAnalyzer:
    """
    Performs clustering analysis on traffic junctions.
    
    Groups junctions into clusters based on their traffic patterns
    to identify different congestion levels and behaviors.
    """
    
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Initialize the clustering analyzer.
        
        Args:
            n_clusters: Number of clusters (default 3 for Low/Medium/High congestion)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
        self.scaled_features = None
        self.cluster_labels = None
        self.cluster_profiles = None
        
    def select_features_for_clustering(self, junction_features: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare features for clustering.
        
        Focuses on hourly patterns and overall statistics that best capture
        congestion characteristics.
        
        Args:
            junction_features: DataFrame with junction-level features
            
        Returns:
            DataFrame with selected features for clustering
        """
        print("\nSelecting features for clustering...")
        
        # Select hourly average features (captures daily pattern)
        hourly_cols = [col for col in junction_features.columns if 'Hour_' in col and '_Avg' in col]
        
        # Select key statistical features
        stat_cols = [
            'Vehicles_mean',      # Overall traffic level
            'Vehicles_std',       # Traffic variability
            'Vehicles_max',       # Maximum congestion
            'PeakHourTraffic',    # Peak hour intensity
            'Weekend_Weekday_Ratio'  # Weekend vs weekday pattern
        ]
        
        # Combine selected features
        selected_cols = hourly_cols + [col for col in stat_cols if col in junction_features.columns]
        features = junction_features[selected_cols].copy()
        
        print(f"Selected {len(selected_cols)} features for clustering")
        print(f"Features include: {len(hourly_cols)} hourly patterns + {len(stat_cols)} statistical metrics")
        
        return features
    
    def find_optimal_clusters(self, features: pd.DataFrame, max_clusters: int = 10) -> int:
        """
        Use Elbow Method and Silhouette Score to find optimal number of clusters.
        
        Tests different cluster numbers and evaluates quality metrics:
        - Inertia (within-cluster sum of squares): Lower is better
        - Silhouette Score: Higher is better (range -1 to 1)
        
        Args:
            features: Features to cluster
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Recommended number of clusters
        """
        print("\nFinding optimal number of clusters...")
        
        # Standardize features before clustering
        scaled_features = self.scaler.fit_transform(features)
        
        inertias = []
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters + 1, len(features)))
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(scaled_features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_features, labels))
            
        # Find elbow point (where inertia reduction diminishes)
        # Recommend cluster with best silhouette score in reasonable range
        best_k = cluster_range[np.argmax(silhouette_scores)]
        
        print(f"\nCluster Evaluation Results:")
        for i, k in enumerate(cluster_range):
            print(f"  K={k}: Inertia={inertias[i]:.2f}, Silhouette={silhouette_scores[i]:.3f}")
        
        print(f"\nRecommended clusters: {best_k} (highest silhouette score)")
        
        return best_k
    
    def perform_clustering(self, junction_features: pd.DataFrame, 
                          auto_select_k: bool = False) -> pd.DataFrame:
        """
        Perform K-Means clustering on junction features.
        
        Args:
            junction_features: DataFrame with junction-level features
            auto_select_k: If True, automatically find optimal number of clusters
            
        Returns:
            DataFrame with cluster assignments for each junction
        """
        print("\n=== Starting Clustering Analysis ===")
        
        # Select relevant features
        features = self.select_features_for_clustering(junction_features)
        
        # Optionally find optimal number of clusters
        if auto_select_k:
            optimal_k = self.find_optimal_clusters(features)
            print(f"\nUsing {optimal_k} clusters (auto-selected)")
            self.n_clusters = optimal_k
        else:
            print(f"\nUsing {self.n_clusters} clusters (pre-defined)")
        
        # Standardize features (important for K-Means as it uses distance metrics)
        # This ensures all features contribute equally regardless of their scale
        print("\nStandardizing features...")
        self.scaled_features = self.scaler.fit_transform(features)
        
        # Perform K-Means clustering
        print(f"Performing K-Means clustering with {self.n_clusters} clusters...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=20,  # Run algorithm 20 times with different initializations
            max_iter=300  # Maximum iterations for convergence
        )
        
        self.cluster_labels = self.kmeans.fit_predict(self.scaled_features)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Junction': junction_features.index,
            'Cluster': self.cluster_labels
        })
        
        # Calculate clustering quality metrics
        silhouette = silhouette_score(self.scaled_features, self.cluster_labels)
        davies_bouldin = davies_bouldin_score(self.scaled_features, self.cluster_labels)
        
        print(f"\n=== Clustering Complete ===")
        print(f"Silhouette Score: {silhouette:.3f} (higher is better, range: -1 to 1)")
        print(f"Davies-Bouldin Index: {davies_bouldin:.3f} (lower is better)")
        print(f"\nJunctions per cluster:")
        print(results['Cluster'].value_counts().sort_index())
        
        return results
    
    def create_cluster_profiles(self, junction_features: pd.DataFrame, 
                               cluster_assignments: pd.DataFrame,
                               timeseries_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create detailed profiles for each cluster.
        
        Profiles include:
        - Average traffic statistics
        - Peak hours
        - Traffic patterns
        - Congestion level classification
        
        Args:
            junction_features: Original junction features
            cluster_assignments: Cluster labels for each junction
            timeseries_data: Optional timeseries data for detailed analysis
            
        Returns:
            DataFrame with cluster profiles
        """
        print("\n=== Creating Cluster Profiles ===")
        
        # Merge cluster assignments with features
        data = junction_features.copy()
        data['Cluster'] = cluster_assignments.set_index('Junction')['Cluster']
        
        profiles = []
        
        for cluster_id in sorted(data['Cluster'].unique()):
            cluster_data = data[data['Cluster'] == cluster_id]
            
            # Calculate average hourly pattern for this cluster
            hourly_cols = [col for col in data.columns if 'Hour_' in col and '_Avg' in col]
            hourly_pattern = cluster_data[hourly_cols].mean()
            
            # Find peak hour for this cluster (hour with maximum average traffic)
            peak_hour = hourly_pattern.values.argmax()
            peak_traffic = hourly_pattern.values.max()
            
            # Calculate overall statistics
            profile = {
                'Cluster': cluster_id,
                'NumJunctions': len(cluster_data),
                'AvgTraffic': cluster_data['Vehicles_mean'].mean(),
                'AvgStdDev': cluster_data['Vehicles_std'].mean(),
                'AvgMaxTraffic': cluster_data['Vehicles_max'].mean(),
                'PeakHour': peak_hour,
                'PeakHourAvgTraffic': peak_traffic,
                'AvgWeekendWeekdayRatio': cluster_data.get('Weekend_Weekday_Ratio', pd.Series([0])).mean()
            }
            
            profiles.append(profile)
        
        self.cluster_profiles = pd.DataFrame(profiles)
        
        # Classify congestion levels based on average traffic
        # Sort by average traffic and assign labels
        self.cluster_profiles = self.cluster_profiles.sort_values('AvgTraffic')
        
        if self.n_clusters == 3:
            congestion_labels = ['Low Congestion', 'Medium Congestion', 'High Congestion']
        elif self.n_clusters == 4:
            congestion_labels = ['Low Congestion', 'Medium-Low Congestion', 
                               'Medium-High Congestion', 'High Congestion']
        else:
            congestion_labels = [f'Level {i+1}' for i in range(self.n_clusters)]
        
        self.cluster_profiles['CongestionLevel'] = congestion_labels[:len(self.cluster_profiles)]
        
        # Re-sort by original cluster ID for display
        self.cluster_profiles = self.cluster_profiles.sort_values('Cluster').reset_index(drop=True)
        
        print("\nCluster Profiles:")
        print("=" * 80)
        for _, profile in self.cluster_profiles.iterrows():
            print(f"\nCluster {profile['Cluster']} - {profile['CongestionLevel']}")
            print(f"  Number of Junctions: {profile['NumJunctions']}")
            print(f"  Average Traffic: {profile['AvgTraffic']:.1f} vehicles/hour")
            print(f"  Peak Hour: {int(profile['PeakHour'])}:00 ({profile['PeakHourAvgTraffic']:.1f} vehicles)")
            print(f"  Traffic Variability (Std Dev): {profile['AvgStdDev']:.1f}")
        
        return self.cluster_profiles
    
    def save_model(self, filepath: str):
        """
        Save the trained clustering model and scaler.
        
        Args:
            filepath: Path to save the model (without extension)
        """
        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'n_clusters': self.n_clusters,
            'cluster_profiles': self.cluster_profiles
        }
        joblib.dump(model_data, f"{filepath}.pkl")
        print(f"\nModel saved to {filepath}.pkl")
    
    def load_model(self, filepath: str):
        """
        Load a previously trained clustering model.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        self.kmeans = model_data['kmeans']
        self.scaler = model_data['scaler']
        self.n_clusters = model_data['n_clusters']
        self.cluster_profiles = model_data['cluster_profiles']
        print(f"\nModel loaded from {filepath}")
    
    def predict_cluster(self, junction_features: pd.DataFrame) -> int:
        """
        Predict cluster for new junction data.
        
        Args:
            junction_features: Features for a single junction
            
        Returns:
            Cluster ID
        """
        if self.kmeans is None:
            raise ValueError("Model not trained. Call perform_clustering first.")
        
        features = self.select_features_for_clustering(junction_features)
        scaled = self.scaler.transform(features)
        return self.kmeans.predict(scaled)[0]


def clustering_pipeline(junction_features: pd.DataFrame, 
                        timeseries_data: pd.DataFrame = None,
                        n_clusters: int = 3,
                        auto_select_k: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete clustering pipeline - convenience function.
    
    Args:
        junction_features: Junction-level features
        timeseries_data: Optional timeseries data
        n_clusters: Number of clusters
        auto_select_k: Automatically find optimal number of clusters
        
    Returns:
        Tuple of (cluster_assignments, cluster_profiles)
    """
    analyzer = TrafficClusterAnalyzer(n_clusters=n_clusters)
    cluster_assignments = analyzer.perform_clustering(junction_features, auto_select_k)
    cluster_profiles = analyzer.create_cluster_profiles(
        junction_features, 
        cluster_assignments, 
        timeseries_data
    )
    
    return cluster_assignments, cluster_profiles


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import preprocess_pipeline
    
    data_path = "../data/traffic.csv"
    timeseries, junction_features = preprocess_pipeline(data_path)
    
    # Perform clustering
    cluster_assignments, cluster_profiles = clustering_pipeline(
        junction_features, 
        timeseries,
        n_clusters=3,
        auto_select_k=False
    )
    
    print("\n=== Clustering Analysis Complete ===")
