"""
Phase 3: Machine Learning Engine
Urban Traffic Flow Analysis - ML Models

This module implements:
1. Junction Clustering (Unsupervised) - Groups junctions by traffic patterns
2. Trend Forecasting (Supervised) - Predicts traffic for next hour

The clustering engine identifies congestion levels to optimize signal timing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os


class JunctionClustering:
    """
    Unsupervised clustering to group junctions by traffic patterns.
    
    Creates junction profiles and uses K-Means to identify
    Low/Medium/High congestion areas for targeted traffic management.
    """
    
    def __init__(self, data, n_clusters=3):
        """
        Initialize the clustering model.
        
        Args:
            data: Processed traffic DataFrame
            n_clusters: Number of clusters (default: 3)
        """
        self.data = data
        self.n_clusters = n_clusters
        self.junction_profiles = None
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_labels = None
        
    def create_junction_profiles(self):
        """
        Create aggregated profiles for each junction.
        
        Calculates key metrics:
        - Avg_Vehicles: Overall average traffic
        - Max_Vehicles: Peak traffic volume
        - Morning_Peak: Average traffic 8am-10am
        - Evening_Peak: Average traffic 5pm-7pm
        
        Returns:
            DataFrame with junction profiles
        """
        print("\n" + "=" * 70)
        print("TASK 1: JUNCTION CLUSTERING - Creating Profiles")
        print("=" * 70)
        print("Aggregating traffic data for each junction...")
        
        profiles = []
        
        for junction in sorted(self.data['Junction'].unique()):
            junction_data = self.data[self.data['Junction'] == junction]
            
            # Calculate overall statistics
            avg_vehicles = junction_data['Vehicles'].mean()
            max_vehicles = junction_data['Vehicles'].max()
            
            # Morning Peak: 8am-10am average
            morning_data = junction_data[junction_data['Hour'].between(8, 10)]
            morning_peak = morning_data['Vehicles'].mean() if len(morning_data) > 0 else 0
            
            # Evening Peak: 5pm-7pm average
            evening_data = junction_data[junction_data['Hour'].between(17, 19)]
            evening_peak = evening_data['Vehicles'].mean() if len(evening_data) > 0 else 0
            
            profiles.append({
                'Junction': junction,
                'Avg_Vehicles': avg_vehicles,
                'Max_Vehicles': max_vehicles,
                'Morning_Peak': morning_peak,
                'Evening_Peak': evening_peak
            })
        
        self.junction_profiles = pd.DataFrame(profiles)
        
        print(f"✓ Created profiles for {len(self.junction_profiles)} junctions")
        print("\nJunction Profiles:")
        print("-" * 70)
        print(self.junction_profiles.to_string(index=False))
        
        return self.junction_profiles
    
    def perform_clustering(self):
        """
        Apply K-Means clustering to group junctions.
        
        Uses standardized features to ensure fair clustering.
        
        Returns:
            DataFrame with cluster assignments
        """
        print("\n" + "-" * 70)
        print("Performing K-Means Clustering...")
        
        # Select features for clustering
        features = self.junction_profiles[['Avg_Vehicles', 'Max_Vehicles', 
                                          'Morning_Peak', 'Evening_Peak']].values
        
        # Standardize features (important for K-Means)
        # This ensures all features contribute equally
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply K-Means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=20)
        self.cluster_labels = self.kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to profiles
        self.junction_profiles['Cluster'] = self.cluster_labels
        
        print(f"✓ Clustering complete with {self.n_clusters} clusters")
        print(f"\nCluster Distribution:")
        print(self.junction_profiles['Cluster'].value_counts().sort_index())
        
        return self.junction_profiles
    
    def label_clusters(self):
        """
        Automatically assign meaningful labels to clusters.
        
        Analyzes cluster centers and assigns labels based on
        average traffic levels: 'Low Congestion', 'Medium Traffic', 
        'High Congestion'.
        
        Returns:
            DataFrame with labeled clusters
        """
        print("\n" + "-" * 70)
        print("Analyzing and Labeling Clusters...")
        
        # Calculate mean traffic for each cluster
        cluster_means = self.junction_profiles.groupby('Cluster')['Avg_Vehicles'].mean().sort_values()
        
        # Assign labels based on average traffic (sorted low to high)
        labels = ['Low Congestion', 'Medium Traffic', 'High Congestion']
        
        # Create mapping from cluster ID to label
        cluster_to_label = {}
        for i, cluster_id in enumerate(cluster_means.index):
            cluster_to_label[cluster_id] = labels[i] if i < len(labels) else f'Level {i+1}'
        
        # Add label column
        self.junction_profiles['Congestion_Level'] = self.junction_profiles['Cluster'].map(cluster_to_label)
        
        print("\nCluster Centers Analysis:")
        print("-" * 70)
        
        # Get inverse transformed centers for interpretation
        centers_original = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        center_df = pd.DataFrame(
            centers_original,
            columns=['Avg_Vehicles', 'Max_Vehicles', 'Morning_Peak', 'Evening_Peak']
        )
        center_df['Cluster'] = range(self.n_clusters)
        center_df['Label'] = center_df['Cluster'].map(cluster_to_label)
        
        print(center_df.to_string(index=False))
        
        print("\n" + "=" * 70)
        print("CLUSTERING RESULTS")
        print("=" * 70)
        print(self.junction_profiles[['Junction', 'Cluster', 'Congestion_Level', 
                                     'Avg_Vehicles', 'Morning_Peak', 'Evening_Peak']].to_string(index=False))
        
        return self.junction_profiles
    
    def visualize_clusters(self, save=True):
        """
        Visualize the clustering results.
        
        Args:
            save: Whether to save the plot
        """
        print("\n" + "-" * 70)
        print("Creating cluster visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Scatter plot of Avg_Vehicles vs Max_Vehicles
        ax1 = axes[0]
        for cluster in sorted(self.junction_profiles['Cluster'].unique()):
            cluster_data = self.junction_profiles[self.junction_profiles['Cluster'] == cluster]
            label = cluster_data['Congestion_Level'].iloc[0]
            ax1.scatter(cluster_data['Avg_Vehicles'], cluster_data['Max_Vehicles'],
                       s=200, alpha=0.7, label=label, edgecolors='black', linewidth=2)
            
            # Annotate junctions
            for _, row in cluster_data.iterrows():
                ax1.annotate(f"J{row['Junction']}", 
                           (row['Avg_Vehicles'], row['Max_Vehicles']),
                           fontsize=9, fontweight='bold', ha='center')
        
        ax1.set_xlabel('Average Vehicles', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Max Vehicles', fontsize=11, fontweight='bold')
        ax1.set_title('Junction Clusters: Avg vs Max Traffic', fontsize=12, fontweight='bold')
        ax1.legend(title='Congestion Level')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Morning vs Evening Peak
        ax2 = axes[1]
        for cluster in sorted(self.junction_profiles['Cluster'].unique()):
            cluster_data = self.junction_profiles[self.junction_profiles['Cluster'] == cluster]
            label = cluster_data['Congestion_Level'].iloc[0]
            ax2.scatter(cluster_data['Morning_Peak'], cluster_data['Evening_Peak'],
                       s=200, alpha=0.7, label=label, edgecolors='black', linewidth=2)
            
            # Annotate junctions
            for _, row in cluster_data.iterrows():
                ax2.annotate(f"J{row['Junction']}", 
                           (row['Morning_Peak'], row['Evening_Peak']),
                           fontsize=9, fontweight='bold', ha='center')
        
        ax2.set_xlabel('Morning Peak (8-10am)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Evening Peak (5-7pm)', fontsize=11, fontweight='bold')
        ax2.set_title('Junction Clusters: Morning vs Evening Rush', fontsize=12, fontweight='bold')
        ax2.legend(title='Congestion Level')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_dir = "../output/figures"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'junction_clusters.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Cluster visualization saved to: {save_path}")
        
        plt.show()
    
    def run_clustering_pipeline(self):
        """
        Execute complete clustering pipeline.
        
        Returns:
            DataFrame with clustered junctions
        """
        self.create_junction_profiles()
        self.perform_clustering()
        self.label_clusters()
        self.visualize_clusters()
        
        return self.junction_profiles


class TrafficForecasting:
    """
    Supervised learning to forecast traffic trends.
    
    Uses Random Forest to predict next hour's traffic based on
    historical patterns and lag features.
    """
    
    def __init__(self, data, junction_id=1):
        """
        Initialize the forecasting model.
        
        Args:
            data: Processed traffic DataFrame
            junction_id: Junction to forecast (default: 1)
        """
        self.data = data
        self.junction_id = junction_id
        self.junction_data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self):
        """
        Filter data for selected junction and sort by time.
        
        Returns:
            Sorted DataFrame for the junction
        """
        print("\n" + "=" * 70)
        print(f"TASK 2: TREND FORECASTING - Junction {self.junction_id}")
        print("=" * 70)
        print(f"Preparing data for Junction {self.junction_id}...")
        
        # Filter for selected junction
        self.junction_data = self.data[self.data['Junction'] == self.junction_id].copy()
        
        # Sort by DateTime
        self.junction_data = self.junction_data.sort_values('DateTime').reset_index(drop=True)
        
        print(f"✓ Loaded {len(self.junction_data):,} records for Junction {self.junction_id}")
        print(f"Date range: {self.junction_data['DateTime'].min()} to {self.junction_data['DateTime'].max()}")
        
        return self.junction_data
    
    def create_lag_features(self):
        """
        Create lag features for time series forecasting.
        
        Features:
        - Traffic_Last_Hour: Traffic from previous hour (lag 1)
        - Traffic_Yesterday_Same_Hour: Traffic from same hour yesterday (lag 24)
        
        Returns:
            DataFrame with lag features
        """
        print("\n" + "-" * 70)
        print("Creating lag features...")
        
        # Create lag features
        # Traffic_Last_Hour: Traffic from 1 hour ago
        self.junction_data['Traffic_Last_Hour'] = self.junction_data['Vehicles'].shift(1)
        
        # Traffic_Yesterday_Same_Hour: Traffic from 24 hours ago (same hour yesterday)
        self.junction_data['Traffic_Yesterday_Same_Hour'] = self.junction_data['Vehicles'].shift(24)
        
        # Drop rows with NaN values (first 24 hours won't have all features)
        self.junction_data = self.junction_data.dropna()
        
        print(f"✓ Created lag features")
        print(f"✓ Dataset size after removing NaN: {len(self.junction_data):,} records")
        print(f"\nSample of features:")
        print(self.junction_data[['DateTime', 'Hour', 'Vehicles', 
                                  'Traffic_Last_Hour', 'Traffic_Yesterday_Same_Hour']].head(10))
        
        return self.junction_data
    
    def train_model(self):
        """
        Train Random Forest Regressor to predict next hour's traffic.
        
        Features: Hour, DayOfWeek, IsWeekend, Traffic_Last_Hour, Traffic_Yesterday_Same_Hour
        Target: Vehicles (next hour)
        
        Returns:
            Trained model
        """
        print("\n" + "-" * 70)
        print("Training Random Forest Regressor...")
        
        # Select features
        feature_cols = ['Hour', 'DayOfWeek', 'IsWeekend', 
                       'Traffic_Last_Hour', 'Traffic_Yesterday_Same_Hour']
        
        X = self.junction_data[feature_cols]
        y = self.junction_data['Vehicles']
        
        # Split data: 80% train, 20% test
        # Use temporal split (not random) for time series
        split_idx = int(len(X) * 0.8)
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        print(f"Training set: {len(self.X_train):,} records")
        print(f"Test set: {len(self.X_test):,} records")
        
        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        print("✓ Model trained successfully")
        
        return self.model
    
    def evaluate_model(self):
        """
        Evaluate model performance using MAE and R2 Score.
        
        Prints:
        - Mean Absolute Error (MAE): Average prediction error
        - R2 Score: Proportion of variance explained
        """
        print("\n" + "-" * 70)
        print("Evaluating Model Performance...")
        
        # Make predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        # Calculate metrics
        mae_train = mean_absolute_error(self.y_train, y_pred_train)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)
        
        r2_train = r2_score(self.y_train, y_pred_train)
        r2_test = r2_score(self.y_test, y_pred_test)
        
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE METRICS")
        print("=" * 70)
        
        print("\nTraining Set:")
        print(f"  - MAE (Mean Absolute Error): {mae_train:.2f} vehicles")
        print(f"  - R2 Score: {r2_train:.4f}")
        
        print("\nTest Set:")
        print(f"  - MAE (Mean Absolute Error): {mae_test:.2f} vehicles")
        print(f"  - R2 Score: {r2_test:.4f}")
        
        print("\nInterpretation:")
        print(f"  - On average, predictions are off by {mae_test:.2f} vehicles")
        print(f"  - Model explains {r2_test*100:.1f}% of variance in traffic")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': ['Hour', 'DayOfWeek', 'IsWeekend', 
                       'Traffic_Last_Hour', 'Traffic_Yesterday_Same_Hour'],
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n" + "-" * 70)
        print("Feature Importance:")
        print("-" * 70)
        print(feature_importance.to_string(index=False))
        
        return {
            'mae_train': mae_train,
            'mae_test': mae_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'feature_importance': feature_importance
        }
    
    def visualize_predictions(self, save=True):
        """
        Visualize actual vs predicted traffic.
        
        Args:
            save: Whether to save the plot
        """
        print("\n" + "-" * 70)
        print("Creating prediction visualization...")
        
        # Make predictions for test set
        y_pred = self.model.predict(self.X_test)
        
        # Get corresponding datetimes
        test_dates = self.junction_data.iloc[len(self.X_train):]['DateTime'].values
        
        # Plot first 200 test samples for clarity
        n_samples = min(200, len(y_pred))
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(range(n_samples), self.y_test.values[:n_samples], 
               label='Actual', linewidth=2, alpha=0.8, color='#2ecc71')
        ax.plot(range(n_samples), y_pred[:n_samples], 
               label='Predicted', linewidth=2, alpha=0.8, color='#e74c3c', linestyle='--')
        
        ax.set_xlabel('Time Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Vehicles', fontsize=11, fontweight='bold')
        ax.set_title(f'Traffic Forecasting - Junction {self.junction_id} (Test Set)', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_dir = "../output/figures"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'traffic_forecast.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Forecast visualization saved to: {save_path}")
        
        plt.show()
    
    def run_forecasting_pipeline(self):
        """
        Execute complete forecasting pipeline.
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.prepare_data()
        self.create_lag_features()
        self.train_model()
        metrics = self.evaluate_model()
        self.visualize_predictions()
        
        return metrics


def run_ml_engine(data_path=None, data=None):
    """
    Run complete ML pipeline with clustering and forecasting.
    
    Args:
        data_path: Path to processed CSV file
        data: Processed DataFrame (if already loaded)
        
    Returns:
        Tuple of (clustering_results, forecasting_metrics)
    """
    # Load data if not provided
    if data is None:
        if data_path is None:
            data_path = "../output/data/processed_traffic_phase1.csv"
        
        print(f"Loading processed data from: {data_path}")
        data = pd.read_csv(data_path)
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        print(f"✓ Loaded {len(data):,} records")
    
    # Task 1: Junction Clustering
    clustering = JunctionClustering(data, n_clusters=3)
    cluster_results = clustering.run_clustering_pipeline()
    
    # Task 2: Traffic Forecasting
    forecasting = TrafficForecasting(data, junction_id=1)
    forecast_metrics = forecasting.run_forecasting_pipeline()
    
    return cluster_results, forecast_metrics


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STARTING PHASE 3: MACHINE LEARNING ENGINE")
    print("=" * 70)
    
    # Run complete ML pipeline
    clusters, metrics = run_ml_engine()
    
    print("\n" + "=" * 70)
    print("✓ Phase 3: Machine Learning Engine Complete!")
    print("=" * 70)
    
    print("\nKey Outputs:")
    print("1. Junction Clusters: Identified congestion levels for traffic optimization")
    print("2. Traffic Forecast: Predicted next-hour traffic with Random Forest")
    print("\nVisualization files:")
    print("  - junction_clusters.png")
    print("  - traffic_forecast.png")
