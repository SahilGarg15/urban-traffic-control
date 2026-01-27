"""
Visualization Module for Urban Traffic Flow Analysis

This module creates visualizations for:
- Traffic patterns over time
- Cluster analysis results
- Junction comparisons
- Peak hour heatmaps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class TrafficVisualizer:
    """
    Creates various visualizations for traffic analysis and clustering results.
    """
    
    def __init__(self, output_dir: str = "../output/figures"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        self.color_palette = sns.color_palette("husl", 10)
        
    def plot_traffic_timeseries(self, timeseries_data: pd.DataFrame, 
                               junctions: List[int] = None,
                               save_path: str = None):
        """
        Plot traffic volume over time for selected junctions.
        
        Shows temporal patterns and trends in vehicle counts.
        
        Args:
            timeseries_data: Timeseries data with DateTime, Junction, Vehicles
            junctions: List of junction IDs to plot (default: first 5)
            save_path: Optional path to save figure
        """
        if junctions is None:
            # Plot first 5 junctions if not specified
            junctions = sorted(timeseries_data['Junction'].unique())[:5]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, junction in enumerate(junctions):
            junction_data = timeseries_data[timeseries_data['Junction'] == junction]
            ax.plot(junction_data['DateTime'], junction_data['Vehicles'], 
                   label=f'Junction {junction}', alpha=0.7, linewidth=1.5,
                   color=self.color_palette[i])
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Vehicles', fontsize=12, fontweight='bold')
        ax.set_title('Traffic Volume Over Time by Junction', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Timeseries plot saved to {save_path}")
        
        return fig
    
    def plot_hourly_patterns(self, timeseries_data: pd.DataFrame,
                            cluster_assignments: pd.DataFrame = None,
                            save_path: str = None):
        """
        Plot average traffic by hour of day.
        
        Shows daily traffic patterns. If clusters provided, shows pattern per cluster.
        
        Args:
            timeseries_data: Timeseries data with Hour and Vehicles
            cluster_assignments: Optional cluster assignments to group by
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if cluster_assignments is not None:
            # Merge cluster information
            data = timeseries_data.merge(cluster_assignments, on='Junction', how='left')
            
            # Plot hourly pattern for each cluster
            for cluster_id in sorted(data['Cluster'].unique()):
                cluster_data = data[data['Cluster'] == cluster_id]
                hourly_avg = cluster_data.groupby('Hour')['Vehicles'].mean()
                
                ax.plot(hourly_avg.index, hourly_avg.values, 
                       marker='o', linewidth=2.5, markersize=6,
                       label=f'Cluster {cluster_id}',
                       color=self.color_palette[cluster_id])
        else:
            # Plot overall hourly pattern
            hourly_avg = timeseries_data.groupby('Hour')['Vehicles'].mean()
            ax.plot(hourly_avg.index, hourly_avg.values, 
                   marker='o', linewidth=2.5, markersize=6,
                   color=self.color_palette[0], label='Overall')
        
        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Number of Vehicles', fontsize=12, fontweight='bold')
        ax.set_title('Average Traffic Pattern by Hour of Day', fontsize=14, fontweight='bold')
        ax.set_xticks(range(24))
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Hourly pattern plot saved to {save_path}")
        
        return fig
    
    def plot_weekday_weekend_comparison(self, timeseries_data: pd.DataFrame,
                                       save_path: str = None):
        """
        Compare traffic patterns between weekdays and weekends.
        
        Helps identify different congestion patterns for different day types.
        
        Args:
            timeseries_data: Timeseries data with IsWeekend, Hour, Vehicles
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate hourly averages for weekdays and weekends
        weekday_data = timeseries_data[timeseries_data['IsWeekend'] == 0]
        weekend_data = timeseries_data[timeseries_data['IsWeekend'] == 1]
        
        weekday_hourly = weekday_data.groupby('Hour')['Vehicles'].mean()
        weekend_hourly = weekend_data.groupby('Hour')['Vehicles'].mean()
        
        ax.plot(weekday_hourly.index, weekday_hourly.values, 
               marker='o', linewidth=2.5, markersize=6,
               label='Weekday', color=self.color_palette[0])
        ax.plot(weekend_hourly.index, weekend_hourly.values, 
               marker='s', linewidth=2.5, markersize=6,
               label='Weekend', color=self.color_palette[3])
        
        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Number of Vehicles', fontsize=12, fontweight='bold')
        ax.set_title('Weekday vs Weekend Traffic Patterns', fontsize=14, fontweight='bold')
        ax.set_xticks(range(24))
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Weekday/Weekend comparison saved to {save_path}")
        
        return fig
    
    def plot_cluster_distribution(self, cluster_assignments: pd.DataFrame,
                                  cluster_profiles: pd.DataFrame = None,
                                  save_path: str = None):
        """
        Visualize distribution of junctions across clusters.
        
        Shows how many junctions fall into each congestion level.
        
        Args:
            cluster_assignments: Cluster labels for each junction
            cluster_profiles: Optional profiles with congestion levels
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count junctions per cluster
        cluster_counts = cluster_assignments['Cluster'].value_counts().sort_index()
        
        # Create labels with congestion levels if available
        if cluster_profiles is not None:
            labels = []
            for cluster_id in cluster_counts.index:
                profile = cluster_profiles[cluster_profiles['Cluster'] == cluster_id]
                if not profile.empty:
                    label = f"Cluster {cluster_id}\n{profile['CongestionLevel'].values[0]}"
                else:
                    label = f"Cluster {cluster_id}"
                labels.append(label)
        else:
            labels = [f"Cluster {i}" for i in cluster_counts.index]
        
        # Bar plot
        bars = ax1.bar(range(len(cluster_counts)), cluster_counts.values, 
                      color=[self.color_palette[i] for i in range(len(cluster_counts))],
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        ax1.set_xticks(range(len(cluster_counts)))
        ax1.set_xticklabels(labels, fontsize=10)
        ax1.set_ylabel('Number of Junctions', fontsize=12, fontweight='bold')
        ax1.set_title('Junction Distribution Across Clusters', fontsize=13, fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        colors = [self.color_palette[i] for i in range(len(cluster_counts))]
        wedges, texts, autotexts = ax2.pie(cluster_counts.values, labels=labels,
                                           autopct='%1.1f%%', startangle=90,
                                           colors=colors, textprops={'fontsize': 10})
        ax2.set_title('Cluster Distribution (Percentage)', fontsize=13, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster distribution plot saved to {save_path}")
        
        return fig
    
    def plot_cluster_profiles(self, cluster_profiles: pd.DataFrame,
                             save_path: str = None):
        """
        Visualize cluster characteristics (traffic stats, peak hours).
        
        Compares clusters across multiple metrics.
        
        Args:
            cluster_profiles: DataFrame with cluster profile statistics
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        clusters = cluster_profiles['Cluster'].values
        colors = [self.color_palette[i] for i in range(len(clusters))]
        
        # 1. Average Traffic
        ax = axes[0, 0]
        bars = ax.bar(clusters, cluster_profiles['AvgTraffic'].values, 
                     color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Vehicles/Hour', fontsize=11, fontweight='bold')
        ax.set_title('Average Traffic Volume by Cluster', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, val in zip(bars, cluster_profiles['AvgTraffic'].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2. Peak Hour
        ax = axes[0, 1]
        bars = ax.bar(clusters, cluster_profiles['PeakHour'].values,
                     color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax.set_ylabel('Peak Hour (24-hour format)', fontsize=11, fontweight='bold')
        ax.set_title('Peak Traffic Hour by Cluster', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 24)
        ax.grid(True, axis='y', alpha=0.3)
        for bar, val in zip(bars, cluster_profiles['PeakHour'].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{int(val)}:00', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 3. Traffic Variability (Standard Deviation)
        ax = axes[1, 0]
        bars = ax.bar(clusters, cluster_profiles['AvgStdDev'].values,
                     color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
        ax.set_title('Traffic Variability by Cluster', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, val in zip(bars, cluster_profiles['AvgStdDev'].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 4. Number of Junctions
        ax = axes[1, 1]
        bars = ax.bar(clusters, cluster_profiles['NumJunctions'].values,
                     color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Junctions', fontsize=11, fontweight='bold')
        ax.set_title('Junctions per Cluster', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, val in zip(bars, cluster_profiles['NumJunctions'].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster profiles plot saved to {save_path}")
        
        return fig
    
    def plot_heatmap_hourly_traffic(self, timeseries_data: pd.DataFrame,
                                   junction: int = None,
                                   save_path: str = None):
        """
        Create heatmap showing traffic intensity by hour and day of week.
        
        Visualizes weekly traffic patterns with color-coded intensity.
        
        Args:
            timeseries_data: Timeseries data with Hour, DayOfWeek, Vehicles
            junction: Optional specific junction (default: all junctions)
            save_path: Optional path to save figure
        """
        if junction is not None:
            data = timeseries_data[timeseries_data['Junction'] == junction]
            title = f'Traffic Intensity Heatmap - Junction {junction}'
        else:
            data = timeseries_data
            title = 'Traffic Intensity Heatmap - All Junctions'
        
        # Create pivot table: rows=hour, columns=day of week
        heatmap_data = data.groupby(['Hour', 'DayOfWeek'])['Vehicles'].mean().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.1f', 
                   cbar_kws={'label': 'Average Vehicles'}, ax=ax,
                   linewidths=0.5, linecolor='gray')
        
        ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
        ax.set_ylabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set day labels
        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax.set_xticklabels(day_labels, rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        return fig
    
    def plot_junction_comparison(self, timeseries_data: pd.DataFrame,
                                cluster_assignments: pd.DataFrame,
                                cluster_id: int,
                                max_junctions: int = 5,
                                save_path: str = None):
        """
        Compare hourly patterns of junctions within a specific cluster.
        
        Shows variability within a cluster.
        
        Args:
            timeseries_data: Timeseries data
            cluster_assignments: Cluster assignments
            cluster_id: Cluster to analyze
            max_junctions: Maximum number of junctions to plot
            save_path: Optional path to save figure
        """
        # Get junctions in this cluster
        cluster_junctions = cluster_assignments[
            cluster_assignments['Cluster'] == cluster_id
        ]['Junction'].values[:max_junctions]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, junction in enumerate(cluster_junctions):
            junction_data = timeseries_data[timeseries_data['Junction'] == junction]
            hourly_avg = junction_data.groupby('Hour')['Vehicles'].mean()
            
            ax.plot(hourly_avg.index, hourly_avg.values,
                   marker='o', linewidth=2, markersize=5,
                   label=f'Junction {junction}',
                   color=self.color_palette[i], alpha=0.8)
        
        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Vehicles', fontsize=12, fontweight='bold')
        ax.set_title(f'Hourly Traffic Patterns - Cluster {cluster_id} Junctions', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(24))
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Junction comparison plot saved to {save_path}")
        
        return fig


def create_all_visualizations(timeseries_data: pd.DataFrame,
                              cluster_assignments: pd.DataFrame,
                              cluster_profiles: pd.DataFrame,
                              output_dir: str = "../output/figures"):
    """
    Generate all standard visualizations for the analysis.
    
    Args:
        timeseries_data: Timeseries traffic data
        cluster_assignments: Cluster assignments
        cluster_profiles: Cluster profiles
        output_dir: Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = TrafficVisualizer(output_dir)
    
    print("\n=== Generating Visualizations ===")
    
    # 1. Traffic timeseries
    visualizer.plot_traffic_timeseries(
        timeseries_data,
        save_path=f"{output_dir}/traffic_timeseries.png"
    )
    
    # 2. Hourly patterns by cluster
    visualizer.plot_hourly_patterns(
        timeseries_data,
        cluster_assignments,
        save_path=f"{output_dir}/hourly_patterns_by_cluster.png"
    )
    
    # 3. Weekday vs Weekend
    visualizer.plot_weekday_weekend_comparison(
        timeseries_data,
        save_path=f"{output_dir}/weekday_weekend_comparison.png"
    )
    
    # 4. Cluster distribution
    visualizer.plot_cluster_distribution(
        cluster_assignments,
        cluster_profiles,
        save_path=f"{output_dir}/cluster_distribution.png"
    )
    
    # 5. Cluster profiles
    visualizer.plot_cluster_profiles(
        cluster_profiles,
        save_path=f"{output_dir}/cluster_profiles.png"
    )
    
    # 6. Heatmap
    visualizer.plot_heatmap_hourly_traffic(
        timeseries_data,
        save_path=f"{output_dir}/traffic_heatmap.png"
    )
    
    print(f"\nAll visualizations saved to {output_dir}/")
    print("Generated files:")
    print("  - traffic_timeseries.png")
    print("  - hourly_patterns_by_cluster.png")
    print("  - weekday_weekend_comparison.png")
    print("  - cluster_distribution.png")
    print("  - cluster_profiles.png")
    print("  - traffic_heatmap.png")


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import preprocess_pipeline
    from clustering import clustering_pipeline
    
    data_path = "../data/traffic.csv"
    timeseries, junction_features = preprocess_pipeline(data_path)
    cluster_assignments, cluster_profiles = clustering_pipeline(junction_features, timeseries)
    
    create_all_visualizations(timeseries, cluster_assignments, cluster_profiles)
