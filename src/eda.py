"""
Phase 2: Pattern Analysis
Urban Traffic Flow Analysis - Exploratory Data Analysis (EDA)

Understanding rush hour peaks is crucial for signal timing optimization.
This module analyzes traffic patterns across different temporal dimensions
to identify congestion periods and inform traffic management strategies.

Visualizations include:
- Hourly trends to identify peak traffic times
- Weekly patterns to understand day-to-day variations
- Junction comparisons to identify high-traffic locations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class TrafficEDA:
    """
    Exploratory Data Analysis for traffic patterns.
    
    Analyzes temporal patterns, junction differences, and identifies
    peak congestion periods for traffic optimization.
    """
    
    def __init__(self, data):
        """
        Initialize the EDA analyzer.
        
        Args:
            data: Processed pandas DataFrame with traffic data
        """
        self.data = data
        self.output_dir = "../output/figures"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_hourly_trend(self, save=True):
        """
        Plot Average Vehicles vs. Hour of Day, colored by Junction.
        
        This visualization reveals rush hour patterns and helps identify
        when traffic peaks occur at each junction. Understanding these
        peaks is crucial for optimizing signal timing to reduce congestion.
        
        Args:
            save: Whether to save the plot to file
        """
        print("\n" + "=" * 70)
        print("HOURLY TREND ANALYSIS")
        print("=" * 70)
        print("Analyzing average traffic by hour of day for each junction...")
        
        # Calculate average vehicles per hour for each junction
        hourly_avg = self.data.groupby(['Hour', 'Junction'])['Vehicles'].mean().reset_index()
        
        # Create the plot
        plt.figure(figsize=(14, 6))
        
        # Plot line for each junction with different colors
        for junction in sorted(self.data['Junction'].unique()):
            junction_data = hourly_avg[hourly_avg['Junction'] == junction]
            plt.plot(junction_data['Hour'], junction_data['Vehicles'], 
                    marker='o', linewidth=2.5, markersize=6, 
                    label=f'Junction {junction}', alpha=0.8)
        
        plt.xlabel('Hour of Day', fontsize=12, fontweight='bold')
        plt.ylabel('Average Vehicles', fontsize=12, fontweight='bold')
        plt.title('Hourly Traffic Trend by Junction', fontsize=14, fontweight='bold')
        plt.xticks(range(24))
        plt.legend(title='Junction', fontsize=10, title_fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add annotation for typical rush hours
        plt.axvspan(7, 9, alpha=0.1, color='red', label='Morning Rush (7-9 AM)')
        plt.axvspan(17, 19, alpha=0.1, color='orange', label='Evening Rush (5-7 PM)')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'hourly_trend.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Hourly trend plot saved to: {save_path}")
        
        plt.show()
        
        # Print peak hours for each junction
        print("\nPeak Hours by Junction:")
        print("-" * 70)
        for junction in sorted(self.data['Junction'].unique()):
            junction_hourly = hourly_avg[hourly_avg['Junction'] == junction]
            peak_hour = junction_hourly.loc[junction_hourly['Vehicles'].idxmax()]
            print(f"Junction {junction}: Peak at {int(peak_hour['Hour'])}:00 "
                  f"({peak_hour['Vehicles']:.1f} vehicles)")
        
        return hourly_avg
    
    def plot_weekly_pattern(self, save=True):
        """
        Plot Average Vehicles vs. Day of Week.
        
        This reveals how traffic patterns differ between weekdays and weekends,
        informing different signal timing strategies for different days.
        
        Args:
            save: Whether to save the plot to file
        """
        print("\n" + "=" * 70)
        print("WEEKLY PATTERN ANALYSIS")
        print("=" * 70)
        print("Analyzing average traffic by day of week...")
        
        # Calculate average vehicles per day of week
        weekly_avg = self.data.groupby('DayOfWeek')['Vehicles'].mean().reset_index()
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Define day labels
        day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Create bar plot with color distinction for weekends
        colors = ['#3498db' if day < 5 else '#e74c3c' for day in weekly_avg['DayOfWeek']]
        
        bars = plt.bar(weekly_avg['DayOfWeek'], weekly_avg['Vehicles'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        plt.xlabel('Day of Week', fontsize=12, fontweight='bold')
        plt.ylabel('Average Vehicles', fontsize=12, fontweight='bold')
        plt.title('Weekly Traffic Pattern', fontsize=14, fontweight='bold')
        plt.xticks(range(7), day_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#3498db', label='Weekday'),
                          Patch(facecolor='#e74c3c', label='Weekend')]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'weekly_pattern.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Weekly pattern plot saved to: {save_path}")
        
        plt.show()
        
        # Print statistics
        print("\nWeekly Statistics:")
        print("-" * 70)
        weekday_avg = weekly_avg[weekly_avg['DayOfWeek'] < 5]['Vehicles'].mean()
        weekend_avg = weekly_avg[weekly_avg['DayOfWeek'] >= 5]['Vehicles'].mean()
        print(f"Average Weekday Traffic: {weekday_avg:.1f} vehicles")
        print(f"Average Weekend Traffic: {weekend_avg:.1f} vehicles")
        print(f"Difference: {abs(weekday_avg - weekend_avg):.1f} vehicles "
              f"({abs(weekday_avg - weekend_avg)/weekday_avg*100:.1f}%)")
        
        return weekly_avg
    
    def plot_junction_comparison(self, save=True):
        """
        Create Boxplot of Vehicles for each Junction.
        
        This shows the distribution of traffic at each junction, revealing
        spread, outliers, and variability. Helps identify which junctions
        need more sophisticated traffic management strategies.
        
        Args:
            save: Whether to save the plot to file
        """
        print("\n" + "=" * 70)
        print("JUNCTION COMPARISON ANALYSIS")
        print("=" * 70)
        print("Comparing traffic distribution across junctions...")
        
        # Create the boxplot
        plt.figure(figsize=(12, 6))
        
        # Create boxplot with custom styling
        box_plot = sns.boxplot(
            data=self.data, 
            x='Junction', 
            y='Vehicles',
            palette='Set2',
            linewidth=2
        )
        
        plt.xlabel('Junction', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Vehicles', fontsize=12, fontweight='bold')
        plt.title('Traffic Distribution by Junction (Boxplot)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add mean markers
        means = self.data.groupby('Junction')['Vehicles'].mean()
        positions = range(len(means))
        plt.scatter(positions, means.values, color='red', s=100, zorder=3, 
                   marker='D', label='Mean', edgecolors='black', linewidth=1.5)
        plt.legend(loc='best')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'junction_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Junction comparison plot saved to: {save_path}")
        
        plt.show()
        
        # Print detailed statistics for each junction
        print("\nJunction Statistics (Spread & Outliers):")
        print("-" * 70)
        print(f"{'Junction':<10} {'Mean':<10} {'Median':<10} {'Std Dev':<10} {'IQR':<10} {'Outliers':<10}")
        print("-" * 70)
        
        for junction in sorted(self.data['Junction'].unique()):
            junction_data = self.data[self.data['Junction'] == junction]['Vehicles']
            
            mean_val = junction_data.mean()
            median_val = junction_data.median()
            std_val = junction_data.std()
            
            # Calculate IQR
            q1 = junction_data.quantile(0.25)
            q3 = junction_data.quantile(0.75)
            iqr = q3 - q1
            
            # Count outliers (values beyond 1.5 * IQR from quartiles)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((junction_data < lower_bound) | (junction_data > upper_bound)).sum()
            
            print(f"{junction:<10} {mean_val:<10.1f} {median_val:<10.1f} "
                  f"{std_val:<10.1f} {iqr:<10.1f} {outliers:<10}")
        
        return self.data.groupby('Junction')['Vehicles'].describe()
    
    def generate_all_visualizations(self):
        """
        Generate all EDA visualizations in one go.
        
        Produces hourly trends, weekly patterns, and junction comparisons.
        """
        print("\n" + "=" * 70)
        print("PHASE 2: PATTERN ANALYSIS - GENERATING ALL VISUALIZATIONS")
        print("=" * 70)
        
        # Generate all plots
        self.plot_hourly_trend(save=True)
        self.plot_weekly_pattern(save=True)
        self.plot_junction_comparison(save=True)
        
        print("\n" + "=" * 70)
        print("✓ All visualizations generated successfully!")
        print(f"✓ Saved to: {self.output_dir}")
        print("=" * 70)


def run_pattern_analysis(data_path=None, data=None):
    """
    Run complete pattern analysis pipeline.
    
    Args:
        data_path: Path to processed CSV file (if data not provided)
        data: Processed DataFrame (if already loaded)
        
    Returns:
        TrafficEDA object with analysis results
    """
    # Load data if not provided
    if data is None:
        if data_path is None:
            data_path = "../output/data/processed_traffic_phase1.csv"
        
        print(f"Loading processed data from: {data_path}")
        data = pd.read_csv(data_path)
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        print(f"✓ Loaded {len(data):,} records")
    
    # Create EDA analyzer
    eda = TrafficEDA(data)
    
    # Generate all visualizations
    eda.generate_all_visualizations()
    
    return eda


if __name__ == "__main__":
    # Run Phase 2: Pattern Analysis
    print("\n" + "=" * 70)
    print("STARTING PHASE 2: PATTERN ANALYSIS")
    print("=" * 70)
    
    # Load the processed data from Phase 1
    eda = run_pattern_analysis()
    
    print("\n✓ Phase 2: Pattern Analysis Complete!")
    print("\nKey Insights:")
    print("- Check hourly_trend.png for rush hour identification")
    print("- Check weekly_pattern.png for weekday/weekend differences")
    print("- Check junction_comparison.png for junction-specific characteristics")
