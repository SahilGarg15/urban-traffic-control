"""
Phase 4: Decision Logic
Urban Traffic Flow Analysis - Traffic Optimizer

This is the core logic layer that generates actionable recommendations
based on clustering results and traffic pattern analysis.

The optimizer applies rule-based logic to suggest traffic management
strategies tailored to each junction's congestion level and behavior.
"""

import pandas as pd
import numpy as np


class TrafficOptimizer:
    """
    Generates traffic optimization recommendations based on ML insights.
    
    Applies rule-based decision logic to create actionable strategies
    for traffic signal optimization and infrastructure planning.
    """
    
    def __init__(self):
        """Initialize the traffic optimizer."""
        self.recommendations_history = []
        
    def generate_recommendations(self, junction_id, cluster_label, peak_hour_data):
        """
        Generate traffic optimization recommendations for a junction.
        
        Applies multiple rule-based checks to create tailored strategies
        based on congestion level and traffic patterns.
        
        Args:
            junction_id: Junction identifier
            cluster_label: Cluster label ('Low Congestion', 'Medium Traffic', 'High Congestion')
            peak_hour_data: Dictionary with keys:
                - 'Morning_Peak': Average traffic 8am-10am
                - 'Evening_Peak': Average traffic 5pm-7pm
                - 'Weekend_Traffic': Average weekend traffic (optional)
                - 'Weekday_Traffic': Average weekday traffic (optional)
        
        Returns:
            List of string recommendations
        """
        recommendations = []
        
        # Extract peak hour metrics
        morning_peak = peak_hour_data.get('Morning_Peak', 0)
        evening_peak = peak_hour_data.get('Evening_Peak', 0)
        weekend_traffic = peak_hour_data.get('Weekend_Traffic', 0)
        weekday_traffic = peak_hour_data.get('Weekday_Traffic', 0)
        
        # Rule 1: High Congestion Cluster
        # Critical infrastructure intervention needed
        if cluster_label == 'High Congestion':
            recommendations.append(
                "Priority Intersection: Consider constructing a flyover or deploying "
                "traffic police during peak hours."
            )
        
        # Rule 2: Morning Peak Dominance
        # Indicates commuter/inbound traffic pattern
        if morning_peak > evening_peak:
            recommendations.append(
                "Commuter Route: Optimize signal timing for inbound traffic 08:00-10:00."
            )
        
        # Rule 3: Weekend Traffic Dominance
        # Indicates leisure/shopping destination
        if weekend_traffic > 0 and weekday_traffic > 0:
            if weekend_traffic > weekday_traffic:
                recommendations.append(
                    "Leisure Zone: Expect delays near shopping/recreational areas on Saturdays."
                )
        
        # Additional Rule 4: Medium Traffic - Smart Signal Optimization
        if cluster_label == 'Medium Traffic':
            recommendations.append(
                "Moderate Congestion: Implement adaptive signal control systems to "
                "dynamically adjust timing based on real-time flow."
            )
        
        # Additional Rule 5: Low Congestion - Efficiency Focus
        if cluster_label == 'Low Congestion':
            recommendations.append(
                "Low Traffic Zone: Maintain current signal timings. Consider reducing "
                "cycle time to improve pedestrian crossing efficiency."
            )
        
        # Additional Rule 6: Balanced Peak Hours
        if morning_peak > 0 and evening_peak > 0:
            peak_ratio = abs(morning_peak - evening_peak) / max(morning_peak, evening_peak)
            if peak_ratio < 0.15:  # Less than 15% difference
                recommendations.append(
                    "Balanced Traffic: Similar morning and evening peaks detected. "
                    "Use time-of-day signal plans with consistent green time allocation."
                )
        
        # Additional Rule 7: Evening Peak Dominance
        if evening_peak > morning_peak:
            recommendations.append(
                "Evening Rush Dominant: Optimize signal timing for outbound traffic 17:00-19:00. "
                "Prioritize routes away from city center."
            )
        
        # Store in history for analysis
        self.recommendations_history.append({
            'Junction': junction_id,
            'Cluster': cluster_label,
            'Morning_Peak': morning_peak,
            'Evening_Peak': evening_peak,
            'Recommendations': recommendations
        })
        
        return recommendations
    
    def generate_batch_recommendations(self, junction_profiles):
        """
        Generate recommendations for multiple junctions at once.
        
        Args:
            junction_profiles: DataFrame with columns:
                - Junction
                - Congestion_Level (cluster label)
                - Morning_Peak
                - Evening_Peak
                - Avg_Vehicles (optional)
                - Weekend_Traffic (optional)
                - Weekday_Traffic (optional)
        
        Returns:
            DataFrame with junction_id and recommendations
        """
        print("\n" + "=" * 70)
        print("GENERATING OPTIMIZATION RECOMMENDATIONS")
        print("=" * 70)
        
        all_recommendations = []
        
        for _, row in junction_profiles.iterrows():
            junction_id = row['Junction']
            cluster_label = row.get('Congestion_Level', row.get('Cluster', 'Unknown'))
            
            # Build peak hour data dictionary
            peak_hour_data = {
                'Morning_Peak': row.get('Morning_Peak', 0),
                'Evening_Peak': row.get('Evening_Peak', 0),
                'Weekend_Traffic': row.get('Weekend_Traffic', 0),
                'Weekday_Traffic': row.get('Weekday_Traffic', 0)
            }
            
            # Generate recommendations
            recommendations = self.generate_recommendations(
                junction_id, cluster_label, peak_hour_data
            )
            
            all_recommendations.append({
                'Junction': junction_id,
                'Congestion_Level': cluster_label,
                'Num_Recommendations': len(recommendations),
                'Recommendations': recommendations
            })
        
        results_df = pd.DataFrame(all_recommendations)
        return results_df
    
    def print_recommendations(self, junction_id, recommendations):
        """
        Print formatted recommendations for a junction.
        
        Args:
            junction_id: Junction identifier
            recommendations: List of recommendation strings
        """
        print("\n" + "=" * 70)
        print(f"RECOMMENDATIONS FOR JUNCTION {junction_id}")
        print("=" * 70)
        
        if not recommendations:
            print("No specific recommendations - traffic patterns are normal.")
        else:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec}")
        
        print("\n" + "=" * 70)
    
    def print_all_recommendations(self, recommendations_df):
        """
        Print all recommendations in a formatted report.
        
        Args:
            recommendations_df: DataFrame with recommendations
        """
        print("\n" + "=" * 70)
        print("TRAFFIC OPTIMIZATION REPORT")
        print("=" * 70)
        
        for _, row in recommendations_df.iterrows():
            print(f"\n{'─' * 70}")
            print(f"Junction {row['Junction']} - {row['Congestion_Level']}")
            print(f"{'─' * 70}")
            
            if row['Recommendations']:
                for i, rec in enumerate(row['Recommendations'], 1):
                    print(f"{i}. {rec}")
            else:
                print("No specific recommendations - traffic patterns are normal.")
        
        print("\n" + "=" * 70)
    
    def export_recommendations(self, recommendations_df, output_path):
        """
        Export recommendations to CSV file.
        
        Args:
            recommendations_df: DataFrame with recommendations
            output_path: Path to save CSV file
        """
        import os
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Create export format with expanded recommendations
        export_data = []
        
        for _, row in recommendations_df.iterrows():
            # Join recommendations into a single text field
            recs_text = " | ".join(row['Recommendations']) if row['Recommendations'] else "None"
            
            export_data.append({
                'Junction': row['Junction'],
                'Congestion_Level': row['Congestion_Level'],
                'Num_Recommendations': row['Num_Recommendations'],
                'Recommendations': recs_text
            })
        
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Recommendations exported to: {output_path}")
        
        return export_df


def apply_optimizer_to_clusters(cluster_results, traffic_data):
    """
    Apply optimizer to clustering results with enhanced traffic metrics.
    
    Args:
        cluster_results: DataFrame from clustering (junction profiles with clusters)
        traffic_data: Original traffic data for calculating weekend/weekday metrics
        
    Returns:
        DataFrame with recommendations
    """
    print("\n" + "=" * 70)
    print("PHASE 4: DECISION LOGIC - TRAFFIC OPTIMIZER")
    print("=" * 70)
    
    # Calculate weekend vs weekday traffic for each junction
    enhanced_profiles = cluster_results.copy()
    
    weekend_traffic = []
    weekday_traffic = []
    
    for junction in enhanced_profiles['Junction']:
        junction_data = traffic_data[traffic_data['Junction'] == junction]
        
        # Calculate weekend average
        weekend_avg = junction_data[junction_data['IsWeekend'] == 1]['Vehicles'].mean()
        weekend_traffic.append(weekend_avg if not pd.isna(weekend_avg) else 0)
        
        # Calculate weekday average
        weekday_avg = junction_data[junction_data['IsWeekend'] == 0]['Vehicles'].mean()
        weekday_traffic.append(weekday_avg if not pd.isna(weekday_avg) else 0)
    
    enhanced_profiles['Weekend_Traffic'] = weekend_traffic
    enhanced_profiles['Weekday_Traffic'] = weekday_traffic
    
    print("\nEnhanced Junction Profiles:")
    print("-" * 70)
    print(enhanced_profiles[['Junction', 'Congestion_Level', 'Morning_Peak', 
                            'Evening_Peak', 'Weekend_Traffic', 'Weekday_Traffic']].to_string(index=False))
    
    # Generate recommendations
    optimizer = TrafficOptimizer()
    recommendations_df = optimizer.generate_batch_recommendations(enhanced_profiles)
    
    # Print formatted report
    optimizer.print_all_recommendations(recommendations_df)
    
    # Export to CSV
    output_path = "../output/traffic_recommendations.csv"
    optimizer.export_recommendations(recommendations_df, output_path)
    
    return recommendations_df, optimizer


def demo_single_junction():
    """
    Demonstrate recommendation generation for a single junction.
    """
    print("\n" + "=" * 70)
    print("DEMO: SINGLE JUNCTION RECOMMENDATION")
    print("=" * 70)
    
    optimizer = TrafficOptimizer()
    
    # Example: High congestion junction with morning peak
    junction_id = 1
    cluster_label = 'High Congestion'
    peak_hour_data = {
        'Morning_Peak': 65.5,
        'Evening_Peak': 48.3,
        'Weekend_Traffic': 35.2,
        'Weekday_Traffic': 52.8
    }
    
    print(f"\nJunction {junction_id} Profile:")
    print(f"  - Cluster: {cluster_label}")
    print(f"  - Morning Peak (8-10am): {peak_hour_data['Morning_Peak']:.1f} vehicles")
    print(f"  - Evening Peak (5-7pm): {peak_hour_data['Evening_Peak']:.1f} vehicles")
    print(f"  - Weekend Traffic: {peak_hour_data['Weekend_Traffic']:.1f} vehicles")
    print(f"  - Weekday Traffic: {peak_hour_data['Weekday_Traffic']:.1f} vehicles")
    
    recommendations = optimizer.generate_recommendations(
        junction_id, cluster_label, peak_hour_data
    )
    
    optimizer.print_recommendations(junction_id, recommendations)
    
    return recommendations


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STARTING PHASE 4: DECISION LOGIC")
    print("=" * 70)
    
    # Demo with single junction
    demo_single_junction()
    
    # Load data and apply to all junctions
    print("\n\n" + "=" * 70)
    print("APPLYING OPTIMIZER TO ALL JUNCTIONS")
    print("=" * 70)
    
    # Load processed data
    traffic_data = pd.read_csv("../output/data/processed_traffic_phase1.csv")
    traffic_data['DateTime'] = pd.to_datetime(traffic_data['DateTime'])
    
    # Load clustering results (assuming Phase 3 was run)
    try:
        # Try to load from ml_models output
        # For now, we'll create sample data
        from ml_models import JunctionClustering
        
        clustering = JunctionClustering(traffic_data, n_clusters=3)
        cluster_results = clustering.run_clustering_pipeline()
        
        # Apply optimizer
        recommendations_df, optimizer = apply_optimizer_to_clusters(cluster_results, traffic_data)
        
        print("\n" + "=" * 70)
        print("✓ Phase 4: Decision Logic Complete!")
        print("=" * 70)
        print("\nGenerated personalized recommendations for each junction")
        print("based on congestion level and traffic patterns.")
        
    except Exception as e:
        print(f"\nNote: Run ml_models.py first to generate clustering results")
        print(f"Error: {e}")
