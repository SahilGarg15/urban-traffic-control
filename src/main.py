"""
Main Analysis Script for Urban Traffic Flow Analysis & Optimization System

This script orchestrates the complete analysis pipeline:
1. Data preprocessing and feature engineering
2. K-Means clustering analysis
3. Visualization generation
4. Traffic optimization rules generation

Run this script to perform the complete analysis.
"""

import os
import sys
import pandas as pd
import argparse
from datetime import datetime

# Import custom modules
from data_preprocessing import TrafficDataPreprocessor, preprocess_pipeline
from clustering import TrafficClusterAnalyzer, clustering_pipeline
from visualization import create_all_visualizations
from rules_generator import TrafficRulesGenerator, generate_optimization_rules


def create_output_directories():
    """
    Create necessary output directories for results.
    """
    directories = [
        '../output',
        '../output/figures',
        '../output/data',
        '../output/models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Output directories created/verified")


def run_complete_analysis(data_path: str = '../data/traffic.csv',
                         n_clusters: int = 3,
                         auto_select_clusters: bool = False,
                         save_models: bool = True) -> dict:
    """
    Run the complete traffic flow analysis pipeline.
    
    Args:
        data_path: Path to traffic CSV file
        n_clusters: Number of clusters for K-Means (default: 3)
        auto_select_clusters: If True, automatically determine optimal clusters
        save_models: If True, save trained models and processed data
        
    Returns:
        Dictionary containing all analysis results
    """
    print("\n" + "="*80)
    print("URBAN TRAFFIC FLOW ANALYSIS & OPTIMIZATION SYSTEM")
    print("="*80)
    print(f"\nAnalysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data source: {data_path}")
    print(f"Number of clusters: {n_clusters if not auto_select_clusters else 'Auto-select'}")
    
    # Create output directories
    create_output_directories()
    
    # ============================================================
    # STEP 1: DATA PREPROCESSING
    # ============================================================
    print("\n" + "="*80)
    print("STEP 1: DATA PREPROCESSING")
    print("="*80)
    
    preprocessor = TrafficDataPreprocessor(data_path)
    
    # Load raw data
    preprocessor.load_data()
    
    # Clean data (handle missing values, outliers)
    preprocessor.clean_data()
    
    # Extract temporal features (hour, day of week, weekend flag, etc.)
    preprocessor.extract_temporal_features()
    
    # Create junction-level features for clustering
    preprocessor.create_junction_features()
    
    # Get processed datasets
    data = preprocessor.get_processed_data()
    timeseries_data = data['timeseries']
    junction_features = data['junction_features']
    
    # Display data summary
    summary = preprocessor.get_data_summary()
    print("\n--- Data Summary ---")
    print(f"Total records: {summary['total_records']:,}")
    print(f"Number of junctions: {summary['num_junctions']}")
    print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Duration: {summary['date_range']['days']} days")
    print(f"Vehicle count - Mean: {summary['vehicle_stats']['mean']}, "
          f"Median: {summary['vehicle_stats']['median']}, "
          f"Range: [{summary['vehicle_stats']['min']}, {summary['vehicle_stats']['max']}]")
    
    # Save processed data
    if save_models:
        timeseries_data.to_csv('../output/data/processed_timeseries.csv', index=False)
        junction_features.to_csv('../output/data/junction_features.csv')
        print("\nProcessed data saved to ../output/data/")
    
    # ============================================================
    # STEP 2: CLUSTERING ANALYSIS
    # ============================================================
    print("\n" + "="*80)
    print("STEP 2: CLUSTERING ANALYSIS")
    print("="*80)
    
    analyzer = TrafficClusterAnalyzer(n_clusters=n_clusters)
    
    # Perform clustering
    cluster_assignments = analyzer.perform_clustering(
        junction_features,
        auto_select_k=auto_select_clusters
    )
    
    # Create cluster profiles
    cluster_profiles = analyzer.create_cluster_profiles(
        junction_features,
        cluster_assignments,
        timeseries_data
    )
    
    # Save clustering results
    if save_models:
        cluster_assignments.to_csv('../output/data/cluster_assignments.csv', index=False)
        cluster_profiles.to_csv('../output/data/cluster_profiles.csv', index=False)
        analyzer.save_model('../output/models/clustering_model')
        print("\nClustering results and model saved")
    
    # ============================================================
    # STEP 3: VISUALIZATION GENERATION
    # ============================================================
    print("\n" + "="*80)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("="*80)
    
    create_all_visualizations(
        timeseries_data,
        cluster_assignments,
        cluster_profiles,
        output_dir='../output/figures'
    )
    
    # ============================================================
    # STEP 4: TRAFFIC RULES GENERATION
    # ============================================================
    print("\n" + "="*80)
    print("STEP 4: GENERATING TRAFFIC OPTIMIZATION RULES")
    print("="*80)
    
    generator = TrafficRulesGenerator()
    generator.generate_cluster_rules(
        cluster_profiles,
        timeseries_data,
        cluster_assignments
    )
    
    # Print rules summary
    generator.print_rules_summary()
    
    # Export rules
    rules_df = generator.export_rules_to_dataframe()
    rules_df.to_csv('../output/traffic_optimization_rules.csv', index=False)
    print(f"\n\nTraffic rules exported to ../output/traffic_optimization_rules.csv")
    
    # ============================================================
    # ANALYSIS COMPLETE
    # ============================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n--- Generated Outputs ---")
    print("Data:")
    print("  - ../output/data/processed_timeseries.csv")
    print("  - ../output/data/junction_features.csv")
    print("  - ../output/data/cluster_assignments.csv")
    print("  - ../output/data/cluster_profiles.csv")
    print("\nVisualizations:")
    print("  - ../output/figures/traffic_timeseries.png")
    print("  - ../output/figures/hourly_patterns_by_cluster.png")
    print("  - ../output/figures/weekday_weekend_comparison.png")
    print("  - ../output/figures/cluster_distribution.png")
    print("  - ../output/figures/cluster_profiles.png")
    print("  - ../output/figures/traffic_heatmap.png")
    print("\nRules:")
    print("  - ../output/traffic_optimization_rules.csv")
    print("\nModels:")
    print("  - ../output/models/clustering_model.pkl")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("1. Review the generated visualizations in ../output/figures/")
    print("2. Examine cluster profiles in ../output/data/cluster_profiles.csv")
    print("3. Review traffic optimization rules in ../output/traffic_optimization_rules.csv")
    print("4. Run the Streamlit dashboard: streamlit run app.py")
    print("="*80)
    
    # Return all results
    return {
        'timeseries_data': timeseries_data,
        'junction_features': junction_features,
        'cluster_assignments': cluster_assignments,
        'cluster_profiles': cluster_profiles,
        'rules': rules_df,
        'preprocessor': preprocessor,
        'analyzer': analyzer,
        'generator': generator
    }


def main():
    """
    Main entry point with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description='Urban Traffic Flow Analysis & Optimization System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (3 clusters)
  python main.py
  
  # Run with 4 clusters
  python main.py --clusters 4
  
  # Auto-select optimal number of clusters
  python main.py --auto-clusters
  
  # Use custom data file
  python main.py --data ../data/my_traffic.csv
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='../data/traffic.csv',
        help='Path to traffic CSV file (default: ../data/traffic.csv)'
    )
    
    parser.add_argument(
        '--clusters',
        type=int,
        default=3,
        help='Number of clusters for K-Means (default: 3)'
    )
    
    parser.add_argument(
        '--auto-clusters',
        action='store_true',
        help='Automatically determine optimal number of clusters'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save models and intermediate data'
    )
    
    args = parser.parse_args()
    
    # Validate data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    # Run analysis
    try:
        results = run_complete_analysis(
            data_path=args.data,
            n_clusters=args.clusters,
            auto_select_clusters=args.auto_clusters,
            save_models=not args.no_save
        )
        
        print("\n✓ Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
