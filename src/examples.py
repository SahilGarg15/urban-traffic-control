"""
Example script demonstrating basic usage of the analysis modules.

This shows how to use individual components without running the full pipeline.
"""

from data_preprocessing import TrafficDataPreprocessor
from clustering import TrafficClusterAnalyzer
from visualization import TrafficVisualizer
from rules_generator import TrafficRulesGenerator

# Example 1: Load and preprocess data
print("=" * 60)
print("EXAMPLE 1: Data Preprocessing")
print("=" * 60)

preprocessor = TrafficDataPreprocessor('../data/traffic.csv')
preprocessor.load_data()
preprocessor.clean_data()
preprocessor.extract_temporal_features()
preprocessor.create_junction_features()

summary = preprocessor.get_data_summary()
print(f"\nDataset contains {summary['num_junctions']} junctions")
print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")

# Example 2: Perform clustering
print("\n" + "=" * 60)
print("EXAMPLE 2: Clustering Analysis")
print("=" * 60)

data = preprocessor.get_processed_data()
timeseries_data = data['timeseries']
junction_features = data['junction_features']

analyzer = TrafficClusterAnalyzer(n_clusters=3)
cluster_assignments = analyzer.perform_clustering(junction_features)
cluster_profiles = analyzer.create_cluster_profiles(
    junction_features, 
    cluster_assignments, 
    timeseries_data
)

print(f"\nClustering complete. Found {len(cluster_profiles)} clusters.")

# Example 3: Create a single visualization
print("\n" + "=" * 60)
print("EXAMPLE 3: Visualization")
print("=" * 60)

visualizer = TrafficVisualizer()
print("\nCreating hourly pattern visualization...")
visualizer.plot_hourly_patterns(
    timeseries_data, 
    cluster_assignments,
    save_path='../output/example_hourly_pattern.png'
)
print("Visualization saved!")

# Example 4: Generate rules for a specific cluster
print("\n" + "=" * 60)
print("EXAMPLE 4: Rules Generation")
print("=" * 60)

generator = TrafficRulesGenerator()
generator.generate_cluster_rules(cluster_profiles, timeseries_data, cluster_assignments)

# Get only high priority rules
high_priority_rules = generator.get_rules_by_priority('High')
print(f"\nFound {len(high_priority_rules)} high-priority rules:")
for rule in high_priority_rules:
    print(f"\n- {rule['rule_type']}")
    print(f"  {rule['description'][:100]}...")

print("\n" + "=" * 60)
print("Examples complete!")
print("=" * 60)
