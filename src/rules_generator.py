"""
Traffic Rules Generator Module

Generates actionable traffic optimization rules based on:
- Cluster analysis results
- Peak hour patterns
- Congestion levels
- Temporal traffic patterns

Rules suggest specific interventions (e.g., signal timing, lane management)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import time


class TrafficRulesGenerator:
    """
    Generates traffic optimization rules based on analysis results.
    
    Creates actionable recommendations for traffic management
    tailored to each cluster's congestion patterns.
    """
    
    def __init__(self):
        """Initialize the rules generator."""
        self.rules = []
        
    def generate_cluster_rules(self, 
                              cluster_profiles: pd.DataFrame,
                              timeseries_data: pd.DataFrame,
                              cluster_assignments: pd.DataFrame) -> List[Dict]:
        """
        Generate traffic rules for each cluster based on their characteristics.
        
        Args:
            cluster_profiles: Cluster profile statistics
            timeseries_data: Timeseries traffic data
            cluster_assignments: Junction-to-cluster mappings
            
        Returns:
            List of rule dictionaries
        """
        print("\n=== Generating Traffic Optimization Rules ===")
        
        self.rules = []
        
        for _, profile in cluster_profiles.iterrows():
            cluster_id = profile['Cluster']
            congestion_level = profile['CongestionLevel']
            
            # Get junctions in this cluster
            cluster_junctions = cluster_assignments[
                cluster_assignments['Cluster'] == cluster_id
            ]['Junction'].tolist()
            
            # Generate rules based on congestion level and patterns
            cluster_rules = self._generate_rules_for_cluster(
                cluster_id, profile, timeseries_data, cluster_junctions
            )
            
            self.rules.extend(cluster_rules)
        
        print(f"\nGenerated {len(self.rules)} optimization rules")
        return self.rules
    
    def _generate_rules_for_cluster(self,
                                   cluster_id: int,
                                   profile: pd.Series,
                                   timeseries_data: pd.DataFrame,
                                   junctions: List[int]) -> List[Dict]:
        """
        Generate specific rules for a single cluster.
        
        Args:
            cluster_id: Cluster identifier
            profile: Cluster profile data
            timeseries_data: Full timeseries data
            junctions: List of junctions in this cluster
            
        Returns:
            List of rules for this cluster
        """
        rules = []
        
        # Get data for junctions in this cluster
        cluster_data = timeseries_data[timeseries_data['Junction'].isin(junctions)]
        
        # Extract key metrics
        avg_traffic = profile['AvgTraffic']
        peak_hour = int(profile['PeakHour'])
        congestion_level = profile['CongestionLevel']
        avg_std = profile['AvgStdDev']
        
        # Rule 1: Peak Hour Management
        peak_rule = self._generate_peak_hour_rule(
            cluster_id, congestion_level, peak_hour, avg_traffic, junctions
        )
        rules.append(peak_rule)
        
        # Rule 2: Signal Timing Optimization
        signal_rule = self._generate_signal_timing_rule(
            cluster_id, congestion_level, avg_traffic, avg_std, junctions
        )
        rules.append(signal_rule)
        
        # Rule 3: Rush Hour Strategy
        rush_hour_rules = self._generate_rush_hour_strategy(
            cluster_id, congestion_level, cluster_data, junctions
        )
        rules.extend(rush_hour_rules)
        
        # Rule 4: Weekend Management (if different from weekday)
        weekend_rule = self._generate_weekend_rule(
            cluster_id, congestion_level, cluster_data, junctions
        )
        if weekend_rule:
            rules.append(weekend_rule)
        
        return rules
    
    def _generate_peak_hour_rule(self,
                                cluster_id: int,
                                congestion_level: str,
                                peak_hour: int,
                                avg_traffic: float,
                                junctions: List[int]) -> Dict:
        """
        Generate rule for peak hour management.
        
        Returns:
            Rule dictionary
        """
        # Determine signal duration adjustment based on congestion
        if 'High' in congestion_level:
            duration_increase = "40-50%"
            priority = "Critical"
            additional_action = "Consider implementing adaptive signal control"
        elif 'Medium' in congestion_level:
            duration_increase = "25-35%"
            priority = "High"
            additional_action = "Monitor traffic flow and adjust as needed"
        else:
            duration_increase = "10-15%"
            priority = "Medium"
            additional_action = "Maintain standard signal timing with minor adjustments"
        
        # Create readable time range (peak hour ± 1 hour)
        start_hour = max(0, peak_hour - 1)
        end_hour = min(23, peak_hour + 1)
        time_range = f"{start_hour:02d}:00 - {end_hour:02d}:00"
        
        return {
            'cluster_id': cluster_id,
            'congestion_level': congestion_level,
            'rule_type': 'Peak Hour Management',
            'priority': priority,
            'description': (
                f"Increase green signal duration by {duration_increase} during peak hours "
                f"({time_range}) at junctions in {congestion_level} areas."
            ),
            'specific_action': (
                f"At {self._format_junction_list(junctions)}: "
                f"Extend green light phase by {duration_increase} at {peak_hour:02d}:00. "
                f"{additional_action}."
            ),
            'expected_impact': 'Reduce queue length and waiting time during peak congestion',
            'implementation_time': f'{time_range} daily'
        }
    
    def _generate_signal_timing_rule(self,
                                    cluster_id: int,
                                    congestion_level: str,
                                    avg_traffic: float,
                                    variability: float,
                                    junctions: List[int]) -> Dict:
        """
        Generate rule for overall signal timing optimization.
        
        Returns:
            Rule dictionary
        """
        # High variability suggests need for adaptive signals
        if variability > avg_traffic * 0.5:  # Std dev > 50% of mean
            timing_type = "Adaptive/Dynamic"
            description = (
                f"Implement adaptive signal timing that responds to real-time traffic conditions. "
                f"High variability (σ={variability:.1f}) indicates fluctuating traffic patterns."
            )
            priority = "High"
        else:
            timing_type = "Fixed with Time-of-Day"
            description = (
                f"Use time-of-day signal timing plans with consistent patterns. "
                f"Low variability (σ={variability:.1f}) indicates predictable traffic."
            )
            priority = "Medium"
        
        # Determine cycle length based on congestion
        if 'High' in congestion_level:
            cycle_length = "120-150 seconds"
        elif 'Medium' in congestion_level:
            cycle_length = "90-120 seconds"
        else:
            cycle_length = "60-90 seconds"
        
        return {
            'cluster_id': cluster_id,
            'congestion_level': congestion_level,
            'rule_type': 'Signal Timing Strategy',
            'priority': priority,
            'description': description,
            'specific_action': (
                f"At {self._format_junction_list(junctions)}: "
                f"Set signal cycle length to {cycle_length}. "
                f"Use {timing_type} signal control system."
            ),
            'expected_impact': 'Optimize traffic flow and minimize delays',
            'implementation_time': 'Continuous'
        }
    
    def _generate_rush_hour_strategy(self,
                                    cluster_id: int,
                                    congestion_level: str,
                                    cluster_data: pd.DataFrame,
                                    junctions: List[int]) -> List[Dict]:
        """
        Generate rules for morning and evening rush hours.
        
        Returns:
            List of rush hour rules
        """
        rules = []
        
        # Identify morning rush (6-10 AM) and evening rush (4-8 PM)
        morning_data = cluster_data[cluster_data['Hour'].between(6, 10)]
        evening_data = cluster_data[cluster_data['Hour'].between(16, 20)]
        
        morning_avg = morning_data['Vehicles'].mean()
        evening_avg = evening_data['Vehicles'].mean()
        
        # Morning Rush Rule
        if morning_avg > cluster_data['Vehicles'].mean() * 1.2:  # 20% above average
            if 'High' in congestion_level:
                action = "Deploy traffic officers and consider reversible lanes"
                priority = "Critical"
            elif 'Medium' in congestion_level:
                action = "Increase signal green time by 30% for main routes"
                priority = "High"
            else:
                action = "Monitor traffic flow and maintain standard timing"
                priority = "Medium"
            
            rules.append({
                'cluster_id': cluster_id,
                'congestion_level': congestion_level,
                'rule_type': 'Morning Rush Hour Strategy',
                'priority': priority,
                'description': (
                    f"Morning rush hour (06:00-10:00) shows elevated traffic "
                    f"({morning_avg:.1f} vehicles/hour, {((morning_avg/cluster_data['Vehicles'].mean()-1)*100):.1f}% above average)."
                ),
                'specific_action': (
                    f"At {self._format_junction_list(junctions)}: {action}. "
                    f"Prioritize routes toward city center/business districts."
                ),
                'expected_impact': 'Reduce morning commute delays by 15-25%',
                'implementation_time': '06:00-10:00 on weekdays'
            })
        
        # Evening Rush Rule
        if evening_avg > cluster_data['Vehicles'].mean() * 1.2:
            if 'High' in congestion_level:
                action = "Maximize outbound route capacity and extend green times by 40%"
                priority = "Critical"
            elif 'Medium' in congestion_level:
                action = "Increase signal green time by 30% for outbound routes"
                priority = "High"
            else:
                action = "Adjust timing to favor outbound traffic flow"
                priority = "Medium"
            
            rules.append({
                'cluster_id': cluster_id,
                'congestion_level': congestion_level,
                'rule_type': 'Evening Rush Hour Strategy',
                'priority': priority,
                'description': (
                    f"Evening rush hour (16:00-20:00) shows elevated traffic "
                    f"({evening_avg:.1f} vehicles/hour, {((evening_avg/cluster_data['Vehicles'].mean()-1)*100):.1f}% above average)."
                ),
                'specific_action': (
                    f"At {self._format_junction_list(junctions)}: {action}. "
                    f"Prioritize routes away from city center/toward residential areas."
                ),
                'expected_impact': 'Reduce evening commute delays by 15-25%',
                'implementation_time': '16:00-20:00 on weekdays'
            })
        
        return rules
    
    def _generate_weekend_rule(self,
                              cluster_id: int,
                              congestion_level: str,
                              cluster_data: pd.DataFrame,
                              junctions: List[int]) -> Dict:
        """
        Generate rule for weekend traffic management if pattern differs significantly.
        
        Returns:
            Weekend rule dictionary or None
        """
        # Compare weekend vs weekday traffic
        weekend_data = cluster_data[cluster_data['IsWeekend'] == 1]
        weekday_data = cluster_data[cluster_data['IsWeekend'] == 0]
        
        if len(weekend_data) == 0 or len(weekday_data) == 0:
            return None
        
        weekend_avg = weekend_data['Vehicles'].mean()
        weekday_avg = weekday_data['Vehicles'].mean()
        
        # Only create rule if weekend traffic differs by more than 20%
        ratio = weekend_avg / weekday_avg
        
        if ratio < 0.8:  # Weekend traffic is significantly lower
            return {
                'cluster_id': cluster_id,
                'congestion_level': congestion_level,
                'rule_type': 'Weekend Traffic Management',
                'priority': 'Low',
                'description': (
                    f"Weekend traffic is {((1-ratio)*100):.1f}% lower than weekdays "
                    f"({weekend_avg:.1f} vs {weekday_avg:.1f} vehicles/hour)."
                ),
                'specific_action': (
                    f"At {self._format_junction_list(junctions)}: "
                    f"Reduce signal cycle lengths by 20-30% on weekends. "
                    f"Consider using pedestrian-priority timing in retail/leisure areas."
                ),
                'expected_impact': 'Improve pedestrian flow and reduce unnecessary delays',
                'implementation_time': 'Saturdays and Sundays'
            }
        elif ratio > 1.2:  # Weekend traffic is significantly higher
            return {
                'cluster_id': cluster_id,
                'congestion_level': congestion_level,
                'rule_type': 'Weekend Traffic Management',
                'priority': 'Medium',
                'description': (
                    f"Weekend traffic is {((ratio-1)*100):.1f}% higher than weekdays "
                    f"({weekend_avg:.1f} vs {weekday_avg:.1f} vehicles/hour)."
                ),
                'specific_action': (
                    f"At {self._format_junction_list(junctions)}: "
                    f"Increase signal capacity on weekends. "
                    f"Focus on routes to shopping/entertainment districts."
                ),
                'expected_impact': 'Prevent weekend congestion buildup',
                'implementation_time': 'Saturdays and Sundays'
            }
        
        return None
    
    def _format_junction_list(self, junctions: List[int], max_display: int = 5) -> str:
        """
        Format junction list for display in rules.
        
        Args:
            junctions: List of junction IDs
            max_display: Maximum junctions to show before using "and X others"
            
        Returns:
            Formatted string
        """
        if len(junctions) <= max_display:
            junction_str = ', '.join([f'Junction {j}' for j in junctions])
        else:
            displayed = ', '.join([f'Junction {j}' for j in junctions[:max_display]])
            remaining = len(junctions) - max_display
            junction_str = f"{displayed} and {remaining} others"
        
        return junction_str
    
    def export_rules_to_dataframe(self) -> pd.DataFrame:
        """
        Export rules to a pandas DataFrame for easy viewing and export.
        
        Returns:
            DataFrame with all rules
        """
        if not self.rules:
            return pd.DataFrame()
        
        return pd.DataFrame(self.rules)
    
    def export_rules_to_csv(self, filepath: str):
        """
        Export rules to CSV file.
        
        Args:
            filepath: Path to save CSV file
        """
        df = self.export_rules_to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"\nRules exported to {filepath}")
    
    def print_rules_summary(self):
        """
        Print a formatted summary of all generated rules.
        """
        if not self.rules:
            print("No rules generated yet.")
            return
        
        print("\n" + "="*80)
        print("TRAFFIC OPTIMIZATION RULES SUMMARY")
        print("="*80)
        
        # Group rules by cluster
        clusters = {}
        for rule in self.rules:
            cluster_id = rule['cluster_id']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(rule)
        
        for cluster_id in sorted(clusters.keys()):
            cluster_rules = clusters[cluster_id]
            congestion = cluster_rules[0]['congestion_level']
            
            print(f"\n{'─'*80}")
            print(f"CLUSTER {cluster_id} - {congestion}")
            print(f"{'─'*80}")
            
            for i, rule in enumerate(cluster_rules, 1):
                print(f"\n{i}. {rule['rule_type']} [Priority: {rule['priority']}]")
                print(f"   Description: {rule['description']}")
                print(f"   Action: {rule['specific_action']}")
                print(f"   Expected Impact: {rule['expected_impact']}")
                print(f"   When: {rule['implementation_time']}")
        
        print("\n" + "="*80)
    
    def get_rules_by_priority(self, priority: str) -> List[Dict]:
        """
        Filter rules by priority level.
        
        Args:
            priority: Priority level ('Critical', 'High', 'Medium', 'Low')
            
        Returns:
            List of rules matching the priority
        """
        return [rule for rule in self.rules if rule['priority'] == priority]
    
    def get_rules_by_cluster(self, cluster_id: int) -> List[Dict]:
        """
        Get all rules for a specific cluster.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            List of rules for the cluster
        """
        return [rule for rule in self.rules if rule['cluster_id'] == cluster_id]


def generate_optimization_rules(cluster_profiles: pd.DataFrame,
                                timeseries_data: pd.DataFrame,
                                cluster_assignments: pd.DataFrame,
                                export_path: str = None) -> pd.DataFrame:
    """
    Complete rules generation pipeline - convenience function.
    
    Args:
        cluster_profiles: Cluster profile statistics
        timeseries_data: Timeseries traffic data
        cluster_assignments: Junction-to-cluster mappings
        export_path: Optional path to export rules as CSV
        
    Returns:
        DataFrame with all generated rules
    """
    generator = TrafficRulesGenerator()
    generator.generate_cluster_rules(cluster_profiles, timeseries_data, cluster_assignments)
    generator.print_rules_summary()
    
    rules_df = generator.export_rules_to_dataframe()
    
    if export_path:
        generator.export_rules_to_csv(export_path)
    
    return rules_df


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import preprocess_pipeline
    from clustering import clustering_pipeline
    
    data_path = "../data/traffic.csv"
    timeseries, junction_features = preprocess_pipeline(data_path)
    cluster_assignments, cluster_profiles = clustering_pipeline(junction_features, timeseries)
    
    # Generate rules
    rules_df = generate_optimization_rules(
        cluster_profiles,
        timeseries,
        cluster_assignments,
        export_path="../output/traffic_rules.csv"
    )
    
    print(f"\n\nTotal rules generated: {len(rules_df)}")
