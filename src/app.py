"""
Streamlit Dashboard for Urban Traffic Flow Analysis & Optimization System

Interactive web-based dashboard for:
- Exploring traffic patterns
- Viewing cluster analysis results
- Examining traffic optimization rules
- Comparing junctions and time periods
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Import custom modules
from data_preprocessing import TrafficDataPreprocessor
from clustering import TrafficClusterAnalyzer
from visualization import TrafficVisualizer
from rules_generator import TrafficRulesGenerator

# Page configuration
st.set_page_config(
    page_title="Urban Traffic Analysis Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #ff7f0e;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    """
    Load processed data and analysis results.
    
    Returns:
        Dictionary with all data
    """
    try:
        # Check if processed data exists
        if os.path.exists('../output/data/processed_timeseries.csv'):
            timeseries = pd.read_csv('../output/data/processed_timeseries.csv')
            timeseries['DateTime'] = pd.to_datetime(timeseries['DateTime'])
            junction_features = pd.read_csv('../output/data/junction_features.csv', index_col=0)
            cluster_assignments = pd.read_csv('../output/data/cluster_assignments.csv')
            cluster_profiles = pd.read_csv('../output/data/cluster_profiles.csv')
            rules = pd.read_csv('../output/traffic_optimization_rules.csv')
            
            return {
                'timeseries': timeseries,
                'junction_features': junction_features,
                'cluster_assignments': cluster_assignments,
                'cluster_profiles': cluster_profiles,
                'rules': rules,
                'loaded': True
            }
        else:
            return {'loaded': False}
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {'loaded': False}


def show_overview(data):
    """Display overview page with key metrics."""
    st.title("🚦 Urban Traffic Flow Analysis Dashboard")
    st.markdown("---")
    
    # Key Metrics
    st.header("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_junctions = data['cluster_assignments']['Junction'].nunique()
        st.metric("Total Junctions", total_junctions)
    
    with col2:
        total_records = len(data['timeseries'])
        st.metric("Total Records", f"{total_records:,}")
    
    with col3:
        num_clusters = data['cluster_profiles']['Cluster'].nunique()
        st.metric("Number of Clusters", num_clusters)
    
    with col4:
        num_rules = len(data['rules'])
        st.metric("Optimization Rules", num_rules)
    
    st.markdown("---")
    
    # Cluster Overview
    st.header("🎯 Cluster Overview")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Cluster Profiles")
        # Display cluster profiles table
        display_profiles = data['cluster_profiles'][['Cluster', 'CongestionLevel', 
                                                     'NumJunctions', 'AvgTraffic', 
                                                     'PeakHour']].copy()
        display_profiles.columns = ['Cluster', 'Congestion Level', 'Junctions', 
                                   'Avg Traffic', 'Peak Hour']
        display_profiles['Avg Traffic'] = display_profiles['Avg Traffic'].round(1)
        display_profiles['Peak Hour'] = display_profiles['Peak Hour'].astype(int).astype(str) + ':00'
        st.dataframe(display_profiles, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("Junction Distribution")
        # Pie chart of cluster distribution
        fig, ax = plt.subplots(figsize=(6, 6))
        cluster_counts = data['cluster_assignments']['Cluster'].value_counts().sort_index()
        labels = [f"Cluster {i}" for i in cluster_counts.index]
        colors = sns.color_palette("husl", len(cluster_counts))
        ax.pie(cluster_counts.values, labels=labels, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax.set_title('Junction Distribution Across Clusters')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Traffic Statistics
    st.header("📈 Traffic Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Traffic by Hour")
        hourly_avg = data['timeseries'].groupby('Hour')['Vehicles'].mean()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hourly_avg.index, hourly_avg.values, marker='o', 
               linewidth=2.5, markersize=6, color='#1f77b4')
        ax.set_xlabel('Hour of Day', fontweight='bold')
        ax.set_ylabel('Average Vehicles', fontweight='bold')
        ax.set_title('Daily Traffic Pattern', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(24))
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Weekday vs Weekend Traffic")
        weekday_avg = data['timeseries'][data['timeseries']['IsWeekend'] == 0].groupby('Hour')['Vehicles'].mean()
        weekend_avg = data['timeseries'][data['timeseries']['IsWeekend'] == 1].groupby('Hour')['Vehicles'].mean()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(weekday_avg.index, weekday_avg.values, marker='o', 
               linewidth=2.5, label='Weekday', color='#1f77b4')
        ax.plot(weekend_avg.index, weekend_avg.values, marker='s', 
               linewidth=2.5, label='Weekend', color='#ff7f0e')
        ax.set_xlabel('Hour of Day', fontweight='bold')
        ax.set_ylabel('Average Vehicles', fontweight='bold')
        ax.set_title('Weekday vs Weekend Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(24))
        st.pyplot(fig)
        plt.close()


def show_cluster_analysis(data):
    """Display cluster analysis page."""
    st.title("🎯 Cluster Analysis")
    st.markdown("---")
    
    # Cluster selection
    clusters = sorted(data['cluster_profiles']['Cluster'].unique())
    selected_cluster = st.selectbox("Select Cluster to Analyze", clusters)
    
    # Get cluster information
    cluster_info = data['cluster_profiles'][data['cluster_profiles']['Cluster'] == selected_cluster].iloc[0]
    cluster_junctions = data['cluster_assignments'][
        data['cluster_assignments']['Cluster'] == selected_cluster
    ]['Junction'].tolist()
    
    # Display cluster info
    st.header(f"Cluster {selected_cluster} - {cluster_info['CongestionLevel']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Number of Junctions", int(cluster_info['NumJunctions']))
    
    with col2:
        st.metric("Average Traffic", f"{cluster_info['AvgTraffic']:.1f} vehicles/hour")
    
    with col3:
        st.metric("Peak Hour", f"{int(cluster_info['PeakHour'])}:00")
    
    with col4:
        st.metric("Variability (σ)", f"{cluster_info['AvgStdDev']:.1f}")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Traffic Pattern")
        # Get hourly pattern for this cluster
        cluster_data = data['timeseries'][data['timeseries']['Junction'].isin(cluster_junctions)]
        hourly_pattern = cluster_data.groupby('Hour')['Vehicles'].mean()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hourly_pattern.index, hourly_pattern.values, 
               marker='o', linewidth=2.5, markersize=6, color='#ff7f0e')
        ax.set_xlabel('Hour of Day', fontweight='bold')
        ax.set_ylabel('Average Vehicles', fontweight='bold')
        ax.set_title(f'Cluster {selected_cluster} - Hourly Pattern', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(24))
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Traffic Heatmap")
        # Create heatmap for this cluster
        heatmap_data = cluster_data.groupby(['Hour', 'DayOfWeek'])['Vehicles'].mean().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.0f', 
                   cbar_kws={'label': 'Avg Vehicles'}, ax=ax)
        ax.set_xlabel('Day of Week', fontweight='bold')
        ax.set_ylabel('Hour of Day', fontweight='bold')
        ax.set_title(f'Cluster {selected_cluster} - Weekly Pattern', fontweight='bold')
        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax.set_xticklabels(day_labels, rotation=0)
        st.pyplot(fig)
        plt.close()
    
    # Junction list
    st.markdown("---")
    st.subheader("Junctions in this Cluster")
    
    junction_cols = st.columns(5)
    for i, junction in enumerate(cluster_junctions):
        with junction_cols[i % 5]:
            st.info(f"Junction {junction}")


def show_traffic_rules(data):
    """Display traffic optimization rules page."""
    st.title("📋 Traffic Optimization Rules")
    st.markdown("---")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        priority_filter = st.multiselect(
            "Filter by Priority",
            options=['Critical', 'High', 'Medium', 'Low'],
            default=['Critical', 'High', 'Medium', 'Low']
        )
    
    with col2:
        cluster_filter = st.multiselect(
            "Filter by Cluster",
            options=sorted(data['rules']['cluster_id'].unique()),
            default=sorted(data['rules']['cluster_id'].unique())
        )
    
    # Apply filters
    filtered_rules = data['rules'][
        (data['rules']['priority'].isin(priority_filter)) &
        (data['rules']['cluster_id'].isin(cluster_filter))
    ]
    
    st.info(f"Showing {len(filtered_rules)} rules (out of {len(data['rules'])} total)")
    
    # Display rules by cluster
    for cluster_id in sorted(filtered_rules['cluster_id'].unique()):
        cluster_rules = filtered_rules[filtered_rules['cluster_id'] == cluster_id]
        cluster_info = data['cluster_profiles'][data['cluster_profiles']['Cluster'] == cluster_id].iloc[0]
        
        st.markdown("---")
        st.header(f"Cluster {cluster_id} - {cluster_info['CongestionLevel']}")
        
        for idx, rule in cluster_rules.iterrows():
            # Color code by priority
            if rule['priority'] == 'Critical':
                st.error(f"**{rule['rule_type']}** [Priority: {rule['priority']}]")
            elif rule['priority'] == 'High':
                st.warning(f"**{rule['rule_type']}** [Priority: {rule['priority']}]")
            else:
                st.info(f"**{rule['rule_type']}** [Priority: {rule['priority']}]")
            
            with st.expander("View Details"):
                st.write(f"**Description:** {rule['description']}")
                st.write(f"**Action:** {rule['specific_action']}")
                st.write(f"**Expected Impact:** {rule['expected_impact']}")
                st.write(f"**Implementation Time:** {rule['implementation_time']}")


def show_junction_explorer(data):
    """Display junction exploration page."""
    st.title("🔍 Junction Explorer")
    st.markdown("---")
    
    # Junction selection
    all_junctions = sorted(data['cluster_assignments']['Junction'].unique())
    selected_junction = st.selectbox("Select Junction", all_junctions)
    
    # Get junction info
    junction_cluster = data['cluster_assignments'][
        data['cluster_assignments']['Junction'] == selected_junction
    ]['Cluster'].values[0]
    
    cluster_info = data['cluster_profiles'][
        data['cluster_profiles']['Cluster'] == junction_cluster
    ].iloc[0]
    
    # Display info
    st.header(f"Junction {selected_junction}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cluster", f"{junction_cluster} ({cluster_info['CongestionLevel']})")
    
    with col2:
        junction_data = data['timeseries'][data['timeseries']['Junction'] == selected_junction]
        avg_traffic = junction_data['Vehicles'].mean()
        st.metric("Average Traffic", f"{avg_traffic:.1f} vehicles/hour")
    
    with col3:
        peak_hour_traffic = junction_data.groupby('Hour')['Vehicles'].mean().max()
        peak_hour = junction_data.groupby('Hour')['Vehicles'].mean().idxmax()
        st.metric("Peak Hour", f"{int(peak_hour)}:00 ({peak_hour_traffic:.1f})")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Traffic Pattern")
        hourly_avg = junction_data.groupby('Hour')['Vehicles'].mean()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hourly_avg.index, hourly_avg.values, 
               marker='o', linewidth=2.5, markersize=6, color='#2ca02c')
        ax.set_xlabel('Hour of Day', fontweight='bold')
        ax.set_ylabel('Average Vehicles', fontweight='bold')
        ax.set_title(f'Junction {selected_junction} - Daily Pattern', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(24))
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Traffic Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(junction_data['Vehicles'], bins=30, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax.axvline(avg_traffic, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_traffic:.1f}')
        ax.set_xlabel('Number of Vehicles', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'Junction {selected_junction} - Traffic Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close()
    
    # Time series
    st.markdown("---")
    st.subheader("Traffic Over Time")
    
    # Allow date range selection
    min_date = junction_data['DateTime'].min().date()
    max_date = junction_data['DateTime'].max().date()
    
    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        filtered_data = junction_data[
            (junction_data['DateTime'].dt.date >= date_range[0]) &
            (junction_data['DateTime'].dt.date <= date_range[1])
        ]
        
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(filtered_data['DateTime'], filtered_data['Vehicles'], 
               linewidth=1.5, color='#2ca02c', alpha=0.7)
        ax.set_xlabel('Date/Time', fontweight='bold')
        ax.set_ylabel('Number of Vehicles', fontweight='bold')
        ax.set_title(f'Junction {selected_junction} - Traffic Timeline', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()


def main():
    """Main application."""
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Cluster Analysis", "Traffic Rules", "Junction Explorer"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Urban Traffic Flow Analysis & Optimization System**\n\n"
        "This dashboard provides interactive visualizations and insights "
        "from traffic pattern analysis and clustering."
    )
    
    # Load data
    data = load_data()
    
    if not data['loaded']:
        st.error("⚠️ No processed data found!")
        st.info(
            "Please run the main analysis first:\n\n"
            "```bash\n"
            "cd src\n"
            "python main.py\n"
            "```"
        )
        st.stop()
    
    # Display selected page
    if page == "Overview":
        show_overview(data)
    elif page == "Cluster Analysis":
        show_cluster_analysis(data)
    elif page == "Traffic Rules":
        show_traffic_rules(data)
    elif page == "Junction Explorer":
        show_junction_explorer(data)


if __name__ == "__main__":
    main()
