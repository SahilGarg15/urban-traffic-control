"""
Phase 5: User Interface
Urban Traffic Flow Analysis - Streamlit Dashboard

Professional web-based interface for traffic analysis and optimization.
Integrates data processing, pattern analysis, ML models, and recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

# Import custom modules
from data_processor import DataProcessor
from eda import TrafficEDA
from ml_models import JunctionClustering, TrafficForecasting
from optimizer import TrafficOptimizer

# Page configuration
st.set_page_config(
    page_title="Urban Traffic Analysis System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #ffffff;
        border-radius: 8px 8px 0px 0px;
        font-weight: 600;
        color: #424242;
        border: 1px solid #e0e0e0;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1976d2;
        color: white;
        border-color: #1976d2;
        box-shadow: 0 -2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .recommendation-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #212121;
    }
    h1 {
        color: #0d47a1;
        border-bottom: 3px solid #1976d2;
        padding-bottom: 10px;
    }
    h2 {
        color: #1a237e;
    }
    h3 {
        color: #263238;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    .stMetric label {
        color: #424242 !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #1976d2 !important;
        font-size: 1.8rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_traffic_data():
    """Load and cache processed traffic data."""
    data_path = "../output/data/processed_traffic_phase1.csv"
    
    if not os.path.exists(data_path):
        st.error("❌ Processed data not found! Please run data_processor.py first.")
        st.stop()
    
    data = pd.read_csv(data_path)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    return data


@st.cache_data
def get_clustering_results(data):
    """Get or compute junction clustering results."""
    clustering = JunctionClustering(data, n_clusters=3)
    clustering.create_junction_profiles()
    clustering.perform_clustering()
    cluster_results = clustering.label_clusters()
    return cluster_results


@st.cache_data
def get_recommendations(cluster_results, data):
    """Generate optimization recommendations."""
    optimizer = TrafficOptimizer()
    
    # Enhance profiles with weekend/weekday data
    enhanced_profiles = cluster_results.copy()
    
    weekend_traffic = []
    weekday_traffic = []
    
    for junction in enhanced_profiles['Junction']:
        junction_data = data[data['Junction'] == junction]
        weekend_avg = junction_data[junction_data['IsWeekend'] == 1]['Vehicles'].mean()
        weekday_avg = junction_data[junction_data['IsWeekend'] == 0]['Vehicles'].mean()
        
        weekend_traffic.append(weekend_avg if not pd.isna(weekend_avg) else 0)
        weekday_traffic.append(weekday_avg if not pd.isna(weekday_avg) else 0)
    
    enhanced_profiles['Weekend_Traffic'] = weekend_traffic
    enhanced_profiles['Weekday_Traffic'] = weekday_traffic
    
    recommendations_df = optimizer.generate_batch_recommendations(enhanced_profiles)
    
    return recommendations_df


def predict_next_24_hours(data, junction_id):
    """Predict traffic for next 24 hours using trained model."""
    try:
        forecaster = TrafficForecasting(data, junction_id=junction_id)
        forecaster.prepare_data()
        forecaster.create_lag_features()
        forecaster.train_model()
        
        # Get the last available data point
        last_data = forecaster.junction_data.tail(30).copy()  # Get last 30 hours for context
        
        if len(last_data) < 24:
            return None
        
        # Predict next 24 hours
        predictions = []
        
        # Start from the most recent actual data
        current_vehicles = last_data.iloc[-1]['Vehicles']
        current_hour = int(last_data.iloc[-1]['Hour'])
        current_dow = int(last_data.iloc[-1]['DayOfWeek'])
        
        for i in range(24):
            # Calculate next hour and day of week
            next_hour = (current_hour + i + 1) % 24
            next_dow = current_dow if next_hour != 0 else (current_dow + 1) % 7
            next_is_weekend = 1 if next_dow >= 5 else 0
            
            # Get lag features
            # Traffic_Last_Hour: Use previous prediction or actual data
            if i == 0:
                traffic_last_hour = current_vehicles
            else:
                traffic_last_hour = predictions[-1]
            
            # Traffic_Yesterday_Same_Hour: Use data from 24 hours ago
            if len(last_data) >= (24 - i):
                yesterday_idx = -(24 - i)
                traffic_yesterday = last_data.iloc[yesterday_idx]['Vehicles']
            else:
                traffic_yesterday = current_vehicles
            
            # Prepare features
            next_features = pd.DataFrame({
                'Hour': [next_hour],
                'DayOfWeek': [next_dow],
                'IsWeekend': [next_is_weekend],
                'Traffic_Last_Hour': [traffic_last_hour],
                'Traffic_Yesterday_Same_Hour': [traffic_yesterday]
            })
            
            # Make prediction
            pred = forecaster.model.predict(next_features)[0]
            predictions.append(max(0, pred))  # Ensure non-negative
        
        return predictions
    
    except Exception as e:
        st.warning(f"Prediction unavailable: {str(e)}")
        return None


def plot_hourly_pattern_single_junction(data, junction_id):
    """Plot hourly pattern for a single junction."""
    junction_data = data[data['Junction'] == junction_id]
    hourly_avg = junction_data.groupby('Hour')['Vehicles'].mean()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(hourly_avg.index, hourly_avg.values, 
           marker='o', linewidth=3, markersize=8, 
           color='#1f77b4', label=f'Junction {junction_id}')
    
    # Highlight rush hours
    ax.axvspan(7, 9, alpha=0.1, color='red', label='Morning Rush')
    ax.axvspan(17, 19, alpha=0.1, color='orange', label='Evening Rush')
    
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Vehicles', fontsize=12, fontweight='bold')
    ax.set_title(f'Hourly Traffic Pattern - Junction {junction_id}', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(24))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_weekly_pattern_single_junction(data, junction_id):
    """Plot weekly pattern for a single junction."""
    junction_data = data[data['Junction'] == junction_id]
    weekly_avg = junction_data.groupby('DayOfWeek')['Vehicles'].mean()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    colors = ['#3498db' if day < 5 else '#e74c3c' for day in range(7)]
    
    bars = ax.bar(range(7), weekly_avg.values, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Vehicles', fontsize=12, fontweight='bold')
    ax.set_title(f'Weekly Traffic Pattern - Junction {junction_id}', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_24hour_forecast(predictions):
    """Plot 24-hour traffic forecast."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    hours = list(range(24))
    
    # Plot with professional blue color scheme
    ax.plot(hours, predictions, marker='o', linewidth=3, markersize=8,
           color='#1976d2', label='Predicted Traffic', zorder=3)
    
    # Fill area under curve with light blue
    ax.fill_between(hours, predictions, alpha=0.2, color='#2196f3')
    
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Vehicles', fontsize=12, fontweight='bold')
    ax.set_title('24-Hour Traffic Forecast (Next Day)', fontsize=14, fontweight='bold', color='#1a237e')
    ax.set_xticks(range(24))
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🚦 Urban Traffic Flow Analysis & Optimization System")
    st.markdown("**Final Year Project - Intelligent Traffic Management**")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading traffic data..."):
        data = load_traffic_data()
        cluster_results = get_clustering_results(data)
        recommendations = get_recommendations(cluster_results, data)
    
    # Sidebar - Junction Selection
    st.sidebar.header("📍 Junction Selection")
    st.sidebar.markdown("Select a junction to analyze")
    
    available_junctions = sorted(data['Junction'].unique())
    selected_junction = st.sidebar.selectbox(
        "Choose Junction:",
        available_junctions,
        format_func=lambda x: f"Junction {x}"
    )
    
    # Get junction information
    junction_info = cluster_results[cluster_results['Junction'] == selected_junction].iloc[0]
    junction_recs = recommendations[recommendations['Junction'] == selected_junction].iloc[0]
    
    # Sidebar - Junction Stats
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Quick Stats")
    st.sidebar.metric("Average Traffic", f"{junction_info['Avg_Vehicles']:.1f} vehicles/hour")
    st.sidebar.metric("Peak Traffic", f"{junction_info['Max_Vehicles']:.0f} vehicles")
    st.sidebar.metric("Morning Peak (8-10am)", f"{junction_info['Morning_Peak']:.1f} vehicles")
    st.sidebar.metric("Evening Peak (5-7pm)", f"{junction_info['Evening_Peak']:.1f} vehicles")
    
    # Main Header - Junction Category
    congestion_level = junction_info['Congestion_Level']
    
    # Color code based on congestion with proper text contrast
    if congestion_level == 'High Congestion':
        header_color = '#d32f2f'  # Dark Red
        text_color = 'white'
        emoji = '🔴'
    elif congestion_level == 'Medium Traffic':
        header_color = '#f57c00'  # Dark Orange
        text_color = 'white'
        emoji = '🟠'
    else:
        header_color = '#388e3c'  # Dark Green
        text_color = 'white'
        emoji = '🟢'
    
    st.markdown(f"""
        <div style='background-color: {header_color}; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h2 style='color: {text_color}; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);'>{emoji} Junction {selected_junction} - {congestion_level}</h2>
            <p style='color: {text_color}; margin: 5px 0 0 0; font-size: 16px; opacity: 0.95;'>
                Cluster-based classification using K-Means algorithm
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2 = st.tabs(["📈 Traffic Patterns", "🎯 Optimization Report"])
    
    # ============================================
    # TAB 1: TRAFFIC PATTERNS
    # ============================================
    with tab1:
        st.header("Traffic Pattern Analysis")
        st.markdown("Historical traffic patterns for selected junction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Hourly Pattern")
            hourly_fig = plot_hourly_pattern_single_junction(data, selected_junction)
            st.pyplot(hourly_fig)
            plt.close()
            
            peak_val = junction_info['Morning_Peak'] if junction_info['Morning_Peak'] > junction_info['Evening_Peak'] else junction_info['Evening_Peak']
            st.info(f"**Peak Hour:** {int(peak_val)} vehicles during rush hours")
        
        with col2:
            st.subheader("📆 Weekly Pattern")
            weekly_fig = plot_weekly_pattern_single_junction(data, selected_junction)
            st.pyplot(weekly_fig)
            plt.close()
            
            # Calculate weekday vs weekend
            junction_data = data[data['Junction'] == selected_junction]
            weekday_avg = junction_data[junction_data['IsWeekend'] == 0]['Vehicles'].mean()
            weekend_avg = junction_data[junction_data['IsWeekend'] == 1]['Vehicles'].mean()
            
            if weekend_avg > weekday_avg:
                st.info(f"🏖️ **Weekend Traffic Higher:** {weekend_avg:.1f} vs {weekday_avg:.1f} vehicles (likely leisure zone)")
            else:
                st.info(f"💼 **Weekday Traffic Higher:** {weekday_avg:.1f} vs {weekend_avg:.1f} vehicles (likely commuter route)")
        
        # Additional Statistics
        st.markdown("---")
        st.subheader("📊 Detailed Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(junction_data):,}")
        
        with col2:
            st.metric("Average Traffic", f"{junction_data['Vehicles'].mean():.1f}")
        
        with col3:
            st.metric("Standard Deviation", f"{junction_data['Vehicles'].std():.1f}")
        
        with col4:
            st.metric("Peak Traffic", f"{junction_data['Vehicles'].max():.0f}")
    
    # ============================================
    # TAB 2: OPTIMIZATION REPORT
    # ============================================
    with tab2:
        st.header("Traffic Optimization Report")
        st.markdown("AI-powered predictions and recommendations")
        
        # 24-Hour Forecast
        st.subheader("🔮 24-Hour Traffic Forecast")
        
        with st.spinner("Generating predictions..."):
            predictions = predict_next_24_hours(data, selected_junction)
        
        if predictions:
            forecast_fig = plot_24hour_forecast(predictions)
            st.pyplot(forecast_fig)
            plt.close()
            
            # Forecast insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_pred = np.mean(predictions)
                st.metric("Avg Predicted Traffic", f"{avg_pred:.1f} vehicles")
            
            with col2:
                peak_hour = np.argmax(predictions)
                st.metric("Predicted Peak Hour", f"{peak_hour}:00")
            
            with col3:
                peak_traffic = np.max(predictions)
                st.metric("Predicted Peak Traffic", f"{peak_traffic:.1f} vehicles")
        else:
            st.warning("⚠️ Unable to generate forecast. Need more historical data.")
        
        # AI Recommendations
        st.markdown("---")
        st.subheader("🤖 AI-Generated Recommendations")
        st.markdown(f"**Based on:** {congestion_level} classification and traffic patterns")
        
        recs = junction_recs['Recommendations']
        
        if recs and len(recs) > 0:
            for i, rec in enumerate(recs, 1):
                st.markdown(f"""
                    <div class='recommendation-box'>
                        <strong>Recommendation {i}:</strong><br>
                        {rec}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("✅ No specific recommendations - traffic patterns are normal and well-managed.")
        
        # Download Recommendations
        st.markdown("---")
        st.subheader("📥 Export Report")
        
        report_data = {
            'Junction': [selected_junction],
            'Congestion_Level': [congestion_level],
            'Avg_Traffic': [junction_info['Avg_Vehicles']],
            'Morning_Peak': [junction_info['Morning_Peak']],
            'Evening_Peak': [junction_info['Evening_Peak']],
            'Recommendations': [' | '.join(recs) if recs else 'None']
        }
        
        report_df = pd.DataFrame(report_data)
        
        csv = report_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Report as CSV",
            data=csv,
            file_name=f"junction_{selected_junction}_report.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
            <p><strong>Urban Traffic Flow Analysis & Optimization System</strong></p>
            <p>Final Year Project | Computer Science | 2026</p>
            <p>Powered by K-Means Clustering, Random Forest, and Rule-based AI</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
