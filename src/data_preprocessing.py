"""
Data Preprocessing Module for Urban Traffic Flow Analysis

This module handles:
- Loading raw traffic data
- Data cleaning and validation
- Feature engineering (temporal features, statistical aggregates)
- Data transformation for ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class TrafficDataPreprocessor:
    """
    Preprocesses traffic data for analysis and machine learning.
    
    Handles data loading, cleaning, feature extraction, and transformation
    to prepare traffic data for clustering and pattern analysis.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the preprocessor with data path.
        
        Args:
            data_path: Path to the traffic CSV file
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.junction_features = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load traffic data from CSV file.
        
        Returns:
            DataFrame with loaded traffic data
        """
        print("Loading traffic data...")
        self.raw_data = pd.read_csv(self.data_path)
        
        # Convert DateTime column to datetime type for temporal analysis
        self.raw_data['DateTime'] = pd.to_datetime(self.raw_data['DateTime'])
        
        # Drop ID column as it's not needed for analysis
        if 'ID' in self.raw_data.columns:
            self.raw_data = self.raw_data.drop('ID', axis=1)
        
        print(f"Loaded {len(self.raw_data)} records from {self.raw_data['Junction'].nunique()} junctions")
        print(f"Date range: {self.raw_data['DateTime'].min()} to {self.raw_data['DateTime'].max()}")
        
        return self.raw_data
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the loaded data by handling missing values and outliers.
        
        Returns:
            Cleaned DataFrame
        """
        print("\nCleaning data...")
        df = self.raw_data.copy()
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values - filling with interpolation")
            # Use forward fill then backward fill for any remaining nulls
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any duplicate entries (same junction at same time)
        duplicates = df.duplicated(subset=['DateTime', 'Junction']).sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate records")
            df = df.drop_duplicates(subset=['DateTime', 'Junction'], keep='first')
        
        # Detect and handle outliers using IQR method
        # Outliers are vehicle counts significantly outside normal range
        Q1 = df['Vehicles'].quantile(0.25)
        Q3 = df['Vehicles'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # Using 3*IQR for less aggressive outlier removal
        upper_bound = Q3 + 3 * IQR
        
        outliers = ((df['Vehicles'] < lower_bound) | (df['Vehicles'] > upper_bound)).sum()
        if outliers > 0:
            print(f"Found {outliers} outliers - capping to bounds")
            df['Vehicles'] = df['Vehicles'].clip(lower=lower_bound, upper=upper_bound)
        
        self.processed_data = df
        print("Data cleaning completed")
        
        return self.processed_data
    
    def extract_temporal_features(self) -> pd.DataFrame:
        """
        Extract temporal features from DateTime column.
        
        Features include:
        - Hour of day (0-23): Captures daily traffic patterns
        - Day of week (0-6): Identifies weekday vs weekend patterns
        - Month: Seasonal variations
        - Is weekend: Binary flag for weekend days
        
        Returns:
            DataFrame with temporal features added
        """
        print("\nExtracting temporal features...")
        df = self.processed_data.copy()
        
        # Extract hour (0-23) to identify peak/off-peak hours
        df['Hour'] = df['DateTime'].dt.hour
        
        # Extract day of week (0=Monday, 6=Sunday) for weekday/weekend patterns
        df['DayOfWeek'] = df['DateTime'].dt.dayofweek
        
        # Extract month for seasonal analysis
        df['Month'] = df['DateTime'].dt.month
        
        # Binary feature for weekend (1) vs weekday (0)
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Create period of day categories for better interpretation
        # Early Morning: 0-6, Morning Rush: 7-9, Midday: 10-16, 
        # Evening Rush: 17-19, Night: 20-23
        df['PeriodOfDay'] = pd.cut(
            df['Hour'], 
            bins=[-1, 6, 9, 16, 19, 23],
            labels=['Early Morning', 'Morning Rush', 'Midday', 'Evening Rush', 'Night']
        )
        
        self.processed_data = df
        print("Temporal features extracted")
        
        return self.processed_data
    
    def create_junction_features(self) -> pd.DataFrame:
        """
        Aggregate data per junction to create features for clustering.
        
        Creates statistical features for each junction:
        - Average traffic by hour
        - Peak hour traffic
        - Variability metrics (std deviation)
        - Weekend vs weekday patterns
        
        Returns:
            DataFrame with one row per junction containing aggregated features
        """
        print("\nCreating junction-level features for clustering...")
        df = self.processed_data.copy()
        
        # Calculate average vehicles per junction for each hour of day
        # This creates a traffic profile for each junction across 24 hours
        hourly_avg = df.groupby(['Junction', 'Hour'])['Vehicles'].mean().unstack(fill_value=0)
        hourly_avg.columns = [f'Hour_{h}_Avg' for h in hourly_avg.columns]
        
        # Calculate overall statistics per junction
        junction_stats = df.groupby('Junction').agg({
            'Vehicles': ['mean', 'std', 'min', 'max', 'median']
        }).round(2)
        junction_stats.columns = ['_'.join(col).strip() for col in junction_stats.columns]
        
        # Calculate peak hour (hour with maximum average traffic) for each junction
        peak_hours = df.groupby(['Junction', 'Hour'])['Vehicles'].mean().reset_index()
        peak_hours = peak_hours.loc[peak_hours.groupby('Junction')['Vehicles'].idxmax()]
        peak_hours = peak_hours.set_index('Junction')[['Hour', 'Vehicles']]
        peak_hours.columns = ['PeakHour', 'PeakHourTraffic']
        
        # Calculate weekend vs weekday average traffic
        weekend_avg = df[df['IsWeekend'] == 1].groupby('Junction')['Vehicles'].mean()
        weekday_avg = df[df['IsWeekend'] == 0].groupby('Junction')['Vehicles'].mean()
        weekend_comparison = pd.DataFrame({
            'Weekend_Avg': weekend_avg,
            'Weekday_Avg': weekday_avg,
            'Weekend_Weekday_Ratio': (weekend_avg / weekday_avg).round(2)
        })
        
        # Combine all features
        self.junction_features = pd.concat([
            hourly_avg,
            junction_stats,
            peak_hours,
            weekend_comparison
        ], axis=1)
        
        # Fill any missing values with 0 (for junctions with missing weekend/weekday data)
        self.junction_features = self.junction_features.fillna(0)
        
        print(f"Created {self.junction_features.shape[1]} features for {self.junction_features.shape[0]} junctions")
        
        return self.junction_features
    
    def get_processed_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get all processed datasets.
        
        Returns:
            Dictionary containing:
            - 'timeseries': Processed time-series data with temporal features
            - 'junction_features': Aggregated features per junction for clustering
        """
        return {
            'timeseries': self.processed_data,
            'junction_features': self.junction_features
        }
    
    def get_data_summary(self) -> Dict:
        """
        Generate summary statistics about the processed data.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.processed_data is None:
            return {"error": "No processed data available"}
        
        return {
            'total_records': len(self.processed_data),
            'num_junctions': self.processed_data['Junction'].nunique(),
            'date_range': {
                'start': str(self.processed_data['DateTime'].min()),
                'end': str(self.processed_data['DateTime'].max()),
                'days': (self.processed_data['DateTime'].max() - 
                        self.processed_data['DateTime'].min()).days
            },
            'vehicle_stats': {
                'mean': round(self.processed_data['Vehicles'].mean(), 2),
                'median': round(self.processed_data['Vehicles'].median(), 2),
                'min': int(self.processed_data['Vehicles'].min()),
                'max': int(self.processed_data['Vehicles'].max())
            }
        }


def preprocess_pipeline(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline - convenience function.
    
    Args:
        data_path: Path to traffic CSV file
        
    Returns:
        Tuple of (timeseries_data, junction_features)
    """
    preprocessor = TrafficDataPreprocessor(data_path)
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.extract_temporal_features()
    preprocessor.create_junction_features()
    
    data = preprocessor.get_processed_data()
    return data['timeseries'], data['junction_features']


if __name__ == "__main__":
    # Example usage
    data_path = "../data/traffic.csv"
    timeseries, junction_features = preprocess_pipeline(data_path)
    
    print("\n=== Preprocessing Complete ===")
    print(f"\nTimeseries data shape: {timeseries.shape}")
    print(f"Junction features shape: {junction_features.shape}")
    print(f"\nFirst few rows of junction features:")
    print(junction_features.head())
