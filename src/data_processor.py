"""
Phase 1: Data Ingestion
Urban Traffic Flow Analysis - Data Processor

This module handles:
- Loading raw traffic CSV data
- Parsing datetime columns
- Extracting temporal features (Hour, DayOfWeek, Month, IsWeekend)
- Cleaning duplicate records by summing vehicle counts
- Displaying summary statistics
"""

import pandas as pd
import numpy as np
import os


class DataProcessor:
    """
    Processes raw traffic data for analysis.
    
    Handles data loading, datetime parsing, feature extraction, and cleaning.
    """
    
    def __init__(self, file_path):
        """
        Initialize the data processor.
        
        Args:
            file_path: Path to the traffic CSV file
        """
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        """
        Load the traffic CSV file.
        
        Reads CSV and stores in dataframe.
        """
        print("=" * 70)
        print("PHASE 1: DATA INGESTION")
        print("=" * 70)
        print(f"\nLoading data from: {self.file_path}")
        
        # Read CSV file
        self.df = pd.read_csv(self.file_path)
        
        print(f"Data loaded successfully!")
        print(f"Total records: {len(self.df):,}")
        print(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def parse_datetime(self):
        """
        Convert DateTime column to pandas datetime object.
        
        This enables temporal operations and feature extraction.
        """
        print("\n" + "-" * 70)
        print("Parsing DateTime column...")
        
        # Convert DateTime string to datetime object
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        
        print(f"DateTime parsed successfully!")
        print(f"Date range: {self.df['DateTime'].min()} to {self.df['DateTime'].max()}")
        
        return self.df
    
    def extract_features(self):
        """
        Extract temporal features from DateTime column.
        
        Creates new columns:
        - Hour: Hour of day (0-23)
        - DayOfWeek: Day of week (0=Monday, 6=Sunday)
        - Month: Month (1-12)
        - IsWeekend: Binary flag (1 if Saturday/Sunday, 0 otherwise)
        """
        print("\n" + "-" * 70)
        print("Extracting temporal features...")
        
        # Extract Hour (0-23) from DateTime
        self.df['Hour'] = self.df['DateTime'].dt.hour
        
        # Extract DayOfWeek (0=Monday, 6=Sunday) from DateTime
        self.df['DayOfWeek'] = self.df['DateTime'].dt.dayofweek
        
        # Extract Month (1-12) from DateTime
        self.df['Month'] = self.df['DateTime'].dt.month
        
        # Create IsWeekend flag (1 if Saturday or Sunday, else 0)
        # Saturday=5, Sunday=6 in dayofweek
        self.df['IsWeekend'] = (self.df['DayOfWeek'] >= 5).astype(int)
        
        print("Features extracted successfully!")
        print(f"Created columns: Hour, DayOfWeek, Month, IsWeekend")
        print(f"\nFeature Statistics:")
        print(f"  - Unique Hours: {self.df['Hour'].nunique()}")
        print(f"  - Unique Days of Week: {self.df['DayOfWeek'].nunique()}")
        print(f"  - Unique Months: {self.df['Month'].nunique()}")
        print(f"  - Weekend Records: {self.df['IsWeekend'].sum():,} ({self.df['IsWeekend'].sum()/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def clean_duplicates(self):
        """
        Check for and handle duplicate records.
        
        If multiple records exist for the same Junction-Time combination,
        sum the Vehicle counts to combine them into a single record.
        """
        print("\n" + "-" * 70)
        print("Checking for duplicate Junction-Time combinations...")
        
        # Check for duplicates based on Junction and DateTime
        duplicate_count = self.df.duplicated(subset=['Junction', 'DateTime']).sum()
        
        if duplicate_count > 0:
            print(f"Found {duplicate_count:,} duplicate records")
            print("Combining duplicates by summing Vehicle counts...")
            
            # Group by Junction and DateTime, summing Vehicles for duplicates
            # Keep first occurrence of other columns
            self.df = self.df.groupby(['Junction', 'DateTime'], as_index=False).agg({
                'Vehicles': 'sum',  # Sum vehicle counts for duplicates
                'ID': 'first',      # Keep first ID
                'Hour': 'first',    # Keep temporal features (they're the same)
                'DayOfWeek': 'first',
                'Month': 'first',
                'IsWeekend': 'first'
            })
            
            print(f"Duplicates combined successfully!")
            print(f"Records after cleaning: {len(self.df):,}")
        else:
            print("No duplicate records found - data is clean!")
        
        return self.df
    
    def display_summary(self):
        """
        Display summary information about the processed data.
        
        Shows:
        - First 5 rows of the dataframe
        - Total vehicles by junction (sorted by traffic volume)
        """
        print("\n" + "=" * 70)
        print("DATA SUMMARY")
        print("=" * 70)
        
        # Display first 5 rows
        print("\nFirst 5 rows of processed data:")
        print("-" * 70)
        print(self.df.head())
        
        # Calculate and display total vehicles by junction
        print("\n" + "=" * 70)
        print("TRAFFIC SUMMARY BY JUNCTION")
        print("=" * 70)
        
        junction_summary = self.df.groupby('Junction').agg({
            'Vehicles': ['sum', 'mean', 'max']
        }).round(2)
        
        # Flatten column names
        junction_summary.columns = ['Total_Vehicles', 'Avg_Vehicles', 'Max_Vehicles']
        
        # Sort by total vehicles (descending)
        junction_summary = junction_summary.sort_values('Total_Vehicles', ascending=False)
        
        # Reset index to make Junction a column
        junction_summary = junction_summary.reset_index()
        
        print(f"\nTotal Junctions: {len(junction_summary)}")
        print("\nTop 10 Junctions by Traffic Volume:")
        print("-" * 70)
        print(junction_summary.head(10).to_string(index=False))
        
        print("\n" + "=" * 70)
        print("Overall Statistics:")
        print(f"  - Total Records: {len(self.df):,}")
        print(f"  - Total Vehicles (all junctions): {self.df['Vehicles'].sum():,}")
        print(f"  - Average Vehicles per Record: {self.df['Vehicles'].mean():.2f}")
        print(f"  - Date Range: {self.df['DateTime'].min().date()} to {self.df['DateTime'].max().date()}")
        print("=" * 70)
        
        return junction_summary
    
    def get_processed_data(self):
        """
        Get the processed dataframe.
        
        Returns:
            Processed pandas DataFrame
        """
        return self.df
    
    def save_processed_data(self, output_path):
        """
        Save the processed data to CSV.
        
        Args:
            output_path: Path to save the processed CSV file
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nCreated output directory: {output_dir}")
        
        self.df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")


def process_traffic_data(file_path, save_output=False, output_path=None):
    """
    Complete data ingestion pipeline.
    
    Args:
        file_path: Path to input CSV file
        save_output: Whether to save processed data
        output_path: Path to save output (if save_output=True)
        
    Returns:
        Tuple of (processed_dataframe, junction_summary)
    """
    # Initialize processor
    processor = DataProcessor(file_path)
    
    # Step 1: Load data
    processor.load_data()
    
    # Step 2: Parse datetime
    processor.parse_datetime()
    
    # Step 3: Extract features
    processor.extract_features()
    
    # Step 4: Clean duplicates
    processor.clean_duplicates()
    
    # Step 5: Display summary
    junction_summary = processor.display_summary()
    
    # Optional: Save processed data
    if save_output and output_path:
        processor.save_processed_data(output_path)
    
    return processor.get_processed_data(), junction_summary


if __name__ == "__main__":
    # Run Phase 1: Data Ingestion
    input_file = "../data/traffic.csv"
    output_file = "../output/data/processed_traffic_phase1.csv"
    
    # Process the data
    df, summary = process_traffic_data(
        file_path=input_file,
        save_output=True,
        output_path=output_file
    )
    
    print("\n✓ Phase 1: Data Ingestion Complete!")
    print(f"\nProcessed dataframe shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
