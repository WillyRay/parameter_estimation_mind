#!/usr/bin/env python3
"""
Test script to validate the training dataset format.
"""

import pandas as pd
import numpy as np

def test_training_dataset(file_path):
    """Test that the training dataset has the correct format."""
    print(f"Testing training dataset: {file_path}")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Test basic structure
    print(f"✓ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Check required columns
    required_cols = ['run', 'start_day', 'decayRate', 'surfaceTransferFraction']
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    print(f"✓ Required metadata columns present: {required_cols}")
    
    # Check time series columns (4 variables × 56 days = 224 columns)
    time_series_cols = []
    for var in ['count', 'CDIFF', 'occupancy', 'anyCP']:
        for day in range(56):
            col_name = f"{var}_{day}"
            time_series_cols.append(col_name)
            assert col_name in df.columns, f"Missing time series column: {col_name}"
    
    print(f"✓ All {len(time_series_cols)} time series columns present")
    
    # Test that we have the expected number of columns
    expected_cols = len(required_cols) + len(time_series_cols)  # 4 + 224 = 228
    assert len(df.columns) == expected_cols, f"Expected {expected_cols} columns, got {len(df.columns)}"
    print(f"✓ Total column count correct: {len(df.columns)}")
    
    # Test that we have data from multiple runs
    unique_runs = df['run'].nunique()
    print(f"✓ Data from {unique_runs} unique runs")
    assert unique_runs >= 20, f"Expected at least 20 runs, got {unique_runs}"
    
    # Test that each run has multiple sequences
    sequences_per_run = df.groupby('run').size()
    min_sequences = sequences_per_run.min()
    max_sequences = sequences_per_run.max()
    print(f"✓ Sequences per run: min={min_sequences}, max={max_sequences}")
    assert min_sequences >= 200, f"Expected at least 200 sequences per run, got {min_sequences}"
    
    # Test that parameters are constant within each run
    for run_id in df['run'].unique()[:5]:  # Test first 5 runs
        run_data = df[df['run'] == run_id]
        decay_rates = run_data['decayRate'].unique()
        surface_fractions = run_data['surfaceTransferFraction'].unique()
        
        assert len(decay_rates) == 1, f"Run {run_id} has multiple decayRate values: {decay_rates}"
        assert len(surface_fractions) == 1, f"Run {run_id} has multiple surfaceTransferFraction values: {surface_fractions}"
    
    print(f"✓ Parameters are constant within each run")
    
    # Test that start_day values are sequential
    for run_id in df['run'].unique()[:3]:  # Test first 3 runs
        run_data = df[df['run'] == run_id].sort_values('start_day')
        start_days = run_data['start_day'].values
        expected_start_days = np.arange(len(start_days))
        
        assert np.array_equal(start_days, expected_start_days), f"Run {run_id} has non-sequential start_day values"
    
    print(f"✓ Start day values are sequential")
    
    # Test that we have no missing values in critical columns
    for col in ['decayRate', 'surfaceTransferFraction'] + time_series_cols[:10]:  # Test first 10 time series cols
        missing_count = df[col].isna().sum()
        assert missing_count == 0, f"Column {col} has {missing_count} missing values"
    
    print(f"✓ No missing values in critical columns")
    
    # Show some statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(df):,}")
    print(f"Unique runs: {df['run'].nunique()}")
    print(f"DecayRate range: {df['decayRate'].min():.4f} to {df['decayRate'].max():.4f}")
    print(f"SurfaceTransferFraction range: {df['surfaceTransferFraction'].min():.4f} to {df['surfaceTransferFraction'].max():.4f}")
    
    # Show sample of each time series variable
    print("\n=== Sample Time Series Values ===")
    sample_row = df.iloc[0]
    for var in ['count', 'CDIFF', 'occupancy', 'anyCP']:
        values = [sample_row[f"{var}_{i}"] for i in range(5)]  # First 5 days
        print(f"{var}: {values}...")
    
    print("\n✅ All tests passed! Training dataset is correctly formatted.")
    return True

if __name__ == "__main__":
    test_training_dataset("./data/training_data.csv")