#!/usr/bin/env python3
"""
Verification script to demonstrate how to use the reshaped sim_data.

Created by: GitHub Copilot (first version)

This script loads and examines the reshaped data created by reshape_sim_data.py
"""

import numpy as np
import pandas as pd

def load_and_examine_reshaped_data():
    """Load and examine the reshaped data files."""
    
    print("Loading reshaped data...")
    
    # Load the data
    time_series_list = np.load('data/time_series_by_run.npy', allow_pickle=True)
    targets = np.load('data/targets_by_run.npy')
    run_ids = np.load('data/run_ids.npy')
    metadata = np.load('data/metadata.npy', allow_pickle=True).item()
    
    print(f"Number of runs: {len(time_series_list)}")
    print(f"Variable names: {metadata['variable_names']}")
    print(f"Target names: {metadata['target_names']}")
    print(f"Run IDs: {run_ids[:10]}...")  # Show first 10 run IDs
    
    # Examine the structure
    print("\n" + "="*50)
    print("DATA STRUCTURE EXAMINATION")
    print("="*50)
    
    for i in range(min(3, len(time_series_list))):  # Show first 3 runs
        run_id = run_ids[i]
        time_series = time_series_list[i]
        target_vals = targets[i]
        
        print(f"\nRun {run_id}:")
        print(f"  Time series shape: {time_series.shape} (4 variables x {time_series.shape[1]} time points)")
        print(f"  Target values: decayRate={target_vals[0]:.4f}, surfaceTransferFraction={target_vals[1]:.4f}")
        
        # Show first few time points for each variable
        print(f"  First 5 time points:")
        for j, var_name in enumerate(metadata['variable_names']):
            values = time_series[j, :5]  # First 5 time points
            print(f"    {var_name}: {values}")
    
    # Show summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    # Run lengths
    run_lengths = [ts.shape[1] for ts in time_series_list]
    print(f"Run lengths - Min: {min(run_lengths)}, Max: {max(run_lengths)}, Mean: {np.mean(run_lengths):.1f}")
    
    # Target variable statistics
    decay_rates = targets[:, 0]
    surface_transfers = targets[:, 1]
    
    print(f"\ndecayRate statistics:")
    print(f"  Min: {np.min(decay_rates):.4f}, Max: {np.max(decay_rates):.4f}")
    print(f"  Mean: {np.mean(decay_rates):.4f}, Std: {np.std(decay_rates):.4f}")
    
    print(f"\nsurfaceTransferFraction statistics:")
    print(f"  Min: {np.min(surface_transfers):.4f}, Max: {np.max(surface_transfers):.4f}")
    print(f"  Mean: {np.mean(surface_transfers):.4f}, Std: {np.std(surface_transfers):.4f}")
    
    # Time series variable statistics (across all runs and time points)
    print(f"\nTime series variables statistics:")
    for i, var_name in enumerate(metadata['variable_names']):
        all_values = np.concatenate([ts[i, :] for ts in time_series_list])
        print(f"  {var_name}: Min={np.min(all_values):.2f}, Max={np.max(all_values):.2f}, "
              f"Mean={np.mean(all_values):.2f}, Std={np.std(all_values):.2f}")
    
    return time_series_list, targets, run_ids, metadata

def demonstrate_ml_preparation(time_series_list, targets, run_ids, metadata):
    """Demonstrate how to prepare this data for machine learning."""
    
    print("\n" + "="*50)
    print("MACHINE LEARNING PREPARATION EXAMPLE")
    print("="*50)
    
    # For machine learning, you might want to:
    # 1. Standardize the run lengths (pad or truncate)
    # 2. Flatten the time series or use them as sequences
    # 3. Split into train/test sets
    
    # Example 1: Find common length and truncate/pad
    run_lengths = [ts.shape[1] for ts in time_series_list]
    min_length = min(run_lengths)
    max_length = max(run_lengths)
    
    print(f"Run length range: {min_length} to {max_length}")
    
    # Option A: Truncate to minimum length
    truncated_series = np.array([ts[:, :min_length] for ts in time_series_list])
    print(f"Truncated to min length - Shape: {truncated_series.shape}")
    print(f"  Interpretation: {len(time_series_list)} runs x 4 variables x {min_length} time points")
    
    # Option B: Pad to maximum length (with zeros or last value)
    padded_series = []
    for ts in time_series_list:
        if ts.shape[1] < max_length:
            # Pad with the last value
            padding = np.repeat(ts[:, -1:], max_length - ts.shape[1], axis=1)
            padded_ts = np.concatenate([ts, padding], axis=1)
        else:
            padded_ts = ts
        padded_series.append(padded_ts)
    
    padded_series = np.array(padded_series)
    print(f"Padded to max length - Shape: {padded_series.shape}")
    print(f"  Interpretation: {len(time_series_list)} runs x 4 variables x {max_length} time points")
    
    # Example 2: Flatten for traditional ML algorithms
    flattened_features = truncated_series.reshape(len(time_series_list), -1)
    print(f"Flattened features shape: {flattened_features.shape}")
    print(f"  Interpretation: {len(time_series_list)} runs x {4 * min_length} features")
    
    # Show how targets are structured
    print(f"Targets shape: {targets.shape}")
    print(f"  Interpretation: {len(time_series_list)} runs x 2 target variables")
    
    print(f"\nExample for ML:")
    print(f"  X (features): {flattened_features.shape} - each row is one run, flattened time series")
    print(f"  y (targets): {targets.shape} - each row is (decayRate, surfaceTransferFraction) for one run")

if __name__ == "__main__":
    # Load and examine the data
    time_series_list, targets, run_ids, metadata = load_and_examine_reshaped_data()
    
    # Demonstrate ML preparation
    demonstrate_ml_preparation(time_series_list, targets, run_ids, metadata)
    
    print(f"\n{'='*50}")
    print("NEXT STEPS FOR MACHINE LEARNING:")
    print("="*50)
    print("""
1. Choose how to handle different run lengths:
   - Truncate all to minimum length
   - Pad all to maximum length
   - Use sequence models that handle variable lengths

2. Feature engineering:
   - Use raw time series values
   - Extract statistical features (mean, std, trends)
   - Use time series transformations (FFT, wavelets, etc.)

3. Model selection:
   - Traditional ML: Use flattened features with Random Forest, SVM, etc.
   - Deep Learning: Use LSTM/GRU for sequences, CNN for patterns
   - Multi-output regression for predicting both target variables

4. Split data:
   - Train/validation/test splits by runs (not by time points)
   - Consider stratified splits based on target variable ranges
""")
