#!/usr/bin/env python3
"""
Script to reshape sim_data.csv for machine learning training.

Created by: GitHub Copilot (first version)
"""
"""

This script reads the sim_data.csv file and reshapes it so that:
1. Observations are grouped by the "run" column
2. For each run, we extract: count, CDIFF, occupancy, and anyCP as time series
3. decayRate and surfaceTransferFraction are stored as one pair per run
4. The result is a dictionary where each run has a 4 x (run_length) array for the time series variables
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

def read_and_reshape_sim_data(file_path: str) -> Dict[str, Any]:
    """
    Read and reshape the simulation data.
    
    Args:
        file_path: Path to the sim_data.csv file
        
    Returns:
        Dictionary containing:
        - 'time_series_data': Dict with run_id as key and 4x(run_length) numpy array as value
        - 'target_variables': Dict with run_id as key and (decayRate, surfaceTransferFraction) tuple as value
        - 'variable_names': List of variable names for the time series (in order)
    """
    
    # Read the CSV file
    print("Reading sim_data.csv...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Get unique runs
    unique_runs = sorted(df['run'].unique())
    print(f"Number of unique runs: {len(unique_runs)}")
    
    # Define the time series variables and target variables
    time_series_vars = ['count', 'CDIFF', 'occupancy', 'anyCP']
    target_vars = ['decayRate', 'surfaceTransferFraction']
    
    # Initialize result dictionaries
    time_series_data = {}
    target_variables = {}
    
    # Process each run
    for run_id in unique_runs:
        # Filter data for this run
        run_data = df[df['run'] == run_id].copy()
        
        # Sort by tick to ensure proper time series order
        run_data = run_data.sort_values('tick')
        
        # Extract time series data (4 x run_length array)
        time_series_matrix = np.array([
            run_data[var].values for var in time_series_vars
        ])
        
        # Extract target variables (should be the same for all rows in a run)
        decay_rate = run_data[target_vars[0]].iloc[0]
        surface_transfer = run_data[target_vars[1]].iloc[0]
        
        # Verify that target variables are constant within the run
        assert run_data[target_vars[0]].nunique() == 1, f"decayRate varies within run {run_id}"
        assert run_data[target_vars[1]].nunique() == 1, f"surfaceTransferFraction varies within run {run_id}"
        
        # Store results
        time_series_data[run_id] = time_series_matrix
        target_variables[run_id] = (decay_rate, surface_transfer)
        
        print(f"Run {run_id}: {time_series_matrix.shape[1]} time points, "
              f"decayRate={decay_rate:.4f}, surfaceTransferFraction={surface_transfer:.4f}")
    
    return {
        'time_series_data': time_series_data,
        'target_variables': target_variables,
        'variable_names': time_series_vars,
        'target_names': target_vars
    }

def print_summary(reshaped_data: Dict[str, Any]):
    """Print a summary of the reshaped data."""
    
    time_series_data = reshaped_data['time_series_data']
    target_variables = reshaped_data['target_variables']
    variable_names = reshaped_data['variable_names']
    target_names = reshaped_data['target_names']
    
    print("\n" + "="*60)
    print("RESHAPED DATA SUMMARY")
    print("="*60)
    
    print(f"Number of runs: {len(time_series_data)}")
    print(f"Time series variables: {variable_names}")
    print(f"Target variables: {target_names}")
    
    # Get run lengths
    run_lengths = [data.shape[1] for data in time_series_data.values()]
    print(f"\nRun lengths - Min: {min(run_lengths)}, Max: {max(run_lengths)}, Mean: {np.mean(run_lengths):.1f}")
    
    # Show example data for first run
    first_run = min(time_series_data.keys())
    print(f"\nExample - Run {first_run}:")
    print(f"Time series shape: {time_series_data[first_run].shape}")
    print(f"Target values: decayRate={target_variables[first_run][0]:.4f}, "
          f"surfaceTransferFraction={target_variables[first_run][1]:.4f}")
    
    print(f"\nFirst 5 time points of time series data for run {first_run}:")
    example_data = time_series_data[first_run][:, :5]  # First 5 time points
    for i, var_name in enumerate(variable_names):
        print(f"  {var_name}: {example_data[i]}")

def save_reshaped_data(reshaped_data: Dict[str, Any], output_dir: str = "data"):
    """Save the reshaped data to numpy files."""
    
    import os
    
    time_series_data = reshaped_data['time_series_data']
    target_variables = reshaped_data['target_variables']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save time series data
    runs = sorted(time_series_data.keys())
    
    # Create arrays for all runs
    all_time_series = []
    all_targets = []
    run_ids = []
    
    for run_id in runs:
        all_time_series.append(time_series_data[run_id])
        all_targets.append(target_variables[run_id])
        run_ids.append(run_id)
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, "time_series_by_run.npy"), all_time_series, allow_pickle=True)
    np.save(os.path.join(output_dir, "targets_by_run.npy"), np.array(all_targets))
    np.save(os.path.join(output_dir, "run_ids.npy"), np.array(run_ids))
    
    # Save metadata
    metadata = {
        'variable_names': reshaped_data['variable_names'],
        'target_names': reshaped_data['target_names'],
        'num_runs': len(runs)
    }
    np.save(os.path.join(output_dir, "metadata.npy"), metadata)
    
    print(f"\nData saved to {output_dir}/ directory:")
    print(f"  - time_series_by_run.npy: List of 4x(run_length) arrays")
    print(f"  - targets_by_run.npy: Array of (decayRate, surfaceTransferFraction) pairs")
    print(f"  - run_ids.npy: Array of run IDs")
    print(f"  - metadata.npy: Variable names and other metadata")

if __name__ == "__main__":
    # File path
    data_file = "data/sim_data.csv"
    
    # Read and reshape the data
    reshaped_data = read_and_reshape_sim_data(data_file)
    
    # Print summary
    print_summary(reshaped_data)
    
    # Save the reshaped data
    save_reshaped_data(reshaped_data)
    
    print(f"\n{'='*60}")
    print("USAGE EXAMPLE:")
    print("="*60)
    print("""
# To load the saved data later:
import numpy as np

# Load the data
time_series_list = np.load('data/time_series_by_run.npy', allow_pickle=True)
targets = np.load('data/targets_by_run.npy')
run_ids = np.load('data/run_ids.npy')
metadata = np.load('data/metadata.npy', allow_pickle=True).item()

# Access data for a specific run (e.g., first run)
run_0_time_series = time_series_list[0]  # Shape: (4, run_length)
run_0_targets = targets[0]               # Shape: (2,) - (decayRate, surfaceTransferFraction)

print(f"Run {run_ids[0]} time series shape: {run_0_time_series.shape}")
print(f"Variables: {metadata['variable_names']}")
print(f"Targets: {metadata['target_names']}")
""")
