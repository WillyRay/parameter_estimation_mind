#!/usr/bin/env python3
"""
Simple example showing how to access the reshaped data exactly as requested.

Created by: GitHub Copilot (first version)

This script demonstrates the final data structure:
- Observations grouped by run
- 4 x (run length) arrays for each run containing: count, CDIFF, occupancy, anyCP
- One pair of target variables per run: decayRate, surfaceTransferFraction
"""

import numpy as np

def load_reshaped_data():
    """Load the reshaped data and return in the requested format."""
    
    # Load the saved data
    time_series_list = np.load('data/time_series_by_run.npy', allow_pickle=True)
    targets = np.load('data/targets_by_run.npy')
    run_ids = np.load('data/run_ids.npy')
    metadata = np.load('data/metadata.npy', allow_pickle=True).item()
    
    return time_series_list, targets, run_ids, metadata

def demonstrate_data_access():
    """Demonstrate how to access the data in the requested format."""
    
    # Load data
    time_series_list, targets, run_ids, metadata = load_reshaped_data()
    
    print("RESHAPED SIM_DATA ACCESS EXAMPLE")
    print("="*50)
    print(f"Variable order in time series: {metadata['variable_names']}")
    print(f"Target variables: {metadata['target_names']}")
    print(f"Total number of runs: {len(run_ids)}")
    
    # Example: Access data for the first few runs
    for i in range(min(3, len(run_ids))):
        run_id = run_ids[i]
        
        # Get the 4 x (run_length) array for this run
        time_series_data = time_series_list[i]  # Shape: (4, run_length)
        
        # Get target variables for this run
        decay_rate, surface_transfer_fraction = targets[i]
        
        print(f"\nRun {run_id}:")
        print(f"  Time series shape: {time_series_data.shape}")
        print(f"  decayRate: {decay_rate:.6f}")
        print(f"  surfaceTransferFraction: {surface_transfer_fraction:.6f}")
        
        # Access individual time series
        count = time_series_data[0, :]        # count time series
        cdiff = time_series_data[1, :]        # CDIFF time series  
        occupancy = time_series_data[2, :]    # occupancy time series
        any_cp = time_series_data[3, :]       # anyCP time series
        
        print(f"  count (first 10): {count[:10]}")
        print(f"  CDIFF (first 10): {cdiff[:10]}")
        print(f"  occupancy (first 10): {occupancy[:10]}")
        print(f"  anyCP (first 10): {any_cp[:10]}")

def create_ml_ready_dataset():
    """Create datasets ready for machine learning."""
    
    # Load data
    time_series_list, targets, run_ids, metadata = load_reshaped_data()
    
    print(f"\n{'='*50}")
    print("MACHINE LEARNING READY DATASETS")
    print("="*50)
    
    # Convert to numpy arrays for easier handling
    # All runs have the same length (276), so we can stack them
    X = np.stack(time_series_list)  # Shape: (num_runs, 4, run_length)
    y = targets                     # Shape: (num_runs, 2)
    
    print(f"Features (X) shape: {X.shape}")
    print(f"  Interpretation: {X.shape[0]} runs x {X.shape[1]} variables x {X.shape[2]} time points")
    print(f"Targets (y) shape: {y.shape}")
    print(f"  Interpretation: {y.shape[0]} runs x {y.shape[1]} target variables")
    
    # Option 1: Flatten for traditional ML
    X_flattened = X.reshape(X.shape[0], -1)  # Shape: (num_runs, 4*run_length)
    print(f"\nFlattened features shape: {X_flattened.shape}")
    print(f"  Use this for: Random Forest, SVM, Linear Regression, etc.")
    
    # Option 2: Keep as sequences for deep learning
    print(f"\nSequence features shape: {X.shape}")
    print(f"  Use this for: LSTM, GRU, 1D CNN, etc.")
    
    # Individual target variables
    decay_rates = y[:, 0]
    surface_transfers = y[:, 1]
    
    print(f"\nIndividual targets:")
    print(f"  Decay rates shape: {decay_rates.shape}")
    print(f"  Surface transfer fractions shape: {surface_transfers.shape}")
    
    return X, y, run_ids

if __name__ == "__main__":
    # Demonstrate data access
    demonstrate_data_access()
    
    # Create ML-ready datasets
    X, y, run_ids = create_ml_ready_dataset()
    
    print(f"\n{'='*50}")
    print("SUMMARY - DATA STRUCTURE AS REQUESTED:")
    print("="*50)
    print("""
✓ Observations grouped by 'run' column
✓ For each run: 4 x (run_length) array containing:
  - Row 0: count time series
  - Row 1: CDIFF time series  
  - Row 2: occupancy time series
  - Row 3: anyCP time series
✓ Target variables: one (decayRate, surfaceTransferFraction) pair per run
✓ tick and run columns discarded (only used for grouping)

Access pattern:
- time_series_list[run_index] gives 4 x (run_length) array
- targets[run_index] gives (decayRate, surfaceTransferFraction) tuple
- All ready for machine learning!
""")
