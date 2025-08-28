#!/usr/bin/env python3
"""
Script to process sim_data.csv into a training dataset.
Creates one row per sequence with 56-day time series for each of the 4 variables.
"""

import pandas as pd
import numpy as np
import sys

def create_training_dataset(sim_data_path, output_path, sequence_length=56):
    """
    Process simulation data into training format.
    
    Args:
        sim_data_path: Path to sim_data.csv
        output_path: Path where to save the training dataset CSV
        sequence_length: Length of sequences to extract (default 56 to match observed data)
    """
    
    # Load the simulation data
    print(f"Loading simulation data from {sim_data_path}...")
    data = pd.read_csv(sim_data_path, index_col=['run']).sort_index().sort_values(by='tick')
    
    print(f"Found {len(data.index.unique())} unique runs")
    
    # Prepare the training data
    training_rows = []
    
    for run_id in data.index.unique():
        run_data = data.loc[run_id].sort_values(by='tick')
        
        # Extract parameter values (constant within each run)
        decay_rate = run_data['decayRate'].iloc[0]
        surface_transfer_fraction = run_data['surfaceTransferFraction'].iloc[0]
        
        # Extract time series
        counts = run_data['count'].values
        cdffs = run_data['CDIFF'].values
        occupancies = run_data['occupancy'].values
        any_cps = run_data['anyCP'].values
        
        # Calculate how many sequences we can extract
        num_sequences = len(counts) - sequence_length + 1
        
        if num_sequences <= 0:
            print(f"Warning: Run {run_id} has insufficient data ({len(counts)} ticks), skipping")
            continue
            
        print(f"Run {run_id}: extracting {num_sequences} sequences from {len(counts)} ticks")
        
        # Extract overlapping sequences
        for start_idx in range(num_sequences):
            end_idx = start_idx + sequence_length
            
            # Create a row for this sequence
            row = {
                'run': run_id,
                'start_day': start_idx,
                'decayRate': decay_rate,
                'surfaceTransferFraction': surface_transfer_fraction
            }
            
            # Add the 56-day sequences for each variable
            for day in range(sequence_length):
                row[f'count_{day}'] = counts[start_idx + day]
                row[f'CDIFF_{day}'] = cdffs[start_idx + day]
                row[f'occupancy_{day}'] = occupancies[start_idx + day]
                row[f'anyCP_{day}'] = any_cps[start_idx + day]
            
            training_rows.append(row)
    
    # Convert to DataFrame
    training_df = pd.DataFrame(training_rows)
    
    print(f"Created training dataset with {len(training_df)} rows")
    print(f"Each row has {len(training_df.columns)} columns")
    
    # Save to CSV
    print(f"Saving training dataset to {output_path}...")
    training_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\nDataset summary:")
    print(f"- Total samples: {len(training_df)}")
    print(f"- Unique runs: {training_df['run'].nunique()}")
    print(f"- Target variables: decayRate, surfaceTransferFraction")
    print(f"- Feature variables: 4 time series × {sequence_length} days = {4 * sequence_length} features")
    print(f"- Total columns: {len(training_df.columns)}")
    
    # Show sample of the data
    print("\nFirst few rows (showing first 10 columns):")
    print(training_df.iloc[:5, :10])
    
    return training_df

if __name__ == "__main__":
    # Configuration
    sim_data_path = "./data/sim_data.csv"
    output_path = "./data/training_data.csv"
    
    # Generate the training dataset
    try:
        dataset = create_training_dataset(sim_data_path, output_path)
        print(f"\nSuccess! Training dataset saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)