"""
Data classes for parameter estimation.
Contains class definitions for handling simulation runs and training samples.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Run:
    """Represents a single simulation run with all its time series data."""
    id: int
    decay: float
    touchTransferFraction: float
    counts: List[float]
    occupancies: List[float]
    cdffs: List[float]
    anyCps: List[float]


@dataclass
class Sample:
    """Represents a training sample: a 56-day sequence from a run."""
    run: int
    startDay: int
    decay: float
    touchTransferFractions: float
    counts: List[float]
    occupancies: List[float]
    cdiffs: List[float]
    anyCps: List[float]


def split_sequences(sequences, n_steps):
    """
    Split a sequence into overlapping subsequences of length n_steps.
    
    Args:
        sequences: Input sequence to split
        n_steps: Length of each subsequence
        
    Returns:
        Array of subsequences
    """
    retlist = []
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x = sequences[i:end_ix]
        retlist.append(seq_x)
    
    return np.array(retlist)


def get_samples(run, n_steps):
    """
    Extract training samples from a run (placeholder function).
    
    Args:
        run: Run object containing time series data
        n_steps: Length of sequences to extract
    """
    # This function can be implemented to extract samples from Run objects
    # For now, the main data processing is handled by generate_training_data.py
    pass