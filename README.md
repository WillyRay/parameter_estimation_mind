# Parameter Estimation Training Dataset

This repository contains scripts to process simulation data for machine learning-based parameter estimation.

## Contributors

- **Data Reshaping Scripts**: GitHub Copilot created the first version of the data reshaping utilities (`reshape_sim_data.py`, `verify_reshaped_data.py`, `access_reshaped_data.py`)

## Overview

The goal is to train a machine learning model to predict `decayRate` and `surfaceTransferFraction` parameters based on 56-day time series data from 4 variables:
- `count`
- `CDIFF` 
- `occupancy`
- `anyCP`

## Files

### Data Processing
- `generate_training_data.py` - Main script to convert `sim_data.csv` into ML training format
- `test_training_data.py` - Validation script to ensure correct dataset format
- `demo_ml_usage.py` - Demonstration of how to use the training data for ML
- `predict_observed_data.py` - Script to predict parameters for observed data using trained model
- `reshape_sim_data.py` - Script to reshape sim_data.csv into grouped time series format (Created by GitHub Copilot)
- `verify_reshaped_data.py` - Verification and analysis of reshaped data (Created by GitHub Copilot)
- `access_reshaped_data.py` - Example showing how to access reshaped data (Created by GitHub Copilot)

### Data Files
- `data/sim_data.csv` - Raw simulation data (21 runs, 276 time steps each)
- `data/observed_data.csv` - Observed data (56 days) used as target length
- `data/training_data.csv` - Processed training dataset (generated)

### Utilities
- `DataClasses.py` - Data class definitions for runs and samples

## Usage

### Generate Training Dataset
```bash
python generate_training_data.py
```

This creates `data/training_data.csv` with:
- 4,641 training samples (221 sequences × 21 runs)
- Each row represents one 56-day sequence
- Columns: `run`, `start_day`, `decayRate`, `surfaceTransferFraction`, plus 224 time series features

### Validate Dataset
```bash
python test_training_data.py
```

### Run ML Demo
```bash
python demo_ml_usage.py
```

### Predict Parameters for Observed Data
```bash
python predict_observed_data.py
```

This script:
- Trains a neural network model on the simulation data
- Processes the observed data into the correct input format
- Predicts `decayRate` and `surfaceTransferFraction` for the observed data
- Displays the predictions with validation information

## Dataset Format

The training dataset has the following structure:

| Column | Description |
|--------|-------------|
| `run` | Simulation run ID |
| `start_day` | Starting day of the 56-day sequence |
| `decayRate` | Target parameter 1 |
| `surfaceTransferFraction` | Target parameter 2 |
| `count_0` to `count_55` | Count time series (56 values) |
| `CDIFF_0` to `CDIFF_55` | CDIFF time series (56 values) |
| `occupancy_0` to `occupancy_55` | Occupancy time series (56 values) |
| `anyCP_0` to `anyCP_55` | AnyCP time series (56 values) |

Total: 228 columns (4 metadata + 224 features)

## Machine Learning Usage

The dataset is designed for multi-output regression where:
- **Input (X)**: 224 time series features (4 variables × 56 days)
- **Output (y)**: 2 target parameters (`decayRate`, `surfaceTransferFraction`)

Example using PyTorch:
```python
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load data
df = pd.read_csv('data/training_data.csv')

# Prepare features and targets
feature_cols = [col for col in df.columns if any(var in col for var in ['count_', 'CDIFF_', 'occupancy_', 'anyCP_'])]
X = df[feature_cols].values
y = df[['decayRate', 'surfaceTransferFraction']].values

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Create neural network
model = nn.Sequential(
    nn.Linear(224, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Dropout(0.2),
    nn.Linear(128, 2)
)

# Train model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```