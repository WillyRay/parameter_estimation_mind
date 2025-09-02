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

### 4x56 CNN Implementation
- `cnn_4x56_model.py` - Convolutional Neural Network implementation for 4x56 time series arrays
- `test_cnn_4x56.py` - Comprehensive test suite for the CNN implementation  
- `compare_approaches.py` - Side-by-side comparison of flattened vs CNN approaches

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

### Run ML Demo (Flattened Approach)
```bash
python demo_ml_usage.py
```

### Run CNN Demo (4x56 Arrays)
```bash
python cnn_4x56_model.py
```

### Compare Both Approaches
```bash
python compare_approaches.py
```

### Test CNN Implementation
```bash
python test_cnn_4x56.py
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

## Machine Learning Approaches

This repository now offers two distinct approaches for training models:

### Approach 1: Flattened Features (Original)

Uses traditional machine learning with flattened time series:
- **Data format**: 224 features per sample (4 variables × 56 days)
- **Model types**: Random Forest, SVM, Neural Networks with dense layers
- **Benefits**: Simple, interpretable, works with traditional ML algorithms
- **Use case**: When temporal structure is less important

Example:
```python
# Each sample is a flat vector: [count_0, count_1, ..., count_55, CDIFF_0, ...]
X_flat = data.reshape(num_samples, 224)
```

### Approach 2: 4x56 CNN Arrays with Sliding Windows (New)

Uses convolutional neural networks with structured arrays and sliding window approach:
- **Data format**: 4×56 arrays per sample (variables × time)
- **Sliding windows**: Multiple 56-day sequences per run (100-155, 101-156, 102-157, etc.)
- **Training samples**: ~4,400 windows from 21 runs (211 windows per run)
- **Model types**: Convolutional Neural Networks (CNN)
- **Benefits**: Preserves temporal structure, captures spatial relationships, much more training data
- **Use case**: When chronological dependencies are important and more training examples are needed

Example:
```python
# Each sample is a 4×56 array preserving structure:
# Row 0: count time series [56 consecutive days]
# Row 1: CDIFF time series [56 consecutive days]  
# Row 2: occupancy time series [56 consecutive days]
# Row 3: anyCP time series [56 consecutive days]

# Multiple sliding windows per run:
# Window 1: days 100-155
# Window 2: days 101-156  
# Window 3: days 102-157
# ... and so on
X_4x56 = data.reshape(num_windows, 4, 56, 1)  # Add channel dimension for CNN
```

### CNN Architecture

The 4x56 CNN implementation includes:
- **Sliding Window Data Generation**: Creates ~4,400 training examples from 21 runs
- Conv2D layers to capture spatial patterns between variables
- MaxPooling along time dimension to reduce dimensionality
- BatchNormalization for stable training
- Dropout for regularization
- Dense layers for final regression to 2 target parameters

Key advantage: The sliding window approach provides much more training data by creating multiple overlapping 56-day sequences from each simulation run, significantly improving model training compared to using only one sequence per run.

```python
model = Sequential([
    Conv2D(32, (2, 3), activation='relu', input_shape=(4, 56, 1)),
    BatchNormalization(),
    MaxPooling2D((1, 2)),  # Pool along time dimension
    # ... additional layers
    Dense(2)  # Output: decayRate, surfaceTransferFraction
])
```

## Machine Learning Usage

The dataset is designed for multi-output regression where:
- **Input (X)**: Time series data in two formats:
  - Flattened: 224 features (4 variables × 56 days)  
  - Structured: 4×56 arrays (variables × time)
- **Output (y)**: 2 target parameters (`decayRate`, `surfaceTransferFraction`)

### Example using PyTorch (Flattened approach):
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

### Example using TensorFlow/Keras (4x56 CNN with Sliding Windows):
```python
from cnn_4x56_model import load_and_prepare_4x56_data, create_cnn_model
from sklearn.model_selection import train_test_split
import numpy as np

# Load sliding window data
X, y, run_ids, window_starts, metadata = load_and_prepare_4x56_data()
print(f"Loaded {X.shape[0]} windows from {len(np.unique(run_ids))} runs")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN (add channel dimension)
X_train_cnn = X_train.reshape(X_train.shape[0], 4, 56, 1)
X_test_cnn = X_test.reshape(X_test.shape[0], 4, 56, 1)

# Create and train CNN model
model = create_cnn_model(input_shape=(4, 56, 1))
model.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test), epochs=100)
```

## Quick Start

**Option 1: Use the flattened approach (traditional ML)**
```bash
python demo_ml_usage.py
```

**Option 2: Use the 4x56 CNN approach with sliding windows**
```bash
python cnn_4x56_model.py
```

**Option 3: Compare both approaches**
```bash
python compare_approaches.py
```

**Run tests**
```bash
python test_cnn_4x56.py
```