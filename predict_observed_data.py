#!/usr/bin/env python3
"""
Script to predict decayRate and surfaceTransferFraction parameters for the observed data.
This script trains a model on the existing training data and then uses it to make predictions
on the observed data.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class ParameterPredictor(nn.Module):
    """Neural network for predicting decayRate and surfaceTransferFraction from time series."""
    
    def __init__(self, input_size=224, hidden_sizes=[512, 256, 128], output_size=2, dropout_rate=0.2):
        super(ParameterPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, criterion, optimizer, device, epochs=100, verbose=False):
    """Train the PyTorch model."""
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"   Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    return train_losses

def process_observed_data(observed_data_path):
    """
    Process the observed data into the same format as the training data.
    
    Args:
        observed_data_path: Path to the observed_data.csv file
        
    Returns:
        numpy array with 224 features (4 variables × 56 days)
    """
    print(f"\n📥 Loading observed data from {observed_data_path}...")
    
    # Load the observed data
    observed_df = pd.read_csv(observed_data_path)
    
    print(f"   Observed data shape: {observed_df.shape}")
    print(f"   Columns: {list(observed_df.columns)}")
    
    # Check if we have exactly 56 days of data
    if len(observed_df) != 56:
        print(f"   Warning: Expected 56 days of data, but found {len(observed_df)} days")
    
    # Extract the 4 time series variables
    count_series = observed_df['count'].values
    cdiff_series = observed_df['CDIFF'].values
    occupancy_series = observed_df['occupancy'].values
    anycp_series = observed_df['anyCP'].values
    
    # Ensure we have exactly 56 values by taking the first 56 or padding with zeros
    def ensure_56_length(series):
        if len(series) >= 56:
            return series[:56]
        else:
            # Pad with the last value if we have fewer than 56 days
            padded = np.zeros(56)
            padded[:len(series)] = series
            padded[len(series):] = series[-1]  # Repeat last value
            return padded
    
    count_series = ensure_56_length(count_series)
    cdiff_series = ensure_56_length(cdiff_series)
    occupancy_series = ensure_56_length(occupancy_series)
    anycp_series = ensure_56_length(anycp_series)
    
    # Create the feature vector in the same order as training data
    features = []
    for day in range(56):
        features.append(count_series[day])
    for day in range(56):
        features.append(cdiff_series[day])
    for day in range(56):
        features.append(occupancy_series[day])
    for day in range(56):
        features.append(anycp_series[day])
    
    # Convert to numpy array and reshape for single sample prediction
    features_array = np.array(features).reshape(1, -1)
    
    print(f"   Processed features shape: {features_array.shape}")
    print(f"   Sample feature values: {features_array[0][:5]}... (showing first 5)")
    
    return features_array

def predict_observed_parameters(training_data_path, observed_data_path):
    """
    Train a model on the training data and use it to predict parameters for observed data.
    
    Args:
        training_data_path: Path to the training dataset CSV
        observed_data_path: Path to the observed data CSV
    """
    print("🔬 Predicting Parameters for Observed Data")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load and prepare training data
    print("\n📥 Loading training data...")
    df = pd.read_csv(training_data_path)
    print(f"   Training dataset shape: {df.shape}")
    
    # Prepare features (X) and targets (y) from training data
    print("\n🔧 Preparing training features and targets...")
    
    # Features: all time series columns (4 variables × 56 days = 224 features)
    feature_cols = []
    for var in ['count', 'CDIFF', 'occupancy', 'anyCP']:
        for day in range(56):
            feature_cols.append(f"{var}_{day}")
    
    X_train = df[feature_cols].values
    y_train = df[['decayRate', 'surfaceTransferFraction']].values
    
    print(f"   Training features shape: {X_train.shape}")
    print(f"   Training targets shape: {y_train.shape}")
    
    # Standardize training features
    print("\n🔄 Standardizing training features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    
    # Create data loader
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Create and train model
    print(f"\n🧠 Creating and training neural network model...")
    model = ParameterPredictor(
        input_size=X_train.shape[1],
        hidden_sizes=[512, 256, 128],
        output_size=2,
        dropout_rate=0.2
    )
    model = model.to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    print(f"\n🤖 Training neural network (this may take a moment)...")
    train_losses = train_model(
        model, train_loader, criterion, optimizer, device, 
        epochs=100, verbose=False
    )
    
    print(f"   Training completed! Final loss: {train_losses[-1]:.6f}")
    
    # Process observed data
    observed_features = process_observed_data(observed_data_path)
    
    # Standardize observed features using the same scaler
    print(f"\n🔄 Standardizing observed data features...")
    observed_features_scaled = scaler.transform(observed_features)
    
    # Convert to PyTorch tensor
    observed_tensor = torch.FloatTensor(observed_features_scaled).to(device)
    
    # Make prediction
    print(f"\n🎯 Making predictions for observed data...")
    model.eval()
    with torch.no_grad():
        predictions = model(observed_tensor)
        predictions_np = predictions.cpu().numpy()
    
    # Display results
    print(f"\n📈 Prediction Results:")
    print("=" * 30)
    
    predicted_decay_rate = predictions_np[0, 0]
    predicted_surface_transfer = predictions_np[0, 1]
    
    print(f"Predicted decayRate: {predicted_decay_rate:.6f}")
    print(f"Predicted surfaceTransferFraction: {predicted_surface_transfer:.6f}")
    
    # Show some context about the training data ranges
    training_decay_range = f"{y_train[:, 0].min():.4f} - {y_train[:, 0].max():.4f}"
    training_surface_range = f"{y_train[:, 1].min():.4f} - {y_train[:, 1].max():.4f}"
    
    print(f"\nℹ️  Training data ranges for context:")
    print(f"   decayRate range: {training_decay_range}")
    print(f"   surfaceTransferFraction range: {training_surface_range}")
    
    # Check if predictions are within reasonable bounds
    if y_train[:, 0].min() <= predicted_decay_rate <= y_train[:, 0].max():
        decay_status = "✅ within training range"
    else:
        decay_status = "⚠️  outside training range"
        
    if y_train[:, 1].min() <= predicted_surface_transfer <= y_train[:, 1].max():
        surface_status = "✅ within training range"
    else:
        surface_status = "⚠️  outside training range"
    
    print(f"\n🔍 Prediction validation:")
    print(f"   decayRate: {decay_status}")
    print(f"   surfaceTransferFraction: {surface_status}")
    
    print(f"\n✅ Prediction completed successfully!")
    
    return predicted_decay_rate, predicted_surface_transfer

if __name__ == "__main__":
    # Configuration
    training_data_path = "./data/training_data.csv"
    observed_data_path = "./data/observed_data.csv"
    
    try:
        predictions = predict_observed_parameters(training_data_path, observed_data_path)
        print(f"\n🎯 Final Results:")
        print(f"   Decay Rate: {predictions[0]:.6f}")
        print(f"   Surface Transfer Fraction: {predictions[1]:.6f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()