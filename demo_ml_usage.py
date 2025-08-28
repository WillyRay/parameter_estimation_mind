#!/usr/bin/env python3
"""
Demo script showing how to use the training dataset for machine learning with PyTorch.
This demonstrates the intended use case for predicting decayRate and surfaceTransferFraction.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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

def train_model(model, train_loader, criterion, optimizer, device, epochs=100, verbose=True):
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

def evaluate_model(model, test_loader, device):
    """Evaluate the model and return predictions."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    return predictions, targets

def calculate_r2_score(y_true, y_pred):
    """Calculate R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def demo_ml_usage(training_data_path):
    """Demonstrate ML usage of the training dataset with PyTorch."""
    print("🔬 PyTorch Neural Network Demo with Training Dataset")
    print("=" * 55)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load the training data
    print("\n📥 Loading training data...")
    df = pd.read_csv(training_data_path)
    print(f"   Dataset shape: {df.shape}")
    
    # Prepare features (X) and targets (y)
    print("\n🔧 Preparing features and targets...")
    
    # Features: all time series columns (4 variables × 56 days = 224 features)
    feature_cols = []
    for var in ['count', 'CDIFF', 'occupancy', 'anyCP']:
        for day in range(56):
            feature_cols.append(f"{var}_{day}")
    
    X = df[feature_cols].values
    y = df[['decayRate', 'surfaceTransferFraction']].values
    
    print(f"   Features shape: {X.shape}")
    print(f"   Targets shape: {y.shape}")
    
    # Split the data
    print("\n📊 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Standardize features
    print("\n🔄 Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print(f"\n🧠 Creating neural network model...")
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
    print(f"\n🤖 Training neural network...")
    train_losses = train_model(
        model, train_loader, criterion, optimizer, device, 
        epochs=100, verbose=True
    )
    
    # Make predictions
    print(f"\n🎯 Making predictions...")
    predictions, targets = evaluate_model(model, test_loader, device)
    
    # Evaluate performance
    print(f"\n📈 Model Performance:")
    print("-" * 30)
    
    # Calculate metrics for each target
    target_names = ['decayRate', 'surfaceTransferFraction']
    for i, target_name in enumerate(target_names):
        mse = np.mean((targets[:, i] - predictions[:, i]) ** 2)
        r2 = calculate_r2_score(targets[:, i], predictions[:, i])
        
        print(f"{target_name}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  R²:  {r2:.4f}")
        print(f"  RMSE: {np.sqrt(mse):.6f}")
        
        # Show actual vs predicted range
        actual_range = f"{targets[:, i].min():.4f} - {targets[:, i].max():.4f}"
        pred_range = f"{predictions[:, i].min():.4f} - {predictions[:, i].max():.4f}"
        print(f"  Actual range: {actual_range}")
        print(f"  Predicted range: {pred_range}")
        print()
    
    # Show some example predictions
    print("🔍 Sample Predictions (first 5):")
    print("-" * 30)
    print("   Actual vs Predicted [decayRate, surfaceTransferFraction]")
    for i in range(min(5, len(targets))):
        actual = f"[{targets[i, 0]:.4f}, {targets[i, 1]:.4f}]"
        predicted = f"[{predictions[i, 0]:.4f}, {predictions[i, 1]:.4f}]"
        print(f"   {actual} -> {predicted}")
    
    # Show training progress
    print(f"\n📉 Training Progress:")
    print("-" * 30)
    print(f"   Initial loss: {train_losses[0]:.6f}")
    print(f"   Final loss:   {train_losses[-1]:.6f}")
    print(f"   Improvement:  {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
    
    print(f"\n✅ PyTorch demo completed successfully!")
    print(f"\nℹ️  This demonstrates how the training dataset can be used to:")
    print(f"   • Train deep neural networks to predict decayRate and surfaceTransferFraction")
    print(f"   • Use 56-day time series as input features")
    print(f"   • Leverage GPU acceleration with PyTorch")
    print(f"   • Apply modern techniques like batch normalization and dropout")
    print(f"   • Evaluate model performance on unseen data")

if __name__ == "__main__":
    demo_ml_usage("./data/training_data.csv")