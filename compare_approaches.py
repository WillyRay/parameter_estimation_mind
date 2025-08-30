#!/usr/bin/env python3
"""
Comparison demo between the original flattened approach and new 4x56 CNN approach.

This script demonstrates both implementations side by side to show the differences
in how they handle the time series data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import our new CNN implementation
from cnn_4x56_model import load_and_prepare_4x56_data, create_cnn_model

def load_flattened_data():
    """Load data in the original flattened format (similar to training_data.csv approach)."""
    print("📥 Loading data for flattened approach...")
    
    # Load the 4x56 sliding window data first
    X_4x56, y, run_ids, window_starts, metadata = load_and_prepare_4x56_data()
    
    # Flatten each window's 4x56 array into a 1D vector of 224 features
    X_flattened = X_4x56.reshape(X_4x56.shape[0], -1)  # Shape: (windows, 224)
    
    print(f"   Flattened shape: {X_flattened.shape}")
    print(f"   Features: 4 variables × 56 days = {X_flattened.shape[1]} features")
    print(f"   Total windows: {X_flattened.shape[0]} (sliding window approach)")
    
    return X_flattened, y, run_ids, window_starts, metadata

def train_flattened_model(X_train, X_test, y_train, y_test):
    """Train a Random Forest model on flattened features."""
    print("\n🌲 Training Random Forest on flattened features...")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42,
        max_depth=10,
        min_samples_split=2
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    return y_pred, rf_model

def train_cnn_model_simple(X_train, X_test, y_train, y_test):
    """Train a simplified CNN model on 4x56 arrays."""
    print("\n🧠 Training CNN on 4x56 arrays...")
    
    # Normalize data (simple normalization)
    X_train_norm = (X_train - X_train.mean()) / (X_train.std() + 1e-8)
    X_test_norm = (X_test - X_train.mean()) / (X_train.std() + 1e-8)
    
    # Reshape for CNN (add channel dimension)
    X_train_cnn = X_train_norm.reshape(X_train_norm.shape[0], 4, 56, 1)
    X_test_cnn = X_test_norm.reshape(X_test_norm.shape[0], 4, 56, 1)
    
    # Create a simpler CNN model for comparison
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (2, 3), activation='relu', input_shape=(4, 56, 1)),
        tf.keras.layers.MaxPooling2D((1, 2)),
        tf.keras.layers.Conv2D(32, (2, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((1, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2)  # 2 outputs
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train with minimal epochs for demo
    model.fit(
        X_train_cnn, y_train, 
        validation_data=(X_test_cnn, y_test),
        epochs=50, 
        batch_size=4, 
        verbose=0
    )
    
    # Make predictions
    y_pred = model.predict(X_test_cnn, verbose=0)
    
    return y_pred, model

def compare_approaches():
    """Compare the flattened approach vs 4x56 CNN approach with sliding windows."""
    print("🔄 Comparison: Flattened vs 4x56 CNN Approaches (Sliding Windows)")
    print("="*70)
    
    # Load data in both formats
    X_4x56, y, run_ids, window_starts, metadata = load_and_prepare_4x56_data()
    X_flattened, _, _, _, _ = load_flattened_data()
    
    print(f"\n📊 Data Summary:")
    print(f"   Number of unique runs: {len(np.unique(run_ids))}")
    print(f"   Total windows: {len(run_ids)} (sliding window approach)")
    print(f"   Windows per run: {metadata.get('windows_per_run', 'unknown')}")
    print(f"   Target variables: {metadata['target_names']}")
    print(f"   Time series variables: {metadata['variable_names']}")
    print(f"   Window size: 56 time points")
    print(f"   Window range: {metadata.get('window_range', 'unknown')}")
    
    # Split data
    print(f"\n📊 Data Split:")
    indices = np.arange(len(run_ids))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
    
    # Flattened data splits
    X_flat_train, X_flat_test = X_flattened[train_idx], X_flattened[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 4x56 data splits
    X_4x56_train, X_4x56_test = X_4x56[train_idx], X_4x56[test_idx]
    
    print(f"   Training: {len(train_idx)} windows")
    print(f"   Testing: {len(test_idx)} windows")
    
    # Approach 1: Flattened features with Random Forest
    print(f"\n" + "="*60)
    print("APPROACH 1: FLATTENED FEATURES")
    print("="*60)
    print(f"Data representation: Each window as 1D vector of {X_flattened.shape[1]} features")
    print(f"Model: Random Forest Regressor")
    print(f"Features: Variables flattened across time [var1_day100, var1_day101, ..., var4_day155]")
    print(f"Training samples: {len(train_idx)} sliding windows")
    
    y_pred_flat, rf_model = train_flattened_model(X_flat_train, X_flat_test, y_train, y_test)
    
    # Approach 2: 4x56 arrays with CNN
    print(f"\n" + "="*60)
    print("APPROACH 2: 4x56 ARRAYS WITH CNN")
    print("="*60)
    print(f"Data representation: Each window as 4x56 array (variables × time)")
    print(f"Model: Convolutional Neural Network")
    print(f"Features: 2D arrays preserving variable-time structure")
    print(f"Training samples: {len(train_idx)} sliding windows")
    
    y_pred_cnn, cnn_model = train_cnn_model_simple(X_4x56_train, X_4x56_test, y_train, y_test)
    
    # Compare results
    print(f"\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    target_names = metadata['target_names']
    
    print(f"{'Metric':<25} {'Approach 1 (Flat)':<20} {'Approach 2 (CNN)':<20}")
    print("-" * 65)
    
    for i, target_name in enumerate(target_names):
        # Flattened approach metrics
        mse_flat = mean_squared_error(y_test[:, i], y_pred_flat[:, i])
        r2_flat = r2_score(y_test[:, i], y_pred_flat[:, i])
        
        # CNN approach metrics
        mse_cnn = mean_squared_error(y_test[:, i], y_pred_cnn[:, i])
        r2_cnn = r2_score(y_test[:, i], y_pred_cnn[:, i])
        
        print(f"\n{target_name}:")
        print(f"{'  MSE':<25} {mse_flat:<20.6f} {mse_cnn:<20.6f}")
        print(f"{'  R²':<25} {r2_flat:<20.4f} {r2_cnn:<20.4f}")
        print(f"{'  RMSE':<25} {np.sqrt(mse_flat):<20.6f} {np.sqrt(mse_cnn):<20.6f}")
    
    # Show sample predictions
    print(f"\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    print(f"{'Run':<6} {'Actual':<20} {'Flat Pred':<20} {'CNN Pred':<20}")
    print("-" * 66)
    
    for i in range(min(5, len(y_test))):
        run_id = run_ids[test_idx[i]]
        actual = f"[{y_test[i,0]:.3f}, {y_test[i,1]:.3f}]"
        flat_pred = f"[{y_pred_flat[i,0]:.3f}, {y_pred_flat[i,1]:.3f}]"
        cnn_pred = f"[{y_pred_cnn[i,0]:.3f}, {y_pred_cnn[i,1]:.3f}]"
        print(f"{run_id:<6} {actual:<20} {flat_pred:<20} {cnn_pred:<20}")
    
    # Summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("APPROACH 1 - Flattened Features:")
    print("  ✓ Simple and interpretable")
    print("  ✓ Works well with traditional ML algorithms")
    print("  ✓ Each time point treated as independent feature")
    print("  ✗ Loses temporal structure information")
    print("  ✗ No spatial relationships between variables")
    
    print("\nAPPROACH 2 - 4x56 CNN Arrays with Sliding Windows:")
    print("  ✓ Preserves temporal structure") 
    print("  ✓ Captures spatial relationships between variables")
    print("  ✓ Can learn complex patterns across time and variables")
    print("  ✓ More suitable for chronologically dependent data")
    print("  ✓ Sliding windows provide much more training data")
    print(f"  ✓ {metadata.get('total_windows', 'Many')} training examples from {len(np.unique(run_ids))} runs")
    print("  ✗ More complex model architecture")
    print("  ✗ Requires more careful hyperparameter tuning")
    
    print(f"\n🎯 Both approaches are now available in the repository!")
    print(f"   • Use flattened approach: generate_training_data.py + demo_ml_usage.py")
    print(f"   • Use 4x56 CNN sliding window approach: cnn_4x56_model.py")
    print(f"   • Compare both approaches: compare_approaches.py")
    print(f"\n📈 Sliding Window Benefits:")
    print(f"   • Multiple 56-length sequences per run (100-155, 101-156, etc.)")
    print(f"   • Significantly more training examples: {metadata.get('total_windows', 'N/A')} vs {len(np.unique(run_ids))}")
    print(f"   • Better utilization of available time series data")

if __name__ == "__main__":
    compare_approaches()