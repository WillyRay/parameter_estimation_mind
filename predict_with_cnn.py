#!/usr/bin/env python3
"""
Enhanced prediction script that uses the CNN sliding window approach to predict
decayRate and surfaceTransferFraction parameters for the observed data.

This script leverages the CNN model trained on 4x56 time series arrays with
sliding windows to make predictions on the observed data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_cnn_training_data():
    """
    Load the reshaped training data and create sliding windows for CNN training.
    
    Returns:
        X: Array of shape (num_windows, 4, 56) containing time series data
        y: Array of shape (num_windows, 2) containing target variables  
        scalers: List of scalers for each variable
        metadata: Dictionary with variable names and other info
    """
    print("📥 Loading training data for CNN...")
    
    # Load the saved data
    time_series_list = np.load('data/time_series_by_run.npy', allow_pickle=True)
    targets = np.load('data/targets_by_run.npy')
    run_ids = np.load('data/run_ids.npy')
    metadata = np.load('data/metadata.npy', allow_pickle=True).item()
    
    print(f"   Training data: {len(time_series_list)} runs")
    print(f"   Variables: {metadata['variable_names']}")
    print(f"   Targets: {metadata['target_names']}")
    
    # Create sliding windows (same as in cnn_4x56_model.py)
    window_size = 56
    start_tick = 100
    tick_start_in_data = 90
    
    start_index = start_tick - tick_start_in_data  # 10
    max_data_index = time_series_list[0].shape[1] - 1  # 275
    last_start_index = max_data_index - window_size + 1  # 220
    num_windows_per_run = last_start_index - start_index + 1  # 211
    
    all_windows = []
    all_targets = []
    
    for run_idx, (ts, target) in enumerate(zip(time_series_list, targets)):
        for window_idx in range(num_windows_per_run):
            window_start_index = start_index + window_idx
            window_end_index = window_start_index + window_size
            
            # Extract the 4x56 window
            window_data = ts[:, window_start_index:window_end_index]  # Shape: (4, 56)
            
            all_windows.append(window_data)
            all_targets.append(target)
    
    # Convert to numpy arrays
    X = np.stack(all_windows)  # Shape: (total_windows, 4, 56)
    y = np.stack(all_targets)  # Shape: (total_windows, 2)
    
    print(f"   Final training shape: {X.shape}")
    print(f"   Total training examples: {X.shape[0]}")
    
    # Normalize data
    print("🔄 Normalizing training data...")
    X_norm = X.copy()
    scalers = []
    
    # Normalize each variable separately
    for var_idx in range(4):
        # Reshape to combine time and examples for fitting scaler
        var_data = X[:, var_idx, :].reshape(-1, 1)
        
        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(var_data)
        scalers.append(scaler)
        
        # Transform training data
        normalized = scaler.transform(var_data).reshape(X.shape[0], -1)
        X_norm[:, var_idx, :] = normalized
    
    return X_norm, y, scalers, metadata

def create_cnn_model(input_shape=(4, 56, 1)):
    """Create the CNN model architecture."""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (2, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(32, (2, 3), activation='relu', padding='same'),
        MaxPooling2D((1, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (2, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (2, 3), activation='relu', padding='same'),
        MaxPooling2D((1, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (2, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Flatten and dense layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # Output layer
        Dense(2, activation='linear')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def process_observed_data_for_cnn(observed_data_path):
    """
    Process the observed data into 4x56 format for CNN prediction.
    
    Args:
        observed_data_path: Path to the observed_data.csv file
        
    Returns:
        numpy array of shape (1, 4, 56) for CNN prediction
    """
    print(f"\n📥 Processing observed data for CNN prediction...")
    
    # Load the observed data
    observed_df = pd.read_csv(observed_data_path)
    print(f"   Observed data shape: {observed_df.shape}")
    
    # Extract the 4 time series variables
    count_series = observed_df['count'].values
    cdiff_series = observed_df['CDIFF'].values  
    occupancy_series = observed_df['occupancy'].values
    anycp_series = observed_df['anyCP'].values
    
    # Ensure we have exactly 56 values
    def ensure_56_length(series):
        if len(series) >= 56:
            return series[:56]
        else:
            # Pad with the last value if we have fewer than 56 days
            padded = np.zeros(56)
            padded[:len(series)] = series
            if len(series) > 0:
                padded[len(series):] = series[-1]
            return padded
    
    count_series = ensure_56_length(count_series)
    cdiff_series = ensure_56_length(cdiff_series)
    occupancy_series = ensure_56_length(occupancy_series)
    anycp_series = ensure_56_length(anycp_series)
    
    # Create 4x56 array
    observed_4x56 = np.array([
        count_series,
        cdiff_series,
        occupancy_series,
        anycp_series
    ])  # Shape: (4, 56)
    
    # Add batch dimension
    observed_batch = observed_4x56.reshape(1, 4, 56)  # Shape: (1, 4, 56)
    
    print(f"   Processed to 4x56 format: {observed_batch.shape}")
    
    return observed_batch

def predict_with_cnn_approach(observed_data_path):
    """
    Train a CNN model and use it to predict parameters for observed data.
    
    Args:
        observed_data_path: Path to the observed data CSV
    """
    print("🧠 CNN-Based Parameter Prediction for Observed Data")
    print("=" * 55)
    
    # Load and prepare training data
    X_train, y_train, scalers, metadata = load_and_prepare_cnn_training_data()
    
    # Normalize the observed data
    observed_4x56 = process_observed_data_for_cnn(observed_data_path)
    
    print("🔄 Normalizing observed data...")
    observed_norm = observed_4x56.copy()
    
    # Apply the same normalization as training data
    for var_idx in range(4):
        # Extract variable data and reshape for scaler
        var_data = observed_4x56[:, var_idx, :].reshape(-1, 1)
        
        # Apply the fitted scaler
        normalized = scalers[var_idx].transform(var_data).reshape(1, -1)
        observed_norm[:, var_idx, :] = normalized
    
    # Reshape for CNN (add channel dimension)
    X_train_cnn = X_train.reshape(X_train.shape[0], 4, 56, 1)
    observed_cnn = observed_norm.reshape(1, 4, 56, 1)
    
    print(f"   Training data shape: {X_train_cnn.shape}")
    print(f"   Observed data shape: {observed_cnn.shape}")
    
    # Create and train CNN model
    print("\n🤖 Training CNN model...")
    model = create_cnn_model(input_shape=(4, 56, 1))
    
    # Train the model
    history = model.fit(
        X_train_cnn, y_train,
        epochs=50,  # Reduced for faster prediction
        batch_size=4,
        verbose=0,  # Silent training for cleaner output
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    print(f"   Training completed! Final loss: {history.history['loss'][-1]:.6f}")
    
    # Make prediction on observed data
    print("\n🎯 Making CNN prediction...")
    prediction = model.predict(observed_cnn, verbose=0)
    
    predicted_decay_rate = prediction[0, 0]
    predicted_surface_transfer = prediction[0, 1]
    
    # Display results
    print(f"\n📈 CNN Prediction Results:")
    print("=" * 30)
    print(f"Predicted decayRate: {predicted_decay_rate:.6f}")
    print(f"Predicted surfaceTransferFraction: {predicted_surface_transfer:.6f}")
    
    # Show training data context
    training_decay_range = f"{y_train[:, 0].min():.4f} - {y_train[:, 0].max():.4f}"
    training_surface_range = f"{y_train[:, 1].min():.4f} - {y_train[:, 1].max():.4f}"
    
    print(f"\nℹ️  Training data ranges for context:")
    print(f"   decayRate range: {training_decay_range}")
    print(f"   surfaceTransferFraction range: {training_surface_range}")
    
    # Validation
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
    
    print(f"\n📊 Model Details:")
    print(f"   • Training examples: {X_train.shape[0]:,} (sliding windows)")
    print(f"   • Input format: 4×56 time series arrays")
    print(f"   • CNN architecture: Conv2D → BatchNorm → MaxPool → Dense")
    print(f"   • Variables: {metadata['variable_names']}")
    print(f"   • Epochs trained: {len(history.history['loss'])}")
    
    print(f"\n✅ CNN prediction completed successfully!")
    
    return predicted_decay_rate, predicted_surface_transfer

if __name__ == "__main__":
    # Configuration
    observed_data_path = "./data/observed_data.csv"
    
    try:
        predictions = predict_with_cnn_approach(observed_data_path)
        print(f"\n🎯 Final CNN Results:")
        print(f"   Decay Rate: {predictions[0]:.6f}")
        print(f"   Surface Transfer Fraction: {predictions[1]:.6f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()