#!/usr/bin/env python3
"""
Convolutional Neural Network implementation for 4x56 time series arrays.

This implementation treats the time series data as 4x56 arrays and uses 
convolutional layers to capture spatial and temporal dependencies.

Created for issue: Treat the time series data as a 4x56 array and use a convolutional network
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_4x56_data():
    """
    Load the reshaped data and create sliding windows of 4x56 arrays.
    
    Each sliding window represents 56 consecutive time points starting from tick 100.
    For each run, multiple windows are created: 100-155, 101-156, 102-157, etc.
    
    Returns:
        X: Array of shape (num_windows, 4, 56) containing time series data
        y: Array of shape (num_windows, 2) containing target variables  
        run_ids: Array of run identifiers for each window
        window_starts: Array of starting tick for each window
        metadata: Dictionary with variable names and other info
    """
    print("📥 Loading reshaped data and creating sliding windows...")
    
    # Load the saved data
    time_series_list = np.load('data/time_series_by_run.npy', allow_pickle=True)
    targets = np.load('data/targets_by_run.npy')
    run_ids = np.load('data/run_ids.npy')
    metadata = np.load('data/metadata.npy', allow_pickle=True).item()
    
    print(f"   Original data shape per run: {time_series_list[0].shape}")
    print(f"   Variable names: {metadata['variable_names']}")
    print(f"   Target variables: {metadata['target_names']}")
    
    # Time series mapping: tick 90 = index 0, tick 100 = index 10, etc.
    # Each time series has 276 time points (ticks 90-365)
    window_size = 56
    start_tick = 100
    tick_start_in_data = 90  # First tick in the time series data
    
    # Calculate the range of possible starting positions
    # tick 100 = index 10, last possible start for 56-window = tick 310 = index 220
    start_index = start_tick - tick_start_in_data  # 100 - 90 = 10
    max_data_index = time_series_list[0].shape[1] - 1  # 275 (for tick 365)
    last_start_index = max_data_index - window_size + 1  # 275 - 56 + 1 = 220
    
    last_start_tick = tick_start_in_data + last_start_index  # 90 + 220 = 310
    num_windows_per_run = last_start_index - start_index + 1  # 220 - 10 + 1 = 211
    
    print(f"   Creating sliding windows:")
    print(f"   - Window size: {window_size} time points")
    print(f"   - First window: ticks {start_tick}-{start_tick + window_size - 1}")
    print(f"   - Last window: ticks {last_start_tick}-{last_start_tick + window_size - 1}")
    print(f"   - Windows per run: {num_windows_per_run}")
    print(f"   - Total windows: {len(time_series_list) * num_windows_per_run}")
    
    # Create sliding windows
    all_windows = []
    all_targets = []
    all_run_ids = []
    all_window_starts = []
    
    for run_idx, (ts, target, run_id) in enumerate(zip(time_series_list, targets, run_ids)):
        for window_idx in range(num_windows_per_run):
            window_start_index = start_index + window_idx
            window_end_index = window_start_index + window_size
            
            # Extract the 4x56 window
            window_data = ts[:, window_start_index:window_end_index]  # Shape: (4, 56)
            
            # Calculate the actual tick for this window start
            window_start_tick = tick_start_in_data + window_start_index
            
            all_windows.append(window_data)
            all_targets.append(target)  # Same target for all windows from this run
            all_run_ids.append(run_id)
            all_window_starts.append(window_start_tick)
    
    # Convert to numpy arrays
    X = np.stack(all_windows)  # Shape: (total_windows, 4, 56)
    y = np.stack(all_targets)  # Shape: (total_windows, 2)
    run_ids_array = np.array(all_run_ids)
    window_starts_array = np.array(all_window_starts)
    
    print(f"   Final X shape: {X.shape}")
    print(f"   Final y shape: {y.shape}")
    print(f"   Run IDs shape: {run_ids_array.shape}")
    print(f"   Window starts shape: {window_starts_array.shape}")
    
    # Update metadata
    metadata_4x56 = metadata.copy()
    metadata_4x56['time_points'] = window_size
    metadata_4x56['window_range'] = f'{start_tick}-{start_tick + window_size - 1}'
    metadata_4x56['windows_per_run'] = num_windows_per_run
    metadata_4x56['total_windows'] = len(all_windows)
    metadata_4x56['sliding_windows'] = True
    
    return X, y, run_ids_array, window_starts_array, metadata_4x56

def create_cnn_model(input_shape=(4, 56, 1), learning_rate=0.001):
    """
    Create a CNN model for 4x56 time series arrays.
    
    Args:
        input_shape: Shape of input data (height, width, channels)
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    print(f"🧠 Creating CNN model with input shape: {input_shape}")
    
    model = Sequential([
        # First convolutional block
        Conv2D(32, (2, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(32, (2, 3), activation='relu', padding='same'),
        MaxPooling2D((1, 2)),  # Pool along time dimension
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (2, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (2, 3), activation='relu', padding='same'), 
        MaxPooling2D((1, 2)),  # Pool along time dimension
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (2, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  # Pool along both dimensions
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
        
        # Output layer (2 target variables)
        Dense(2, activation='linear')  # Linear for regression
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def normalize_data(X_train, X_test):
    """
    Normalize the input data by standardizing each variable across time and runs.
    
    Args:
        X_train: Training data of shape (n_train, 4, 56)
        X_test: Test data of shape (n_test, 4, 56)
        
    Returns:
        X_train_norm, X_test_norm: Normalized data
        scalers: List of scalers for each variable (for inverse transform if needed)
    """
    print("🔄 Normalizing data...")
    
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    scalers = []
    
    # Normalize each variable (across all time points and runs)
    for var_idx in range(4):  # 4 variables
        # Reshape to combine time and run dimensions for fitting scaler
        train_var_data = X_train[:, var_idx, :].reshape(-1, 1)  # Shape: (n_train*56, 1)
        
        # Fit scaler on training data
        scaler = StandardScaler()
        scaler.fit(train_var_data)
        scalers.append(scaler)
        
        # Transform training data
        train_normalized = scaler.transform(train_var_data).reshape(X_train.shape[0], -1)
        X_train_norm[:, var_idx, :] = train_normalized
        
        # Transform test data
        test_var_data = X_test[:, var_idx, :].reshape(-1, 1)
        test_normalized = scaler.transform(test_var_data).reshape(X_test.shape[0], -1)
        X_test_norm[:, var_idx, :] = test_normalized
    
    return X_train_norm, X_test_norm, scalers

def train_and_evaluate_cnn():
    """
    Complete pipeline to train and evaluate the CNN model on 4x56 data.
    """
    print("🎯 CNN Training Pipeline for 4x56 Time Series Arrays")
    print("="*60)
    
    # Load and prepare data
    X, y, run_ids, window_starts, metadata = load_and_prepare_4x56_data()
    
    # Split data
    print("\n📊 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test, run_ids_train, run_ids_test, window_starts_train, window_starts_test = train_test_split(
        X, y, run_ids, window_starts, test_size=0.2, random_state=42
    )
    
    print(f"   Training set: {X_train.shape[0]} windows")
    print(f"   Test set: {X_test.shape[0]} windows")
    
    # Normalize data
    X_train_norm, X_test_norm, scalers = normalize_data(X_train, X_test)
    
    # Reshape for CNN (add channel dimension)
    X_train_cnn = X_train_norm.reshape(X_train_norm.shape[0], 4, 56, 1)
    X_test_cnn = X_test_norm.reshape(X_test_norm.shape[0], 4, 56, 1)
    
    print(f"   CNN input shape: {X_train_cnn.shape}")
    
    # Create model
    model = create_cnn_model(input_shape=(4, 56, 1))
    
    # Print model summary
    print("\n🏗️  Model Architecture:")
    model.summary()
    
    # Train model
    print("\n🤖 Training CNN model...")
    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_test_cnn, y_test),
        epochs=100,
        batch_size=4,  # Small batch size due to limited data
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]
    )
    
    # Make predictions
    print("\n🎯 Making predictions...")
    y_pred = model.predict(X_test_cnn)
    
    # Evaluate performance
    print("\n📈 Model Performance:")
    print("-" * 40)
    
    target_names = metadata['target_names']
    for i, target_name in enumerate(target_names):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        print(f"\n{target_name}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {np.sqrt(mse):.6f}")
        print(f"  R²: {r2:.4f}")
        
        # Show range comparison
        actual_range = f"{y_test[:, i].min():.4f} - {y_test[:, i].max():.4f}"
        pred_range = f"{y_pred[:, i].min():.4f} - {y_pred[:, i].max():.4f}"
        print(f"  Actual range: {actual_range}")
        print(f"  Predicted range: {pred_range}")
    
    # Show sample predictions
    print("\n🔍 Sample Predictions:")
    print("-" * 70)
    print("Run ID | Window Start | Actual [decay, surface] | Predicted [decay, surface]")
    print("-" * 70)
    for i in range(min(5, len(y_test))):
        run_id = run_ids_test[i]
        window_start = window_starts_test[i]
        actual = f"[{y_test[i, 0]:.4f}, {y_test[i, 1]:.4f}]"
        predicted = f"[{y_pred[i, 0]:.4f}, {y_pred[i, 1]:.4f}]"
        print(f"{run_id:6d} | {window_start:12.0f} | {actual:23s} | {predicted}")
    
    # Training history
    print(f"\n📉 Training Summary:")
    print("-" * 40)
    final_epoch = len(history.history['loss'])
    print(f"   Epochs trained: {final_epoch}")
    print(f"   Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"   Final validation loss: {history.history['val_loss'][-1]:.6f}")
    print(f"   Best validation loss: {min(history.history['val_loss']):.6f}")
    
    print(f"\n✅ CNN training completed!")
    print(f"\nℹ️  This demonstrates sliding window approach with 4x56 time series arrays:")
    print(f"   • Creates {metadata['windows_per_run']} windows per run")
    print(f"   • Total dataset: {metadata['total_windows']} training examples")
    print(f"   • Each window: {metadata['window_range']} time points")
    print(f"   • 2D Convolutional layers to capture spatial patterns")
    print(f"   • MaxPooling to reduce dimensionality") 
    print(f"   • BatchNormalization for stable training")
    print(f"   • Dropout for regularization")
    print(f"   • Early stopping to prevent overfitting")
    
    return model, history, y_test, y_pred, run_ids_test

if __name__ == "__main__":
    # Run the complete pipeline
    model, history, y_test, y_pred, run_ids_test = train_and_evaluate_cnn()