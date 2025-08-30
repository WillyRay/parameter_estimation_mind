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
    Load the reshaped data and truncate to 4x56 arrays (days 100-155).
    
    Returns:
        X: Array of shape (num_runs, 4, 56) containing time series data
        y: Array of shape (num_runs, 2) containing target variables
        run_ids: Array of run identifiers
        metadata: Dictionary with variable names and other info
    """
    print("📥 Loading reshaped data...")
    
    # Load the saved data
    time_series_list = np.load('data/time_series_by_run.npy', allow_pickle=True)
    targets = np.load('data/targets_by_run.npy')
    run_ids = np.load('data/run_ids.npy')
    metadata = np.load('data/metadata.npy', allow_pickle=True).item()
    
    print(f"   Original data shape per run: {time_series_list[0].shape}")
    print(f"   Variable names: {metadata['variable_names']}")
    print(f"   Target variables: {metadata['target_names']}")
    
    # The original data starts at tick 90, so:
    # - tick 90 = index 0
    # - tick 100 = index 10  
    # - tick 155 = index 65
    # We want days 100-155 (56 days total)
    start_day_index = 10  # tick 100
    end_day_index = 66    # tick 155 + 1 (exclusive)
    
    print(f"   Truncating to days 100-155 (indices {start_day_index}:{end_day_index})")
    
    # Truncate all time series to 4x56
    truncated_time_series = []
    for ts in time_series_list:
        truncated_ts = ts[:, start_day_index:end_day_index]  # Shape: (4, 56)
        truncated_time_series.append(truncated_ts)
    
    # Convert to numpy array
    X = np.stack(truncated_time_series)  # Shape: (num_runs, 4, 56)
    y = targets  # Shape: (num_runs, 2)
    
    print(f"   Final X shape: {X.shape}")
    print(f"   Final y shape: {y.shape}")
    
    # Update metadata
    metadata_4x56 = metadata.copy()
    metadata_4x56['time_points'] = 56
    metadata_4x56['day_range'] = '100-155'
    
    return X, y, run_ids, metadata_4x56

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
    X, y, run_ids, metadata = load_and_prepare_4x56_data()
    
    # Split data
    print("\n📊 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test, run_ids_train, run_ids_test = train_test_split(
        X, y, run_ids, test_size=0.2, random_state=42
    )
    
    print(f"   Training set: {X_train.shape[0]} runs")
    print(f"   Test set: {X_test.shape[0]} runs")
    
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
    print("-" * 40)
    print("Run ID | Actual [decay, surface] | Predicted [decay, surface]")
    print("-" * 65)
    for i in range(min(5, len(y_test))):
        run_id = run_ids_test[i]
        actual = f"[{y_test[i, 0]:.4f}, {y_test[i, 1]:.4f}]"
        predicted = f"[{y_pred[i, 0]:.4f}, {y_pred[i, 1]:.4f}]"
        print(f"{run_id:6d} | {actual:23s} | {predicted}")
    
    # Training history
    print(f"\n📉 Training Summary:")
    print("-" * 40)
    final_epoch = len(history.history['loss'])
    print(f"   Epochs trained: {final_epoch}")
    print(f"   Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"   Final validation loss: {history.history['val_loss'][-1]:.6f}")
    print(f"   Best validation loss: {min(history.history['val_loss']):.6f}")
    
    print(f"\n✅ CNN training completed!")
    print(f"\nℹ️  This demonstrates how 4x56 time series arrays can be used with:")
    print(f"   • 2D Convolutional layers to capture spatial patterns")
    print(f"   • MaxPooling to reduce dimensionality") 
    print(f"   • BatchNormalization for stable training")
    print(f"   • Dropout for regularization")
    print(f"   • Early stopping to prevent overfitting")
    
    return model, history, y_test, y_pred, run_ids_test

if __name__ == "__main__":
    # Run the complete pipeline
    model, history, y_test, y_pred, run_ids_test = train_and_evaluate_cnn()