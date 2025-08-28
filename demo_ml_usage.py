#!/usr/bin/env python3
"""
Demo script showing how to use the training dataset for machine learning.
This demonstrates the intended use case for predicting decayRate and surfaceTransferFraction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def demo_ml_usage(training_data_path):
    """Demonstrate ML usage of the training dataset."""
    print("🔬 Machine Learning Demo with Training Dataset")
    print("=" * 50)
    
    # Load the training data
    print("📥 Loading training data...")
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
    
    # Train a simple model
    print("\n🤖 Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Make predictions
    print("\n🎯 Making predictions...")
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    print("\n📈 Model Performance:")
    print("-" * 30)
    
    # Calculate metrics for each target
    target_names = ['decayRate', 'surfaceTransferFraction']
    for i, target_name in enumerate(target_names):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        print(f"{target_name}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  R²:  {r2:.4f}")
        print(f"  RMSE: {np.sqrt(mse):.6f}")
        
        # Show actual vs predicted range
        actual_range = f"{y_test[:, i].min():.4f} - {y_test[:, i].max():.4f}"
        pred_range = f"{y_pred[:, i].min():.4f} - {y_pred[:, i].max():.4f}"
        print(f"  Actual range: {actual_range}")
        print(f"  Predicted range: {pred_range}")
        print()
    
    # Show some example predictions
    print("🔍 Sample Predictions (first 5):")
    print("-" * 30)
    print("   Actual vs Predicted [decayRate, surfaceTransferFraction]")
    for i in range(min(5, len(y_test))):
        actual = f"[{y_test[i, 0]:.4f}, {y_test[i, 1]:.4f}]"
        predicted = f"[{y_pred[i, 0]:.4f}, {y_pred[i, 1]:.4f}]"
        print(f"   {actual} -> {predicted}")
    
    # Show feature importance
    print("\n📊 Most Important Features (top 10):")
    print("-" * 30)
    feature_importance = model.feature_importances_
    feature_names = feature_cols
    
    # Get top 10 most important features
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for i, idx in enumerate(top_indices):
        importance = feature_importance[idx]
        feature_name = feature_names[idx]
        print(f"   {i+1:2d}. {feature_name:15s} ({importance:.4f})")
    
    print("\n✅ Demo completed successfully!")
    print("\nℹ️  This demonstrates how the training dataset can be used to:")
    print("   • Train ML models to predict decayRate and surfaceTransferFraction")
    print("   • Use 56-day time series as input features")
    print("   • Evaluate model performance on unseen data")
    print("   • Analyze which time points are most predictive")

if __name__ == "__main__":
    demo_ml_usage("./data/training_data.csv")