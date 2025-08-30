#!/usr/bin/env python3
"""
Test script to validate the 4x56 CNN implementation.
This ensures the new CNN approach works correctly with the data structure.
"""

import numpy as np
import sys
import os

# Add the current directory to path for imports
sys.path.append('/home/runner/work/parameter_estimation_mind/parameter_estimation_mind')

from cnn_4x56_model import load_and_prepare_4x56_data, create_cnn_model

def test_data_loading_and_structure():
    """Test that data loading works and produces correct 4x56 structure."""
    print("🧪 Testing data loading and 4x56 structure...")
    
    # Load data
    X, y, run_ids, metadata = load_and_prepare_4x56_data()
    
    # Test shapes
    assert X.shape[1] == 4, f"Expected 4 variables, got {X.shape[1]}"
    assert X.shape[2] == 56, f"Expected 56 time points, got {X.shape[2]}"
    assert y.shape[1] == 2, f"Expected 2 target variables, got {y.shape[1]}"
    assert len(run_ids) == X.shape[0], f"Number of run_ids ({len(run_ids)}) doesn't match data ({X.shape[0]})"
    
    print(f"   ✓ Data shape: {X.shape} (runs, variables, time_points)")
    print(f"   ✓ Target shape: {y.shape} (runs, target_variables)")
    print(f"   ✓ Run IDs: {len(run_ids)} unique runs")
    
    # Test that data values are reasonable
    assert not np.any(np.isnan(X)), "Data contains NaN values"
    assert not np.any(np.isnan(y)), "Targets contain NaN values"
    
    print(f"   ✓ No NaN values found")
    
    # Test metadata
    assert 'variable_names' in metadata, "Missing variable_names in metadata"
    assert 'target_names' in metadata, "Missing target_names in metadata"
    assert len(metadata['variable_names']) == 4, "Should have 4 variable names"
    assert len(metadata['target_names']) == 2, "Should have 2 target names"
    
    print(f"   ✓ Metadata structure correct")
    print(f"   ✓ Variables: {metadata['variable_names']}")
    print(f"   ✓ Targets: {metadata['target_names']}")
    
    return X, y, run_ids, metadata

def test_model_creation():
    """Test that the CNN model can be created with correct architecture."""
    print("\n🧪 Testing CNN model creation...")
    
    model = create_cnn_model(input_shape=(4, 56, 1))
    
    # Test model structure
    assert model is not None, "Model creation failed"
    
    # Test input shape
    expected_input_shape = (None, 4, 56, 1)
    actual_input_shape = model.input_shape
    assert actual_input_shape == expected_input_shape, f"Expected input shape {expected_input_shape}, got {actual_input_shape}"
    
    # Test output shape (should be 2 for our target variables)
    expected_output_shape = (None, 2)
    actual_output_shape = model.output_shape
    assert actual_output_shape == expected_output_shape, f"Expected output shape {expected_output_shape}, got {actual_output_shape}"
    
    print(f"   ✓ Model created successfully")
    print(f"   ✓ Input shape: {actual_input_shape}")
    print(f"   ✓ Output shape: {actual_output_shape}")
    print(f"   ✓ Total parameters: {model.count_params():,}")
    
    return model

def test_data_preprocessing():
    """Test that data can be correctly preprocessed for CNN."""
    print("\n🧪 Testing data preprocessing...")
    
    # Load test data
    X, y, run_ids, metadata = load_and_prepare_4x56_data()
    
    # Test reshape for CNN
    X_cnn = X.reshape(X.shape[0], 4, 56, 1)
    expected_shape = (X.shape[0], 4, 56, 1)
    assert X_cnn.shape == expected_shape, f"Expected CNN shape {expected_shape}, got {X_cnn.shape}"
    
    print(f"   ✓ CNN reshape: {X.shape} -> {X_cnn.shape}")
    
    # Test that model can process the data
    model = create_cnn_model(input_shape=(4, 56, 1))
    
    # Test prediction (should not crash)
    try:
        predictions = model.predict(X_cnn[:1], verbose=0)  # Test with 1 sample
        assert predictions.shape == (1, 2), f"Expected prediction shape (1, 2), got {predictions.shape}"
        print(f"   ✓ Model prediction test passed: {predictions.shape}")
    except Exception as e:
        assert False, f"Model prediction failed: {e}"
    
    return True

def test_data_truncation():
    """Test that the time series is correctly truncated to days 100-155."""
    print("\n🧪 Testing time series truncation (days 100-155)...")
    
    # Load original data for comparison
    time_series_list = np.load('data/time_series_by_run.npy', allow_pickle=True)
    original_length = time_series_list[0].shape[1]  # Should be 276
    
    # Load truncated data
    X, y, run_ids, metadata = load_and_prepare_4x56_data()
    truncated_length = X.shape[2]  # Should be 56
    
    print(f"   ✓ Original length: {original_length} time points")
    print(f"   ✓ Truncated length: {truncated_length} time points")
    
    # Verify truncation
    assert original_length == 276, f"Expected original length 276, got {original_length}"
    assert truncated_length == 56, f"Expected truncated length 56, got {truncated_length}"
    
    # Test that truncation corresponds to correct time range
    # Days 100-155 correspond to indices 10:66 (tick 90=index 0, tick 100=index 10)
    start_idx, end_idx = 10, 66
    expected_truncated = time_series_list[0][:, start_idx:end_idx]
    actual_truncated = X[0]
    
    assert np.allclose(expected_truncated, actual_truncated), "Truncation doesn't match expected range"
    
    print(f"   ✓ Truncation correctly extracts days 100-155 (indices {start_idx}:{end_idx})")
    print(f"   ✓ Truncated data matches expected range")
    
    return True

def test_comparison_with_original_data():
    """Test that our 4x56 approach preserves data integrity."""
    print("\n🧪 Testing data integrity comparison...")
    
    # Load our 4x56 data
    X_4x56, y_4x56, run_ids_4x56, metadata_4x56 = load_and_prepare_4x56_data()
    
    # Load original data
    time_series_list = np.load('data/time_series_by_run.npy', allow_pickle=True)
    targets_original = np.load('data/targets_by_run.npy')
    run_ids_original = np.load('data/run_ids.npy')
    
    # Test that run_ids match
    assert np.array_equal(run_ids_4x56, run_ids_original), "Run IDs don't match between datasets"
    
    # Test that targets match
    assert np.allclose(y_4x56, targets_original), "Target variables don't match between datasets"
    
    # Test that truncated time series are subset of original
    for i in range(len(run_ids_4x56)):
        original_ts = time_series_list[i][:, 10:66]  # Days 100-155
        truncated_ts = X_4x56[i]
        assert np.allclose(original_ts, truncated_ts), f"Run {run_ids_4x56[i]} truncated data doesn't match"
    
    print(f"   ✓ Run IDs preserved: {len(run_ids_4x56)} runs")
    print(f"   ✓ Target variables preserved")
    print(f"   ✓ Truncated time series are correct subsets of original data")
    
    return True

def run_all_tests():
    """Run all tests for the 4x56 CNN implementation."""
    print("🎯 Testing 4x56 CNN Implementation")
    print("=" * 50)
    
    try:
        # Test data loading
        test_data_loading_and_structure()
        
        # Test model creation
        test_model_creation()
        
        # Test data preprocessing
        test_data_preprocessing()
        
        # Test truncation logic
        test_data_truncation()
        
        # Test data integrity
        test_comparison_with_original_data()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("The 4x56 CNN implementation is working correctly.")
        print("\nKey features validated:")
        print("  • Data correctly truncated to 56 time points (days 100-155)")
        print("  • 4x56 array structure maintained")
        print("  • CNN model architecture accepts 4x56x1 input")
        print("  • Data integrity preserved from original dataset")
        print("  • Model can make predictions on the data")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)