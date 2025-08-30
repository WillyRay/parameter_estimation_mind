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
    """Test that data loading works and produces correct 4x56 sliding window structure."""
    print("🧪 Testing data loading and 4x56 sliding window structure...")
    
    # Load data
    X, y, run_ids, window_starts, metadata = load_and_prepare_4x56_data()
    
    # Test shapes
    assert X.shape[1] == 4, f"Expected 4 variables, got {X.shape[1]}"
    assert X.shape[2] == 56, f"Expected 56 time points, got {X.shape[2]}"
    assert y.shape[1] == 2, f"Expected 2 target variables, got {y.shape[1]}"
    assert len(run_ids) == X.shape[0], f"Number of run_ids ({len(run_ids)}) doesn't match data ({X.shape[0]})"
    assert len(window_starts) == X.shape[0], f"Number of window_starts ({len(window_starts)}) doesn't match data ({X.shape[0]})"
    
    print(f"   ✓ Data shape: {X.shape} (windows, variables, time_points)")
    print(f"   ✓ Target shape: {y.shape} (windows, target_variables)")
    print(f"   ✓ Window starts: {len(window_starts)} windows")
    print(f"   ✓ Run IDs: {len(np.unique(run_ids))} unique runs, {len(run_ids)} total windows")
    
    # Test that we have the expected number of windows
    expected_windows_per_run = metadata.get('windows_per_run', 0)
    unique_runs = len(np.unique(run_ids))
    expected_total = unique_runs * expected_windows_per_run
    
    print(f"   ✓ Expected {expected_windows_per_run} windows per run")
    print(f"   ✓ Expected total: {expected_total}, actual: {X.shape[0]}")
    
    # Test that data values are reasonable
    assert not np.any(np.isnan(X)), "Data contains NaN values"
    assert not np.any(np.isnan(y)), "Targets contain NaN values"
    
    print(f"   ✓ No NaN values found")
    
    # Test sliding window properties
    assert np.min(window_starts) >= 100, f"Window starts should be >= 100, got {np.min(window_starts)}"
    window_range = np.max(window_starts) - np.min(window_starts) + 1
    print(f"   ✓ Window start range: {np.min(window_starts)} to {np.max(window_starts)} ({window_range} different starts)")
    
    # Test metadata
    assert 'variable_names' in metadata, "Missing variable_names in metadata"
    assert 'target_names' in metadata, "Missing target_names in metadata"
    assert len(metadata['variable_names']) == 4, "Should have 4 variable names"
    assert len(metadata['target_names']) == 2, "Should have 2 target names"
    assert metadata.get('sliding_windows', False), "Should indicate sliding windows are used"
    
    print(f"   ✓ Metadata structure correct")
    print(f"   ✓ Variables: {metadata['variable_names']}")
    print(f"   ✓ Targets: {metadata['target_names']}")
    
    return X, y, run_ids, window_starts, metadata

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
    X, y, run_ids, window_starts, metadata = load_and_prepare_4x56_data()
    
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

def test_sliding_windows():
    """Test that sliding windows are correctly created."""
    print("\n🧪 Testing sliding window creation...")
    
    # Load original data for comparison
    time_series_list = np.load('data/time_series_by_run.npy', allow_pickle=True)
    original_length = time_series_list[0].shape[1]  # Should be 276
    
    # Load sliding window data
    X, y, run_ids, window_starts, metadata = load_and_prepare_4x56_data()
    window_length = X.shape[2]  # Should be 56
    
    print(f"   ✓ Original length: {original_length} time points")
    print(f"   ✓ Window length: {window_length} time points")
    print(f"   ✓ Total windows: {X.shape[0]}")
    
    # Verify window properties
    assert original_length == 276, f"Expected original length 276, got {original_length}"
    assert window_length == 56, f"Expected window length 56, got {window_length}"
    
    # Test that we have multiple windows per run
    unique_runs = len(np.unique(run_ids))
    windows_per_run = X.shape[0] // unique_runs
    print(f"   ✓ Unique runs: {unique_runs}")
    print(f"   ✓ Windows per run: {windows_per_run}")
    
    assert windows_per_run > 1, f"Expected multiple windows per run, got {windows_per_run}"
    
    # Test sliding window properties
    # First window should start at tick 100, last should be much later
    min_start = np.min(window_starts)
    max_start = np.max(window_starts)
    print(f"   ✓ Window starts range: {min_start} to {max_start}")
    
    assert min_start == 100, f"Expected first window to start at tick 100, got {min_start}"
    assert max_start > min_start + 100, f"Expected significant range in window starts, got {max_start - min_start}"
    
    # Test that consecutive windows from same run are shifted by 1
    # Find windows from first run
    first_run = np.unique(run_ids)[0]
    first_run_mask = run_ids == first_run
    first_run_starts = window_starts[first_run_mask]
    first_run_starts_sorted = np.sort(first_run_starts)
    
    # Check that consecutive starts differ by 1
    if len(first_run_starts_sorted) > 1:
        diffs = np.diff(first_run_starts_sorted)
        assert np.all(diffs == 1), f"Expected consecutive windows to differ by 1 tick, got {diffs[:5]}"
        print(f"   ✓ Consecutive windows are shifted by 1 tick")
    
    return True

def test_comparison_with_original_data():
    """Test that our sliding window approach preserves data integrity."""
    print("\n🧪 Testing data integrity comparison...")
    
    # Load our sliding window data
    X_4x56, y_4x56, run_ids_4x56, window_starts_4x56, metadata_4x56 = load_and_prepare_4x56_data()
    
    # Load original data
    time_series_list = np.load('data/time_series_by_run.npy', allow_pickle=True)
    targets_original = np.load('data/targets_by_run.npy')
    run_ids_original = np.load('data/run_ids.npy')
    
    # Test that all original runs are represented
    unique_runs_4x56 = np.unique(run_ids_4x56)
    assert np.array_equal(unique_runs_4x56, run_ids_original), "Not all original runs are represented"
    
    # Test that targets are correctly replicated for sliding windows
    for original_run_id, original_target in zip(run_ids_original, targets_original):
        # Get all windows for this run
        run_mask = run_ids_4x56 == original_run_id
        run_targets = y_4x56[run_mask]
        
        # All targets for this run should be identical to the original
        assert np.all(np.allclose(run_targets, original_target)), f"Targets for run {original_run_id} don't match"
    
    # Test that sliding windows contain correct data
    # Test first window of first run should match original data at correct indices
    first_run = run_ids_original[0]
    first_run_mask = run_ids_4x56 == first_run
    first_run_starts = window_starts_4x56[first_run_mask]
    first_window_idx = np.where(first_run_starts == 100)[0][0]  # Find window starting at tick 100
    
    # Get the first window and compare with original
    first_window = X_4x56[first_run_mask][first_window_idx]  # Shape: (4, 56)
    original_first_run = time_series_list[0][:, 10:66]  # Tick 100-155 = indices 10:66
    
    assert np.allclose(first_window, original_first_run), "First sliding window doesn't match expected original data slice"
    
    print(f"   ✓ All {len(run_ids_original)} original runs represented in sliding windows")
    print(f"   ✓ Target variables correctly replicated across windows")  
    print(f"   ✓ Sliding windows contain correct data slices from original")
    print(f"   ✓ Total sliding windows created: {len(X_4x56)}")
    
    return True

def run_all_tests():
    """Run all tests for the 4x56 CNN implementation."""
    print("🎯 Testing 4x56 CNN Implementation")
    print("=" * 50)
    
    try:
        # Test data loading
        X, y, run_ids, window_starts, metadata_4x56 = test_data_loading_and_structure()
        
        # Test model creation
        test_model_creation()
        
        # Test data preprocessing
        test_data_preprocessing()
        
        # Test sliding window creation
        test_sliding_windows()
        
        # Test data integrity
        test_comparison_with_original_data()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("The 4x56 CNN implementation is working correctly.")
        print("\nKey features validated:")
        print("  • Sliding windows correctly created from time series data")
        print("  • Multiple 56-point windows per run (100-155, 101-156, etc.)")
        print("  • 4x56 array structure maintained for each window")
        print("  • CNN model architecture accepts 4x56x1 input")
        print("  • Data integrity preserved from original dataset")
        print("  • Model can make predictions on the sliding window data")
        print(f"  • Total training examples: {metadata_4x56.get('total_windows', 'unknown')}")
        
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