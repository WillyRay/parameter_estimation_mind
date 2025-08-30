#!/usr/bin/env python3
"""
Comprehensive prediction script that uses both approaches to predict 
decayRate and surfaceTransferFraction for the observed data.

This script provides predictions from:
1. Flattened Neural Network approach (original method)
2. CNN with sliding windows approach (new 4x56 method)
"""

import numpy as np
import pandas as pd
import subprocess
import sys

def run_flattened_prediction():
    """Run the original flattened neural network prediction."""
    print("🔬 Running Flattened Neural Network Prediction...")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, 'predict_observed_data.py'], 
                              capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            # Extract the final predictions from output
            lines = result.stdout.split('\n')
            decay_rate = None
            surface_transfer = None
            
            for line in lines:
                if 'Decay Rate:' in line:
                    decay_rate = float(line.split(':')[1].strip())
                elif 'Surface Transfer Fraction:' in line:
                    surface_transfer = float(line.split(':')[1].strip())
            
            print(f"✅ Flattened NN Results:")
            print(f"   Decay Rate: {decay_rate:.6f}")
            print(f"   Surface Transfer Fraction: {surface_transfer:.6f}")
            
            return decay_rate, surface_transfer
        else:
            print(f"❌ Error running flattened prediction: {result.stderr}")
            return None, None
            
    except Exception as e:
        print(f"❌ Exception in flattened prediction: {e}")
        return None, None

def run_cnn_prediction():
    """Run the CNN sliding window prediction."""
    print("\n🧠 Running CNN Sliding Window Prediction...")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, 'predict_with_cnn.py'], 
                              capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            # Extract the final predictions from output
            lines = result.stdout.split('\n')
            decay_rate = None
            surface_transfer = None
            
            for line in lines:
                if 'Decay Rate:' in line and 'Final CNN Results:' in result.stdout:
                    # Get the last occurrence (final results)
                    decay_rate = float(line.split(':')[1].strip())
                elif 'Surface Transfer Fraction:' in line and 'Final CNN Results:' in result.stdout:
                    surface_transfer = float(line.split(':')[1].strip())
            
            print(f"✅ CNN Results:")
            print(f"   Decay Rate: {decay_rate:.6f}")
            print(f"   Surface Transfer Fraction: {surface_transfer:.6f}")
            
            return decay_rate, surface_transfer
        else:
            print(f"❌ Error running CNN prediction: {result.stderr}")
            return None, None
            
    except Exception as e:
        print(f"❌ Exception in CNN prediction: {e}")
        return None, None

def summarize_predictions(flattened_results, cnn_results):
    """Provide a comprehensive summary of both predictions."""
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE PREDICTION SUMMARY FOR OBSERVED DATA")
    print("="*80)
    
    print(f"\n📊 Prediction Results Comparison:")
    print("-" * 60)
    print(f"{'Method':<25} {'Decay Rate':<15} {'Surface Transfer':<15}")
    print("-" * 60)
    
    if flattened_results[0] is not None:
        print(f"{'Flattened Neural Net':<25} {flattened_results[0]:<15.6f} {flattened_results[1]:<15.6f}")
    else:
        print(f"{'Flattened Neural Net':<25} {'Failed':<15} {'Failed':<15}")
    
    if cnn_results[0] is not None:
        print(f"{'CNN Sliding Windows':<25} {cnn_results[0]:<15.6f} {cnn_results[1]:<15.6f}")
    else:
        print(f"{'CNN Sliding Windows':<25} {'Failed':<15} {'Failed':<15}")
    
    print("-" * 60)
    
    # Calculate differences if both methods worked
    if (flattened_results[0] is not None and cnn_results[0] is not None):
        decay_diff = abs(flattened_results[0] - cnn_results[0])
        surface_diff = abs(flattened_results[1] - cnn_results[1])
        
        print(f"\n📈 Method Comparison:")
        print(f"   Decay Rate difference: {decay_diff:.6f}")
        print(f"   Surface Transfer difference: {surface_diff:.6f}")
        
        # Average predictions
        avg_decay = (flattened_results[0] + cnn_results[0]) / 2
        avg_surface = (flattened_results[1] + cnn_results[1]) / 2
        
        print(f"\n🎯 Ensemble Average Predictions:")
        print(f"   Average Decay Rate: {avg_decay:.6f}")
        print(f"   Average Surface Transfer Fraction: {avg_surface:.6f}")
        
        # Provide confidence assessment
        if decay_diff < 0.1 and surface_diff < 0.1:
            confidence = "🟢 High confidence - methods agree closely"
        elif decay_diff < 0.2 and surface_diff < 0.2:
            confidence = "🟡 Moderate confidence - some disagreement between methods"
        else:
            confidence = "🔴 Low confidence - significant disagreement between methods"
        
        print(f"\n🔍 Prediction Confidence: {confidence}")
    
    print(f"\n📝 Method Details:")
    print(f"   • Flattened Neural Net: Uses 224 features (4 variables × 56 days)")
    print(f"   • CNN Sliding Windows: Uses 4×56 arrays with 4,431 training examples")
    print(f"   • Both methods trained on simulation data with sliding windows")
    
    print(f"\n✅ Prediction analysis completed!")

def main():
    """Main function to run both prediction approaches and summarize results."""
    print("🚀 PARAMETER ESTIMATION FOR OBSERVED DATA")
    print("="*80)
    print("Running both prediction approaches for comprehensive analysis...")
    
    # Run both prediction methods
    flattened_results = run_flattened_prediction() 
    cnn_results = run_cnn_prediction()
    
    # Summarize and compare results
    summarize_predictions(flattened_results, cnn_results)
    
    # Return the best estimate
    if flattened_results[0] is not None and cnn_results[0] is not None:
        # Use ensemble average as best estimate
        best_decay = (flattened_results[0] + cnn_results[0]) / 2
        best_surface = (flattened_results[1] + cnn_results[1]) / 2
        
        print(f"\n🏆 FINAL RECOMMENDATION:")
        print(f"   Estimated Decay Rate: {best_decay:.6f}")
        print(f"   Estimated Surface Transfer Fraction: {best_surface:.6f}")
        
        return best_decay, best_surface
    elif flattened_results[0] is not None:
        print(f"\n🏆 FINAL RECOMMENDATION (Flattened NN only):")
        print(f"   Estimated Decay Rate: {flattened_results[0]:.6f}")
        print(f"   Estimated Surface Transfer Fraction: {flattened_results[1]:.6f}")
        return flattened_results
    elif cnn_results[0] is not None:
        print(f"\n🏆 FINAL RECOMMENDATION (CNN only):")
        print(f"   Estimated Decay Rate: {cnn_results[0]:.6f}")
        print(f"   Estimated Surface Transfer Fraction: {cnn_results[1]:.6f}")
        return cnn_results
    else:
        print(f"\n❌ PREDICTION FAILED - Both methods encountered errors")
        return None, None

if __name__ == "__main__":
    results = main()