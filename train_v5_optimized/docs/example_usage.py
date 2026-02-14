"""
Quick Start Example for TwinBrain V5 Optimizations
===================================================

This example demonstrates how to integrate V5 optimizations
into your existing training pipeline.

Run this example:
    python train_v5_optimized/example_usage.py
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

# Import V5 optimizations
from train_v5_optimized import (
    AdaptiveLossBalancer,
    EnhancedEEGHandler,
    EnhancedMultiStepPredictor,
    get_default_config,
)


def example_adaptive_loss_balancing():
    """Example: Using adaptive loss balancing."""
    print("\n" + "="*60)
    print("Example 1: Adaptive Loss Balancing")
    print("="*60)
    
    # Initialize balancer
    balancer = AdaptiveLossBalancer(
        task_names=['reconstruction', 'temporal_prediction', 'alignment'],
        modality_names=['eeg', 'fmri'],
        modality_energy_ratios={'eeg': 0.02, 'fmri': 1.0},
        alpha=1.5,
        learning_rate=0.025,
        warmup_epochs=5,
    )
    
    # Simulate training loop
    print("\nSimulating training with adaptive loss balancing...")
    
    for epoch in range(1, 11):
        balancer.set_epoch(epoch)
        
        # Simulate losses
        losses = {
            'reconstruction': torch.tensor(1.5) + torch.randn(1) * 0.1,
            'temporal_prediction': torch.tensor(0.8) + torch.randn(1) * 0.05,
            'alignment': torch.tensor(2.0) + torch.randn(1) * 0.2,
        }
        
        # Compute weighted loss
        total_loss, weights = balancer(losses)
        
        # Print current weights
        if epoch % 2 == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Losses: {' '.join(f'{k}={v.item():.3f}' for k, v in losses.items())}")
            print(f"  Weights: {' '.join(f'{k}={v:.3f}' for k, v in weights.items())}")
            print(f"  Total: {total_loss.item():.3f}")
    
    print("\n✓ Adaptive loss balancing maintains balanced training")


def example_eeg_channel_handling():
    """Example: Enhanced EEG channel handling."""
    print("\n" + "="*60)
    print("Example 2: Enhanced EEG Channel Handling")
    print("="*60)
    
    # Initialize EEG handler
    num_channels = 64
    handler = EnhancedEEGHandler(
        num_channels=num_channels,
        enable_monitoring=True,
        enable_dropout=True,
        enable_attention=True,
        enable_regularization=True,
        dropout_rate=0.1,
    )
    
    # Simulate EEG data
    batch_size = 4
    seq_len = 200
    eeg_data = torch.randn(batch_size, seq_len, num_channels)
    
    # Add some silent channels (simulate low-energy channels)
    silent_channels = [10, 20, 30]
    eeg_data[:, :, silent_channels] *= 0.01  # Very low energy
    
    print(f"\nProcessing EEG data:")
    print(f"  Shape: {eeg_data.shape}")
    print(f"  Silent channels (simulated): {silent_channels}")
    
    # Process with handler
    eeg_processed, info = handler(eeg_data, training=True)
    
    print(f"\nProcessing results:")
    print(f"  Processed shape: {eeg_processed.shape}")
    if 'channel_health' in info:
        print(f"  Average channel health: {info['channel_health']:.3f}")
    if 'regularization_loss' in info:
        print(f"  Regularization loss: {info['regularization_loss'].item():.4f}")
    
    # Simulate multiple updates to show monitoring
    print("\nSimulating training updates...")
    for step in range(5):
        eeg_data_new = torch.randn(batch_size, seq_len, num_channels)
        eeg_data_new[:, :, silent_channels] *= 0.01
        
        eeg_processed, info = handler(eeg_data_new, training=True)
        
        if step % 2 == 0 and 'channel_health' in info:
            print(f"  Step {step}: health={info['channel_health']:.3f}")
    
    print("\n✓ EEG handler successfully monitors and processes channels")


def example_advanced_prediction():
    """Example: Advanced multi-step prediction."""
    print("\n" + "="*60)
    print("Example 3: Advanced Multi-Step Prediction")
    print("="*60)
    
    # Initialize predictor
    input_dim = 128
    predictor = EnhancedMultiStepPredictor(
        input_dim=input_dim,
        hidden_dim=256,
        context_length=50,
        prediction_steps=10,
        use_hierarchical=True,
        use_transformer=True,
        use_uncertainty=True,
        num_scales=3,
        num_windows=3,
    )
    
    # Simulate sequences
    batch_size = 2
    seq_len = 200
    sequences = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"\nPredicting with advanced predictor:")
    print(f"  Input shape: {sequences.shape}")
    print(f"  Context length: 50")
    print(f"  Prediction steps: 10")
    print(f"  Number of scales: 3")
    print(f"  Number of windows: 3")
    
    # Predict
    predictions, targets, uncertainties = predictor(
        sequences=sequences,
        return_uncertainty=True,
    )
    
    print(f"\nPrediction results:")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Targets shape: {targets.shape}")
    if uncertainties is not None:
        print(f"  Uncertainties shape: {uncertainties.shape}")
        print(f"  Mean uncertainty: {uncertainties.mean().item():.4f}")
    
    # Compute loss
    loss = predictor.compute_loss(predictions, targets, uncertainties)
    print(f"  Prediction loss: {loss.item():.4f}")
    
    print("\n✓ Advanced predictor successfully predicts with uncertainty")


def example_full_integration():
    """Example: Full integration of all V5 components."""
    print("\n" + "="*60)
    print("Example 4: Full Integration")
    print("="*60)
    
    # Get default config
    config = get_default_config()
    
    print("\nV5 Optimization Configuration:")
    for component, comp_config in config.items():
        print(f"\n{component}:")
        for key, value in comp_config.items():
            print(f"  {key}: {value}")
    
    # Initialize all components
    print("\nInitializing all V5 components...")
    
    # 1. Adaptive loss balancer
    loss_balancer = AdaptiveLossBalancer(
        task_names=['recon', 'temp_pred', 'align'],
        modality_names=['eeg', 'fmri'],
        **{k: v for k, v in config['adaptive_loss'].items() 
           if k not in ['enabled', 'modality_energy_ratios']},
        modality_energy_ratios=config['adaptive_loss']['modality_energy_ratios'],
    )
    print("  ✓ Adaptive loss balancer initialized")
    
    # 2. EEG handler
    eeg_handler = EnhancedEEGHandler(
        num_channels=64,
        **{k: v for k, v in config['eeg_enhancement'].items() if k != 'enabled'},
    )
    print("  ✓ EEG handler initialized")
    
    # 3. Advanced predictor
    predictor = EnhancedMultiStepPredictor(
        input_dim=128,
        context_length=50,
        prediction_steps=10,
        **{k: v for k, v in config['advanced_prediction'].items() if k != 'enabled'},
    )
    print("  ✓ Advanced predictor initialized")
    
    print("\n✓ All V5 components ready for training!")
    print("\nTo integrate into your trainer:")
    print("  1. Add these components to your trainer's __init__")
    print("  2. Use adaptive_loss_balancer for loss weighting")
    print("  3. Process EEG data through eeg_handler")
    print("  4. Use advanced predictor for multi-step prediction")


def main():
    """Run all examples."""
    print("="*60)
    print("TwinBrain V5 Optimizations - Quick Start Examples")
    print("="*60)
    
    try:
        example_adaptive_loss_balancing()
    except Exception as e:
        print(f"\n✗ Example 1 failed: {e}")
    
    try:
        example_eeg_channel_handling()
    except Exception as e:
        print(f"\n✗ Example 2 failed: {e}")
    
    try:
        example_advanced_prediction()
    except Exception as e:
        print(f"\n✗ Example 3 failed: {e}")
    
    try:
        example_full_integration()
    except Exception as e:
        print(f"\n✗ Example 4 failed: {e}")
    
    print("\n" + "="*60)
    print("Quick Start Examples Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review the V5 README: train_v5_optimized/README.md")
    print("  2. Check configuration: train_v5_optimized/__init__.py::get_default_config()")
    print("  3. Integrate into your training workflow")
    print("  4. Monitor improvements in training metrics")
    print("\nFor questions or issues, refer to the documentation.")


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run examples
    main()
