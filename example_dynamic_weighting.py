"""
Example: Using Dynamic Variability Weighting

This example demonstrates how to use the dynamic weighting system
with a simple mock training scenario.
"""

import torch
import numpy as np
from train.variability_weighting import DynamicVariabilityWeighting


def example_basic_usage():
    """Basic usage example."""
    print("="*80)
    print("Example 1: Basic Usage")
    print("="*80)
    
    # Create the weighting system
    weighting = DynamicVariabilityWeighting(
        eeg_window_size=50,
        fmri_window_size=150,
        warmup_epochs=5,
        main_epochs=60,
        finetune_epochs=30,
        enabled=True
    )
    
    # Simulate EEG data (batch=1, time=200, channels=64)
    eeg_data = torch.randn(1, 200, 64)
    # Some channels have low variability (simulate silent channels)
    eeg_data[:, :, 50:60] *= 0.1
    
    # Compute weights at different epochs
    print("\nEEG weights across training stages:")
    print(f"{'Epoch':<10} {'Stage':<12} {'Temp':<10} {'Max Weight':<12} {'Min Weight':<12}")
    print("-" * 66)
    
    for epoch in [1, 5, 30, 66, 90]:
        weights = weighting.compute_modality_weights(eeg_data, 'eeg', epoch)
        stage_info = weighting.get_stage_info(epoch)
        
        print(f"{epoch:<10} {stage_info['stage']:<12} {stage_info['temperature']:<10.3f} "
              f"{weights.max():.6f}    {weights.min():.6f}")
    
    print("\n✓ Weights adapt across training stages")


def example_with_loss_computation():
    """Example showing how weights are used in loss computation."""
    print("\n" + "="*80)
    print("Example 2: Weighted Loss Computation")
    print("="*80)
    
    # Setup
    weighting = DynamicVariabilityWeighting(
        warmup_epochs=5,
        main_epochs=60,
        finetune_epochs=30,
        warmup_temp=0.1,
        enabled=True
    )
    
    # Mock data
    eeg_data = torch.randn(2, 200, 64)  # [batch, time, channels]
    eeg_data[:, :, 50:60] *= 0.1  # Low variance channels
    
    # Mock predictions and targets
    predictions = torch.randn(2, 200, 64)
    targets = eeg_data
    
    # Compute weights at warmup (epoch 1)
    epoch = 1
    weights = weighting.compute_modality_weights(eeg_data, 'eeg', epoch)
    
    # Standard MSE loss (no weighting)
    standard_loss = ((predictions - targets) ** 2).mean()
    
    # Weighted loss (per-channel)
    per_channel_loss = ((predictions - targets) ** 2).mean(dim=(0, 1))  # [channels]
    weighted_loss = (per_channel_loss * weights).sum()
    
    print(f"\nEpoch {epoch} (Warmup Stage):")
    print(f"  Standard MSE Loss:    {standard_loss:.6f}")
    print(f"  Weighted Loss:        {weighted_loss:.6f}")
    print(f"  Weight on silent channels (50-60): {weights[50:60].mean():.6f}")
    print(f"  Weight on active channels (0-50):  {weights[0:50].mean():.6f}")
    
    print("\n✓ Weighted loss focuses on active channels")


def example_comparison_enabled_disabled():
    """Compare behavior with weighting enabled vs disabled."""
    print("\n" + "="*80)
    print("Example 3: Enabled vs Disabled Comparison")
    print("="*80)
    
    # Create two systems
    weighting_enabled = DynamicVariabilityWeighting(enabled=True, warmup_temp=0.1)
    weighting_disabled = DynamicVariabilityWeighting(enabled=False)
    
    # Mock EEG data
    eeg_data = torch.randn(1, 200, 64)
    eeg_data[:, :, 50:60] *= 0.1
    
    # Compute weights
    weights_enabled = weighting_enabled.compute_modality_weights(eeg_data, 'eeg', epoch=1)
    weights_disabled = weighting_disabled.compute_modality_weights(eeg_data, 'eeg', epoch=1)
    
    print("\nWeights Comparison (Epoch 1, Warmup):")
    print(f"{'Mode':<15} {'Max Weight':<15} {'Min Weight':<15} {'Std':<15}")
    print("-" * 60)
    print(f"{'Enabled':<15} {weights_enabled.max():.6f}        {weights_enabled.min():.6f}        {weights_enabled.std():.6f}")
    print(f"{'Disabled':<15} {weights_disabled.max():.6f}        {weights_disabled.min():.6f}        {weights_disabled.std():.6f}")
    
    print("\n✓ Enabled mode creates focused weights, disabled mode is uniform")


def example_stage_progression():
    """Show how weights evolve across training stages."""
    print("\n" + "="*80)
    print("Example 4: Weight Evolution Across Stages")
    print("="*80)
    
    weighting = DynamicVariabilityWeighting(
        warmup_epochs=5,
        main_epochs=60,
        finetune_epochs=30,
        warmup_temp=0.1,
        main_temp_start=0.1,
        main_temp_end=1.0,
        finetune_temp=2.0,
        enabled=True
    )
    
    # Mock data with clear variability pattern
    eeg_data = torch.randn(1, 200, 64)
    eeg_data[:, :, 0:10] *= 3.0   # Very high variability
    eeg_data[:, :, 10:30] *= 1.5  # Medium variability
    eeg_data[:, :, 30:64] *= 0.2  # Low variability
    
    print("\nWeight Distribution Evolution:")
    print(f"{'Epoch':<8} {'Stage':<12} {'Top-10 Mean':<15} {'Mid-20 Mean':<15} {'Bottom-34 Mean':<15}")
    print("-" * 68)
    
    for epoch in [1, 5, 6, 30, 65, 66, 95]:
        weights = weighting.compute_modality_weights(eeg_data, 'eeg', epoch)
        stage_info = weighting.get_stage_info(epoch)
        
        top10 = weights[0:10].mean().item()
        mid20 = weights[10:30].mean().item()
        bottom34 = weights[30:64].mean().item()
        
        print(f"{epoch:<8} {stage_info['stage']:<12} {top10:<15.6f} {mid20:<15.6f} {bottom34:<15.6f}")
    
    print("\n✓ Weights gradually broaden from warmup to finetune")


def example_config_based():
    """Example using configuration dictionary."""
    print("\n" + "="*80)
    print("Example 5: Configuration-Based Initialization")
    print("="*80)
    
    # Configuration dictionary (similar to YAML config)
    config = {
        'dynamic_weighting': {
            'enabled': True,
            'eeg_window_size': 50,
            'fmri_window_size': 150,
            'min_weight': 0.01,
            'warmup_temp': 0.1,
            'main_temp_start': 0.1,
            'main_temp_end': 1.0,
            'finetune_temp': 2.0,
            'eeg_use_first_order_diff': True,
            'fmri_use_fc_change': True,
        },
        'training': {
            'warmup_epochs': 5,
            'main_epochs': 60,
            'finetune_epochs': 30,
        }
    }
    
    # Create from config
    from train.variability_weighting import create_dynamic_weighting_from_config
    weighting = create_dynamic_weighting_from_config(config)
    
    # Test it
    eeg_data = torch.randn(1, 200, 64)
    weights = weighting.compute_modality_weights(eeg_data, 'eeg', epoch=1)
    
    print("\nCreated from config:")
    print(f"  Enabled: {weighting.enabled}")
    print(f"  EEG window size: {weighting.eeg_computer.window_size}")
    print(f"  fMRI window size: {weighting.fmri_computer.window_size}")
    print(f"  Warmup epochs: {weighting.stage_scheduler.warmup_epochs}")
    print(f"  Computed weights shape: {weights.shape}")
    
    print("\n✓ Config-based initialization works")


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*80)
    print("DYNAMIC VARIABILITY WEIGHTING - USAGE EXAMPLES")
    print("="*80)
    
    try:
        example_basic_usage()
        example_with_loss_computation()
        example_comparison_enabled_disabled()
        example_stage_progression()
        example_config_based()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY ✓")
        print("="*80)
        print("\nKey Takeaways:")
        print("  1. Weights adapt across training stages (warmup → main → finetune)")
        print("  2. Weighted loss focuses gradient on high-variability channels")
        print("  3. Can be easily enabled/disabled for ablation studies")
        print("  4. Integrates seamlessly with existing config system")
        print("\nTo use in your training:")
        print("  Set 'dynamic_weighting.enabled: true' in config/default.yaml")
        
    except Exception as e:
        print(f"\n❌ EXAMPLE FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
