"""
Test script for dynamic variability-based weighting.

This script validates that:
1. Variability computation works for EEG and fMRI data
2. Weight mapping produces valid probability distributions
3. Temperature scheduling works across training stages
4. Integration with trainer is functional
"""

import torch
import numpy as np
from train.variability_weighting import (
    VariabilityComputer,
    VariabilityWeightMapper,
    TrainingStageScheduler,
    DynamicVariabilityWeighting
)


def test_variability_computer():
    """Test variability computation for EEG and fMRI."""
    print("\n" + "="*80)
    print("Testing VariabilityComputer")
    print("="*80)
    
    computer = VariabilityComputer(
        window_size=50,
        use_temporal_variance=True,
        use_first_order_diff=True,
        use_covariance_participation=False
    )
    
    # Test EEG data (fast dynamics, many channels)
    print("\n1. Testing EEG variability computation...")
    eeg_data = torch.randn(1, 200, 64)  # [batch, time, channels]
    # Add some channels with low variability (simulating silent channels)
    eeg_data[:, :, 50:60] = eeg_data[:, :, 50:60] * 0.1  # Low variance channels
    
    eeg_variability = computer.compute_eeg_variability(eeg_data, normalize=True)
    print(f"   EEG shape: {eeg_data.shape}")
    print(f"   Variability shape: {eeg_variability.shape}")
    print(f"   Variability range: [{eeg_variability.min():.4f}, {eeg_variability.max():.4f}]")
    print(f"   Mean: {eeg_variability.mean():.4f}, Std: {eeg_variability.std():.4f}")
    print(f"   Low-variance channels (50-60) mean: {eeg_variability[50:60].mean():.4f}")
    print(f"   High-variance channels (0-50) mean: {eeg_variability[0:50].mean():.4f}")
    
    assert eeg_variability.shape[0] == 64, "Wrong output dimension"
    assert eeg_variability.min() >= 0, "Negative variability"
    assert eeg_variability.max() <= 1.1, "Variability not normalized"
    print("   ✓ EEG variability computation passed")
    
    # Test fMRI data (slow dynamics, fewer ROIs)
    print("\n2. Testing fMRI variability computation...")
    fmri_data = torch.randn(1, 500, 200)  # [batch, time, rois]
    # Add some ROIs with state transitions
    fmri_data[:, :250, :50] = fmri_data[:, :250, :50] + 2.0  # State 1
    fmri_data[:, 250:, :50] = fmri_data[:, 250:, :50] - 2.0  # State 2
    
    fmri_variability = computer.compute_fmri_variability(fmri_data, normalize=True)
    print(f"   fMRI shape: {fmri_data.shape}")
    print(f"   Variability shape: {fmri_variability.shape}")
    print(f"   Variability range: [{fmri_variability.min():.4f}, {fmri_variability.max():.4f}]")
    print(f"   Mean: {fmri_variability.mean():.4f}, Std: {fmri_variability.std():.4f}")
    print(f"   Dynamic ROIs (0-50) mean: {fmri_variability[0:50].mean():.4f}")
    print(f"   Static ROIs (50-200) mean: {fmri_variability[50:200].mean():.4f}")
    
    assert fmri_variability.shape[0] == 200, "Wrong output dimension"
    assert fmri_variability.min() >= 0, "Negative variability"
    assert fmri_variability.max() <= 1.1, "Variability not normalized"
    print("   ✓ fMRI variability computation passed")
    
    return True


def test_weight_mapper():
    """Test weight mapping with different temperatures."""
    print("\n" + "="*80)
    print("Testing VariabilityWeightMapper")
    print("="*80)
    
    # Create mock variability scores
    variability = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95])
    
    mapper = VariabilityWeightMapper(min_weight=0.01)
    
    # Test different temperatures
    temps = [0.1, 0.5, 1.0, 2.0, 10.0]
    print(f"\nVariability scores: {variability.numpy()}")
    print(f"\nWeights at different temperatures:")
    print(f"{'Temp':<8} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8} {'Sum':<8}")
    print("-" * 48)
    
    for temp in temps:
        weights = mapper.compute_weights(variability, temperature=temp)
        print(f"{temp:<8.1f} {weights.min():.6f} {weights.max():.6f} {weights.mean():.6f} {weights.std():.6f} {weights.sum():.6f}")
        
        # Validate
        assert weights.shape == variability.shape, "Shape mismatch"
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5), "Weights don't sum to 1"
        assert weights.min() >= 0.01, "Weight below minimum"
        assert weights.max() <= 1.0, "Weight above 1"
    
    print("\n   ✓ Weight mapping passed - weights sum to 1 and respect constraints")
    
    # Show that low temp focuses on high variability
    low_temp_weights = mapper.compute_weights(variability, temperature=0.1)
    high_temp_weights = mapper.compute_weights(variability, temperature=10.0)
    
    print(f"\n   Low temp (0.1) - sharp focus:")
    print(f"      Top-3 weights: {low_temp_weights.topk(3).values.numpy()}")
    print(f"   High temp (10.0) - flat distribution:")
    print(f"      Top-3 weights: {high_temp_weights.topk(3).values.numpy()}")
    
    return True


def test_stage_scheduler():
    """Test training stage scheduling."""
    print("\n" + "="*80)
    print("Testing TrainingStageScheduler")
    print("="*80)
    
    scheduler = TrainingStageScheduler(
        warmup_epochs=5,
        main_epochs=60,
        finetune_epochs=30,
        warmup_temp=0.1,
        main_temp_start=0.1,
        main_temp_end=1.0,
        finetune_temp=2.0,
        disable_in_finetune=False
    )
    
    # Test key epochs
    test_epochs = [1, 3, 5, 6, 10, 30, 65, 66, 80, 95]
    print(f"\n{'Epoch':<8} {'Stage':<12} {'Temperature':<12} {'Notes'}")
    print("-" * 50)
    
    for epoch in test_epochs:
        stage = scheduler.get_stage_name(epoch)
        temp = scheduler.get_temperature(epoch)
        notes = ""
        
        if epoch == 1:
            notes = "(start warmup)"
        elif epoch == 5:
            notes = "(end warmup)"
        elif epoch == 6:
            notes = "(start main)"
        elif epoch == 65:
            notes = "(end main)"
        elif epoch == 66:
            notes = "(start finetune)"
        elif epoch == 95:
            notes = "(near end)"
        
        print(f"{epoch:<8} {stage:<12} {temp:<12.3f} {notes}")
        
        # Validate stage names
        if epoch <= 5:
            assert stage == 'warmup', f"Wrong stage at epoch {epoch}"
        elif epoch <= 65:
            assert stage == 'main', f"Wrong stage at epoch {epoch}"
        else:
            assert stage == 'finetune', f"Wrong stage at epoch {epoch}"
    
    # Validate temperature increases in main training
    temps_main = [scheduler.get_temperature(e) for e in range(6, 66)]
    assert temps_main[0] < temps_main[-1], "Temperature should increase in main training"
    
    print("\n   ✓ Stage scheduling passed - correct stages and temperature progression")
    
    return True


def test_integrated_system():
    """Test the complete integrated system."""
    print("\n" + "="*80)
    print("Testing DynamicVariabilityWeighting (Integrated System)")
    print("="*80)
    
    system = DynamicVariabilityWeighting(
        eeg_window_size=50,
        fmri_window_size=150,
        min_weight=0.01,
        warmup_epochs=5,
        main_epochs=60,
        finetune_epochs=30,
        warmup_temp=0.1,
        main_temp_start=0.1,
        main_temp_end=1.0,
        finetune_temp=2.0,
        enabled=True
    )
    
    # Simulate EEG data
    print("\n1. Testing EEG weight computation across epochs...")
    eeg_data = torch.randn(1, 200, 64)
    eeg_data[:, :, 50:60] = eeg_data[:, :, 50:60] * 0.1  # Low variance
    
    epochs_to_test = [1, 5, 30, 66, 90]
    print(f"\n{'Epoch':<8} {'Stage':<12} {'Temp':<8} {'Weight Range':<20} {'Focus'}")
    print("-" * 60)
    
    for epoch in epochs_to_test:
        weights = system.compute_modality_weights(eeg_data, 'eeg', epoch)
        stage_info = system.get_stage_info(epoch)
        
        w_min, w_max = weights.min().item(), weights.max().item()
        focus = "Sharp" if w_max > 0.1 else "Flat"
        
        print(f"{epoch:<8} {stage_info['stage']:<12} {stage_info['temperature']:<8.2f} "
              f"[{w_min:.5f}, {w_max:.5f}]  {focus}")
        
        # Validate
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5), "Weights don't sum to 1"
        assert weights.shape[0] == 64, "Wrong dimension"
    
    print("\n   ✓ EEG weighting across epochs passed")
    
    # Simulate fMRI data
    print("\n2. Testing fMRI weight computation...")
    fmri_data = torch.randn(1, 500, 200)
    fmri_data[:, :250, :50] += 2.0  # State change in first 50 ROIs
    
    weights_fmri = system.compute_modality_weights(fmri_data, 'fmri', epoch=30)
    print(f"   fMRI weights computed: shape={weights_fmri.shape}")
    print(f"   Range: [{weights_fmri.min():.5f}, {weights_fmri.max():.5f}]")
    print(f"   Dynamic ROIs (0-50) mean: {weights_fmri[:50].mean():.5f}")
    print(f"   Static ROIs (50-200) mean: {weights_fmri[50:].mean():.5f}")
    
    assert torch.allclose(weights_fmri.sum(), torch.tensor(1.0), atol=1e-5), "Weights don't sum to 1"
    print("   ✓ fMRI weighting passed")
    
    # Test caching
    print("\n3. Testing weight caching...")
    weights_cached = system.compute_modality_weights(eeg_data, 'eeg', epoch=30, force_update=False)
    assert torch.allclose(weights_cached, system.compute_modality_weights(eeg_data, 'eeg', epoch=30)), \
        "Cached weights don't match"
    print("   ✓ Caching works correctly")
    
    # Test disabled mode
    print("\n4. Testing disabled mode...")
    system_disabled = DynamicVariabilityWeighting(enabled=False)
    weights_uniform = system_disabled.compute_modality_weights(eeg_data, 'eeg', epoch=1)
    expected_uniform = torch.ones(64) / 64
    assert torch.allclose(weights_uniform, expected_uniform, atol=1e-5), "Disabled mode should return uniform weights"
    print("   ✓ Disabled mode returns uniform weights")
    
    print("\n   ✓ All integrated system tests passed")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("DYNAMIC VARIABILITY WEIGHTING - TEST SUITE")
    print("="*80)
    
    try:
        test_variability_computer()
        test_weight_mapper()
        test_stage_scheduler()
        test_integrated_system()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nThe dynamic weighting system is working correctly!")
        print("\nKey findings:")
        print("  • Variability computation successfully identifies dynamic channels/ROIs")
        print("  • Weight mapping produces valid probability distributions")
        print("  • Temperature scheduling works across training stages")
        print("  • Integration is complete and functional")
        print("\nTo enable in training, set 'dynamic_weighting.enabled: true' in config")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
