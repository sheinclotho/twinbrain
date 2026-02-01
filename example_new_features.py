"""
Example script demonstrating the new prediction and monitoring features.
This script shows how to use the enhanced capabilities without running full training.
"""

import torch
import torch.nn as nn
from pathlib import Path

# Import new modules
from train.predictor import PredictorHead, ConditionalPredictor, prediction_loss
from utils.metrics_tracker import MetricsTracker, TrainingMonitor


def demo_predictor():
    """Demonstrate the PredictorHead module."""
    print("=" * 80)
    print("Demo 1: Multi-step Future Prediction with Context Length")
    print("=" * 80)
    
    # Configuration
    batch_size = 2
    seq_length = 100  # Total available history
    context_length = 50  # Use only last 50 steps
    hidden_dim = 128
    n_future_steps = 10
    
    print(f"\nScenario: Use last {context_length} steps to predict next {n_future_steps} steps")
    print(f"Available history: {seq_length} steps")
    
    # Create predictor with context length
    predictor = PredictorHead(
        hidden_dim=hidden_dim,
        n_future_steps=n_future_steps,
        context_length=context_length,  # NEW: Specify context length
        num_layers=3,
        num_heads=8,
        dropout=0.1
    )
    
    # Create sample historical sequence
    latent_seq = torch.randn(batch_size, seq_length, hidden_dim)
    
    print(f"\nInput shape: {latent_seq.shape}")
    print(f"Context length: {context_length}")
    print(f"Prediction horizon: {n_future_steps}")
    
    # Predict future
    predictions, attention = predictor(latent_seq, return_attention=True)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Attention weights shape: {attention.shape if attention is not None else 'None'}")
    print(f"→ Attention is over {context_length} context steps")
    
    # Compute loss against ground truth
    target_future = torch.randn(batch_size, n_future_steps, hidden_dim)
    loss = prediction_loss(predictions, target_future, loss_type='mse')
    
    print(f"Prediction loss: {loss.item():.4f}")
    print()


def demo_conditional_predictor():
    """Demonstrate the ConditionalPredictor for stimulation."""
    print("=" * 80)
    print("Demo 2: Conditional Prediction (Stimulation Response)")
    print("=" * 80)
    
    batch_size = 2
    hidden_dim = 128
    n_regions = 200
    n_future_steps = 10
    
    # Create conditional predictor
    predictor = ConditionalPredictor(
        hidden_dim=hidden_dim,
        condition_dim=n_regions,
        n_future_steps=n_future_steps,
        num_layers=3
    )
    
    # Current brain state
    current_state = torch.randn(batch_size, hidden_dim)
    
    # Stimulation pattern (e.g., TMS to specific regions)
    stimulation = torch.zeros(batch_size, n_future_steps, n_regions)
    # Stimulate regions 10 and 15 with amplitude 0.5
    stimulation[:, :, [10, 15]] = 0.5
    
    print(f"Current state shape: {current_state.shape}")
    print(f"Stimulation shape: {stimulation.shape}")
    
    # Predict response
    predicted_states = predictor(current_state, stimulation)
    
    print(f"Predicted states shape: {predicted_states.shape}")
    print(f"Stimulation applied to regions: [10, 15]")
    print()


def demo_metrics_tracker():
    """Demonstrate the MetricsTracker."""
    print("=" * 80)
    print("Demo 3: Enhanced Metrics Tracking")
    print("=" * 80)
    
    # Create metrics tracker
    output_dir = Path("/tmp/twinbrain_demo_metrics")
    tracker = MetricsTracker(output_dir=output_dir, enabled=True)
    
    print(f"Metrics output directory: {output_dir}")
    print()
    
    # Simulate training epochs
    print("Simulating 20 epochs of training...")
    for epoch in range(1, 21):
        # Simulate loss values that improve over time
        total_loss = 2.0 - (epoch * 0.05) + torch.randn(1).item() * 0.1
        recon_loss = 0.8 - (epoch * 0.02) + torch.randn(1).item() * 0.05
        temp_loss = 0.7 - (epoch * 0.015) + torch.randn(1).item() * 0.05
        align_loss = 0.5 - (epoch * 0.015) + torch.randn(1).item() * 0.05
        
        # Log loss components
        tracker.log_loss_components(
            epoch=epoch,
            recon_loss=recon_loss,
            temp_loss=temp_loss,
            align_loss=align_loss,
            total_loss=total_loss
        )
        
        # Log gradient statistics
        grad_norm = 3.0 - (epoch * 0.1) + torch.randn(1).item() * 0.5
        tracker.log_gradient_stats(
            epoch=epoch,
            grad_norm=max(0.1, grad_norm)
        )
        
        # Log relative errors
        fmri_error = 0.15 - (epoch * 0.003) + torch.randn(1).item() * 0.01
        eeg_error = 0.18 - (epoch * 0.004) + torch.randn(1).item() * 0.01
        tracker.log_epoch(epoch, {
            'rel_error/fmri': max(0.01, fmri_error),
            'rel_error/eeg': max(0.01, eeg_error)
        })
    
    print("Training simulation completed!")
    print()
    
    # Save metrics
    tracker.save_metrics()
    print(f"Metrics saved to: {output_dir / 'metrics_history.json'}")
    print()
    
    # Print summary
    tracker.print_summary(last_n_epochs=10)
    print()
    
    # Get best epoch
    best_epoch = tracker.get_best_epoch('loss/total', mode='min')
    print(f"Best epoch (lowest total loss): {best_epoch}")
    print()


def demo_training_monitor():
    """Demonstrate the TrainingMonitor."""
    print("=" * 80)
    print("Demo 4: Training Progress Monitoring")
    print("=" * 80)
    
    # Create monitor
    monitor = TrainingMonitor(patience=5, min_delta=0.01, check_interval=1)
    
    print("Simulating training with stagnation detection...")
    print()
    
    # Simulate training with improvement then stagnation
    losses = [
        2.0, 1.8, 1.6, 1.5, 1.45,  # Improving
        1.44, 1.43, 1.43, 1.44, 1.43,  # Stagnating
        1.42, 1.41  # Slight improvement
    ]
    
    for epoch, loss in enumerate(losses, 1):
        status = monitor.check_progress(epoch, loss)
        
        print(f"Epoch {epoch:2d}: loss={loss:.2f}", end="")
        
        if status['improved']:
            print(" ✓ Improved!")
        elif status['warnings']:
            print(f" ⚠ {status['warnings'][0]}")
        else:
            print()
        
        if status['should_stop']:
            print("Training should stop due to issues!")
            break
    
    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "TwinBrain New Features Demo" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        demo_predictor()
        demo_conditional_predictor()
        demo_metrics_tracker()
        demo_training_monitor()
        
        print("=" * 80)
        print("All demos completed successfully! ✓")
        print("=" * 80)
        print()
        print("Next steps:")
        print("1. Enable prediction in config: prediction.enabled = true")
        print("2. Enable metrics in config: metrics.enabled = true")
        print("3. Run training: python main.py train --config config/default.yaml")
        print()
        
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
