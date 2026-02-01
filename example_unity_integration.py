#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unity Integration Example
==========================

This script demonstrates how to use the Unity integration features:
1. Export brain states to JSON
2. Simulate virtual stimulation
3. Generate time series for Unity animation

Run this after training a model to export brain states for Unity visualization.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from unity_integration import BrainStateExporter, StimulationSimulator, StimulationConfig


def load_example_data():
    """Load or generate example brain activity data."""
    print("Loading example data...")
    
    # For demonstration, generate synthetic data
    # In practice, you would load actual model output
    n_regions = 200
    n_timepoints = 400
    n_features = 1
    
    # Generate synthetic fMRI data
    fmri_data = torch.randn(n_regions, n_timepoints, n_features)
    
    # Generate synthetic EEG data
    eeg_data = torch.randn(n_regions, n_timepoints, n_features)
    
    # Generate connectivity matrix
    connectivity = np.random.rand(n_regions, n_regions)
    connectivity = (connectivity + connectivity.T) / 2  # Make symmetric
    connectivity[connectivity < 0.7] = 0  # Sparse connectivity
    
    return {
        'fmri': fmri_data,
        'eeg': eeg_data,
        'connectivity': {'structural': connectivity}
    }


def load_atlas_info():
    """Load atlas information."""
    print("Loading atlas info...")
    
    # Example atlas info (Schaefer 200 regions)
    # In practice, load from actual atlas file
    atlas_info = {
        'name': 'Schaefer200',
        'regions': {}
    }
    
    # Generate example region info
    for i in range(200):
        atlas_info['regions'][str(i + 1)] = {
            'label': f'Region_{i+1}',
            'xyz': [
                np.random.uniform(-80, 80),
                np.random.uniform(-100, 80),
                np.random.uniform(-60, 80)
            ]
        }
    
    return atlas_info


def example_1_export_single_state():
    """Example 1: Export a single brain state."""
    print("\n" + "="*60)
    print("Example 1: Export Single Brain State")
    print("="*60)
    
    # Load data and atlas
    data = load_example_data()
    atlas_info = load_atlas_info()
    
    # Create exporter
    exporter = BrainStateExporter(atlas_info, model_version="v4")
    
    # Export brain state at time point 100
    output_path = Path("output/brain_state_t100.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    brain_state = exporter.export_brain_state(
        brain_activity={'fmri': data['fmri'], 'eeg': data['eeg']},
        connectivity=data['connectivity'],
        time_point=100,
        time_second=200.0,
        subject_id="example_subject",
        output_path=output_path
    )
    
    print(f"✓ Exported brain state to: {output_path}")
    print(f"  - Number of regions: {len(brain_state['brain_state']['regions'])}")
    print(f"  - Number of connections: {len(brain_state['brain_state']['connections'])}")
    print(f"  - Global mean activity: {brain_state['brain_state']['global_metrics']['mean_activity']:.3f}")


def example_2_export_sequence():
    """Example 2: Export a time sequence for animation."""
    print("\n" + "="*60)
    print("Example 2: Export Time Sequence")
    print("="*60)
    
    # Load data and atlas
    data = load_example_data()
    atlas_info = load_atlas_info()
    
    # Create exporter
    exporter = BrainStateExporter(atlas_info, model_version="v4")
    
    # Export sequence
    output_dir = Path("output/brain_sequence")
    
    exporter.export_sequence(
        brain_activity={'fmri': data['fmri'], 'eeg': data['eeg']},
        output_dir=output_dir,
        start=0,
        end=200,
        step=5,
        connectivity=data['connectivity'],
        subject_id="example_subject"
    )
    
    print(f"✓ Exported sequence to: {output_dir}")
    print(f"  - Time range: 0 to 200")
    print(f"  - Step: 5")
    print(f"  - Number of frames: {len(list(output_dir.glob('brain_state_*.json')))}")
    print(f"  - See sequence_index.json for frame list")


def example_3_simulate_stimulation():
    """Example 3: Simulate virtual stimulation."""
    print("\n" + "="*60)
    print("Example 3: Simulate Virtual Stimulation")
    print("="*60)
    
    # Load data
    data = load_example_data()
    n_regions = data['fmri'].shape[0]
    
    # Create simulator
    simulator = StimulationSimulator(
        n_regions=n_regions,
        connectivity=data['connectivity']['structural']
    )
    
    # Configure stimulation
    stim_config = StimulationConfig(
        target_regions=[10, 15, 20],  # Stimulate visual cortex
        amplitude=0.5,
        duration=20,
        pattern="sine",
        frequency=10.0,  # 10 Hz alpha band
        spatial_spread=5.0
    )
    
    print(f"Stimulation configuration:")
    print(f"  - Target regions: {stim_config.target_regions}")
    print(f"  - Pattern: {stim_config.pattern}")
    print(f"  - Amplitude: {stim_config.amplitude}")
    print(f"  - Frequency: {stim_config.frequency} Hz")
    print(f"  - Duration: {stim_config.duration} time steps")
    
    # Simulate response
    initial_state = data['fmri'][:, 0:1, :]  # Initial state
    
    trajectory, metrics = simulator.simulate_response(
        initial_state=initial_state,
        config=stim_config,
        n_steps=50
    )
    
    print(f"\n✓ Simulation complete:")
    print(f"  - Trajectory shape: {trajectory.shape}")
    print(f"  - Number of time steps: {len(metrics)}")
    
    # Export trajectory as sequence
    atlas_info = load_atlas_info()
    exporter = BrainStateExporter(atlas_info)
    
    output_dir = Path("output/stimulation_response")
    
    for t, state in enumerate(trajectory):
        # Add time dimension if needed
        if len(state.shape) == 2:
            state = state.unsqueeze(1)
        
        exporter.export_brain_state(
            brain_activity={'fmri': state},
            time_point=t,
            time_second=float(t),
            subject_id="stimulation_example",
            stimulation=simulator.to_json(stim_config) if metrics[t]['stimulation_active'] else None,
            output_path=output_dir / f"response_{t:03d}.json"
        )
    
    print(f"  - Exported to: {output_dir}")
    print(f"  - Target activity increased by: {(metrics[-1]['target_activity'] - metrics[0]['target_activity']):.3f}")


def example_4_inverse_stimulation():
    """Example 4: Design stimulation to reach target state."""
    print("\n" + "="*60)
    print("Example 4: Inverse Stimulation Design")
    print("="*60)
    
    # Load data
    data = load_example_data()
    n_regions = data['fmri'].shape[0]
    
    # Create simulator
    simulator = StimulationSimulator(
        n_regions=n_regions,
        connectivity=data['connectivity']['structural']
    )
    
    # Define initial and target states
    initial_state = data['fmri'][:, 0:1, :]
    target_state = data['fmri'][:, 100:101, :]  # Use a later state as target
    
    print("Designing stimulation to reach target state...")
    print(f"  - Initial state mean activity: {initial_state.mean().item():.3f}")
    print(f"  - Target state mean activity: {target_state.mean().item():.3f}")
    
    # Design stimulation
    optimal_config = simulator.design_inverse_stimulation(
        initial_state=initial_state,
        target_state=target_state,
        max_amplitude=1.0,
        n_iterations=50,
        learning_rate=0.01
    )
    
    print(f"\n✓ Optimal stimulation design:")
    print(f"  - Target regions: {optimal_config.target_regions[:5]}... (showing first 5)")
    print(f"  - Amplitude: {optimal_config.amplitude:.3f}")
    print(f"  - Duration: {optimal_config.duration}")
    print(f"  - Total regions to stimulate: {len(optimal_config.target_regions)}")
    
    # Save config
    output_path = Path("output/optimal_stimulation_config.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(simulator.to_json(optimal_config), f, indent=2)
    
    print(f"  - Saved config to: {output_path}")


def main():
    """Run all examples."""
    print("="*60)
    print("TwinBrain Unity Integration Examples")
    print("="*60)
    
    try:
        # Run examples
        example_1_export_single_state()
        example_2_export_sequence()
        example_3_simulate_stimulation()
        example_4_inverse_stimulation()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        print("\nOutput files are in the 'output/' directory.")
        print("You can now use these JSON files in Unity for visualization.")
        print("\nNext steps:")
        print("1. Check the JSON files in output/")
        print("2. Load them in Unity using the provided C# scripts")
        print("3. Visualize brain activity in 3D")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
