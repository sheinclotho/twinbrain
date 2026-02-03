"""
Example script for using TwinBrain V2 with cross-modal bidirectional prediction.

This example demonstrates:
1. How to use the V2 trainer with cross-modal prediction
2. How to configure cross-modal prediction parameters
3. How to interpret cross-modal prediction results
"""

import torch
import yaml
from pathlib import Path

# Import V2 trainer
from train.hetero_trainer_v2 import DynamicHeteroTrainerV2, create_trainer_v2

# Import original components
from mapper.multi_modal_mapper import MultiModalMapper
from utils.config import load_config


def example_basic_usage():
    """
    Basic example of using V2 trainer with cross-modal prediction.
    """
    print("=" * 80)
    print("TwinBrain V2 - Cross-Modal Bidirectional Prediction Example")
    print("=" * 80)
    
    # Load configuration
    config_path = "config/default_v2.yaml"
    print(f"\n1. Loading V2 configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"   Cross-modal prediction enabled: {config['cross_modal_prediction']['enabled']}")
    print(f"   Direction: {config['cross_modal_prediction']['direction']}")
    print(f"   Weight: {config['cross_modal_prediction']['weight']}")
    
    # Load data (example - replace with actual data loading)
    print("\n2. Loading multi-modal data...")
    print("   Note: This requires actual fMRI and EEG data")
    print("   Skipping actual data loading in this example")
    
    # Create V2 trainer
    print("\n3. Creating V2 trainer with cross-modal prediction...")
    
    # Example parameters (use actual data in practice)
    example_params = {
        'hidden_dim': config['model']['hidden_dim'],
        'num_layers': config['model']['num_layers'],
        'dropout': config['model']['dropout'],
        'lr': config['training']['learning_rate'],
        'num_epochs': config['training']['finetune_epochs'],
        
        # Single-modality prediction
        'enable_prediction': config['prediction']['enabled'],
        'prediction_context_length': config['prediction']['context_length'],
        'prediction_steps': config['prediction']['steps'],
        'prediction_weight': config['prediction']['weight'],
        
        # Cross-modal prediction (V2)
        'enable_cross_modal_prediction': config['cross_modal_prediction']['enabled'],
        'cross_modal_weight': config['cross_modal_prediction']['weight'],
        'cross_modal_context_length': config['cross_modal_prediction']['context_length'],
        'cross_modal_steps': config['cross_modal_prediction']['steps'],
        'cross_modal_direction': config['cross_modal_prediction']['direction'],
        'cross_modal_use_bridge': config['cross_modal_prediction']['use_bridge'],
        'cross_modal_share_attention': config['cross_modal_prediction']['share_attention'],
    }
    
    print(f"   Parameters configured:")
    print(f"   - Single-modality prediction: {example_params['enable_prediction']}")
    print(f"   - Cross-modal prediction: {example_params['enable_cross_modal_prediction']}")
    print(f"   - Cross-modal direction: {example_params['cross_modal_direction']}")
    
    # Trainer initialization would go here with actual data
    # trainer = DynamicHeteroTrainerV2(hetero_data=your_data, **example_params)
    
    print("\n4. Training would begin here with:")
    print("   trainer.train(save_dir='results_v2')")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


def example_comparison_v1_vs_v2():
    """
    Example showing the difference between V1 (single-modality) and V2 (cross-modal).
    """
    print("\n" + "=" * 80)
    print("Comparison: V1 (Single-Modality) vs V2 (Cross-Modal)")
    print("=" * 80)
    
    print("\nV1 (Original) - Single-Modality Prediction:")
    print("  ┌─────────┐       ┌─────────┐")
    print("  │  fMRI   │  -->  │  fMRI   │  (predicts fMRI from fMRI)")
    print("  │ history │       │ future  │")
    print("  └─────────┘       └─────────┘")
    print()
    print("  ┌─────────┐       ┌─────────┐")
    print("  │  EEG    │  -->  │  EEG    │  (predicts EEG from EEG)")
    print("  │ history │       │ future  │")
    print("  └─────────┘       └─────────┘")
    
    print("\nV2 (New) - Cross-Modal Bidirectional Prediction:")
    print("  ┌─────────┐       ┌─────────┐")
    print("  │  fMRI   │  -->  │  fMRI   │  (single-modality)")
    print("  │ history │       │ future  │")
    print("  └─────────┘       └─────────┘")
    print("       │                  ↑")
    print("       │                  │")
    print("       ↓                  │")
    print("  ┌─────────┐       ┌─────────┐")
    print("  │  EEG    │  -->  │  EEG    │  (single-modality)")
    print("  │ history │       │ future  │")
    print("  └─────────┘       └─────────┘")
    print()
    print("  Plus cross-modal predictions:")
    print("  - fMRI history  -->  EEG future   (fMRI → EEG)")
    print("  - EEG history   -->  fMRI future  (EEG → fMRI)")
    
    print("\nKey Differences:")
    print("  1. V2 learns cross-modal dependencies during training")
    print("  2. V2 can predict EEG activity from fMRI features (and vice versa)")
    print("  3. V2 enables deeper understanding of multi-modal brain dynamics")
    print("  4. V2 uses cross-attention mechanisms between modalities")


def example_configuration_options():
    """
    Example showing different configuration options for V2.
    """
    print("\n" + "=" * 80)
    print("V2 Configuration Options")
    print("=" * 80)
    
    print("\n1. Bidirectional (Recommended):")
    print("   cross_modal_prediction:")
    print("     direction: 'both'")
    print("   → Learns both fMRI→EEG and EEG→fMRI")
    
    print("\n2. Unidirectional (fMRI → EEG):")
    print("   cross_modal_prediction:")
    print("     direction: 'fmri_to_eeg'")
    print("   → Only learns to predict EEG from fMRI")
    
    print("\n3. Unidirectional (EEG → fMRI):")
    print("   cross_modal_prediction:")
    print("     direction: 'eeg_to_fmri'")
    print("   → Only learns to predict fMRI from EEG")
    
    print("\n4. Weight Adjustment:")
    print("   cross_modal_prediction:")
    print("     weight: 0.05  # Lower weight (less emphasis)")
    print("     weight: 0.1   # Default (balanced)")
    print("     weight: 0.2   # Higher weight (more emphasis)")
    
    print("\n5. Architecture Options:")
    print("   cross_modal_prediction:")
    print("     use_bridge: true        # Use modality translation networks")
    print("     share_attention: false  # Separate attention for each direction")
    print("     num_layers: 3           # GRU layers")
    print("     num_heads: 8            # Attention heads")


def example_expected_benefits():
    """
    Example explaining the expected benefits of V2.
    """
    print("\n" + "=" * 80)
    print("Expected Benefits of V2 Cross-Modal Prediction")
    print("=" * 80)
    
    print("\n1. Enhanced Multi-Modal Understanding:")
    print("   - Model learns relationships between fMRI and EEG")
    print("   - Captures cross-modal dynamics not visible in single modality")
    
    print("\n2. Improved Prediction Accuracy:")
    print("   - Cross-modal information helps constrain predictions")
    print("   - Complementary information from both modalities")
    
    print("\n3. Better Latent Representations:")
    print("   - Forces model to learn modality-invariant features")
    print("   - More robust internal representations")
    
    print("\n4. Novel Applications:")
    print("   - Predict EEG when only fMRI is available")
    print("   - Predict fMRI when only EEG is available")
    print("   - Fill in missing modality data")
    
    print("\n5. Research Insights:")
    print("   - Understand how fMRI and EEG relate dynamically")
    print("   - Discover cross-modal biomarkers")
    print("   - Study multi-modal brain connectivity")


def main():
    """
    Main function to run all examples.
    """
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 20 + "TwinBrain V2 Examples" + " " * 37 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    
    # Run examples
    example_basic_usage()
    example_comparison_v1_vs_v2()
    example_configuration_options()
    example_expected_benefits()
    
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 25 + "Examples Complete" + " " * 36 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    
    print("\nNext Steps:")
    print("  1. Prepare your multi-modal data (fMRI + EEG)")
    print("  2. Configure V2 parameters in config/default_v2.yaml")
    print("  3. Run training: python main.py train --config config/default_v2.yaml")
    print("  4. Monitor cross-modal prediction losses during training")
    print("  5. Evaluate cross-modal prediction accuracy")
    
    print("\nNote: V2 files are separate from original implementation")
    print("  - train/predictor_v2.py (new predictor modules)")
    print("  - train/hetero_trainer_v2.py (extended trainer)")
    print("  - config/default_v2.yaml (V2 configuration)")
    print("\nOriginal files remain unchanged for stability.")


if __name__ == "__main__":
    main()
