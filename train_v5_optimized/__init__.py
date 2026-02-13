"""
TwinBrain V5 Optimized Training Components
==========================================

This package contains enhanced training components for improved:
1. EEG-fMRI energy balance
2. EEG silent channel handling
3. Multi-step prediction capabilities
4. fMRI prediction accuracy

Modules:
- adaptive_loss_balancer: GradNorm-based adaptive loss weighting
- eeg_channel_handler: Enhanced EEG channel processing
- advanced_prediction: Hierarchical multi-step prediction

Usage:
    from train_v5_optimized import (
        AdaptiveLossBalancer,
        EnhancedEEGHandler,
        EnhancedMultiStepPredictor,
    )
"""

__version__ = '5.0.0'
__author__ = 'TwinBrain Team'

# Import main components
try:
    from .adaptive_loss_balancer import (
        AdaptiveLossBalancer,
        ModalityGradientScaler,
    )
except ImportError as e:
    print(f"Warning: Could not import adaptive_loss_balancer: {e}")
    AdaptiveLossBalancer = None
    ModalityGradientScaler = None

try:
    from .eeg_channel_handler import (
        EnhancedEEGHandler,
        ChannelActivityMonitor,
        AdaptiveChannelDropout,
        ChannelAttention,
        AntiCollapseRegularizer,
    )
except ImportError as e:
    print(f"Warning: Could not import eeg_channel_handler: {e}")
    EnhancedEEGHandler = None
    ChannelActivityMonitor = None
    AdaptiveChannelDropout = None
    ChannelAttention = None
    AntiCollapseRegularizer = None

try:
    from .advanced_prediction import (
        EnhancedMultiStepPredictor,
        HierarchicalPredictor,
        TransformerPredictor,
        StratifiedWindowSampler,
        UncertaintyAwarePredictor,
    )
except ImportError as e:
    print(f"Warning: Could not import advanced_prediction: {e}")
    EnhancedMultiStepPredictor = None
    HierarchicalPredictor = None
    TransformerPredictor = None
    StratifiedWindowSampler = None
    UncertaintyAwarePredictor = None

__all__ = [
    # Adaptive loss balancing
    'AdaptiveLossBalancer',
    'ModalityGradientScaler',
    
    # EEG channel handling
    'EnhancedEEGHandler',
    'ChannelActivityMonitor',
    'AdaptiveChannelDropout',
    'ChannelAttention',
    'AntiCollapseRegularizer',
    
    # Advanced prediction
    'EnhancedMultiStepPredictor',
    'HierarchicalPredictor',
    'TransformerPredictor',
    'StratifiedWindowSampler',
    'UncertaintyAwarePredictor',
]


def get_default_config():
    """
    Get default V5 optimization configuration.
    
    Returns:
        dict: Default configuration for V5 optimizations
    """
    return {
        'adaptive_loss': {
            'enabled': True,
            'alpha': 1.5,
            'learning_rate': 0.025,
            'update_frequency': 10,
            'warmup_epochs': 5,
            'modality_energy_ratios': {
                'eeg': 0.02,
                'fmri': 1.0,
            },
        },
        'eeg_enhancement': {
            'enabled': True,
            'enable_monitoring': True,
            'enable_dropout': True,
            'enable_attention': True,
            'enable_regularization': True,
            'dropout_rate': 0.1,
            'attention_hidden_dim': 64,
            'entropy_weight': 0.01,
            'diversity_weight': 0.01,
            'activity_weight': 0.01,
        },
        'advanced_prediction': {
            'enabled': True,
            'use_hierarchical': True,
            'use_transformer': True,
            'use_uncertainty': True,
            'num_scales': 3,
            'num_windows': 3,
            'sampling_strategy': 'uniform',
            'hidden_dim': 256,
        },
    }


def check_dependencies():
    """
    Check if all required dependencies are available.
    
    Returns:
        dict: Status of each component
    """
    status = {
        'adaptive_loss_balancer': AdaptiveLossBalancer is not None,
        'eeg_channel_handler': EnhancedEEGHandler is not None,
        'advanced_prediction': EnhancedMultiStepPredictor is not None,
    }
    
    all_available = all(status.values())
    
    return {
        'all_available': all_available,
        'components': status,
    }


# Print version info on import
print(f"TwinBrain V5 Optimized Training Components v{__version__}")
dep_status = check_dependencies()
if dep_status['all_available']:
    print("✓ All components loaded successfully")
else:
    print("⚠ Some components failed to load:")
    for comp, loaded in dep_status['components'].items():
        status = "✓" if loaded else "✗"
        print(f"  {status} {comp}")
