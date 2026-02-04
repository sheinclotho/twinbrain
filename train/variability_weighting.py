"""
Dynamic Variability-Based Weighting for EEG-fMRI Training
===========================================================

Implements a learnability/variability-based dynamic weighting mechanism for
unsupervised multimodal training. Prevents zero-solution collapse in EEG 
resting-state data and emphasizes network state transitions in fMRI.

Key Features:
- Channel/ROI-level variability computation
- Temperature-based softmax weight mapping
- Three-stage training scheduler (warmup, main, finetune)
- Modality-specific time scales and statistics
- No supervision, no hard masking, fully differentiable

References:
    Problem statement: OPTIMIZATION_DIRECTIONS.md section on dynamic weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import numpy as np


class VariabilityComputer:
    """
    Computes variability/learnability metrics for channels/ROIs.
    
    This class computes intrinsic variability measures from data without
    using any external supervision or task labels.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        use_temporal_variance: bool = True,
        use_first_order_diff: bool = True,
        use_spectral_change: bool = False,
        use_covariance_participation: bool = False,
        epsilon: float = 1e-8,
    ):
        """
        Initialize variability computer.
        
        Args:
            window_size: Sliding window size for computing statistics
            use_temporal_variance: Use time-series variance
            use_first_order_diff: Use first-order difference energy
            use_spectral_change: Use spectral structure changes (KL divergence)
            use_covariance_participation: Use channel covariance participation
            epsilon: Small constant for numerical stability
        """
        self.window_size = window_size
        self.use_temporal_variance = use_temporal_variance
        self.use_first_order_diff = use_first_order_diff
        self.use_spectral_change = use_spectral_change
        self.use_covariance_participation = use_covariance_participation
        self.epsilon = epsilon
    
    def compute_eeg_variability(
        self,
        x: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute EEG channel variability (fast time scale, channel-level).
        
        Combines multiple variability measures:
        - Temporal variance: Var(x_i)
        - First-order difference energy: Var(x_i(t) - x_i(t-1))
        - Channel covariance participation: Σ_j |corr(i, j)|
        
        Args:
            x: EEG data [batch, time, channels] or [time, channels]
            normalize: Whether to normalize to unit scale
            
        Returns:
            variability: Channel-wise variability scores [channels]
        """
        if x.dim() == 3:
            # [batch, time, channels] -> [time, channels] by averaging batch
            x = x.mean(dim=0)
        
        assert x.dim() == 2, f"Expected 2D tensor [time, channels], got {x.shape}"
        
        T, C = x.shape
        
        if T < self.window_size:
            # If sequence too short, use full sequence
            window_size = T
        else:
            window_size = self.window_size
        
        variabilities = []
        
        # 1. Temporal variance (scale-normalized)
        if self.use_temporal_variance:
            # Use sliding window variance
            var = torch.zeros(C, device=x.device)
            num_windows = T - window_size + 1
            
            for start in range(0, num_windows, max(1, window_size // 4)):
                end = start + window_size
                if end > T:
                    break
                window = x[start:end, :]
                # Variance normalized by local scale
                window_std = window.std(dim=0) + self.epsilon
                window_var = (window.var(dim=0) / (window_std + self.epsilon))
                var += window_var
            
            var = var / max(1, (num_windows + window_size // 4 - 1) // (window_size // 4))
            variabilities.append(var)
        
        # 2. First-order difference energy
        if self.use_first_order_diff:
            diff = x[1:] - x[:-1]  # [T-1, C]
            diff_var = diff.var(dim=0)
            # Normalize by signal scale
            signal_scale = x.std(dim=0) + self.epsilon
            diff_var_norm = diff_var / (signal_scale ** 2 + self.epsilon)
            variabilities.append(diff_var_norm)
        
        # 3. Covariance participation (how much each channel correlates with others)
        if self.use_covariance_participation and C > 1:
            # Compute correlation matrix
            x_centered = x - x.mean(dim=0, keepdim=True)
            x_norm = x_centered / (x_centered.std(dim=0, keepdim=True) + self.epsilon)
            corr_matrix = torch.matmul(x_norm.T, x_norm) / T
            
            # Sum of absolute correlations for each channel (excluding self)
            corr_participation = corr_matrix.abs().sum(dim=1) - 1.0  # -1 to remove diagonal
            corr_participation = corr_participation / (C - 1) if C > 1 else corr_participation
            variabilities.append(corr_participation)
        
        # Combine variabilities
        if len(variabilities) == 0:
            # Fallback: use simple variance
            variability = x.var(dim=0)
        else:
            # Average different measures
            variability = torch.stack(variabilities).mean(dim=0)
        
        # Normalize to [0, 1] range if requested
        if normalize:
            v_min = variability.min()
            v_max = variability.max()
            if v_max > v_min:
                variability = (variability - v_min) / (v_max - v_min + self.epsilon)
        
        return variability
    
    def compute_fmri_variability(
        self,
        x: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute fMRI ROI variability (slow time scale, network-level).
        
        Emphasizes state transitions and network reorganization:
        - Sliding window functional connectivity (FC) changes
        - ROI temporal variance after global signal removal
        - Low-frequency power changes
        
        Args:
            x: fMRI data [batch, time, rois] or [time, rois]
            normalize: Whether to normalize to unit scale
            
        Returns:
            variability: ROI-wise variability scores [rois]
        """
        if x.dim() == 3:
            # [batch, time, rois] -> [time, rois] by averaging batch
            x = x.mean(dim=0)
        
        assert x.dim() == 2, f"Expected 2D tensor [time, rois], got {x.shape}"
        
        T, R = x.shape
        
        # fMRI typically has longer time scale
        window_size = min(self.window_size * 3, T)  # 3x longer window for fMRI
        
        variabilities = []
        
        # 1. Global signal removal + ROI variance
        # Global signal = mean across all ROIs at each time point
        global_signal = x.mean(dim=1, keepdim=True)
        x_residual = x - global_signal
        
        # Variance of residual signal
        residual_var = x_residual.var(dim=0)
        variabilities.append(residual_var)
        
        # 2. Functional connectivity (FC) changes
        if T >= window_size * 2:
            fc_change = torch.zeros(R, device=x.device)
            num_pairs = T - window_size * 2 + 1
            
            for start in range(0, num_pairs, max(1, window_size // 2)):
                mid = start + window_size
                end = mid + window_size
                if end > T:
                    break
                
                # Compute FC in two consecutive windows
                window1 = x_residual[start:mid, :]
                window2 = x_residual[mid:end, :]
                
                # Correlation matrices
                fc1 = self._compute_fc(window1)
                fc2 = self._compute_fc(window2)
                
                # FC change per ROI (mean absolute change in connections)
                fc_diff = (fc1 - fc2).abs()
                roi_fc_change = fc_diff.mean(dim=1)  # Mean change per ROI
                fc_change += roi_fc_change
            
            fc_change = fc_change / max(1, (num_pairs + window_size // 2 - 1) // (window_size // 2))
            variabilities.append(fc_change)
        
        # 3. Low-frequency power (using simple moving average as lowpass)
        if T >= window_size:
            # Apply moving average to get low-frequency component
            kernel_size = min(11, T // 4)  # Use ~25% of sequence as kernel
            if kernel_size >= 3:
                x_lowfreq = self._moving_average(x_residual, kernel_size)
                lowfreq_var = x_lowfreq.var(dim=0)
                variabilities.append(lowfreq_var)
        
        # Combine variabilities
        if len(variabilities) == 0:
            variability = x.var(dim=0)
        else:
            variability = torch.stack(variabilities).mean(dim=0)
        
        # Normalize to [0, 1] range if requested
        if normalize:
            v_min = variability.min()
            v_max = variability.max()
            if v_max > v_min:
                variability = (variability - v_min) / (v_max - v_min + self.epsilon)
        
        return variability
    
    def _compute_fc(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute functional connectivity (correlation) matrix.
        
        Args:
            x: Time series data [time, rois]
            
        Returns:
            fc: Correlation matrix [rois, rois]
        """
        x_centered = x - x.mean(dim=0, keepdim=True)
        x_norm = x_centered / (x_centered.std(dim=0, keepdim=True) + self.epsilon)
        fc = torch.matmul(x_norm.T, x_norm) / x.shape[0]
        return fc
    
    def _moving_average(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Apply moving average filter along time dimension.
        
        Args:
            x: Input [time, features]
            kernel_size: Size of averaging window
            
        Returns:
            Smoothed signal [time, features]
        """
        # Pad at edges
        pad = kernel_size // 2
        x_padded = F.pad(x.T.unsqueeze(0), (pad, pad), mode='replicate')  # [1, features, time]
        
        # Create averaging kernel
        kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
        
        # Apply convolution for each feature independently
        x_smooth = F.conv1d(x_padded, kernel, groups=1)  # [1, features, time]
        
        return x_smooth.squeeze(0).T  # [time, features]


class VariabilityWeightMapper:
    """
    Maps variability scores to training weights using temperature-controlled softmax.
    
    Implements the mapping: w = softmax(C / τ)
    where C is variability, τ is temperature, and w are the weights.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        min_weight: float = 0.01,
        epsilon: float = 1e-8
    ):
        """
        Initialize weight mapper.
        
        Args:
            temperature: Temperature parameter for softmax
            min_weight: Minimum weight to prevent complete suppression
            epsilon: Small constant for numerical stability
        """
        self.temperature = temperature
        self.min_weight = min_weight
        self.epsilon = epsilon
    
    def compute_weights(
        self,
        variability: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute training weights from variability scores.
        
        Args:
            variability: Variability scores [channels/rois]
            temperature: Optional temperature override
            
        Returns:
            weights: Normalized weights [channels/rois], sum to 1
        """
        if temperature is None:
            temperature = self.temperature
        
        # Apply temperature scaling
        scaled = variability / (temperature + self.epsilon)
        
        # Softmax to get weights
        weights = F.softmax(scaled, dim=0)
        
        # Apply minimum weight threshold
        weights = torch.maximum(weights, torch.tensor(self.min_weight, device=weights.device))
        
        # Re-normalize after applying minimum
        weights = weights / (weights.sum() + self.epsilon)
        
        return weights


class TrainingStageScheduler:
    """
    Schedules temperature and other parameters across three training stages:
    - Warmup: Small τ (sharp focus on high-variability channels)
    - Main Training: Gradually increase τ (broaden participation)
    - Finetuning: Large τ or disable (prevent overfitting to local patterns)
    """
    
    def __init__(
        self,
        warmup_epochs: int = 5,
        main_epochs: int = 60,
        finetune_epochs: int = 30,
        warmup_temp: float = 0.1,
        main_temp_start: float = 0.1,
        main_temp_end: float = 1.0,
        finetune_temp: float = 2.0,
        disable_in_finetune: bool = False
    ):
        """
        Initialize training stage scheduler.
        
        Args:
            warmup_epochs: Number of warmup epochs
            main_epochs: Number of main training epochs
            finetune_epochs: Number of finetuning epochs
            warmup_temp: Temperature during warmup (small = sharp focus)
            main_temp_start: Starting temperature for main training
            main_temp_end: Ending temperature for main training
            finetune_temp: Temperature during finetuning (large = flat distribution)
            disable_in_finetune: If True, return uniform weights in finetune
        """
        self.warmup_epochs = warmup_epochs
        self.main_epochs = main_epochs
        self.finetune_epochs = finetune_epochs
        self.warmup_temp = warmup_temp
        self.main_temp_start = main_temp_start
        self.main_temp_end = main_temp_end
        self.finetune_temp = finetune_temp
        self.disable_in_finetune = disable_in_finetune
        
        self.total_epochs = warmup_epochs + main_epochs + finetune_epochs
    
    def get_temperature(self, epoch: int) -> float:
        """
        Get temperature for current epoch.
        
        Args:
            epoch: Current epoch (1-indexed)
            
        Returns:
            temperature: Temperature value for this epoch
        """
        if epoch <= self.warmup_epochs:
            # Warmup: use small temperature
            return self.warmup_temp
        
        elif epoch <= self.warmup_epochs + self.main_epochs:
            # Main training: linearly increase temperature
            progress = (epoch - self.warmup_epochs) / max(1, self.main_epochs)
            temp = self.main_temp_start + progress * (self.main_temp_end - self.main_temp_start)
            return temp
        
        else:
            # Finetuning: use large temperature or disable
            if self.disable_in_finetune:
                return float('inf')  # Results in uniform weights
            return self.finetune_temp
    
    def get_stage_name(self, epoch: int) -> str:
        """
        Get name of current training stage.
        
        Args:
            epoch: Current epoch (1-indexed)
            
        Returns:
            stage_name: 'warmup', 'main', or 'finetune'
        """
        if epoch <= self.warmup_epochs:
            return 'warmup'
        elif epoch <= self.warmup_epochs + self.main_epochs:
            return 'main'
        else:
            return 'finetune'
    
    def is_warmup(self, epoch: int) -> bool:
        """Check if in warmup stage."""
        return epoch <= self.warmup_epochs
    
    def is_main(self, epoch: int) -> bool:
        """Check if in main training stage."""
        return self.warmup_epochs < epoch <= self.warmup_epochs + self.main_epochs
    
    def is_finetune(self, epoch: int) -> bool:
        """Check if in finetuning stage."""
        return epoch > self.warmup_epochs + self.main_epochs


class DynamicVariabilityWeighting:
    """
    Complete dynamic variability-based weighting system.
    
    Integrates variability computation, weight mapping, and stage scheduling
    for unsupervised multimodal training with EEG and fMRI.
    """
    
    def __init__(
        self,
        # Variability computation
        eeg_window_size: int = 50,
        fmri_window_size: int = 150,  # 3x longer for slower fMRI dynamics
        # Weight mapping
        min_weight: float = 0.01,
        # Stage scheduling
        warmup_epochs: int = 5,
        main_epochs: int = 60,
        finetune_epochs: int = 30,
        warmup_temp: float = 0.1,
        main_temp_start: float = 0.1,
        main_temp_end: float = 1.0,
        finetune_temp: float = 2.0,
        disable_in_finetune: bool = False,
        # Modality-specific options
        eeg_use_first_order_diff: bool = True,
        eeg_use_covariance: bool = False,
        fmri_use_fc_change: bool = True,
        # General
        epsilon: float = 1e-8,
        enabled: bool = True
    ):
        """
        Initialize complete dynamic weighting system.
        
        Args:
            eeg_window_size: Window size for EEG variability computation
            fmri_window_size: Window size for fMRI variability computation
            min_weight: Minimum weight to prevent complete suppression
            warmup_epochs: Number of warmup epochs
            main_epochs: Number of main training epochs
            finetune_epochs: Number of finetuning epochs
            warmup_temp: Temperature during warmup
            main_temp_start: Starting temperature for main training
            main_temp_end: Ending temperature for main training
            finetune_temp: Temperature during finetuning
            disable_in_finetune: If True, use uniform weights in finetune
            eeg_use_first_order_diff: Use first-order differences for EEG
            eeg_use_covariance: Use covariance participation for EEG
            fmri_use_fc_change: Use FC changes for fMRI
            epsilon: Small constant for numerical stability
            enabled: If False, return uniform weights (for ablation studies)
        """
        self.enabled = enabled
        self.epsilon = epsilon
        
        # Initialize components
        self.eeg_computer = VariabilityComputer(
            window_size=eeg_window_size,
            use_temporal_variance=True,
            use_first_order_diff=eeg_use_first_order_diff,
            use_spectral_change=False,  # Too expensive for real-time
            use_covariance_participation=eeg_use_covariance,
            epsilon=epsilon
        )
        
        self.fmri_computer = VariabilityComputer(
            window_size=fmri_window_size,
            use_temporal_variance=True,
            use_first_order_diff=False,  # fMRI focuses on FC and slow dynamics
            use_spectral_change=False,
            use_covariance_participation=fmri_use_fc_change,  # Reuse this for FC
            epsilon=epsilon
        )
        
        self.weight_mapper = VariabilityWeightMapper(
            temperature=1.0,  # Will be overridden by scheduler
            min_weight=min_weight,
            epsilon=epsilon
        )
        
        self.stage_scheduler = TrainingStageScheduler(
            warmup_epochs=warmup_epochs,
            main_epochs=main_epochs,
            finetune_epochs=finetune_epochs,
            warmup_temp=warmup_temp,
            main_temp_start=main_temp_start,
            main_temp_end=main_temp_end,
            finetune_temp=finetune_temp,
            disable_in_finetune=disable_in_finetune
        )
        
        # Cache for variability scores (updated periodically)
        self.variability_cache = {}
        self.weight_cache = {}
    
    def compute_modality_weights(
        self,
        x: torch.Tensor,
        modality: str,
        epoch: int,
        force_update: bool = False
    ) -> torch.Tensor:
        """
        Compute weights for a specific modality at given epoch.
        
        Args:
            x: Input data [batch, time, channels/rois] or [time, channels/rois]
            modality: 'eeg' or 'fmri'
            epoch: Current training epoch
            force_update: Force recomputation even if cached
            
        Returns:
            weights: Channel/ROI weights [channels/rois], sum to 1
        """
        if not self.enabled:
            # Return uniform weights
            if x.dim() == 3:
                num_features = x.shape[2]
            else:
                num_features = x.shape[1]
            return torch.ones(num_features, device=x.device) / num_features
        
        # Check cache
        cache_key = f"{modality}_epoch{epoch}"
        if cache_key in self.weight_cache and not force_update:
            cached_weights = self.weight_cache[cache_key]
            # Ensure weights are on correct device
            if cached_weights.device != x.device:
                cached_weights = cached_weights.to(x.device)
            return cached_weights
        
        # Compute variability
        if modality.lower() == 'eeg':
            variability = self.eeg_computer.compute_eeg_variability(x, normalize=True)
        elif modality.lower() == 'fmri':
            variability = self.fmri_computer.compute_fmri_variability(x, normalize=True)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Get temperature for current stage
        temperature = self.stage_scheduler.get_temperature(epoch)
        
        # Map to weights
        if temperature == float('inf'):
            # Uniform weights in finetune if disabled
            weights = torch.ones_like(variability) / len(variability)
        else:
            weights = self.weight_mapper.compute_weights(variability, temperature)
        
        # Cache results
        self.variability_cache[cache_key] = variability.detach().cpu()
        self.weight_cache[cache_key] = weights.detach()
        
        return weights
    
    def get_stage_info(self, epoch: int) -> Dict[str, Union[str, float]]:
        """
        Get information about current training stage.
        
        Args:
            epoch: Current epoch
            
        Returns:
            info: Dictionary with stage name and temperature
        """
        return {
            'stage': self.stage_scheduler.get_stage_name(epoch),
            'temperature': self.stage_scheduler.get_temperature(epoch),
            'enabled': self.enabled
        }
    
    def clear_cache(self):
        """Clear variability and weight caches."""
        self.variability_cache.clear()
        self.weight_cache.clear()


def create_dynamic_weighting_from_config(config: Dict) -> DynamicVariabilityWeighting:
    """
    Create DynamicVariabilityWeighting from configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'dynamic_weighting' section
        
    Returns:
        DynamicVariabilityWeighting instance
    """
    dw_config = config.get('dynamic_weighting', {})
    
    # Training stage epochs (fallback to training config if not in dynamic_weighting)
    training_config = config.get('training', {})
    warmup_epochs = dw_config.get('warmup_epochs', training_config.get('warmup_epochs', 5))
    main_epochs = dw_config.get('main_epochs', training_config.get('main_epochs', 60))
    finetune_epochs = dw_config.get('finetune_epochs', training_config.get('finetune_epochs', 30))
    
    return DynamicVariabilityWeighting(
        # Variability computation
        eeg_window_size=dw_config.get('eeg_window_size', 50),
        fmri_window_size=dw_config.get('fmri_window_size', 150),
        # Weight mapping
        min_weight=dw_config.get('min_weight', 0.01),
        # Stage scheduling
        warmup_epochs=warmup_epochs,
        main_epochs=main_epochs,
        finetune_epochs=finetune_epochs,
        warmup_temp=dw_config.get('warmup_temp', 0.1),
        main_temp_start=dw_config.get('main_temp_start', 0.1),
        main_temp_end=dw_config.get('main_temp_end', 1.0),
        finetune_temp=dw_config.get('finetune_temp', 2.0),
        disable_in_finetune=dw_config.get('disable_in_finetune', False),
        # Modality options
        eeg_use_first_order_diff=dw_config.get('eeg_use_first_order_diff', True),
        eeg_use_covariance=dw_config.get('eeg_use_covariance', False),
        fmri_use_fc_change=dw_config.get('fmri_use_fc_change', True),
        # General
        enabled=dw_config.get('enabled', True)
    )
