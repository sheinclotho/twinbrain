"""
Adaptive Loss Balancing for Multi-Task Multi-Modal Learning
============================================================

Implements GradNorm-inspired adaptive loss weighting to handle:
1. EEG-fMRI energy imbalance (10-100x difference)
2. Multi-task learning with varying difficulty
3. Dynamic loss weight adjustment based on training dynamics

Key Features:
- Per-modality gradient normalization
- Automatic loss weight adaptation
- Energy-aware scaling for EEG/fMRI balance
- Gradient magnitude-based weighting

References:
    - GradNorm: Gradient Normalization for Adaptive Loss Balancing (Chen et al., 2018)
    - Multi-Task Learning Using Uncertainty to Weigh Losses (Kendall et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AdaptiveLossBalancer(nn.Module):
    """
    Adaptive loss balancing using gradient normalization.
    
    Dynamically adjusts loss weights based on:
    1. Relative training rates across tasks
    2. Gradient magnitudes per task
    3. Modality-specific energy characteristics
    """
    
    def __init__(
        self,
        task_names: List[str],
        modality_names: Optional[List[str]] = None,
        initial_weights: Optional[Dict[str, float]] = None,
        alpha: float = 1.5,
        update_frequency: int = 10,
        learning_rate: float = 0.025,
        warmup_epochs: int = 5,
        enable_modality_scaling: bool = True,
        modality_energy_ratios: Optional[Dict[str, float]] = None,
        min_weight: float = 0.01,
        max_weight: float = 100.0,
    ):
        """
        Initialize adaptive loss balancer.
        
        Args:
            task_names: List of task names (e.g., ['recon', 'temp_pred', 'align'])
            modality_names: List of modality names (e.g., ['eeg', 'fmri'])
            initial_weights: Initial task weights (defaults to 1.0 for all)
            alpha: Restoring force for balancing (higher = more aggressive)
            update_frequency: Update weights every N steps
            learning_rate: Learning rate for weight updates
            warmup_epochs: Number of epochs before enabling adaptation
            enable_modality_scaling: Enable per-modality gradient scaling
            modality_energy_ratios: Energy ratio for each modality (e.g., {'eeg': 0.01, 'fmri': 1.0})
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
        """
        super().__init__()
        
        self.task_names = task_names
        self.modality_names = modality_names or []
        self.alpha = alpha
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.enable_modality_scaling = enable_modality_scaling
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Initialize task weights
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in task_names}
        
        # Create learnable weights (in log space for stability)
        log_weights = {
            name: torch.log(torch.tensor(initial_weights.get(name, 1.0)))
            for name in task_names
        }
        self.log_weights = nn.ParameterDict({
            name: nn.Parameter(w) for name, w in log_weights.items()
        })
        
        # Modality energy ratios (for EEG-fMRI balancing)
        if modality_energy_ratios is None:
            # Default: fMRI has ~50x more energy than EEG
            modality_energy_ratios = {'eeg': 0.02, 'fmri': 1.0}
        
        self.register_buffer(
            'modality_energy_ratios',
            torch.tensor([modality_energy_ratios.get(m, 1.0) for m in self.modality_names])
            if self.modality_names else None
        )
        
        # Track training dynamics
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('epoch_count', torch.tensor(0))
        
        # Track initial loss values for normalization
        self.register_buffer('initial_losses', torch.zeros(len(task_names)))
        self.register_buffer('initial_losses_set', torch.tensor(False))
        
        # Track loss history for adaptive adjustment
        self.loss_history = {name: [] for name in task_names}
        self.grad_norm_history = {name: [] for name in task_names}
        
    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        modality_losses: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        return_weighted: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss.
        
        Args:
            losses: Dictionary of task losses {task_name: loss_value}
            modality_losses: Optional nested dict {modality: {task: loss}} for per-modality scaling
            return_weighted: If True, return weighted sum; else return per-task weighted losses
            
        Returns:
            total_loss: Weighted sum of losses
            weight_dict: Current weights for logging
        """
        # Get current weights
        weights = {name: torch.exp(self.log_weights[name]).clamp(self.min_weight, self.max_weight) 
                   for name in self.task_names}
        
        # Set initial losses on first call (for normalization)
        if not self.initial_losses_set:
            self.initial_losses = torch.tensor([
                losses.get(name, torch.tensor(0.0)).detach().item()
                for name in self.task_names
            ], device=self.initial_losses.device)
            self.initial_losses_set = torch.tensor(True)
        
        # Compute weighted losses
        weighted_losses = {}
        for i, name in enumerate(self.task_names):
            if name in losses:
                # Normalize by initial loss to prevent scale differences
                loss_norm = losses[name]
                if self.initial_losses[i] > 1e-6:
                    loss_norm = loss_norm / (self.initial_losses[i] + 1e-8)
                
                weighted_losses[name] = weights[name] * loss_norm
            else:
                weighted_losses[name] = torch.tensor(0.0, device=list(losses.values())[0].device)
        
        # Apply modality-specific scaling if enabled
        if self.enable_modality_scaling and modality_losses is not None:
            weighted_losses = self._apply_modality_scaling(
                weighted_losses, modality_losses
            )
        
        # Compute total loss
        if return_weighted:
            total_loss = sum(weighted_losses.values())
        else:
            total_loss = weighted_losses
        
        # Return weights for logging (detached)
        weight_dict = {name: w.detach().item() for name, w in weights.items()}
        
        return total_loss, weight_dict
    
    def _apply_modality_scaling(
        self,
        weighted_losses: Dict[str, torch.Tensor],
        modality_losses: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Apply per-modality energy scaling to handle EEG-fMRI imbalance.
        
        Args:
            weighted_losses: Current weighted losses
            modality_losses: Per-modality losses {modality: {task: loss}}
            
        Returns:
            scaled_losses: Losses with modality scaling applied
        """
        if self.modality_energy_ratios is None or len(self.modality_names) == 0:
            return weighted_losses
        
        scaled_losses = {}
        
        for task_name, loss_val in weighted_losses.items():
            # Check if we have modality breakdown for this task
            task_modality_losses = {}
            for i, modality in enumerate(self.modality_names):
                if modality in modality_losses and task_name in modality_losses[modality]:
                    task_modality_losses[modality] = modality_losses[modality][task_name]
            
            if len(task_modality_losses) > 0:
                # Apply energy scaling to each modality component
                scaled_components = []
                for i, modality in enumerate(self.modality_names):
                    if modality in task_modality_losses:
                        # Scale by inverse of energy ratio to balance contributions
                        # EEG (low energy) gets higher weight, fMRI (high energy) gets lower weight
                        scale = 1.0 / (self.modality_energy_ratios[i] + 1e-8)
                        scaled_components.append(task_modality_losses[modality] * scale)
                
                # Average scaled components
                if len(scaled_components) > 0:
                    scaled_losses[task_name] = torch.stack(scaled_components).mean()
                else:
                    scaled_losses[task_name] = loss_val
            else:
                # No modality breakdown, use original
                scaled_losses[task_name] = loss_val
        
        return scaled_losses
    
    def update_weights(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        shared_params: Optional[List[nn.Parameter]] = None,
    ):
        """
        Update loss weights based on gradient magnitudes (GradNorm).
        
        Args:
            losses: Dictionary of task losses
            model: Model to compute gradients on
            shared_params: Shared parameters to compute gradient norms on
                          (if None, uses all model parameters)
        """
        # Only update after warmup and at specified frequency
        if self.epoch_count < self.warmup_epochs:
            return
        
        if self.step_count % self.update_frequency != 0:
            self.step_count += 1
            return
        
        self.step_count += 1
        
        # Get shared parameters
        if shared_params is None:
            shared_params = [p for p in model.parameters() if p.requires_grad]
        
        # Compute gradient norms for each task
        grad_norms = {}
        
        for name in self.task_names:
            if name not in losses:
                continue
            
            # Compute gradients for this task
            task_loss = losses[name]
            
            # Get gradients
            grads = torch.autograd.grad(
                task_loss,
                shared_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
            
            # Compute gradient norm
            grad_norm = 0.0
            for g in grads:
                if g is not None:
                    grad_norm += g.detach().norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            grad_norms[name] = grad_norm
            self.grad_norm_history[name].append(grad_norm)
        
        # Compute average gradient norm
        avg_grad_norm = sum(grad_norms.values()) / max(len(grad_norms), 1)
        
        # Compute relative inverse training rates
        # Tasks with smaller gradients are training slower, so increase their weight
        with torch.no_grad():
            for name in self.task_names:
                if name not in grad_norms or grad_norms[name] < 1e-8:
                    continue
                
                # Relative gradient norm
                rel_grad = grad_norms[name] / (avg_grad_norm + 1e-8)
                
                # Target: balanced gradient norms (rel_grad ≈ 1.0)
                # If rel_grad < 1: task is training slower, increase weight
                # If rel_grad > 1: task is training faster, decrease weight
                target_rel = 1.0
                
                # Compute gradient for weight update (GradNorm loss)
                # L_grad = |G_i(t) / avg(G(t)) - r_i(t)|^α
                # where r_i(t) is the relative inverse training rate
                grad_loss = abs(rel_grad - target_rel) ** self.alpha
                
                # Update weight (gradient descent on grad_loss)
                # Increase weight if task is training slower
                weight_update = -self.learning_rate * (rel_grad - target_rel)
                
                # Update in log space for stability
                self.log_weights[name].data += weight_update
                
                # Clamp to reasonable range
                max_log_weight = torch.log(torch.tensor(self.max_weight))
                min_log_weight = torch.log(torch.tensor(self.min_weight))
                self.log_weights[name].data.clamp_(min_log_weight, max_log_weight)
        
        # Log current weights
        if logger.isEnabledFor(logging.DEBUG):
            weights = {name: torch.exp(self.log_weights[name]).item() 
                      for name in self.task_names}
            logger.debug(f"Updated loss weights: {weights}")
            logger.debug(f"Gradient norms: {grad_norms}")
    
    def set_epoch(self, epoch: int):
        """Set current epoch for warmup tracking."""
        self.epoch_count = torch.tensor(epoch)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights as dictionary."""
        return {
            name: torch.exp(self.log_weights[name]).clamp(self.min_weight, self.max_weight).item()
            for name in self.task_names
        }
    
    def reset_history(self):
        """Reset loss and gradient history."""
        self.loss_history = {name: [] for name in self.task_names}
        self.grad_norm_history = {name: [] for name in self.task_names}


class ModalityGradientScaler:
    """
    Per-modality gradient scaling to handle energy imbalances.
    
    Scales gradients for each modality based on their relative energy levels.
    This ensures EEG (low energy) and fMRI (high energy) contribute equally to training.
    """
    
    def __init__(
        self,
        modality_names: List[str],
        energy_ratios: Optional[Dict[str, float]] = None,
        scale_method: str = 'inverse',  # 'inverse' or 'sqrt_inverse'
        adaptive: bool = True,
        update_frequency: int = 100,
    ):
        """
        Initialize modality gradient scaler.
        
        Args:
            modality_names: List of modality names
            energy_ratios: Relative energy for each modality
            scale_method: How to compute scale from energy ratio
            adaptive: Adapt scales based on gradient statistics
            update_frequency: Update adaptive scales every N steps
        """
        self.modality_names = modality_names
        self.scale_method = scale_method
        self.adaptive = adaptive
        self.update_frequency = update_frequency
        
        # Initialize energy ratios
        if energy_ratios is None:
            # Default: fMRI 50x more energy than EEG
            energy_ratios = {'eeg': 0.02, 'fmri': 1.0}
        
        self.energy_ratios = energy_ratios
        
        # Compute initial scales
        self.scales = self._compute_scales(energy_ratios)
        
        # Track gradient statistics for adaptive scaling
        self.grad_stats = {name: {'mean': 0.0, 'count': 0} for name in modality_names}
        self.step_count = 0
    
    def _compute_scales(self, energy_ratios: Dict[str, float]) -> Dict[str, float]:
        """Compute gradient scales from energy ratios."""
        scales = {}
        
        for name in self.modality_names:
            energy = energy_ratios.get(name, 1.0)
            
            if self.scale_method == 'inverse':
                # Inverse: low energy -> high scale
                scale = 1.0 / (energy + 1e-8)
            elif self.scale_method == 'sqrt_inverse':
                # Square root inverse: softer scaling
                scale = 1.0 / (energy ** 0.5 + 1e-8)
            else:
                scale = 1.0
            
            scales[name] = scale
        
        # Normalize scales to have mean = 1.0
        mean_scale = sum(scales.values()) / len(scales)
        scales = {name: s / mean_scale for name, s in scales.items()}
        
        return scales
    
    def scale_gradients(
        self,
        modality_losses: Dict[str, torch.Tensor],
        model: nn.Module,
        modality_params: Dict[str, List[nn.Parameter]],
    ):
        """
        Scale gradients for each modality.
        
        Args:
            modality_losses: Loss for each modality
            model: Model
            modality_params: Parameters associated with each modality
        """
        self.step_count += 1
        
        for modality, loss in modality_losses.items():
            if modality not in self.modality_names:
                continue
            
            # Get scale for this modality
            scale = self.scales.get(modality, 1.0)
            
            # Get parameters for this modality
            params = modality_params.get(modality, [])
            if len(params) == 0:
                continue
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
            
            # Scale gradients
            for param, grad in zip(params, grads):
                if grad is not None and param.grad is not None:
                    param.grad.data.mul_(scale)
            
            # Track gradient statistics
            if self.adaptive:
                grad_norm = sum(g.norm().item() for g in grads if g is not None)
                self._update_grad_stats(modality, grad_norm)
        
        # Update scales adaptively
        if self.adaptive and self.step_count % self.update_frequency == 0:
            self._update_scales()
    
    def _update_grad_stats(self, modality: str, grad_norm: float):
        """Update gradient statistics for adaptive scaling."""
        stats = self.grad_stats[modality]
        stats['mean'] = (stats['mean'] * stats['count'] + grad_norm) / (stats['count'] + 1)
        stats['count'] += 1
    
    def _update_scales(self):
        """Update scales based on gradient statistics."""
        # Compute average gradient norm across modalities
        grad_norms = {name: stats['mean'] for name, stats in self.grad_stats.items()}
        avg_grad = sum(grad_norms.values()) / max(len(grad_norms), 1)
        
        # Adjust scales to balance gradient norms
        for modality in self.modality_names:
            if grad_norms[modality] < 1e-8:
                continue
            
            # If gradient too small, increase scale; if too large, decrease scale
            rel_grad = grad_norms[modality] / (avg_grad + 1e-8)
            
            # Exponential moving average update
            adjust = 1.0 / (rel_grad + 1e-8)
            self.scales[modality] = 0.9 * self.scales[modality] + 0.1 * adjust
        
        # Normalize scales
        mean_scale = sum(self.scales.values()) / len(self.scales)
        self.scales = {name: s / mean_scale for name, s in self.scales.items()}
        
        # Reset statistics
        self.grad_stats = {name: {'mean': 0.0, 'count': 0} for name in self.modality_names}
    
    def get_scales(self) -> Dict[str, float]:
        """Get current gradient scales."""
        return self.scales.copy()
