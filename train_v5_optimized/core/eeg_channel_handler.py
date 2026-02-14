"""
Enhanced EEG Channel Handling
==============================

Addresses the silent channel problem in EEG data where:
1. Channels mapped to more outputs + low energy → many silent channels
2. Zero-value fitting becomes excellent → difficult to train
3. Channels are completely ignored in gradient flow

Solutions:
- SNR-based channel pre-filtering
- Adaptive channel dropout/masking  
- Channel activity monitoring and auto-scaling
- Soft channel attention mechanism
- Anti-collapse regularization

Key Features:
- Real-time channel health monitoring
- Dynamic channel importance weighting
- Gradient-aware channel dropout
- Anti-zero-collapse loss terms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ChannelActivityMonitor:
    """
    Monitors channel activity and health metrics in real-time.
    
    Tracks:
    - Signal-to-noise ratio (SNR)
    - Temporal variance
    - Gradient magnitude
    - Reconstruction error
    - Activity ratio (non-zero proportion)
    """
    
    def __init__(
        self,
        num_channels: int,
        window_size: int = 100,
        snr_threshold: float = 1.0,
        variance_threshold: float = 1e-6,
        gradient_threshold: float = 1e-8,
        activity_threshold: float = 0.1,
        update_frequency: int = 10,
    ):
        """
        Initialize channel activity monitor.
        
        Args:
            num_channels: Number of EEG channels
            window_size: Window size for computing statistics
            snr_threshold: Minimum SNR for healthy channel
            variance_threshold: Minimum variance for healthy channel
            gradient_threshold: Minimum gradient for active training
            activity_threshold: Minimum non-zero proportion
            update_frequency: Update metrics every N batches
        """
        self.num_channels = num_channels
        self.window_size = window_size
        self.snr_threshold = snr_threshold
        self.variance_threshold = variance_threshold
        self.gradient_threshold = gradient_threshold
        self.activity_threshold = activity_threshold
        self.update_frequency = update_frequency
        
        # Initialize metric buffers
        self.snr_buffer = torch.zeros(num_channels)
        self.variance_buffer = torch.zeros(num_channels)
        self.gradient_buffer = torch.zeros(num_channels)
        self.activity_buffer = torch.zeros(num_channels)
        self.error_buffer = torch.zeros(num_channels)
        
        # Track update count
        self.update_count = 0
        self.step_count = 0
        
        # Channel health status
        self.channel_health = torch.ones(num_channels)  # 1.0 = healthy, 0.0 = dead
        self.channel_importance = torch.ones(num_channels)  # Relative importance weights
    
    def update(
        self,
        signals: torch.Tensor,
        gradients: Optional[torch.Tensor] = None,
        reconstructions: Optional[torch.Tensor] = None,
    ):
        """
        Update channel statistics.
        
        Args:
            signals: Input signals [batch, time, channels] or [time, channels]
            gradients: Gradients w.r.t. signals (same shape)
            reconstructions: Reconstructed signals (same shape)
        """
        self.step_count += 1
        
        if self.step_count % self.update_frequency != 0:
            return
        
        # Ensure 2D: [time, channels]
        if signals.dim() == 3:
            signals = signals.reshape(-1, signals.shape[-1])
        if gradients is not None and gradients.dim() == 3:
            gradients = gradients.reshape(-1, gradients.shape[-1])
        if reconstructions is not None and reconstructions.dim() == 3:
            reconstructions = reconstructions.reshape(-1, reconstructions.shape[-1])
        
        # Move to CPU for statistics (avoid GPU memory issues)
        signals_cpu = signals.detach().cpu()
        
        # 1. Compute SNR (signal power / noise power estimate)
        signal_power = signals_cpu.var(dim=0)
        noise_estimate = (signals_cpu[1:] - signals_cpu[:-1]).var(dim=0)  # High-freq noise
        snr = signal_power / (noise_estimate + 1e-10)
        self.snr_buffer = 0.9 * self.snr_buffer + 0.1 * snr
        
        # 2. Compute temporal variance
        variance = signals_cpu.var(dim=0)
        self.variance_buffer = 0.9 * self.variance_buffer + 0.1 * variance
        
        # 3. Compute gradient magnitude
        if gradients is not None:
            grad_mag = gradients.detach().cpu().abs().mean(dim=0)
            self.gradient_buffer = 0.9 * self.gradient_buffer + 0.1 * grad_mag
        
        # 4. Compute activity ratio (proportion of non-near-zero values)
        threshold = signals_cpu.abs().mean() * 0.01  # 1% of mean absolute value
        activity = (signals_cpu.abs() > threshold).float().mean(dim=0)
        self.activity_buffer = 0.9 * self.activity_buffer + 0.1 * activity
        
        # 5. Compute reconstruction error
        if reconstructions is not None:
            recon_cpu = reconstructions.detach().cpu()
            error = (signals_cpu - recon_cpu).pow(2).mean(dim=0)
            self.error_buffer = 0.9 * self.error_buffer + 0.1 * error
        
        # Update health status
        self._update_channel_health()
        
        self.update_count += 1
    
    def _update_channel_health(self):
        """Update channel health status based on metrics."""
        # Health criteria (all must pass for healthy channel)
        snr_healthy = (self.snr_buffer > self.snr_threshold).float()
        var_healthy = (self.variance_buffer > self.variance_threshold).float()
        grad_healthy = (self.gradient_buffer > self.gradient_threshold).float()
        activity_healthy = (self.activity_buffer > self.activity_threshold).float()
        
        # Combine criteria (soft AND with multiplication)
        health = snr_healthy * var_healthy * grad_healthy * activity_healthy
        
        # Smooth health status to avoid abrupt changes
        self.channel_health = 0.95 * self.channel_health + 0.05 * health
        
        # Compute importance weights (higher for healthier channels)
        # Use sigmoid to create smooth transition
        self.channel_importance = torch.sigmoid(5.0 * (self.channel_health - 0.5))
    
    def get_healthy_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary mask of healthy channels.
        
        Args:
            threshold: Health threshold for mask
            
        Returns:
            mask: Boolean mask [channels]
        """
        return self.channel_health > threshold
    
    def get_importance_weights(self) -> torch.Tensor:
        """Get channel importance weights [channels]."""
        return self.channel_importance
    
    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get all channel statistics."""
        return {
            'snr': self.snr_buffer,
            'variance': self.variance_buffer,
            'gradient': self.gradient_buffer,
            'activity': self.activity_buffer,
            'error': self.error_buffer,
            'health': self.channel_health,
            'importance': self.channel_importance,
        }
    
    def log_channel_status(self, logger_obj: Optional[logging.Logger] = None):
        """Log current channel status."""
        if logger_obj is None:
            logger_obj = logger
        
        healthy_count = (self.channel_health > 0.5).sum().item()
        dead_count = (self.channel_health < 0.2).sum().item()
        
        logger_obj.info(
            f"Channel Status: {healthy_count}/{self.num_channels} healthy, "
            f"{dead_count} dead/silent"
        )
        
        if dead_count > 0:
            dead_indices = (self.channel_health < 0.2).nonzero(as_tuple=True)[0].tolist()
            logger_obj.warning(f"Silent channels: {dead_indices[:10]}...")


class AdaptiveChannelDropout(nn.Module):
    """
    Adaptive dropout that drops channels based on their health/importance.
    
    Unlike standard dropout:
    - Drops unhealthy/silent channels with higher probability
    - Keeps healthy/active channels with higher probability
    - Adapts dropout rate based on training progress
    """
    
    def __init__(
        self,
        dropout_rate: float = 0.1,
        importance_based: bool = True,
        invert_importance: bool = True,  # Drop unhealthy channels more
        warmup_steps: int = 1000,
    ):
        """
        Initialize adaptive channel dropout.
        
        Args:
            dropout_rate: Base dropout rate
            importance_based: Use channel importance for dropout probability
            invert_importance: Drop less important channels more (recommended)
            warmup_steps: Steps before enabling adaptive dropout
        """
        super().__init__()
        
        self.dropout_rate = dropout_rate
        self.importance_based = importance_based
        self.invert_importance = invert_importance
        self.warmup_steps = warmup_steps
        
        self.step_count = 0
    
    def forward(
        self,
        x: torch.Tensor,
        channel_importance: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Apply adaptive channel dropout.
        
        Args:
            x: Input tensor [batch, time, channels]
            channel_importance: Importance weights [channels]
            training: Whether in training mode
            
        Returns:
            x_dropped: Tensor with dropout applied
        """
        if not training or self.dropout_rate == 0.0:
            return x
        
        self.step_count += 1
        
        # Warmup: use standard dropout
        if self.step_count < self.warmup_steps or channel_importance is None:
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.dropout_rate))
            return x * mask / (1 - self.dropout_rate + 1e-8)
        
        # Adaptive dropout based on importance
        if self.importance_based:
            # Expand importance to match x shape
            importance = channel_importance.view(1, 1, -1).expand_as(x)
            
            if self.invert_importance:
                # Drop unhealthy channels more frequently
                # Low importance → high dropout prob
                dropout_prob = self.dropout_rate * (1.0 - importance)
            else:
                # Drop healthy channels more frequently (regularization)
                # High importance → high dropout prob
                dropout_prob = self.dropout_rate * importance
            
            # Clamp dropout probability
            dropout_prob = dropout_prob.clamp(0.0, 0.9)
            
            # Sample dropout mask
            keep_prob = 1.0 - dropout_prob
            mask = torch.bernoulli(keep_prob)
            
            # Apply dropout with importance-based scaling
            return x * mask / (keep_prob + 1e-8)
        else:
            # Standard dropout
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.dropout_rate))
            return x * mask / (1 - self.dropout_rate + 1e-8)


class ChannelAttention(nn.Module):
    """
    Soft channel attention mechanism that learns channel importance.
    
    Unlike hard masking, attention provides:
    - Smooth channel weighting
    - Learnable channel relationships
    - Gradient flow to all channels
    - Context-dependent importance
    """
    
    def __init__(
        self,
        num_channels: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        use_temporal_context: bool = True,
    ):
        """
        Initialize channel attention.
        
        Args:
            num_channels: Number of input channels
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            use_temporal_context: Use temporal information for attention
        """
        super().__init__()
        
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_temporal_context = use_temporal_context
        
        # Channel embedding
        self.channel_embed = nn.Parameter(torch.randn(num_channels, hidden_dim))
        
        # Attention computation
        if use_temporal_context:
            # Use temporal pooling + channel embedding
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)
            self.query_proj = nn.Linear(1 + hidden_dim, hidden_dim)
        else:
            # Use only channel embedding
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        channel_importance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute channel attention and apply to input.
        
        Args:
            x: Input tensor [batch, time, channels]
            channel_importance: Optional importance prior [channels]
            
        Returns:
            x_attended: Attended output [batch, time, channels]
            attention_weights: Attention weights [batch, channels]
        """
        batch_size, seq_len, num_channels = x.shape
        
        # Compute attention scores
        if self.use_temporal_context:
            # Pool temporal dimension: [batch, channels, time] -> [batch, channels, 1]
            x_pooled = self.temporal_pool(x.transpose(1, 2)).squeeze(-1)  # [batch, channels]
            
            # Combine with channel embedding
            channel_embed = self.channel_embed.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, channels, hidden]
            combined = torch.cat([x_pooled.unsqueeze(-1), channel_embed], dim=-1)  # [batch, channels, 1+hidden]
            
            queries = self.query_proj(combined)  # [batch, channels, hidden]
        else:
            # Use only channel embedding
            channel_embed = self.channel_embed.unsqueeze(0).expand(batch_size, -1, -1)
            queries = self.query_proj(channel_embed)
        
        # Keys and values from channel embedding
        keys = self.key_proj(channel_embed)
        values = self.value_proj(channel_embed)
        
        # Compute attention scores
        # [batch, channels, hidden] @ [batch, hidden, channels] = [batch, channels, channels]
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))
        attention_scores = attention_scores / (self.hidden_dim ** 0.5)
        
        # Apply importance prior if provided
        if channel_importance is not None:
            # Add importance as bias to attention scores
            importance_bias = channel_importance.log().unsqueeze(0).unsqueeze(1)  # [1, 1, channels]
            attention_scores = attention_scores + importance_bias
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, channels, channels]
        
        # Apply attention to values
        attended = torch.bmm(attention_weights, values)  # [batch, channels, hidden]
        
        # Project to get channel weights
        channel_weights = torch.sigmoid(self.output_proj(attended)).squeeze(-1)  # [batch, channels]
        
        # Apply weights to input
        x_attended = x * channel_weights.unsqueeze(1)  # [batch, time, channels]
        
        return x_attended, channel_weights


class AntiCollapseRegularizer:
    """
    Regularization to prevent zero-solution collapse.
    
    Adds penalty terms that prevent the model from:
    - Outputting all zeros (easy low loss)
    - Ignoring low-energy channels
    - Collapsing to constant predictions
    """
    
    def __init__(
        self,
        entropy_weight: float = 0.01,
        diversity_weight: float = 0.01,
        activity_weight: float = 0.01,
        min_activity_ratio: float = 0.1,
    ):
        """
        Initialize anti-collapse regularizer.
        
        Args:
            entropy_weight: Weight for output entropy term
            diversity_weight: Weight for channel diversity term
            activity_weight: Weight for channel activity term
            min_activity_ratio: Minimum expected activity ratio
        """
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        self.activity_weight = activity_weight
        self.min_activity_ratio = min_activity_ratio
    
    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute anti-collapse regularization loss.
        
        Args:
            outputs: Model outputs [batch, time, channels]
            targets: Optional target values (for activity matching)
            
        Returns:
            loss: Regularization loss
            losses_dict: Dictionary of component losses for logging
        """
        losses_dict = {}
        total_loss = torch.tensor(0.0, device=outputs.device)
        
        # 1. Output entropy: penalize low-entropy (constant) predictions
        if self.entropy_weight > 0:
            # Normalize outputs to [0, 1]
            outputs_norm = torch.sigmoid(outputs)
            
            # Compute entropy per channel: -p*log(p) - (1-p)*log(1-p)
            eps = 1e-8
            entropy = -(outputs_norm * torch.log(outputs_norm + eps) + 
                       (1 - outputs_norm) * torch.log(1 - outputs_norm + eps))
            
            # Penalize low entropy (encourage diversity)
            # Maximum entropy = log(2) ≈ 0.693
            max_entropy = 0.693
            entropy_loss = (max_entropy - entropy.mean()).clamp(min=0.0)
            
            total_loss = total_loss + self.entropy_weight * entropy_loss
            losses_dict['entropy'] = entropy_loss.item()
        
        # 2. Channel diversity: penalize channels that are too similar
        if self.diversity_weight > 0:
            # Compute pairwise correlations between channels
            outputs_flat = outputs.reshape(-1, outputs.shape[-1])  # [batch*time, channels]
            
            # Normalize
            outputs_centered = outputs_flat - outputs_flat.mean(dim=0, keepdim=True)
            outputs_std = outputs_centered.std(dim=0, keepdim=True) + 1e-8
            outputs_normed = outputs_centered / outputs_std
            
            # Correlation matrix
            corr = torch.mm(outputs_normed.T, outputs_normed) / outputs_flat.shape[0]
            
            # Penalize high off-diagonal correlations
            # Extract off-diagonal elements
            mask = ~torch.eye(corr.shape[0], dtype=torch.bool, device=corr.device)
            off_diag = corr[mask]
            
            # Penalize high absolute correlations
            diversity_loss = off_diag.abs().mean()
            
            total_loss = total_loss + self.diversity_weight * diversity_loss
            losses_dict['diversity'] = diversity_loss.item()
        
        # 3. Channel activity: penalize channels with too many near-zero values
        if self.activity_weight > 0:
            # Compute activity ratio (proportion of non-near-zero values)
            threshold = outputs.abs().mean() * 0.01
            activity_ratio = (outputs.abs() > threshold).float().mean(dim=(0, 1))  # Per channel
            
            # Penalize channels with low activity
            activity_loss = F.relu(self.min_activity_ratio - activity_ratio).mean()
            
            total_loss = total_loss + self.activity_weight * activity_loss
            losses_dict['activity'] = activity_loss.item()
        
        return total_loss, losses_dict


class EnhancedEEGHandler(nn.Module):
    """
    Complete enhanced EEG handling system.
    
    Combines all components:
    - Channel activity monitoring
    - Adaptive dropout
    - Channel attention
    - Anti-collapse regularization
    """
    
    def __init__(
        self,
        num_channels: int,
        enable_monitoring: bool = True,
        enable_dropout: bool = True,
        enable_attention: bool = True,
        enable_regularization: bool = True,
        # Monitoring params
        monitor_window_size: int = 100,
        monitor_update_freq: int = 10,
        # Dropout params
        dropout_rate: float = 0.1,
        # Attention params
        attention_hidden_dim: int = 64,
        attention_heads: int = 4,
        # Regularization params
        entropy_weight: float = 0.01,
        diversity_weight: float = 0.01,
        activity_weight: float = 0.01,
    ):
        """
        Initialize enhanced EEG handler.
        
        Args:
            num_channels: Number of EEG channels
            enable_monitoring: Enable channel monitoring
            enable_dropout: Enable adaptive dropout
            enable_attention: Enable channel attention
            enable_regularization: Enable anti-collapse regularization
            Other params: See individual component docs
        """
        super().__init__()
        
        self.num_channels = num_channels
        self.enable_monitoring = enable_monitoring
        self.enable_dropout = enable_dropout
        self.enable_attention = enable_attention
        self.enable_regularization = enable_regularization
        
        # Initialize components
        if enable_monitoring:
            self.monitor = ChannelActivityMonitor(
                num_channels=num_channels,
                window_size=monitor_window_size,
                update_frequency=monitor_update_freq,
            )
        else:
            self.monitor = None
        
        if enable_dropout:
            self.dropout = AdaptiveChannelDropout(dropout_rate=dropout_rate)
        else:
            self.dropout = None
        
        if enable_attention:
            self.attention = ChannelAttention(
                num_channels=num_channels,
                hidden_dim=attention_hidden_dim,
                num_heads=attention_heads,
            )
        else:
            self.attention = None
        
        if enable_regularization:
            self.regularizer = AntiCollapseRegularizer(
                entropy_weight=entropy_weight,
                diversity_weight=diversity_weight,
                activity_weight=activity_weight,
            )
        else:
            self.regularizer = None
    
    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
        compute_regularization: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        Process EEG signals.
        
        Args:
            x: Input EEG signals [batch, time, channels]
            training: Whether in training mode
            compute_regularization: Whether to compute regularization loss
            
        Returns:
            x_processed: Processed signals
            info_dict: Dictionary with processing info and losses
        """
        info_dict = {}
        
        # Get channel importance from monitor
        channel_importance = None
        if self.monitor is not None and training:
            self.monitor.update(x)
            channel_importance = self.monitor.get_importance_weights().to(x.device)
            info_dict['channel_health'] = self.monitor.channel_health.mean().item()
        
        # Apply adaptive dropout
        if self.dropout is not None and training:
            x = self.dropout(x, channel_importance, training=training)
        
        # Apply channel attention
        attention_weights = None
        if self.attention is not None:
            x, attention_weights = self.attention(x, channel_importance)
            info_dict['attention_weights'] = attention_weights.detach()
        
        # Compute regularization loss
        reg_loss = None
        if self.regularizer is not None and compute_regularization and training:
            reg_loss, reg_losses = self.regularizer.compute_loss(x)
            info_dict['regularization_loss'] = reg_loss
            info_dict.update({f'reg_{k}': v for k, v in reg_losses.items()})
        
        return x, info_dict
    
    def update_gradients(self, gradients: torch.Tensor):
        """Update monitor with gradient information."""
        if self.monitor is not None:
            self.monitor.update(gradients, gradients=gradients)
    
    def log_status(self):
        """Log current channel status."""
        if self.monitor is not None:
            self.monitor.log_channel_status()
