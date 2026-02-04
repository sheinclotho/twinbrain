"""
Predictor modules for future state prediction in TwinBrain.
Implements multi-step prediction with attention mechanisms.
"""

import random
import numpy as np

# Initialize random seeds before torch import to prevent THPGenerator errors
_INIT_SEED = 42
random.seed(_INIT_SEED)
np.random.seed(_INIT_SEED)

import torch
# MUST call manual_seed immediately after torch import
torch.manual_seed(_INIT_SEED)

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple


class PredictorHead(nn.Module):
    """
    Multi-step future state predictor with attention mechanism.
    
    Predicts future latent states from historical sequence using:
    - GRU for temporal modeling
    - Multi-head attention for focusing on relevant history
    - Residual connections for stability
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_future_steps: int = 10,
        context_length: Optional[int] = None,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_gradient_checkpointing: bool = True
    ):
        """
        Initialize predictor head.
        
        Args:
            hidden_dim: Dimension of hidden/latent states
            n_future_steps: Number of future steps to predict
            context_length: Number of historical steps to use as context.
                          If None, uses all available steps. Default: None
            num_layers: Number of GRU layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_residual: Whether to use residual connections
            use_gradient_checkpointing: Whether to use gradient checkpointing to save memory.
                                       Default: True (recommended for large models)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_future_steps = n_future_steps
        self.context_length = context_length
        self.use_residual = use_residual
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Temporal prediction network
        self.predictor_gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Temporal attention: focus on relevant historical timesteps
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection with optional residual
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        latent_seq: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict future states from historical sequence.
        
        Args:
            latent_seq: [B, T_past, H] - Historical latent sequence
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: [B, T_future, H] - Predicted future sequence
            attention_weights: [B, T_future, T_past] if return_attention=True
            
        Note:
            If context_length is set, only the last context_length steps are used.
            For example, if context_length=50 and latent_seq has 100 steps,
            only the last 50 steps will be used for prediction.
        """
        batch_size = latent_seq.shape[0]
        
        # Use only the last context_length steps if specified
        if self.context_length is not None and latent_seq.shape[1] > self.context_length:
            context_seq = latent_seq[:, -self.context_length:, :]
        else:
            context_seq = latent_seq
        
        # Initialize with context sequence
        # Use gradient checkpointing for memory efficiency during training
        if self.use_gradient_checkpointing and self.training:
            # Wrap GRU call in checkpoint to reduce memory usage
            # Pass None explicitly as hidden state for clarity
            _, hidden = checkpoint(self.predictor_gru, context_seq, None, use_reentrant=False)
        else:
            _, hidden = self.predictor_gru(context_seq)
        
        # Auto-regressive prediction
        predictions = []
        attention_weights_list = [] if return_attention else None
        
        # Start with last state from context
        current = context_seq[:, -1:, :]  # [B, 1, H]
        
        for t in range(self.n_future_steps):
            # Predict next step with GRU
            # Use gradient checkpointing for memory efficiency
            if self.use_gradient_checkpointing and self.training:
                pred, hidden = checkpoint(
                    lambda c, h: self.predictor_gru(c, h),
                    current, hidden,
                    use_reentrant=False
                )
            else:
                pred, hidden = self.predictor_gru(current, hidden)  # [B, 1, H]
            
            # Apply attention to historical context
            attended, attn_weights = self.temporal_attention(
                pred,  # query: [B, 1, H]
                context_seq,  # key: [B, T_context, H]
                context_seq,  # value: [B, T_context, H]
            )
            
            # Combine with residual connection
            if self.use_residual:
                combined = pred + attended
            else:
                combined = attended
            
            # Project to output space
            output = self.output_proj(combined)
            
            # Layer normalization
            output = self.layer_norm(output)
            
            predictions.append(output)
            
            if return_attention:
                attention_weights_list.append(attn_weights)
            
            # Use prediction as input for next step
            current = output
        
        # Concatenate predictions
        predictions = torch.cat(predictions, dim=1)  # [B, T_future, H]
        
        if return_attention:
            attention_weights = torch.stack(attention_weights_list, dim=1)  # [B, T_future, num_heads, 1, T_past]
            # Average over heads and squeeze
            attention_weights = attention_weights.mean(dim=2).squeeze(3)  # [B, T_future, T_past]
            return predictions, attention_weights
        
        return predictions, None
    
    def predict_single_step(self, latent_seq: torch.Tensor) -> torch.Tensor:
        """
        Predict only the next single step (faster than multi-step).
        
        Args:
            latent_seq: [B, T_past, H] - Historical sequence
            
        Returns:
            prediction: [B, 1, H] - Next step prediction
            
        Note:
            If context_length is set, only the last context_length steps are used.
        """
        # Use only the last context_length steps if specified
        if self.context_length is not None and latent_seq.shape[1] > self.context_length:
            context_seq = latent_seq[:, -self.context_length:, :]
        else:
            context_seq = latent_seq
        
        # Use GRU to get next hidden state
        _, hidden = self.predictor_gru(context_seq)
        
        # Get last state from context
        current = context_seq[:, -1:, :]
        
        # Predict one step
        pred, _ = self.predictor_gru(current, hidden)
        
        # Apply attention to context
        attended, _ = self.temporal_attention(pred, context_seq, context_seq)
        
        # Combine and project
        if self.use_residual:
            combined = pred + attended
        else:
            combined = attended
        
        output = self.output_proj(combined)
        output = self.layer_norm(output)
        
        return output


class ConditionalPredictor(nn.Module):
    """
    Conditional predictor that incorporates external stimulation/conditions.
    
    Useful for predicting brain response to stimulation.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int,
        n_future_steps: int = 10,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize conditional predictor.
        
        Args:
            hidden_dim: Dimension of hidden states
            condition_dim: Dimension of condition/stimulation input
            n_future_steps: Number of future steps to predict
            num_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_future_steps = n_future_steps
        
        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Conditional GRU (takes concatenated [state, condition])
        self.conditional_gru = nn.GRU(
            hidden_dim * 2,  # state + condition
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        current_state: torch.Tensor,
        condition_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict future states conditioned on external input.
        
        Args:
            current_state: [B, H] - Current brain state
            condition_sequence: [B, T_future, C] - Future condition/stimulation
            
        Returns:
            predictions: [B, T_future, H] - Predicted states
        """
        batch_size = current_state.shape[0]
        
        # Encode condition sequence
        # Reshape for batch processing
        T = condition_sequence.shape[1]
        condition_flat = condition_sequence.reshape(-1, condition_sequence.shape[-1])
        condition_embed = self.condition_encoder(condition_flat)
        condition_embed = condition_embed.reshape(batch_size, T, -1)
        
        # Initialize hidden state
        hidden = None
        predictions = []
        
        # Current state for auto-regression
        state = current_state.unsqueeze(1)  # [B, 1, H]
        
        # Predict each future step with corresponding condition
        for t in range(self.n_future_steps):
            # Get condition for this timestep
            cond_t = condition_embed[:, t:t+1, :]  # [B, 1, H]
            
            # Concatenate state and condition
            conditioned_input = torch.cat([state, cond_t], dim=-1)  # [B, 1, 2H]
            
            # Predict
            pred, hidden = self.conditional_gru(conditioned_input, hidden)
            
            # Project and normalize
            output = self.output_proj(pred)
            output = self.layer_norm(output)
            
            predictions.append(output)
            
            # Use prediction for next step
            state = output
        
        return torch.cat(predictions, dim=1)  # [B, T_future, H]


def prediction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = 'mse'
) -> torch.Tensor:
    """
    Compute prediction loss.
    
    Args:
        predictions: [B, T, H] - Predicted states
        targets: [B, T, H] - Ground truth future states
        loss_type: Type of loss ('mse', 'mae', or 'huber')
        
    Returns:
        loss: Scalar loss value
    """
    if loss_type == 'mse':
        return F.mse_loss(predictions, targets)
    elif loss_type == 'mae':
        return F.l1_loss(predictions, targets)
    elif loss_type == 'huber':
        return F.smooth_l1_loss(predictions, targets)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
