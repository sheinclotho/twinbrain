"""
Predictor modules V2 for future state prediction in TwinBrain.
Implements multi-step prediction with attention mechanisms and cross-modal bidirectional prediction.

This version adds cross-modal prediction capabilities:
- fMRI -> EEG prediction
- EEG -> fMRI prediction
- Bidirectional cross-modal prediction training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


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
        use_residual: bool = True
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
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_future_steps = n_future_steps
        self.context_length = context_length
        self.use_residual = use_residual
        
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
        _, hidden = self.predictor_gru(context_seq)
        
        # Auto-regressive prediction
        predictions = []
        attention_weights_list = [] if return_attention else None
        
        # Start with last state from context
        current = context_seq[:, -1:, :]  # [B, 1, H]
        
        for t in range(self.n_future_steps):
            # Predict next step with GRU
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


class CrossModalPredictor(nn.Module):
    """
    Cross-modal predictor for bidirectional prediction between modalities.
    
    Supports:
    - fMRI -> EEG prediction
    - EEG -> fMRI prediction
    
    Uses cross-attention mechanism to capture cross-modal dependencies.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_future_steps: int = 10,
        context_length: Optional[int] = None,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_bridge: bool = True
    ):
        """
        Initialize cross-modal predictor.
        
        Args:
            hidden_dim: Dimension of hidden states
            n_future_steps: Number of future steps to predict
            context_length: Number of historical steps to use as context
            num_layers: Number of GRU layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_bridge: Whether to use a bridge network for modality translation
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_future_steps = n_future_steps
        self.context_length = context_length
        self.use_bridge = use_bridge
        
        # Bridge networks for modality translation (optional)
        if use_bridge:
            self.fmri_to_eeg_bridge = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            self.eeg_to_fmri_bridge = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Cross-modal attention: source modality attends to target modality
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Temporal GRU for sequential prediction
        self.temporal_gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        source_seq: torch.Tensor,
        target_seq: torch.Tensor,
        source_to_target: str = "fmri_to_eeg",
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict future states of target modality from source modality.
        
        Args:
            source_seq: [B, T_source, H] - Source modality historical sequence
            target_seq: [B, T_target, H] - Target modality historical sequence (for conditioning)
            source_to_target: Direction of prediction ('fmri_to_eeg' or 'eeg_to_fmri')
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: [B, T_future, H] - Predicted target modality states
            attention_weights: Optional attention weights
        """
        batch_size = source_seq.shape[0]
        
        # Use only last context_length steps if specified
        if self.context_length is not None:
            if source_seq.shape[1] > self.context_length:
                source_context = source_seq[:, -self.context_length:, :]
            else:
                source_context = source_seq
                
            if target_seq.shape[1] > self.context_length:
                target_context = target_seq[:, -self.context_length:, :]
            else:
                target_context = target_seq
        else:
            source_context = source_seq
            target_context = target_seq
        
        # Apply bridge network if enabled
        if self.use_bridge:
            if source_to_target == "fmri_to_eeg":
                source_context = self.fmri_to_eeg_bridge(source_context)
            elif source_to_target == "eeg_to_fmri":
                source_context = self.eeg_to_fmri_bridge(source_context)
        
        # Initialize GRU with target context
        _, hidden = self.temporal_gru(target_context)
        
        # Auto-regressive prediction with cross-modal attention
        predictions = []
        attention_weights_list = [] if return_attention else None
        
        # Start from last state of target modality
        current = target_context[:, -1:, :]  # [B, 1, H]
        
        for t in range(self.n_future_steps):
            # Predict next step
            pred, hidden = self.temporal_gru(current, hidden)  # [B, 1, H]
            
            # Cross-modal attention: attend to source modality
            attended, attn_weights = self.cross_attention(
                pred,  # query from prediction: [B, 1, H]
                source_context,  # key from source: [B, T_source, H]
                source_context,  # value from source: [B, T_source, H]
            )
            
            # Combine prediction with cross-modal information
            combined = pred + attended
            
            # Project to output
            output = self.output_proj(combined)
            output = self.layer_norm(output)
            
            predictions.append(output)
            
            if return_attention:
                attention_weights_list.append(attn_weights)
            
            # Use prediction for next step
            current = output
        
        # Concatenate predictions
        predictions = torch.cat(predictions, dim=1)  # [B, T_future, H]
        
        if return_attention:
            attention_weights = torch.stack(attention_weights_list, dim=1)
            attention_weights = attention_weights.mean(dim=2).squeeze(3)  # [B, T_future, T_source]
            return predictions, attention_weights
        
        return predictions, None


class BidirectionalCrossModalPredictor(nn.Module):
    """
    Bidirectional cross-modal predictor that handles both directions.
    
    Combines two CrossModalPredictors for:
    - fMRI -> EEG prediction
    - EEG -> fMRI prediction
    """
    
    def __init__(
        self,
        hidden_dim: int,
        n_future_steps: int = 10,
        context_length: Optional[int] = None,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_bridge: bool = True,
        share_attention: bool = False
    ):
        """
        Initialize bidirectional cross-modal predictor.
        
        Args:
            hidden_dim: Dimension of hidden states
            n_future_steps: Number of future steps to predict
            context_length: Number of historical steps to use
            num_layers: Number of GRU layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_bridge: Whether to use bridge networks
            share_attention: Whether to share attention weights between directions
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.share_attention = share_attention
        
        if share_attention:
            # Use single shared predictor
            self.shared_predictor = CrossModalPredictor(
                hidden_dim=hidden_dim,
                n_future_steps=n_future_steps,
                context_length=context_length,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                use_bridge=use_bridge
            )
        else:
            # Separate predictors for each direction
            self.fmri_to_eeg = CrossModalPredictor(
                hidden_dim=hidden_dim,
                n_future_steps=n_future_steps,
                context_length=context_length,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                use_bridge=use_bridge
            )
            
            self.eeg_to_fmri = CrossModalPredictor(
                hidden_dim=hidden_dim,
                n_future_steps=n_future_steps,
                context_length=context_length,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                use_bridge=use_bridge
            )
    
    def forward(
        self,
        fmri_seq: torch.Tensor,
        eeg_seq: torch.Tensor,
        direction: str = "both",
        return_attention: bool = False
    ) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Predict cross-modal future states.
        
        Args:
            fmri_seq: [B, T, H] - fMRI sequence
            eeg_seq: [B, T, H] - EEG sequence
            direction: Prediction direction ('fmri_to_eeg', 'eeg_to_fmri', or 'both')
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with predictions for requested directions:
            {
                'fmri_to_eeg': (predictions, attention_weights),
                'eeg_to_fmri': (predictions, attention_weights)
            }
        """
        results = {}
        
        if direction in ["fmri_to_eeg", "both"]:
            if self.share_attention:
                pred, attn = self.shared_predictor(
                    fmri_seq, eeg_seq, 
                    source_to_target="fmri_to_eeg",
                    return_attention=return_attention
                )
            else:
                pred, attn = self.fmri_to_eeg(
                    fmri_seq, eeg_seq,
                    source_to_target="fmri_to_eeg",
                    return_attention=return_attention
                )
            results['fmri_to_eeg'] = (pred, attn)
        
        if direction in ["eeg_to_fmri", "both"]:
            if self.share_attention:
                pred, attn = self.shared_predictor(
                    eeg_seq, fmri_seq,
                    source_to_target="eeg_to_fmri",
                    return_attention=return_attention
                )
            else:
                pred, attn = self.eeg_to_fmri(
                    eeg_seq, fmri_seq,
                    source_to_target="eeg_to_fmri",
                    return_attention=return_attention
                )
            results['eeg_to_fmri'] = (pred, attn)
        
        return results


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


def cross_modal_prediction_loss(
    predictions_dict: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]],
    targets_dict: Dict[str, torch.Tensor],
    loss_type: str = 'mse',
    weights: Optional[Dict[str, float]] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute cross-modal prediction loss for both directions.
    
    Args:
        predictions_dict: Dictionary with predictions for each direction
        targets_dict: Dictionary with target sequences for each direction
        loss_type: Type of loss ('mse', 'mae', or 'huber')
        weights: Optional weights for each direction
        
    Returns:
        total_loss: Weighted total loss
        losses_dict: Individual losses for each direction
    """
    if weights is None:
        weights = {'fmri_to_eeg': 1.0, 'eeg_to_fmri': 1.0}
    
    losses = {}
    total_loss = 0.0
    
    for direction, (pred, _) in predictions_dict.items():
        if direction in targets_dict:
            target = targets_dict[direction]
            loss = prediction_loss(pred, target, loss_type=loss_type)
            losses[direction] = loss
            total_loss += weights.get(direction, 1.0) * loss
    
    return total_loss, losses
