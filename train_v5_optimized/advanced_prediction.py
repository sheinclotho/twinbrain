"""
Advanced Multi-Step Prediction
===============================

Implements enhanced multi-step prediction strategies:
1. Hierarchical multi-scale prediction (coarse-to-fine)
2. Transformer-based prediction with long-range attention
3. Stratified window sampling (beginning, middle, end)
4. Uncertainty-aware prediction
5. Autoregressive with teacher forcing schedule

Key Improvements over baseline:
- Better long-range temporal modeling
- Memory-efficient sampling strategy
- Multi-scale temporal hierarchies
- Confidence estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [batch, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor for long-range temporal modeling.
    
    Advantages over GRU:
    - Parallel processing
    - Better long-range dependencies
    - Multi-head attention for different temporal scales
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        use_gradient_checkpointing: bool = True,
    ):
        """
        Initialize transformer predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
            use_gradient_checkpointing: Use gradient checkpointing for memory
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-normalization for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, input_dim]
            mask: Attention mask [seq_len, seq_len]
            
        Returns:
            Output [batch, seq_len, input_dim]
        """
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            x = torch.utils.checkpoint.checkpoint(
                self.transformer,
                x,
                mask,
                use_reentrant=False
            )
        else:
            x = self.transformer(x, mask=mask)
        
        # Normalize and project output
        x = self.norm(x)
        x = self.output_proj(x)
        
        return x


class HierarchicalPredictor(nn.Module):
    """
    Hierarchical multi-scale prediction.
    
    Predicts at multiple temporal scales:
    - Coarse: long-term trends (low frequency)
    - Medium: intermediate dynamics
    - Fine: short-term fluctuations (high frequency)
    
    Combines predictions in coarse-to-fine manner.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_scales: int = 3,
        scale_factors: Optional[List[int]] = None,
        predictor_type: str = 'transformer',  # 'transformer' or 'gru'
        dropout: float = 0.1,
    ):
        """
        Initialize hierarchical predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for each scale
            num_scales: Number of temporal scales
            scale_factors: Downsampling factors for each scale (e.g., [4, 2, 1])
            predictor_type: Type of predictor ('transformer' or 'gru')
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        if scale_factors is None:
            # Default: exponentially decreasing scales
            scale_factors = [2 ** (num_scales - 1 - i) for i in range(num_scales)]
        self.scale_factors = scale_factors
        
        # Create predictor for each scale
        self.predictors = nn.ModuleList()
        for scale in range(num_scales):
            if predictor_type == 'transformer':
                predictor = TransformerPredictor(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_heads=4,
                    num_layers=2,
                    dropout=dropout,
                )
            elif predictor_type == 'gru':
                predictor = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=2,
                    dropout=dropout,
                    batch_first=True,
                )
            else:
                raise ValueError(f"Unknown predictor type: {predictor_type}")
            
            self.predictors.append(predictor)
        
        # Upsampling layers to reconstruct fine scale
        self.upsamplers = nn.ModuleList()
        for i, scale_factor in enumerate(scale_factors[:-1]):  # Skip finest scale
            upsampler = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=input_dim,
                    out_channels=input_dim,
                    kernel_size=scale_factor * 2,
                    stride=scale_factor,
                    padding=scale_factor // 2,
                ),
                nn.LayerNorm(input_dim),
                nn.GELU(),
            )
            self.upsamplers.append(upsampler)
        
        # Fusion layer to combine multi-scale predictions
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * num_scales, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        future_steps: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Hierarchical prediction.
        
        Args:
            x: Context [batch, context_len, input_dim]
            future_steps: Number of future steps to predict
            
        Returns:
            prediction: Final prediction [batch, future_steps, input_dim]
            scale_predictions: Predictions at each scale
        """
        batch_size = x.shape[0]
        
        scale_predictions = []
        
        # Predict at each scale
        for i, (predictor, scale_factor) in enumerate(zip(self.predictors, self.scale_factors)):
            # Downsample context
            if scale_factor > 1:
                # Average pooling to downsample
                x_down = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=scale_factor,
                    stride=scale_factor,
                ).transpose(1, 2)
            else:
                x_down = x
            
            # Predict at this scale
            if isinstance(predictor, TransformerPredictor):
                # For transformer, we need to extend sequence with future slots
                future_steps_scaled = future_steps // scale_factor
                
                # Create future slots (initialized with last context value)
                future_init = x_down[:, -1:, :].repeat(1, future_steps_scaled, 1)
                x_extended = torch.cat([x_down, future_init], dim=1)
                
                # Create causal mask
                seq_len = x_extended.shape[1]
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
                    diagonal=1
                )
                
                # Predict
                pred_full = predictor(x_extended, mask=causal_mask)
                pred = pred_full[:, -future_steps_scaled:, :]
            else:
                # For GRU, predict autoregressively
                pred = self._autoregressive_predict(
                    predictor, x_down, future_steps // scale_factor
                )
            
            # Upsample to finest resolution if needed
            if i < len(self.upsamplers):
                # Transpose for Conv1d: [batch, channels, time]
                pred_up = self.upsamplers[i](pred.transpose(1, 2)).transpose(1, 2)
                
                # Adjust length if needed
                if pred_up.shape[1] != future_steps:
                    pred_up = F.interpolate(
                        pred_up.transpose(1, 2),
                        size=future_steps,
                        mode='linear',
                        align_corners=False,
                    ).transpose(1, 2)
            else:
                pred_up = pred
            
            scale_predictions.append(pred_up)
        
        # Fuse multi-scale predictions
        # Stack along feature dimension
        multi_scale = torch.cat(scale_predictions, dim=-1)  # [batch, future_steps, input_dim * num_scales]
        
        # Fusion
        final_prediction = self.fusion(multi_scale)
        
        return final_prediction, scale_predictions
    
    def _autoregressive_predict(
        self,
        predictor: nn.GRU,
        context: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Autoregressive prediction with GRU."""
        batch_size = context.shape[0]
        
        # Get initial hidden state
        _, hidden = predictor(context)
        
        # Start with last context value
        current = context[:, -1:, :]
        
        predictions = []
        for _ in range(num_steps):
            # Predict next step
            output, hidden = predictor(current, hidden)
            predictions.append(output)
            
            # Use prediction as next input
            current = output
        
        # Concatenate predictions
        return torch.cat(predictions, dim=1)


class StratifiedWindowSampler:
    """
    Stratified window sampling for memory-efficient training.
    
    Instead of using only beginning and end windows (baseline),
    samples windows from:
    - Beginning (early dynamics)
    - Middle (intermediate patterns)
    - End (recent context)
    
    This provides better coverage of temporal patterns.
    """
    
    def __init__(
        self,
        context_length: int = 50,
        prediction_steps: int = 10,
        num_windows: int = 3,
        sampling_strategy: str = 'uniform',  # 'uniform', 'random', or 'adaptive'
    ):
        """
        Initialize stratified window sampler.
        
        Args:
            context_length: Length of context window
            prediction_steps: Number of steps to predict
            num_windows: Number of windows to sample
            sampling_strategy: How to sample windows
        """
        self.context_length = context_length
        self.prediction_steps = prediction_steps
        self.num_windows = num_windows
        self.sampling_strategy = sampling_strategy
        
        self.window_length = context_length + prediction_steps
    
    def sample_windows(
        self,
        sequence: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
        """
        Sample windows from sequence.
        
        Args:
            sequence: Full sequence [batch, time, features]
            importance_weights: Optional importance weights for adaptive sampling
            
        Returns:
            windows: List of (context, target, start_idx) tuples
        """
        batch_size, seq_len, _ = sequence.shape
        
        # Check if sequence is long enough
        if seq_len < self.window_length:
            # Use full sequence
            context = sequence
            target = sequence[:, -self.prediction_steps:, :]
            return [(context, target, 0)]
        
        # Compute possible start indices
        max_start = seq_len - self.window_length
        
        if self.sampling_strategy == 'uniform':
            # Uniformly spaced windows
            if self.num_windows == 1:
                start_indices = [max_start // 2]
            else:
                start_indices = [
                    int(i * max_start / (self.num_windows - 1))
                    for i in range(self.num_windows)
                ]
        
        elif self.sampling_strategy == 'random':
            # Random sampling
            start_indices = torch.randint(0, max_start + 1, (self.num_windows,)).tolist()
        
        elif self.sampling_strategy == 'adaptive':
            # Adaptive sampling based on importance weights
            if importance_weights is None:
                # Fall back to uniform
                start_indices = [
                    int(i * max_start / (self.num_windows - 1))
                    for i in range(self.num_windows)
                ]
            else:
                # Sample based on importance
                # importance_weights: [time]
                # Compute importance for each possible window
                window_importance = []
                for start in range(max_start + 1):
                    end = start + self.window_length
                    importance = importance_weights[start:end].mean().item()
                    window_importance.append(importance)
                
                # Sample windows proportional to importance
                window_importance = torch.tensor(window_importance)
                probs = F.softmax(window_importance, dim=0)
                start_indices = torch.multinomial(probs, self.num_windows, replacement=False).tolist()
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Extract windows
        windows = []
        for start_idx in start_indices:
            end_idx = start_idx + self.window_length
            window = sequence[:, start_idx:end_idx, :]
            
            # Split into context and target
            context = window[:, :self.context_length, :]
            target = window[:, self.context_length:, :]
            
            windows.append((context, target, start_idx))
        
        return windows


class UncertaintyAwarePredictor(nn.Module):
    """
    Uncertainty-aware prediction with confidence estimation.
    
    Estimates both:
    - Prediction mean (expected value)
    - Prediction uncertainty (confidence)
    
    Uses uncertainty for:
    - Loss weighting (downweight uncertain predictions)
    - Confidence-based ensembling
    - Uncertainty quantification for downstream tasks
    """
    
    def __init__(
        self,
        base_predictor: nn.Module,
        input_dim: int,
        uncertainty_method: str = 'gaussian',  # 'gaussian' or 'dropout'
        num_mc_samples: int = 10,
    ):
        """
        Initialize uncertainty-aware predictor.
        
        Args:
            base_predictor: Base prediction model
            input_dim: Input dimension
            uncertainty_method: Method for uncertainty estimation
            num_mc_samples: Number of MC samples for dropout-based uncertainty
        """
        super().__init__()
        
        self.base_predictor = base_predictor
        self.input_dim = input_dim
        self.uncertainty_method = uncertainty_method
        self.num_mc_samples = num_mc_samples
        
        if uncertainty_method == 'gaussian':
            # Predict mean and log variance
            self.uncertainty_head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, input_dim),
            )
    
    def forward(
        self,
        x: torch.Tensor,
        future_steps: Optional[int] = None,
        return_uncertainty: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x: Context [batch, context_len, input_dim]
            future_steps: Number of future steps to predict
            return_uncertainty: Whether to return uncertainty
            
        Returns:
            prediction_mean: Mean prediction [batch, future_steps, input_dim]
            prediction_std: Uncertainty (None if not requested)
        """
        if self.uncertainty_method == 'gaussian':
            # Forward through base predictor
            if isinstance(self.base_predictor, HierarchicalPredictor):
                prediction_mean, _ = self.base_predictor(x, future_steps)
            else:
                prediction_mean = self.base_predictor(x)
            
            if not return_uncertainty:
                return prediction_mean, None
            
            # Predict log variance
            log_var = self.uncertainty_head(prediction_mean)
            prediction_std = torch.exp(0.5 * log_var)
            
            return prediction_mean, prediction_std
        
        elif self.uncertainty_method == 'dropout':
            # MC dropout for uncertainty estimation
            if not return_uncertainty:
                # Single forward pass
                if isinstance(self.base_predictor, HierarchicalPredictor):
                    prediction, _ = self.base_predictor(x, future_steps)
                else:
                    prediction = self.base_predictor(x)
                return prediction, None
            
            # Multiple forward passes with dropout
            self.base_predictor.train()  # Enable dropout
            
            predictions = []
            for _ in range(self.num_mc_samples):
                if isinstance(self.base_predictor, HierarchicalPredictor):
                    pred, _ = self.base_predictor(x, future_steps)
                else:
                    pred = self.base_predictor(x)
                predictions.append(pred)
            
            # Compute mean and std
            predictions = torch.stack(predictions, dim=0)  # [num_samples, batch, time, dim]
            prediction_mean = predictions.mean(dim=0)
            prediction_std = predictions.std(dim=0)
            
            return prediction_mean, prediction_std
        
        else:
            raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute uncertainty-aware loss.
        
        Args:
            predictions: Predicted values [batch, time, dim]
            targets: Target values [batch, time, dim]
            uncertainty: Predicted uncertainty (std) [batch, time, dim]
            
        Returns:
            loss: Uncertainty-weighted loss
        """
        if uncertainty is None or self.uncertainty_method == 'dropout':
            # Standard MSE loss
            return F.mse_loss(predictions, targets)
        
        # Gaussian negative log likelihood
        # Loss = 0.5 * (log(2π) + log(σ²) + (y - μ)² / σ²)
        # Simplified: 0.5 * (log(σ²) + (y - μ)² / σ²)
        
        squared_error = (predictions - targets) ** 2
        log_var = 2.0 * torch.log(uncertainty + 1e-8)
        
        # NLL loss
        nll_loss = 0.5 * (log_var + squared_error / (uncertainty ** 2 + 1e-8))
        
        return nll_loss.mean()


class EnhancedMultiStepPredictor(nn.Module):
    """
    Complete enhanced multi-step prediction system.
    
    Combines:
    - Hierarchical multi-scale prediction
    - Transformer-based modeling
    - Stratified window sampling
    - Uncertainty estimation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        context_length: int = 50,
        prediction_steps: int = 10,
        use_hierarchical: bool = True,
        use_transformer: bool = True,
        use_uncertainty: bool = True,
        num_scales: int = 3,
        num_windows: int = 3,
        sampling_strategy: str = 'uniform',
        dropout: float = 0.1,
    ):
        """
        Initialize enhanced multi-step predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            context_length: Length of context window
            prediction_steps: Number of steps to predict
            use_hierarchical: Use hierarchical multi-scale prediction
            use_transformer: Use transformer (else GRU)
            use_uncertainty: Enable uncertainty estimation
            num_scales: Number of scales for hierarchical prediction
            num_windows: Number of windows for stratified sampling
            sampling_strategy: Window sampling strategy
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.prediction_steps = prediction_steps
        self.use_hierarchical = use_hierarchical
        self.use_uncertainty = use_uncertainty
        
        # Create base predictor
        if use_hierarchical:
            predictor_type = 'transformer' if use_transformer else 'gru'
            base_predictor = HierarchicalPredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_scales=num_scales,
                predictor_type=predictor_type,
                dropout=dropout,
            )
        elif use_transformer:
            base_predictor = TransformerPredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=8,
                num_layers=4,
                dropout=dropout,
            )
        else:
            # Simple GRU baseline
            base_predictor = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=3,
                dropout=dropout,
                batch_first=True,
            )
        
        # Wrap with uncertainty estimation if enabled
        if use_uncertainty:
            self.predictor = UncertaintyAwarePredictor(
                base_predictor=base_predictor,
                input_dim=input_dim,
                uncertainty_method='gaussian',
            )
        else:
            self.predictor = base_predictor
        
        # Window sampler
        self.window_sampler = StratifiedWindowSampler(
            context_length=context_length,
            prediction_steps=prediction_steps,
            num_windows=num_windows,
            sampling_strategy=sampling_strategy,
        )
    
    def forward(
        self,
        sequences: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with stratified sampling.
        
        Args:
            sequences: Full sequences [batch, time, input_dim]
            importance_weights: Optional importance for adaptive sampling [time]
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            predictions: Predictions for all sampled windows
            targets: Targets for all sampled windows
            uncertainties: Uncertainty estimates (None if not requested)
        """
        # Sample windows
        windows = self.window_sampler.sample_windows(sequences, importance_weights)
        
        # Predict for each window
        all_predictions = []
        all_targets = []
        all_uncertainties = [] if return_uncertainty else None
        
        for context, target, _ in windows:
            # Predict
            if self.use_uncertainty:
                pred_mean, pred_std = self.predictor(
                    context,
                    future_steps=self.prediction_steps,
                    return_uncertainty=return_uncertainty,
                )
            elif self.use_hierarchical:
                pred_mean, _ = self.predictor(context, self.prediction_steps)
                pred_std = None
            else:
                # Simple GRU
                _, hidden = self.predictor(context)
                
                # Autoregressive prediction
                current = context[:, -1:, :]
                predictions = []
                for _ in range(self.prediction_steps):
                    output, hidden = self.predictor(current, hidden)
                    predictions.append(output)
                    current = output
                pred_mean = torch.cat(predictions, dim=1)
                pred_std = None
            
            all_predictions.append(pred_mean)
            all_targets.append(target)
            if pred_std is not None:
                all_uncertainties.append(pred_std)
        
        # Stack predictions
        predictions = torch.stack(all_predictions, dim=0)  # [num_windows, batch, time, dim]
        targets = torch.stack(all_targets, dim=0)
        uncertainties = torch.stack(all_uncertainties, dim=0) if all_uncertainties else None
        
        return predictions, targets, uncertainties
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute prediction loss.
        
        Args:
            predictions: Predictions [num_windows, batch, time, dim]
            targets: Targets [num_windows, batch, time, dim]
            uncertainties: Uncertainty estimates (optional)
            
        Returns:
            loss: Mean loss across all windows
        """
        if self.use_uncertainty and uncertainties is not None:
            # Uncertainty-aware loss for each window
            losses = []
            for i in range(predictions.shape[0]):
                loss = self.predictor.compute_loss(
                    predictions[i],
                    targets[i],
                    uncertainties[i] if uncertainties is not None else None,
                )
                losses.append(loss)
            return torch.stack(losses).mean()
        else:
            # Standard MSE loss
            return F.mse_loss(predictions, targets)
