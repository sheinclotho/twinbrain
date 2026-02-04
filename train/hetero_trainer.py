import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import random

# Set CUDA memory allocator configuration to reduce fragmentation
# This must be set before any CUDA operations (i.e., at module import time)
# Uses setdefault to avoid overwriting user-specified configurations
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ============================================================================
# CRITICAL: Initialize random seeds BEFORE importing torch modules
# This prevents THPGenerator_initDefaultGenerator errors during CUDA operations
# ============================================================================
_INIT_SEED = 42
random.seed(_INIT_SEED)
np.random.seed(_INIT_SEED)

# Now import torch - but seed it immediately BEFORE any CUDA operations
import torch
# MUST call manual_seed BEFORE any cuda operations
torch.manual_seed(_INIT_SEED)

# Now safe to import other torch modules
import torch.nn as nn
import torch.nn.functional as F_nn
from torch_geometric.data import HeteroData

from train.dynamic_hetero_gnn import DynamicHeteroGNN
from train.coder import GraphEncoder

# Aligners may be LatentAligner or TemporalCrossAligner depending on availability
try:
    from train.aligner import LatentAligner
except Exception:
    try:
        from train.aligner import TemporalCrossAligner as LatentAligner
    except Exception:
        LatentAligner = None

# Optional utility: compute_batch_alpha
try:
    from utils.utils import compute_batch_alpha
except Exception:
    compute_batch_alpha = None

# Lowpass loss helper (optional)
try:
    from train.loss_helpers import lowpass_mse_loss
except Exception:
    def lowpass_mse_loss(a, b, kernel_size=11):
        return torch.tensor(0.0, device=(a.device if a is not None else "cpu"))

# Dynamic variability weighting (optional)
try:
    from train.variability_weighting import DynamicVariabilityWeighting
except Exception:
    DynamicVariabilityWeighting = None


class DynamicHeteroTrainer:
    """
    DynamicHeteroTrainer (B 方案，GraphEncoder + DynamicHeteroGNN 分工版)

    主要职责：
    - 使用 GraphEncoder 统一负责：HeteroData -> (x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict)
    - 使用 DynamicHeteroGNN 负责：encoded + edges -> GNN+GRU+TemporalDecoder+NodeDecoder
    - 定义并优化多种损失（重构、时序预测、对齐、频域等）
    - 提供 train(), save_model(), load_model(), _temporal_prediction_loss() 等公开接口

    设计原则：
    - fail-fast：接口/shape 不符合预期时直接抛出异常；
    - 逻辑分层清晰：不在 Trainer 里做隐式编码/解码，唯一编码入口是 GraphEncoder；
    - 保留你原有的复杂损失与 early stopping 逻辑，但尽量保证实现优雅简洁。
    """

    def __init__(
        self,
        hetero_data: Union[HeteroData, List[HeteroData], Dict[Any, Any]],
        input_dims: Optional[Dict[str, int]] = None,
        hidden_dim: int = 128,
        num_layers: int = 8,
        dropout: float = 0.3,
        lr: float = 4e-4,
        num_epochs: int = 100,
        recon_weight: float = 1.0,
        recon_norm_weight: float = 1.0,
        recon_corr_weight: float = 0.0,
        temp_weight: float = 0.5,
        align_weight: float = 1.0,
        temporal_T: int = 200,
        spatial_T: int = 384,
        use_amp: bool = False,
        weight_decay: float = 1e-5,
        debug: bool = False,
        grad_clip: float = 1.0,
        warmup_epochs: int = 0,
        freeze_scale_during_warmup: bool = True,
        scale_lr_mul: float = 5.0,
        feature_lr_mul: float = 5.0,
        batch_rescale_fn: Optional[Callable] = None,
        batch_rescale_cfg: Optional[Dict] = None,
        recon_feat_var_weight: float = 0.0,
        scale_only_epochs: int = 0,
        scale_only_lr_mul: float = 20.0,
        spec_loss_weight: float = 0.0,
        spec_kernel_size: int = 11,
        shift_invariant_range: int = 0,
        shift_invariant_temp: float = 1.0,
        auto_align: bool = False,
        auto_align_max_lag: int = 120,
        # New parameters for enhanced features
        enable_prediction: bool = False,
        prediction_context_length: Optional[int] = None,
        prediction_steps: int = 10,
        prediction_weight: float = 0.1,
        enable_metrics_tracking: bool = True,
        metrics_output_dir: Optional[str] = None,
        gradient_accumulation_steps: int = 1,  # NEW: Gradient accumulation
        clear_cache_frequency: int = 1,  # NEW: Clear CUDA cache every N batches (1=every batch)
        # NEW: Dynamic variability weighting parameters
        enable_dynamic_weighting: bool = False,
        dynamic_weighting_config: Optional[Dict] = None,
    ):
        # ---------- Early random seed initialization (runtime call) ----------
        # NOTE: Module-level seed initialization has ALREADY occurred (lines 13-20)
        # with _INIT_SEED=42 to prevent THPGenerator errors during module import.
        # 
        # This runtime call to set_random_seed() will:
        # 1. Re-initialize with the same seed for consistency
        # 2. Ensure deterministic behavior is properly configured
        # 3. Apply any additional CUDA-specific settings (if deterministic=True)
        #
        # TrainingWorkflow may call set_random_seed() again with config.random_seed
        # before creating the trainer, which will be the actual seed for training.
        try:
            from utils.utils import set_random_seed
            # Re-initialize with same seed to ensure proper CUDA configuration
            set_random_seed(_INIT_SEED)
        except ImportError as e:
            # Fallback: Module-level initialization has already occurred,
            # so this is just for additional CUDA configuration if needed
            _logger = logging.getLogger(__name__)
            _logger.warning(f"Could not import set_random_seed, using module-level initialization: {e}")
            
            # Try to configure CUDA explicitly if available
            try:
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(_INIT_SEED)
            except RuntimeError as cuda_err:
                # Log CUDA initialization issues but continue
                _logger.warning(f"CUDA seed initialization failed: {cuda_err}")
        
        # ---------- logger ----------
        self.logger = logging.getLogger("DynamicHeteroTrainer")
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.logger.addHandler(ch)
        self.logger.setLevel(logging.INFO if debug else logging.WARNING)

        # ---------- device & config ----------
        # Device detection is now safe after random seed initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hetero_data = hetero_data
        self.input_dims = input_dims or {}
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.recon_weight = float(recon_weight)
        self.recon_norm_weight = float(recon_norm_weight)
        self.recon_corr_weight = float(recon_corr_weight)
        self.temp_weight = float(temp_weight)
        self.align_weight = float(align_weight)
        self.temporal_T = int(temporal_T)
        self.spatial_T = int(spatial_T)
        self.use_amp = bool(use_amp and torch.cuda.is_available())
        self.weight_decay = float(weight_decay)
        self.debug = bool(debug)
        self.grad_clip = float(grad_clip)

        # warmup & scale handling
        self.warmup_epochs = int(warmup_epochs)
        self.freeze_scale_during_warmup = bool(freeze_scale_during_warmup)
        self.scale_lr_mul = float(scale_lr_mul)
        self.feature_lr_mul = float(feature_lr_mul)

        # batch rescale utility
        self.batch_rescale_fn = batch_rescale_fn
        self.batch_rescale_cfg = batch_rescale_cfg or {"enable": False, "only": [], "warmup_epochs": 0}

        # recon feature variance regularizer
        self.recon_feat_var_weight = float(recon_feat_var_weight)

        # scale-only fine-tune options
        self.scale_only_epochs = int(scale_only_epochs)
        self.scale_only_lr_mul = float(scale_only_lr_mul)

        # SPEC lowpass loss options
        self.spec_loss_weight = float(spec_loss_weight)
        self.spec_kernel_size = int(spec_kernel_size)

        # shift-invariant options
        self.shift_invariant_range = int(shift_invariant_range)
        self.shift_invariant_temp = max(1e-6, float(shift_invariant_temp))

        # auto align config
        self.auto_align = bool(auto_align)
        self.auto_align_max_lag = int(auto_align_max_lag)
        self._auto_align_cache: Dict[str, int] = {}

        # New: prediction config
        self.enable_prediction = bool(enable_prediction)
        self.prediction_context_length = int(prediction_context_length) if prediction_context_length is not None else None
        self.prediction_steps = int(prediction_steps)
        self.prediction_weight = float(prediction_weight)
        
        # New: gradient accumulation
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        
        # New: CUDA cache clearing frequency
        self.clear_cache_frequency = max(1, int(clear_cache_frequency))

        # New: metrics tracking
        self.enable_metrics_tracking = bool(enable_metrics_tracking)
        if self.enable_metrics_tracking:
            try:
                from utils.metrics_tracker import MetricsTracker
                self.metrics_tracker = MetricsTracker(
                    output_dir=metrics_output_dir,
                    enabled=True
                )
                self.logger.info("Metrics tracking enabled")
            except ImportError:
                self.logger.warning("Failed to import MetricsTracker, metrics tracking disabled")
                self.metrics_tracker = None
                self.enable_metrics_tracking = False
        else:
            self.metrics_tracker = None

        # New: dynamic variability weighting
        self.enable_dynamic_weighting = bool(enable_dynamic_weighting)
        self.dynamic_weighting = None
        self.modality_weights = {}  # Cache for per-modality weights
        
        if self.enable_dynamic_weighting and DynamicVariabilityWeighting is not None:
            try:
                # Create config with training stage parameters
                if dynamic_weighting_config is None:
                    dynamic_weighting_config = {}
                
                # Merge with warmup_epochs if not explicitly set
                if 'warmup_epochs' not in dynamic_weighting_config:
                    dynamic_weighting_config['warmup_epochs'] = self.warmup_epochs
                
                self.dynamic_weighting = DynamicVariabilityWeighting(**dynamic_weighting_config)
                self.logger.info("Dynamic variability weighting enabled")
                self.logger.info(f"  - Warmup: {dynamic_weighting_config.get('warmup_epochs', self.warmup_epochs)} epochs")
                self.logger.info(f"  - Main: {dynamic_weighting_config.get('main_epochs', 60)} epochs")
                self.logger.info(f"  - Finetune: {dynamic_weighting_config.get('finetune_epochs', 30)} epochs")
            except Exception as e:
                self.logger.warning(f"Failed to initialize dynamic weighting: {e}")
                self.enable_dynamic_weighting = False
        elif self.enable_dynamic_weighting:
            self.logger.warning("Dynamic weighting requested but DynamicVariabilityWeighting not available")
            self.enable_dynamic_weighting = False

        # optional logger config helper
        try:
            from utils.log_control import configure_trainer_logger
            configure_trainer_logger(self.logger, debug=self.debug, suppress_in_debug=True, interval=30.0, initial_allow=1)
        except Exception:
            pass

        # ---------- GraphEncoder ----------
        try:
            self.graph_encoder = GraphEncoder(
                spatial_T=self.spatial_T,
                hidden_dim=self.hidden_dim,
                debug=self.debug,
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"[Init] Failed to construct GraphEncoder: {e}")

        # flatten hetero_data into list
        self.data_list = self._flatten_data(self.hetero_data)
        if len(self.data_list) == 0:
            raise ValueError("[Init] No hetero_data provided to trainer")

        # ---------- metadata inference ----------
        sample = self.data_list[0]
        if hasattr(sample, "metadata"):
            self.metadata = sample.metadata()
        else:
            node_types = list(sample.keys())
            # 注意：edge_types 这里只作占位；GNN 中真正用的是 data.edge_types
            self.metadata = (node_types, [])

        # ---------- infer input dims if not provided ----------
        if not self.input_dims:
            inferred: Dict[str, int] = {}
            for nt in self.metadata[0]:
                try:
                    x_seq = getattr(sample[nt], "x_seq", None)
                    if isinstance(x_seq, torch.Tensor) and x_seq.ndim == 3:
                        inferred[nt] = int(x_seq.shape[2])
                    else:
                        inferred[nt] = 1
                except Exception:
                    inferred[nt] = 1
            self.input_dims = inferred
            self.logger.info(f"[Init] inferred input dims: {self.input_dims}")

        # ---------- instantiate model & aligner ----------
        self.model = DynamicHeteroGNN(
            metadata=self.metadata,
            node_feature_dims=self.input_dims,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            temporal_T=self.temporal_T,
            spatial_T=self.spatial_T,
            debug=self.debug,
        ).to(self.device)

        try:
            if LatentAligner is None:
                raise RuntimeError("LatentAligner not available")
            self.aligner = LatentAligner(
                hidden_dim=self.hidden_dim,
                mode="nodewise",
                lambda_align=1.0,
                temperature=0.3,
            ).to(self.device)
        except Exception:
            try:
                from train.aligner import TemporalCrossAligner
                self.aligner = TemporalCrossAligner(hidden_dim=self.hidden_dim, dropout=self.dropout).to(self.device)
            except Exception:
                class _DummyAligner(nn.Module):
                    def forward(self, a, b):
                        return torch.tensor(0.0, device=a.device if a is not None else "cpu")
                self.aligner = _DummyAligner()

        # ---------- New: predictor module ----------
        self.predictor = None
        if self.enable_prediction:
            try:
                from train.predictor import PredictorHead
                self.predictor = PredictorHead(
                    hidden_dim=self.hidden_dim,
                    n_future_steps=self.prediction_steps,
                    context_length=self.prediction_context_length,
                    num_layers=3,
                    num_heads=8,
                    dropout=self.dropout,
                    use_residual=True,
                    use_gradient_checkpointing=True  # Enable gradient checkpointing for memory efficiency
                ).to(self.device)
                context_info = f"context={self.prediction_context_length}" if self.prediction_context_length else "full sequence"
                self.logger.info(f"Predictor enabled: use {context_info} to predict {self.prediction_steps} steps ahead")
                self.logger.info("  ✓ Gradient checkpointing enabled for memory efficiency")
            except ImportError:
                self.logger.warning("Failed to import PredictorHead, prediction disabled")
                self.enable_prediction = False

        # ---------- lazy params: dummy forward ----------
        try:
            sample_graph = self.data_list[0].to(self.device)
            with torch.no_grad():
                x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict = self.graph_encoder(sample_graph)
                _ = self.model(
                    data=sample_graph,
                    edge_index_dict=edge_index_dict,
                    encoded_dict=x_dict,
                    num_nodes_dict=num_nodes_dict,
                    stats_dict=stats_dict,
                )
        except Exception as e:
            self.logger.debug(f"[Init] dummy forward failed or skipped: {e}")

        # ---------- optimizer param groups ----------
        base_params: List[torch.nn.Parameter] = []
        feature_params: List[torch.nn.Parameter] = []
        scale_params: List[torch.nn.Parameter] = []
        aligner_params = list(self.aligner.parameters())

        def is_feature_decoder_name(name: str) -> bool:
            substrs = ["feature_decoders", "decoder_input_proj", "proj_head", "temporal_projs", "denorm_decoders"]
            return any(s in name for s in substrs)

        def is_scale_name(name: str) -> bool:
            # log_scale_* 或 scale_*（排除 scale_fixed_* buffer）
            if "log_scale" in name:
                return True
            if "scale_" in name and "scale_fixed" not in name:
                return True
            return False

        seen_param_ids = set()
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen_param_ids:
                continue
            seen_param_ids.add(pid)
            if is_feature_decoder_name(name):
                feature_params.append(p)
            elif is_scale_name(name):
                scale_params.append(p)
            else:
                base_params.append(p)

        param_groups: List[Dict[str, Any]] = []
        if base_params:
            param_groups.append({"params": base_params})
        if feature_params:
            param_groups.append({"params": feature_params, "lr": self.lr * self.feature_lr_mul})
        if scale_params:
            param_groups.append({"params": scale_params, "lr": self.lr * self.scale_lr_mul})
        if aligner_params:
            param_groups.append({"params": aligner_params})
        
        # Add predictor parameters if enabled
        if self.predictor is not None:
            predictor_params = list(self.predictor.parameters())
            if predictor_params:
                param_groups.append({"params": predictor_params})
                self.logger.info(f"Added {len(predictor_params)} predictor parameters to optimizer")

        self.optimizer = torch.optim.Adam(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.7)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # ---------- bookkeeping ----------
        self.loss_log = {"total": [], "align": [], "temp": [], "recon": [], "recon_norm": [], "spec": []}
        self.diagnostic_dir = None
        self.use_batch_rescale = bool(self.batch_rescale_cfg.get("enable", False) and self.batch_rescale_fn is not None)

        # 保存初始 requires_grad 映射，便于 scale freeze/unfreeze
        self._orig_requires_grad_map: Dict[str, bool] = {
            name: p.requires_grad for name, p in self.model.named_parameters()
        }

        # debug optimizer summary
        if self.debug:
            try:
                grp_info = []
                for i, g in enumerate(self.optimizer.param_groups):
                    names = []
                    for p in g["params"][:10]:
                        for nm, par in self.model.named_parameters():
                            if par is p:
                                names.append(nm)
                                break
                    grp_info.append(
                        {"index": i, "lr": float(g.get("lr", self.lr)), "n_params": len(g["params"]), "example_names": names}
                    )
                self.logger.info(
                    f"[Init] Trainer initialized on device={self.device}. "
                    f"model params={sum(p.numel() for p in self.model.parameters())}"
                )
                self.logger.info(f"[Init] Optimizer param groups: {grp_info}")
            except Exception:
                pass
        else:
            self.logger.info(
                f"[Init] Trainer initialized on device={self.device}. "
                f"model params={sum(p.numel() for p in self.model.parameters())}"
            )
            self.logger.info(
                f"[Init] Optimizer param groups: base={len(base_params)}, "
                f"feature={len(feature_params)}, scale={len(scale_params)}, aligner={len(aligner_params)}"
            )

    # ----------------------------------------------------------------------
    # Utils
    # ----------------------------------------------------------------------
    def _flatten_data(self, data: Union[HeteroData, List, Dict]) -> List[HeteroData]:
        if data is None:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            flat: List[HeteroData] = []
            for v in data.values():
                if isinstance(v, list):
                    flat.extend(v)
                else:
                    flat.append(v)
            return flat
        return [data]

    # ----------------------------------------------------------------------
    # Temporal prediction loss (latent) - 你原来的逻辑，保持不变，只在接口上更严谨
    # ----------------------------------------------------------------------
    def _temporal_prediction_loss(
        self,
        proj_seq: torch.Tensor,
        nt: str,
        context_len: int = 40,
        predict_len: int = 4,
        teacher_forcing_ratio: float = 0.3,
        return_preds: bool = False,
        stats_nt: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Autoregressive multi-step prediction loss based on proj_seq for node-type nt.

        - proj_seq: (N, T, H)
        - 返回:
          - 若 return_preds=False: loss (Tensor)
          - 若 return_preds=True: (loss, pred_feat, pred_denorm)
        """
        device = self.device

        if proj_seq is None or proj_seq.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            if return_preds:
                return zero, None, None
            return zero

        if proj_seq.ndim != 3:
            raise ValueError(f"[TempLoss] proj_seq must be (N,T,H), got {tuple(proj_seq.shape)}")

        N, T, C = proj_seq.shape
        if T < 2:
            zero = torch.tensor(0.0, device=device)
            if return_preds:
                return zero, None, None
            return zero

        # get GRU for node type
        if nt not in self.model.grus:
            zero = torch.tensor(0.0, device=device)
            if return_preds:
                return zero, None, None
            return zero
        gru = self.model.grus[nt]

        # 调整 context_len 至合理范围
        context_len = min(context_len, T - 1)
        if context_len <= 0:
            context_len = max(1, T - 1)

        context = proj_seq[:, :context_len, :]
        out_ctx, h = gru(context)

        next_input = out_ctx[:, -1:, :].contiguous()
        future_targets = proj_seq[:, context_len:context_len + predict_len, :] if T > context_len else None

        preds: List[torch.Tensor] = []
        for step in range(predict_len):
            out_step, h = gru(next_input, h)
            pred = out_step[:, -1:, :].contiguous()
            preds.append(pred)

            do_teacher = (
                self.model.training
                and future_targets is not None
                and torch.rand(1, device=device).item() < float(teacher_forcing_ratio)
            )
            if do_teacher and step < future_targets.shape[1]:
                next_input = future_targets[:, step:step + 1, :].contiguous()
            else:
                next_input = pred

        pred_feat = torch.cat(preds, dim=1) if preds else torch.zeros((N, 0, C), device=device)

        time_loss = torch.tensor(0.0, device=device)
        freq_loss = torch.tensor(0.0, device=device)

        if future_targets is not None and future_targets.numel() != 0:
            K = min(predict_len, future_targets.shape[1])
            if K > 0:
                time_loss = F_nn.mse_loss(pred_feat[:, :K, :], future_targets[:, :K, :])

                # 频域 loss
                try:
                    pred_fft = torch.fft.rfft(pred_feat[:, :K, :].permute(0, 2, 1), dim=-1)
                    targ_fft = torch.fft.rfft(future_targets[:, :K, :].permute(0, 2, 1), dim=-1)
                    pred_mag = pred_fft.abs()
                    targ_mag = targ_fft.abs()
                    n_bins = pred_mag.size(-1)
                    freq_idx = torch.arange(n_bins, device=device).float()
                    weights = (freq_idx / max(1.0, (n_bins - 1))).view(1, 1, -1)
                    freq_loss = F_nn.mse_loss(pred_mag * weights, targ_mag * weights)
                except Exception:
                    freq_loss = torch.tensor(0.0, device=device)

        a_t = getattr(self, "temporal_loss_alpha", 1.0)
        a_f = getattr(self, "temporal_loss_beta", 0.5)
        loss = a_t * time_loss + a_f * freq_loss

        pred_denorm = None
        if return_preds:
            den = getattr(self.model, "denorm_decoders", None)
            if den is not None:
                try:
                    if hasattr(den, "forward_feature_and_denorm"):
                        _, pred_denorm = den.forward_feature_and_denorm(pred_feat, stats_nt or {}, node_type=nt)
                    else:
                        pred_denorm = den(pred_feat, stats_nt or {}, node_type=nt)
                except Exception:
                    pred_denorm = None
            return loss, pred_feat, pred_denorm

        return loss

    # ----------------------------------------------------------------------
    # 训练主循环（对接 GraphEncoder + DynamicHeteroGNN）
    # ----------------------------------------------------------------------
    def train(self, num_epochs: Optional[int] = None, verbose: bool = True):
        """
        训练循环（严格版）：
        - 每个 batch：
          1) graph_encoder(data) -> x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict
          2) model(data, edge_index_dict, encoded_dict=x_dict, num_nodes_dict, stats_dict) -> 8-tuple
          3) 计算对齐 / 时序预测 / 重构 / 频域等 loss
        - 对接口不满足的情况直接抛异常，不做隐式 fallback。
        """

        epochs = num_epochs or self.num_epochs
        self.model.train()
        self.aligner.train()
        
        # NEW: Log training configuration at start
        self.logger.info("=" * 80)
        self.logger.info(f"Starting Training: {epochs} epochs")
        self.logger.info(f"  Loss weights: recon={self.recon_weight}, temp={self.temp_weight}, align={self.align_weight}")
        if self.enable_prediction:
            self.logger.info(f"  ✓ Prediction ENABLED: weight={self.prediction_weight}, context={self.prediction_context_length}, steps={self.prediction_steps}")
            self.logger.info(f"    Using autoregressive multi-step prediction with PredictorHead")
        else:
            self.logger.info(f"  ✗ Prediction DISABLED")
        self.logger.info("=" * 80)

        # batch_rescale 默认配置防御
        if "warmup_epochs" not in self.batch_rescale_cfg:
            self.batch_rescale_cfg["warmup_epochs"] = 0
        if not hasattr(self, "scale_only_epochs"):
            self.scale_only_epochs = 0
        if not hasattr(self, "scale_only_lr_mul"):
            self.scale_only_lr_mul = 20.0

        monitor_nt = "fmri"
        patience = getattr(self, "early_stop_patience", 5)
        lr_shrink = float(getattr(self, "early_stop_lr_shrink", 0.5))
        min_lr = float(getattr(self, "early_stop_min_lr", 1e-7))

        # warmup: freeze scale
        if self.warmup_epochs > 0 and self.freeze_scale_during_warmup:
            self._set_scale_requires_grad(False)
            self.logger.info(f"[Warmup] freezing scale params for {self.warmup_epochs} epochs")

        best_rel = float("inf")
        best_epoch = 0
        no_improve = 0
        best_model_state = None
        best_opt_state = None

        # helper: model 输出校验
        def _validate_model_outputs(outputs):
            if not (isinstance(outputs, (list, tuple)) and len(outputs) == 8):
                raise RuntimeError(
                    "model.forward must return 8-tuple "
                    "(z_dict, gru_out, proj_seq_dict, recon_seq_denorm, recon_seq_scaled, "
                    "global_seq, recon_feature_dict, recon_denorm_dict). "
                    f"Got type={type(outputs)}, len={len(outputs) if isinstance(outputs,(list,tuple)) else 'N/A'}"
                )
            return outputs

        for epoch in range(1, epochs + 1):
            # Clear CUDA cache at the start of each epoch for a clean slate
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            start = time.time()
            total_loss = total_align = total_temp = total_recon = total_recon_norm = total_spec = 0.0
            total_predictor = 0.0  # NEW: Track PredictorHead loss
            batches = 0
            
            # Check for empty data_list to prevent silent hangs
            if len(self.data_list) == 0:
                raise RuntimeError(
                    f"[Train] data_list is empty at epoch {epoch}. "
                    "Training cannot proceed without data."
                )

            # warmup 结束后解冻 scale
            if epoch == self.warmup_epochs + 1 and self.freeze_scale_during_warmup:
                self._set_scale_requires_grad(True)
                self.logger.info("[Warmup] unfreezing scale params, resuming full training")

            # batch_rescale 的 warmup 结束后禁用
            if self.batch_rescale_cfg.get("enable", False) and epoch > int(self.batch_rescale_cfg.get("warmup_epochs", 0)):
                self.batch_rescale_cfg["enable"] = False
                self.logger.info(
                    f"[BatchRescale] warmup_epochs passed ({self.batch_rescale_cfg.get('warmup_epochs')}), disabling batch_rescale"
                )

            for data_idx, data in enumerate(self.data_list):
                data = data.to(self.device)
                
                # Zero gradients only at start of accumulation cycle
                if data_idx % self.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                # 1) GraphEncoder
                enc_out = self.graph_encoder(data)
                if not (isinstance(enc_out, (list, tuple)) and len(enc_out) >= 5):
                    raise RuntimeError(
                        "graph_encoder(data) must return at least 5-tuple "
                        "(x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict, ...). "
                        f"Received type={type(enc_out)}"
                    )
                x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict = enc_out[:5]

                # 2) model forward
                outputs = self.model(
                    data=data,
                    edge_index_dict=edge_index_dict,
                    encoded_dict=x_dict,
                    num_nodes_dict=num_nodes_dict,
                    stats_dict=stats_dict,
                )
                # Validate model outputs before unpacking
                outputs = _validate_model_outputs(outputs)
                (
                    z_dict,
                    gru_seq_dict,
                    proj_seq_dict,
                    recon_seq_dict,
                    recon_seq_scaled,
                    global_seq,
                    recon_feature_dict,
                    recon_denorm_dict,
                ) = _validate_model_outputs(outputs)

                # 3) z_dict 清洗为 (N,H)
                sanitized_z: Dict[str, Optional[torch.Tensor]] = {}
                if z_dict is None:
                    for nt in self.metadata[0]:
                        sanitized_z[nt] = None
                else:
                    if not isinstance(z_dict, dict):
                        raise TypeError("z_dict must be dict[node_type]->Tensor or None")
                    for nt in self.metadata[0]:
                        z = z_dict.get(nt, None)
                        if z is None:
                            sanitized_z[nt] = None
                            continue
                        if not isinstance(z, torch.Tensor):
                            raise TypeError(f"z_dict['{nt}'] must be Tensor, got {type(z)}")
                        if z.ndim == 1:
                            raise ValueError(
                                f"z_dict['{nt}'] is 1D (shape {tuple(z.shape)}). "
                                "Please return (N,H) or (N,T,H) from model."
                            )
                        if z.ndim == 3:
                            z = z.mean(dim=1)
                        if z.ndim != 2:
                            raise ValueError(f"z_dict['{nt}'] after processing must be (N,H), got {tuple(z.shape)}")
                        sanitized_z[nt] = z

                # NEW: Compute dynamic variability weights for each modality
                # Note: Weights are computed once per epoch at the first batch
                if self.enable_dynamic_weighting and self.dynamic_weighting is not None:
                    # Compute weights for each modality based on current data
                    for nt in self.metadata[0]:
                        # Use raw input data for variability computation
                        x_seq = getattr(data[nt], "x_seq", None)
                        if x_seq is not None and isinstance(x_seq, torch.Tensor):
                            try:
                                weights = self.dynamic_weighting.compute_modality_weights(
                                    x=x_seq,
                                    modality=nt,
                                    epoch=epoch,
                                    force_update=(data_idx == 0)  # Update at start of each epoch
                                )
                                self.modality_weights[nt] = weights
                            except Exception as e:
                                self.logger.warning(f"Failed to compute weights for {nt}: {e}")
                                # Fallback to uniform weights
                                self.modality_weights[nt] = self._uniform_weights(x_seq)
                else:
                    # If disabled, use uniform weights
                    for nt in self.metadata[0]:
                        x_seq = getattr(data[nt], "x_seq", None)
                        if x_seq is not None and isinstance(x_seq, torch.Tensor):
                            self.modality_weights[nt] = self._uniform_weights(x_seq)

                # 4) align loss
                a_z_f = sanitized_z.get("fmri", None)
                a_z_e = sanitized_z.get("eeg", None)
                if a_z_f is not None and a_z_e is not None:
                    align_loss = self.aligner(a_z_f, a_z_e)
                    if not isinstance(align_loss, torch.Tensor):
                        raise TypeError("aligner must return a torch.Tensor")
                else:
                    align_loss = torch.tensor(0.0, device=self.device)

                # 5) temporal prediction loss + raw 预测 loss
                temp_loss = torch.tensor(0.0, device=self.device)
                raw_pred_loss_total = torch.tensor(0.0, device=self.device)
                raw_pred_weight = float(getattr(self, "raw_pred_weight", self.temp_weight))
                
                # NEW: Use PredictorHead for prediction if enabled
                predictor_loss = torch.tensor(0.0, device=self.device)
                if self.enable_prediction and self.predictor is not None:
                    # Use PredictorHead for multi-step prediction with sliding window
                    # Log once per epoch to confirm prediction is running
                    if data_idx == 0 and epoch % 10 == 0 and verbose:
                        self.logger.info(f"[Prediction] Running autoregressive prediction (context={self.prediction_context_length}, steps={self.prediction_steps})")
                    
                    for nt in self.metadata[0]:
                        seq = None
                        if isinstance(proj_seq_dict, dict):
                            seq = proj_seq_dict.get(nt, None)
                        elif isinstance(proj_seq_dict, torch.Tensor):
                            seq = proj_seq_dict
                        if seq is None or (isinstance(seq, torch.Tensor) and seq.numel() == 0):
                            continue
                        if not isinstance(seq, torch.Tensor) or seq.ndim != 3:
                            continue
                        
                        N, T, H = seq.shape
                        min_required = (self.prediction_context_length or 10) + self.prediction_steps
                        if T < min_required:
                            continue
                        
                        # Use fewer sliding windows to reduce memory usage
                        # Process only 2-3 windows per sequence to minimize memory footprint
                        context_len = self.prediction_context_length or 50
                        
                        # Calculate positions for 2-3 evenly-spaced windows
                        max_start = T - context_len - self.prediction_steps
                        if max_start <= 0:
                            continue
                            
                        # Use only 2 windows: beginning and end of sequence
                        # Sort to ensure proper ordering for loop control
                        window_starts = sorted([0, max_start])
                        
                        num_windows = 0
                        window_loss = torch.tensor(0.0, device=self.device)
                        
                        # Process only selected windows to reduce memory
                        for start_idx in window_starts:
                            context_start = start_idx
                            context_end = start_idx + context_len
                            target_start = context_end
                            target_end = context_end + self.prediction_steps
                            
                            if target_end > T:
                                break  # This and any subsequent windows don't fit
                            
                            # Extract context and target
                            context_seq = seq[:, context_start:context_end, :]
                            target_seq = seq[:, target_start:target_end, :]
                            
                            # Predict using PredictorHead (autoregressive)
                            predictions, _ = self.predictor(context_seq, return_attention=False)
                            
                            # Compute prediction loss for this window
                            pred_loss = F_nn.mse_loss(predictions, target_seq)
                            window_loss = window_loss + pred_loss
                            num_windows += 1
                            
                            # Clean up intermediate tensors immediately to free memory
                            del predictions, pred_loss, context_seq, target_seq
                        
                        # Average over all windows
                        if num_windows > 0:
                            avg_window_loss = window_loss / num_windows
                            predictor_loss = predictor_loss + avg_window_loss
                            # Clean up window loss tensor
                            del window_loss
                            # Log window count on first batch to show prediction is active
                            if data_idx == 0 and epoch % 10 == 0 and verbose:
                                self.logger.info(f"  [{nt}] Trained on {num_windows} prediction windows (avg loss: {float(avg_window_loss):.6f})")

                for nt in self.metadata[0]:
                    seq = None
                    if isinstance(proj_seq_dict, dict):
                        seq = proj_seq_dict.get(nt, None)
                    elif isinstance(proj_seq_dict, torch.Tensor):
                        seq = proj_seq_dict
                    if seq is None or (isinstance(seq, torch.Tensor) and seq.numel() == 0):
                        continue
                    if not isinstance(seq, torch.Tensor) or seq.ndim != 3:
                        raise ValueError(f"proj_seq_dict['{nt}'] must be Tensor (N,T,H), got {None if seq is None else tuple(seq.shape)}")

                    loss_t, pred_feat, pred_denorm = self._temporal_prediction_loss(
                        seq, nt, return_preds=True, stats_nt=stats_dict.get(nt, None)
                    )
                    temp_loss = temp_loss + loss_t

                    # raw 预测 loss
                    if pred_denorm is not None:
                        if pred_denorm.ndim != 3:
                            raise ValueError(f"pred_denorm for '{nt}' must be (N,T_pred,F_raw), got {tuple(pred_denorm.shape)}")
                        targ_raw = getattr(data[nt], "x_seq", None)
                        if targ_raw is None:
                            raise RuntimeError(
                                f"pred_denorm returned for '{nt}' but data['{nt}'].x_seq is missing (cannot compute raw forecast loss)"
                            )
                        Np, Tp, Fp = pred_denorm.shape
                        Nt, Tt, Ft = targ_raw.shape
                        common_F = min(Fp, Ft)
                        if Tt < Tp:
                            raise ValueError(f"Target length {Tt} < predicted length {Tp} for '{nt}', cannot align")
                        target_future = targ_raw.to(self.device)[:, -Tp:, :common_F]
                        pred_crop = pred_denorm[:, :Tp, :common_F].to(self.device)
                        raw_l = F_nn.mse_loss(pred_crop, target_future)
                        raw_pred_loss_total = raw_pred_loss_total + raw_l

                # 6) 构建 recon_final_map（优先级：recon_feature_dict+denorm → recon_seq_scaled → recon_seq_dict+denorm）
                recon_final_map: Dict[str, torch.Tensor] = {}
                den = getattr(self.model, "denorm_decoders", None)

                for nt in self.metadata[0]:
                    chosen = None

                    # A: recon_feature_dict + denorm_decoders.forward_feature_and_denorm
                    if isinstance(recon_feature_dict, dict) and nt in recon_feature_dict and den is not None:
                        recon_feat = recon_feature_dict[nt]
                        if not isinstance(recon_feat, torch.Tensor):
                            raise TypeError(f"recon_feature_dict['{nt}'] must be Tensor")
                        if recon_feat.ndim != 3:
                            raise ValueError(f"recon_feature_dict['{nt}'] must be (N,T,F), got {tuple(recon_feat.shape)}")
                        if hasattr(den, "forward_feature_and_denorm"):
                            out = den.forward_feature_and_denorm(
                                recon_feat, stats_dict.get(nt, None), node_type=nt
                            )
                            if not (isinstance(out, (list, tuple)) and len(out) >= 2):
                                raise RuntimeError("denorm_decoders.forward_feature_and_denorm must return (feature, denorm)")
                            _, recon_den = out[:2]
                            if not isinstance(recon_den, torch.Tensor):
                                raise TypeError("denorm_decoders.forward_feature_and_denorm must return Tensor as denorm")
                            chosen = recon_den.to(self.device)

                    # B: recon_seq_scaled（已经是 denorm）
                    if chosen is None and isinstance(recon_seq_scaled, dict) and nt in recon_seq_scaled:
                        candidate = recon_seq_scaled[nt]
                        if not isinstance(candidate, torch.Tensor):
                            raise TypeError(f"recon_seq_scaled['{nt}'] must be Tensor")
                        if candidate.ndim != 3:
                            raise ValueError(f"recon_seq_scaled['{nt}'] must be (N,T,F), got {tuple(candidate.shape)}")
                        chosen = candidate.to(self.device)

                    # C: recon_seq_dict（normalized）+ denorm_decoders
                    if chosen is None and isinstance(recon_seq_dict, dict) and nt in recon_seq_dict and den is not None:
                        candidate = recon_seq_dict[nt]
                        if not isinstance(candidate, torch.Tensor):
                            raise TypeError(f"recon_seq_dict['{nt}'] must be Tensor")
                        recon_den = den(candidate, stats_dict.get(nt, None), node_type=nt)
                        if not isinstance(recon_den, torch.Tensor):
                            raise TypeError("denorm_decoders(candidate, stats) must return Tensor")
                        chosen = recon_den.to(self.device)

                    if chosen is not None:
                        recon_final_map[nt] = chosen

                if self.recon_weight > 0.0 and len(recon_final_map) == 0:
                    raise RuntimeError(
                        "recon_weight > 0 but no usable reconstruction in recon_final_map. "
                        "Ensure model returns recon_seq_scaled or recon_feature_dict and denorm_decoders is valid."
                    )

                # 7) batch_rescale（如启用）
                if self.batch_rescale_cfg.get("enable", False):
                    if self.batch_rescale_fn is None:
                        raise RuntimeError("batch_rescale_cfg.enable=True but batch_rescale_fn is None")
                    temp_map: Dict[str, torch.Tensor] = {}
                    for nt, recon in recon_final_map.items():
                        apply_here = (not self.batch_rescale_cfg.get("only")) or (nt in self.batch_rescale_cfg.get("only", []))
                        if not apply_here:
                            temp_map[nt] = recon
                            continue
                        target = getattr(data[nt], "x_seq", None)
                        if target is None:
                            raise RuntimeError(f"batch_rescale requested for '{nt}' but data['{nt}'].x_seq missing")
                        r_det = recon.detach()
                        t_res = self._resample_time(target.to(self.device), r_det.shape[1])
                        alpha = self.batch_rescale_fn(r_det, t_res, self.batch_rescale_cfg)
                        if not (isinstance(alpha, (float, torch.Tensor))):
                            raise TypeError("batch_rescale_fn must return float or Tensor")
                        alpha_val = alpha if isinstance(alpha, torch.Tensor) else float(alpha)
                        temp_map[nt] = recon * alpha_val
                    recon_final_map = temp_map

                # 8) 重构 loss
                recon_loss = torch.tensor(0.0, device=self.device)
                recon_losses_per_nt: Dict[str, float] = {}
                for nt, recon in recon_final_map.items():
                    target = getattr(data[nt], "x_seq", None)
                    if target is None:
                        raise RuntimeError(f"recon available for '{nt}' but data['{nt}'].x_seq is missing")
                    target_res = self._resample_time(target.to(self.device), recon.shape[1])
                    Nr, Tr, Fr = recon.shape
                    Nt, Tt, Ft = target_res.shape
                    mN, mT, mF = min(Nr, Nt), min(Tr, Tt), min(Fr, Ft)
                    if mN <= 0 or mT <= 0 or mF <= 0:
                        raise ValueError(f"No overlap between recon and target for '{nt}'")
                    r_crop = recon[:mN, :mT, :mF]
                    t_crop = target_res[:mN, :mT, :mF]
                    
                    # NEW: Apply dynamic weights if available
                    if nt in self.modality_weights:
                        weights = self.modality_weights[nt][:mF]  # Match feature dimension
                        # Compute per-feature MSE
                        per_feature_loss = ((r_crop - t_crop) ** 2).mean(dim=(0, 1))  # [F]
                        # Weight and sum
                        l_nt = (per_feature_loss * weights).sum()
                    else:
                        # Standard MSE loss
                        l_nt = F_nn.mse_loss(r_crop, t_crop)
                    
                    recon_losses_per_nt[nt] = float(l_nt.detach().cpu())
                    recon_loss = recon_loss + l_nt

                # 9) recon_norm/spec loss（如果启用）
                recon_norm_loss = torch.tensor(0.0, device=self.device)
                spec_loss_total = torch.tensor(0.0, device=self.device)
                if isinstance(recon_feature_dict, dict) and self.recon_norm_weight > 0.0:
                    for nt in self.metadata[0]:
                        if nt not in recon_feature_dict:
                            continue
                        recon_feat = recon_feature_dict[nt]
                        if not isinstance(recon_feat, torch.Tensor) or recon_feat.ndim != 3:
                            raise ValueError(f"recon_feature_dict['{nt}'] must be (N,T,F)")
                        stats = stats_dict.get(nt, None)
                        target = getattr(data[nt], "x_seq", None)
                        if stats is None or target is None:
                            continue
                        target_res = self._resample_time(target.to(self.device), recon_feat.shape[1])
                        Nr, Tr, Fr = recon_feat.shape
                        Nt, Tt, Ft = target_res.shape
                        mN, mT, mF = min(Nr, Nt), min(Tr, Tt), min(Fr, Ft)
                        if mN <= 0 or mT <= 0 or mF <= 0:
                            continue
                        rf = recon_feat[:mN, :mT, :mF]
                        mean_expand = stats["mean"].expand(-1, recon_feat.shape[1], -1)[:mN, :mT, :mF]
                        std_expand = stats["std"].expand(-1, recon_feat.shape[1], -1)[:mN, :mT, :mF]
                        tnorm = (target_res[:mN, :mT, :mF] - mean_expand) / (std_expand + 1e-8)
                        
                        # NEW: Apply dynamic weights if available
                        if nt in self.modality_weights:
                            weights = self.modality_weights[nt][:mF]  # Match feature dimension
                            # Compute per-feature MSE
                            per_feature_loss = ((rf - tnorm) ** 2).mean(dim=(0, 1))  # [F]
                            # Weight and sum
                            recon_norm_loss = recon_norm_loss + (per_feature_loss * weights).sum()
                        else:
                            recon_norm_loss = recon_norm_loss + F_nn.mse_loss(rf, tnorm)
                        
                        if self.spec_loss_weight > 0.0:
                            spec_loss_total = spec_loss_total + lowpass_mse_loss(
                                rf, tnorm, kernel_size=self.spec_kernel_size
                            )

                # 10) recon_corr loss（如启用）
                recon_corr_loss = torch.tensor(0.0, device=self.device)
                if isinstance(recon_feature_dict, dict) and self.recon_corr_weight > 0.0:
                    r_list = []
                    eps = 1e-8
                    for nt in self.metadata[0]:
                        if nt not in recon_feature_dict:
                            continue
                        recon_feat = recon_feature_dict[nt]
                        stats = stats_dict.get(nt, None)
                        target = getattr(data[nt], "x_seq", None)
                        if (
                            not isinstance(recon_feat, torch.Tensor)
                            or recon_feat.ndim != 3
                            or stats is None
                            or target is None
                        ):
                            continue
                        target_res = self._resample_time(target.to(self.device), recon_feat.shape[1])
                        Nr, Tr, Fr = recon_feat.shape
                        Nt, Tt, Ft = target_res.shape
                        mN, mT, mF = min(Nr, Nt), min(Tr, Tt), min(Fr, Ft)
                        if mN <= 0 or mT <= 0 or mF <= 0:
                            continue
                        rf = recon_feat[:mN, :mT, :mF].reshape(-1, mF)
                        mean_expand = stats["mean"].expand(-1, recon_feat.shape[1], -1)[:mN, :mT, :mF]
                        std_expand = stats["std"].expand(-1, recon_feat.shape[1], -1)[:mN, :mT, :mF]
                        tnorm = ((target_res[:mN, :mT, :mF] - mean_expand) / (std_expand + eps)).reshape(-1, mF)
                        rf_center = rf - rf.mean(dim=0, keepdim=True)
                        t_center = tnorm - tnorm.mean(dim=0, keepdim=True)
                        num = (rf_center * t_center).sum(dim=0)
                        den = torch.sqrt((rf_center ** 2).sum(dim=0) * (t_center ** 2).sum(dim=0) + eps)
                        r_feat = num / (den + eps)
                        r_list.append(r_feat.mean())
                    if r_list:
                        mean_r = torch.stack(r_list).mean()
                        recon_corr_loss = 1.0 - mean_r

                # 11) 总 loss
                loss = (
                    self.align_weight * align_loss
                    + self.temp_weight * temp_loss
                    + self.recon_weight * recon_loss
                )
                if self.recon_norm_weight > 0.0:
                    loss = loss + self.recon_norm_weight * recon_norm_loss
                if self.recon_corr_weight > 0.0:
                    loss = loss + self.recon_corr_weight * recon_corr_loss
                if self.spec_loss_weight > 0.0:
                    loss = loss + self.spec_loss_weight * spec_loss_total
                if raw_pred_loss_total.numel() != 0 and float(raw_pred_loss_total.detach().cpu()) != 0.0:
                    loss = loss + raw_pred_weight * raw_pred_loss_total
                # NEW: Add PredictorHead loss if enabled
                if self.enable_prediction and predictor_loss.numel() != 0:
                    loss = loss + self.prediction_weight * predictor_loss
                
                # Scale loss for gradient accumulation
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                # 12) backward + step (with gradient accumulation)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    
                    # Only step optimizer every N accumulation steps
                    if (data_idx + 1) % self.gradient_accumulation_steps == 0 or (data_idx + 1) == len(self.data_list):
                        if self.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            params_to_clip = list(self.model.parameters()) + list(self.aligner.parameters())
                            if self.predictor is not None:
                                params_to_clip += list(self.predictor.parameters())
                            torch.nn.utils.clip_grad_norm_(params_to_clip, self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    loss.backward()
                    
                    # Only step optimizer every N accumulation steps
                    if (data_idx + 1) % self.gradient_accumulation_steps == 0 or (data_idx + 1) == len(self.data_list):
                        if self.grad_clip > 0:
                            params_to_clip = list(self.model.parameters()) + list(self.aligner.parameters())
                            if self.predictor is not None:
                                params_to_clip += list(self.predictor.parameters())
                            torch.nn.utils.clip_grad_norm_(params_to_clip, self.grad_clip)
                        self.optimizer.step()

                # 13) accumulate stats (convert to python floats to free GPU memory)
                total_loss += float(loss.detach().cpu())
                total_align += float(align_loss.detach().cpu())
                total_temp += float(temp_loss.detach().cpu())
                total_recon += float(recon_loss.detach().cpu())
                total_recon_norm += float(recon_norm_loss.detach().cpu())
                total_spec += float(spec_loss_total.detach().cpu())
                # NEW: Track predictor loss
                if self.enable_prediction and predictor_loss.numel() != 0:
                    total_predictor += float(predictor_loss.detach().cpu())
                batches += 1
                
                # Explicitly delete large intermediate tensors to free GPU memory
                # While Python GC will eventually collect these, explicit deletion before
                # torch.cuda.empty_cache() allows CUDA to reclaim GPU memory immediately
                # rather than waiting for the next GC cycle. This is critical for preventing OOM.
                del loss, align_loss, temp_loss, recon_loss, recon_norm_loss, spec_loss_total
                # Conditionally delete raw_pred_loss_total if it was computed
                try:
                    if raw_pred_loss_total.numel() != 0:
                        del raw_pred_loss_total
                except (NameError, AttributeError):
                    pass  # Variable doesn't exist or already deleted
                if self.enable_prediction:
                    del predictor_loss
                # Delete other large tensors
                del z_dict, gru_seq_dict, proj_seq_dict, recon_seq_dict, recon_seq_scaled
                del global_seq, recon_feature_dict, recon_denorm_dict
                # DO NOT delete recon_final_map - save it for epoch-level metrics
                # Delete encoded tensors
                del x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict, enc_out
                # Delete sanitized tensors
                del sanitized_z
                # Move data back to CPU to free GPU memory before next iteration
                data = data.cpu()

                if data_idx == 0 and verbose:
                    self.logger.info(f"[Train] epoch={epoch} batch=0 recon_losses={recon_losses_per_nt}")
                
                # Clear CUDA cache after all tensors are deleted for maximum effectiveness
                # This is critical for preventing memory fragmentation on small GPUs
                if torch.cuda.is_available() and (data_idx + 1) % self.clear_cache_frequency == 0:
                    torch.cuda.empty_cache()

            if batches == 0:
                raise RuntimeError(
                    f"[Train] No batches processed in epoch {epoch}. "
                    "All batches may have been skipped or failed validation. "
                    "Check data_list content and graph_encoder outputs."
                )

            # 14) epoch-level log & scheduler
            avg_total = total_loss / batches
            avg_align = total_align / batches
            avg_temp = total_temp / batches
            avg_recon = total_recon / batches
            avg_recon_norm = total_recon_norm / batches
            avg_spec = total_spec / batches
            avg_predictor = total_predictor / batches  # NEW: Average predictor loss

            self.loss_log["total"].append(avg_total)
            self.loss_log["align"].append(avg_align)
            self.loss_log["temp"].append(avg_temp)
            self.loss_log["recon"].append(avg_recon)
            self.loss_log["recon_norm"].append(avg_recon_norm)
            self.loss_log["spec"].append(avg_spec)
            # NEW: Log predictor loss
            if self.enable_prediction:
                if "predictor" not in self.loss_log:
                    self.loss_log["predictor"] = []
                self.loss_log["predictor"].append(avg_predictor)

            try:
                self.scheduler.step()
            except Exception:
                self.logger.warning("[Train] scheduler.step() failed; continuing")

            # 15) relative error (使用最后一个 batch 的 recon_final_map 和 data)
            rel_error_epoch: Dict[str, float] = {}
            try:
                rel_error_epoch = self._compute_relative_error(recon_final_map, data, self.metadata)
            except Exception as e:
                import traceback
                self.logger.warning(f"[Train] relative error computation failed for this epoch: {e}")
                self.logger.debug(traceback.format_exc())
            finally:
                # Clean up recon_final_map after use to free memory
                try:
                    del recon_final_map
                except NameError:
                    pass  # recon_final_map was not defined

            # New: Log metrics
            if self.metrics_tracker is not None:
                self.metrics_tracker.log_loss_components(
                    epoch=epoch,
                    recon_loss=avg_recon,
                    temp_loss=avg_temp,
                    align_loss=avg_align,
                    total_loss=avg_total,
                    recon_norm=avg_recon_norm,
                    spec=avg_spec
                )
                # NEW: Log prediction loss if enabled
                if self.enable_prediction:
                    self.metrics_tracker.log_epoch(epoch, {'loss/prediction': avg_predictor})
                # Log relative errors
                for nt, err in rel_error_epoch.items():
                    self.metrics_tracker.log_epoch(epoch, {f'rel_error/{nt}': err})

            # 16) early stopping / LR 调整
            monitor_val = rel_error_epoch.get(monitor_nt, avg_recon)
            improved = monitor_val < best_rel - 1e-12
            if improved:
                best_rel = monitor_val
                best_epoch = epoch
                no_improve = 0
                best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                try:
                    best_opt_state = self.optimizer.state_dict()
                except Exception:
                    best_opt_state = None
                self.logger.info(f"[Monitor] new best {monitor_nt} rel={best_rel:.6f} at epoch={epoch}")
            else:
                no_improve += 1
                self.logger.info(
                    f"[Monitor] no_improve={no_improve}/{patience} "
                    f"(current {monitor_nt}={monitor_val:.6f} best={best_rel:.6f})"
                )
                if no_improve >= patience:
                    self.logger.info(
                        f"[Monitor] patience reached ({patience}). "
                        f"Rolling back to epoch={best_epoch} and shrinking LR by {lr_shrink}"
                    )
                    if best_model_state is not None:
                        self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
                    if best_opt_state is not None:
                        try:
                            self.optimizer.load_state_dict(best_opt_state)
                        except Exception:
                            self.logger.warning("[Monitor] Failed to restore optimizer state")
                    for g in self.optimizer.param_groups:
                        old_lr = float(g.get("lr", self.lr))
                        new_lr = max(old_lr * lr_shrink, min_lr)
                        g["lr"] = new_lr
                    no_improve = 0

            # Clear CUDA cache after each epoch to prevent memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if verbose:
                # Build log message with prediction loss if enabled
                log_msg = (
                    f"[Epoch {epoch:3d}] total={avg_total:.6f} align={avg_align:.6f} "
                    f"temp={avg_temp:.6f} recon={avg_recon:.6f} recon_norm={avg_recon_norm:.6f} "
                    f"spec={avg_spec:.6f}"
                )
                # NEW: Add predictor loss to logging
                if self.enable_prediction:
                    log_msg += f" pred={avg_predictor:.6f}"
                log_msg += f" time={time.time()-start:.2f}s"
                
                self.logger.info(log_msg)
                self.logger.info(f"[Epoch {epoch}] relative_error={rel_error_epoch}")
                
                # NEW: Log dynamic weighting info
                if self.enable_dynamic_weighting and self.dynamic_weighting is not None:
                    stage_info = self.dynamic_weighting.get_stage_info(epoch)
                    self.logger.info(
                        f"[Epoch {epoch}] Dynamic Weighting: stage={stage_info['stage']}, "
                        f"temperature={stage_info['temperature']:.3f}"
                    )
                    # Log weight statistics for each modality
                    for nt, weights in self.modality_weights.items():
                        if weights is not None and len(weights) > 0:
                            w_min = weights.min().item()
                            w_max = weights.max().item()
                            w_mean = weights.mean().item()
                            w_std = weights.std().item()
                            self.logger.info(
                                f"  {nt}: weight_range=[{w_min:.4f}, {w_max:.4f}], "
                                f"mean={w_mean:.4f}, std={w_std:.4f}"
                            )

        # Training completed - save metrics and print summary
        if self.metrics_tracker is not None:
            self.metrics_tracker.save_metrics()
            self.metrics_tracker.print_summary(last_n_epochs=min(10, epochs))

    # ----------------------------------------------------------------------
    # Scale freeze / unfreeze
    # ----------------------------------------------------------------------
    def _uniform_weights(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Create uniform weights for a tensor.
        
        Args:
            x_seq: Input sequence tensor
            
        Returns:
            Uniform weights [features]
        """
        num_features = x_seq.shape[-1] if x_seq.ndim >= 2 else 1
        return torch.ones(num_features, device=self.device) / num_features
    
    def _set_scale_requires_grad(self, flag: bool):
        """
        只对 scale/log_scale 参数的 requires_grad 做切换；
        - flag=False: 冻结这些 scale 参数；
        - flag=True: 恢复到初始化时记录的 requires_grad。
        """
        if flag:
            for name, p in self.model.named_parameters():
                if name in self._orig_requires_grad_map:
                    try:
                        p.requires_grad = self._orig_requires_grad_map[name]
                    except Exception:
                        pass
            self.logger.info("[Scale] restored requires_grad from original map")
            return

        changed = 0
        for name, p in self.model.named_parameters():
            if ("log_scale" in name) or ("scale_" in name and "scale_fixed" not in name):
                try:
                    p.requires_grad = False
                    changed += 1
                except Exception:
                    pass
        self.logger.info(f"[Scale] set requires_grad=False for {changed} scale/log_scale params")

    # ----------------------------------------------------------------------
    # Relative error & hist utils（保持你原有接口，只精简实现）
    # ----------------------------------------------------------------------
    def _compute_relative_error(self, recon_seq_scaled: Dict[str, torch.Tensor], data: HeteroData,
                                metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                                eps: float = 1e-8, debug: bool = False) -> Dict[str, float]:
        rel_error: Dict[str, float] = {}
        for nt in metadata[0]:
            if nt not in recon_seq_scaled:
                continue
            recon = recon_seq_scaled[nt]
            target = getattr(data[nt], "x_seq", None)
            if target is None:
                continue
            target_res = self._resample_time(target.to(self.device), recon.shape[1])
            Nr, Tr, Fr = recon.shape
            Nt, Tt, Ft = target_res.shape
            mN, mT, mF = min(Nr, Nt), min(Tr, Tt), min(Fr, Ft)
            if mN <= 0 or mT <= 0 or mF <= 0:
                continue
            r = recon[:mN, :mT, :mF]
            t = target_res[:mN, :mT, :mF]
            diff_norm = torch.norm(r - t)
            target_norm = torch.norm(t) + eps
            rel = diff_norm / target_norm
            rel_error[nt] = float(rel.item())
            if debug:
                self.logger.info(
                    f"[RelError:{nt}] recon mean={r.mean().item():.5f} std={r.std().item():.5f} "
                    f"target mean={t.mean().item():.5f} std={t.std().item():.5f} rel={rel.item():.5f}"
                )
        return rel_error

    def _resample_time(self, x: torch.Tensor, target_T: int) -> torch.Tensor:
        if x is None or x.numel() == 0:
            return x
        if x.ndim != 3:
            raise ValueError(f"_resample_time expects (N,T,F), got {tuple(x.shape)}")
        N, T_orig, F_dim = x.shape
        xp = x.permute(0, 2, 1)
        if T_orig > target_T:
            out = F_nn.adaptive_avg_pool1d(xp, target_T)
        elif T_orig < target_T:
            out = F_nn.interpolate(xp, size=target_T, mode="linear", align_corners=False)
        else:
            out = xp
        return out.permute(0, 2, 1).contiguous()

    # ----------------------------------------------------------------------
    # 保存 / 加载
    # ----------------------------------------------------------------------
    def save_model(self, path: Union[str, os.PathLike]):
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "aligner": self.aligner.state_dict() if hasattr(self, "aligner") else None,
            "optimizer": self.optimizer.state_dict() if hasattr(self, "optimizer") else None,
            "scheduler": self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
        }
        torch.save(payload, str(path))
        self.logger.info(f"[Save] saved {path}")

    def load_model(self, path: Union[str, os.PathLike]):
        ckpt = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(ckpt.get("model", ckpt))
        if "aligner" in ckpt and ckpt["aligner"] is not None and hasattr(self, "aligner"):
            try:
                self.aligner.load_state_dict(ckpt["aligner"])
            except Exception:
                self.logger.warning("[Load] Failed to load aligner state_dict")
        if "optimizer" in ckpt and ckpt["optimizer"] is not None and hasattr(self, "optimizer"):
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                self.logger.warning("[Load] Failed to load optimizer state_dict")
        if "scheduler" in ckpt and ckpt["scheduler"] is not None and hasattr(self, "scheduler"):
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                self.logger.warning("[Load] Failed to load scheduler state_dict")
        self.logger.info(f"[Load] loaded {path}")