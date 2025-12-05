import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch_geometric.data import HeteroData
from scipy.signal import correlate

from train.dynamic_hetero_gnn import DynamicHeteroGNN
from train.coder import GraphEncoder  # assuming GraphEncoder is defined in coder.py or elsewhere

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


class DynamicHeteroTrainer:
    """
    Trainer orchestrating model, losses and optimization.

    Enhancements included:
      - spec_loss (lowpass MSE) to encourage low-frequency trend matching
      - optional shift-invariant normalized-space loss (soft-min over shifts)
      - optional auto_align heuristic (integer lag estimation + roll) to fix large static time shifts
      - decoder-only warmup & scale-only fine-tune support
    """

    def __init__(
        self,
        hetero_data,
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
        # SPEC/lowpass loss defaults:
        spec_loss_weight: float = 0.0,
        spec_kernel_size: int = 11,
        # shift-invariant normalized-space loss defaults:
        shift_invariant_range: int = 0,       # e.g., 5 -> allow shifts -5..+5 frames
        shift_invariant_temp: float = 1.0,    # softmin temperature
        # auto-align heuristic (integer shift estimation)
        auto_align: bool = False,             # enable integer-lag auto alignment
        auto_align_max_lag: int = 120,        # maximum lag (frames) to search for
        auto_align_scope: str = "warmup",     # "warmup" or "always"
    ):
        # logger setup
        self.logger = logging.getLogger("DynamicHeteroTrainer")
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.logger.addHandler(ch)
        self.logger.setLevel(logging.INFO if debug else logging.WARNING)

        # device & config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hetero_data = hetero_data
        self.input_dims = input_dims or {}
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.num_epochs = num_epochs
        self.recon_weight = recon_weight
        self.recon_norm_weight = recon_norm_weight
        self.recon_corr_weight = recon_corr_weight
        self.temp_weight = temp_weight
        self.align_weight = align_weight
        self.temporal_T = temporal_T
        self.spatial_T = spatial_T
        self.use_amp = use_amp and torch.cuda.is_available()
        self.weight_decay = weight_decay
        self.debug = debug
        self.grad_clip = grad_clip

        # warmup & scale handling
        self.warmup_epochs = warmup_epochs
        self.freeze_scale_during_warmup = freeze_scale_during_warmup
        self.scale_lr_mul = scale_lr_mul
        self.feature_lr_mul = feature_lr_mul

        # batch rescale utility
        self.batch_rescale_fn = batch_rescale_fn
        self.batch_rescale_cfg = batch_rescale_cfg or {"enable": False, "only": [], "warmup_epochs": 0}

        # recon feature variance regularizer
        self.recon_feat_var_weight = recon_feat_var_weight

        # scale-only fine-tune options
        self.scale_only_epochs = scale_only_epochs
        self.scale_only_lr_mul = scale_only_lr_mul

        # SPEC lowpass loss options
        self.spec_loss_weight = spec_loss_weight
        self.spec_kernel_size = spec_kernel_size

        # SHIFT-INVARIANT options
        self.shift_invariant_range = shift_invariant_range
        self.shift_invariant_temp = max(1e-6, float(shift_invariant_temp))

        # AUTO ALIGN heuristic options
        self.auto_align = auto_align
        self.auto_align_max_lag = int(auto_align_max_lag)
        self.auto_align_scope = auto_align_scope  # "warmup" or "always"
        # cache for computed lags (reset each epoch)
        self._auto_align_cache: Dict[str, int] = {}

        # Graph encoder placeholder
        try:
            self.graph_encoder = GraphEncoder()
        except Exception:
            self.graph_encoder = getattr(self, "graph_encoder", None)

        # flatten input hetero_data into list
        self.data_list = self._flatten_data(self.hetero_data)
        if len(self.data_list) == 0:
            raise ValueError("No hetero_data provided to trainer")

        # metadata inference
        sample = self.data_list[0]
        if hasattr(sample, "metadata"):
            self.metadata = sample.metadata()
        else:
            node_types = list(sample.keys())
            self.metadata = (node_types, [])

        # infer input dims if not provided
        if not self.input_dims:
            inferred = {}
            for nt in self.metadata[0]:
                try:
                    x_seq = getattr(sample[nt], "x_seq", None)
                    if x_seq is not None and len(x_seq.shape) == 3:
                        inferred[nt] = int(x_seq.shape[2])
                    else:
                        inferred[nt] = 1
                except Exception:
                    inferred[nt] = 1
            self.input_dims = inferred
            self.logger.info(f"[Init] inferred input dims: {self.input_dims}")

        # instantiate model and aligner
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

        # aligner: try LatentAligner if available else fallback
        if LatentAligner is not None:
            try:
                self.aligner = LatentAligner(hidden_dim=self.hidden_dim, mode="nodewise", lambda_align=1.0, temperature=0.3).to(self.device)
            except Exception:
                try:
                    from train.aligner import TemporalCrossAligner
                    self.aligner = TemporalCrossAligner(hidden_dim=self.hidden_dim, dropout=self.dropout).to(self.device)
                except Exception:
                    class _DummyAligner(nn.Module):
                        def forward(self, a, b): return torch.tensor(0.0, device=next(self.parameters()).device)
                    self.aligner = _DummyAligner()
        else:
            try:
                from train.aligner import TemporalCrossAligner
                self.aligner = TemporalCrossAligner(hidden_dim=self.hidden_dim, dropout=self.dropout).to(self.device)
            except Exception:
                class _DummyAligner(nn.Module):
                    def forward(self, a, b): return torch.tensor(0.0, device=next(self.parameters()).device)
                self.aligner = _DummyAligner()

        # ensure lazy params created: run one dummy forward
        try:
            sample_graph = self.data_list[0].to(self.device)
            with torch.no_grad():
                enc_out = self.graph_encoder(sample_graph)
                _, _, _, _, edge_index_dict = enc_out
                _ = self.model(sample_graph, edge_index_dict=edge_index_dict)
        except Exception as e:
            self.logger.debug(f"[Init] dummy forward failed or skipped: {e}")

        # Build optimizer parameter groups
        base_params = []
        feature_params = []
        scale_params = []
        aligner_params = list(self.aligner.parameters())

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # Group by substring to detect decoder params robustly
            if "feature_decoders" in name:
                feature_params.append(p)
                continue
            # scale params detection
            if ("log_scale" in name) or ("scale_" in name and "scale_fixed" not in name):
                scale_params.append(p)
                continue
            base_params.append(p)

        param_groups = []
        if len(base_params) > 0:
            param_groups.append({"params": base_params})
        if len(feature_params) > 0:
            param_groups.append({"params": feature_params, "lr": self.lr * float(self.feature_lr_mul)})
        if len(scale_params) > 0:
            param_groups.append({"params": scale_params, "lr": self.lr * float(self.scale_lr_mul)})
        if len(aligner_params) > 0:
            param_groups.append({"params": aligner_params})

        self.optimizer = torch.optim.Adam(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.7)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # bookkeeping
        self.loss_log = {"total": [], "align": [], "temp": [], "recon": [], "recon_norm": [], "spec": []}
        self.diagnostic_dir = None
        self.use_batch_rescale = bool(self.batch_rescale_cfg.get("enable", False) and self.batch_rescale_fn is not None)

        self.logger.info(f"[Init] Trainer initialized on device={self.device}. model params={sum(p.numel() for p in self.model.parameters())}")
        self.logger.info(f"[Init] Optimizer param groups: base={len(base_params)}, feature={len(feature_params)}, scale={len(scale_params)}, aligner={len(aligner_params)}")

    # -------------------------
    # Utilities
    # -------------------------
    def _flatten_data(self, data) -> List[HeteroData]:
        if data is None:
            return []
        if isinstance(data, HeteroData):
            return [data]
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            flat = []
            for v in data.values():
                flat.extend(v if isinstance(v, list) else [v])
            return flat
        return [data]

    def _temporal_prediction_loss(self, proj_seq: torch.Tensor, nt: str) -> torch.Tensor:
        device = self.device
        if proj_seq is None or proj_seq.numel() == 0:
            return torch.tensor(0.0, device=device)
        if proj_seq.shape[1] < 2:
            return torch.tensor(0.0, device=device)
        if nt not in self.model.grus:
            return torch.tensor(0.0, device=device)
        gru = self.model.grus[nt]
        inp = proj_seq[:, :-1, :]
        targ = proj_seq[:, 1:, :]
        out, _ = gru(inp)
        time_loss = F_nn.mse_loss(out, targ)
        # frequency-aware loss (optional)
        try:
            pred = out.permute(0, 2, 1)
            targ_f = targ.permute(0, 2, 1)
            pred_fft = torch.fft.rfft(pred, dim=-1)
            targ_fft = torch.fft.rfft(targ_f, dim=-1)
            pred_mag = pred_fft.abs()
            targ_mag = targ_fft.abs()
            n_bins = pred_mag.size(-1)
            freq_idx = torch.arange(n_bins, device=device).float()
            weights = (freq_idx / (n_bins - 1)).view(1, 1, -1)
            freq_loss = F_nn.mse_loss(pred_mag * weights, targ_mag * weights)
        except Exception:
            freq_loss = torch.tensor(0.0, device=device)
        a_t = getattr(self, "temporal_loss_alpha", 1.0)
        a_f = getattr(self, "temporal_loss_beta", 0.5)
        return a_t * time_loss + a_f * freq_loss

    def _estimate_best_lag_cpu(self, ref: np.ndarray, query: np.ndarray, max_lag: int) -> int:
        """
        Compute best integer lag (in frames) between ref and query using cross-correlation on CPU.
        ref, query: 1D numpy arrays of same length T
        returns best_lag in [-max_lag, +max_lag]
        """
        T = len(ref)
        if T <= 0:
            return 0
        corr_full = correlate(ref, query, mode='full')
        lags = np.arange(-T + 1, T)
        mask = (lags >= -max_lag) & (lags <= max_lag)
        if not mask.any():
            return 0
        corr_masked = corr_full.copy()
        corr_masked[~mask] = -1e18
        best_idx = int(np.argmax(corr_masked))
        best_lag = int(lags[best_idx])
        return best_lag

    def _align_tensor_by_lag(self, tensor: torch.Tensor, lag: int, fill_value: float = 0.0) -> torch.Tensor:
        """
        Align tensor by integer lag via roll and zero-fill rolled-in regions.
        tensor: (B, T, F)
        lag: int, positive means recon lags target -> shift left by lag (roll -lag)
        """
        if lag == 0:
            return tensor
        B, T, F = tensor.shape
        rolled = torch.roll(tensor, shifts=-lag, dims=1)
        if lag > 0:
            # zero last lag frames
            rolled[:, T - lag : T, :] = fill_value
        else:
            # negative lag: zero first -lag frames
            rolled[:, : -lag, :] = fill_value
        return rolled

    def _resample_time(self, x: torch.Tensor, target_T: int) -> torch.Tensor:
        if x is None or x.numel() == 0:
            return x
        N, T_orig, F_dim = x.shape
        xp = x.permute(0, 2, 1)
        if T_orig > target_T:
            out = F_nn.adaptive_avg_pool1d(xp, target_T)
        elif T_orig < target_T:
            out = F_nn.interpolate(xp, size=target_T, mode="linear", align_corners=False)
        else:
            out = xp
        return out.permute(0, 2, 1).contiguous()

    # -------------------------
    # Train loop
    # -------------------------
    def train(self, num_epochs: Optional[int] = None, verbose: bool = True):
        # defaults
        if not hasattr(self, "recon_feat_var_weight"):
            self.recon_feat_var_weight = 0.0
        if not hasattr(self, "batch_rescale_cfg"):
            self.batch_rescale_cfg = {"enable": False, "warmup_epochs": 0}
        if "warmup_epochs" not in self.batch_rescale_cfg:
            self.batch_rescale_cfg["warmup_epochs"] = 0
        if not hasattr(self, "scale_only_epochs"):
            self.scale_only_epochs = 0
        if not hasattr(self, "scale_only_lr_mul"):
            self.scale_only_lr_mul = 20.0

        epochs = num_epochs or self.num_epochs
        self.model.train()
        self.aligner.train()

        # warmup freeze
        if getattr(self, "warmup_epochs", 0) > 0 and getattr(self, "freeze_scale_during_warmup", False):
            self._set_scale_requires_grad(False)
            self.logger.info(f"[Warmup] freezing scale params for {self.warmup_epochs} epochs")

        for epoch in range(1, epochs + 1):
            # reset per-epoch auto-align cache
            self._auto_align_cache = {}

            start = time.time()
            total_loss = total_align = total_temp = total_recon = total_recon_norm = total_spec = 0.0
            batches = 0

            # auto-disable batch_rescale after warmup_epochs
            if self.batch_rescale_cfg.get("enable", False) and epoch > int(self.batch_rescale_cfg.get("warmup_epochs", 0)):
                self.batch_rescale_cfg["enable"] = False
                self.logger.info(f"[BatchRescale] warmup_epochs passed ({self.batch_rescale_cfg.get('warmup_epochs')}), disabling batch_rescale")

            # unfreeze after warmup
            if epoch == getattr(self, "warmup_epochs", 0) + 1 and getattr(self, "freeze_scale_during_warmup", False):
                self._set_scale_requires_grad(True)
                self.logger.info("[Warmup] unfreezing scale params, resuming full training")

            for data_idx, data in enumerate(self.data_list):
                data = data.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                # encoder
                encoder_out = self.graph_encoder(data)
                x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict = encoder_out

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(data, edge_index_dict=edge_index_dict)
                    if len(outputs) >= 7:
                        z_dict, gru_seq_dict, proj_seq_dict, recon_seq_dict, recon_seq_scaled, global_seq, recon_feature_dict = outputs[:7]
                    else:
                        z_dict, gru_seq_dict, proj_seq_dict, recon_seq_dict, recon_seq_scaled, global_seq = outputs[:6]
                        recon_feature_dict = None
                    # ---- START AUTO-ALIGN APPLY (paste immediately after outputs unpacking) ----
                    # If auto_align enabled and in scope, estimate integer lag and ALIGN the model outputs
                    try:
                        if self.auto_align and (self.auto_align_scope == "always" or epoch <= getattr(self, "warmup_epochs", 0)):
                            # do per-node auto-align (we typically only want to align fmri)
                            for nt in self.metadata[0]:
                                if nt not in recon_feature_dict:
                                    continue
                                if nt != "fmri":
                                    continue  # limit to fmri for now (avoid messing eeg)
                                # if cached for this epoch/sample, reuse
                                cache_key = f"{data_idx}_{nt}"
                                if cache_key in self._auto_align_cache:
                                    best_lag = int(self._auto_align_cache[cache_key])
                                else:
                                    # build 1D references (mean across batch & features)
                                    try:
                                        target_raw = getattr(data[nt], "x_seq", None)
                                        if target_raw is None:
                                            continue
                                        # resample target to same length as recon_feature time
                                        t_res = self._resample_time(target_raw.to(self.device), recon_feature_dict[nt].shape[1])
                                        ref = t_res.detach().cpu().numpy().mean(axis=(0,2))  # shape (T,)
                                        qry = recon_feature_dict[nt].detach().cpu().numpy().mean(axis=(0,2))
                                        best_lag = int(self._estimate_best_lag_cpu(ref, qry, max_lag=self.auto_align_max_lag))
                                    except Exception as e:
                                        if self.debug:
                                            self.logger.warning(f"[AutoAlign] estimation failed for nt={nt}: {e}")
                                        best_lag = 0
                                    # cache per data_idx & nt for this epoch
                                    self._auto_align_cache[cache_key] = int(best_lag)

                                if best_lag != 0:
                                    # apply same integer roll/zero-fill to recon_feature_dict, recon_seq_dict and recon_seq_scaled
                                    try:
                                        recon_feature_dict[nt] = self._align_tensor_by_lag(recon_feature_dict[nt], best_lag, fill_value=0.0)
                                    except Exception:
                                        pass
                                    try:
                                        if nt in recon_seq_dict:
                                            recon_seq_dict[nt] = self._align_tensor_by_lag(recon_seq_dict[nt], best_lag, fill_value=0.0)
                                        if nt in recon_seq_scaled:
                                            recon_seq_scaled[nt] = self._align_tensor_by_lag(recon_seq_scaled[nt], best_lag, fill_value=0.0)
                                    except Exception:
                                        pass
                                    if self.debug:
                                        self.logger.info(f"[AutoAlign] applied lag={best_lag} to nt={nt} (data_idx={data_idx}, epoch={epoch})")
                    # swallow any issues to avoid crashing training
                    except Exception as _e:
                        if self.debug:
                            self.logger.warning(f"[AutoAlign] unexpected error: {_e}")
                    # ---- END AUTO-ALIGN APPLY ----
                    # align
                    a_z_f = z_dict.get("fmri", None) if isinstance(z_dict, dict) else None
                    a_z_e = z_dict.get("eeg", None) if isinstance(z_dict, dict) else None
                    align_loss = self.aligner(a_z_f, a_z_e) if (a_z_f is not None and a_z_e is not None) else torch.tensor(0.0, device=self.device)

                    # temporal
                    temp_loss = torch.tensor(0.0, device=self.device)
                    for nt in self.metadata[0]:
                        seq = proj_seq_dict.get(nt, None)
                        temp_loss = temp_loss + self._temporal_prediction_loss(seq, nt)

                    # batch rescale
                    recon_to_use = recon_seq_dict
                    batch_alphas = {}
                    if self.batch_rescale_cfg.get("enable", False) and self.batch_rescale_fn is not None:
                        recon_to_use = {}
                        for nt, recon in recon_seq_dict.items():
                            apply_here = (not self.batch_rescale_cfg.get("only")) or (nt in self.batch_rescale_cfg.get("only", []))
                            if apply_here:
                                t = getattr(data[nt], "x_seq")
                                if t is None:
                                    alpha = 1.0
                                else:
                                    r_det = recon.detach()
                                    t_res = self._resample_time(t.to(self.device), r_det.shape[1])
                                    alpha = self.batch_rescale_fn(r_det, t_res, self.batch_rescale_cfg)
                                batch_alphas[nt] = alpha
                                recon_to_use[nt] = recon * (alpha if isinstance(alpha, torch.Tensor) else float(alpha))
                            else:
                                recon_to_use[nt] = recon

                    # recon loss
                    recon_loss = torch.tensor(0.0, device=self.device)
                    recon_losses_per_nt = {}
                    for nt in self.metadata[0]:
                        recon = recon_to_use.get(nt, recon_seq_dict.get(nt))
                        target = getattr(data[nt], "x_seq")
                        target_res = self._resample_time(target.to(self.device), recon.shape[1])

                        Nr, Tr, Fr = recon.shape
                        Nt, Tt, Ft = target_res.shape
                        min_N, min_T, min_F = min(Nr, Nt), min(Tr, Tt), min(Fr, Ft)

                        recon_crop = recon[:min_N, :min_T, :min_F]
                        target_crop = target_res[:min_N, :min_T, :min_F]

                        l_nt = F_nn.mse_loss(recon_crop, target_crop)
                        recon_losses_per_nt[nt] = float(l_nt.detach().cpu())
                        recon_loss = recon_loss + l_nt

                    # recon_norm_loss (normalized-space) + spec lowpass loss per nt
                    recon_norm_loss = torch.tensor(0.0, device=self.device)
                    spec_loss_total = torch.tensor(0.0, device=self.device)
                    if recon_feature_dict is not None and getattr(self, "recon_norm_weight", 0.0) and self.recon_norm_weight > 0:
                        for nt in self.metadata[0]:
                            if nt not in recon_feature_dict:
                                continue
                            recon_feat = recon_feature_dict[nt]
                            stats = stats_dict.get(nt, {"mean": None, "std": None})
                            mean = stats.get("mean", None)
                            std = stats.get("std", None)
                            if mean is None or std is None:
                                continue
                            target = getattr(data[nt], "x_seq")
                            target_res = self._resample_time(target.to(self.device), recon_feat.shape[1])

                            Nr, Tr, Fr = recon_feat.shape
                            Nt, Tt, Ft = target_res.shape
                            min_N, min_T, min_F = min(Nr, Nt), min(Tr, Tt), min(Fr, Ft)

                            recon_feat_crop = recon_feat[:min_N, :min_T, :min_F]
                            mean_expand = mean.expand(-1, recon_feat.shape[1], -1)[:min_N, :min_T, :min_F]
                            std_expand = std.expand(-1, recon_feat.shape[1], -1)[:min_N, :min_T, :min_F]
                            target_norm = (target_res[:min_N, :min_T, :min_F] - mean_expand) / (std_expand + 1e-8)

                            # --- AUTO ALIGN HEURISTIC (integer lag) ---
                            if self.auto_align and (self.auto_align_scope == "always" or epoch <= getattr(self, "warmup_epochs", 0)):
                                # only align fmri by default to avoid unwanted effects on eeg
                                if nt == "fmri":
                                    try:
                                        if nt in self._auto_align_cache:
                                            best_lag = self._auto_align_cache[nt]
                                        else:
                                            # compute mean over batch and features as 1D signals
                                            ref = target_norm.detach().cpu().numpy().mean(axis=(0,2))
                                            qry = recon_feat_crop.detach().cpu().numpy().mean(axis=(0,2))
                                            best_lag = self._estimate_best_lag_cpu(ref, qry, max_lag=self.auto_align_max_lag)
                                            self._auto_align_cache[nt] = int(best_lag)
                                        if best_lag != 0:
                                            recon_feat_crop = self._align_tensor_by_lag(recon_feat_crop, best_lag, fill_value=0.0)
                                            if self.debug:
                                                self.logger.info(f"[AutoAlign] nt={nt} applied best_lag={best_lag} (epoch={epoch})")
                                    except Exception as e:
                                        if self.debug:
                                            self.logger.warning(f"[AutoAlign] failed for nt={nt}: {e}")
                            # --- END AUTO ALIGN ---

                            # SHIFT-INVARIANT normalized-space handling (soft-min over small shifts)
                            if getattr(self, "shift_invariant_range", 0) and int(self.shift_invariant_range) > 0:
                                sR = int(self.shift_invariant_range)
                                shifts = list(range(-sR, sR + 1))
                                mse_shifts = []
                                for s in shifts:
                                    if s == 0:
                                        targ_s = target_norm
                                    else:
                                        targ_s = torch.roll(target_norm, shifts=s, dims=1)
                                    try:
                                        mse_s = F_nn.mse_loss(recon_feat_crop, targ_s)
                                    except Exception:
                                        mse_s = torch.tensor(float("nan"), device=self.device)
                                    mse_shifts.append(mse_s)
                                mse_stack = torch.stack([m if torch.isfinite(m) else torch.tensor(1e9, device=self.device) for m in mse_shifts])
                                temp = max(1e-6, float(getattr(self, "shift_invariant_temp", 1.0)))
                                weights = torch.softmax((-mse_stack / temp), dim=0)
                                combined_mse = torch.sum(weights * mse_stack)
                                recon_norm_loss = recon_norm_loss + combined_mse

                                # SPEC/lowpass: compute weighted spec over same shifts for consistency
                                if getattr(self, "spec_loss_weight", 0.0) and self.spec_loss_weight > 0:
                                    spec_shifts = []
                                    for idx, s in enumerate(shifts):
                                        if s == 0:
                                            targ_s = target_norm
                                        else:
                                            targ_s = torch.roll(target_norm, shifts=s, dims=1)
                                        try:
                                            spec_shifts.append(lowpass_mse_loss(recon_feat_crop, targ_s, kernel_size=self.spec_kernel_size))
                                        except Exception:
                                            spec_shifts.append(torch.tensor(0.0, device=self.device))
                                    spec_stack = torch.stack(spec_shifts)
                                    spec_weighted = torch.sum(weights * spec_stack)
                                    spec_loss_total = spec_loss_total + spec_weighted
                            else:
                                # direct normalized MSE
                                recon_norm_loss = recon_norm_loss + F_nn.mse_loss(recon_feat_crop, target_norm)

                                # variance regularizer
                                if getattr(self, "recon_feat_var_weight", 0.0) and self.recon_feat_var_weight > 0:
                                    rf_flat = recon_feat_crop.reshape(recon_feat_crop.shape[0], -1)
                                    rf_std = rf_flat.std(dim=1).mean()
                                    recon_norm_loss = recon_norm_loss + self.recon_feat_var_weight * F_nn.mse_loss(rf_std, torch.tensor(1.0, device=rf_std.device))

                                # SPEC / lowpass loss
                                if getattr(self, "spec_loss_weight", 0.0) and self.spec_loss_weight > 0:
                                    try:
                                        spec_loss_nt = lowpass_mse_loss(recon_feat_crop, target_norm, kernel_size=self.spec_kernel_size)
                                        spec_loss_total = spec_loss_total + spec_loss_nt
                                    except Exception as e:
                                        if self.debug:
                                            self.logger.debug(f"[SpecLoss] failed for nt={nt}: {e}")

                    # recon_corr_loss (differentiable Pearson avg loss)
                    recon_corr_loss = torch.tensor(0.0, device=self.device)
                    if recon_feature_dict is not None and getattr(self, "recon_corr_weight", 0.0) and self.recon_corr_weight > 0:
                        r_list = []
                        eps = 1e-8
                        for nt in self.metadata[0]:
                            if nt not in recon_feature_dict:
                                continue
                            recon_feat = recon_feature_dict[nt]
                            stats = stats_dict.get(nt, {"mean": None, "std": None})
                            mean = stats.get("mean", None)
                            std = stats.get("std", None)
                            if mean is None or std is None:
                                continue
                            target = getattr(data[nt], "x_seq")
                            target_res = self._resample_time(target.to(self.device), recon_feat.shape[1])

                            Nr, Tr, Fr = recon_feat.shape
                            Nt, Tt, Ft = target_res.shape
                            min_N, min_T, min_F = min(Nr, Nt), min(Tr, Tt), min(Fr, Ft)

                            rf = recon_feat[:min_N, :min_T, :min_F].reshape(-1, min_F)  # (S, F)
                            mean_expand = mean.expand(-1, recon_feat.shape[1], -1)[:min_N, :min_T, :min_F]
                            std_expand = std.expand(-1, recon_feat.shape[1], -1)[:min_N, :min_T, :min_F]
                            tnorm = ((target_res[:min_N, :min_T, :min_F] - mean_expand) / (std_expand + eps)).reshape(-1, min_F)
                            rf_center = rf - rf.mean(dim=0, keepdim=True)
                            t_center = tnorm - tnorm.mean(dim=0, keepdim=True)
                            num = (rf_center * t_center).sum(dim=0)
                            den = torch.sqrt((rf_center ** 2).sum(dim=0) * (t_center ** 2).sum(dim=0) + eps)
                            r_feat = num / (den + eps)
                            r_mean = torch.mean(r_feat)
                            r_list.append(r_mean)
                        if len(r_list) > 0:
                            mean_r = torch.stack(r_list).mean()
                            recon_corr_loss = 1.0 - mean_r
                        else:
                            recon_corr_loss = torch.tensor(0.0, device=self.device)

                    # total
                    loss = self.align_weight * align_loss + self.temp_weight * temp_loss + self.recon_weight * recon_loss
                    if getattr(self, "recon_norm_weight", 0.0) and self.recon_norm_weight > 0:
                        loss = loss + self.recon_norm_weight * recon_norm_loss
                    if getattr(self, "recon_corr_weight", 0.0) and self.recon_corr_weight > 0:
                        loss = loss + self.recon_corr_weight * recon_corr_loss
                    # add spec_loss if configured
                    if getattr(self, "spec_loss_weight", 0.0) and self.spec_loss_weight > 0:
                        loss = loss + float(self.spec_loss_weight) * spec_loss_total

                # backward & step
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        try:
                            self.scaler.unscale_(self.optimizer)
                        except Exception:
                            pass
                        torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.aligner.parameters()), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.aligner.parameters()), self.grad_clip)
                    self.optimizer.step()

                # stats
                total_loss += float(loss.detach().cpu())
                total_align += float(align_loss.detach().cpu())
                total_temp += float(temp_loss.detach().cpu())
                total_recon += float(recon_loss.detach().cpu())
                total_recon_norm += float(recon_norm_loss.detach().cpu()) if isinstance(recon_norm_loss, torch.Tensor) else 0.0
                total_spec += float(spec_loss_total.detach().cpu()) if isinstance(spec_loss_total, torch.Tensor) else 0.0
                batches += 1

                if data_idx == 0:
                    self.logger.info(f"[Train] epoch={epoch} batch=0 recon_losses={recon_losses_per_nt}")
                    if batch_alphas:
                        for nt, a in batch_alphas.items():
                            if isinstance(a, torch.Tensor):
                                try:
                                    self.logger.info(f"[Train] batch alpha {nt} sample first5: {a.view(-1)[:5].cpu().tolist()}")
                                except Exception:
                                    self.logger.info(f"[Train] batch alpha {nt}: tensor")
                            else:
                                self.logger.info(f"[Train] batch alpha {nt}: {a}")

            # epoch end
            avg_total = total_loss / batches
            avg_align = total_align / batches
            avg_temp = total_temp / batches
            avg_recon = total_recon / batches
            avg_recon_norm = total_recon_norm / batches
            avg_spec = total_spec / batches

            self.loss_log.setdefault("total", []).append(avg_total)
            self.loss_log.setdefault("align", []).append(avg_align)
            self.loss_log.setdefault("temp", []).append(avg_temp)
            self.loss_log.setdefault("recon", []).append(avg_recon)
            self.loss_log.setdefault("recon_norm", []).append(avg_recon_norm)
            self.loss_log.setdefault("spec", []).append(avg_spec)

            self.scheduler.step()

            # epoch diagnostics
            rel_error_epoch = {}
            for nt in self.metadata[0]:
                recon_final = recon_seq_dict[nt]
                rel_error_epoch[nt] = self._compute_relative_error({nt: recon_final}, data, self.metadata)[nt]
                if verbose:
                    try:
                        self.logger.info(f"[Debug:{nt}] recon mean={recon_final.mean():.5f}, std={recon_final.std():.5f}")
                    except Exception:
                        pass

            # scale diagnostics
            try:
                scale_info = {}
                for nt in self.metadata[0]:
                    val = None
                    grad_norm = None
                    dec = getattr(self.model, "denorm_decoders", None)
                    if dec is not None and hasattr(dec, "get_scale"):
                        try:
                            scale_tensor = dec.get_scale(nt)
                            val = float(scale_tensor.detach().mean().cpu())
                            if hasattr(dec, f"log_scale_{nt}"):
                                p = getattr(dec, f"log_scale_{nt}")
                                grad_norm = float(p.grad.detach().norm().cpu()) if (p.grad is not None) else None
                            elif hasattr(dec, f"scale_{nt}"):
                                p = getattr(dec, f"scale_{nt}")
                                grad_norm = float(p.grad.detach().norm().cpu()) if (p.grad is not None) else None
                        except Exception:
                            val = None
                            grad_norm = None
                    else:
                        for name, p in self.model.named_parameters():
                            if nt in name and ("scale" in name or "log_scale" in name):
                                try:
                                    if "log_scale" in name:
                                        val = float(torch.exp(p.detach()).mean().cpu())
                                    else:
                                        val = float(p.detach().mean().cpu())
                                    grad_norm = float(p.grad.detach().norm().cpu()) if (p.grad is not None) else None
                                    break
                                except Exception:
                                    continue
                    scale_info[nt] = {"scale_mean": val, "scale_grad_norm": grad_norm}
                self.logger.info(f"[ScaleDiag epoch={epoch}] " + ", ".join([f"{k}: mean={v['scale_mean']:.4f} grad_norm={v['scale_grad_norm']}" for k, v in scale_info.items()]))
            except Exception as e:
                self.logger.warning(f"[ScaleDiag] failed: {e}")

            if verbose:
                self.logger.info(
                    f"[Epoch {epoch:3d}] total={avg_total:.6f} align={avg_align:.6f} "
                    f"temp={avg_temp:.6f} recon={avg_recon:.6f} recon_norm={avg_recon_norm:.6f} spec={avg_spec:.6f} time={time.time()-start:.2f}s"
                )
                self.logger.info(f"[Epoch {epoch}] relative_error={rel_error_epoch}")
                self._log_reconstruction_histogram(recon_seq_dict, self.metadata)

        # optional scale-only fine-tune
        if getattr(self, "scale_only_epochs", 0) > 0:
            scale_params = []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if ("scale" in name or "log_scale" in name):
                    scale_params.append(p)
            if len(scale_params) == 0:
                self.logger.info("[ScaleOnly] no scale/log_scale parameters found; skipping scale-only phase.")
            else:
                self.logger.info(f"[ScaleOnly] starting scale-only fine-tune for {self.scale_only_epochs} epochs with lr_mul={self.scale_only_lr_mul}")
                opt_scale = torch.optim.Adam([{"params": scale_params, "lr": float(self.lr * self.scale_only_lr_mul)}], weight_decay=self.weight_decay)
                for so in range(1, self.scale_only_epochs + 1):
                    start_so = time.time()
                    total_so_loss = 0.0
                    for data in self.data_list:
                        data = data.to(self.device)
                        opt_scale.zero_grad(set_to_none=True)
                        encoder_out = self.graph_encoder(data)
                        x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict = encoder_out
                        with torch.cuda.amp.autocast(enabled=self.use_amp):
                            outputs = self.model(data, edge_index_dict=edge_index_dict)
                            if len(outputs) >= 7:
                                _, _, _, recon_seq_dict, _, _, recon_feature_dict = outputs[:7]
                            else:
                                _, _, _, recon_seq_dict, _, _ = outputs[:6]
                                recon_feature_dict = None
                            recon_loss = torch.tensor(0.0, device=self.device)
                            for nt in self.metadata[0]:
                                recon = recon_seq_dict[nt]
                                target = getattr(data[nt], "x_seq")
                                target_res = self._resample_time(target.to(self.device), recon.shape[1])
                                Nr, Tr, Fr = recon.shape
                                Nt, Tt, Ft = target_res.shape
                                min_N, min_T, min_F = min(Nr, Nt), min(Tr, Tt), min(Fr, Ft)
                                recon_crop = recon[:min_N, :min_T, :min_F]
                                target_crop = target_res[:min_N, :min_T, :min_F]
                                recon_loss = recon_loss + F_nn.mse_loss(recon_crop, target_crop)
                            loss = recon_loss
                        if self.use_amp:
                            self.scaler.scale(loss).backward()
                            self.scaler.step(opt_scale)
                            self.scaler.update()
                        else:
                            loss.backward()
                            opt_scale.step()
                        total_so_loss += float(loss.detach().cpu())
                    self.logger.info(f"[ScaleOnly] epoch={so} avg_loss={total_so_loss/len(self.data_list):.6f} time={time.time()-start_so:.2f}s")
                # rebuild full optimizer
                try:
                    base_params = []
                    scale_params = []
                    for name, p in self.model.named_parameters():
                        if not p.requires_grad:
                            continue
                        if ("scale" in name or "log_scale" in name):
                            scale_params.append(p)
                        else:
                            base_params.append(p)
                    param_groups = [{"params": base_params}, {"params": scale_params, "lr": self.lr * float(getattr(self, "scale_lr_mul", 1.0))}, {"params": list(self.aligner.parameters())}]
                    self.optimizer = torch.optim.Adam(param_groups, lr=self.lr, weight_decay=self.weight_decay)
                    self.logger.info("[ScaleOnly] rebuilt full optimizer after scale-only phase.")
                except Exception as e:
                    self.logger.warning(f"[ScaleOnly] failed to rebuild full optimizer: {e}")

    # -------------------------
    # Helpers
    # -------------------------
    def _set_scale_requires_grad(self, flag: bool):
        changed = 0
        for name, p in self.model.named_parameters():
            if "scale" in name or "log_scale" in name:
                p.requires_grad = flag
                changed += 1
        self.logger.info(f"[Scale] set requires_grad={flag} for {changed} params")

    def _compute_relative_error(self, recon_seq_scaled, data, metadata, eps=1e-8, debug=False):
        rel_error = {}
        for nt in metadata[0]:
            if nt not in recon_seq_scaled:
                continue
            recon = recon_seq_scaled[nt]
            target = getattr(data[nt], "x_seq")
            target_res = self._resample_time(target.to(self.device), recon.shape[1])
            Nr, Tr, Fr = recon.shape
            Nt, Tt, Ft = target_res.shape
            mN, mT, mF = min(Nr, Nt), min(Tr, Tt), min(Fr, Ft)
            r = recon[:mN, :mT, :mF]
            t = target_res[:mN, :mT, :mF]
            diff_norm = torch.norm(r - t)
            target_norm = torch.norm(t) + eps
            rel = diff_norm / target_norm
            rel_error[nt] = rel.item()
            if debug:
                r_mean, r_std = r.mean().item(), r.std().item()
                t_mean, t_std = t.mean().item(), t.std().item()
                print(f"[RelError:{nt}] recon mean={r_mean:.5f} std={r_std:.5f} target mean={t_mean:.5f} std={t_std:.5f} rel_error={rel.item():.5f}")
        return rel_error

    def _log_reconstruction_histogram(self, recon_seq_scaled, metadata, bins=10):
        for nt in metadata[0]:
            if nt not in recon_seq_scaled:
                continue
            x = recon_seq_scaled[nt].detach().flatten().cpu().numpy()
            if x.size == 0:
                continue
            hist, bin_edges = np.histogram(x, bins=bins)
            self.logger.info(
                f"[Hist] {nt} bins={bins} | min={x.min():.4f} max={x.max():.4f} mean={x.mean():.4f} std={x.std():.4f}"
            )

    def save_model(self, path: Union[str, os.PathLike]):
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "aligner": self.aligner.state_dict(),
            "optimizer": self.optimizer.state_dict() if hasattr(self, "optimizer") else None,
            "scheduler": self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
        }
        torch.save(payload, str(path))
        self.logger.info(f"[Save] saved {path}")

    def load_model(self, path: Union[str, os.PathLike]):
        ckpt = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        try:
            if "aligner" in ckpt and ckpt["aligner"] is not None:
                self.aligner.load_state_dict(ckpt["aligner"])
        except Exception:
            self.logger.warning("[Load] aligner state incompatible")
        try:
            if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                self.optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            self.logger.warning("[Load] optimizer state incompatible")
        self.logger.info(f"[Load] loaded {path}")