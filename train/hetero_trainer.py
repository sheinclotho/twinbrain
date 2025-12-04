# Clean, responsibility-separated DynamicHeteroTrainer
# - Trainer orchestrates training only.
# - Heavy modality-specific logic (rescaling helpers, statistics helpers) should live in utils or model/decoder.
# - Trainer exposes simple hooks/flags to enable warmup, per-parameter lr multipliers, and an optional batch_rescale function.
import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch_geometric.data import HeteroData
from train.dynamic_hetero_gnn import DynamicHeteroGNN
from train.coder import GraphEncoder, GraphDecoder
from train.aligner import LatentAligner
import numpy as np
import torch.fft as fft

# Optional helper import (trainer will not fail if utils absent)
try:
    from train.utils import compute_batch_alpha
except Exception:
    compute_batch_alpha = None


class DynamicHeteroTrainer:
    """
    Cleaner trainer: orchestrates data -> model -> losses -> optimizer.
    Responsibilities:
      - Build model, aligner, optimizer (with optional param groups for 'scale' params)
      - Provide warm-up (optionally freeze scale params), scheduler
      - Compute losses: alignment, temporal, recon (denorm), recon_norm (normalized-space)
      - Optionally call an external batch_rescale function (pure function) to compute per-batch alphas
      - Keep diagnostics minimal and clear (one concise per-epoch summary + optional one-time detailed dump)

    Configuration notes (most important knobs are constructor args):
      - recon_weight, recon_norm_weight, temp_weight, align_weight: loss weights
      - warmup_epochs: when >0 and freeze_scale_during_warmup True, scale params are frozen for warmup_epochs
      - scale_lr_mul: multiplier applied to parameters whose name contains 'scale' or 'log_scale' to speed scale learning
      - batch_rescale_fn: optional callable (r_det, t_res, cfg) -> alpha; if provided, trainer will use it before recon loss.
          A reference implementation compute_batch_alpha is available in train/utils.py (std_ratio or ls).
    """

    def __init__(
        self,
        hetero_data: Union[HeteroData, List[HeteroData], Dict[str, List[HeteroData]]],
        input_dims: Optional[Dict[str, int]] = None,
        hidden_dim: int = 128,
        num_layers: int = 8,
        dropout: float = 0.3,
        lr: float = 4e-4,
        num_epochs: int = 100,
        recon_weight: float = 1.0,
        recon_norm_weight: float = 1.0,
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
        batch_rescale_fn: Optional[Callable] = None,
        batch_rescale_cfg: Optional[Dict] = None,
    ):
        self.logger = logging.getLogger("DynamicHeteroTrainer")
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.logger.addHandler(ch)
        self.logger.setLevel(logging.INFO if debug else logging.WARNING)

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

        # hooks & utils
        self.batch_rescale_fn = batch_rescale_fn  # callable(r_det, t_res, cfg) -> alpha (detached)
        self.batch_rescale_cfg = batch_rescale_cfg or {}

        # simple defaults
        self.graph_encoder = GraphEncoder()
        self.graph_decoder = GraphDecoder()
        self.temporal_loss_alpha = 1.0
        self.temporal_loss_beta = 0.5

        # dataset flattening
        self.data_list = self._flatten_data(self.hetero_data)
        if not self.data_list:
            raise ValueError("No HeteroData provided to trainer")
        sample = self.data_list[0]
        self.metadata = sample.metadata() if hasattr(sample, "metadata") else (list(self.input_dims.keys()), [])
        if not self.input_dims:
            self.input_dims = self._infer_input_dims(sample, self.metadata[0])

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

        # aligner
        self.aligner = LatentAligner(hidden_dim=self.hidden_dim, mode="nodewise", lambda_align=1.0, temperature=0.3).to(self.device)

        # Ensure lazy modules created where possible: run one forward with sample to register lazy params
        try:
            sample_graph = self.data_list[0].to(self.device)
            with torch.no_grad():
                enc_out = self.graph_encoder(sample_graph)
                _, _, _, _, edge_index_dict = enc_out
                _ = self.model(sample_graph, edge_index_dict=edge_index_dict)
        except Exception as e:
            # If forward fails here, it's not fatal; optimizer will be built from parameters present
            self.logger.debug(f"[Init] dummy forward failed: {e}")

        # build optimizer with optional scale param group
        base_params = []
        scale_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # pick scale params by name convention (scale, log_scale)
            if ("scale" in name or "log_scale" in name) and p.ndim >= 0:
                scale_params.append(p)
            else:
                base_params.append(p)

        param_groups = [{"params": base_params}]
        if len(scale_params) > 0:
            param_groups.append({"params": scale_params, "lr": self.lr * float(self.scale_lr_mul)})

        # include aligner params
        param_groups.append({"params": list(self.aligner.parameters())})

        self.optimizer = torch.optim.Adam(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.7)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.loss_log = {"total": [], "align": [], "temp": [], "recon": [], "recon_norm": []}

    # -------------------------
    # Utilities
    # -------------------------
    def _flatten_data(self, data) -> List[HeteroData]:
        if isinstance(data, HeteroData):
            return [data]
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            flat = []
            for v in data.values():
                flat.extend(v if isinstance(v, list) else [v])
            return flat
        return []

    def _infer_input_dims(self, sample: HeteroData, node_types: List[str]) -> Dict[str, int]:
        inferred: Dict[str, int] = {}
        for nt in node_types:
            F_dim = 1
            try:
                x_seq = getattr(sample[nt], "x_seq", None)
                if x_seq is not None and len(x_seq.shape) == 3:
                    F_dim = int(x_seq.shape[2])
            except Exception:
                F_dim = 1
            inferred[nt] = F_dim
        self.logger.info(f"[Init] inferred input dims: {inferred}")
        return inferred

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
        pred = out.permute(0, 2, 1)
        targ_f = targ.permute(0, 2, 1)
        pred_fft = fft.rfft(pred, dim=-1)
        targ_fft = fft.rfft(targ_f, dim=-1)
        pred_mag = pred_fft.abs()
        targ_mag = targ_fft.abs()
        n_bins = pred_mag.size(-1)
        freq_idx = torch.arange(n_bins, device=device).float()
        weights = (freq_idx / (n_bins - 1)).view(1, 1, -1)
        freq_loss = F_nn.mse_loss(pred_mag * weights, targ_mag * weights)
        a_t = getattr(self, "temporal_loss_alpha", 1.0)
        a_f = getattr(self, "temporal_loss_beta", 0.5)
        return a_t * time_loss + a_f * freq_loss

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
        epochs = num_epochs or self.num_epochs
        self.model.train()
        self.aligner.train()

        # Warmup: optionally freeze scale params for warmup_epochs
        if self.warmup_epochs > 0 and self.freeze_scale_during_warmup:
            self._set_scale_requires_grad(False)
            self.logger.info(f"[Warmup] freezing scale params for {self.warmup_epochs} epochs")

        for epoch in range(1, epochs + 1):
            start = time.time()
            total_loss = total_align = total_temp = total_recon = total_recon_norm = 0.0
            batches = 0

            # Unfreeze scales after warmup
            if epoch == self.warmup_epochs + 1 and self.freeze_scale_during_warmup:
                self._set_scale_requires_grad(True)
                self.logger.info("[Warmup] unfreezing scale params, resuming full training")

            for data_idx, data in enumerate(self.data_list):
                data = data.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                # encode
                encoder_out = self.graph_encoder(data)
                x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict = encoder_out

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(data, edge_index_dict=edge_index_dict)
                    # model expected to return (z_dict, gru_out, proj_seq_dict, recon_seq_denorm, recon_seq_scaled, global_seq [, recon_feature_dict])
                    if len(outputs) >= 7:
                        z_dict, gru_seq_dict, proj_seq_dict, recon_seq_dict, recon_seq_scaled, global_seq, recon_feature_dict = outputs[:7]
                    else:
                        z_dict, gru_seq_dict, proj_seq_dict, recon_seq_dict, recon_seq_scaled, global_seq = outputs[:6]
                        recon_feature_dict = None

                    # Align loss
                    a_z_f = z_dict.get("fmri", None)
                    a_z_e = z_dict.get("eeg", None)
                    align_loss = self.aligner(a_z_f, a_z_e) if (a_z_f is not None and a_z_e is not None) else torch.tensor(0.0, device=self.device)

                    # Temporal loss
                    temp_loss = torch.tensor(0.0, device=self.device)
                    for nt in self.metadata[0]:
                        seq = proj_seq_dict[nt]
                        temp_loss = temp_loss + self._temporal_prediction_loss(seq, nt)

                    # Optionally compute per-batch alpha externally (trainer keeps no modality-specific hacks)
                    recon_to_use = recon_seq_dict
                    batch_alphas = {}
                    if self.batch_rescale_fn is not None:
                        # batch_rescale_fn expected: (r_det, t_res, cfg) -> alpha (tensor or scalar), uses detached r
                        recon_to_use = {}
                        for nt, recon in recon_seq_dict.items():
                            apply_here = (not self.batch_rescale_cfg.get("only")) or (nt in self.batch_rescale_cfg.get("only", []))
                            if self.batch_rescale_cfg.get("enable", True) and apply_here:
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

                    # Reconstruction losses
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

                    # Normalized-space auxiliary loss (if model provides recon_feature_dict)
                    recon_norm_loss = torch.tensor(0.0, device=self.device)
                    if recon_feature_dict is not None and self.recon_norm_weight > 0:
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
                            recon_norm_loss = recon_norm_loss + F_nn.mse_loss(recon_feat_crop, target_norm)

                    # total
                    loss = self.align_weight * align_loss + self.temp_weight * temp_loss + self.recon_weight * recon_loss
                    if self.recon_norm_weight > 0:
                        loss = loss + self.recon_norm_weight * recon_norm_loss

                # Backward & step
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

                # accumulate
                total_loss += float(loss.detach().cpu())
                total_align += float(align_loss.detach().cpu())
                total_temp += float(temp_loss.detach().cpu())
                total_recon += float(recon_loss.detach().cpu())
                total_recon_norm += float(recon_norm_loss.detach().cpu()) if isinstance(recon_norm_loss, torch.Tensor) else 0.0
                batches += 1

                # one-line batch summary for first batch in epoch
                if data_idx == 0:
                    self.logger.info(f"[Train] epoch={epoch} batch=0 recon_losses={recon_losses_per_nt}")
                    if batch_alphas:
                        for nt, a in batch_alphas.items():
                            if isinstance(a, torch.Tensor):
                                self.logger.info(f"[Train] batch alpha {nt} sample first5: {a.view(-1)[:5].cpu().tolist()}")
                            else:
                                self.logger.info(f"[Train] batch alpha {nt}: {a}")

            # epoch end
            avg_total = total_loss / batches
            avg_align = total_align / batches
            avg_temp = total_temp / batches
            avg_recon = total_recon / batches
            avg_recon_norm = total_recon_norm / batches

            self.loss_log["total"].append(avg_total)
            self.loss_log["align"].append(avg_align)
            self.loss_log["temp"].append(avg_temp)
            self.loss_log["recon"].append(avg_recon)
            self.loss_log["recon_norm"].append(avg_recon_norm)

            self.scheduler.step()

            # epoch diagnostics
            rel_error_epoch = {}
            # compute relative errors on last batch `data` (simple, consistent)
            for nt in self.metadata[0]:
                recon_final = recon_seq_dict[nt]
                rel_error_epoch[nt] = self._compute_relative_error({nt: recon_final}, data, self.metadata)[nt]
                if verbose:
                    self.logger.info(f"[Debug:{nt}] recon mean={recon_final.mean():.5f}, std={recon_final.std():.5f}")

            if verbose:
                self.logger.info(
                    f"[Epoch {epoch:3d}] total={avg_total:.6f} align={avg_align:.6f} "
                    f"temp={avg_temp:.6f} recon={avg_recon:.6f} recon_norm={avg_recon_norm:.6f} time={time.time()-start:.2f}s"
                )
                self.logger.info(f"[Epoch {epoch}] relative_error={rel_error_epoch}")
                self._log_reconstruction_histogram(recon_seq_dict, self.metadata)

    # -------------------------
    # Helpers
    # -------------------------
    def _set_scale_requires_grad(self, flag: bool):
        """Set requires_grad for any parameter that looks like a scale parameter (by name)."""
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

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "aligner": self.aligner.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(payload, path)
        self.logger.info(f"[Save] saved {path}")

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.aligner.load_state_dict(ckpt["aligner"])
        try:
            if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                self.optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            self.logger.warning("[Load] optimizer state incompatible")
        self.logger.info(f"[Load] loaded {path}")