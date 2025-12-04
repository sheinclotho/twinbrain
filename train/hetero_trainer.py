# Refactored DynamicHeteroTrainer
# Assumes model encoders produce flattened time-major inputs and return denormalized reconstructions
import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch_geometric.data import HeteroData
from train.dynamic_hetero_gnn import DynamicHeteroGNN
from train.coder import GraphEncoder, GraphDecoder
from train.aligner import LatentAligner
import numpy as np
import torch.fft as fft

class DynamicHeteroTrainer:
    """
    Cleaner trainer that delegates encoding/decoding to DynamicHeteroGNN.
    Model.forward accepts HeteroData and returns:
      (z_dict, gru_out, proj_seq_dict, recon_seq_denorm, global_seq, attn_stats)
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
        align_weight: float = 1.0,
        temp_weight: float = 0.5,
        recon_weight: float = 1.0,
        temporal_T: int = 200,
        spatial_T: int = 384,
        max_T: int = 2000,
        use_amp: bool = False,
        weight_decay: float = 1e-5,
        debug: bool = False,
        grad_clip: float = 1.0,
        aligner_mode: str = "nodewise",
        aligner_lambda: float = 1.0,
        aligner_temperature: float = 0.3,
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
        self.align_weight = align_weight
        self.temp_weight = temp_weight
        self.recon_weight = recon_weight
        self.temporal_T = temporal_T
        self.spatial_T = spatial_T
        self.max_T = max_T
        self.use_amp = use_amp and torch.cuda.is_available()
        self.weight_decay = weight_decay
        self.debug = debug
        self.grad_clip = grad_clip
        self.aligner_mode = aligner_mode
        self.aligner_lambda = aligner_lambda
        self.aligner_temperature = aligner_temperature
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

        # NOTE: we DO NOT create graph_encoder/graph_decoder here to avoid double-encoding.
        self.aligner = LatentAligner(
            hidden_dim=self.hidden_dim,
            mode=self.aligner_mode,
            lambda_align=self.aligner_lambda,
            temperature=self.aligner_temperature,
        ).to(self.device)

        params = list(self.model.parameters()) + list(self.aligner.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.7)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.loss_log = {"total": [], "align": [], "temp": [], "recon": []}

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
        """
        Time-domain + frequency-domain prediction loss
        proj_seq: (N, T, H)
        """
        device = self.device

        if proj_seq is None or proj_seq.numel() == 0:
            return torch.tensor(0.0, device=device)
        if proj_seq.shape[1] < 2:
            return torch.tensor(0.0, device=device)
        if nt not in self.model.grus:
            return torch.tensor(0.0, device=device)

        gru = self.model.grus[nt]

        # ----- Time-domain one-step prediction -----
        inp = proj_seq[:, :-1, :]      # (N,T-1,H)
        targ = proj_seq[:, 1:, :]      # (N,T-1,H)
        out, _ = gru(inp)              # (N,T-1,H)

        time_loss = F_nn.mse_loss(out, targ)

        # ----- Frequency-domain magnitude loss -----
        pred = out.permute(0, 2, 1)      # (N,H,L)
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
        """
        Resample (N, T, F) -> (N, target_T, F) using adaptive avg / linear interpolation.
        """
        if x is None or x.numel() == 0:
            return x
        N, T_orig, F_dim = x.shape
        xp = x.permute(0, 2, 1)  # (N, F, T)
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

        for epoch in range(1, epochs + 1):
            start = time.time()
            total_loss = total_align = total_temp = total_recon = 0.0
            batches = 0

            for data_idx, data in enumerate(self.data_list):
                data = data.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)

                # --- call graph_encoder ---
                encoder_out = self.graph_encoder(data)
                x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict = encoder_out

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(data, edge_index_dict=edge_index_dict)
                    if len(outputs) == 6:
                        z_dict, gru_seq_dict, proj_seq_dict, recon_seq_dict, recon_seq_scaled, global_seq = outputs
                    else:
                        z_dict, gru_seq_dict, proj_seq_dict, recon_seq_dict, recon_seq_scaled, global_seq = outputs[:6]

                    # --- Align loss ---
                    a_z_f = z_dict.get("fmri", None)
                    a_z_e = z_dict.get("eeg", None)
                    if a_z_f is not None and a_z_e is not None:
                        align_loss = self.aligner(a_z_f, a_z_e)
                    else:
                        align_loss = torch.tensor(0.0, device=self.device)

                    # --- Temporal loss ---
                    temp_loss = torch.tensor(0.0, device=self.device)
                    for nt in self.metadata[0]:
                        seq = proj_seq_dict[nt]
                        temp_loss += self._temporal_prediction_loss(seq, nt)

                    # --- Reconstruction loss ---
                    recon_loss = torch.tensor(0.0, device=self.device)
                    for nt in self.metadata[0]:
                        recon = recon_seq_scaled[nt]
                        target = getattr(data[nt], "x_seq")
                        target_res = self._resample_time(target.to(self.device), recon.shape[1])

                        Nr, Tr, Fr = recon.shape
                        Nt, Tt, Ft = target_res.shape
                        min_N, min_T, min_F = min(Nr, Nt), min(Tr, Tt), min(Fr, Ft)

                        recon_crop = recon[:min_N, :min_T, :min_F]
                        target_crop = target_res[:min_N, :min_T, :min_F]

                        recon_loss += F_nn.mse_loss(recon_crop, target_crop)

                    # --- Total loss ---
                    loss = self.align_weight * align_loss + self.temp_weight * temp_loss + self.recon_weight * recon_loss

                # --- Backward ---
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.parameters()) + list(self.aligner.parameters()), self.grad_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.parameters()) + list(self.aligner.parameters()), self.grad_clip
                        )
                    self.optimizer.step()

                # --- Stats ---
                total_loss += float(loss.detach().cpu())
                total_align += float(align_loss.detach().cpu())
                total_temp += float(temp_loss.detach().cpu())
                total_recon += float(recon_loss.detach().cpu())
                batches += 1

            # --- Epoch end logging ---
            avg_total = total_loss / batches
            avg_align = total_align / batches
            avg_temp = total_temp / batches
            avg_recon = total_recon / batches

            self.loss_log['total'].append(avg_total)
            self.loss_log['align'].append(avg_align)
            self.loss_log['temp'].append(avg_temp)
            self.loss_log['recon'].append(avg_recon)

            self.scheduler.step()

            # --- Epoch end: compute relative error for all node types ---
            rel_error_epoch = {}
            for nt in self.metadata[0]:
                recon_final = recon_seq_scaled[nt]
                rel_error_epoch[nt] = self._compute_relative_error({nt: recon_final}, data, self.metadata)[nt]
                if verbose:
                    self.logger.info(f"[Debug:{nt}] recon mean={recon_final.mean():.5f}, std={recon_final.std():.5f}")

            if verbose:
                self.logger.info(
                    f"[Epoch {epoch:3d}] total={avg_total:.6f} align={avg_align:.6f} "
                    f"temp={avg_temp:.6f} recon={avg_recon:.6f} time={time.time()-start:.2f}s"
                )
                self.logger.info(f"[Epoch {epoch}] relative_error={rel_error_epoch}")
                self._log_reconstruction_histogram(recon_seq_scaled, self.metadata)

    # -------------------------
    # Plot / Save / Load
    # -------------------------
    def _plot_loss_curves(self):
        import matplotlib.pyplot as plt
        epochs = range(1, len(self.loss_log['total']) + 1)
        plt.figure(figsize=(9,5))
        plt.plot(epochs, self.loss_log['total'], label='total')
        plt.plot(epochs, self.loss_log['align'], label='align')
        plt.plot(epochs, self.loss_log['temp'], label='temp')
        plt.plot(epochs, self.loss_log['recon'], label='recon')
        plt.legend()
        path = 'loss_curves.png'
        plt.savefig(path, dpi=150)
        plt.close()
        self.logger.info(f"[Plot] saved {path}")

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not getattr(self.model, '_fully_initialized', False):
            try:
                sample_graph = None
                for split in ['train','val','test']:
                    if split in self.hetero_data and len(self.hetero_data[split])>0:
                        sample_graph = self.hetero_data[split][0]
                        break
                if sample_graph is None:
                    sample_graph = self.hetero_data[0] if isinstance(self.hetero_data, list) else list(self.hetero_data.values())[0][0]
                sample_graph = sample_graph.to(self.device)
                with torch.no_grad():
                    _ = self.model(sample_graph)
                self.model._fully_initialized = True
            except Exception as e:
                self.logger.warning(f"[Save] register forward failed: {e}")
        torch.save({'model': self.model.state_dict(), 'aligner': self.aligner.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(), 'recon_weight': self.recon_weight}, path)
        self.logger.info(f"[Save] saved {path}")

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.aligner.load_state_dict(ckpt['aligner'])
        try:
            if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
        except Exception:
            self.logger.warning('[Load] optimizer state incompatible')
        self.logger.info(f"[Load] loaded {path}")

    def _compute_relative_error(self, recon_seq_scaled, data, metadata, eps=1e-8, debug=False):
        """
        计算每个 node_type 的整体相对误差（Frobenius norm）
        recon_seq_scaled: dict[node_type] -> (N, T, F)
        data: 原始数据对象
        metadata: 数据描述，metadata[0] 内含 node_type 列表
        """
        rel_error = {}

        for nt in metadata[0]:
            if nt not in recon_seq_scaled:
                continue

            recon = recon_seq_scaled[nt]  # (N, T, F)
            target = getattr(data[nt], "x_seq")  # (N, T_raw, F)
            target_res = self._resample_time(target.to(self.device), recon.shape[1])

            # 对齐尺寸
            Nr, Tr, Fr = recon.shape
            Nt, Tt, Ft = target_res.shape
            mN, mT, mF = min(Nr, Nt), min(Tr, Tt), min(Fr, Ft)
            r = recon[:mN, :mT, :mF]
            t = target_res[:mN, :mT, :mF]

            # 计算整体 Frobenius norm 相对误差
            diff_norm = torch.norm(r - t)  # sqrt(sum((r-t)^2))
            target_norm = torch.norm(t) + eps
            rel = diff_norm / target_norm
            rel_error[nt] = rel.item()  # 小数形式

            if debug:
                r_mean, r_std = r.mean().item(), r.std().item()
                t_mean, t_std = t.mean().item(), t.std().item()
                print(f"[RelError:{nt}] recon mean={r_mean:.5f} std={r_std:.5f} "
                      f"target mean={t_mean:.5f} std={t_std:.5f} rel_error={rel.item():.5f}")

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
                f"[Hist] {nt} bins={bins} | "
                f"min={x.min():.4f} max={x.max():.4f} mean={x.mean():.4f} std={x.std():.4f}"
            )
