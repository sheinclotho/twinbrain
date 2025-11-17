# train/hetero_trainer.py — 修复 T_orig 维度错误 + AMP 兼容 + 完整 Debug
import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, GCNConv
from torch_geometric.nn import SAGEConv

# -------------------------
# Logger
# -------------------------
logger = logging.getLogger("hetero_trainer")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# -------------------------
# TemporalCrossAligner
# -------------------------
class TemporalCrossAligner(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, seq1: torch.Tensor, seq2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        seq1_b = seq1.unsqueeze(0)
        seq2_b = seq2.unsqueeze(0)
        aligned1, w1 = self.attn(seq1_b, seq2_b, seq2_b, need_weights=True)
        aligned2, w2 = self.attn(seq2_b, seq1_b, seq1_b, need_weights=True)
        eps = 1e-12
        entropy1 = -(w1 * (w1 + eps).log()).sum(dim=-1).mean().item()
        entropy2 = -(w2 * (w2 + eps).log()).sum(dim=-1).mean().item()
        return (
            aligned1.squeeze(0),
            aligned2.squeeze(0),
            {"attn_f2e_entropy": entropy1, "attn_e2f_entropy": entropy2},
        )

# -------------------------
# LatentAligner (robust for unequal sizes)
# -------------------------
class LatentAligner(nn.Module):
    def __init__(self, hidden_dim: int = 64, temperature: float = 0.3, safe_clip: float = 1e9):
        super().__init__()
        self.temperature = temperature
        self.proj_fmri = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_eeg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.safe_clip = safe_clip

    def forward(self, z_fmri: torch.Tensor, z_eeg: torch.Tensor) -> torch.Tensor:
        # handle empty
        if z_fmri.numel() == 0 or z_eeg.numel() == 0:
            return torch.tensor(0.0, device=z_fmri.device if z_fmri.numel() else z_eeg.device)

        try:
            # project + normalize
            z1 = F.normalize(self.proj_fmri(z_fmri), dim=-1)  # (Nf, D)
            z2 = F.normalize(self.proj_eeg(z_eeg), dim=-1)    # (Ne, D)

            sim = torch.matmul(z1, z2.t()) / self.temperature  # (Nf, Ne)

            # numerical safety: replace non-finite with large negative
            if not torch.isfinite(sim).all():
                sim = torch.where(torch.isfinite(sim), sim, torch.full_like(sim, -self.safe_clip))

            device = sim.device

            # i -> e: for each fmri row pick argmax column as positive class (0..Ne-1)
            labels_i2e = torch.argmax(sim, dim=1, keepdim=False).to(device=device, dtype=torch.long)
            # e -> i: operate on transposed sim (Ne, Nf)
            sim_e2i = sim.t()
            labels_e2i = torch.argmax(sim_e2i, dim=1, keepdim=False).to(device=device, dtype=torch.long)

            loss_i2e = F.cross_entropy(sim, labels_i2e)
            loss_e2i = F.cross_entropy(sim_e2i, labels_e2i)
            return (loss_i2e + loss_e2i) * 0.5

        except Exception as e:
            # 遇到任何运行时错误时记录并返回 0，避免把 CUDA 状态搞脏
            logger.error(f"[LatentAligner ERROR] shapes zf={tuple(z_fmri.shape)}, ze={tuple(z_eeg.shape)} | err={e}")
            return torch.tensor(0.0, device=z_fmri.device if z_fmri.numel() else z_eeg.device)

# ===================================================== #
#  DynamicHeteroGNN
# ===================================================== #
class DynamicHeteroGNN(nn.Module):
    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        temporal_T: int = 200,
        spatial_T: int = 384,
        debug: bool = False,
    ):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.temporal_T = temporal_T
        self.spatial_T = spatial_T
        self.debug = debug

        self.convs = nn.ModuleList()
        self.grus = nn.ModuleDict({nt: nn.GRU(hidden_dim, hidden_dim, batch_first=True) for nt in self.node_types})
        self.temporal_projs = nn.ModuleDict({nt: nn.ModuleDict() for nt in self.node_types})
        self.temporal_align = TemporalCrossAligner(hidden_dim=hidden_dim, dropout=dropout)

        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _build_conv_for_layer(self, layer_idx: int, in_dims: Dict[str, int]):
        conv_dict = {}
        Conv = SAGEConv
        for src, rel, dst in self.edge_types:
            in_src = in_dims[src]
            in_dst = in_dims[dst]
            conv_dict[(src, rel, dst)] = Conv((in_src, in_dst), self.hidden_dim)
        return HeteroConv(conv_dict, aggr="mean")

    def _get_or_create_temporal_proj(self, nt: str, T_src: int) -> nn.Linear:
        key = str(T_src)
        if key not in self.temporal_projs[nt]:
            proj = nn.Linear(T_src, self.temporal_T, bias=False)
            nn.init.xavier_uniform_(proj.weight)
            self.temporal_projs[nt][key] = proj.to(next(self.parameters()).device)
        return self.temporal_projs[nt][key]

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]):
        device = next(self.parameters()).device

        # 1. 重采样 spatial_T
        x_resampled = {}
        for nt, x in x_dict.items():
            N, T_orig, F_dim = x.shape
            x = x.permute(0, 2, 1)
            if T_orig > self.spatial_T:
                x = F.adaptive_avg_pool1d(x, self.spatial_T)
            elif T_orig < self.spatial_T:
                x = F.interpolate(x, size=self.spatial_T, mode="linear", align_corners=False)
            x_resampled[nt] = x.permute(0, 2, 1).contiguous()

        num_nodes = {nt: x_resampled[nt].shape[0] for nt in self.node_types}
        T = self.spatial_T

        # 2. 展平成 time-major
        x_flat = {nt: x_resampled[nt].permute(1, 0, 2).contiguous().view(T * num_nodes[nt], -1) for nt in self.node_types}

        # 3. block-diagonal temporal lift
        temporal_edge_index_dict = {}
        for key, edge_index in edge_index_dict.items():
            if edge_index is None or edge_index.numel() == 0:
                continue
            src, _, dst = key
            n_src, n_dst = num_nodes[src], num_nodes[dst]
            e_base = edge_index.clone().long()
            e_base[0] = e_base[0] % n_src
            e_base[1] = e_base[1] % n_dst
            lifted_list = []
            for t in range(T):
                e_t = e_base.clone()
                e_t[0] += t * n_src
                e_t[1] += t * n_dst
                lifted_list.append(e_t)
            temporal_edge_index_dict[key] = torch.cat(lifted_list, dim=1)

        # 4. GNN 层
        h = x_flat
        for layer_idx in range(self.num_layers):
            in_dims = {nt: h[nt].shape[1] for nt in self.node_types} if layer_idx == 0 else {nt: self.hidden_dim for nt in self.node_types}
            if layer_idx >= len(self.convs):
                conv = self._build_conv_for_layer(layer_idx, in_dims).to(device)
                self.convs.append(conv)
            h_new = self.convs[layer_idx](h, temporal_edge_index_dict)
            h = {nt: F.dropout(F.relu(h_new[nt]), p=self.dropout.p, training=self.training) for nt in self.node_types}

        # 5. reshape + GRU
        h_reshaped = {}
        for nt in self.node_types:
            if num_nodes[nt] == 0:
                h_reshaped[nt] = torch.zeros(0, T, self.hidden_dim, device=device)
            else:
                h_reshaped[nt] = h[nt].reshape(num_nodes[nt], T, self.hidden_dim)

        gru_out = {nt: self.grus[nt](seq)[0] if seq.numel() > 0 else torch.zeros(0, T, self.hidden_dim, device=device) for nt, seq in h_reshaped.items()}

        # 6. modality pooling + cross-modal alignment
        modality_pool = {nt: seq.mean(dim=0) for nt, seq in gru_out.items() if seq.numel() > 0}
        aligned_pools = modality_pool.copy()
        attn_stats = {}
        if "fmri" in modality_pool and "eeg" in modality_pool:
            a_f, a_e, stats = self.temporal_align(modality_pool["fmri"], modality_pool["eeg"])
            aligned_pools["fmri"] = a_f
            aligned_pools["eeg"] = a_e
            attn_stats.update(stats)

        # 7. fuse
        fused_seq = {nt: gru_out[nt] + aligned_pools.get(nt, torch.zeros(self.hidden_dim, device=device)).unsqueeze(0) for nt in gru_out}

        # 8. temporal projection
        proj_seq_dict = fused_seq
        if self.temporal_T is not None and self.temporal_T != self.spatial_T:
            proj_seq_dict = {}
            for nt, seq in fused_seq.items():
                if seq.shape[1] == 0:
                    proj_seq_dict[nt] = torch.zeros(num_nodes[nt], self.temporal_T, self.hidden_dim, device=device)
                    continue
                flat = seq.permute(0, 2, 1).reshape(-1, T)
                proj = self._get_or_create_temporal_proj(nt, T)
                out_flat = proj(flat)
                proj_seq_dict[nt] = out_flat.view(num_nodes[nt], self.hidden_dim, self.temporal_T).permute(0, 2, 1)

        # 9. z_dict + global_seq
        z_dict = {nt: seq.mean(dim=1) for nt, seq in proj_seq_dict.items() if seq.numel() > 0}
        valid_means = [seq.mean(dim=0) for seq in proj_seq_dict.values() if seq.numel() > 0]
        global_seq = self.global_proj(torch.stack(valid_means).mean(dim=0)) if valid_means else torch.zeros(self.hidden_dim, device=device)

        return z_dict, gru_out, proj_seq_dict, global_seq, attn_stats


# ===================================================== #
#  DynamicHeteroTrainer
# ===================================================== #
class DynamicHeteroTrainer:
    def __init__(
        self,
        hetero_data: Union[HeteroData, List[HeteroData], Dict[str, List[HeteroData]]],
        input_dims: Optional[Dict[str, int]] = None,
        hidden_dim: int = 64,
        num_layers: int = 5,
        dropout: float = 0.3,
        lr: float = 3e-4,
        num_epochs: int = 100,
        align_weight: float = 1.0,
        temp_weight: float = 0.5,
        max_T: int = 2000,
        temporal_T: int = 200,
        spatial_T: int = 384,
        use_amp: bool = False,
        weight_decay: float = 1e-5,
        debug: bool = False,
        grad_clip: float = 1.0,
    ):

        logger = logging.getLogger("DynamicHeteroTrainer")
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(ch)
            logger.setLevel(logging.CRITICAL)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------- 基本超参 -------------------- #
        self.hetero_data = hetero_data
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.num_epochs = num_epochs
        self.align_weight = align_weight
        self.temp_weight = temp_weight
        self.max_T = max_T
        self.temporal_T = temporal_T
        self.spatial_T = spatial_T
        self.use_amp = use_amp and torch.cuda.is_available()
        self.weight_decay = weight_decay
        self.debug = debug
        self.grad_clip = grad_clip

        # -------------------- 数据展平 -------------------- #
        self.data_list = self._flatten_data(hetero_data)
        if not self.data_list:
            raise ValueError("No HeteroData provided")

        # -------------------- Metadata -------------------- #
        sample = self.data_list[0]
        self.metadata = sample.metadata() if hasattr(sample, "metadata") else (list(self.input_dims.keys()), [])

        # -------------------- 模型 -------------------- #
        self.model = DynamicHeteroGNN(
            metadata=self.metadata,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            temporal_T=self.temporal_T,
            spatial_T=self.spatial_T,
            debug=self.debug,
        ).to(self.device)

        # -------------------- 对齐器 -------------------- #
        self.aligner = LatentAligner(hidden_dim=self.hidden_dim).to(self.device)

        # -------------------- 优化器 -------------------- #
        params = list(self.model.parameters()) + list(self.aligner.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.7)

        # -------------------- AMP -------------------- #
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        logger.critical(
            f"[Trainer] Initialized | graphs={len(self.data_list)} | "
            f"temporal_T={self.temporal_T} | spatial_T={self.spatial_T} | "
            f"AMP={self.use_amp} | DEBUG={self.debug}"
        )


    # --------------------------------------------------------------------- #
    #  1.  数据展平
    # --------------------------------------------------------------------- #
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

    # --------------------------------------------------------------------- #
    #  2.  输入准备（z-score + clamp）
    # --------------------------------------------------------------------- #
    def _prepare_x_dict(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict: Dict[str, torch.Tensor] = {}
        src = data.x_seq_dict if hasattr(data, "x_seq_dict") and data.x_seq_dict else None

        for nt in data.node_types:
            if src and nt in src:
                x = src[nt]
            else:
                x = getattr(data[nt], "x_seq", None)
                if x is None:
                    raise RuntimeError(f"Missing {nt}.x_seq")

            T_cut = min(x.shape[1], self.max_T)
            x_t = x[:, :T_cut, :].to(self.device)  # (N, T, F)

            # ---- z-score per sample (dim=1) ----
            mean = x_t.mean(dim=1, keepdim=True)
            std = x_t.std(dim=1, keepdim=True) + 1e-8
            x_t = (x_t - mean) / std
            x_t = torch.clamp(x_t, -10.0, 10.0)

            # 方案B: 保留时间维，feature 投影在 GNN forward 中处理
            x_dict[nt] = x_t

        if self.debug:
            for nt, tensor in x_dict.items():
                logger.debug(f"[Input] {nt} shape={tensor.shape} mean={tensor.mean().item():.4f} std={tensor.std().item():.4f}")

        return x_dict

    # --------------------------------------------------------------------- #
    #  3.  Temporal prediction loss
    # --------------------------------------------------------------------- #
    def _temporal_prediction_loss(self, proj_seq: torch.Tensor, nt: str) -> torch.Tensor:
        if proj_seq.shape[1] < 2:
            return torch.tensor(0.0, device=proj_seq.device)
        gru = self.model.grus[nt]
        out, _ = gru(proj_seq[:, :-1, :])
        targ = proj_seq[:, 1:, :]
        return F.mse_loss(out, targ)

    # --------------------------------------------------------------------- #
    #  4.  训练主循环（完整 debug + AMP）
    # --------------------------------------------------------------------- #
    def train(self, num_epochs: Optional[int] = None, verbose: bool = True):
        epochs = num_epochs or self.num_epochs
        self.model.train()
        self.aligner.train()
        nan_cnt = inf_cnt = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = total_align = total_temp = 0.0

            logger.critical(f"===== [EPOCH {epoch}] START =====")

            for data_idx, data in enumerate(self.data_list):
                x_dict = {}
                try:
                    # ==================== 1. 数据迁移 + 打印 ====================
                    data = data.to(self.device)

                    if hasattr(data, 'x_seq_dict') and data.x_seq_dict:
                        data.x_seq_dict = {k: v.to(self.device) for k, v in data.x_seq_dict.items()}
                    # ==================== 2. 准备 x_dict ====================
                    x_dict = self._prepare_x_dict(data)
                    # ==================== 3. 前向 + loss 计算 ====================
                    # 更安全的 zero_grad
                    self.optimizer.zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        z_dict, gru_seq_dict, proj_seq_dict, global_seq, attn_info = self.model(
                            x_dict, data.edge_index_dict
                        )
                        # ----- Align loss (robust) -----
                        align_loss = torch.tensor(0.0, device=self.device)
                        if "fmri" in proj_seq_dict and "eeg" in proj_seq_dict:
                            try:
                                zf = proj_seq_dict["fmri"].mean(dim=1)
                                ze = proj_seq_dict["eeg"].mean(dim=1)
                                logger.critical(f"   [Align] zf={zf.shape}, ze={ze.shape}")
                                align_loss = self.aligner(zf, ze)
                                if not isinstance(align_loss, torch.Tensor):
                                    align_loss = torch.tensor(float(align_loss), device=self.device)
                            except Exception as e:
                                logger.error(f"[AlignErr] shapes zf={proj_seq_dict['fmri'].shape}, ze={proj_seq_dict['eeg'].shape} | err={e}")
                                align_loss = torch.tensor(0.0, device=self.device)
                        # 延后打印 align_loss.item() 到检查非有限之后

                        # ----- Temporal loss (robust) -----
                        temp_loss = torch.tensor(0.0, device=self.device)
                        for nt in self.metadata[0]:
                            seq = proj_seq_dict.get(nt)
                            if seq is not None and seq.numel() > 0:
                                try:
                                    t_loss = self._temporal_prediction_loss(seq, nt)
                                    if not isinstance(t_loss, torch.Tensor):
                                        t_loss = torch.tensor(float(t_loss), device=self.device)
                                    temp_loss = temp_loss + t_loss
                                    logger.critical(f"   [Temp] {nt}: loss={t_loss.detach().cpu().item():.6f}")
                                except Exception as e:
                                    logger.error(f"[TempLossErr] nt={nt} err={e}")
                                    temp_loss = temp_loss + torch.tensor(0.0, device=self.device)

                        loss = self.align_weight * align_loss + self.temp_weight * temp_loss

                    # ==================== 4. 检查 loss 的有效性 ====================
                    if not isinstance(loss, torch.Tensor):
                        loss = torch.tensor(float(loss), device=self.device, requires_grad=True)

                    if not torch.isfinite(loss):
                        if torch.isnan(loss):
                            nan_cnt += 1
                        if torch.isinf(loss):
                            inf_cnt += 1
                        logger.critical(f"[LOSS] Non-finite loss detected (epoch {epoch} idx {data_idx}). loss={loss}")
                        # 清空梯度并跳过该步，避免污染 scaler / optimizer
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    # ==================== 5. 反向传播（AMP-safe） ====================
                    if self.use_amp:
                        # scale -> backward
                        self.scaler.scale(loss).backward()

                        # unscale for clipping
                        if self.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                list(self.model.parameters()) + list(self.aligner.parameters()),
                                self.grad_clip,
                            )
                            # clip_grad_norm_ may return float or Tensor
                            try:
                                gnorm = float(grad_norm)
                            except Exception:
                                gnorm = float(getattr(grad_norm, "item", lambda: float('nan'))())
                            logger.critical(f"   [Grad] norm={gnorm:.4f}")
                        # step
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip > 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                list(self.model.parameters()) + list(self.aligner.parameters()),
                                self.grad_clip,
                            )
                            try:
                                gnorm = float(grad_norm)
                            except Exception:
                                gnorm = float(getattr(grad_norm, "item", lambda: float('nan'))())
                            logger.critical(f"   [Grad] norm={gnorm:.4f}")
                        self.optimizer.step()

                    # ==================== 6. 统计与日志 ====================
                    total_loss += float(loss.detach().cpu().item())
                    try:
                        total_align += float(align_loss.detach().cpu().item()) if isinstance(align_loss, torch.Tensor) else float(align_loss)
                    except Exception:
                        total_align += 0.0
                    try:
                        total_temp += float(temp_loss.detach().cpu().item()) if isinstance(temp_loss, torch.Tensor) else float(temp_loss)
                    except Exception:
                        total_temp += 0.0

                    if self.debug and data_idx == 0 and attn_info:
                        logger.debug(f"[Attn] f2e_entropy={attn_info.get('attn_f2e_entropy'):.4f} "
                                    f"e2f_entropy={attn_info.get('attn_e2f_entropy'):.4f}")

                except Exception as e:
                    # 在捕获异常时打印尽可能多的调试信息，再抛出
                    logger.critical(f"[CRASH] data_idx={data_idx} 崩溃: {e}")
                    try:
                        logger.critical(f"   x_dict: { {k: v.shape for k,v in x_dict.items() } }")
                    except Exception:
                        logger.critical("   x_dict: (unavailable)")
                    try:
                        logger.critical(f"   edge_index max: { {str(k): v.max().item() if v.numel()>0 else -1 for k,v in data.edge_index_dict.items() } }")
                    except Exception:
                        logger.critical("   edge_index max: (unavailable)")
                    # 重新抛出以便上层能中断或检查
                    raise

            # epoch end
            self.scheduler.step()
            if verbose:
                logger.info(
                    f"[Epoch {epoch:3d}] Loss: {total_loss:.6f} | "
                    f"Align: {total_align:.4f} | Temp: {total_temp:.4f} | "
                    f"Time: {time.time() - start_time:.2f}s | "
                    f"NaN:{nan_cnt} Inf:{inf_cnt}"
                )
                nan_cnt = inf_cnt = 0

        logger.critical("===== [TRAIN] 全部完成 =====")

    #  5.  推理 / 保存 / 加载
    # --------------------------------------------------------------------- #
    def get_embeddings(self, data=None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        返回每个数据索引的每种 node_type embedding，格式：
        { 'task_1': {node_type: np.ndarray}, ..., 'task_4': {...} }
        """
        self.model.eval()
        data_list = self._flatten_data(data) if data else self.data_list

        node_types = self.metadata[0]
        results: Dict[str, Dict[str, List[np.ndarray]]] = {}

        with torch.no_grad():
            for idx, data in enumerate(data_list):
                data = data.to(self.device)
                task_name = f"task_{idx+1}"  # 按索引命名 task
                results[task_name] = {nt: [] for nt in node_types}

                z_dict, _, _, _, _ = self.model(self._prepare_x_dict(data), data.edge_index_dict)
                for nt in node_types:
                    if nt in z_dict:
                        results[task_name][nt].append(z_dict[nt].cpu().numpy())

        # 合并 list -> array
        final_results: Dict[str, Dict[str, np.ndarray]] = {}
        for t, nt_dict in results.items():
            final_results[t] = {}
            for nt, embeds in nt_dict.items():
                if embeds:
                    final_results[t][nt] = np.concatenate(embeds, axis=0)
                else:
                    final_results[t][nt] = np.zeros((0, self.model.hidden_dim))

        # 调试打印
        for t, nt_dict in final_results.items():
            logging.info(f"[Embed] Task={t}")
            for nt, arr in nt_dict.items():
                logging.info(f"   node_type={nt}, shape={arr.shape}")

        return final_results

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "aligner": self.aligner.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            path,
        )
        logger.info(f"[Save] Model saved to {path}")

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.aligner.load_state_dict(ckpt["aligner"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        logger.info(f"[Load] Model loaded from {path}")