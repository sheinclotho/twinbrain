import math
from typing import Any, Dict, List, Optional, Tuple

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
import torch.nn.functional as F_nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData

from train.coder import NodeDecoder, ProjectionHead, TemporalDecoder

# TemporalCrossAligner 可选
try:
    from train.aligner import TemporalCrossAligner
    _HAS_TEMPORAL_ALIGNER = True
except Exception:
    TemporalCrossAligner = None
    _HAS_TEMPORAL_ALIGNER = False


class DynamicHeteroGNN(nn.Module):
    """
    DynamicHeteroGNN
    ----------------
    职责：
    - 在 GraphEncoder 之后接手：
        x_dict:      node_type -> (N, T_spatial, H_enc)    # 来自 GraphEncoder(NodeEncoder)
        edge_index_dict: (src,rel,dst) -> edge_index(2,E)
        stats_dict:  node_type -> {"mean","std","orig_T"}  # 用于 NodeDecoder 反归一化
    - 通过 HeteroConv / SAGEConv 融合图结构；
    - 用 GRU 沿时间维建模时序；
    - 用 TemporalDecoder + NodeDecoder 重构原始信号；
    - 用 ProjectionHead + TemporalCrossAligner 做模态对齐和 global 表征；
    - 为 temporal prediction（在 latent 空间预测未来）提供 proj_seq_dict。

    重要说明：
    - 该类不再负责“原始 x_seq -> 编码”，这一部分由 GraphEncoder 管理；
      这里的 forward 必须接受 x_dict / edge_index_dict 的形式（在 Trainer 里已经包装好）。
    - EEG 相关配置：
        - eeg_input_scale 由 GraphEncoder 里处理时序放大（你已经在 GraphEncoder 前面做了标准化）；
          这里主要通过 eeg_decoder_cfg 给 EEG 一个独立的 TemporalDecoder 配置。
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        node_feature_dims: Dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        temporal_T: int = 200,
        spatial_T: int = 384,
        debug: bool = False,
        eeg_decoder_cfg: Optional[Dict[str, Any]] = None,
    ):
        """
        参数：
        - metadata: (node_types, edge_types)
            node_types: ["fmri","eeg",...]
            edge_types: [("fmri","connects","fmri"), ("eeg","connects","eeg"), ("fmri","projects_to","eeg"), ...]
        - node_feature_dims: node_type -> 原始特征维度 F_in（用于确定 decoder 输出 dim）
        - hidden_dim: 编码后的隐变量维度 H
        - num_layers: GNN 层数
        - dropout: GNN / 全连接层中的 Dropout 比例
        - temporal_T: Temporal projection 的目标时间长度（proj_seq_dict[*].shape[1]）
        - spatial_T: 保留以兼容 NodeEncoder 的 T_spatial（这里仅做记录）
        - eeg_decoder_cfg: 对 EEG 的 TemporalDecoder 进行的特殊配置：
            {
                "channels": int,
                "kernel_size": int,
                "num_layers": int,
                "dropout": float
            }
        """
        super().__init__()
        self.node_types, self.edge_types = metadata
        if not isinstance(self.node_types, (list, tuple)):
            raise ValueError("metadata 必须是 (node_types, edge_types)，node_types 为 list/tuple")

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.temporal_T = int(temporal_T)
        self.spatial_T = int(spatial_T)
        self.debug = bool(debug)

        self.node_feature_dims = {nt: int(node_feature_dims.get(nt, 1)) for nt in self.node_types}
        self.dropout = nn.Dropout(float(dropout))

        # ====== GNN backbone: HeteroConv + SAGEConv ======
        self.convs = nn.ModuleList()

        # ====== GRU for temporal modeling per node_type ======
        self.grus = nn.ModuleDict({
            nt: nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
            for nt in self.node_types
        })

        # ====== per-node_type temporal projection: T_src -> temporal_T ======
        self.temporal_projs = nn.ModuleDict({nt: nn.ModuleDict() for nt in self.node_types})

        # ====== feature decoders (TemporalDecoder) per node_type ======
        self.eeg_decoder_cfg = dict(eeg_decoder_cfg) if eeg_decoder_cfg is not None else {}
        self.feature_decoders = nn.ModuleDict()
        self._build_feature_decoders()

        # ====== decoder_input_proj: concat(proj, gru) 2H -> H ======
        self.decoder_input_proj = nn.ModuleDict({
            nt: nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            for nt in self.node_types
        })

        # ====== Node-level denorm decoders (NodeDecoder) ======
        self.denorm_decoders = NodeDecoder(
            node_types=self.node_types,
            out_dims=self.node_feature_dims,
            hidden_dim=self.hidden_dim,
            extra_hidden=self.hidden_dim,
            num_layers=3,
            init_scales=None,
            scale_mode="log",
        )

        # ====== Temporal aligner (可选) ======
        if _HAS_TEMPORAL_ALIGNER:
            self.temporal_align = TemporalCrossAligner(hidden_dim=self.hidden_dim, dropout=0.0)
        else:
            class _IdentityAlign(nn.Module):
                def forward(self, a, b):
                    loss = torch.tensor(0.0, device=(a.device if a is not None else "cpu"))
                    return a, b, {"loss": loss}
            self.temporal_align = _IdentityAlign()

        # ====== Global projection & per-modality ProjectionHead ======
        self.global_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.proj_head = ProjectionHead(hidden_dim=self.hidden_dim, latent_dim=self.hidden_dim, dropout=dropout)

    # ------------------------------------------------------------------
    # Feature decoder 构造：给 EEG 单独配置
    # ------------------------------------------------------------------
    def _build_feature_decoders(self):
        for nt in self.node_types:
            out_dim = self.node_feature_dims.get(nt, 1)

            # base config
            base_channels = min(256, self.hidden_dim)
            base_kernel = 5
            base_layers = 3
            base_dropout = 0.1

            if nt == "eeg":
                channels = int(self.eeg_decoder_cfg.get("channels", min(128, self.hidden_dim)))
                kernel_size = int(self.eeg_decoder_cfg.get("kernel_size", base_kernel))
                num_layers = int(self.eeg_decoder_cfg.get("num_layers", base_layers))
                dec_dropout = float(self.eeg_decoder_cfg.get("dropout", base_dropout))
            else:
                channels = base_channels
                kernel_size = base_kernel
                num_layers = base_layers
                dec_dropout = base_dropout

            if channels <= 0:
                raise ValueError(f"[DynamicHeteroGNN] Decoder channels 必须 > 0, node_type={nt}, got {channels}")
            if kernel_size <= 0:
                raise ValueError(f"[DynamicHeteroGNN] kernel_size 必须 > 0, node_type={nt}, got {kernel_size}")
            if num_layers <= 0:
                raise ValueError(f"[DynamicHeteroGNN] num_layers 必须 > 0, node_type={nt}, got {num_layers}")
            if not (0.0 <= dec_dropout < 1.0):
                raise ValueError(f"[DynamicHeteroGNN] dropout 必须在 [0,1) 内, node_type={nt}, got {dec_dropout}")

            self.feature_decoders[nt] = TemporalDecoder(
                in_dim=self.hidden_dim,
                out_dim=out_dim,
                channels=channels,
                kernel_size=kernel_size,
                num_layers=num_layers,
                dropout=dec_dropout,
            )

    # ------------------------------------------------------------------
    # HeteroConv 构造
    # ------------------------------------------------------------------
    def _build_conv_for_layer(self, layer_idx: int, in_dims: Dict[str, int]) -> HeteroConv:
        conv_dict = {}
        for src, rel, dst in self.edge_types:
            in_src = int(in_dims.get(src, self.hidden_dim))
            in_dst = int(in_dims.get(dst, self.hidden_dim))
            conv_dict[(src, rel, dst)] = SAGEConv((in_src, in_dst), self.hidden_dim)
        return HeteroConv(conv_dict, aggr="mean")

    # ------------------------------------------------------------------
    # Temporal projection 构造（lazy）
    # ------------------------------------------------------------------
    def _get_or_create_temporal_proj(self, nt: str, T_src: int) -> nn.Linear:
        key = str(int(T_src))
        if key not in self.temporal_projs[nt]:
            proj = nn.Linear(T_src, self.temporal_T, bias=False)
            nn.init.xavier_uniform_(proj.weight)
            proj = proj.to(next(self.parameters()).device)
            self.temporal_projs[nt][key] = proj
        return self.temporal_projs[nt][key]

    # ------------------------------------------------------------------
    # forward: 整个图 + 时序 + 解码 pipeline
    # ------------------------------------------------------------------
    def forward(
        self,
        data: HeteroData,
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        encoded_dict: Optional[Dict[str, torch.Tensor]] = None,
        num_nodes_dict: Optional[Dict[str, int]] = None,
        stats_dict: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
        """
        参数：
        - data: HeteroData（主要用于 shapes / 兼容接口，这里一般不直接用 x_seq）
        - edge_index_dict: (src,rel,dst) -> LongTensor(2,E)，来自 GraphEncoder
        - encoded_dict: nt -> (N,T_spatial,H)，来自 GraphEncoder.x_dict
        - num_nodes_dict: nt -> N
        - stats_dict: nt -> {"mean","std","orig_T"}，来自 GraphEncoder.stats_dict

        返回（严格 8-tuple）：
        - z_dict:             nt -> pooled latent (N,H)
        - gru_out:            nt -> (N,T_spatial,H)
        - proj_seq_dict:      nt -> (N, T_proj, H)
        - recon_seq_denorm:   nt -> (N, T_dec, F_norm)  # 实际为 recon_feature（normalized space）
        - recon_seq_scaled:   nt -> (N, T_dec, F_raw)   # denorm 后原量纲
        - global_seq:         Tensor(H,)                # 全局 pooled 表征
        - recon_feature_dict: nt -> (N, T_dec, F_norm)  # 同 recon_seq_denorm
        - recon_denorm_dict:  dict{"recon_denorm_dict": recon_seq_scaled}
        """
        device = next(self.parameters()).device

        if encoded_dict is None or num_nodes_dict is None or stats_dict is None:
            raise RuntimeError(
                "DynamicHeteroGNN.forward 现在要求显式传入 encoded_dict / num_nodes_dict / stats_dict；"
                "请在 Trainer 中先调用 GraphEncoder，再把结果传进来。"
            )

        # ---------- 1) flatten time for GNN ----------
        # encoded_dict: nt -> (N,T_spatial,H)
        x_flat: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            seq = encoded_dict.get(nt, None)
            if seq is None:
                x_flat[nt] = torch.zeros((0, self.hidden_dim), device=device)
                continue
            if seq.ndim != 3:
                raise ValueError(f"[DynamicHeteroGNN] encoded_dict['{nt}'] must be (N,T_spatial,H), got {tuple(seq.shape)}")
            N_nt, T_sp, H_dim = seq.shape
            if H_dim != self.hidden_dim:
                raise ValueError(
                    f"[DynamicHeteroGNN] encoded_dict['{nt}'].shape[-1]={H_dim} != hidden_dim={self.hidden_dim}"
                )
            x_flat[nt] = seq.permute(1, 0, 2).reshape(T_sp * N_nt, H_dim)  # (T_sp*N, H)

        # ---------- 2) lift edges in time ----------
        T_pool = self.spatial_T
        temporal_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for key, eidx in edge_index_dict.items():
            if not isinstance(eidx, torch.Tensor):
                raise TypeError(f"[DynamicHeteroGNN] edge_index_dict[{key}] must be Tensor, got {type(eidx)}")
            src, _, dst = key
            n_src = num_nodes_dict.get(src, None)
            n_dst = num_nodes_dict.get(dst, None)
            if n_src is None or n_dst is None:
                raise RuntimeError(f"[DynamicHeteroGNN] num_nodes_dict 缺少 {src} 或 {dst}")
            e_base = eidx.clone().long().to(device)
            if e_base.ndim != 2 or e_base.shape[0] != 2:
                raise ValueError(f"[DynamicHeteroGNN] edge_index[{key}] must be (2,E), got {tuple(e_base.shape)}")

            if e_base.numel() == 0:
                temporal_edge_index_dict[key] = e_base
                continue

            # 检查索引范围
            if torch.any(e_base[0] < 0) or torch.any(e_base[1] < 0):
                raise ValueError(f"[DynamicHeteroGNN] edge_index[{key}] has negative indices")
            if torch.any(e_base[0] >= n_src) or torch.any(e_base[1] >= n_dst):
                raise ValueError(f"[DynamicHeteroGNN] edge_index[{key}] out of range (n_src={n_src}, n_dst={n_dst})")

            lifted_list = []
            for t in range(T_pool):
                lifted = e_base + torch.tensor([[t * n_src], [t * n_dst]], device=device)
                lifted_list.append(lifted)
            temporal_edge_index_dict[key] = torch.cat(lifted_list, dim=1)

        # ---------- 3) GNN 堆叠 ----------
        h = x_flat
        for layer_idx in range(self.num_layers):
            in_dims = {
                nt: (h[nt].shape[1] if (nt in h and h[nt].numel() > 0) else self.hidden_dim)
                for nt in self.node_types
            }
            if layer_idx >= len(self.convs):
                conv = self._build_conv_for_layer(layer_idx, in_dims).to(device)
                self.convs.append(conv)
            h_new = self.convs[layer_idx](h, temporal_edge_index_dict)
            h = {nt: self.dropout(F_nn.relu(h_new[nt])) for nt in self.node_types}

        # ---------- 4) reshape 回 (N,T_sp,H) ----------
        h_reshaped: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            seq_flat = h.get(nt, None)
            if seq_flat is None or seq_flat.numel() == 0:
                h_reshaped[nt] = torch.zeros((0, T_pool, self.hidden_dim), device=device)
                continue
            n_nodes = num_nodes_dict.get(nt, None)
            if n_nodes is None or n_nodes <= 0:
                raise RuntimeError(f"[DynamicHeteroGNN] num_nodes_dict[{nt}] invalid: {n_nodes}")
            rows = seq_flat.shape[0]
            expected = n_nodes * T_pool
            if rows != expected:
                raise ValueError(
                    f"[DynamicHeteroGNN] GNN output rows={rows} for '{nt}' != n_nodes*T_pool={expected}"
                )
            h_reshaped[nt] = seq_flat.view(T_pool, n_nodes, self.hidden_dim).permute(1, 0, 2).contiguous()

        # ---------- 5) GRU 沿时间维建模 ----------
        gru_out: Dict[str, torch.Tensor] = {}
        for nt, seq in h_reshaped.items():
            if seq.numel() == 0:
                gru_out[nt] = seq
                continue
            out_seq, _ = self.grus[nt](seq)
            gru_out[nt] = out_seq

        # ---------- 6) 投影 & temporal align ----------
        proj_pool: Dict[str, Optional[torch.Tensor]] = {}
        for nt in self.node_types:
            seq = gru_out.get(nt, None)
            if seq is None or seq.numel() == 0:
                proj_pool[nt] = None
                continue
            proj_out = self.proj_head(seq, modality=nt if nt in ["fmri", "eeg"] else "fmri")
            proj_pool[nt] = proj_out

        aligned_pools: Dict[str, Optional[torch.Tensor]] = {}
        if "fmri" in proj_pool and "eeg" in proj_pool and proj_pool["fmri"] is not None and proj_pool["eeg"] is not None:
            a_f, a_e, _ = self.temporal_align(proj_pool["fmri"], proj_pool["eeg"])
            aligned_pools["fmri"] = a_f
            aligned_pools["eeg"] = a_e
        else:
            for nt in self.node_types:
                aligned_pools[nt] = proj_pool.get(nt, None)

        # ---------- 7) aligned 向量加回 GRU 序列（逐时间步 broadcast） ----------
        fused_seq: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            seq = gru_out.get(nt, None)
            aligned = aligned_pools.get(nt, None)
            if seq is None or seq.numel() == 0:
                fused_seq[nt] = seq
                continue
            if aligned is not None:
                if aligned.ndim == 3:
                    aligned_vec = aligned.mean(dim=1)   # (N,H)
                elif aligned.ndim == 2:
                    aligned_vec = aligned
                else:
                    raise ValueError(f"[DynamicHeteroGNN] aligned_pools['{nt}'] ndim={aligned.ndim} invalid")
                if aligned_vec.shape[-1] != self.hidden_dim:
                    raise ValueError(
                        f"[DynamicHeteroGNN] aligned_vec dim={aligned_vec.shape[-1]} != hidden_dim={self.hidden_dim}"
                    )
                aligned_exp = aligned_vec.unsqueeze(1).expand(-1, seq.shape[1], -1)
                fused_seq[nt] = seq + aligned_exp
            else:
                fused_seq[nt] = seq

        # ---------- 8) temporal projection 到统一 temporal_T ----------
        proj_seq_dict: Dict[str, torch.Tensor] = {}
        for nt, seq in fused_seq.items():
            if seq is None or seq.numel() == 0:
                proj_seq_dict[nt] = seq
                continue
            T_src = seq.shape[1]
            if T_src != self.temporal_T:
                proj = self._get_or_create_temporal_proj(nt, T_src)
                flat = seq.permute(0, 2, 1).reshape(-1, T_src)  # (N*H,T_src)
                out_flat = proj(flat)                            # (N*H, temporal_T)
                seq_proj = out_flat.view(seq.shape[0], self.hidden_dim, self.temporal_T).permute(0, 2, 1)
                proj_seq_dict[nt] = seq_proj
            else:
                proj_seq_dict[nt] = seq

        # ---------- 9) 解码：decoder_input_proj + feature_decoder + NodeDecoder ----------
        recon_feature_dict: Dict[str, torch.Tensor] = {}
        recon_seq_denorm: Dict[str, torch.Tensor] = {}
        recon_seq_scaled: Dict[str, torch.Tensor] = {}

        for nt in self.node_types:
            proj_hidden = proj_seq_dict.get(nt, None)
            gru_hidden = gru_out.get(nt, None)

            if (proj_hidden is not None and proj_hidden.numel() != 0) and (gru_hidden is not None and gru_hidden.numel() != 0):
                if proj_hidden.shape[1] != gru_hidden.shape[1]:
                    gru_rs = F_nn.interpolate(
                        gru_hidden.permute(0, 2, 1),
                        size=proj_hidden.shape[1],
                        mode="linear",
                        align_corners=False,
                    ).permute(0, 2, 1)
                else:
                    gru_rs = gru_hidden
                recon_cat = torch.cat([proj_hidden, gru_rs], dim=-1)  # (N,T,2H)
                B, T_tmp, D_tmp = recon_cat.shape
                proj_fn = self.decoder_input_proj[nt]
                recon_hidden = proj_fn(recon_cat.reshape(-1, D_tmp)).view(B, T_tmp, self.hidden_dim)
            elif proj_hidden is not None and proj_hidden.numel() != 0:
                recon_hidden = proj_hidden
            elif gru_hidden is not None and gru_hidden.numel() != 0:
                target_T = self.temporal_T
                if gru_hidden.shape[1] != target_T:
                    recon_hidden = F_nn.interpolate(
                        gru_hidden.permute(0, 2, 1),
                        size=target_T,
                        mode="linear",
                        align_corners=False,
                    ).permute(0, 2, 1)
                else:
                    recon_hidden = gru_hidden
            else:
                # 完全没有可用时序表示
                recon_feature_dict[nt] = torch.zeros((0,), device=device)
                recon_seq_denorm[nt] = torch.zeros((0,), device=device)
                recon_seq_scaled[nt] = torch.zeros((0,), device=device)
                continue

            # feature decoder (TemporalDecoder)
            dec = self.feature_decoders[nt]
            recon_feature = dec(recon_hidden)      # (N,T,F_norm)
            recon_feature_dict[nt] = recon_feature

            # NodeDecoder 做 scale/bias + 反归一化
            stats_nt = stats_dict.get(nt, {"mean": None, "std": None, "orig_T": 0})
            _, recon_denorm = self.denorm_decoders.forward_feature_and_denorm(
                recon_feature,
                stats_nt,
                node_type=nt,
            )
            recon_seq_denorm[nt] = recon_feature
            recon_seq_scaled[nt] = recon_denorm

        # ---------- 10) z_dict: pooling proj_seq_dict 到 (N,H) ----------
        z_dict: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            p = proj_seq_dict.get(nt, None)
            if p is None or p.numel() == 0:
                # fallback: 用 encoded 的均值
                enc = encoded_dict.get(nt, None)
                if enc is None or enc.numel() == 0:
                    z_dict[nt] = torch.zeros((0, self.hidden_dim), device=device)
                else:
                    z_dict[nt] = enc.mean(dim=1)   # (N,H)
            else:
                if p.ndim == 3:
                    z_dict[nt] = p.mean(dim=1)     # (N,H)
                elif p.ndim == 2:
                    z_dict[nt] = p
                else:
                    raise ValueError(f"[DynamicHeteroGNN] proj_seq_dict['{nt}'] ndim={p.ndim} invalid")

        # ---------- 11) global_seq ----------
        valid_means = [
            proj_seq_dict[nt].mean(dim=0)
            for nt in self.node_types
            if proj_seq_dict.get(nt, None) is not None and proj_seq_dict[nt].numel() != 0
        ]
        if len(valid_means) > 0:
            global_seq = self.global_proj(torch.stack(valid_means).mean(dim=0))
        else:
            global_seq = torch.zeros((self.hidden_dim,), device=device)

        recon_denorm_dict = {"recon_denorm_dict": recon_seq_scaled}

        return (
            z_dict,
            gru_out,
            proj_seq_dict,
            recon_seq_denorm,
            recon_seq_scaled,
            global_seq,
            recon_feature_dict,
            recon_denorm_dict,
        )