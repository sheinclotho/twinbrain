import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F_nn


class NodeEncoder(nn.Module):
    def __init__(self, spatial_T: int = 384, hidden_dim: int = 64):
        super().__init__()
        self.spatial_T = spatial_T
        self.hidden_dim = hidden_dim

        # 动态根据输入 F_dim 初始化
        self.dw_conv = None
        self.pw_conv = None

    def forward(self, x: torch.Tensor):
        device = x.device if isinstance(x, torch.Tensor) else "cpu"

        # ----- 空输入 -----
        if x is None or x.numel() == 0:
            if self.pw_conv is None:
                self.pw_conv = nn.Linear(1, self.hidden_dim).to(device)
            return (
                torch.zeros((0, self.spatial_T, self.hidden_dim), device=device),
                0,
                {"mean": None, "std": None, "orig_T": 0},
            )

        if x.ndim != 3:
            raise ValueError(
                f"[NodeEncoder DEBUG] Expected 3D tensor (N,T,F), got {x.ndim}D: shape={x.shape}"
            )

        N, T_raw, F_dim = x.shape

        # ----- Normalize -----
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        x_norm = torch.clamp((x - mean) / std, -10.0, 10.0)

        # ----- Resample to spatial_T -----
        x_perm = x_norm.permute(0, 2, 1)  # (N,F,T)

        if T_raw > self.spatial_T:
            x_resampled = F_nn.adaptive_avg_pool1d(x_perm, self.spatial_T)
        elif T_raw < self.spatial_T:
            x_resampled = F_nn.interpolate(
                x_perm, size=self.spatial_T, mode="linear", align_corners=False
            )
        else:
            x_resampled = x_perm

        x_proc = x_resampled.permute(0, 2, 1).contiguous()  # (N,T,F)

        # ----- Conv 初始化 -----
        if self.dw_conv is None or self.dw_conv.in_channels != F_dim:
            self.dw_conv = nn.Conv1d(
                F_dim, F_dim, kernel_size=3, padding=1, groups=F_dim
            ).to(device)
            self.pw_conv = nn.Conv1d(
                F_dim, self.hidden_dim, kernel_size=1
            ).to(device)

        # Conv 输入: (N,T,F) → (N,F,T)
        x_conv = x_proc.permute(0, 2, 1)

        x_dw = self.dw_conv(x_conv)
        x_pw = self.pw_conv(F_nn.relu(x_dw))

        x_hidden = x_pw.permute(0, 2, 1).contiguous()  # (N,T,H)

        stats = {"mean": mean, "std": std, "orig_T": T_raw}
        return x_hidden, N, stats


class NodeDecoder(nn.Module):
    def __init__(self, hidden_dim=128, extra_hidden=128, num_layers=3):
        super().__init__()
        self.decoders = nn.ModuleDict()      # 每个 node_type 独立 decoder
        self.scale = nn.ParameterDict()      # 可学习缩放
        self.bias = nn.ParameterDict()       # 可学习偏置
        self.pre_scale = nn.ParameterDict()  # GRU 输出前可学习放大
        self.hidden_dim = hidden_dim
        self.extra_hidden = extra_hidden
        self.num_layers = num_layers

    def forward(self, recon_norm: torch.Tensor, stats: dict, node_type: str = "generic", debug=False):
        """
        recon_norm: (N, T, H_in)
        stats: {"mean": (N,1,F), "std": (N,1,F)}
        node_type: str
        """
        if recon_norm is None or recon_norm.numel() == 0:
            return recon_norm

        N, T, H_in = recon_norm.shape
        mean = stats.get("mean", None)
        std = stats.get("std", None)
        F_rec = mean.shape[-1] if mean is not None else H_in

        # 初始化 per-node_type decoder
        if node_type not in self.decoders:
            layers = []
            in_dim = H_in
            for i in range(self.num_layers):
                out_dim = self.extra_hidden if i < self.num_layers - 1 else F_rec
                layers.append(nn.Linear(in_dim, out_dim))
                if i < self.num_layers - 1:
                    layers.append(nn.LeakyReLU(0.1))
                in_dim = out_dim
            self.decoders[node_type] = nn.Sequential(*layers).to(recon_norm.device)

            # scale / bias 初始化
            init_scale = 5.0 if node_type == "fmri" else 8.0
            self.scale[node_type] = nn.Parameter(torch.ones(F_rec, device=recon_norm.device) * init_scale)
            self.bias[node_type] = nn.Parameter(torch.zeros(F_rec, device=recon_norm.device))
            self.pre_scale[node_type] = nn.Parameter(torch.ones(1, device=recon_norm.device))

        # 可选 GRU 输出放大
        recon_norm = recon_norm * self.pre_scale[node_type]

        # decoder 投影
        recon_flat = recon_norm.reshape(-1, H_in)
        recon_learned = self.decoders[node_type](recon_flat).reshape(N, T, F_rec)

        # scale + bias
        recon_scaled = recon_learned * self.scale[node_type] + self.bias[node_type]

        # 残差重建（如果 mean/std 已知）
        if mean is not None and std is not None:
            mean_expand = mean.expand(-1, T, -1)
            std_expand = std.expand(-1, T, -1)
            recon_denorm = recon_scaled * std_expand + mean_expand
        else:
            print("No mean and std")
            recon_denorm = recon_scaled

        # if debug:
        #     print(f"[NodeDecoder:{node_type}] recon_learned mean={recon_learned.mean():.5f} std={recon_learned.std():.5f}")
        #     print(f"[NodeDecoder:{node_type}] recon_scaled mean={recon_scaled.mean():.5f} std={recon_scaled.std():.5f}")
        #     print(f"[NodeDecoder:{node_type}] recon_denorm mean={recon_denorm.mean():.5f} std={recon_denorm.std():.5f}")

        return recon_denorm



class GraphEncoder(nn.Module):
    """
    GraphEncoder: 管理每个 node_type 的 NodeEncoder，输出字典：
      - x_dict: nt -> (N, T_spatial, hidden_dim)
      - num_nodes_dict: nt -> N
      - stats_dict: nt -> {"mean","std","orig_T"}
      - x_raw_map: nt -> 原始输入
      - edge_index_dict: (src, rel, dst) -> edge_index
    """
    def __init__(self, spatial_T: int = 384, hidden_dim: int = 64, debug: bool = False):
        super().__init__()
        self.spatial_T = spatial_T
        self.hidden_dim = hidden_dim
        self.encoders = nn.ModuleDict()
        self.debug = debug

    def forward(self, data: HeteroData):
        x_dict: Dict[str, torch.Tensor] = {}
        num_nodes_dict: Dict[str, int] = {}
        stats_dict: Dict[str, Dict[str, torch.Tensor]] = {}
        x_raw_map: Dict[str, torch.Tensor] = {}
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        
        # --- 节点编码 ---
        for nt in data.node_types:
            x_seq = getattr(data[nt], 'x_seq', None)

            if nt not in self.encoders:
                self.encoders[nt] = NodeEncoder(self.spatial_T, self.hidden_dim)

            x_hidden, N, stats = self.encoders[nt](x_seq)

            x_dict[nt] = x_hidden
            num_nodes_dict[nt] = N
            stats_dict[nt] = stats
            x_raw_map[nt] = x_seq


        # --- 构建 edge_index_dict ---
        for src, rel, dst in data.edge_types:
            edge_index = getattr(data[src, rel, dst], 'edge_index', None)
            if edge_index is not None:
                edge_index_dict[(src, rel, dst)] = edge_index

            else:
                edge_index_dict[(src, rel, dst)] = torch.empty((2,0), dtype=torch.long, device=x_hidden.device)

        return x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict



class GraphDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoders = nn.ModuleDict()

    def forward(self, recon_dict: Dict[str, torch.Tensor], stats_dict: Dict[str, Dict[str, torch.Tensor]]):
        out = {}
        for nt, recon_norm in recon_dict.items():
            if nt not in self.decoders:
                self.decoders[nt] = NodeDecoder()
            stats = stats_dict.get(nt, {"mean": None, "std": None})
            out[nt] = self.decoders[nt](recon_norm, stats, node_type=nt)
        return out


class ProjectionHead(nn.Module):
    """
    Per-modality small projector head.
    Use separate heads for 'fmri' and 'eeg' to avoid one modality dominating.
    Input: (N, T, H) -> Output: (N, latent_dim)
    We'll pool over time (mean) after projection to get a modality vector.
    """
    def __init__(self, hidden_dim: int, latent_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(dropout)
        # separate heads
        self.heads = nn.ModuleDict({
            "fmri": nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim)
            ),
            "eeg": nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim)
            )
        })

    def forward(self, seq: torch.Tensor, modality: str):
        """
        seq: (N, T, H) -> returns (N, latent_dim)
        """
        if seq is None or seq.numel() == 0:
            return seq
        # temporal pooling (mean)
        pooled = seq.mean(dim=1)  # (N, H)
        proj = self.heads[modality](pooled)  # (N, latent_dim)
        return proj
