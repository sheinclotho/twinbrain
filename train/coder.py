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


import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from typing import Dict

class NodeDecoder(nn.Module):
    """
    NodeDecoder:
    - Builds per-node-type MLP decoders at init (no lazy creation).
    - Supports scale modes:
        - "fixed": scale is fixed scalar (init_scale), requires_grad=False
        - "learned": direct per-feature scale parameter (scale vector)
        - "log": learn s and compute scale = exp(s) (stable, ensures positive)
    - Provides denorm(...) helper to map normalized predictions back to original space.
    - Exposes convenience methods to freeze/unfreeze scale parameters.
    """

    def __init__(self, node_types: list, out_dims: Dict[str, int], hidden_dim=128, extra_hidden=128, num_layers=3,
                 init_scales: Dict[str, float] = None, scale_mode: str = "log"):
        """
        node_types: list of node_type names (e.g. ["fmri","eeg"])
        out_dims: dict node_type -> output feature dim (F_rec)
        init_scales: map node_type -> initial scale (float)
        scale_mode: "fixed" | "learned" | "log"
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.extra_hidden = extra_hidden
        self.num_layers = num_layers
        self.node_types = list(node_types)
        self.out_dims = out_dims
        self.scale_mode = scale_mode
        self.decoders = nn.ModuleDict()
        self.bias = nn.ParameterDict()
        # scale containers:
        # - learned: ParameterDict of vectors
        # - log: ParameterDict of scalar s where scale = exp(s)
        # - fixed: buffers (registered as tensors) in self.register_buffer
        self._init_scales = init_scales or {}
        self._create_decoders_and_scales()

    def _create_decoder_mlp(self, in_dim, out_dim):
        layers = []
        in_d = in_dim
        for i in range(self.num_layers):
            out_d = self.extra_hidden if i < self.num_layers - 1 else out_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < self.num_layers - 1:
                layers.append(nn.LeakyReLU(0.1))
            in_d = out_d
        return nn.Sequential(*layers)

    def _create_decoders_and_scales(self):
        for nt in self.node_types:
            F_rec = int(self.out_dims.get(nt, self.hidden_dim))
            # decoder MLP
            self.decoders[nt] = self._create_decoder_mlp(self.hidden_dim, F_rec)
            # bias
            self.bias[nt] = nn.Parameter(torch.zeros(F_rec))
            # scale handling
            init_s = float(self._init_scales.get(nt, 1.0))
            if self.scale_mode == "fixed":
                # register as buffer (non-trainable)
                self.register_buffer(f"scale_fixed_{nt}", torch.ones(F_rec) * init_s)
            elif self.scale_mode == "learned":
                self.register_parameter(f"scale_{nt}", nn.Parameter(torch.ones(F_rec) * init_s))
            elif self.scale_mode == "log":
                # keep scalar log parameter per feature for stability
                # store s as parameter; scale = exp(s)
                self.register_parameter(f"log_scale_{nt}", nn.Parameter(torch.log(torch.ones(F_rec) * init_s)))
            else:
                raise ValueError(f"Unknown scale_mode {self.scale_mode}")

        # expose accessors for convenience
        # e.g. self.get_scale("fmri") returns a tensor (may depend on device)
    def get_scale(self, node_type: str):
        if self.scale_mode == "fixed":
            return getattr(self, f"scale_fixed_{node_type}")
        elif self.scale_mode == "learned":
            return getattr(self, f"scale_{node_type}")
        elif self.scale_mode == "log":
            s = getattr(self, f"log_scale_{node_type}")
            return torch.exp(s)
        else:
            raise ValueError(self.scale_mode)

    def freeze_scale(self, node_type: str = None):
        if node_type is None:
            for nt in self.node_types:
                self._set_scale_requires_grad(nt, False)
        else:
            self._set_scale_requires_grad(node_type, False)

    def unfreeze_scale(self, node_type: str = None):
        if node_type is None:
            for nt in self.node_types:
                self._set_scale_requires_grad(nt, True)
        else:
            self._set_scale_requires_grad(node_type, True)

    def _set_scale_requires_grad(self, nt, flag: bool):
        if self.scale_mode == "fixed":
            return
        if self.scale_mode == "learned":
            p = getattr(self, f"scale_{nt}")
            p.requires_grad = flag
        elif self.scale_mode == "log":
            p = getattr(self, f"log_scale_{nt}")
            p.requires_grad = flag

    def forward(self, recon_norm: torch.Tensor, stats: Dict[str, torch.Tensor], node_type: str = "generic"):
        """
        recon_norm: (N, T, H_in)
        stats: {"mean": (N,1,F), "std": (N,1,F)}
        Returns: recon_denorm (N,T,F)
        """
        if recon_norm is None or recon_norm.numel() == 0:
            return recon_norm
        N, T, H_in = recon_norm.shape
        mean = stats.get("mean", None)
        std = stats.get("std", None)
        F_rec = mean.shape[-1] if mean is not None else self.decoders[node_type][-1].out_features

        # project via decoder
        recon_flat = recon_norm.reshape(-1, H_in)
        recon_learned = self.decoders[node_type](recon_flat).reshape(N, T, F_rec)
        # apply scale + bias
        scale = self.get_scale(node_type)
        bias = self.bias[node_type].view(1, 1, -1)
        recon_scaled = recon_learned * scale.view(1, 1, -1) + bias
        # denorm
        if mean is not None and std is not None:
            mean_expand = mean.expand(-1, T, -1)
            std_expand = std.expand(-1, T, -1)
            recon_denorm = recon_scaled * std_expand + mean_expand
        else:
            recon_denorm = recon_scaled
        return recon_denorm

    def forward_feature_and_denorm(self, recon_norm: torch.Tensor, stats: Dict[str, torch.Tensor], node_type: str = "generic"):
        """
        Returns tuple (recon_feature, recon_denorm)
        - recon_feature: recon_learned after decoder but BEFORE scale/bias (N,T,F)
        - recon_denorm: after scale/bias and denorm (N,T,F)
        Useful if trainer wants to compute normalized-space losses.
        """
        if recon_norm is None or recon_norm.numel() == 0:
            return recon_norm, recon_norm
        N, T, H_in = recon_norm.shape
        mean = stats.get("mean", None)
        std = stats.get("std", None)
        F_rec = mean.shape[-1] if mean is not None else self.decoders[node_type][-1].out_features

        recon_flat = recon_norm.reshape(-1, H_in)
        recon_learned = self.decoders[node_type](recon_flat).reshape(N, T, F_rec)
        scale = self.get_scale(node_type)
        bias = self.bias[node_type].view(1, 1, -1)
        recon_scaled = recon_learned * scale.view(1, 1, -1) + bias
        if mean is not None and std is not None:
            mean_expand = mean.expand(-1, T, -1)
            std_expand = std.expand(-1, T, -1)
            recon_denorm = recon_scaled * std_expand + mean_expand
        else:
            recon_denorm = recon_scaled
        return recon_learned, recon_denorm
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
