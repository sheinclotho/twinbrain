import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import math


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
    """
    NodeDecoder:
    - 在初始化阶段一次性为每个 node_type 创建 decoder MLP、scale/bias 等参数（避免 lazy creation）。
    - 支持三种 scale 模式："fixed" | "learned" | "log"（推荐 "log" 用于正值与稳定性）。
    - forward(recon_feature, stats, node_type=nt) -> recon_denorm (N,T,F)
    - forward_feature_and_denorm(...) -> (recon_feature_decoded_before_scale, recon_denorm)
    """

    def __init__(
        self,
        node_types: List[str],
        out_dims: Dict[str, int],
        hidden_dim: int = 128,
        extra_hidden: int = 128,
        num_layers: int = 3,
        init_scales: Dict[str, float] = None,
        scale_mode: str = "log",
    ):
        """
        node_types: list of node type names (e.g., ["fmri","eeg"])
        out_dims: dict node_type -> output feature dim (F_rec)
        hidden_dim/extra_hidden/num_layers: MLP size params (hidden layers sizes)
        init_scales: optional initial scale per node_type
        scale_mode: "fixed" | "learned" | "log"
        """
        super().__init__()
        self.node_types = list(node_types)
        self.out_dims = {nt: int(out_dims.get(nt, 1)) for nt in self.node_types}
        self.hidden_dim = hidden_dim
        self.extra_hidden = extra_hidden
        self.num_layers = num_layers
        self.scale_mode = scale_mode
        self.init_scales = init_scales or {}

        # containers
        self.decoders = nn.ModuleDict()
        self.bias = nn.ParameterDict()
        # create decoders & scales immediately
        self._create_decoders_and_scales()

    def _create_decoder_mlp(self, in_dim: int, out_dim: int) -> nn.Sequential:
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
            F_rec = self.out_dims.get(nt, 1)
            # NOTE: Input to decoder MLP should be the recon_feature dimension (F_rec),
            # because recon_feature is produced by feature_decoders: Linear(hidden_dim -> F_rec)
            in_dim = F_rec
            out_dim = F_rec
            self.decoders[nt] = self._create_decoder_mlp(in_dim, out_dim)
            # bias per feature
            self.bias[nt] = nn.Parameter(torch.zeros(out_dim))
            # scale handling
            init_val = float(self.init_scales.get(nt, 1.0))
            if self.scale_mode == "fixed":
                # register buffer holding fixed scale vector
                self.register_buffer(f"scale_fixed_{nt}", torch.ones(out_dim) * init_val)
            elif self.scale_mode == "learned":
                # learned direct scale vector
                self.register_parameter(f"scale_{nt}", nn.Parameter(torch.ones(out_dim) * init_val))
            elif self.scale_mode == "log":
                # learn log-scale vector s; scale = exp(s)
                self.register_parameter(f"log_scale_{nt}", nn.Parameter(torch.log(torch.ones(out_dim) * init_val)))
            else:
                raise ValueError(f"Unknown scale_mode: {self.scale_mode}")

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

    def _set_scale_requires_grad(self, nt: str, flag: bool):
        if self.scale_mode == "fixed":
            return
        if self.scale_mode == "learned":
            p = getattr(self, f"scale_{nt}")
            p.requires_grad = flag
        elif self.scale_mode == "log":
            p = getattr(self, f"log_scale_{nt}")
            p.requires_grad = flag

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

    def forward(self, recon_feature: torch.Tensor, stats: dict, node_type: str = "generic"):
        """
        recon_feature: (N, T, F_rec) — this is the model's prediction in normalized space (before scale)
        stats: {"mean": (N,1,F), "std": (N,1,F)}
        node_type: modality name
        returns recon_denorm: (N, T, F_rec)
        """
        if recon_feature is None or recon_feature.numel() == 0:
            return recon_feature

        N, T, F_in = recon_feature.shape
        F_rec = self.out_dims.get(node_type, F_in)

        # decoder expects input dim == F_rec
        recon_flat = recon_feature.reshape(-1, F_in)  # (N*T, F_rec)
        recon_learned = self.decoders[node_type](recon_flat).reshape(N, T, F_rec)  # (N,T,F_rec)

        # apply scale + bias
        scale = self.get_scale(node_type).view(1, 1, -1)
        bias = self.bias[node_type].view(1, 1, -1)
        recon_scaled = recon_learned * scale + bias

        # denormalize if mean/std provided
        mean = stats.get("mean", None)
        std = stats.get("std", None)
        if mean is not None and std is not None:
            mean_expand = mean.expand(-1, T, -1)
            std_expand = std.expand(-1, T, -1)
            recon_denorm = recon_scaled * std_expand + mean_expand
        else:
            recon_denorm = recon_scaled

        return recon_denorm

    def forward_feature_and_denorm(self, recon_feature: torch.Tensor, stats: dict, node_type: str = "generic"):
        """
        Return (recon_learned_before_scale, recon_denorm)
        recon_learned_before_scale: (N,T,F)
        recon_denorm: after scale/bias and denorm
        """
        if recon_feature is None or recon_feature.numel() == 0:
            return recon_feature, recon_feature

        N, T, F_in = recon_feature.shape
        F_rec = self.out_dims.get(node_type, F_in)

        recon_flat = recon_feature.reshape(-1, F_in)
        recon_learned = self.decoders[node_type](recon_flat).reshape(N, T, F_rec)

        scale = self.get_scale(node_type).view(1, 1, -1)
        bias = self.bias[node_type].view(1, 1, -1)
        recon_scaled = recon_learned * scale + bias

        mean = stats.get("mean", None)
        std = stats.get("std", None)
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


# train/coder.py


class TemporalDecoder(nn.Module):
    """
    TemporalDecoder with residual shortcut and conditional LayerNorm:
    - input: recon_hidden (N, T, H)
    - output: recon_feature (N, T, F)
    Notes:
      - If out_dim == 1, LayerNorm over feature dim collapses to zero; use Identity instead.
      - Uses Conv1d temporal stack with same padding and a residual shortcut from input.
      - Stable initialization for Conv1d/Linear and LayerNorm defaults.
    """
    def __init__(self, in_dim: int, out_dim: int, channels: int = 128, kernel_size: int = 5, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channels = channels
        self.kernel_size = kernel_size

        # initial projection in_dim -> channels
        self.pre_proj = nn.Conv1d(in_dim, channels, kernel_size=1, bias=False)
        self.pre_act = nn.ReLU()

        # temporal conv stack
        convs = []
        for i in range(num_layers):
            convs.append(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=(kernel_size // 2), bias=False))
            convs.append(nn.ReLU())
            if dropout and dropout > 0:
                convs.append(nn.Dropout(dropout))
        self.conv_stack = nn.Sequential(*convs)

        # final projection to out_dim
        self.final_proj = nn.Conv1d(channels, out_dim, kernel_size=1, bias=True)

        # shortcut projection (in_dim -> out_dim) to preserve signal & avoid collapse
        self.shortcut = nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False)

        # LayerNorm: use only if out_dim > 1; otherwise use identity to avoid collapse
        if out_dim > 1:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.Identity()

        # Stable initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                # layer norm weight=1 bias=0
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # ensure final_proj.bias small non-zero to avoid exact zero outputs at init
        if hasattr(self.final_proj, "bias") and self.final_proj.bias is not None:
            with torch.no_grad():
                self.final_proj.bias.fill_(1e-2)

    def forward(self, x: torch.Tensor):
        """
        x: (N, T, H)
        returns: (N, T, out_dim)
        """
        if x is None or x.numel() == 0:
            return x
        n, t, h = x.shape
        # to (N, H, T)
        x_t = x.permute(0, 2, 1).contiguous()
        pre = self.pre_proj(x_t)   # (N, channels, T)
        pre = self.pre_act(pre)
        out = self.conv_stack(pre)  # (N, channels, T)
        out = self.final_proj(out)  # (N, out_dim, T)

        # shortcut from input
        try:
            sc = self.shortcut(x_t)     # (N, out_dim, T)
        except Exception as e:
            # fallback: if shapes mismatch, skip shortcut (use zeros with correct shape)
            # This maintains gradient flow while avoiding dimension mismatches
            logging.warning(f"[TemporalDecoder] Shortcut failed: {e}. Using zero shortcut.")
            sc = torch.zeros((x_t.size(0), self.out_dim, x_t.size(2)), device=x_t.device)

        out = out + sc              # residual add
        # back to (N, T, out_dim)
        out = out.permute(0, 2, 1).contiguous()
        # conditional normalization (Identity for out_dim==1)
        out = self.norm(out)
        return out