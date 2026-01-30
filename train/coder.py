import os
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch_geometric.data import HeteroData

# 诊断控制：设置环境变量 DUMP_RUNTIME_DIAG=1 可以在第一次 forward 时打印一些形状信息
_DIAG_ENABLED = os.getenv("DUMP_RUNTIME_DIAG", "0") == "1"


# ============================================================
# NodeEncoder: 单模态节点时序编码器
# ============================================================
class NodeEncoder(nn.Module):
    """
    NodeEncoder
    -----------
    职责：
    - 接收单模态节点时间序列 x_seq: (N, T_raw, F_in)
    - 做标准化 (per-node, per-feature)
    - 在时间维上重采样到固定 T_spatial
    - 用 Depthwise + Pointwise 1D Conv 提取时序特征
    - 输出时序隐变量 h: (N, T_spatial, H)

    设计目标：
    - 输入接口简单，不要求显式声明 in_dim；
    - 对于不同 node_type（fmri/eeg）统一编码逻辑；
    - stats 中保存 mean/std/orig_T，供 NodeDecoder 反归一化使用。
    """

    def __init__(self, spatial_T: int = 384, hidden_dim: int = 64):
        super().__init__()
        self.spatial_T = int(spatial_T)
        self.hidden_dim = int(hidden_dim)

        # 动态根据输入 F_dim 初始化 conv
        self.dw_conv: Optional[nn.Conv1d] = None
        self.pw_conv: Optional[nn.Conv1d] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, Dict[str, torch.Tensor]]:
        """
        x: (N, T_raw, F_in)

        返回:
        - h: (N, T_spatial, hidden_dim)
        - N: 节点数
        - stats: {"mean": (N,1,F_in), "std": (N,1,F_in), "orig_T": int}
        """
        device = x.device if isinstance(x, torch.Tensor) else "cpu"

        # ----- 空输入 -----
        if x is None or (isinstance(x, torch.Tensor) and x.numel() == 0):
            if self.pw_conv is None:
                # lazy init 保证 hidden_dim 一致
                self.pw_conv = nn.Conv1d(1, self.hidden_dim, kernel_size=1, bias=False).to(device)
            h_empty = torch.zeros((0, self.spatial_T, self.hidden_dim), device=device)
            stats_empty = {"mean": None, "std": None, "orig_T": 0}
            return h_empty, 0, stats_empty

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"[NodeEncoder] x must be torch.Tensor, got {type(x)}")

        if x.ndim != 3:
            raise ValueError(f"[NodeEncoder] Expected 3D tensor (N,T,F), got {x.ndim}D: shape={x.shape}")

        N, T_raw, F_dim = x.shape

        # ----- Normalize (per-node, per-feature) -----
        mean = x.mean(dim=1, keepdim=True)              # (N,1,F)
        std = x.std(dim=1, keepdim=True) + 1e-8
        x_norm = torch.clamp((x - mean) / std, -10.0, 10.0)

        # ----- Resample to spatial_T -----
        # x_norm: (N,T,F) -> (N,F,T) for conv
        x_perm = x_norm.permute(0, 2, 1)  # (N,F,T_raw)

        if T_raw > self.spatial_T:
            # 自适应平均池化到固定长度
            x_resampled = F_nn.adaptive_avg_pool1d(x_perm, self.spatial_T)
        elif T_raw < self.spatial_T:
            # 线性插值到固定长度
            x_resampled = F_nn.interpolate(
                x_perm, size=self.spatial_T, mode="linear", align_corners=False
            )
        else:
            x_resampled = x_perm

        # 现在 x_resampled: (N, F, T_spatial)
        x_proc = x_resampled  # (N,F,T_spatial)

        # ----- Conv 初始化 -----
        if self.dw_conv is None or self.dw_conv.in_channels != F_dim:
            # Depthwise Conv: 每个特征一条独立的 1D conv
            self.dw_conv = nn.Conv1d(
                F_dim, F_dim, kernel_size=3, padding=1, groups=F_dim, bias=False
            ).to(device)
            # Pointwise Conv: 融合到 hidden_dim
            self.pw_conv = nn.Conv1d(
                F_dim, self.hidden_dim, kernel_size=1, bias=False
            ).to(device)

        x_dw = self.dw_conv(x_proc)                # (N,F,T_spatial)
        x_pw = self.pw_conv(F_nn.relu(x_dw))       # (N,H,T_spatial)
        x_hidden = x_pw.permute(0, 2, 1).contiguous()  # (N,T_spatial,H)

        stats = {"mean": mean, "std": std, "orig_T": int(T_raw)}

        if _DIAG_ENABLED:
            print(f"[NodeEncoder DIAG] x: {tuple(x.shape)} -> h: {tuple(x_hidden.shape)}, orig_T={T_raw}")

        return x_hidden, N, stats


# ============================================================
# NodeDecoder: per-modality 解码 + 反归一化
# ============================================================
class NodeDecoder(nn.Module):
    """
    NodeDecoder
    -----------
    职责：
    - 按 node_type 管理解码 MLP、scale/bias，负责从 normalized 特征恢复原量纲时序。
    - 输入 recon_feature 通常是 feature decoder 的输出 (N,T,F_rec)（仍然处于 normalized 空间）。
    - stats 则来自 NodeEncoder（mean/std/orig_T），用于反归一化。

    接口：
    - __init__(node_types, out_dims, hidden_dim, extra_hidden, num_layers, init_scales, scale_mode)
    - forward(recon_feature, stats, node_type) -> recon_denorm (N,T,F_rec)
    - forward_feature_and_denorm(...) -> (recon_learned_before_scale, recon_denorm)
    """

    def __init__(
        self,
        node_types: List[str],
        out_dims: Dict[str, int],
        hidden_dim: int = 128,
        extra_hidden: int = 128,
        num_layers: int = 3,
        init_scales: Optional[Dict[str, float]] = None,
        scale_mode: str = "log",
    ):
        """
        参数：
        - node_types: 模态名称列表（如 ["fmri", "eeg"]）
        - out_dims: 每个 node_type 的输出特征维度（F_rec）
        - hidden_dim / extra_hidden / num_layers: decoder MLP 的结构参数
        - init_scales: 初始 scale 配置（nt -> float）
        - scale_mode: "fixed" | "learned" | "log"（推荐 "log"）
        """
        super().__init__()
        self.node_types = list(node_types)
        self.out_dims = {nt: int(out_dims.get(nt, 1)) for nt in self.node_types}
        self.hidden_dim = int(hidden_dim)
        self.extra_hidden = int(extra_hidden)
        self.num_layers = int(num_layers)
        self.scale_mode = str(scale_mode)
        self.init_scales = dict(init_scales) if init_scales is not None else {}

        self.decoders = nn.ModuleDict()
        self.bias = nn.ParameterDict()
        self._create_decoders_and_scales()

        self._diag_printed = False

    def _create_decoder_mlp(self, in_dim: int, out_dim: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        d_in = in_dim
        for i in range(self.num_layers):
            d_out = self.extra_hidden if i < self.num_layers - 1 else out_dim
            layers.append(nn.Linear(d_in, d_out))
            if i < self.num_layers - 1:
                layers.append(nn.LeakyReLU(0.1))
            d_in = d_out
        return nn.Sequential(*layers)

    def _create_decoders_and_scales(self):
        for nt in self.node_types:
            F_rec = self.out_dims.get(nt, 1)
            in_dim = F_rec
            out_dim = F_rec

            self.decoders[nt] = self._create_decoder_mlp(in_dim, out_dim)
            # bias per feature
            self.bias[nt] = nn.Parameter(torch.zeros(out_dim))

            init_val = float(self.init_scales.get(nt, 1.0))
            if self.scale_mode == "fixed":
                self.register_buffer(f"scale_fixed_{nt}", torch.ones(out_dim) * init_val)
            elif self.scale_mode == "learned":
                self.register_parameter(f"scale_{nt}", nn.Parameter(torch.ones(out_dim) * init_val))
            elif self.scale_mode == "log":
                self.register_parameter(f"log_scale_{nt}", nn.Parameter(torch.log(torch.ones(out_dim) * init_val)))
            else:
                raise ValueError(f"[NodeDecoder] Unknown scale_mode: {self.scale_mode}")

    def get_scale(self, node_type: str) -> torch.Tensor:
        if self.scale_mode == "fixed":
            return getattr(self, f"scale_fixed_{node_type}")
        elif self.scale_mode == "learned":
            return getattr(self, f"scale_{node_type}")
        elif self.scale_mode == "log":
            s = getattr(self, f"log_scale_{node_type}")
            return torch.exp(s)
        else:
            raise ValueError(f"[NodeDecoder] Unknown scale_mode: {self.scale_mode}")

    def _set_scale_requires_grad(self, nt: str, flag: bool):
        if self.scale_mode == "fixed":
            return
        if self.scale_mode == "learned":
            p = getattr(self, f"scale_{nt}")
            p.requires_grad = flag
        elif self.scale_mode == "log":
            p = getattr(self, f"log_scale_{nt}")
            p.requires_grad = flag

    def freeze_scale(self, node_type: Optional[str] = None):
        if node_type is None:
            for nt in self.node_types:
                self._set_scale_requires_grad(nt, False)
        else:
            self._set_scale_requires_grad(node_type, False)

    def unfreeze_scale(self, node_type: Optional[str] = None):
        if node_type is None:
            for nt in self.node_types:
                self._set_scale_requires_grad(nt, True)
        else:
            self._set_scale_requires_grad(node_type, True)

    def forward(self, recon_feature: torch.Tensor, stats: Dict[str, Any], node_type: str = "generic") -> torch.Tensor:
        """
        recon_feature: (N, T, F_rec) — normalized 空间的重构特征
        stats: {"mean": (N,1,F), "std": (N,1,F), "orig_T": int}
        返回 recon_denorm: (N, T, F_rec)，为原量纲时序重构
        """
        if recon_feature is None or recon_feature.numel() == 0:
            return recon_feature

        if node_type not in self.decoders:
            raise KeyError(f"[NodeDecoder] Unknown node_type '{node_type}' in decoders.")

        N, T, F_in = recon_feature.shape
        F_rec = self.out_dims.get(node_type, F_in)

        recon_flat = recon_feature.reshape(-1, F_in)           # (N*T, F_rec)
        recon_learned = self.decoders[node_type](recon_flat).reshape(N, T, F_rec)  # (N,T,F_rec)

        scale = self.get_scale(node_type).view(1, 1, -1)       # (1,1,F_rec)
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

        if _DIAG_ENABLED and not self._diag_printed:
            print(f"[NodeDecoder DIAG] node_type={node_type}, recon_feature={tuple(recon_feature.shape)}, "
                  f"recon_denorm={tuple(recon_denorm.shape)}")
            self._diag_printed = True

        return recon_denorm

    def forward_feature_and_denorm(
        self,
        recon_feature: torch.Tensor,
        stats: Dict[str, Any],
        node_type: str = "generic",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
        - recon_learned: scale/bias 之前的 decoder 输出 (N,T,F_rec)
        - recon_denorm: scale/bias + 反归一化之后的原量纲输出 (N,T,F_rec)
        """
        if recon_feature is None or recon_feature.numel() == 0:
            return recon_feature, recon_feature

        if node_type not in self.decoders:
            raise KeyError(f"[NodeDecoder] Unknown node_type '{node_type}' in decoders.")

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


# ============================================================
# GraphEncoder: 从 HeteroData 统一调用 NodeEncoder
# ============================================================
class GraphEncoder(nn.Module):
    """
    GraphEncoder
    ------------
    职责：
    - 管理每个 node_type 的 NodeEncoder；
    - 从 HeteroData 中取出 data[nt].x_seq，编码为统一时长的隐表示；
    - 输出：
      - x_dict: nt -> (N, T_spatial, hidden_dim)
      - num_nodes_dict: nt -> N
      - stats_dict: nt -> {"mean","std","orig_T"}
      - x_raw_map: nt -> 原始 x_seq
      - edge_index_dict: (src, rel, dst) -> edge_index (来自 HeteroData)

    说明：
    - 这是“信号到 latent”唯一的入口。DynamicHeteroGNN 不再自己构造 NodeEncoder。
    """

    def __init__(self, spatial_T: int = 384, hidden_dim: int = 64, debug: bool = False):
        super().__init__()
        self.spatial_T = int(spatial_T)
        self.hidden_dim = int(hidden_dim)
        self.encoders = nn.ModuleDict()
        self.debug = bool(debug)

    def forward(
        self,
        data: HeteroData,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]:
        """
        data: HeteroData，要求每个 node_type 有 .x_seq: (N,T,F)

        返回:
        - x_dict: node_type -> (N, T_spatial, H)
        - num_nodes_dict: node_type -> N
        - stats_dict: node_type -> {"mean","std","orig_T"}
        - x_raw_map: node_type -> 原始 x_seq (N,T,F)
        - edge_index_dict: (src,rel,dst) -> edge_index Tensor(2,E)
        """
        if not isinstance(data, HeteroData):
            raise TypeError(f"[GraphEncoder] data must be HeteroData, got {type(data)}")

        x_dict: Dict[str, torch.Tensor] = {}
        num_nodes_dict: Dict[str, int] = {}
        stats_dict: Dict[str, Dict[str, torch.Tensor]] = {}
        x_raw_map: Dict[str, torch.Tensor] = {}
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}

        device = None

        # --- 节点编码 ---
        for nt in data.node_types:
            x_seq = getattr(data[nt], "x_seq", None)
            if x_seq is None:
                raise RuntimeError(f"[GraphEncoder] data['{nt}'] missing x_seq attribute.")

            if not isinstance(x_seq, torch.Tensor):
                raise TypeError(f"[GraphEncoder] data['{nt}'].x_seq must be torch.Tensor, got {type(x_seq)}")

            if device is None:
                device = x_seq.device

            # 初始化 NodeEncoder
            if nt not in self.encoders:
                self.encoders[nt] = NodeEncoder(self.spatial_T, self.hidden_dim).to(x_seq.device)

            enc: NodeEncoder = self.encoders[nt]
            h, N, stats = enc(x_seq)

            x_dict[nt] = h
            num_nodes_dict[nt] = N
            stats_dict[nt] = stats
            x_raw_map[nt] = x_seq

            if self.debug:
                print(f"[GraphEncoder] nt={nt}, x_seq={tuple(x_seq.shape)}, h={tuple(h.shape)}, N={N}")

        # --- edge_index_dict ---
        if device is None:
            # 如果没有节点（极端情况），用 CPU 设备
            device = torch.device("cpu")

        for (src, rel, dst) in data.edge_types:
            obj = data[(src, rel, dst)]
            edge_index = getattr(obj, "edge_index", None)
            if edge_index is not None and isinstance(edge_index, torch.Tensor):
                edge_index_dict[(src, rel, dst)] = edge_index.to(device).long()
            else:
                edge_index_dict[(src, rel, dst)] = torch.empty((2, 0), dtype=torch.long, device=device)

        return x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict


# ============================================================
# 图解码器（目前建议你直接在 DynamicHeteroGNN 里用 NodeDecoder）
# ============================================================
class GraphDecoder(nn.Module):
    """
    GraphDecoder (旧接口)
    ----------------------
    警告：
    - 原有版本尝试在这里 new NodeDecoder()，但没有传入 node_types/out_dims，实际上是错误的。
    - 建议在 DynamicHeteroGNN 中集中管理 NodeDecoder，使其知道每个 node_type 的 out_dims。

    这里的 GraphDecoder 保留一个简单、明确的接口：
    - 你必须在构造时显式提供 node_types 和 out_dims；
    - forward 时只做按模态调用 NodeDecoder。
    """

    def __init__(
        self,
        node_types: List[str],
        out_dims: Dict[str, int],
        hidden_dim: int = 128,
        extra_hidden: int = 128,
        num_layers: int = 3,
        init_scales: Optional[Dict[str, float]] = None,
        scale_mode: str = "log",
    ):
        super().__init__()
        self.node_types = list(node_types)
        self.out_dims = dict(out_dims)
        self.decoders = NodeDecoder(
            node_types=self.node_types,
            out_dims=self.out_dims,
            hidden_dim=hidden_dim,
            extra_hidden=extra_hidden,
            num_layers=num_layers,
            init_scales=init_scales,
            scale_mode=scale_mode,
        )

    def forward(
        self,
        recon_dict: Dict[str, torch.Tensor],
        stats_dict: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        recon_dict: node_type -> recon_feature (N,T,F_rec) in normalized space
        stats_dict: node_type -> stats from NodeEncoder
        """
        out: Dict[str, torch.Tensor] = {}
        for nt, recon_norm in recon_dict.items():
            stats = stats_dict.get(nt, {"mean": None, "std": None, "orig_T": 0})
            out[nt] = self.decoders(recon_norm, stats, node_type=nt)
        return out


# ============================================================
# ProjectionHead: 每个模态一个小投影头
# ============================================================
class ProjectionHead(nn.Module):
    """
    ProjectionHead
    --------------
    职责：
    - 为每种模态提供一个小的投影头；
    - 输入为 (N, T, H) 的时序隐变量，先在时间维上做池化，再过两层 MLP 得到 (N, latent_dim)。

    说明：
    - 当前实现为 'fmri' 和 'eeg' 分别建一个 head；
    - 如果将来你有更多 node_type，可以在构造时扩展 heads 字典。
    """

    def __init__(self, hidden_dim: int, latent_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.dropout = nn.Dropout(float(dropout))

        self.heads = nn.ModuleDict({
            "fmri": nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.latent_dim),
            ),
            "eeg": nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.latent_dim),
            ),
        })

    def forward(self, seq: torch.Tensor, modality: str) -> torch.Tensor:
        """
        seq: (N, T, H) -> returns (N, latent_dim)
        modality: "fmri" 或 "eeg"
        """
        if seq is None or seq.numel() == 0:
            return seq
        if modality not in self.heads:
            raise KeyError(f"[ProjectionHead] Unknown modality '{modality}'")

        pooled = seq.mean(dim=1)                # (N,H)
        proj = self.heads[modality](pooled)     # (N,latent_dim)
        return proj


# ============================================================
# TemporalDecoder: 时序卷积解码器
# ============================================================
class TemporalDecoder(nn.Module):
    """
    TemporalDecoder
    ---------------
    职责：
    - 接受 recon_hidden: (N, T, H) 的时序隐状态
    - 通过 1D Conv 堆栈 + residual shortcut 生成 recon_feature: (N, T, F_out)

    设计要点：
    - pre_proj: in_dim -> channels（通道扩展）
    - conv_stack: 多层 Conv1d + ReLU (+ Dropout)
    - final_proj: channels -> out_dim
    - shortcut: in_dim -> out_dim（避免 collapse）
    - LayerNorm: 仅在 out_dim > 1 时启用，否则用 Identity 避免 collapse
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        channels: int = 128,
        kernel_size: int = 5,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)

        # initial projection in_dim -> channels
        self.pre_proj = nn.Conv1d(self.in_dim, self.channels, kernel_size=1, bias=False)
        self.pre_act = nn.ReLU()

        # temporal conv stack
        convs: List[nn.Module] = []
        for _ in range(num_layers):
            convs.append(
                nn.Conv1d(
                    self.channels,
                    self.channels,
                    kernel_size=self.kernel_size,
                    padding=(self.kernel_size // 2),
                    bias=False,
                )
            )
            convs.append(nn.ReLU())
            if dropout and dropout > 0:
                convs.append(nn.Dropout(dropout))
        self.conv_stack = nn.Sequential(*convs)

        # final projection to out_dim
        self.final_proj = nn.Conv1d(self.channels, self.out_dim, kernel_size=1, bias=True)

        # shortcut projection (in_dim -> out_dim) to preserve signal
        self.shortcut = nn.Conv1d(self.in_dim, self.out_dim, kernel_size=1, bias=False)

        # LayerNorm over feature dim (out_dim) if >1
        if self.out_dim > 1:
            self.norm = nn.LayerNorm(self.out_dim)
        else:
            self.norm = nn.Identity()

        self._init_weights()
        self._diag_printed = False

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
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # ensure final_proj.bias small non-zero to avoid exact zero outputs at init
        if hasattr(self.final_proj, "bias") and self.final_proj.bias is not None:
            with torch.no_grad():
                self.final_proj.bias.fill_(1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, T, H_in)
        returns: (N, T, out_dim)
        """
        if x is None or x.numel() == 0:
            return x
        if x.ndim != 3:
            raise ValueError(f"[TemporalDecoder] Expected (N,T,H), got {tuple(x.shape)}")

        N, T, H_in = x.shape
        if H_in != self.in_dim:
            raise ValueError(
                f"[TemporalDecoder] Input feature dim mismatch: H_in={H_in}, expected in_dim={self.in_dim}"
            )

        # to (N,H_in,T)
        x_t = x.permute(0, 2, 1).contiguous()
        pre = self.pre_proj(x_t)    # (N,channels,T)
        pre = self.pre_act(pre)
        out = self.conv_stack(pre)  # (N,channels,T)
        out = self.final_proj(out)  # (N,out_dim,T)

        # shortcut from input
        sc = self.shortcut(x_t)     # (N,out_dim,T)
        out = out + sc              # residual add

        # back to (N,T,out_dim)
        out = out.permute(0, 2, 1).contiguous()
        out = self.norm(out)        # LayerNorm or Identity

        if _DIAG_ENABLED and not self._diag_printed:
            print(f"[TemporalDecoder DIAG] x={tuple(x.shape)} -> out={tuple(out.shape)}")
            self._diag_printed = True

        return out