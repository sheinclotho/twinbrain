import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from train.aligner import TemporalCrossAligner
from train.coder import NodeEncoder, NodeDecoder, ProjectionHead

logger = logging.getLogger()

class DynamicHeteroGNN(nn.Module):
    def __init__(  self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        node_feature_dims: Optional[Dict[str, int]] = None,
        hidden_dim: int = 128,
        num_layers: int = 8,
        dropout: float = 0.3,
        temporal_T: int = 200,
        spatial_T: int = 384,
        debug: bool = False
        ):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.temporal_T = temporal_T
        self.spatial_T = spatial_T
        self.debug = debug

        if node_feature_dims is None:
            self.node_feature_dims = {nt: 1 for nt in self.node_types}
        else:
            self.node_feature_dims = node_feature_dims

        # Encoders & decoders
        self.encoders = nn.ModuleDict({nt: NodeEncoder(self.spatial_T, self.hidden_dim) for nt in self.node_types})
        self.denorm_decoders = nn.ModuleDict({nt: NodeDecoder() for nt in self.node_types})

        # backbone
        self.convs = nn.ModuleList()
        self.grus = nn.ModuleDict({nt: nn.GRU(hidden_dim, hidden_dim, batch_first=True) for nt in self.node_types})
        self.temporal_projs = nn.ModuleDict({nt: nn.ModuleDict() for nt in self.node_types})
        self.temporal_align = TemporalCrossAligner(hidden_dim=hidden_dim, dropout=dropout)
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.feature_decoders = nn.ModuleDict({
            nt: nn.Linear(self.hidden_dim, self.node_feature_dims[nt]) for nt in self.node_types
        })
        # Projection head: per-modality small projection to latent (used for alignment / modality pooling)
        self.proj_head = ProjectionHead(hidden_dim=self.hidden_dim, latent_dim=self.hidden_dim)

    # ---------- forward ----------
    def forward(self, data: HeteroData, edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]):
        device = next(self.parameters()).device

        # edge_index_dict 检查
        if edge_index_dict is None:
            raise RuntimeError("[DynamicHeteroGNN] edge_index_dict cannot be None")

        # 1) 提取 x_dict
        x_dict = {}
        for nt in self.node_types:
            x_seq = getattr(data[nt], 'x_seq', None)
            x_dict[nt] = x_seq.to(device)

        # 2) 编码器 (NodeEncoder 自行处理 resample/normalize)
        encoded = {}
        stats_map = {}
        num_nodes = {}
        for nt in self.node_types:
            h3d, N_nt, stats = self.encoders[nt](x_dict[nt])
            encoded[nt] = h3d
            stats_map[nt] = stats
            num_nodes[nt] = N_nt

        # 3) flatten 时间维度
        x_flat = {nt: encoded[nt].permute(1, 0, 2).reshape(-1, self.hidden_dim) for nt in self.node_types}

        # 4) 构建 temporal edge_index
        T_pool = self.spatial_T
        temporal_edge_index_dict = {}
        for key, edge_index in edge_index_dict.items():
            if edge_index is None or edge_index.numel() == 0:
                raise RuntimeError(f"[DynamicHeteroGNN] edge_index for {key} is None or empty")
            src, _, dst = key
            n_src, n_dst = num_nodes[src], num_nodes[dst]
            e_base = edge_index.clone().long()
            e_base[0] = e_base[0] % n_src
            e_base[1] = e_base[1] % n_dst
            lifted_list = [e_base + torch.tensor([[t * n_src], [t * n_dst]], device=e_base.device) for t in range(T_pool)]
            temporal_edge_index_dict[key] = torch.cat(lifted_list, dim=1)

        # 5) GNN 层
        h = x_flat
        for layer_idx in range(self.num_layers):
            in_dims = {nt: (h[nt].shape[1] if layer_idx == 0 else self.hidden_dim) for nt in self.node_types}
            if layer_idx >= len(self.convs):
                conv = self._build_conv_for_layer(layer_idx, in_dims).to(device)
                self.convs.append(conv)
            h_new = self.convs[layer_idx](h, temporal_edge_index_dict)
            h = {nt: F_nn.dropout(F_nn.relu(h_new[nt]), p=self.dropout.p, training=self.training) for nt in self.node_types}

        # 6) reshape 回 (N, T, H)
        h_reshaped = {nt: h[nt].view(num_nodes[nt], T_pool, self.hidden_dim) for nt in self.node_types}

        # 7) GRU
        gru_out = {}
        for nt, seq in h_reshaped.items():
            out_seq, _ = self.grus[nt](seq)
            gru_out[nt] = out_seq

        # 8) temporal align & fuse (使用 ProjectionHead time-mean pool)
        proj_pool: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            proj_pool[nt] = self.proj_head(gru_out[nt], nt)  # (N, latent_dim)

        aligned_pools = {}
        if "fmri" in proj_pool and "eeg" in proj_pool:
            a_f, a_e, align_stats = self.temporal_align(proj_pool["fmri"], proj_pool["eeg"])
            aligned_pools["fmri"] = a_f
            aligned_pools["eeg"] = a_e
        else:
            # fallback: 使用 proj_head 的 mean-pool
            for nt in self.node_types:
                aligned_pools[nt] = self.proj_head(gru_out[nt], nt) if nt in self.proj_head.heads else gru_out[nt].mean(dim=0)

        # fuse: expand aligned 到 (N, T, H)
        fused_seq = {}
        for nt in self.node_types:
            aligned = aligned_pools.get(nt)
            if aligned is not None:
                if aligned.shape[-1] == self.hidden_dim:
                    aligned_exp = aligned.unsqueeze(1).expand(-1, gru_out[nt].shape[1], -1)
                else:
                    lin = getattr(self, f"_align_to_hidden_{nt}", None)
                    if lin is None:
                        lin = nn.Linear(aligned.shape[-1], self.hidden_dim).to(device)
                        setattr(self, f"_align_to_hidden_{nt}", lin)
                    aligned_exp = lin(aligned).unsqueeze(1).expand(-1, gru_out[nt].shape[1], -1)
                fused_seq[nt] = gru_out[nt] + aligned_exp
            else:
                fused_seq[nt] = gru_out[nt]

        # 9) temporal projection
        proj_seq_dict = {}
        for nt, seq in fused_seq.items():
            if self.temporal_T != T_pool:
                flat = seq.permute(0, 2, 1).reshape(-1, T_pool)
                proj = self._get_or_create_temporal_proj(nt, T_pool)
                seq_proj = proj(flat).view(num_nodes[nt], self.hidden_dim, self.temporal_T).permute(0, 2, 1)
                proj_seq_dict[nt] = seq_proj
            else:
                proj_seq_dict[nt] = seq

        # ---- decode 前/后 对比 ----
        recon_seq_denorm = {}
        recon_seq_scaled = {}   # 用于训练的 scaled 输出（此处设为 denorm 避免混淆）
        recon_feature_dict = {}  # 新增：保留 decoder 前的“归一化预测” (N,T,F)
        for nt in self.node_types:
            recon_hidden = proj_seq_dict[nt]
            recon_feature = self.feature_decoders[nt](recon_hidden)  # (N,T,F), 期望为归一化空间的预测
            recon_feature_dict[nt] = recon_feature  # 返回给 trainer 用于归一化损失

            # NodeDecoder 的 denorm 输出（包括 learnable scale + bias + 使用原始 mean/std）
            recon_denorm = self.denorm_decoders[nt](recon_feature, stats_map[nt], node_type=nt)
            recon_seq_denorm[nt] = recon_denorm

            # 避免训练中使用不一致的量：把 recon_seq_scaled 设置为 denorm（之前的实现存在混淆）
            recon_seq_scaled[nt] = recon_denorm

            # if self.debug:
            #     logger.info(f"[forward][decode_debug] {nt} feature_decoder output mean={recon_feature.mean():.6f}, std={recon_feature.std():.6f}")
            #     logger.info(f"[forward][decode_debug] {nt} denorm output mean={recon_denorm.mean():.6f}, std={recon_denorm.std():.6f}")

        # 返回
        z_dict = {nt: seq.mean(dim=1) for nt, seq in proj_seq_dict.items()}
        valid_means = [seq.mean(dim=0) for seq in proj_seq_dict.values()]
        global_seq = self.global_proj(torch.stack(valid_means).mean(dim=0))

        # 注意：增加 recon_feature_dict 作为额外返回值（兼容旧 trainer 只读取前 6 个输出）
        return z_dict, gru_out, proj_seq_dict, recon_seq_denorm, recon_seq_scaled, global_seq, recon_feature_dict
    # ---------- helpers ----------
    def _build_conv_for_layer(self, layer_idx: int, in_dims: Dict[str, int]) -> HeteroConv:
        conv_dict = {}
        for src, rel, dst in self.edge_types:
            conv_dict[(src, rel, dst)] = SAGEConv((in_dims.get(src, 1), in_dims.get(dst, 1)), self.hidden_dim)
        return HeteroConv(conv_dict, aggr="mean")

    def _get_or_create_temporal_proj(self, nt: str, T_src: int) -> nn.Linear:
        key = str(T_src)
        if key not in self.temporal_projs[nt]:
            proj = nn.Linear(T_src, self.temporal_T, bias=False)
            nn.init.xavier_uniform_(proj.weight)
            proj = proj.to(next(self.parameters()).device)
            self.temporal_projs[nt][key] = proj
        return self.temporal_projs[nt][key]