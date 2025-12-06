import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Any
import logging

# local imports - these must exist in your repo
try:
    from train.aligner import TemporalCrossAligner
except Exception:
    TemporalCrossAligner = None

try:
    from train.coder import NodeEncoder, NodeDecoder, ProjectionHead, TemporalDecoder
except Exception:
    # graceful fallbacks if imports missing; user should provide real implementations
    TemporalDecoder = None

logger = logging.getLogger(__name__)


class DynamicHeteroGNN(nn.Module):
    """
    Heterogeneous dynamic GNN with temporal decoding.

    Key behaviors / design notes:
    - Encodes per-node sequences with NodeEncoder -> (N, T_spatial, H)
    - Lifts to temporal graph convs across time and nodes
    - Decodes per-node sequences via feature_decoders (TemporalDecoder by default)
    - New: merges proj_seq (low-dim projection) with gru_out (raw sequence) before decoding
      by concatenation and a small linear projection back to hidden_dim. This helps restore
      time-domain details lost by projection.
    """
    def __init__(
        self,
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

        # Encoders: NodeEncoder must return (encoded_seq (N,T,H), num_nodes, stats_dict)
        # If NodeEncoder not provided by train.coder, user must supply.
        try:
            self.encoders = nn.ModuleDict({nt: NodeEncoder(self.spatial_T, self.hidden_dim) for nt in self.node_types})
        except Exception:
            # Fallback: identity-like encoder (expects x_seq shape (N,T,F) -> project to H)
            class _SimpleEnc(nn.Module):
                def __init__(self, spatial_T, hidden_dim, in_dim=1):
                    super().__init__()
                    self.spatial_T = spatial_T
                    self.hidden_dim = hidden_dim
                    self.proj = nn.Linear(in_dim, hidden_dim)

                def forward(self, x_seq):
                    if x_seq is None:
                        return torch.zeros((0, self.spatial_T, self.hidden_dim)), 0, {"mean": None, "std": None}
                    # x_seq (N, T, F) -> (N, T, hidden_dim)
                    N, T, F = x_seq.shape
                    out = self.proj(x_seq.view(-1, F)).view(N, T, self.hidden_dim)
                    stats = {"mean": x_seq.mean(dim=(0,1), keepdim=True), "std": x_seq.std(dim=(0,1), keepdim=True)}
                    return out, N, stats
            self.encoders = nn.ModuleDict({nt: _SimpleEnc(self.spatial_T, self.hidden_dim, in_dim=self.node_feature_dims.get(nt, 1)) for nt in self.node_types})

        # Feature decoders: use TemporalDecoder when available, else fallback to simple linear per-time MLP
        self.feature_decoders = nn.ModuleDict()
        for nt in self.node_types:
            out_dim = int(self.node_feature_dims.get(nt, 1))
            if TemporalDecoder is not None:
                # decoder input is expected to be hidden_dim (we will project concat->hidden_dim before feeding)
                self.feature_decoders[nt] = TemporalDecoder(in_dim=self.hidden_dim, out_dim=out_dim, channels=min(256, self.hidden_dim), kernel_size=5, num_layers=3, dropout=0.1)
            else:
                # fallback: time-distributed linear
                self.feature_decoders[nt] = nn.Sequential(
                    nn.Linear(self.hidden_dim, max(self.hidden_dim // 2, out_dim)),
                    nn.ReLU(),
                    nn.Linear(max(self.hidden_dim // 2, out_dim), out_dim)
                )

        # Per-node projection to map concat( proj + gru ) (2*H) back to H for decoder input
        self.decoder_input_proj = nn.ModuleDict()
        for nt in self.node_types:
            # linear 2H -> H (even if not used always)
            self.decoder_input_proj[nt] = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Denorm decoders: NodeDecoder expected to map normalized recon_feature -> denormed recon
        try:
            self.denorm_decoders = NodeDecoder(node_types=self.node_types, out_dims=self.node_feature_dims, hidden_dim=self.hidden_dim, extra_hidden=self.hidden_dim, num_layers=3)
        except Exception:
            # Simple denorm: recon_feature * std + mean
            class _SimpleDenorm(nn.Module):
                def __init__(self, node_types, out_dims):
                    super().__init__()
                    self.node_types = node_types
                    self.out_dims = out_dims

                def get_scale(self, nt):
                    return torch.ones(self.out_dims.get(nt, 1))

                def forward(self, recon_feature, stats, node_type=None):
                    mean = stats.get("mean", None)
                    std = stats.get("std", None)
                    if mean is not None and std is not None:
                        return recon_feature * std.expand(-1, recon_feature.shape[1], -1) + mean.expand(-1, recon_feature.shape[1], -1)
                    return recon_feature
            self.denorm_decoders = _SimpleDenorm(self.node_types, self.node_feature_dims)

        # GNN backbone components
        self.convs = nn.ModuleList()
        self.grus = nn.ModuleDict({nt: nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True) for nt in self.node_types})
        # temporal projection maps per-modality time sizes if needed
        self.temporal_projs = nn.ModuleDict({nt: nn.ModuleDict() for nt in self.node_types})

        # aligner (temporal cross align)
        if TemporalCrossAligner is not None:
            self.temporal_align = TemporalCrossAligner(hidden_dim=self.hidden_dim, dropout=0.0)
        else:
            # fallback dummy aligner that returns inputs unchanged and zero loss
            class _DummyAligner(nn.Module):
                def forward(self, a, b):
                    return a, b, {"loss": torch.tensor(0.0)}
            self.temporal_align = _DummyAligner()

        # global projection and projection head
        try:
            self.global_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.proj_head = ProjectionHead(hidden_dim=self.hidden_dim, latent_dim=self.hidden_dim)
        except Exception:
            # fallback minimal implementations
            self.global_proj = nn.Identity()
            class _PH(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                def forward(self, seq, nt=None):
                    # seq (N,T,H) -> return seq.mean(dim=1)
                    return seq.mean(dim=1)
            self.proj_head = _PH(self.hidden_dim)

    def forward(self, data: HeteroData, edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]):
        device = next(self.parameters()).device

        if edge_index_dict is None:
            raise RuntimeError("edge_index_dict cannot be None")

        # 1) extract sequences
        x_dict = {}
        for nt in self.node_types:
            x_seq = getattr(data[nt], 'x_seq', None)
            x_dict[nt] = x_seq.to(device) if x_seq is not None else None

        # 2) per-node encoding
        encoded = {}
        stats_map = {}
        num_nodes = {}
        for nt in self.node_types:
            enc_out = self.encoders[nt](x_dict[nt])
            # flexible return: either (h3d, N, stats) or (h3d, stats)
            if isinstance(enc_out, tuple) and len(enc_out) == 3:
                h3d, N_nt, stats = enc_out
            else:
                # fallback assume (h3d, N, stats)
                h3d, N_nt, stats = enc_out
            encoded[nt] = h3d
            stats_map[nt] = stats
            num_nodes[nt] = N_nt

        # 3) flatten time for spatial graph convs
        x_flat = {}
        for nt in self.node_types:
            # encoded[nt]: (N, T_spatial, H)
            seq = encoded[nt]
            if seq is None or seq.numel() == 0:
                x_flat[nt] = torch.zeros((0, self.hidden_dim), device=device)
                continue
            # permute to (T, N, H) -> flatten nodes across time: (N*T, H) but we prefer (num_nodes*N, H)
            # We'll reshape as (N*T, H) trading time for batch for graph conv
            N_nt, T_sp, H = seq.shape
            x_flat[nt] = seq.permute(1, 0, 2).reshape(T_sp * N_nt, H)

        # 4) build temporal lifted edge indices (repeat edges across time blocks)
        T_pool = self.spatial_T
        temporal_edge_index_dict = {}
        for key, edge_index in edge_index_dict.items():
            if edge_index is None or edge_index.numel() == 0:
                raise RuntimeError(f"edge_index for {key} is None or empty")
            src, _, dst = key
            n_src, n_dst = num_nodes.get(src, 0), num_nodes.get(dst, 0)
            e_base = edge_index.clone().long()
            e_base[0] = e_base[0] % max(1, n_src)
            e_base[1] = e_base[1] % max(1, n_dst)
            lifted_list = []
            for t in range(T_pool):
                lifted = e_base + torch.tensor([[t * n_src], [t * n_dst]], device=e_base.device)
                lifted_list.append(lifted)
            temporal_edge_index_dict[key] = torch.cat(lifted_list, dim=1)

        # 5) stacked heterogeneous GNN layers
        h = x_flat
        for layer_idx in range(self.num_layers):
            in_dims = {nt: (h[nt].shape[1] if layer_idx == 0 and h[nt].nelement() > 0 else self.hidden_dim) for nt in self.node_types}
            if layer_idx >= len(self.convs):
                conv = self._build_conv_for_layer(layer_idx, in_dims).to(device)
                self.convs.append(conv)
            h_new = self.convs[layer_idx](h, temporal_edge_index_dict)
            h = {nt: F_nn.dropout(F_nn.relu(h_new[nt]), p=self.dropout.p, training=self.training) for nt in self.node_types}

        # 6) reshape back to (N, T_pool, H)
        h_reshaped = {}
        for nt in self.node_types:
            # h[nt] shape: (N*T_pool, H) -> reshape (N, T_pool, H)
            if h.get(nt) is None or h[nt].numel() == 0:
                h_reshaped[nt] = torch.zeros((0, T_pool, self.hidden_dim), device=device)
                continue
            # we need num_nodes[nt]
            n_nodes = max(1, num_nodes.get(nt, 1))
            h_reshaped[nt] = h[nt].view(T_pool, n_nodes, self.hidden_dim).permute(1, 0, 2).contiguous()

        # 7) GRU temporal layer per node type
        gru_out = {}
        for nt, seq in h_reshaped.items():
            if seq is None or seq.numel() == 0:
                gru_out[nt] = torch.zeros((0, 0, 0), device=device)
                continue
            out_seq, _ = self.grus[nt](seq)
            gru_out[nt] = out_seq  # (N, T_pool, H)

        # 8) projection / alignment / fusion
        proj_pool = {}
        for nt in self.node_types:
            proj_pool[nt] = self.proj_head(gru_out[nt], nt) if hasattr(self.proj_head, '__call__') else gru_out[nt].mean(dim=1)

        # temporal align (if both fmri/eeg present) - aligner returns aligned representations
        aligned_pools = {}
        if "fmri" in proj_pool and "eeg" in proj_pool and hasattr(self.temporal_align, '__call__'):
            try:
                a_f, a_e, align_stats = self.temporal_align(proj_pool["fmri"], proj_pool["eeg"])
                aligned_pools["fmri"] = a_f
                aligned_pools["eeg"] = a_e
            except Exception:
                for nt in self.node_types:
                    aligned_pools[nt] = proj_pool.get(nt)
        else:
            for nt in self.node_types:
                aligned_pools[nt] = proj_pool.get(nt, None)

        # fuse aligned into sequence (add aligned vector to gru_out at every timestep)
        fused_seq = {}
        for nt in self.node_types:
            seq = gru_out.get(nt, None)
            aligned = aligned_pools.get(nt, None)
            if seq is None or seq.numel() == 0:
                fused_seq[nt] = seq
                continue
            if aligned is not None:
                if aligned.shape[-1] == self.hidden_dim:
                    aligned_exp = aligned.unsqueeze(1).expand(-1, seq.shape[1], -1)
                else:
                    lin_name = f"_align_to_hidden_{nt}"
                    lin = getattr(self, lin_name, None)
                    if lin is None:
                        lin = nn.Linear(aligned.shape[-1], self.hidden_dim).to(device)
                        setattr(self, lin_name, lin)
                    aligned_exp = lin(aligned).unsqueeze(1).expand(-1, seq.shape[1], -1)
                fused_seq[nt] = seq + aligned_exp
            else:
                fused_seq[nt] = seq

        # 9) temporal projection to desired temporal_T if needed
        proj_seq_dict = {}
        for nt, seq in fused_seq.items():
            if seq is None or seq.numel() == 0:
                proj_seq_dict[nt] = seq
                continue
            if self.temporal_T != seq.shape[1]:
                key = str(seq.shape[1])
                proj = self._get_or_create_temporal_proj(nt, seq.shape[1])
                # flatten per-channel: seq (N, T_src, H) -> (N, H, T_src) -> permute flatten approach
                flat = seq.permute(0, 2, 1).reshape(-1, seq.shape[1])
                seq_proj = proj(flat).view(seq.shape[0], self.hidden_dim, self.temporal_T).permute(0, 2, 1)
                proj_seq_dict[nt] = seq_proj
            else:
                proj_seq_dict[nt] = seq

        # ---- decode: produce recon_feature (normalized-space) and recon_denorm ----
        recon_seq_denorm = {}
        recon_seq_scaled = {}
        recon_feature_dict = {}

        for nt in self.node_types:
            # obtain proj_hidden and gru_hidden
            proj_hidden = proj_seq_dict.get(nt, None)  # (N, T_proj, H)
            gru_hidden = gru_out.get(nt, None)         # (N, T_gru, H)

            recon_hidden = None

            # if both present, resample GRU to proj time and concat, then map back to hidden_dim
            if (proj_hidden is not None and proj_hidden.numel() != 0) and (gru_hidden is not None and gru_hidden.numel() != 0):
                if proj_hidden.shape[1] != gru_hidden.shape[1]:
                    # interpolate gru_hidden (N,H,T) conv style via F.interpolate after permute
                    gru_rs = F_nn.interpolate(gru_hidden.permute(0, 2, 1), size=proj_hidden.shape[1], mode='linear', align_corners=False).permute(0, 2, 1)
                else:
                    gru_rs = gru_hidden
                # concat along feature dim
                recon_hidden_cat = torch.cat([proj_hidden, gru_rs], dim=-1)  # (N, T, 2H)
                B, Ttmp, Dtmp = recon_hidden_cat.shape
                # project 2H -> H
                proj_fn = self.decoder_input_proj[nt]
                recon_hidden = proj_fn(recon_hidden_cat.reshape(-1, Dtmp)).view(B, Ttmp, self.hidden_dim)
            elif proj_hidden is not None and proj_hidden.numel() != 0:
                recon_hidden = proj_hidden
            elif gru_hidden is not None and gru_hidden.numel() != 0:
                # resample gru to target temporal length (use temporal_T)
                target_T = self.temporal_T if hasattr(self, "temporal_T") else gru_hidden.shape[1]
                if gru_hidden.shape[1] != target_T:
                    recon_hidden = F_nn.interpolate(gru_hidden.permute(0, 2, 1), size=target_T, mode='linear', align_corners=False).permute(0, 2, 1)
                else:
                    recon_hidden = gru_hidden
            else:
                recon_hidden = None

            # apply decoder if available
            if recon_hidden is None or recon_hidden.numel() == 0:
                # fallback zero-tensor shaped (N, temporal_T, out_dim)
                out_dim = int(self.node_feature_dims.get(nt, 1))
                recon_feature = torch.zeros((0,), device=device)
            else:
                # safe access to ModuleDict (ModuleDict has __getitem__ but no get)
                try:
                    dec = self.feature_decoders[nt]
                except Exception:
                    dec = None

                if dec is None:
                    # fallback zero
                    recon_feature = torch.zeros((recon_hidden.shape[0], recon_hidden.shape[1], int(self.node_feature_dims.get(nt,1))), device=device)
                else:
                    recon_feature = dec(recon_hidden)  # (N, T, F)
                    # store clone to avoid in-place issues
                    try:
                        recon_feature_dict[nt] = recon_feature.clone()
                    except Exception:
                        recon_feature_dict[nt] = recon_feature

            # Denormalize via denorm_decoders
            stats = stats_map.get(nt, {"mean": None, "std": None})
            try:
                recon_denorm = self.denorm_decoders(recon_feature_dict[nt], stats, node_type=nt)
            except Exception:
                # fallback identity
                recon_denorm = recon_feature_dict.get(nt, recon_feature)

            # IMPORTANT: expose both normalized and denormalized outputs separately:
            # - recon_seq_denorm (outputs slot 3) will contain the normalized-space recon (recon_feature)
            #   so diagnostics can compare normalized vs denorm easily.
            # - recon_seq_scaled (outputs slot 4) will contain the denormalized recon (recon_denorm)
            recon_seq_denorm[nt] = recon_feature_dict.get(nt, recon_denorm)
            recon_seq_scaled[nt] = recon_denorm


        # prepare outputs for trainer
        z_dict = {nt: (proj_seq_dict[nt].mean(dim=1) if proj_seq_dict.get(nt) is not None and proj_seq_dict[nt].numel() != 0 else torch.zeros((self.hidden_dim,), device=device)) for nt in self.node_types}
        valid_means = [proj_seq_dict[nt].mean(dim=0) for nt in proj_seq_dict.keys() if proj_seq_dict.get(nt) is not None and proj_seq_dict[nt].numel() != 0]
        if len(valid_means) > 0:
            global_seq = self.global_proj(torch.stack(valid_means).mean(dim=0))
        else:
            global_seq = torch.zeros((self.hidden_dim,), device=device)

        return z_dict, gru_out, proj_seq_dict, recon_seq_denorm, recon_seq_scaled, global_seq, recon_feature_dict

    # helpers
    def _build_conv_for_layer(self, layer_idx: int, in_dims: Dict[str, int]) -> HeteroConv:
        conv_dict = {}
        for src, rel, dst in self.edge_types:
            in_src = in_dims.get(src, 1)
            in_dst = in_dims.get(dst, 1)
            conv_dict[(src, rel, dst)] = SAGEConv((in_src, in_dst), self.hidden_dim)
        return HeteroConv(conv_dict, aggr="mean")

    def _get_or_create_temporal_proj(self, nt: str, T_src: int) -> nn.Linear:
        key = str(T_src)
        if key not in self.temporal_projs[nt]:
            proj = nn.Linear(T_src, self.temporal_T, bias=False)
            nn.init.xavier_uniform_(proj.weight)
            proj = proj.to(next(self.parameters()).device)
            self.temporal_projs[nt][key] = proj
        return self.temporal_projs[nt][key]