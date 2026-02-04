import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

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
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

# -------------------------
# TemporalCrossAligner
# -------------------------
class TemporalCrossAligner(nn.Module):
    """
    Cross-attention aligner between two temporal sequences.
    Inputs:
        seq1, seq2: (T, H) tensors
    Returns:
        aligned_seq1, aligned_seq2: (T, H)
        stats: dict with attn entropy info
    """
    def __init__(self, hidden_dim: int = 64, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, seq1: torch.Tensor, seq2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        seq1_b = seq1.unsqueeze(0)  # (1, T, H)
        seq2_b = seq2.unsqueeze(0)
        aligned1, w1 = self.attn(seq1_b, seq2_b, seq2_b, need_weights=True)
        aligned2, w2 = self.attn(seq2_b, seq1_b, seq1_b, need_weights=True)
        eps = 1e-12
        entropy1 = -(w1 * (w1 + eps).log()).sum(dim=-1).mean().item()
        entropy2 = -(w2 * (w2 + eps).log()).sum(dim=-1).mean().item()
        return aligned1.squeeze(0), aligned2.squeeze(0), {"attn_f2e_entropy": entropy1, "attn_e2f_entropy": entropy2}


# -------------------------
# LatentAligner (multi-mode)
# -------------------------
class LatentAligner(nn.Module):
    """
    Flexible aligner supporting:
      - nodewise MSE (for datasets where nodes are already aligned/index-matched)
      - contrastive symmetric cross-entropy (robust when nodes are not in 1:1 index mapping)
      - auto mode (prefer nodewise when shapes match)

    Usage notes:
      - If z inputs are [N, T, D] they are collapsed by mean over time before nodewise comparison.
      - Returns a scalar Tensor (with grad when appropriate).
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        mode: str = "auto",            # 'nodewise', 'contrastive' or 'auto'
        lambda_align: float = 1.0,     # weighting for nodewise MSE
        temperature: float = 0.3,      # for contrastive
        safe_clip: float = 1e9,
    ):
        super().__init__()
        self.mode = mode
        self.lambda_align = lambda_align
        self.temperature = temperature
        self.safe_clip = safe_clip

        # projection layers only needed for contrastive mode (kept even if nodewise for flexibility)
        self.proj_fmri = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_eeg = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, z_fmri: torch.Tensor, z_eeg: torch.Tensor) -> torch.Tensor:
        # Basic guards
        if z_fmri is None or z_eeg is None:
            return torch.tensor(0.0, device=(z_fmri.device if z_fmri is not None else z_eeg.device))
        if z_fmri.numel() == 0 or z_eeg.numel() == 0:
            return torch.tensor(0.0, device=(z_fmri.device if z_fmri.numel() else z_eeg.device))

        # collapse time dim if present
        if z_fmri.dim() == 3:
            zf = z_fmri.mean(dim=1)
        else:
            zf = z_fmri
        if z_eeg.dim() == 3:
            ze = z_eeg.mean(dim=1)
        else:
            ze = z_eeg

        # choose mode
        mode = self.mode
        if mode == "auto":
            # if node counts match, prefer nodewise (user said nodes/time are aligned)
            if zf.shape[0] == ze.shape[0]:
                mode = "nodewise"
            else:
                mode = "contrastive"

        if mode == "nodewise":
            # nodewise MSE: assume nodes correspond by index
            try:
                # if node counts differ but one side has N==1, broadcast; else truncate to min(N)
                if zf.shape[0] != ze.shape[0]:
                    if zf.shape[0] == 1:
                        zf = zf.expand(ze.shape[0], -1)
                    elif ze.shape[0] == 1:
                        ze = ze.expand(zf.shape[0], -1)
                    else:
                        nmin = min(zf.shape[0], ze.shape[0])
                        zf = zf[:nmin]
                        ze = ze[:nmin]
                loss = ((zf - ze) ** 2).mean() * self.lambda_align
                return loss
            except Exception as e:
                logger.error(f"[LatentAligner.nodewise ERROR] shapes zf={tuple(zf.shape)} ze={tuple(ze.shape)} err={e}")
                return torch.tensor(0.0, device=zf.device if zf.numel() else ze.device)

        # contrastive branch (symmetric CE on similarity logits)
        if mode == "contrastive":
            try:
                z1 = F.normalize(self.proj_fmri(zf), dim=-1)  # (Nf, D)
                z2 = F.normalize(self.proj_eeg(ze), dim=-1)   # (Ne, D)
                sim = torch.matmul(z1, z2.t()) / self.temperature  # (Nf, Ne)

                # numerical safety
                if not torch.isfinite(sim).all():
                    sim = torch.where(torch.isfinite(sim), sim, torch.full_like(sim, -self.safe_clip))

                device = sim.device
                labels_i2e = torch.argmax(sim, dim=1).to(device=device, dtype=torch.long)
                sim_e2i = sim.t()
                labels_e2i = torch.argmax(sim_e2i, dim=1).to(device=device, dtype=torch.long)

                loss_i2e = F.cross_entropy(sim, labels_i2e)
                loss_e2i = F.cross_entropy(sim_e2i, labels_e2i)
                return 0.5 * (loss_i2e + loss_e2i)
            except Exception as e:
                logger.error(f"[LatentAligner.contrastive ERROR] shapes zf={tuple(zf.shape)} ze={tuple(ze.shape)} err={e}")
                return torch.tensor(0.0, device=zf.device if zf.numel() else ze.device)

        # unknown mode
        logger.error(f"[LatentAligner] Unknown mode={self.mode}. Returning 0.")
        return torch.tensor(0.0, device=zf.device)