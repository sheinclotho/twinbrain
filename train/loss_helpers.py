# 放到 train/loss_helpers.py（新文件或并入现有 trainer 模块）
import random
import numpy as np

# Initialize random seeds before torch import to prevent THPGenerator errors
_INIT_SEED = 42
random.seed(_INIT_SEED)
np.random.seed(_INIT_SEED)

import torch
# MUST call manual_seed immediately after torch import
torch.manual_seed(_INIT_SEED)

import torch.nn.functional as F

def lowpass_mse_loss(recon_norm: torch.Tensor, target_norm: torch.Tensor, kernel_size: int = 11) -> torch.Tensor:
    """
    Compute lowpass (moving-average) MSE between recon_norm and target_norm.
    recon_norm/target_norm: (B, T, F)
    kernel_size: odd integer smoothing window length (in frames)
    Returns scalar loss.
    """
    if recon_norm is None or target_norm is None:
        return torch.tensor(0.0, device=recon_norm.device if recon_norm is not None else "cpu")

    B, T, Ff = recon_norm.shape
    pad = kernel_size // 2
    # kernel shape (out_channels=1, in_channels=1, kernel_size)
    w = torch.ones(1, 1, kernel_size, device=recon_norm.device, dtype=recon_norm.dtype) / float(kernel_size)

    # reshape to (B*F, 1, T)
    r = recon_norm.permute(0, 2, 1).reshape(B * Ff, 1, T)
    t = target_norm.permute(0, 2, 1).reshape(B * Ff, 1, T)

    # reflect-pad to reduce edge artifacts
    r_pad = F.pad(r, (pad, pad), mode="reflect")
    t_pad = F.pad(t, (pad, pad), mode="reflect")

    r_lp = F.conv1d(r_pad, w, groups=1)
    t_lp = F.conv1d(t_pad, w, groups=1)

    # MSE over (B*F, T)
    loss = F.mse_loss(r_lp, t_lp)
    return loss