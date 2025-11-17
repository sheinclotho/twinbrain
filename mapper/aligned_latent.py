import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LatentAligner(nn.Module):
    """
    动态孪生脑潜在对齐模块

    功能:
    1. 支持 EEG/fMRI 时间长度不同，通过线性插值对齐
    2. 动态线性对齐 (nn.Linear)，可选旋转矩阵对齐
    3. 延迟初始化，适应不同 batch 或不同特征维度
    4. 自动标准化输入，保证训练稳定
    """

    def __init__(self, max_ratio_tolerance: float = 0.25, use_rotation: bool = False):
        super().__init__()
        self.max_ratio_tolerance = float(max_ratio_tolerance)
        self.use_rotation = use_rotation

        self.align_linear: nn.Linear | None = None
        self.align_R: nn.Parameter | None = None

    @staticmethod
    def _temporal_align_interpolate(fmri_x: torch.Tensor, eeg_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 EEG 时间序列插值到 fMRI 时间长度
        输入: fmri_x: (Nf, Tf, Ff), eeg_x: (Ne, Te, Fe)
        输出: 对齐后的 fmri_x, eeg_x
        """
        Nf, Tf, Ff = fmri_x.shape
        Ne, Te, Fe = eeg_x.shape
        if Te == Tf:
            return fmri_x, eeg_x.clone()
        # 线性插值到 fMRI 时间长度
        eeg_al = F.interpolate(eeg_x.permute(0, 2, 1), size=Tf, mode='linear', align_corners=False).permute(0, 2, 1)
        return fmri_x, eeg_al

    def forward(self, fmri_x: torch.Tensor, eeg_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fmri_x: (Nf, Tf, Ff) 或 (Ff,) 或 (Tf, Ff)
            eeg_x:  (Ne, Te, Fe) 或 (Fe,) 或 (Te, Fe)
        Returns:
            fmri_out, eeg_out: 对齐后的张量, 形状统一 (N, T, q)
        """
        device = fmri_x.device

        # ---------------- 1. 时间对齐 ----------------
        if fmri_x.ndim == 2:
            fmri_x = fmri_x.unsqueeze(0)  # (1, T, F)
        if eeg_x.ndim == 2:
            eeg_x = eeg_x.unsqueeze(0)
        fmri_proj, eeg_proj = self._temporal_align_interpolate(fmri_x, eeg_x)

        # ---------------- 2. 标准化 ----------------
        eps = 1e-6
        fmri_proj = (fmri_proj - fmri_proj.mean(dim=(0, 1), keepdim=True)) / (fmri_proj.std(dim=(0, 1), keepdim=True) + eps)
        eeg_proj  = (eeg_proj - eeg_proj.mean(dim=(0, 1), keepdim=True)) / (eeg_proj.std(dim=(0, 1), keepdim=True) + eps)

        # ---------------- 3. Flatten -> Linear 对齐 ----------------
        Nf, Tf, Ff = fmri_proj.shape
        Ne, Te, Fe = eeg_proj.shape
        fmri_2d = fmri_proj.reshape(-1, Ff)
        eeg_2d  = eeg_proj.reshape(-1, Fe)

        q = min(Ff, Fe)

        # ---------------- 4. 延迟初始化 align_linear ----------------
        if (self.align_linear is None) or (self.align_linear.in_features != q):
            self.align_linear = nn.Linear(q, q, bias=False).to(device)
            nn.init.eye_(self.align_linear.weight)
            # logger.debug(f"[LatentAligner] Initialized align_linear in_features={q}")

        eeg_aligned = self.align_linear(eeg_2d)

        # ---------------- 5. 可选旋转 ----------------
        if self.use_rotation:
            if (self.align_R is None) or (self.align_R.shape[0] != q):
                self.align_R = nn.Parameter(torch.eye(q, device=device))
            eeg_aligned = eeg_aligned @ self.align_R

        # ---------------- 6. Reshape 回原形 ----------------
        fmri_out = fmri_2d[:, :q].reshape(Nf, Tf, q)
        eeg_out  = eeg_aligned.reshape(Ne, Tf, q)

        return fmri_out.to(device), eeg_out.to(device)
