# -------------------------
# LatentAligner (robust)
# -------------------------
class LatentAligner(nn.Module):
    """
    Computes a robust alignment loss between z_fmri and z_eeg.
    Accepts either [N, D] or [N, T, D]. Converts 2D -> 3D (N,1,D) when needed.
    Loss: mean squared error across nodes, time and dims (weighted by lambda_align).
    """
    def __init__(self, hidden_dim: int, lambda_align: float = 1.0):
        super().__init__()
        self.lambda_align = lambda_align

    def forward(self, z_fmri: torch.Tensor, z_eeg: torch.Tensor) -> torch.Tensor:
        if z_fmri.dim() == 2:
            z_fmri = z_fmri.unsqueeze(1)
        if z_eeg.dim() == 2:
            z_eeg = z_eeg.unsqueeze(1)

        if z_fmri.dim() != 3 or z_eeg.dim() != 3:
            raise ValueError(f"[Aligner] Expect shape [N, T, D], got {z_fmri.shape} and {z_eeg.shape}")

        Nf, Tf, Df = z_fmri.shape
        Ne, Te, De = z_eeg.shape
        if Nf != Ne or Df != De:
            raise ValueError(f"[Aligner] Node or feature dim mismatch: {z_fmri.shape} vs {z_eeg.shape}")

        loss = ((z_fmri - z_eeg) ** 2).mean() * self.lambda_align
        return loss