import numpy as np
import torch
from typing import Optional
from scipy.signal import correlate

def _estimate_best_lag_cpu(ref: np.ndarray, query: np.ndarray, max_lag: int) -> int:
    T = len(ref)
    if T <= 0:
        return 0
    corr_full = correlate(ref, query, mode="full")
    lags = np.arange(-T + 1, T)
    mask = (lags >= -max_lag) & (lags <= max_lag)
    if not mask.any():
        return 0
    corr_full[~mask] = -1e18
    best_idx = int(np.argmax(corr_full))
    best_lag = int(lags[best_idx])
    return best_lag


def compute_and_set_auto_align_lags(trainer,
                                    nt: str = "fmri",
                                    max_lag: int = 150,
                                    agg_method: str = "median",
                                    per_sample_limit: Optional[int] = None) -> dict:
    """
    For trainer.data_list, run model forward on each sample once (no grads),
    compute integer best_lag between recon_feature and target (mean over batch/features),
    aggregate per-sample lags (median/mode/mean) and write a fixed lag into
    trainer._auto_align_cache['fixed_{nt}'].

    Returns info dict with fixed lag and the per-sample lags list.
    """
    device = getattr(trainer, "device", torch.device("cpu"))
    lags = []

    trainer.model.eval()
    with torch.no_grad():
        for i, data in enumerate(trainer.data_list):
            if per_sample_limit is not None and i >= per_sample_limit:
                break
            d = data.to(device)
            # try to get edge_index_dict if encoder exists
            edge_index_dict = None
            try:
                enc_out = trainer.graph_encoder(d)
                # enc_out may be (x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict)
                if isinstance(enc_out, (list, tuple)) and len(enc_out) >= 5:
                    edge_index_dict = enc_out[-1]
            except Exception:
                edge_index_dict = None
            try:
                outputs = trainer.model(d, edge_index_dict=edge_index_dict)
            except Exception:
                continue
            # unpack recon_feature robustly
            recon_feature_dict = None
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 7:
                recon_feature_dict = outputs[6]
            elif isinstance(outputs, dict) and "recon_feature_dict" in outputs:
                recon_feature_dict = outputs["recon_feature_dict"]
            else:
                # cannot find recon_feature, skip
                continue

            if recon_feature_dict is None or nt not in recon_feature_dict:
                continue
            rf = recon_feature_dict[nt]
            if rf is None or rf.numel() == 0:
                continue
            target_raw = getattr(d[nt], "x_seq", None)
            if target_raw is None:
                continue
            try:
                # resample target to recon length using trainer helper (keeps consistency)
                t_res = trainer._resample_time(target_raw.to(device), rf.shape[1])
            except Exception:
                continue

            # compute 1D mean signals on CPU
            qry = rf.detach().cpu().numpy().mean(axis=(0, 2))
            ref = t_res.detach().cpu().numpy().mean(axis=(0, 2))
            # ensure finite
            if not (np.isfinite(ref).all() and np.isfinite(qry).all()):
                continue
            best_lag = _estimate_best_lag_cpu(ref, qry, max_lag=int(max_lag))
            lags.append(int(best_lag))

    if len(lags) == 0:
        fixed = 0
    else:
        arr = np.array(lags)
        if agg_method == "median":
            fixed = int(np.median(arr))
        elif agg_method == "mode":
            vals, counts = np.unique(arr, return_counts=True)
            fixed = int(vals[np.argmax(counts)])
        else:
            fixed = int(np.round(np.mean(arr)))

    if not hasattr(trainer, "_auto_align_cache"):
        trainer._auto_align_cache = {}
    trainer._auto_align_cache[f"fixed_{nt}"] = int(fixed)

    # convenience: enable apply in trainer (so trainer will use cache)
    trainer.auto_align = True
    trainer.auto_align_scope = "always"
    trainer.auto_align_max_lag = int(max_lag)

    return {"fixed_lag": int(fixed), "lags": lags, "agg_method": agg_method}