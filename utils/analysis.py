"""
Analysis utilities for brain imaging data.
Contains shared functions for cross-correlation analysis and other metrics.
"""

import numpy as np
import torch
from scipy.signal import correlate


def compute_xcorr_best_lag(trainer, nt="fmri", node_idx=0, feat_idx=0):
    """
    Compute cross-correlation between recon_feature and target for given node/feature,
    return best_lag (in frames) and best_corr (normalized by length).
    
    Args:
        trainer: DynamicHeteroTrainer instance
        nt: Node type ("fmri" or "eeg")
        node_idx: Index of the node to analyze
        feat_idx: Index of the feature to analyze
        
    Returns:
        dict: Contains best_lag, best_corr, corr_trace_len, or error information
    """
    device = getattr(trainer, "device", torch.device("cpu"))
    data = trainer.data_list[0].to(device)
    try:
        enc_out = trainer.graph_encoder(data)
        x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict = enc_out
    except Exception as e:
        print("[XCORR] graph_encoder failed:", e)
        return {"error": "graph_encoder_failed", "exc": str(e)}

    with torch.no_grad():
        try:
            outputs = trainer.model(
                data=data,
                edge_index_dict=edge_index_dict,
                encoded_dict=x_dict,
                num_nodes_dict=num_nodes_dict,
                stats_dict=stats_dict,
            )
        except Exception as e:
            print("[XCORR] model forward failed:", e)
            return {"error": "model_forward_failed", "exc": str(e)}

    # extract recon_feature_dict robustly
    recon_feature_dict = None
    try:
        if len(outputs) >= 7:
            _, _, _, _, _, _, recon_feature_dict = outputs[:7]
        else:
            # older interface - cannot compute xcorr
            return {"error": "no_recon_feature_in_outputs"}
    except Exception as e:
        print("[XCORR] unpack outputs failed:", e)
        return {"error": "bad_outputs", "exc": str(e)}

    if recon_feature_dict is None or nt not in recon_feature_dict:
        return {"error": "recon_feature_missing"}

    rf = recon_feature_dict[nt]  # (N, T, F)
    if rf is None or rf.numel() == 0:
        return {"error": "recon_feature_empty"}

    # resample target to recon time length
    try:
        target_raw = getattr(data[nt], "x_seq", None)
        if target_raw is None:
            return {"error": "target_missing"}
        target_res = trainer._resample_time(target_raw.to(device), rf.shape[1])
    except Exception as e:
        return {"error": "resample_failed", "exc": str(e)}

    # extract series for node_idx, feat_idx
    try:
        rf0 = rf[node_idx, :, feat_idx].detach().cpu().numpy()
        t0 = target_res[node_idx, :, feat_idx].detach().cpu().numpy()
    except Exception as e:
        return {"error": "index_failed", "exc": str(e)}

    # z-score
    rv = rf0
    tv = t0
    if np.std(rv) < 1e-8 or np.std(tv) < 1e-8:
        return {"error": "zero_variance"}

    rvz = (rv - rv.mean()) / (rv.std() + 1e-8)
    tvz = (tv - tv.mean()) / (tv.std() + 1e-8)

    corr = correlate(tvz, rvz, mode="full")
    lags = np.arange(-len(tvz) + 1, len(tvz))
    best_idx = int(np.argmax(corr))
    best_lag = int(lags[best_idx])
    best_corr = float(corr[best_idx] / len(tvz))
    print(f"[XCORR] nt={nt} node={node_idx} feat={feat_idx} best_lag={best_lag} best_corr={best_corr:.4f}")
    return {"best_lag": best_lag, "best_corr": best_corr, "corr_trace_len": len(corr)}
