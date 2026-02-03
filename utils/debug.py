import os
import math
import random
import numpy as np

# Initialize random seeds before torch import to prevent THPGenerator errors
_INIT_SEED = 42
random.seed(_INIT_SEED)
np.random.seed(_INIT_SEED)

import matplotlib.pyplot as plt
import torch
# MUST call manual_seed immediately after torch import
torch.manual_seed(_INIT_SEED)

from scipy.signal import welch, correlate
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("debug")

def _resample_tensor_time(x: torch.Tensor, target_T: int):
    """
    x: (N, T_src, D) -> returns (N, target_T, D) via linear interpolation along time
    """
    if x is None or x.numel() == 0:
        return x
    N, T_src, D = x.shape
    if T_src == target_T:
        return x
    # permute to (N, D, T) and use interpolate
    xp = x.permute(0, 2, 1).contiguous()  # (N, D, T_src)
    xp_res = F.interpolate(xp, size=target_T, mode="linear", align_corners=False)
    return xp_res.permute(0, 2, 1).contiguous()  # (N, target_T, D)

def diagnostics_plot_all(trainer, nt='fmri', node_idx=0, feat_idx=0, save_dir=None, max_hidden_ch=8):
    """
    Robust diagnostics: produces and saves a set of plots and prints numeric summaries.
    Call this after trainer is initialized (before/after training).
    Returns list of saved file paths.
    """
    trainer.model.eval()
    try:
        device = trainer.device
        save_dir = save_dir or getattr(trainer, "diagnostic_dir", "./diagnostics")
        os.makedirs(save_dir, exist_ok=True)

        data = trainer.data_list[0].to(device)
        with torch.no_grad():
            enc_out = trainer.graph_encoder(data)
            x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict = enc_out
            outputs = trainer.model(
                data=data,
                edge_index_dict=edge_index_dict,
                encoded_dict=x_dict,
                num_nodes_dict=num_nodes_dict,
                stats_dict=stats_dict,
            )
            if len(outputs) >= 7:
                z_dict, gru_seq_dict, proj_seq_dict, recon_seq_dict, recon_seq_scaled, global_seq, recon_feature_dict = outputs[:7]
            else:
                raise RuntimeError("model did not return expected outputs with recon_feature_dict")

        if nt not in recon_feature_dict:
            raise RuntimeError(f"{nt} not found in model outputs")

        rf = recon_feature_dict[nt].detach().cpu()        # (N,T_rf,F)
        recon_denorm = recon_seq_dict[nt].detach().cpu()  # (N,T_rf,F)
        proj = proj_seq_dict[nt].detach().cpu()           # (N,T_proj,Hp)
        gru_out = gru_seq_dict.get(nt, None)
        if gru_out is not None:
            gru_out = gru_out.detach().cpu()              # (N,T_gru,H)

        # target and stats
        target_raw = getattr(data[nt], "x_seq", None)
        stats = stats_dict.get(nt, {})
        mean = stats.get("mean", None)
        std = stats.get("std", None)
        if target_raw is None or mean is None or std is None:
            raise RuntimeError("missing target or stats for diagnostics")

        # resample target to rf time length
        target_res = trainer._resample_time(target_raw.to(device), rf.shape[1]).detach().cpu()  # (N, T_rf, F)
        # resample proj and gru_out to rf time length for consistent plotting
        proj_rs = _resample_tensor_time(proj, rf.shape[1])
        gru_rs = _resample_tensor_time(gru_out, rf.shape[1]) if gru_out is not None else None

        N, T, F = rf.shape
        node_idx = int(min(node_idx, N-1))
        feat_idx = int(min(feat_idx, F-1))

        # build normalized target
        mean_exp = mean.expand(-1, rf.shape[1], -1)[:N, :T, :F].cpu()
        std_exp = std.expand(-1, rf.shape[1], -1)[:N, :T, :F].cpu()
        target_norm = (target_res - mean_exp) / (std_exp + 1e-8)

        t = np.arange(T)

        saved = []

        # 1) time series overlay
        plt.figure(figsize=(10,3))
        plt.plot(t, target_norm[node_idx,:,feat_idx].numpy(), label='target_norm', linewidth=1.2)
        plt.plot(t, rf[node_idx,:,feat_idx].numpy(), label='recon_feat (norm)', linewidth=1.0)
        plt.plot(t, recon_denorm[node_idx,:,feat_idx].numpy(), label='recon_denorm', linewidth=0.8, alpha=0.6)
        plt.legend(); plt.title(f"{nt} node={node_idx} feat={feat_idx}")
        f1 = os.path.join(save_dir, f"{nt}_ts_node{node_idx}_feat{feat_idx}.png")
        plt.tight_layout(); plt.savefig(f1, dpi=150); plt.close()
        saved.append(f1)

        # 2) proj channels (first few)
        Hp = proj_rs.shape[2]
        n_ch = min(max_hidden_ch, Hp)
        fig, axs = plt.subplots(n_ch, 1, figsize=(10, 2*n_ch), sharex=True)
        for i in range(n_ch):
            axs[i].plot(t, proj_rs[node_idx,:,i].numpy(), color='tab:blue')
            axs[i].set_ylabel(f"ch{i}")
        fig.suptitle(f"{nt} proj channels node={node_idx}")
        f2 = os.path.join(save_dir, f"{nt}_proj_node{node_idx}_chs.png")
        plt.tight_layout(); fig.savefig(f2, dpi=150); plt.close()
        saved.append(f2)

        # 3) gru channels if present
        if gru_rs is not None:
            Hg = gru_rs.shape[2]
            n_chg = min(max_hidden_ch, Hg)
            fig, axs = plt.subplots(n_chg, 1, figsize=(10, 2*n_chg), sharex=True)
            for i in range(n_chg):
                axs[i].plot(t, gru_rs[node_idx,:,i].numpy(), color='tab:orange')
                axs[i].set_ylabel(f"gch{i}")
            fig.suptitle(f"{nt} gru_out channels node={node_idx}")
            f3 = os.path.join(save_dir, f"{nt}_gru_node{node_idx}_chs.png")
            plt.tight_layout(); fig.savefig(f3, dpi=150); plt.close()
            saved.append(f3)
        else:
            f3 = None

        # 4) PSD comparison
        from scipy.signal import welch
        fs = 1.0
        f_t, P_t = welch(target_norm[node_idx,:,feat_idx].numpy(), fs=fs, nperseg=min(128, T))
        f_r, P_r = welch(rf[node_idx,:,feat_idx].numpy(), fs=fs, nperseg=min(128, T))
        plt.figure(figsize=(6,3))
        plt.semilogy(f_t, P_t + 1e-12, label='target_norm PSD')
        plt.semilogy(f_r, P_r + 1e-12, label='recon_feat PSD')
        plt.legend(); plt.title(f"{nt} PSD node={node_idx} feat={feat_idx}")
        f4 = os.path.join(save_dir, f"{nt}_psd_node{node_idx}_feat{feat_idx}.png")
        plt.tight_layout(); plt.savefig(f4, dpi=150); plt.close()
        saved.append(f4)

        # 5) xcorr
        tv = target_norm[node_idx,:,feat_idx].numpy()
        rv = rf[node_idx,:,feat_idx].numpy()
        tvz = (tv - tv.mean()) / (tv.std()+1e-8)
        rvz = (rv - rv.mean()) / (rv.std()+1e-8)
        corr = correlate(tvz, rvz, mode='full')
        lags = np.arange(-len(tvz)+1, len(tvz))
        best_idx = np.argmax(corr)
        best_lag = lags[best_idx]
        best_corr = corr[best_idx] / len(tvz)
        plt.figure(figsize=(6,3)); plt.plot(lags, corr); plt.axvline(best_lag, color='r', linestyle='--'); plt.title(f"{nt} xcorr best_lag={best_lag} corr={best_corr:.4f}")
        f5 = os.path.join(save_dir, f"{nt}_xcorr_node{node_idx}_feat{feat_idx}.png")
        plt.tight_layout(); plt.savefig(f5, dpi=150); plt.close()
        saved.append(f5)

        # 6) LS-scaling alpha & rel per-feature summary (flatten over nodes/time)
        r_flat = rf.reshape(-1, rf.shape[-1]).numpy()
        t_flat = target_norm.reshape(-1, target_norm.shape[-1]).numpy()
        alphas = []
        rels = []
        for f in range(r_flat.shape[1]):
            rvf = r_flat[:,f]
            tvf = t_flat[:,f]
            den = (rvf*rvf).sum() + 1e-8
            alpha = (rvf*tvf).sum() / den if den>0 else 0.0
            alphas.append(alpha)
            r_scaled = rvf * alpha
            rel = np.linalg.norm(r_scaled - tvf) / (np.linalg.norm(tvf) + 1e-8)
            rels.append(rel)
        print(f"[DIAG_EXTRA] {nt} alphas mean={np.mean(alphas):.4f} med={np.median(alphas):.4f}")
        print(f"[DIAG_EXTRA] {nt} rels mean={np.mean(rels):.4f} med={np.median(rels):.4f}")
        return saved
    finally:
        trainer.model.train()


def run_forward_diagnostics(trainer,
                            do_autoscale: bool = False,
                            autoscale_target: str = "fmri",
                            max_sample_nodes: int = 8) -> dict:
    """
    Forward-run diagnostics for a DynamicHeteroTrainer instance.

    If trainer.auto_align is True, estimate integer lag(s) (or reuse trainer._auto_align_cache)
    and apply the same integer roll/zero-fill to recon_feature/recon_seq for diagnostics, so
    metrics reflect the aligned outputs used inside training.

    Returns a dict of summary statistics and the raw outputs for further programmatic use.
    """
    out = {}
    trainer.model.eval()
    device = getattr(trainer, "device", torch.device("cpu"))
    data = trainer.data_list[0].to(device)

    # run encoder to get edge_index and stats
    try:
        enc_out = trainer.graph_encoder(data)
    except Exception as e:
        logger.exception(f"[diagnostics] graph_encoder failed: {e}")
        trainer.model.train()
        return {"error": "graph_encoder_failed", "exc": str(e)}

    # enc_out expected: (x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict)
    try:
        x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict = enc_out
    except Exception as e:
        logger.exception(f"[diagnostics] unexpected enc_out format: {e}")
        trainer.model.train()
        return {"error": "bad_encoder_output", "exc": str(e)}

    # forward model
    try:
        with torch.no_grad():
            outputs = trainer.model(
                data=data,
                edge_index_dict=edge_index_dict,
                encoded_dict=x_dict,
                num_nodes_dict=num_nodes_dict,
                stats_dict=stats_dict,
            )
    except Exception as e:
        logger.exception(f"[diagnostics] model forward failed: {e}")
        trainer.model.train()
        return {"error": "model_forward_failed", "exc": str(e)}

    # unpack outputs robustly
    recon_feature_dict = None
    try:
        if len(outputs) >= 7:
            z_dict, gru_seq_dict, proj_seq_dict, recon_seq_dict, recon_seq_scaled, global_seq, recon_feature_dict = outputs[:7]
        else:
            z_dict, gru_seq_dict, proj_seq_dict, recon_seq_dict, recon_seq_scaled, global_seq = outputs[:6]
            recon_feature_dict = None
    except Exception as e:
        logger.exception(f"[diagnostics] unpacking model outputs failed: {e}")
        trainer.model.train()
        return {"error": "bad_model_outputs", "exc": str(e)}

    # If auto_align is enabled on trainer, estimate/apply integer lag(s) to outputs for diagnostics.
    # We operate on local copies so we don't mutate model outputs used elsewhere.
    applied_lags = {}
    if getattr(trainer, "auto_align", False) and recon_feature_dict is not None:
        # only align modalities present in recon_feature_dict and typically only 'fmri' (configurable)
        for nt in list(recon_feature_dict.keys()):
            if getattr(trainer, "auto_align_scope", "warmup") not in ["always", "warmup"]:
                # guard (unlikely) but keep consistent
                continue
            if nt != "fmri" and getattr(trainer, "auto_align_apply_to", "fmri") == "fmri":
                continue
            try:
                # build simple 1D ref/qry by mean over batch and features
                rf = recon_feature_dict[nt]
                if rf is None or rf.numel() == 0:
                    continue
                # resample target to recon time length
                target_raw = getattr(data[nt], "x_seq", None)
                if target_raw is None:
                    continue
                # use trainer._resample_time to match lengths
                t_res = trainer._resample_time(target_raw.to(device), rf.shape[1])
                ref = t_res.detach().cpu().numpy().mean(axis=(0, 2))
                qry = rf.detach().cpu().numpy().mean(axis=(0, 2))
                # check cache first (cache key per nt)
                cache_key = f"diag_{nt}"
                if hasattr(trainer, "_auto_align_cache") and cache_key in trainer._auto_align_cache:
                    best_lag = int(trainer._auto_align_cache[cache_key])
                else:
                    max_lag = int(getattr(trainer, "auto_align_max_lag", 120))
                    # use trainer helper if available else fallback
                    if hasattr(trainer, "_estimate_best_lag_cpu"):
                        best_lag = int(trainer._estimate_best_lag_cpu(ref, qry, max_lag=max_lag))
                    else:
                        # local estimate via numpy correlate
                        from scipy.signal import correlate as _corr
                        T = len(ref)
                        corr_full = _corr(ref, qry, mode="full")
                        lags = np.arange(-T + 1, T)
                        mask = (lags >= -max_lag) & (lags <= max_lag)
                        corr_full[~mask] = -1e18
                        best_idx = int(np.argmax(corr_full))
                        best_lag = int(lags[best_idx])
                    # cache it
                    if not hasattr(trainer, "_auto_align_cache"):
                        trainer._auto_align_cache = {}
                    trainer._auto_align_cache[cache_key] = int(best_lag)

                # only accept reasonably large correlation to avoid noisy shifts
                # compute corr value for reporting
                from scipy.signal import correlate as _corr2
                rvz = (qry - qry.mean()) / (qry.std() + 1e-8)
                tvz = (ref - ref.mean()) / (ref.std() + 1e-8)
                corr = _corr2(tvz, rvz, mode="full")
                lags = np.arange(-len(tvz)+1, len(tvz))
                best_idx = int(np.argmax(corr))
                best_corr = float(corr[best_idx] / len(tvz))

                applied_lags[nt] = {"best_lag": int(best_lag), "best_corr": best_corr}

                if best_lag != 0:
                    # apply to local copies of recon_feature_dict / recon_seq_dict / recon_seq_scaled
                    try:
                        # recon_feature copy
                        recon_feature_dict[nt] = _align_tensor_for_diag(recon_feature_dict[nt], best_lag)
                    except Exception:
                        pass
                    try:
                        if isinstance(recon_seq_dict, dict) and nt in recon_seq_dict:
                            recon_seq_dict[nt] = _align_tensor_for_diag(recon_seq_dict[nt], best_lag)
                        if isinstance(recon_seq_scaled, dict) and nt in recon_seq_scaled:
                            recon_seq_scaled[nt] = _align_tensor_for_diag(recon_seq_scaled[nt], best_lag)
                    except Exception:
                        pass
            except Exception as e:
                if trainer.debug:
                    logger.warning(f"[diag auto_align] failed for {nt}: {e}")

    # proceed with diagnostics printing & summary using (possibly aligned) recon_feature_dict / recon_seq_dict etc.
    out_summary = {}
    print("[DIAG] Per-modality summary (first data_list sample):")
    for nt in trainer.metadata[0]:
        summary = {}
        recon = recon_seq_dict.get(nt) if isinstance(recon_seq_dict, dict) else None
        rf = recon_feature_dict.get(nt) if (recon_feature_dict is not None and isinstance(recon_feature_dict, dict) and nt in recon_feature_dict) else None
        target = getattr(data[nt], "x_seq", None)
        stats = stats_dict.get(nt, {"mean": None, "std": None}) if isinstance(stats_dict, dict) else {"mean": None, "std": None}
        mean = stats.get("mean", None)
        std = stats.get("std", None)

        # print applied lag if any
        if nt in applied_lags:
            bl = applied_lags[nt]
            print(f"[DIAG:{nt}] AUTO_ALIGN applied best_lag={bl['best_lag']} best_corr={bl['best_corr']:.4f}")

        print(f"\n[DIAG:{nt}] recon_denorm shape={None if recon is None else tuple(recon.shape)}")
        summary["recon_denorm_shape"] = None if recon is None else tuple(recon.shape)
        if recon is not None:
            try:
                print(f"[DIAG:{nt}] recon_denorm mean={float(recon.mean()):.6f}, std={float(recon.std()):.6f}")
                summary["recon_denorm_mean"] = float(recon.mean())
                summary["recon_denorm_std"] = float(recon.std())
            except Exception:
                summary["recon_denorm_mean"] = None
                summary["recon_denorm_std"] = None

        if rf is None:
            print(f"[DIAG:{nt}] recon_feature: MISSING")
            summary["recon_feature_shape"] = None
        else:
            try:
                print(f"[DIAG:{nt}] recon_feature shape={tuple(rf.shape)} mean={float(rf.mean()):.6f} std={float(rf.std()):.6f}")
                summary["recon_feature_shape"] = tuple(rf.shape)
                summary["recon_feature_mean"] = float(rf.mean())
                summary["recon_feature_std"] = float(rf.std())
            except Exception:
                summary["recon_feature_shape"] = None

        # print target_norm summary safely if available
        if target is not None and mean is not None and std is not None and rf is not None:
            try:
                target_res = trainer._resample_time(target.to(device), rf.shape[1])
                print(f"[DIAG:{nt}] target_res shape={tuple(target_res.shape)}")
                t_np = target_res.detach().cpu().numpy()
                t_mean = float(t_np.mean()) if t_np.size > 0 else float("nan")
                t_std = float(t_np.std()) if t_np.size > 0 else float("nan")
                print(f"[DIAG:{nt}] target_norm overall mean={t_mean:.6e} std={t_std:.6e}")
                summary["target_res_shape"] = tuple(target_res.shape)
                summary["target_mean"] = t_mean
                summary["target_std"] = t_std
            except Exception as e:
                logger.debug(f"[DIAG:{nt}] cannot compute target stats: {e}")
                summary["target_res_shape"] = None

        # compute per-feature Pearson/alpha only if rf and target_norm present and have valid shapes
        if rf is not None and target is not None and mean is not None and std is not None:
            try:
                rf_np = rf.detach().cpu().numpy()
                if rf_np.ndim == 2:
                    rf_np = rf_np.reshape(rf_np.shape[0], rf_np.shape[1], 1)
                target_res = trainer._resample_time(target.to(device), rf.shape[1])
                t_np = target_res.detach().cpu().numpy()
                Nr, Tr, Fr = rf_np.shape
                Nt, Tt, Ft = t_np.shape
                mN, mT, mF = min(Nr, Nt), min(Tr, Tt), min(Fr, Ft)
                if mF <= 0 or (mN * mT) == 0:
                    print(f"[DIAG:{nt}] insufficient shape for per-feature stats (mN={mN}, mT={mT}, mF={mF})")
                    summary["per_feature_stats"] = None
                else:
                    rf_crop = rf_np[:mN, :mT, :mF]
                    mean_expand = mean.expand(-1, rf.shape[1], -1)[:mN, :mT, :mF].detach().cpu().numpy()
                    std_expand = std.expand(-1, rf.shape[1], -1)[:mN, :mT, :mF].detach().cpu().numpy()
                    tnorm_crop = (t_np[:mN, :mT, :mF] - mean_expand) / (std_expand + 1e-8)
                    rf_flat = rf_crop.reshape(-1, mF)
                    tnorm_flat = tnorm_crop.reshape(-1, mF)

                    corrs, alphas_ls, alphas_std = [], [], []
                    for f in range(mF):
                        r_vec = rf_flat[:, f]
                        t_vec = tnorm_flat[:, f]
                        if np.isfinite(r_vec).all() and np.isfinite(t_vec).all() and r_vec.std() > 1e-8 and t_vec.std() > 1e-8:
                            pr = float(np.corrcoef(r_vec, t_vec)[0, 1])
                            num = float((r_vec * t_vec).sum())
                            den = float((r_vec * r_vec).sum()) + 1e-8
                            al = num / den
                            astd = float(np.std(t_vec) / (np.std(r_vec) + 1e-8))
                        else:
                            pr = float("nan")
                            al = float("nan")
                            astd = float("nan")
                        corrs.append(pr)
                        alphas_ls.append(al)
                        alphas_std.append(astd)

                    def summar(x):
                        arr = np.array(x, dtype=float)
                        return {
                            "mean": float(np.nanmean(arr)),
                            "med": float(np.nanmedian(arr)),
                            "p10": float(np.nanpercentile(arr, 10)),
                            "p90": float(np.nanpercentile(arr, 90))
                        }

                    summary["pearson"] = summar(corrs)
                    summary["alpha_ls"] = summar(alphas_ls)
                    summary["alpha_std"] = summar(alphas_std)

                    print(f"[DIAG:{nt}] Pearson per-feature: mean={summary['pearson']['mean']:.4f}, med={summary['pearson']['med']:.4f}, p10={summary['pearson']['p10']:.4f}, p90={summary['pearson']['p90']:.4f}")
                    print(f"[DIAG:{nt}] alpha_ls per-feature: mean={summary['alpha_ls']['mean']:.4f}, med={summary['alpha_ls']['med']:.4f}, p10={summary['alpha_ls']['p10']:.4f}, p90={summary['alpha_ls']['p90']:.4f}")
                    print(f"[DIAG:{nt}] alpha_std per-feature: mean={summary['alpha_std']['mean']:.4f}, med={summary['alpha_std']['med']:.4f}, p10={summary['alpha_std']['p10']:.4f}, p90={summary['alpha_std']['p90']:.4f}")
            except Exception as e:
                logger.exception(f"[DIAG:{nt}] per-feature stats computation failed: {e}")
                summary["per_feature_stats_error"] = str(e)

        # scale info if available
        dec = getattr(trainer.model, "denorm_decoders", None)
        if dec is not None:
            try:
                if hasattr(dec, "get_scale"):
                    st = dec.get_scale(nt)
                    print(f"[DIAG:{nt}] scale mean={float(st.detach().mean().cpu()):.6f}")
                    summary["scale_mean"] = float(st.detach().mean().cpu())
                else:
                    if hasattr(dec, f"log_scale_{nt}"):
                        p = getattr(dec, f"log_scale_{nt}")
                        val = float(p.detach().mean().cpu())
                        print(f"[DIAG:{nt}] log_scale mean={val:.6f}, exp={float(torch.exp(p).mean().cpu()):.6f}")
                        summary["scale_param_mean"] = val
                    elif hasattr(dec, f"scale_{nt}"):
                        p = getattr(dec, f"scale_{nt}")
                        print(f"[DIAG:{nt}] scale mean={float(p.detach().mean().cpu()):.6f}")
                        summary["scale_param_mean"] = float(p.detach().mean().cpu())
            except Exception as e:
                logger.debug(f"[DIAG:{nt}] scale read failed: {e}")

        out_summary[nt] = summary

    out["summary"] = out_summary
    trainer.model.train()
    print("[DIAG] Done.")
    return out


# small helper for diagnostics alignment (local, non-intrusive)
def _align_tensor_for_diag(tensor: torch.Tensor, lag: int, fill_value: float = 0.0) -> torch.Tensor:
    """
    Align a (N,T,F) tensor by integer lag for diagnostics only (returns new tensor).
    Positive lag => recon lags target => shift left by lag (roll -lag) then zero-fill tail.
    """
    if lag == 0:
        return tensor
    try:
        t = tensor.clone()
        B, T, F = t.shape
        t = torch.roll(t, shifts=-int(lag), dims=1)
        if lag > 0:
            t[:, T - lag : T, :] = fill_value
        else:
            t[:, : -lag, :] = fill_value
        return t
    except Exception:
        return tensor


# utils/debug.py






def run_decoder_only_warmup(trainer,
                            epochs: int = 20,
                            recon_norm_weight: Optional[float] = 4.0,
                            recon_corr_weight: Optional[float] = 3.0,
                            recon_feat_var_weight: Optional[float] = 0.01,
                            feature_lr_mul: Optional[float] = None,
                            # new params for spec + shift scheduling
                            spec_loss_weight: Optional[float] = None,
                            spec_kernel_size: Optional[int] = None,
                            shift_invariant_range: Optional[int] = None,
                            shift_invariant_temp: Optional[float] = None,
                            verbose: bool = True) -> Dict[str, Any]:
    """
    Freeze everything except decoder parameters, optionally enable spec_loss and shift-invariant loss
    for the warmup, rebuild a temp optimizer, run epochs, then restore everything.
    """
    result = {"status": "started"}
    # save originals
    orig_recon_norm = getattr(trainer, "recon_norm_weight", None)
    orig_recon_corr = getattr(trainer, "recon_corr_weight", None)
    orig_recon_feat_var = getattr(trainer, "recon_feat_var_weight", None)
    orig_feature_lr_mul = getattr(trainer, "feature_lr_mul", None)
    orig_spec_loss_weight = getattr(trainer, "spec_loss_weight", None)
    orig_spec_kernel_size = getattr(trainer, "spec_kernel_size", None)
    orig_shift_range = getattr(trainer, "shift_invariant_range", None)
    orig_shift_temp = getattr(trainer, "shift_invariant_temp", None)

    # store original requires_grad flags
    orig_req = {name: p.requires_grad for name, p in trainer.model.named_parameters()}

    # freeze all
    for name, p in trainer.model.named_parameters():
        p.requires_grad = False

    # enable decoder params + decoder_input_proj if present
    enabled_params = []
    for nt, dec in trainer.model.feature_decoders.items():
        for p in dec.parameters():
            p.requires_grad = True
            enabled_params.append(p)
    if hasattr(trainer.model, "decoder_input_proj"):
        for nt, proj in trainer.model.decoder_input_proj.items():
            for p in proj.parameters():
                p.requires_grad = True
                enabled_params.append(p)

    # set stronger supervision weights temporarily
    if recon_norm_weight is not None:
        trainer.recon_norm_weight = recon_norm_weight
    if recon_corr_weight is not None:
        trainer.recon_corr_weight = recon_corr_weight
    if recon_feat_var_weight is not None:
        trainer.recon_feat_var_weight = recon_feat_var_weight
    if feature_lr_mul is not None:
        trainer.feature_lr_mul = feature_lr_mul

    # set spec/shift warmup params if provided
    if spec_loss_weight is not None:
        trainer.spec_loss_weight = spec_loss_weight
    if spec_kernel_size is not None:
        trainer.spec_kernel_size = spec_kernel_size
    if shift_invariant_range is not None:
        trainer.shift_invariant_range = shift_invariant_range
    if shift_invariant_temp is not None:
        trainer.shift_invariant_temp = shift_invariant_temp

    # save old optimizer/scheduler and build temporary one for decoder params
    old_optimizer = getattr(trainer, "optimizer", None)
    old_scheduler = getattr(trainer, "scheduler", None)

    temp_lr = trainer.lr * float(getattr(trainer, "feature_lr_mul", 1.0))
    temp_opt = torch.optim.Adam([p for p in enabled_params if p.requires_grad], lr=temp_lr, weight_decay=getattr(trainer, "weight_decay", 0.0))
    temp_sched = torch.optim.lr_scheduler.StepLR(temp_opt, step_size=30, gamma=0.7)

    trainer.optimizer = temp_opt
    trainer.scheduler = temp_sched

    # run focused training
    try:
        logger.info(f"[decoder-warmup] running {epochs} epochs with lr={temp_lr} spec_w={getattr(trainer,'spec_loss_weight',None)} shift_range={getattr(trainer,'shift_invariant_range',None)}")
        trainer.train(num_epochs=epochs, verbose=verbose)
        result["status"] = "completed"
    except Exception as e:
        logger.exception(f"[decoder-warmup] training failed: {e}")
        result["status"] = "failed"
        result["exc"] = str(e)

    # restore original optimizer/scheduler
    trainer.optimizer = old_optimizer
    trainer.scheduler = old_scheduler

    # restore requires_grad states
    for name, p in trainer.model.named_parameters():
        p.requires_grad = orig_req.get(name, True)

    # restore original hyperparams
    if orig_recon_norm is not None:
        trainer.recon_norm_weight = orig_recon_norm
    if orig_recon_corr is not None:
        trainer.recon_corr_weight = orig_recon_corr
    if orig_recon_feat_var is not None:
        trainer.recon_feat_var_weight = orig_recon_feat_var
    if orig_feature_lr_mul is not None:
        trainer.feature_lr_mul = orig_feature_lr_mul
    # restore spec/shift params
    trainer.spec_loss_weight = orig_spec_loss_weight
    trainer.spec_kernel_size = orig_spec_kernel_size
    trainer.shift_invariant_range = orig_shift_range
    trainer.shift_invariant_temp = orig_shift_temp

    # run diagnostics if possible
    try:
        diag = run_forward_diagnostics(trainer, do_autoscale=False)
        result["diagnostics"] = diag
    except Exception as e:
        logger.debug(f"[decoder-warmup] post-warmup diagnostics failed: {e}")
        result["diagnostics_error"] = str(e)

    return result