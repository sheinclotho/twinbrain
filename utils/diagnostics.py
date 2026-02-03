import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any, List
from scipy.signal import welch, correlate

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
    xp = x.permute(0, 2, 1).contiguous()
    xp_res = F.interpolate(xp, size=target_T, mode="linear", align_corners=False)
    return xp_res.permute(0, 2, 1).contiguous()


def run_comprehensive_diagnostics(
    trainer,
    save_dir: Optional[str] = None,
    save_plots: bool = True,
    plot_nodes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive diagnostics - runs model forward ONCE and analyzes all modalities.
    
    Provides:
    - Reconstruction quality metrics (Pearson, RMSE, SNR)
    - Signal statistics and temporal alignment  
    - Model parameter tracking
    - Optional visualization plots
    - JSON export of metrics
    
    Args:
        trainer: DynamicHeteroTrainer instance
        save_dir: Directory for plots and metrics
        save_plots: Whether to generate plots
        plot_nodes: Node indices to analyze (default [0])
    
    Returns:
        Dictionary with comprehensive diagnostic data
    """
    trainer.model.eval()
    try:
        device = trainer.device
        save_dir = save_dir or getattr(trainer, "diagnostic_dir", "./diagnostics")
        os.makedirs(save_dir, exist_ok=True)
        plot_nodes = plot_nodes or [0]
        
        result = {"modalities": {}, "training_state": {}, "model_info": {}}
        data = trainer.data_list[0].to(device)
        
        # === Single forward pass ===
        with torch.no_grad():
            try:
                enc_out = trainer.graph_encoder(data)
                x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict = enc_out
            except Exception as e:
                logger.exception(f"[diag] graph_encoder failed: {e}")
                return {"error": "graph_encoder_failed", "exception": str(e)}
            
            try:
                outputs = trainer.model(
                    data=data,
                    edge_index_dict=edge_index_dict,
                    encoded_dict=x_dict,
                    num_nodes_dict=num_nodes_dict,
                    stats_dict=stats_dict,
                )
            except Exception as e:
                logger.exception(f"[diag] model forward failed: {e}")
                return {"error": "model_forward_failed", "exception": str(e)}
            
            if len(outputs) >= 7:
                z_dict, gru_seq_dict, proj_seq_dict, recon_seq_dict, recon_seq_scaled, global_seq, recon_feature_dict = outputs[:7]
            else:
                z_dict, gru_seq_dict, proj_seq_dict, recon_seq_dict, recon_seq_scaled, global_seq = outputs[:6]
                recon_feature_dict = None
        
        # === Training state ===
        result["training_state"] = {
            "current_epoch": getattr(trainer, "current_epoch", 0),
            "learning_rate": trainer.optimizer.param_groups[0]["lr"] if hasattr(trainer, "optimizer") else None,
        }
        
        # === Model info ===
        result["model_info"] = {
            "hidden_dim": trainer.hidden_dim,
            "total_params": sum(p.numel() for p in trainer.model.parameters()),
            "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        }
        
        # === Analyze each modality ===
        for nt in trainer.metadata[0]:
            target_raw = getattr(data[nt], "x_seq", None)
            if target_raw is None:
                continue
            
            stats = stats_dict.get(nt, {})
            mean, std = stats.get("mean"), stats.get("std")
            recon_feat = recon_feature_dict.get(nt) if recon_feature_dict else None
            
            if recon_feat is None or mean is None or std is None:
                continue
            
            # Prepare data
            recon_cpu = recon_feat.detach().cpu()
            target_res = trainer._resample_time(target_raw.to(device), recon_feat.shape[1]).detach().cpu()
            
            N, T, F = recon_cpu.shape
            mean_exp = mean.expand(-1, T, -1)[:N, :T, :F].cpu()
            std_exp = std.expand(-1, T, -1)[:N, :T, :F].cpu()
            target_norm = (target_res - mean_exp) / (std_exp + 1e-8)
            
            # === Compute metrics ===
            recon_flat = recon_cpu.reshape(-1, F).numpy()
            target_flat = target_norm.reshape(-1, F).numpy()
            
            pearson_corrs, rmse_values, snr_values = [], [], []
            for f in range(F):
                r, t = recon_flat[:, f], target_flat[:, f]
                if np.isfinite(r).all() and np.isfinite(t).all() and r.std() > 1e-8 and t.std() > 1e-8:
                    pearson_corrs.append(float(np.corrcoef(r, t)[0, 1]))
                    rmse_values.append(float(np.sqrt(np.mean((r - t)**2))))
                    signal_pow, noise_pow = np.mean(t**2), np.mean((r - t)**2)
                    snr_values.append(10 * np.log10(signal_pow / (noise_pow + 1e-10)) if signal_pow > 0 else float('-inf'))
                else:
                    pearson_corrs.extend([float('nan')])
                    rmse_values.extend([float('nan')])
                    snr_values.extend([float('nan')])
            
            # Temporal alignment
            node_idx = min(plot_nodes[0], N - 1) if N > 0 else 0
            if N > 0 and F > 0:
                tv, rv = target_norm[node_idx, :, 0].numpy(), recon_cpu[node_idx, :, 0].numpy()
                if tv.std() > 1e-8 and rv.std() > 1e-8:
                    tvz = (tv - tv.mean()) / tv.std()
                    rvz = (rv - rv.mean()) / rv.std()
                    corr = correlate(tvz, rvz, mode='full')
                    lags = np.arange(-len(tvz) + 1, len(tvz))
                    best_idx = int(np.argmax(corr))
                    best_lag, best_corr = int(lags[best_idx]), float(corr[best_idx] / len(tvz))
                else:
                    best_lag, best_corr = 0, 0.0
            else:
                best_lag, best_corr = 0, 0.0
            
            # Store metrics
            modality_result = {
                "metrics": {
                    "reconstruction": {
                        "pearson_mean": float(np.nanmean(pearson_corrs)),
                        "pearson_std": float(np.nanstd(pearson_corrs)),
                        "rmse_mean": float(np.nanmean(rmse_values)),
                        "snr_mean_db": float(np.nanmean([s for s in snr_values if not np.isinf(s)])) if any(not np.isinf(s) for s in snr_values) else float('nan'),
                    },
                    "alignment": {"lag_frames": best_lag, "correlation": best_corr},
                    "statistics": {
                        "target_mean": float(target_norm.mean()),
                        "target_std": float(target_norm.std()),
                        "recon_mean": float(recon_cpu.mean()),
                        "recon_std": float(recon_cpu.std()),
                    },
                    "shapes": {"nodes": N, "time_steps": T, "features": F},
                }
            }
            
            # Scale parameters
            dec = getattr(trainer.model, "denorm_decoders", None)
            if dec:
                try:
                    if hasattr(dec, "get_scale"):
                        scale = dec.get_scale(nt)
                        modality_result["metrics"]["scale"] = float(scale.detach().mean().cpu())
                    elif hasattr(dec, f"log_scale_{nt}"):
                        log_scale = getattr(dec, f"log_scale_{nt}")
                        modality_result["metrics"]["scale"] = float(torch.exp(log_scale).mean().cpu())
                except:
                    pass
            
            # === Generate plots ===
            if save_plots:
                plot_files = []
                for node_idx in plot_nodes:
                    if node_idx >= N:
                        continue
                    
                    # Reconstruction quality plot
                    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                    for feat_idx in range(min(3, F)):
                        t_vec = target_norm[node_idx, :, feat_idx].numpy()
                        r_vec = recon_cpu[node_idx, :, feat_idx].numpy()
                        axes[0].plot(t_vec, label=f'Target F{feat_idx}', alpha=0.7, linewidth=0.8)
                        axes[1].plot(r_vec, label=f'Recon F{feat_idx}', alpha=0.7, linewidth=0.8)
                    
                    axes[0].set_ylabel('Target (norm)'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
                    axes[1].set_ylabel('Reconstruction'); axes[1].set_xlabel('Time'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
                    fig.suptitle(f'{nt} Node {node_idx} | Pearson: {np.nanmean(pearson_corrs):.3f}, RMSE: {np.nanmean(rmse_values):.3f}')
                    plt.tight_layout()
                    
                    plot_path = os.path.join(save_dir, f'{nt}_node{node_idx}_quality.png')
                    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    plot_files.append(plot_path)
                
                modality_result["plot_files"] = plot_files
            
            result["modalities"][nt] = modality_result
        
        # === Save JSON summary ===
        json_path = os.path.join(save_dir, "diagnostics_summary.json")
        try:
            json_data = {k: v for k, v in result.items() if k != "modalities" or k == "modalities"}
            # Clean modalities for JSON
            json_data["modalities"] = {nt: {"metrics": mod.get("metrics", {})} for nt, mod in result["modalities"].items()}
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            result["summary_file"] = json_path
        except Exception as e:
            logger.warning(f"[diag] Failed to save JSON: {e}")
        
        # === Print summary ===
        print("\n" + "="*70)
        print("DIAGNOSTIC SUMMARY")
        print("="*70)
        print(f"Training: Epoch {result['training_state']['current_epoch']}, LR {result['training_state']['learning_rate']}")
        print(f"Model: {result['model_info']['trainable_params']:,} trainable params\n")
        
        for nt, mod in result["modalities"].items():
            m = mod["metrics"]
            r = m["reconstruction"]
            a = m["alignment"]
            print(f"[{nt.upper()}] Pearson={r['pearson_mean']:.3f}±{r['pearson_std']:.3f}, "
                  f"RMSE={r['rmse_mean']:.4f}, SNR={r['snr_mean_db']:.1f}dB, "
                  f"Lag={a['lag_frames']} frames")
        print("="*70 + "\n")
        
        return result
    finally:
        trainer.model.train()


# Legacy wrappers for backward compatibility
def diagnostics_plot_all(trainer, nt='fmri', node_idx=0, feat_idx=0, save_dir=None, max_hidden_ch=8):
    """DEPRECATED: Use run_comprehensive_diagnostics() instead."""
    logger.warning("diagnostics_plot_all is deprecated. Use run_comprehensive_diagnostics().")
    result = run_comprehensive_diagnostics(trainer, save_dir=save_dir, save_plots=True, plot_nodes=[node_idx])
    return result.get("modalities", {}).get(nt, {}).get("plot_files", [])


def run_forward_diagnostics(trainer, do_autoscale: bool = False, autoscale_target: str = "fmri", max_sample_nodes: int = 8) -> dict:
    """DEPRECATED: Use run_comprehensive_diagnostics() instead."""
    logger.warning("run_forward_diagnostics is deprecated. Use run_comprehensive_diagnostics().")
    result = run_comprehensive_diagnostics(trainer, save_plots=False, plot_nodes=[0])
    return {"summary": {nt: mod.get("metrics", {}) for nt, mod in result.get("modalities", {}).items()}}
