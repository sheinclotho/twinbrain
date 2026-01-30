#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数字孪生脑主函数（批量多任务版）
此版本把 decoder-only warmup 的调用加入到 main 流程中（在 trainer 初始化并做完初始 diagnostics 后执行），
并在 warmup 后自动运行第2步的互相关滞后检验（xcorr best lag）。
"""
import os
import json
import logging
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import correlate

import mne
mne.set_log_level("WARNING")

# diagnostic helpers moved to utils.debug
try:
    from utils.debug import run_forward_diagnostics, diagnostics_plot_all, run_decoder_only_warmup
except Exception:
    run_forward_diagnostics = None
    diagnostics_plot_all = None
    run_decoder_only_warmup = None

from utils.utils import plot_recon_vs_target
from stim_align import batch_generate_stim
from node_generator import generate_nodes_all_regions
from edge_computer import generate_edges_with_dti_fallback
from mapper.atlas_mapper import BrainAtlas
from mapper.bids_mapper import BIDSMapper
from mapper.eeg_mapper import EEGMapper
from mapper.multi_modal_mapper import MultiModalMapper
from train.hetero_trainer import DynamicHeteroTrainer
# optional batch rescale utility
try:
    from utils.utils import compute_batch_alpha as train_compute_batch_alpha
except Exception:
    train_compute_batch_alpha = None

from utils.function import (
    discover_eeg_tasks,
    discover_fmri_tasks,
    load_fmri,
    load_dti,
    load_eeg,
    load_atlas,
    build_nodes,
    save_nodes_json,
    build_hetero_graph
)

plt.rcParams["font.family"] = "Arial"
logger = logging.getLogger("hetero_trainer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _compute_xcorr_best_lag(trainer, nt="fmri", node_idx=0, feat_idx=0):
    """
    Compute cross-correlation between recon_feature and target for given node/feature,
    return best_lag (in frames) and best_corr (normalized by length).
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
            outputs = trainer.model(data, edge_index_dict=edge_index_dict)
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


def main():
    BASE_DIR = Path(__file__).parent / "test_file3"
    SUBJECTS = [d for d in BASE_DIR.glob("sub-*") if d.is_dir()]
    atlas_path = BASE_DIR.parent / "schaefer200_mask_ready.json"
    atlas = load_atlas(atlas_path)

    for subj in SUBJECTS:
        logging.info(f"\n=== Processing {subj.name} ===")
        result_dir = subj / "results"
        os.makedirs(result_dir, exist_ok=True)

        PATHS = {
            "eeg_dir": subj / "eeg",
            "func_dir": subj / "func",
            "dti_npy": subj / "dwi" / f"{subj.name}_acq-AP_dwi_connectome.npy",
            "nodes_json": result_dir / "nodes.json",
            "hetero_model": result_dir / "hetero_gnn_trained.pt",
        }
        for p in PATHS.values():
            os.makedirs(Path(p).parent, exist_ok=True)

        eeg_tasks = discover_eeg_tasks(PATHS["eeg_dir"])
        fmri_tasks = discover_fmri_tasks(PATHS["func_dir"])
        logging.info(f"[Discover] EEG tasks: {eeg_tasks}")
        logging.info(f"[Discover] fMRI tasks: {fmri_tasks}")

        stim_cache = result_dir / "stim.pt"
        eeg_data_cache = result_dir / "eeg_data.pt"
        hetero_graphs_cache = result_dir / "hetero_graphs_for_training.pt"

        if stim_cache.exists() and eeg_data_cache.exists() and hetero_graphs_cache.exists():
            logging.info("[FULL CACHE HIT] Loading cached preprocessed data")
            stim = torch.load(stim_cache, map_location="cpu", weights_only=False)
            eeg_data = torch.load(eeg_data_cache, map_location="cpu", weights_only=False)
            hetero_graphs = torch.load(hetero_graphs_cache, map_location="cpu", weights_only=False)
        else:
            logging.info("[CACHE MISS] running preprocessing + graph building")
            stim = batch_generate_stim(subj)
            torch.save(stim, stim_cache)

            fmri_data = load_fmri(
                func_dir=PATHS["func_dir"],
                tasks=fmri_tasks,
                atlas_file=BASE_DIR.parent / "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii",
                label_file=BASE_DIR.parent / "schaefer200_mask_ready.json",
                brain_atlas=atlas,
                output_root=result_dir
            )
            eeg_data = load_eeg(
                eeg_dir=PATHS["eeg_dir"],
                brain_atlas=atlas,
                output_root=result_dir
            )
            torch.save(eeg_data, eeg_data_cache)

            dti = load_dti(PATHS["dti_npy"])
            nodes = build_nodes(atlas)
            save_nodes_json(nodes, PATHS["nodes_json"])

            hetero_graphs = build_hetero_graph(fmri_data, eeg_data, stim_dict=stim)
            torch.save(hetero_graphs, hetero_graphs_cache)

        # Trainer construction: use stable defaults (edit these in file before running)
        try:
            trainer = DynamicHeteroTrainer(
                hetero_data=hetero_graphs,
                hidden_dim=128,
                num_epochs=120,
                recon_weight=1.0,
                recon_norm_weight=3.0,        # 增强 normalized-space MSE
                recon_corr_weight=2.0,        # 加强 Pearson 相关的监督
                recon_feat_var_weight=0.02,   # 小量方差正则，避免 collapse
                feature_lr_mul=12.0,          # decoder LR 适度高
                scale_lr_mul=10.0,
                warmup_epochs=5,
                freeze_scale_during_warmup=True,
                debug=True,
                batch_rescale_fn=None,
                batch_rescale_cfg={"enable": False},
                spec_loss_weight=0.5, 
                spec_kernel_size=11
            )
        except Exception as e:
            logging.exception(f"[Train] Trainer initialization failed: {e}")
            with open(result_dir / "hetero_graphs_debug.json", "w") as fh:
                json.dump({k: len(v) if v is not None else None for k, v in hetero_graphs.items()}, fh, indent=2)
            continue

        # quick patch: replace fmri decoder with a freshly initialized TemporalDecoder (safe)
        from train.coder import TemporalDecoder
        import math

        # ensure conservative regularizers (avoid immediate collapse)
        trainer.recon_feat_var_weight = 0.0
        trainer.recon_corr_weight = 0.0
        trainer.recon_norm_weight = 1.0

        # build a fresh TemporalDecoder for fmri (channels smaller to be safe)
        new_fmri_decoder = TemporalDecoder(
            in_dim=trainer.model.hidden_dim,
            out_dim=trainer.model.node_feature_dims.get("fmri", 1),
            channels=min(128, trainer.model.hidden_dim),
            kernel_size=5,
            num_layers=3,
            dropout=0.1
        ).to(trainer.device)

        # 启用 auto_align（仅在 warmup/decoder-only 阶段生效）
        trainer.auto_align = True
        trainer.auto_align_max_lag = 150      # 给足够的搜索范围（你的 best_lag=111）


        # stable init function
        def init_temporal_decoder(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
            if isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.weight, 1.0)
                torch.nn.init.constant_(m.bias, 0.0)

        new_fmri_decoder.apply(init_temporal_decoder)
        # make final_proj.bias slightly non-zero to avoid strict zero output (small positive)
        try:
            if hasattr(new_fmri_decoder, "final_proj") and hasattr(new_fmri_decoder.final_proj, "bias"):
                with torch.no_grad():
                    new_fmri_decoder.final_proj.bias.fill_(0.01)
        except Exception:
            pass

        # Replace in model
        trainer.model.feature_decoders["fmri"] = new_fmri_decoder

        trainer.diagnostic_dir = str(result_dir / "diagnostics")
        os.makedirs(trainer.diagnostic_dir, exist_ok=True)

        # Diagnostics (moved to utils.debug)
        if run_forward_diagnostics is not None:
            run_forward_diagnostics(trainer, do_autoscale=False)
        if diagnostics_plot_all is not None:
            try:
                _ = diagnostics_plot_all(trainer, nt='fmri', node_idx=0, feat_idx=0)
                _ = diagnostics_plot_all(trainer, nt='eeg', node_idx=0, feat_idx=0)
            except Exception as e:
                print("[PLOT DIAG] diagnostics_plot_all failed:", e)


        try:
            out_pngs = plot_recon_vs_target(trainer, nt="eeg", node_idxs=[0,1,2], feature_idxs=None, max_plots=3, save_dir=trainer.diagnostic_dir)
            for p in out_pngs:
                print(f"[PLOT] saved {p}")
        except Exception as e:
            print(f"[PLOT] failed for eeg: {e}")

        # --- NEW: decoder-only warmup (focused training for decoders) ---
        if run_decoder_only_warmup is not None:
            try:
                print("[Run] Starting decoder-only warmup (50 epochs) with spec+shift...")
                warmup_res = run_decoder_only_warmup(
                    trainer,
                    epochs=50,
                    recon_norm_weight=5.0,
                    recon_corr_weight=4.0,
                    recon_feat_var_weight=0.0,
                    feature_lr_mul=20.0,
                    # spec/lowpass: encourage slow-trend fit during warmup
                    spec_loss_weight=0.0,
                    spec_kernel_size=21,         # larger kernel to capture slower fmri envelope
                    # allow small time shifts during matching (±5 frames)
                    shift_invariant_range=0,
                    shift_invariant_temp=0.1,    # low temperature => near-min selection
                    verbose=True
                )
                print("[Run] decoder-only warmup finished:", warmup_res.get("status"))
                if "diagnostics" in warmup_res:
                    diag_summary = warmup_res["diagnostics"].get("summary", {})
                    print("[Run] post-warmup diagnostics summary keys:", list(diag_summary.keys()))
            except Exception as e:
                print("[Run] decoder-only warmup failed:", e)
        else:
            print("[Run] run_decoder_only_warmup not available in utils.debug - skipping decoder-only warmup.")

        # --- NEW STEP 2: compute xcorr best lag for fmri node0 feat0 and print ---
        try:
            xcorr_res = _compute_xcorr_best_lag(trainer, nt="fmri", node_idx=0, feat_idx=0)
            print("[XCORR RESULT]", xcorr_res)
        except Exception as e:
            print("[XCORR] failed:", e)

        # Stage 1: warm-up (original pipeline)
        warmup_run_epochs = trainer.warmup_epochs + 10
        trainer.num_epochs = warmup_run_epochs
        logging.info(f"[Run] Stage 1 (warmup). Running {warmup_run_epochs} epochs...")
        trainer.train(num_epochs=warmup_run_epochs, verbose=True)
        try:
            trainer.save_model(result_dir / "hetero_gnn_after_warmup.pt")
        except Exception as e:
            logging.warning(f"[Train] Failed to save warmup checkpoint: {e}")

        # Stage 2: unfreeze scale & finetune
        logging.info("[Run] Stage 2 (unfreeze scale) - enabling align/temp and disabling batch-rescale for finetune.")
        trainer._set_scale_requires_grad(True)
        trainer.batch_rescale_cfg["enable"] = False
        trainer.use_batch_rescale = False
        trainer.align_weight = 1.0
        trainer.temp_weight = 5.0
        # keep conservative recon_feat_var_weight for now
        trainer.recon_feat_var_weight = 0.0
        trainer.scale_only_epochs = 5
        trainer.scale_only_lr_mul = 20

        # rebuild optimizer with feature & scale lr multipliers
        try:
            base_params = []
            feature_params = []
            scale_params = []
            for name, p in trainer.model.named_parameters():
                if not p.requires_grad:
                    continue
                if "feature_decoders" in name:
                    feature_params.append(p)
                elif ("log_scale" in name) or ("scale_" in name and "scale_fixed" not in name):
                    scale_params.append(p)
                else:
                    base_params.append(p)
            param_groups = [{"params": base_params}]
            if len(feature_params) > 0:
                param_groups.append({"params": feature_params, "lr": trainer.lr * float(getattr(trainer, "feature_lr_mul", 1.0))})
            if len(scale_params) > 0:
                param_groups.append({"params": scale_params, "lr": trainer.lr * float(getattr(trainer, "scale_lr_mul", 1.0))})
            param_groups.append({"params": list(trainer.aligner.parameters())})
            trainer.optimizer = torch.optim.Adam(param_groups, lr=trainer.lr, weight_decay=trainer.weight_decay)
            trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=30, gamma=0.7)
            logging.info("[Run] rebuilt optimizer with updated lr multipliers.")
        except Exception as e:
            logging.warning(f"[Run] failed to rebuild optimizer for scale lr: {e}")

            # --- Insert this right after you rebuild trainer.optimizer/trainer.scheduler (i.e. after the "rebuilt optimizer" log)
            # Run a fresh set of diagnostics now that optimizer / lr multipliers are in place.

            # ensure diagnostic dir
            trainer.diagnostic_dir = getattr(trainer, "diagnostic_dir", trainer.diagnostic_dir or str(result_dir / "diagnostics"))
            os.makedirs(trainer.diagnostic_dir, exist_ok=True)

            print("[PRE-FINETUNE DIAG] Running forward diagnostics with new optimizer/lr groups...")
            try:
                if run_forward_diagnostics is not None:
                    diag_out = run_forward_diagnostics(trainer, do_autoscale=False)
                    print("[PRE-FINETUNE DIAG] run_forward_diagnostics returned summary keys:", list(diag_out.get("summary", {}).keys()))
                else:
                    print("[PRE-FINETUNE DIAG] run_forward_diagnostics not available")
            except Exception as e:
                print("[PRE-FINETUNE DIAG] run_forward_diagnostics failed:", e)


            # compute best lag for fmri (useful to decide if shift-invariant needed)
            try:
                xres = _compute_xcorr_best_lag(trainer, nt="fmri", node_idx=0, feat_idx=0)
                print("[PRE-FINETUNE XCORR]", xres)
            except Exception as e:
                print("[PRE-FINETUNE XCORR] failed:", e)

            # optional: save a quick checkpoint before finetune
            try:
                ckpt_path = result_dir / "hetero_gnn_pre_finetune.pt"
                trainer.save_model(ckpt_path)
                print("[PRE-FINETUNE] saved checkpoint to", ckpt_path)
            except Exception as e:
                print("[PRE-FINETUNE] failed to save checkpoint:", e)

        finetune_epochs = 80
        logging.info(f"[Run] Stage 2: finetune for {finetune_epochs} epochs")
        trainer.train(num_epochs=finetune_epochs, verbose=True)

        try:
            trainer.save_model(PATHS["hetero_model"])
        except Exception as e:
            logging.warning(f"[Train] Failed to save model: {e}")

if __name__ == "__main__":

    main()