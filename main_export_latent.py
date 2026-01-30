#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数字孪生脑 - 单次 latent 预测导出（Unity 可视化用）

参考 main_v4.py：
- 使用相同的预处理 / 构图 / DynamicHeteroTrainer 初始化流程
- 不跑训练，只加载/构建一次模型
- 在 latent 空间上做一次未来预测，计算预测指标
- 导出真实 latent future 和 预测 latent future 为 JSON
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F_nn
from torch_geometric.data import HeteroData

import mne
mne.set_log_level("WARNING")

from train.hetero_trainer import DynamicHeteroTrainer

# 复用 main_v4.py 的工具函数
from utils.utils import plot_recon_vs_target  # 这里暂时不使用，只是说明可以用
# Deprecated: from stim_align import batch_generate_stim
# from node_generator import generate_nodes_all_regions  # Unused - graph built via build_hetero_graph
# from edge_computer import generate_edges_with_dti_fallback  # Module does not exist
from mapper.atlas_mapper import BrainAtlas
from mapper.bids_mapper import BIDSMapper
from mapper.eeg_mapper import EEGMapper
from mapper.multi_modal_mapper import MultiModalMapper

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

logger = logging.getLogger("latent_export")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ======== 工具函数：latent 预测 & JSON 导出（B 方案） ========

def _assert_tensor(t, name: str):
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(t)}")


def _ensure_ndim(t: torch.Tensor, nd: int, name: str):
    if t.ndim != nd:
        raise ValueError(f"{name} expected ndim={nd}, got shape={tuple(t.shape)}")


def _latent_norm_reduce(z: torch.Tensor) -> torch.Tensor:
    """
    把 latent z (N, T, H) 用 L2 范数压成 (N, T) 标量，用于可视化。
    """
    _assert_tensor(z, "z")
    _ensure_ndim(z, 3, "z")
    return torch.norm(z, dim=-1)  # (N, T)


def _compute_latent_metrics(z_pred: torch.Tensor,
                            z_real: torch.Tensor,
                            eps: float = 1e-8) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    在 latent 空间上比较预测与真实:
    - z_pred, z_real: (N, T, H)
    返回:
    - global_metrics: {'mse', 'mae', 'rel_l2', 'acc_varnorm', 'pearson'}
    - per_node_metrics: node_id -> 同样字段的字典
    """
    _assert_tensor(z_pred, "z_pred")
    _assert_tensor(z_real, "z_real")
    if z_pred.shape != z_real.shape:
        raise ValueError(f"z_pred and z_real must have same shape; got {tuple(z_pred.shape)} vs {tuple(z_real.shape)}")

    N, T, H = z_pred.shape
    z_pred_np = z_pred.detach().cpu().numpy()
    z_real_np = z_real.detach().cpu().numpy()

    per_node: Dict[str, Dict[str, float]] = {}
    mse_list, mae_list, rel_list, acc_list, pearson_list = [], [], [], [], []

    for i in range(N):
        p = z_pred_np[i].reshape(-1)  # (T*H,)
        r = z_real_np[i].reshape(-1)  # (T*H,)

        diff = p - r
        mse = float((diff ** 2).mean())
        mae = float(np.abs(diff).mean())
        num = float(np.linalg.norm(diff))
        den = float(np.linalg.norm(r) + eps)
        rel = num / den

        var_r = float(np.var(r))
        acc = 1.0 - mse / (var_r + eps)  # variance-normalized "accuracy"

        # Pearson
        if np.allclose(r, r[0]) or np.allclose(p, p[0]):
            pearson = 0.0
        else:
            corr_mat = np.corrcoef(p, r)
            pearson_val = corr_mat[0, 1]
            pearson = float(0.0 if np.isnan(corr_mat[0, 1]) else corr_mat[0, 1])

        per_node[str(i)] = {
            "mse": mse,
            "mae": mae,
            "rel_l2": rel,
            "acc_varnorm": acc,
            "pearson": pearson
        }

        mse_list.append(mse)
        mae_list.append(mae)
        rel_list.append(rel)
        acc_list.append(acc)
        pearson_list.append(pearson)

    global_metrics = {
        "mse": float(np.mean(mse_list)),
        "mae": float(np.mean(mae_list)),
        "rel_l2": float(np.mean(rel_list)),
        "acc_varnorm": float(np.mean(acc_list)),
        "pearson": float(np.mean(pearson_list)),
    }
    return global_metrics, per_node


def _tensor_to_unity_json_scalar(t: torch.Tensor,
                                 start_time: float,
                                 dt: float,
                                 node_type: str,
                                 feature_name: str = "latent_norm",
                                 node_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    把 (N, T) 标量 tensor 转成 Unity 友好的 JSON 结构.
    """
    _assert_tensor(t, "scalar tensor")
    _ensure_ndim(t, 2, "scalar tensor")
    t_cpu = t.detach().cpu()
    N, T = t_cpu.shape

    if node_ids is None:
        node_ids = list(range(N))
    if len(node_ids) != N:
        raise ValueError("len(node_ids) must equal N")

    timestamps = [start_time + i * dt for i in range(T)]
    data_dict = {str(node_ids[i]): t_cpu[i].tolist() for i in range(N)}

    json_obj = {
        "meta": {
            "node_type": node_type,
            "node_count": N,
            "time_steps": T,
            "feature_name": feature_name,
            "start_time": start_time,
            "dt": dt
        },
        "timestamps": timestamps,
        "data": data_dict
    }
    return json_obj


def run_single_latent_prediction_and_export(
    trainer: DynamicHeteroTrainer,
    data: HeteroData,
    node_type: str,
    context_len: int,
    predict_len: int,
    dt: float,
    start_time: float,
    json_real_path: str,
    json_pred_path: str,
    json_metrics_path: str
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    使用现有的 trainer/model，在 latent 空间上做一次预测并导出 JSON（B 方案）。
    """

    device = trainer.device if hasattr(trainer, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = trainer.model.to(device)
    model.eval()

    # 1) graph_encoder
    data = data.to(device)
    encoder_out = trainer.graph_encoder(data)
    if not (isinstance(encoder_out, (list, tuple)) and len(encoder_out) >= 5):
        raise RuntimeError("graph_encoder must return at least 5-tuple (x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict)")
    x_dict, num_nodes_dict, stats_dict, x_raw_map, edge_index_dict = encoder_out[:5]

    # 2) model forward - FIXED: pass the encoded_dict, num_nodes_dict, and stats_dict
    outputs = model(
        data,
        encoded_dict=x_dict,           # Add this
        num_nodes_dict=num_nodes_dict, # Add this
        stats_dict=stats_dict,         # Add this
        edge_index_dict=edge_index_dict
    )
    if not (isinstance(outputs, (list, tuple)) and len(outputs) == 8):
        raise RuntimeError("model.forward must return 8-tuple (z_dict, gru_out, proj_seq_dict, recon_seq_denorm, recon_seq_scaled, global_seq, recon_feature_dict, recon_denorm_dict)")
    z_dict, gru_out, proj_seq_dict, recon_seq_denorm, recon_seq_scaled, global_seq, recon_feature_dict, recon_denorm_dict = outputs

    # ... rest of the function remains the same

    # 3) 取 latent 序列
    if not isinstance(proj_seq_dict, dict):
        raise TypeError("proj_seq_dict must be a dict[node_type] -> Tensor")
    if node_type not in proj_seq_dict:
        raise RuntimeError(f"proj_seq_dict missing node type '{node_type}'")
    proj_seq = proj_seq_dict[node_type]  # (N, T_proj, H)
    _assert_tensor(proj_seq, "proj_seq")
    _ensure_ndim(proj_seq, 3, "proj_seq")
    N, T_proj, H = proj_seq.shape

    if T_proj < context_len + predict_len:
        raise ValueError(f"T_proj={T_proj} too small for context_len={context_len} + predict_len={predict_len}")

    # 4) latent 预测
    loss_t, pred_feat, pred_denorm = trainer._temporal_prediction_loss(
        proj_seq,
        node_type,
        context_len=context_len,
        predict_len=predict_len,
        teacher_forcing_ratio=0.0,  # 纯自回归预测
        return_preds=True,
        stats_nt=None  # 只看 latent，不用 denorm
    )

    if pred_feat is None or not isinstance(pred_feat, torch.Tensor):
        raise RuntimeError("_temporal_prediction_loss did not return pred_feat as Tensor")
    _ensure_ndim(pred_feat, 3, "pred_feat")
    Np, Tp, Hp = pred_feat.shape
    if Np != N or Tp != predict_len or Hp != H:
        raise ValueError(f"pred_feat shape mismatch: expected (N={N}, T={predict_len}, H={H}), got {tuple(pred_feat.shape)}")

    # 真实 latent future
    z_real = proj_seq[:, context_len:context_len+predict_len, :]  # (N, predict_len, H)
    if z_real.shape != pred_feat.shape:
        raise ValueError(f"z_real shape {tuple(z_real.shape)} != pred_feat shape {tuple(pred_feat.shape)}")

    # 5) latent 指标
    global_metrics, per_node_metrics = _compute_latent_metrics(pred_feat, z_real)

    # 6) 压缩为标量并导出 JSON
    z_real_scalar = _latent_norm_reduce(z_real)    # (N, T_pred)
    z_pred_scalar = _latent_norm_reduce(pred_feat) # (N, T_pred)

    start_time_pred = start_time + context_len * dt

    json_real = _tensor_to_unity_json_scalar(
        z_real_scalar,
        start_time=start_time_pred,
        dt=dt,
        node_type=node_type,
        feature_name="latent_norm_real",
    )
    json_pred = _tensor_to_unity_json_scalar(
        z_pred_scalar,
        start_time=start_time_pred,
        dt=dt,
        node_type=node_type,
        feature_name="latent_norm_pred",
    )

    os.makedirs(os.path.dirname(json_real_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(json_pred_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(json_metrics_path) or ".", exist_ok=True)

    with open(json_real_path, "w") as f:
        json.dump(json_real, f)
    with open(json_pred_path, "w") as f:
        json.dump(json_pred, f)
    with open(json_metrics_path, "w") as f:
        json.dump({
            "global": global_metrics,
            "per_node": per_node_metrics
        }, f)

    return global_metrics, per_node_metrics


# ==================== 主流程：参考 main_v4.py ====================

def main():
    BASE_DIR = Path(__file__).parent / "test_file3"
    SUBJECTS = [d for d in BASE_DIR.glob("sub-*") if d.is_dir()]
    atlas_path = BASE_DIR.parent / "schaefer200_mask_ready.json"
    atlas = load_atlas(atlas_path)

    # 这里只做单被试/单 run 的导出示例；你可以按需要遍历 SUBJECTS
    for subj in SUBJECTS:
        logging.info(f"\n=== [LATENT EXPORT] Processing {subj.name} ===")
        result_dir = subj / "results"
        os.makedirs(result_dir, exist_ok=True)

        PATHS = {
            "eeg_dir": subj / "eeg",
            "func_dir": subj / "func",
            "dti_npy": subj / "dwi" / f"{subj.name}_acq-AP_dwi_connectome.npy",
            "nodes_json": result_dir / "nodes.json",
            "hetero_model": result_dir / "hetero_gnn_trained.pt",
            "latent_export_dir": result_dir / "latent_export"
        }
        for p in PATHS.values():
            os.makedirs(Path(p).parent, exist_ok=True)

        eeg_tasks = discover_eeg_tasks(PATHS["eeg_dir"])
        fmri_tasks = discover_fmri_tasks(PATHS["func_dir"])
        logging.info(f"[Discover] EEG tasks: {eeg_tasks}")
        logging.info(f"[Discover] fMRI tasks: {fmri_tasks}")

        eeg_data_cache = result_dir / "eeg_data.pt"
        hetero_graphs_cache = result_dir / "hetero_graphs_for_training.pt"

        # === 和 main_v4 一样的缓存逻辑 ===
        if eeg_data_cache.exists() and hetero_graphs_cache.exists():
            logging.info("[FULL CACHE HIT] Loading cached preprocessed data")
            stim = torch.load(stim_cache, map_location="cpu", weights_only=False)
            hetero_graphs = torch.load(hetero_graphs_cache, map_location="cpu", weights_only=False)
        else:
            logging.info("[CACHE MISS] running preprocessing + graph building")
            # Deprecated: stim = batch_generate_stim(subj)
            stim = None  # stim functionality has been deprecated
            # Removed: torch.save(stim, stim_cache)

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

            # Build heterogeneous graphs without stim (deprecated functionality)
            hetero_graphs = build_hetero_graph(fmri_data, eeg_data, stim_dict=None)
            torch.save(hetero_graphs, hetero_graphs_cache)

        # === Trainer 初始化（参考 main_v4 的设置），但不跑 train ===
        try:
            trainer = DynamicHeteroTrainer(
                hetero_data=hetero_graphs,
                hidden_dim=128,
                num_epochs=1,  # 这里不训练，只用来 forward
                recon_weight=1.0,
                recon_norm_weight=3.0,
                recon_corr_weight=2.0,
                recon_feat_var_weight=0.02,
                feature_lr_mul=12.0,
                scale_lr_mul=10.0,
                warmup_epochs=0,   # 不做 warmup
                freeze_scale_during_warmup=False,
                debug=False,
                batch_rescale_fn=None,
                batch_rescale_cfg={"enable": False},
                spec_loss_weight=0.5,
                spec_kernel_size=11
            )
        except Exception as e:
            logging.exception(f"[LatentExport] Trainer initialization failed: {e}")
            with open(result_dir / "hetero_graphs_debug.json", "w") as fh:
                json.dump({k: len(v) if v is not None else None for k, v in hetero_graphs.items()}, fh, indent=2)
            continue

        # 如果你有已经训练好的 checkpoint，可以在这里加载：
        ckpt_path = result_dir / "hetero_gnn_trained.pt"  # main_v4 最终保存的路径
        if ckpt_path.exists():
            try:
                trainer.load_model(ckpt_path)
                logging.info(f"[LatentExport] Loaded trained model from {ckpt_path}")
            except Exception as e:
                logging.warning(f"[LatentExport] Failed to load checkpoint {ckpt_path}: {e}")
        else:
            logging.warning(f"[LatentExport] Checkpoint {ckpt_path} not found, using current (possibly untrained) weights.")

        # 选第一张图用于导出
        if isinstance(trainer.data_list, (list, tuple)) and len(trainer.data_list) > 0:
            data = trainer.data_list[0]
        else:
            logging.error("[LatentExport] trainer.data_list is empty.")
            continue

        export_dir = PATHS["latent_export_dir"]
        os.makedirs(export_dir, exist_ok=True)

        # === B 方案 latent 导出 ===
        try:
            global_metrics, per_node_metrics = run_single_latent_prediction_and_export(
                trainer=trainer,
                data=data,
                node_type="fmri",  # 如你的 node_type 不同，改这里
                context_len=40,    # 你自己的 context/predict 配置
                predict_len=10,
                dt=1.0,
                start_time=0.0,
                json_real_path=str(export_dir / "fmri_latent_real.json"),
                json_pred_path=str(export_dir / "fmri_latent_pred.json"),
                json_metrics_path=str(export_dir / "fmri_latent_metrics.json"),
            )
            logging.info(f"[LatentExport] Global latent metrics for {subj.name}: {global_metrics}")
        except Exception as e:
            logging.exception(f"[LatentExport] Failed during latent prediction/export for {subj.name}: {e}")
            continue


if __name__ == "__main__":
    main()