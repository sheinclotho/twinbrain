#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数字孪生脑主函数（批量多任务版，结构化路径，支持新版BIDSMapper）
一次性整合：自动发现任务、fMRI/EEG 匹配、region 对齐、结果集中到 results/
增强点：
- 对齐模式使用 'latent'（潜在空间对齐）
- 在计算 mean_diff/cov_diff 前做 safe_time_align_for_stats()
- 加强 debug 日志与空值保护
"""

import os
import json
import glob
import logging
import re
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData

import mne
mne.set_log_level("WARNING")
import torch
import torch.nn as nn
import torch.nn.functional as F
from stim_align import batch_generate_stim
from meta_node import MetaNode
from node_generator import generate_nodes_all_regions
from edge_computer import generate_edges_with_dti_fallback
from mapper.atlas_mapper import BrainAtlas
from mapper.bids_mapper import BIDSMapper
from mapper.eeg_mapper import EEGMapper
from mapper.multi_modal_mapper import MultiModalMapper
from train.hetero_trainer import DynamicHeteroTrainer  # 无监督版

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
if not logger.handlers:  # 防止重复添加
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =============================
# 主流程
# =============================
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
            "png_graph": result_dir / "png_graph.png",
            "png_activations": result_dir / "png_activations.png",
            "png_connectome": result_dir / "png_connectome.png",
            "html_connectome": result_dir / "brain_connectome.html",
        }
        for p in PATHS.values():
            os.makedirs(Path(p).parent, exist_ok=True)

        # ========== 任务发现 ==========
        eeg_tasks = discover_eeg_tasks(PATHS["eeg_dir"])
        fmri_tasks = discover_fmri_tasks(PATHS["func_dir"])
        logging.info(f"[Discover] EEG tasks: {eeg_tasks}")
        logging.info(f"[Discover] fMRI tasks: {fmri_tasks}")

        # ========== stim + eeg_data + 图构建 ==========
        stim_cache = result_dir / "stim.pt"
        eeg_data_cache = result_dir / "eeg_data.pt"
        hetero_graphs_cache = result_dir / "hetero_graphs_for_training.pt"

        if stim_cache.exists() and eeg_data_cache.exists() and hetero_graphs_cache.exists():
            logging.info(f"[FULL CACHE HIT] Loading cached stim + eeg_data + hetero_graphs → 直接进入训练！")
            stim = torch.load(stim_cache, map_location="cpu", weights_only=False)
            eeg_data = torch.load(eeg_data_cache, map_location="cpu", weights_only=False)
            hetero_graphs = torch.load(hetero_graphs_cache, map_location="cpu", weights_only=False)
        else:
            logging.info("[CACHE MISS] Running full preprocessing + graph building (only once!)")
            stim = batch_generate_stim(subj)
            torch.save(stim, stim_cache)
            logging.info(f"[Cache] stim saved to {stim_cache}")

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
            logging.info(f"[Cache] eeg_data saved to {eeg_data_cache}")

            dti = load_dti(PATHS["dti_npy"])
            nodes = build_nodes(atlas)
            save_nodes_json(nodes, PATHS["nodes_json"])

            hetero_graphs = build_hetero_graph(fmri_data, eeg_data, stim_dict=stim)
            torch.save(hetero_graphs, hetero_graphs_cache)
            logging.info(f"[Cache] hetero_graphs saved to {hetero_graphs_cache}")

        # ========== 模型训练与预测 ==========
        try:
            trainer = DynamicHeteroTrainer(
                hetero_data=hetero_graphs,
                hidden_dim=128,
                num_epochs=120,
                align_weight=2.0,
                temp_weight=0.5,
                max_T=2000,
                debug=True
            )
        except Exception as e:
            logging.exception(f"[Train] Trainer initialization failed: {e}")
            with open(result_dir / "hetero_graphs_debug.json", "w") as fh:
                json.dump({k: len(v) if v is not None else None for k, v in hetero_graphs.items()}, fh, indent=2)
            logging.error("[Train] Saved debug summary; skipping subject.")
            continue

        trainer.diagnostic_dir = str(result_dir / "diagnostics")
        os.makedirs(trainer.diagnostic_dir, exist_ok=True)

        # ========== 训练 ==========
        trainer.train()
        try:
            trainer.save_model(PATHS["hetero_model"])
        except Exception as e:
            logging.warning(f"[Train] Failed to save model: {e}")

        from train.embed_utils import save_embeddings
        # save embeddings for whole dataset (or subset)
        subject_ids_list = ["subj1", "subj2", "subj3", "subj4"]  # length == len(graphs)
        save_embeddings(trainer, trainer.data_list, "outputs/embeddings_run1_with_subj.npz",
                        agg="global", save_raw=False, subject_ids_override=subject_ids_list)

if __name__ == "__main__":
    main()