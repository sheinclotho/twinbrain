import os
import json
import logging
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional

# import glob  # Unused - using Path.glob() instead
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# Removed duplicate: from torch_geometric.data import HeteroData

import mne
mne.set_log_level("WARNING")
import torch.nn as nn
import torch.nn.functional as F
from meta_node import MetaNode
from node_generator import generate_nodes_all_regions
from mapper.atlas_mapper import BrainAtlas
# Optional mapper imports - kept for reference:
# from mapper.bids_mapper import BIDSMapper
# from mapper.eeg_mapper import EEGMapper
from mapper.multi_modal_mapper import MultiModalMapper

# =============================
# 自动扫描 EEG/fMRI 任务
# =============================
def discover_eeg_tasks(eeg_dir: Path):
    eeg_dir = Path(eeg_dir)
    files = list(eeg_dir.glob("sub-*_task-*_run-*_eeg.*"))
    tasks = set()
    for f in files:
        m = re.search(r"task-([A-Za-z]+)", f.name)
        if m:
            token = m.group(1)
            token_up = token.upper()
            if token_up.endswith("ON"):
                base = token_up[:-2]
            elif token_up.endswith("OFF"):
                base = token_up[:-3]
            else:
                base = token_up
            tasks.add(base)
    return sorted(list(tasks))


def discover_fmri_tasks(func_dir: Path):
    func_dir = Path(func_dir)
    files = list(func_dir.glob("sub-*_task-*_run-*_bold.nii*"))
    tasks = set()
    for f in files:
        m = re.search(r"task-([A-Za-z0-9]+)", f.name)
        if m:
            token = m.group(1)
            tasks.add(token)
    return sorted(list(tasks))

# =============================
# 数据加载
# =============================
def load_atlas(atlas_path: Path) -> BrainAtlas:
    atlas = BrainAtlas(atlas_path)
    logging.info(f"[Atlas] Loaded: {len(atlas.regions)} regions")
    return atlas

def load_fmri(func_dir, tasks, atlas_file=None, label_file=None, brain_atlas=None, output_root=None):
    fmri_data = {}
    func_dir = Path(func_dir)
    if not func_dir.exists():
        logging.error(f"[load_fmri] func_dir does not exist: {func_dir}")
        return fmri_data
    logging.info(f"[load_fmri] Searching fMRI data under: {func_dir.resolve()}")
    logging.info(f"[load_fmri] Target tasks: {tasks}")
    for t in tasks:
        logging.info(f"\n[Task:{t}] ---------- 开始处理 ----------")
        task_dir = func_dir / t
        nii_files = []
        if task_dir.exists() and task_dir.is_dir():
            nii_files = list(task_dir.glob("*.nii*"))
            if nii_files:
                logging.debug(f"[Task:{t}] Found in subdir: {task_dir}")
        else:
            logging.debug(f"[Task:{t}] Subdir not found, trying func_dir root.")
        if not nii_files:
            nii_files = [p for p in func_dir.glob("*.nii*") if f"task-{t}" in p.name]
            if nii_files:
                logging.debug(f"[Task:{t}] Found in root via task-{t} pattern")
        if not nii_files:
            logging.warning(f"[Task:{t}] No NIfTI file found in {func_dir} or {task_dir}")
            continue
        fmri_file = nii_files[0]
        logging.info(f"[Task:{t}] Selected NIfTI: {fmri_file.name}")
        if output_root:
            output_dir = Path(output_root) / t
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.debug(f"[Task:{t}] Output directory: {output_dir}")
        else:
            output_dir = None
        try:
            # Initialize BIDSMapper
            mapper = BIDSMapper(
                atlas_name="schaefer",
                atlas_file=str(atlas_file) if atlas_file else None,
                label_file=str(label_file) if label_file else None,
                func_dir=str(func_dir),
                task_name=t
            )
            logging.debug(f"[Task:{t}] BIDSMapper initialized.")
        except Exception as e:
            logging.exception(f"[Task:{t}] Failed to initialize BIDSMapper: {e}")
            continue
        try:
            # Load and preprocess fMRI data
            mapper.load_and_preprocess(fmri_file=fmri_file)
            logging.info(f"[Task:{t}] fMRI preprocessing done, data shape={mapper.data.shape}")
        except Exception as e:
            logging.exception(f"[Task:{t}] Error during preprocessing: {e}")
            continue
        if brain_atlas is not None:
            try:
                # Attach brain atlas and generate labels
                mapper.attach_brain_atlas(brain_atlas)
                logging.info(f"[Task:{t}] Attached brain atlas metadata successfully. Labels generated.")
            except Exception as e:
                logging.warning(f"[Task:{t}] Brain atlas attachment failed: {e}")

        # 唯一修改：返回 HeteroData，不是 BIDSMapper
        fmri_data[t] = mapper.to_pyg(node_type="fmri")

        logging.info(f"[Task:{t}] Task completed successfully.")
        logging.debug(f"[Task:{t}] Node count={len(mapper.labels)}, Active nodes={int(mapper.node_mask.sum()) if mapper.node_mask is not None else 'N/A'}")
    logging.info(f"\n[load_fmri] All tasks processed. Loaded {len(fmri_data)} successful mappings.")
    return fmri_data

def load_eeg(eeg_dir, brain_atlas=None, output_root=None):
    """
    返回结构:
    {
        task_name: {
            "on":  HeteroData({'eeg': {...}}),
            "off": HeteroData({'eeg': {...}})
        }
    }
    """
    from torch_geometric.data import HeteroData
    eeg_dir = Path(eeg_dir)
    mapper = EEGMapper()
    all_tasks = mapper.load_task(task_name=None, merge=False, eeg_dir=str(eeg_dir))

    eeg_data = {}

    for t, pair in all_tasks.items():
        logging.info(f"[EEG] Preparing task: {t}")

        out_pair = {}

        for state in ("on", "off"):
            em = pair[state]

            # --------------------------------------------------------
            # 1) 获取 pyg（可能是 Data 或 HeteroData）
            # --------------------------------------------------------
            pyg_data = em.to_pyg()

            # --------------------------------------------------------
            # 2) 若是 Data → 包装成标准 HeteroData('eeg')
            # --------------------------------------------------------
            if isinstance(pyg_data, HeteroData):
                hd = pyg_data
            else:
                d = pyg_data
                hd = HeteroData()

                # 节点: eeg
                hd["eeg"].x = getattr(d, "x", None)
                hd["eeg"].x_seq = getattr(d, "x_seq", None)
                if hasattr(d, "pos"):
                    hd["eeg"].pos = d.pos
                if hasattr(d, "node_mask"):
                    hd["eeg"].node_mask = d.node_mask
                if hasattr(d, "region_ids"):
                    hd["eeg"].region_ids = d.region_ids

                # 边: eeg -> eeg
                if hasattr(d, "edge_index"):
                    hd["eeg", "connects", "eeg"].edge_index = d.edge_index
                if hasattr(d, "edge_attr"):
                    hd["eeg", "connects", "eeg"].edge_attr = d.edge_attr

            # --------------------------------------------------------
            # 3) edge_index 若缺失 → 自动生成
            # --------------------------------------------------------
            if ("eeg", "connects", "eeg") not in hd.edge_types:
                # 生成新的 edge_index
                temp = EEGMapper()
                temp.raw = em.raw if hasattr(em, "raw") else None
                temp.epochs = em.epochs if hasattr(em, "epochs") else None
                temp.compute_functional_connectivity()
                new_ei = temp.get_edge_index(method="threshold", k=5)
                hd["eeg", "connects", "eeg"].edge_index = torch.tensor(new_ei, dtype=torch.long)

                # edge_attr
                if temp.fc_matrix is not None:
                    ei = hd["eeg", "connects", "eeg"].edge_index.cpu().numpy()
                    w = temp.fc_matrix[ei[0], ei[1]]
                    hd["eeg", "connects", "eeg"].edge_attr = torch.tensor(w, dtype=torch.float32)

            # --------------------------------------------------------
            # 4) region_ids 若缺失 → 自动从 atlas 映射
            # --------------------------------------------------------
            if brain_atlas is not None:
                if not hasattr(hd["eeg"], "region_ids") or hd["eeg"].region_ids is None:
                    temp = EEGMapper()
                    temp.channel_names = em.channel_names if hasattr(em, "channel_names") else None
                    temp.add_region_mapping(brain_atlas, cache=True)
                    hd["eeg"].region_ids = torch.tensor(temp.region_ids, dtype=torch.long)

            # --------------------------------------------------------
            # 5) 基础检查
            # --------------------------------------------------------
            assert "eeg" in hd.node_types, f"[EEG] {t}/{state}: HeteroData 缺少 'eeg' 节点类型"
            assert hasattr(hd["eeg"], "x_seq"), f"[EEG] {t}/{state}: 缺少 x_seq"
            assert hd["eeg"].x_seq.dim() == 3, f"[EEG] {t}/{state}: x_seq 必须是 3D (N,T,F)"

            out_pair[state] = hd

        eeg_data[t] = out_pair
        logging.info(f"[EEG] Loaded task {t}")

    logging.info(f"[EEG] Total tasks: {len(eeg_data)} -> {list(eeg_data.keys())}")
    return eeg_data

def load_dti(dti_path: Path) -> np.ndarray | None:
    if not dti_path.exists():
        logging.warning(f"[DTI] File not found: {dti_path}")
        return None
    dti = np.load(dti_path)
    logging.info(f"[DTI] Loaded {dti_path}, shape={dti.shape}")
    return dti

# =============================
# 节点构建
# =============================
def build_nodes(atlas) -> list[dict]:
    meta = MetaNode("base_node", [0, 0, 0], "general", 0.8)
    meta.add_feature("volume", 100.0)
    nodes = generate_nodes_all_regions(
        meta=meta,
        atlas=atlas,
        nodes_per_region=1,
        noise_std=1.0,
        atlas_img_path="Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii",
        mode="mask"
    )
    logging.info(f"[Nodes] Generated {len(nodes)} nodes")
    return nodes

def save_nodes_json(nodes, path: Path):
    formatted = []
    for i, n in enumerate(nodes):
        formatted.append({
            "idx": i,
            "node_id": n.get("node_id", f"node_{i}"),
            "region_id": n.get("region_id", f"region_{i}"),
            "position_3d": n["position_3d"].tolist() if isinstance(n["position_3d"], np.ndarray) else n["position_3d"],
            "position": n["position"].tolist() if isinstance(n["position"], np.ndarray) else n["position"],
        })
    with open(path, "w") as f:
        json.dump(formatted, f, indent=2)
    logging.info(f"[Nodes] Saved to {path}")

# =============================
# 多模态图构建
# =============================
def build_hetero_graph(
    fmri_data: Dict[str, HeteroData],
    eeg_data: Dict[str, Dict[str, HeteroData]],
    stim_dict: Optional[Dict[str, torch.Tensor]] = None,
    save_input_path: str = "graph_input_check.json",
    max_T: Optional[int] = 2000,  # ← 新增：强制截断时间步
    debug: bool = True,
) -> Dict[str, List[HeteroData]]:
    """
    构建异构图，自动截断时间步，避免 edge_index 越界。

    Args:
        fmri_data[t]: HeteroData (node: 'fmri')
        eeg_data[t]: {"on": HeteroData, "off": HeteroData} (node: 'eeg')
        stim_dict[t]: torch.Tensor [T_stim]
        max_T: 强制最大时间步（必须 ≤ spatial_T * N 在 trainer 中）
        debug: 是否打印详细建图信息
    """
    hetero_graphs = {}
    input_summary = {}

    # ====================== 1. 输入检查 & 摘要 ======================
    for t, g in fmri_data.items():
        if isinstance(g, HeteroData):
            input_summary.setdefault(t, {})["fmri"] = {
                "num_nodes": {nt: g[nt].num_nodes for nt in g.node_types},
                "x_seq_shape": getattr(g["fmri"], "x_seq", None).shape if hasattr(g["fmri"], "x_seq") else None,
            }

    for t, pair in eeg_data.items():
        input_summary.setdefault(t, {})["eeg"] = {}
        for key in ["on", "off"]:
            g = pair.get(key)
            if isinstance(g, HeteroData):
                input_summary[t]["eeg"][key] = {
                    "num_nodes": {nt: g[nt].num_nodes for nt in g.node_types},
                    "x_seq_shape": getattr(g["eeg"], "x_seq", None).shape if hasattr(g["eeg"], "x_seq") else None,
                }

    # 保存摘要
    os.makedirs(os.path.dirname(save_input_path) or ".", exist_ok=True)
    try:
        with open(save_input_path, "w", encoding="utf-8") as f:
            json.dump(input_summary, f, indent=2, default=str)
        logging.info(f"[Graph] Input summary saved to {save_input_path}")
    except Exception as e:
        logging.error(f"[Graph] Failed to save input summary: {e}")

    # ====================== 2. 共同任务 ======================
    common_tasks = set(fmri_data.keys()) & set(eeg_data.keys())
    logging.info(f"[Graph] Common tasks: {sorted(common_tasks)}")

    if not common_tasks:
        logging.error("[Graph] No common tasks found!")
        return {}

    # ====================== 3. 构建图 ======================
    for t in sorted(common_tasks):
        fmri_graph = fmri_data[t]
        pair = eeg_data[t]
        on_graph = pair.get("on")
        off_graph = pair.get("off")

        # --- 严格类型检查 ---
        if not isinstance(fmri_graph, HeteroData):
            logging.error(f"[Graph:{t}] fmri is not HeteroData.")
            continue
        if not isinstance(on_graph, HeteroData) or not isinstance(off_graph, HeteroData):
            logging.error(f"[Graph:{t}] eeg on/off is not HeteroData.")
            continue
        if "fmri" not in fmri_graph.node_types:
            logging.error(f"[Graph:{t}] missing 'fmri' node")
            continue
        if "eeg" not in on_graph.node_types or "eeg" not in off_graph.node_types:
            logging.error(f"[Graph:{t}] missing 'eeg' node")
            continue

        # --- 获取 x_seq ---
        fmri_seq = getattr(fmri_graph["fmri"], "x_seq", None)
        eeg_on_seq = getattr(on_graph["eeg"], "x_seq", None)
        eeg_off_seq = getattr(off_graph["eeg"], "x_seq", None)

        if fmri_seq is None or eeg_on_seq is None or eeg_off_seq is None:
            logging.error(f"[Graph:{t}] missing x_seq")
            continue

        # --- 确定 T_cut ---
        T_fmri = fmri_seq.shape[1]
        T_eeg = eeg_on_seq.shape[1]
        T_stim = stim_dict[t].shape[0] if stim_dict and t in stim_dict else float('inf')
        T = min(T_fmri, T_eeg, T_stim)
        if max_T is not None:
            T = min(T, max_T)
            logging.info(f"[Graph:{t}] T_cut = {T} (max_T={max_T})")

        # --- 截断序列 ---
        fmri_seq = fmri_seq[:, :T, :].clone()
        eeg_on_seq = eeg_on_seq[:, :T, :].clone()
        eeg_off_seq = eeg_off_seq[:, :T, :].clone()

        # --- 安全处理 stim ---
        stim_t = stim_dict[t][:T].clone() if stim_dict and t in stim_dict else None

        # --- 防 NaN ---
        fmri_seq = torch.nan_to_num(fmri_seq, nan=0.0)
        eeg_on_seq = torch.nan_to_num(eeg_on_seq, nan=0.0)
        eeg_off_seq = torch.nan_to_num(eeg_off_seq, nan=0.0)

        # --- 临时注入 x_seq ---
        fmri_graph["fmri"].x_seq = fmri_seq
        on_graph["eeg"].x_seq = eeg_on_seq
        off_graph["eeg"].x_seq = eeg_off_seq

        # --- 构建 ---
        mm = MultiModalMapper(verbose=debug)
        try:
            graphs_list = mm.build_dynamic_from_graphs(
                on_graph=on_graph,
                off_graph=off_graph,
                fmri_graph=fmri_graph,
                stim_dict={t: stim_t} if stim_t is not None else None,
                max_T=384
            )

            if not graphs_list:
                logging.warning(f"[Graph:{t}] build_dynamic returned empty list")
                continue

            # --- Debug: 打印每个 graph 的关键信息 ---
            if debug:
                for i, g in enumerate(graphs_list):
                    info = {
                        "task": t,
                        "idx": i,
                        "node_types": g.node_types,
                        "edge_types": [str(et) for et in g.edge_types],
                    }
                    if hasattr(g, 'x_seq_dict'):
                        info["x_seq_shapes"] = {nt: x.shape for nt, x in g.x_seq_dict.items()}
                    for et in g.edge_types:
                        edge = g[et].edge_index
                        if edge.numel() > 0:
                            info[f"edge_{et}_max"] = edge.max().item()
                            info[f"edge_{et}_min"] = edge.min().item()
                    logging.info(f"[Graph:{t}] graph[{i}] {info}")

            hetero_graphs[t] = graphs_list
            logging.info(f"[Graph:{t}] Built {len(graphs_list)} graphs | T={T}")

        except Exception as e:
            logging.error(f"[Graph:{t}] build_dynamic FAILED: {e}")
            import traceback
            traceback.print_exc()

    logging.info(f"[Graph] Final tasks: {list(hetero_graphs.keys())}")
    return hetero_graphs