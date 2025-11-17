# mapper/multi_modal_mapper.py
import os
import json
import logging
from typing import List, Union, Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data

logger = logging.getLogger("MultiModalMapper")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


class MultiModalMapper:

    def __init__(
        self,
        fmri_mapper=None,
        eeg_mapper=None,
        target_dim: int = 64,
        align_dim: int = 64,
        align_mode: str = "latent",
        max_time: int = 2000,
        cross_modal_k: int = 5,
        cross_modal_thr: float = 50.0,
        verbose: bool = True,
    ):
        self.fmri_mapper = fmri_mapper
        self.eeg_mapper = eeg_mapper
        self.target_dim = int(target_dim)
        self.align_dim = int(align_dim)
        self.align_mode = align_mode
        self.max_time = int(max_time)
        self.cross_modal_k = int(cross_modal_k)
        self.cross_modal_thr = float(cross_modal_thr)
        self.verbose = bool(verbose)
        self._proj_layers: Dict[str, nn.Linear] = {}
        self.align_linear: Optional[nn.Linear] = None
        self.align_R: Optional[nn.Parameter] = None

    def _log(self, *args, level="info"):
        if not self.verbose:
            return
        getattr(logger, level)(" ".join(map(str, args)))

    @staticmethod
    def _to_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.tensor(np.asarray(x), dtype=torch.float32)

    @staticmethod
    def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.asarray(x), dtype=torch.float32)
        if x.dim() == 3:
            return x
        if x.dim() == 2:
            a, b = x.shape
            return x.unsqueeze(1) if b <= 20 else x.T.unsqueeze(-1)
        if x.dim() == 1:
            return x.reshape(1, 1, -1)
        raise ValueError(f"Unsupported input tensor ndim={x.dim()}")

    def _get_projection(self, in_dim: int, tag: str) -> nn.Linear:
        if tag not in self._proj_layers:
            layer = nn.Linear(in_dim, self.target_dim, bias=False)
            nn.init.xavier_uniform_(layer.weight)
            self._proj_layers[tag] = layer
            self._log(f"[Proj] created {tag}: {in_dim}->{self.target_dim}")
        return self._proj_layers[tag]

    def _project_and_agg(self, x: torch.Tensor, tag: str, agg_time: str = "mean") -> torch.Tensor:
        x = self._ensure_3d(x)
        N, T, F = x.shape
        if agg_time == "mean":
            feat = x.mean(dim=1)
        elif agg_time == "last":
            feat = x[:, -1, :]
        else:
            feat = x.mean(dim=1)
        if feat.shape[-1] == self.target_dim:
            return feat
        proj = self._get_projection(feat.shape[-1], tag)
        return proj(feat)

    @staticmethod
    def _sanitize_numeric(t: torch.Tensor) -> torch.Tensor:
        t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
        std = t.std(dim=0)
        if (std == 0).any():
            noise = (torch.randn_like(t) * 1e-6)
            t = t + noise
        return t

    def _init_align_params(self, q: int, device: torch.device):
        if self.align_linear is None or self.align_linear.in_features != q:
            self.align_linear = nn.Linear(q, q, bias=False).to(device)
            with torch.no_grad():
                self.align_linear.weight.copy_(torch.eye(q, device=device))
        if self.align_R is None or self.align_R.shape[0] != q:
            self.align_R = nn.Parameter(torch.eye(q, device=device))

    # ================================
    # 关键修复 1：支持 HeteroData["fmri"].times
    # ================================
    def _get_time_vector(self, hetero: HeteroData, node_type: str) -> np.ndarray:
        """
        从 HeteroData[node_type] 提取时间向量（强制简化版本）
        允许字段:
            node.times / node.time / node.timestamps / node.t_seq
            node.debug_info = {'n_tp': int, 'TR': float}
            或通过 node.x_seq 的形状推断
        """
        import numpy as np
        node = hetero[node_type]

        # ---- 1) 直接字段 ----
        for key in ("times", "time", "timestamps", "t_seq"):
            if hasattr(node, key):
                arr = getattr(node, key)
                if arr is not None:
                    return np.asarray(arr, dtype=np.float32).reshape(-1)

        # ---- 2) debug_info ----
        dbg = getattr(node, "debug_info", None)
        if isinstance(dbg, dict):
            if "n_tp" in dbg:
                n_tp = int(dbg["n_tp"])
                tr = float(dbg.get("TR", 1.0))
                return (np.arange(n_tp, dtype=np.float32) * tr)

        # ---- 3) 从 x_seq 推断 ----
        if hasattr(node, "x_seq") and node.x_seq is not None:
            xseq = node.x_seq
            # 格式固定为 (N, T, C)
            if xseq.ndim == 3:
                n_tp = xseq.shape[1]
                return (np.arange(n_tp, dtype=np.float32))

        raise ValueError(f"Cannot extract time from node '{node_type}'")

    def _validate_and_extract(self, hetero: HeteroData, node_type: str) -> Dict[str, Any]:
        """
        从 HeteroData[node_type] 提取 x_seq/x、边、坐标等标准字段。
        假设:
          - hetero 是 HeteroData
          - hetero[node_type] 至少有 x 或 x_seq
          - x_seq 形状 (N, T, C) 或 x (N, C)
        """
        import torch

        if not isinstance(hetero, HeteroData):
            raise ValueError("Input to _validate_and_extract must be HeteroData")

        if node_type not in hetero.node_types:
            raise ValueError(f"Node type '{node_type}' missing in HeteroData")

        node = hetero[node_type]
        out = {}

        # --- x_seq 或 x ---
        xseq = getattr(node, "x_seq", None)
        x = getattr(node, "x", None)

        if xseq is None and x is None:
            raise ValueError(f"{node_type}: missing both x_seq and x")

        # ---- 标准化为 3D ----
        if xseq is not None:
            t = self._to_tensor(xseq)
        else:
            t = self._to_tensor(x)
        t3 = self._ensure_3d(t)            # (N, T, C)
        t3 = self._sanitize_numeric(t3)    # 去 NaN / inf

        out["x_seq"] = t3
        out["x_mean"] = t3.mean(dim=1)     # (N, C)

        # --- edge_index ---
        eidx = getattr(node, "edge_index", None)
        if isinstance(eidx, torch.Tensor) and eidx.numel() > 0:
            out["edge_index"] = eidx.long()
        else:
            out["edge_index"] = torch.empty((2, 0), dtype=torch.long)

        # --- pos ---
        pos = getattr(node, "pos", None)
        out["pos"] = pos.float() if isinstance(pos, torch.Tensor) else None

        # --- fc_matrix (可选) ---
        out["fc_matrix"] = getattr(node, "fc_matrix", None)

        return out


    # ================================
    # 其余函数修复
    # ================================
    def _temporal_align_physical(
        self,
        fmri_x: torch.Tensor,
        eeg_x: torch.Tensor,
        fmri_graph: Any,  # HeteroData
        eeg_graph: Any   # Data
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = fmri_x.device
        t_fmri = torch.from_numpy(self._get_time_vector(fmri_graph, "fmri")).to(device)
        t_eeg = torch.from_numpy(self._get_time_vector(eeg_graph, "eeg")).to(device)

        t_min = max(t_fmri.min(), t_eeg.min())
        t_max = min(t_fmri.max(), t_eeg.max())
        if t_min >= t_max:
            raise ValueError(f"No time overlap: fMRI [{t_fmri.min():.1f}, {t_fmri.max():.1f}], EEG [{t_eeg.min():.1f}, {t_eeg.max():.1f}]")

        duration = t_max - t_min
        n_target = min(self.max_time, int(duration * 10) + 1)
        t_target = torch.linspace(t_min.item(), t_max.item(), steps=n_target, device=device)

        def interp_to_target(x: torch.Tensor, t_src: torch.Tensor):
            x_perm = x.permute(0, 2, 1)
            x_interp = F.interpolate(x_perm, size=n_target, mode="linear", align_corners=False)
            return x_interp.permute(0, 2, 1)

        fmri_interp = interp_to_target(fmri_x, t_fmri)
        eeg_interp = interp_to_target(eeg_x, t_eeg)
        self._log(f"[TemporalAlign] Aligned to [{t_min:.1f}, {t_max:.1f}]s, T={n_target}")
        return fmri_interp, eeg_interp

    def _merge_eeg_runs(self, on_graph: Any, off_graph: Any) -> Dict[str, Any]:
        on_info = self._validate_and_extract(on_graph, "eeg")
        off_info = self._validate_and_extract(off_graph, "eeg")
        x_on = on_info["x_seq"]
        x_off = off_info["x_seq"]
        if x_on.shape[0] != x_off.shape[0]:
            raise ValueError(f"EEG node count mismatch: {x_on.shape[0]} vs {x_off.shape[0]}")
        x_cat = torch.cat([x_on, x_off], dim=1)
        edge_index = on_info["edge_index"] if on_info["edge_index"].numel() > 0 else off_info["edge_index"]
        pos = on_info["pos"] if on_info["pos"] is not None else off_info["pos"]
        return {"x_seq": x_cat, "edge_index": edge_index, "pos": pos}

    def _add_cross_modal_edges(self, fmri_pos: torch.Tensor, eeg_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if fmri_pos is None or eeg_pos is None:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
        dist = torch.cdist(fmri_pos, eeg_pos)
        edge_index_list = []
        edge_attr_list = []
        for i in range(fmri_pos.shape[0]):
            d = dist[i]
            if self.cross_modal_k > 0:
                _, idx = torch.topk(d, k=min(self.cross_modal_k, d.shape[0]), largest=False)
            else:
                idx = torch.where(d <= self.cross_modal_thr)[0]
            if len(idx) == 0:
                continue
            src = torch.full((len(idx),), i, dtype=torch.long)
            dst = idx
            edge_index_list.append(torch.stack([src, dst]))
            edge_attr_list.append(1.0 / (d[idx] + 1e-6))
        if not edge_index_list:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list)
        return edge_index, edge_attr

    def build_dynamic_from_graphs(
        self,
        on_graph: Any,
        off_graph: Any,
        fmri_graph: Any,
        stim_dict: Optional[Dict[str, Any]] = None,
        max_T: Optional[int] = None,  # ← 新增参数！
    ) -> List[HeteroData]:
        """
        完全修复版 + 强制截断 max_T
        """
        def _ensure_seq(name, x):
            if x is None:
                raise ValueError(f"[Dynamic] {name}.x_seq is None")
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"[Dynamic] {name}.x_seq 必须为 torch.Tensor")
            if x.dim() != 3:
                raise ValueError(f"[Dynamic] {name}.x_seq 必须为 3D (N,T,F)，收到 {tuple(x.shape)}")
            self._log(f"[Dynamic] {name}.x_seq shape = {tuple(x.shape)}")
            return x

        # 1) 合并 EEG
        eeg_merged = self._merge_eeg_runs(on_graph, off_graph)
        eeg_x = _ensure_seq("EEG(raw)", eeg_merged["x_seq"])

        # 2) 提取 fMRI
        fmri_info = self._validate_and_extract(fmri_graph, "fmri")
        fmri_x = _ensure_seq("fMRI(raw)", fmri_info["x_seq"])

        # 3) 物理时间对齐
        self._log("[Align] 开始物理时间对齐（physical alignment）")
        fmri_al, eeg_al = self._temporal_align_physical(fmri_x, eeg_x, fmri_graph, on_graph)
        fmri_al = _ensure_seq("fMRI(aligned)", fmri_al)
        eeg_al = _ensure_seq("EEG(aligned)", eeg_al)

        # --- 关键修复：强制截断 max_T ---
        if max_T is not None:
            T_orig = fmri_al.shape[1]
            T = min(fmri_al.shape[1], eeg_al.shape[1])
            T = min(T, max_T)
            if T < T_orig:
                fmri_al = fmri_al[:, :T, :]
                eeg_al = eeg_al[:, :T, :]
                self._log(f"[Align] 强制截断 T={T_orig} → {T} (max_T={max_T})")

        # 4) 构建 eeg_info
        eeg_info = on_graph.clone()
        eeg_info["eeg"].x_seq = eeg_al
        eeg_info["eeg"].x = eeg_al.mean(dim=1)

        # 5) 更新 fmri_info
        fmri_info["x_seq"] = fmri_al

        # 6) 构建
        self._log("[Dynamic] 开始构建 HeteroData")
        combined = self._construct_hetero(fmri_info, eeg_info, on_graph)

        # 7) 刺激注入 + 截断
        if stim_dict:
            self._log("[Dynamic] 注入刺激时序 stim_dict")
            for ntype, stim in stim_dict.items():
                if ntype in combined and hasattr(combined[ntype], "x_seq"):
                    stim_t = torch.as_tensor(stim, dtype=torch.float32, device=combined[ntype].x_seq.device)
                    # 截断 stim
                    if max_T is not None and stim_t.shape[0] > max_T:
                        stim_t = stim_t[:max_T]
                        self._log(f"[Stim] 截断 stim[{ntype}] {stim_t.shape[0]} → {max_T}")
                    if stim_t.shape[0] == combined[ntype].x_seq.shape[1]:
                        combined[ntype].x_seq = combined[ntype].x_seq + stim_t
                        combined[ntype].x = combined[ntype].x_seq.mean(dim=1)

        return [combined]
    def _construct_hetero(self, fmri_info: Any, eeg_info: Any, on_graph: Any) -> HeteroData:
        from torch_geometric.data import HeteroData
        data = HeteroData()

        def _check_ntf(x: Any, name: str) -> torch.Tensor:
            if x is None:
                raise ValueError(f"{name} is None")
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32)
            if x.dim() != 3:
                raise ValueError(f"{name} must be 3D (N,T,F), got {tuple(x.shape)}")
            return x

        self._log("=" * 120)
        self._log("[HETERO BUILD] 进入 _construct_hetero")
        self._log(f"[INPUT] fmri_info keys = {list(fmri_info.keys())}")
        self._log(f"[INPUT] eeg_info type = {type(eeg_info)}")

        # ---------------------------------------------------------------------
        # fMRI node
        # ---------------------------------------------------------------------
        fmri_seq = _check_ntf(fmri_info["x_seq"], "fmri x_seq")
        data["fmri"].x_seq = fmri_seq.clone()
        data["fmri"].x = fmri_seq.mean(dim=1)
        data["fmri"].x_mean = data["fmri"].x

        if fmri_info.get("pos") is not None:
            data["fmri"].pos = fmri_info["pos"]
            self._log(f"[fMRI] pos shape = {tuple(fmri_info['pos'].shape)}")

        self._log(f"[fMRI] x_seq = {tuple(fmri_seq.shape)}  x = {tuple(data['fmri'].x.shape)}")

        # ---------------------------------------------------------------------
        # EEG node
        # ---------------------------------------------------------------------
        if isinstance(eeg_info, dict):
            eeg_seq = _check_ntf(eeg_info.get("x_seq"), "eeg x_seq")
            pos = eeg_info.get("pos")
        else:
            # HeteroData
            eeg_seq = _check_ntf(eeg_info["eeg"].x_seq, "eeg.eeg.x_seq")
            pos = eeg_info["eeg"].pos if hasattr(eeg_info["eeg"], "pos") else None

        data["eeg"].x_seq = eeg_seq.clone()
        data["eeg"].x = eeg_seq.mean(dim=1)
        data["eeg"].x_mean = data["eeg"].x
        self._log(f"[EEG] x_seq = {tuple(eeg_seq.shape)}  x = {tuple(data['eeg'].x.shape)}")

        if pos is not None:
            data["eeg"].pos = pos
            self._log(f"[EEG] pos shape = {tuple(pos.shape)}")

        # ---------------------------------------------------------------------
        # fMRI edges
        # ---------------------------------------------------------------------
        self._log("[fMRI] 尝试读取 edge_index")
        ei_f = fmri_info.get("edge_index")

        if ei_f is not None:
            self._log(f"[fMRI] 输入 edge_index shape = {tuple(ei_f.shape)}  num_edges={ei_f.shape[1]}")
        else:
            self._log("[fMRI] 输入 edge_index is None")

        if ei_f is not None and ei_f.numel() > 0:
            data[("fmri", "connects", "fmri")].edge_index = ei_f
            self._log("[fMRI] 写入 HeteroData -- edge_type ('fmri','connects','fmri')")
            self._log(f"[fMRI] 实际写入 edge_index = {tuple(ei_f.shape)}")

            if fmri_info.get("fc_matrix") is not None:
                src, dst = ei_f[0], ei_f[1]
                edge_attr = torch.tensor(
                    fmri_info["fc_matrix"][src, dst], dtype=torch.float32
                )
                data[("fmri", "connects", "fmri")].edge_attr = edge_attr
                self._log(f"[fMRI] 写入 edge_attr shape = {tuple(edge_attr.shape)}")
        else:
            self._log("[fMRI] 不写入 edge_index（为空）")

        # ---------------------------------------------------------------------
        # EEG edges
        # ---------------------------------------------------------------------
        # --- 原来的 EEG 边处理（替换为下面更稳健的实现） ---
        self._log("[EEG] 尝试读取 edge_index")

        ei_e = None
        # 1) 如果 eeg_info 是 HeteroData，优先从 edge_types 列表读取
        if hasattr(eeg_info, "edge_types"):
            if ("eeg", "connects", "eeg") in getattr(eeg_info, "edge_types", []):
                # 直接安全获取对象并检查
                try:
                    obj = eeg_info[("eeg", "connects", "eeg")]
                    if hasattr(obj, "edge_index") and obj.edge_index is not None and obj.edge_index.numel() > 0:
                        ei_e = obj.edge_index.clone()
                        ei_src = "eeg_info[('eeg','connects','eeg')]"
                        self._log(f"[EEG] Found eeg internal edges in eeg_info via edge_types: {tuple(ei_e.shape)}")
                except Exception as e:
                    self._log(f"[EEG] Failed to read eeg_info[('eeg','connects','eeg')]: {e}", level="warning")
            else:
                # Debug 输出，明确告诉你 edge_types 列表里到底有什么
                self._log(f"[EEG] eeg_info.edge_types does not contain ('eeg','connects','eeg'): {getattr(eeg_info, 'edge_types', None)}", level="debug")

        # 2) 尝试从可能的 dict-like eeg_merged 中拷贝（优先级高于 on_graph）
        if ei_e is None and isinstance(eeg_merged, dict):
            for key in [("eeg","connects","eeg"), "edge_index", "eeg_edge_index"]:
                if key in eeg_merged and eeg_merged[key] is not None:
                    cand = eeg_merged[key]
                    if hasattr(cand, "edge_index"):
                        cand = cand.edge_index
                    if isinstance(cand, np.ndarray):
                        cand = torch.from_numpy(cand).long()
                    if isinstance(cand, torch.Tensor) and cand.numel() > 0:
                        ei_e = cand.clone()
                        ei_src = f"eeg_merged[{key}]"
                        self._log(f"[EEG] Copied edge_index from eeg_merged key={key} -> {tuple(ei_e.shape)}")
                        break

        # 3) 尝试从 on_graph（HeteroData）中找到任何以 eeg->eeg 的 edge_type 并拷贝
        if ei_e is None and hasattr(on_graph, "edge_types"):
            for et in on_graph.edge_types:
                if et[0] == "eeg" and et[2] == "eeg":
                    obj = on_graph[et]
                    if hasattr(obj, "edge_index") and obj.edge_index is not None and obj.edge_index.numel() > 0:
                        ei_e = obj.edge_index.clone()
                        ei_src = f"on_graph edge_type={et}"
                        self._log(f"[EEG] Copied edge_index from on_graph {et} -> {tuple(ei_e.shape)}")
                        # copy edge_attr if present
                        attr = getattr(obj, "edge_attr", None)
                        if attr is not None:
                            tmp_attr = attr.clone()
                        else:
                            tmp_attr = None
                        break

        # 4) 最终注入到 data（如果找到）
        if ei_e is not None and ei_e.numel() > 0:
            data[("eeg", "connects", "eeg")].edge_index = ei_e
            # 优先使用已有的 tmp_attr（来自 on_graph），否则尝试从 on_graph.fc_matrix 生成
            if 'tmp_attr' in locals() and tmp_attr is not None:
                data[("eeg", "connects", "eeg")].edge_attr = tmp_attr
                self._log(f"[EEG] Wrote edge_attr from source {ei_src}: {tuple(tmp_attr.shape)}")
            else:
                if getattr(on_graph, "fc_matrix", None) is not None:
                    src_idx, dst_idx = ei_e[0], ei_e[1]
                    data[("eeg", "connects", "eeg")].edge_attr = torch.tensor(on_graph.fc_matrix[src_idx, dst_idx], dtype=torch.float32)
                    self._log(f"[EEG] Generated edge_attr from on_graph.fc_matrix for source {ei_src}")
            self._log(f"[EEG] 写入 HeteroData -- edge_type ('eeg','connects','eeg') from {ei_src}")
        else:
            self._log("[EEG] No eeg internal edge_index found after exhaustive checks", level="warning")


        # ---------------------------------------------------------------------
        # Cross modal edges
        # ---------------------------------------------------------------------
        self._log("[CM] 开始构建跨模态边")

        fmri_pos = data["fmri"].pos if hasattr(data["fmri"], "pos") else None
        eeg_pos = data["eeg"].pos if hasattr(data["eeg"], "pos") else None

        cross_ei, cross_attr = self._add_cross_modal_edges(fmri_pos, eeg_pos)

        self._log(f"[CM] cross_ei shape = {tuple(cross_ei.shape)}  num_edges={cross_ei.shape[1]}")

        if cross_ei.numel() > 0:
            # forward
            data[("fmri", "projects_to", "eeg")].edge_index = cross_ei
            data[("fmri", "projects_to", "eeg")].edge_attr = cross_attr
            self._log("[CM] 写入 forward ('fmri','projects_to','eeg')")

            # reverse
            rev_ei = cross_ei[[1, 0]]
            data[("eeg", "projects_to", "fmri")].edge_index = rev_ei
            data[("eeg", "projects_to", "fmri")].edge_attr = cross_attr
            self._log("[CM] 写入 reverse ('eeg','projects_to','fmri')")
            self._log(f"[CM] rev_ei shape = {tuple(rev_ei.shape)}")
        else:
            self._log("[CM] 未添加跨模态边")

        # ---------------------------------------------------------------------
        # Summary over edge_index_dict
        # ---------------------------------------------------------------------
        self._log("[SUMMARY] 开始记录 edge_index_dict")

        data.edge_index_dict = {}
        for et in data.edge_types:
            obj = data[et]
            if hasattr(obj, "edge_index") and obj.edge_index is not None:
                data.edge_index_dict[et] = obj.edge_index
                self._log(f"[SUMMARY] {et}: edge_index = {tuple(obj.edge_index.shape)}")
            else:
                self._log(f"[SUMMARY] {et}: edge_index = None")

        # ---------------------------------------------------------------------
        # Node summary
        # ---------------------------------------------------------------------
        self._log("[SUMMARY] 节点信息")
        for nt in data.node_types:
            node = data[nt]
            x_seq = node.x_seq.shape if hasattr(node, 'x_seq') else None
            x = node.x.shape if hasattr(node, 'x') else None
            self._log(f"  {nt}: x_seq={x_seq}, x={x}")

        self._log("=" * 120)
        self._log("[HETERO BUILD] 完成 _construct_hetero")
        self._log("=" * 120)

        return data
