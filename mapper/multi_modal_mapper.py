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
        cross_modal_neighbors: int = 5,
        cross_modal_distance_threshold: float = 50.0,
        verbose: bool = True,
    ):
        """Initialize MultiModalMapper.
        
        Args:
            fmri_mapper: fMRI data mapper instance.
            eeg_mapper: EEG data mapper instance.
            target_dim: Target dimension for projections.
            align_dim: Alignment dimension.
            align_mode: Alignment mode ('latent' or other).
            max_time: Maximum number of time points.
            cross_modal_neighbors: Number of nearest neighbors for cross-modal edges (k-NN).
            cross_modal_distance_threshold: Distance threshold in mm for cross-modal edges.
            verbose: Enable verbose logging.
        """
        self.fmri_mapper = fmri_mapper
        self.eeg_mapper = eeg_mapper
        self.target_dim = int(target_dim)
        self.align_dim = int(align_dim)
        self.align_mode = align_mode
        self.max_time = int(max_time)
        self.cross_modal_k = int(cross_modal_neighbors)
        self.cross_modal_thr = float(cross_modal_distance_threshold)
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
    # Extract time vector from HeteroData node
    # ================================
    def _get_time_vector(self, hetero: HeteroData, node_type: str) -> np.ndarray:
        """Extract time vector from HeteroData node.
        
        Checks multiple possible fields in priority order:
        1. Direct fields: times, time, timestamps, t_seq
        2. debug_info dict with n_tp and TR
        3. Inferred from x_seq shape
        
        Args:
            hetero: HeteroData graph.
            node_type: Node type to extract time from.
            
        Returns:
            Time vector as 1D numpy array.
            
        Raises:
            ValueError: If time cannot be extracted from any source.
        """
        node = hetero[node_type]

        # Direct fields
        for key in ("times", "time", "timestamps", "t_seq"):
            if hasattr(node, key):
                arr = getattr(node, key)
                if arr is not None:
                    return np.asarray(arr, dtype=np.float32).reshape(-1)

        # debug_info
        dbg = getattr(node, "debug_info", None)
        if isinstance(dbg, dict):
            if "n_tp" in dbg:
                n_tp = int(dbg["n_tp"])
                tr = float(dbg.get("TR", 1.0))
                return (np.arange(n_tp, dtype=np.float32) * tr)

        # Infer from x_seq shape
        if hasattr(node, "x_seq") and node.x_seq is not None:
            xseq = node.x_seq
            # Expected format: (N, T, C)
            if xseq.ndim == 3:
                n_tp = xseq.shape[1]
                return (np.arange(n_tp, dtype=np.float32))

        raise ValueError(f"Cannot extract time from node '{node_type}'")

    def _validate_and_extract(self, hetero: HeteroData, node_type: str) -> Dict[str, Any]:
        """Extract x_seq/x, edges, coordinates and other fields from HeteroData node.
        
        Assumes:
          - hetero is HeteroData
          - hetero[node_type] has x or x_seq attribute
          - x_seq shape is (N, T, C) or x shape is (N, C)
          
        Args:
            hetero: HeteroData graph.
            node_type: Node type to extract from.
            
        Returns:
            Dictionary with extracted tensors (x_seq, x_mean, edge_index, pos, fc_matrix).
            
        Raises:
            ValueError: If inputs are invalid.
        """
        if not isinstance(hetero, HeteroData):
            raise ValueError("Input to _validate_and_extract must be HeteroData")

        if node_type not in hetero.node_types:
            raise ValueError(f"Node type '{node_type}' missing in HeteroData")

        node = hetero[node_type]
        out = {}

        # Extract x_seq or x
        xseq = getattr(node, "x_seq", None)
        x = getattr(node, "x", None)

        if xseq is None and x is None:
            raise ValueError(f"{node_type}: missing both x_seq and x")

        # Standardize to 3D
        if xseq is not None:
            t = self._to_tensor(xseq)
        else:
            t = self._to_tensor(x)
        t3 = self._ensure_3d(t)            # (N, T, C)
        t3 = self._sanitize_numeric(t3)    # Remove NaN/inf

        out["x_seq"] = t3
        out["x_mean"] = t3.mean(dim=1)     # (N, C)

        # Extract edge_index
        eidx = getattr(node, "edge_index", None)
        if isinstance(eidx, torch.Tensor) and eidx.numel() > 0:
            out["edge_index"] = eidx.long()
        else:
            out["edge_index"] = torch.empty((2, 0), dtype=torch.long)

        # Extract position
        pos = getattr(node, "pos", None)
        out["pos"] = pos.float() if isinstance(pos, torch.Tensor) else None

        # Extract FC matrix (optional)
        out["fc_matrix"] = getattr(node, "fc_matrix", None)

        return out


    # ================================
    # Temporal alignment and edge building
    # ================================
    def _temporal_align_physical(
        self,
        fmri_x: torch.Tensor,
        eeg_x: torch.Tensor,
        fmri_graph: Any,  # HeteroData
        eeg_graph: Any    # Data or HeteroData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align fMRI and EEG sequences by physical time through interpolation.
        
        Args:
            fmri_x: fMRI tensor (N, T, C).
            eeg_x: EEG tensor (N, T, C).
            fmri_graph: HeteroData containing fMRI time information.
            eeg_graph: Data/HeteroData containing EEG time information.
            
        Returns:
            Tuple of (aligned_fmri, aligned_eeg) tensors.
            
        Raises:
            ValueError: If time ranges don't overlap.
        """
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

    def _add_cross_modal_edges(
        self,
        fmri_pos: torch.Tensor,
        eeg_pos: torch.Tensor,
        template_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build cross-modal edges between fMRI and EEG nodes.
        
        If template_path exists, loads edge structure from file.
        Otherwise, computes edges based on k-NN or distance threshold.
        
        Args:
            fmri_pos: fMRI node positions (N_fmri, 3).
            eeg_pos: EEG node positions (N_eeg, 3).
            template_path: Optional path to save/load edge template.
            
        Returns:
            Tuple of (edge_index, edge_attr) where edge_index is (2, E) and edge_attr is (E,).
        """
        # Generate edges if no template or positions missing
        if fmri_pos is None or eeg_pos is None:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)

        # Compute distance matrix once (used for both template loading and generation)
        dist = torch.cdist(fmri_pos, eeg_pos)
        
        # Load from template if available
        if template_path is not None and os.path.exists(template_path):
            saved = torch.load(template_path)
            edge_index = saved['edge_index']
            self._log(f"[CM] Loaded cross-modal edge template from {template_path}")

            # Compute edge attributes based on distance
            if edge_index.numel() == 0:
                edge_attr = torch.empty((0,), dtype=torch.float32)
            else:
                src, dst = edge_index
                edge_attr = 1.0 / (dist[src, dst] + 1e-6)
            return edge_index, edge_attr

        # Generate edges based on k-NN or distance threshold
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
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.float32)
        else:
            edge_index = torch.cat(edge_index_list, dim=1)
            edge_attr = torch.cat(edge_attr_list)

        # Save template (edge_index only)
        if template_path is not None:
            os.makedirs(os.path.dirname(template_path), exist_ok=True)
            torch.save({'edge_index': edge_index}, template_path)
            self._log(f"[CM] Saved cross-modal edge template to {template_path}")

        return edge_index, edge_attr

    def build_dynamic_from_graphs(
        self,
        on_graph: Any,
        off_graph: Any,
        fmri_graph: Any,
        stim_dict: Optional[Dict[str, Any]] = None,
        max_T: Optional[int] = None,
    ) -> List[HeteroData]:
        """Build dynamic HeteroData from ON/OFF EEG runs and fMRI.
        
        Complete fixed version with max_T truncation support.
        
        Args:
            on_graph: EEG ON condition HeteroData.
            off_graph: EEG OFF condition HeteroData.
            fmri_graph: fMRI HeteroData.
            stim_dict: Optional stimulation time series to inject.
            max_T: Optional maximum time points to truncate to.
            
        Returns:
            List containing single HeteroData with aligned sequences.
        """
        def _ensure_seq(name, x):
            if x is None:
                raise ValueError(f"[Dynamic] {name}.x_seq is None")
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"[Dynamic] {name}.x_seq must be torch.Tensor")
            if x.dim() != 3:
                raise ValueError(f"[Dynamic] {name}.x_seq must be 3D (N,T,F), got {tuple(x.shape)}")
            self._log(f"[Dynamic] {name}.x_seq shape = {tuple(x.shape)}")
            return x

        # Merge EEG runs
        eeg_merged = self._merge_eeg_runs(on_graph, off_graph)
        eeg_x = _ensure_seq("EEG(raw)", eeg_merged["x_seq"])

        # Extract fMRI
        fmri_info = self._validate_and_extract(fmri_graph, "fmri")
        fmri_x = _ensure_seq("fMRI(raw)", fmri_info["x_seq"])

        # Physical time alignment
        self._log("[Align] Starting physical time alignment")
        fmri_al, eeg_al = self._temporal_align_physical(fmri_x, eeg_x, fmri_graph, on_graph)
        fmri_al = _ensure_seq("fMRI(aligned)", fmri_al)
        eeg_al = _ensure_seq("EEG(aligned)", eeg_al)

        # Truncate to max_T if specified
        if max_T is not None:
            T_orig = fmri_al.shape[1]
            T = min(fmri_al.shape[1], eeg_al.shape[1])
            T = min(T, max_T)
            if T < T_orig:
                fmri_al = fmri_al[:, :T, :]
                eeg_al = eeg_al[:, :T, :]
                self._log(f"[Align] Truncated T={T_orig} → {T} (max_T={max_T})")

        # Build eeg_info
        eeg_info = on_graph.clone()
        eeg_info["eeg"].x_seq = eeg_al
        eeg_info["eeg"].x = eeg_al.mean(dim=1)

        # Update fmri_info
        fmri_info["x_seq"] = fmri_al

        # Construct HeteroData
        self._log("[Dynamic] Building HeteroData")
        combined = self._construct_hetero(fmri_info, eeg_info, on_graph)

        # Inject stimulation if provided
        if stim_dict:
            self._log("[Dynamic] Injecting stimulation time series")
            for ntype, stim in stim_dict.items():
                if ntype in combined and hasattr(combined[ntype], "x_seq"):
                    stim_t = torch.as_tensor(stim, dtype=torch.float32, device=combined[ntype].x_seq.device)
                    # Truncate stim if needed
                    if max_T is not None and stim_t.shape[0] > max_T:
                        stim_t = stim_t[:max_T]
                        self._log(f"[Stim] Truncated stim[{ntype}] to max_T={max_T}")
                    if stim_t.shape[0] == combined[ntype].x_seq.shape[1]:
                        combined[ntype].x_seq = combined[ntype].x_seq + stim_t
                        combined[ntype].x = combined[ntype].x_seq.mean(dim=1)

        return [combined]

    def _construct_hetero(self, fmri_info: Any, eeg_info: Any, on_graph: Any) -> HeteroData:
        """Construct HeteroData from fMRI and EEG information.
        
        Enforces fixed relation order and always writes all four relations (even if empty).
        Edge_index can come from fmri_info/eeg_info/on_graph templates.
        Edge_attr always computed dynamically from fc/dist and returned as 1D float32 tensors.
        
        Args:
            fmri_info: Dictionary with fMRI data.
            eeg_info: HeteroData with EEG data.
            on_graph: Original graph for template lookups.
            
        Returns:
            HeteroData with fmri and eeg nodes and all edge types.
        """
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

        # ------------------------------------------------------------
        # Helper: read edge_index candidates from info/on_graph
        # ------------------------------------------------------------
        def _read_edge_index_from_info(info_obj, et_key_candidates):
            """
            Try to extract edge_index tensor from info_obj given list of keys.
            Return (edge_index_tensor or None, source_desc or None)
            """
            if info_obj is None:
                return None, None
            # If info_obj is HeteroData, prefer direct access
            try:
                if hasattr(info_obj, "edge_types"):
                    for et in info_obj.edge_types:
                        if (et[0], et[1], et[2]) in et_key_candidates:
                            obj = info_obj[et]
                            if hasattr(obj, "edge_index") and obj.edge_index is not None and obj.edge_index.numel() > 0:
                                return obj.edge_index.clone(), f"heterodata[{et}]"
            except Exception:
                pass
            # If dict-like, try keys
            if isinstance(info_obj, dict):
                for key in et_key_candidates:
                    if key in info_obj and info_obj[key] is not None:
                        cand = info_obj[key]
                        if hasattr(cand, "edge_index"):
                            cand = cand.edge_index
                        if isinstance(cand, np.ndarray):
                            cand = torch.from_numpy(cand).long()
                        if isinstance(cand, torch.Tensor) and cand.numel() > 0:
                            return cand.clone(), f"dict[{key}]"
            return None, None

        # ------------------------------------------------------------
        # Determine edge_index candidates (priority: fmri_info/eeg_info -> on_graph)
        # ------------------------------------------------------------
        # fMRI
        ei_f, ei_f_src = None, None
        if "edge_index" in fmri_info and fmri_info["edge_index"] is not None:
            ei_f = fmri_info["edge_index"]
            if isinstance(ei_f, np.ndarray):
                ei_f = torch.from_numpy(ei_f).long()
            ei_f_src = "fmri_info['edge_index']"
        else:
            # try on_graph fallback
            ei_f, ei_f_src = _read_edge_index_from_info(on_graph, [( "fmri","connects","fmri")])

        # EEG
        ei_e, ei_e_src = None, None
        if isinstance(eeg_info, dict) and "edge_index" in eeg_info and eeg_info["edge_index"] is not None:
            ei_e = eeg_info["edge_index"]
            if isinstance(ei_e, np.ndarray):
                ei_e = torch.from_numpy(ei_e).long()
            ei_e_src = "eeg_info['edge_index']"
        else:
            ei_e, ei_e_src = _read_edge_index_from_info(on_graph, [("eeg","connects","eeg")])

        # ------------------------------------------------------------
        # Prepare FC matrices (ensure numpy for indexing convenience)
        # ------------------------------------------------------------
        fmri_fc = None
        if fmri_info.get("fc_matrix") is not None:
            fmri_fc = np.asarray(fmri_info["fc_matrix"], dtype=np.float32)
        elif hasattr(on_graph, "fc_matrix") and on_graph.fc_matrix is not None:
            fmri_fc = np.asarray(on_graph.fc_matrix, dtype=np.float32)

        eeg_fc = None
        if isinstance(eeg_info, dict) and eeg_info.get("fc_matrix") is not None:
            eeg_fc = np.asarray(eeg_info["fc_matrix"], dtype=np.float32)
        elif hasattr(on_graph, "fc_matrix") and on_graph.fc_matrix is not None:
            eeg_fc = np.asarray(on_graph.fc_matrix, dtype=np.float32)

        # ------------------------------------------------------------
        # Cross-modal positions
        # ------------------------------------------------------------
        fmri_pos = data["fmri"].pos if hasattr(data["fmri"], "pos") else None
        eeg_pos = data["eeg"].pos if hasattr(data["eeg"], "pos") else None

        # ------------------------------------------------------------
        # Fixed relation order (enforced)
        # ------------------------------------------------------------
        REL_ORDER = [
            ("fmri", "connects", "fmri"),
            ("eeg", "connects", "eeg"),
            ("fmri", "projects_to", "eeg"),
            ("eeg", "projects_to", "fmri"),
        ]

        # ------------------------------------------------------------
        # 1) fmri -> fmri
        # ------------------------------------------------------------
        if ei_f is not None and isinstance(ei_f, torch.Tensor) and ei_f.numel() > 0:
            data[("fmri","connects","fmri")].edge_index = ei_f.long()
            # compute edge_attr from fmri_fc if available
            if fmri_fc is not None:
                src_np = ei_f[0].cpu().numpy().astype(int)
                dst_np = ei_f[1].cpu().numpy().astype(int)
                vals = fmri_fc[src_np, dst_np]
                data[("fmri","connects","fmri")].edge_attr = torch.from_numpy(np.asarray(vals, dtype=np.float32))
                self._log(f"[fMRI] Wrote edge_attr from {ei_f_src or 'fmri_fc'} shape={data[('fmri','connects','fmri')].edge_attr.shape}")
            else:
                data[("fmri","connects","fmri")].edge_attr = torch.empty((ei_f.size(1),), dtype=torch.float32)
        else:
            data[("fmri","connects","fmri")].edge_index = torch.zeros((2,0), dtype=torch.long)
            data[("fmri","connects","fmri")].edge_attr = torch.empty((0,), dtype=torch.float32)
            self._log("[fMRI] No fmri edges written (empty)")

        # ------------------------------------------------------------
        # 2) eeg -> eeg
        # ------------------------------------------------------------
        if ei_e is not None and isinstance(ei_e, torch.Tensor) and ei_e.numel() > 0:
            data[("eeg","connects","eeg")].edge_index = ei_e.long()
            if eeg_fc is not None:
                src_np = ei_e[0].cpu().numpy().astype(int)
                dst_np = ei_e[1].cpu().numpy().astype(int)
                vals = eeg_fc[src_np, dst_np]
                data[("eeg","connects","eeg")].edge_attr = torch.from_numpy(np.asarray(vals, dtype=np.float32))
                self._log(f"[EEG] Wrote edge_attr from {ei_e_src or 'eeg_fc'} shape={data[('eeg','connects','eeg')].edge_attr.shape}")
            else:
                data[("eeg","connects","eeg")].edge_attr = torch.empty((ei_e.size(1),), dtype=torch.float32)
        else:
            data[("eeg","connects","eeg")].edge_index = torch.zeros((2,0), dtype=torch.long)
            data[("eeg","connects","eeg")].edge_attr = torch.empty((0,), dtype=torch.float32)
            self._log("[EEG] No eeg edges written (empty)")

        # ------------------------------------------------------------
        # 3) cross-modal: use self._add_cross_modal_edges (which may accept template_path)
        # ------------------------------------------------------------
        cross_ei, cross_attr = self._add_cross_modal_edges(fmri_pos, eeg_pos)

        if cross_ei is not None and cross_ei.numel() > 0:
            # forward
            data[("fmri","projects_to","eeg")].edge_index = cross_ei.long()
            # ensure cross_attr is 1D float tensor
            if isinstance(cross_attr, torch.Tensor):
                data[("fmri","projects_to","eeg")].edge_attr = cross_attr.to(dtype=torch.float32).view(-1)
            else:
                data[("fmri","projects_to","eeg")].edge_attr = torch.from_numpy(np.asarray(cross_attr, dtype=np.float32))
            # reverse
            rev = torch.stack([cross_ei[1], cross_ei[0]], dim=0)
            data[("eeg","projects_to","fmri")].edge_index = rev.long()
            # reverse attr: same values but aligned with rev ordering (here symmetric so same)
            data[("eeg","projects_to","fmri")].edge_attr = data[("fmri","projects_to","eeg")].edge_attr.clone()
            self._log(f"[CM] cross_ei shape = {tuple(cross_ei.shape)}  num_edges={cross_ei.shape[1]}")
        else:
            data[("fmri","projects_to","eeg")].edge_index = torch.zeros((2,0), dtype=torch.long)
            data[("fmri","projects_to","eeg")].edge_attr = torch.empty((0,), dtype=torch.float32)
            data[("eeg","projects_to","fmri")].edge_index = torch.zeros((2,0), dtype=torch.long)
            data[("eeg","projects_to","fmri")].edge_attr = torch.empty((0,), dtype=torch.float32)
            self._log("[CM] No cross-modal edges (empty)")

        # ------------------------------------------------------------
        #  Finalize: build edge_index_dict following REL_ORDER to guarantee order/stability
        # ------------------------------------------------------------
        data.edge_index_dict = {}
        for et in REL_ORDER:
            obj = data[et]
            # ensure edge_index is present (it should be)
            ei = getattr(obj, "edge_index", None)
            data.edge_index_dict[et] = ei if ei is not None else torch.zeros((2,0), dtype=torch.long)
            self._log(f"[SUMMARY] {et}: edge_index = {tuple(data.edge_index_dict[et].shape) if isinstance(data.edge_index_dict[et], torch.Tensor) else None}")

        # Node summary
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