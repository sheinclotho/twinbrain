# mapper/bids_mapper.py
import nibabel as nib
import numpy as np
import json
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from pathlib import Path
import torch
from torch_geometric.data import HeteroData
from preprocess.fmri_preprocessor import FMRI_Preprocessor
import xml.etree.ElementTree as ET
import logging
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

class BIDSMapper:
    """
    fMRI -> ROI -> 功能连接图构建器 (修复版)
    规范：
      - self.data: (N_nodes, T_time)
      - get_node_features(with_stats=False): (N, T, 1) 原始序列
      - get_node_features(with_stats=True): (N, T, 1) 标准化序列
      - to_pyg:
          x_seq: (N, T, 1) 原始序列
          x: (N, 1) 时间均值
    """
    def __init__(self, atlas_name: str = 'schaefer',
                 atlas_file: str = None,
                 label_file: str = None,
                 func_dir: str = None,
                 task_name: str = None):
        self.atlas_name = atlas_name.lower()
        self.atlas_img = None
        self.labels = []
        self.data = None  # (N, T)
        self.node_mask = None
        self.fc_matrix = None
        self.edge_index = None
        self.edge_attr = None
        self.region_ids = None
        self.region_positions = None
        self.region_functions = None
        self.node_labels = None
        self.func_dir = Path(func_dir) if func_dir else None
        self.task_name = task_name
        self.fmri_file = None
        self.confounds_file = None
        self.debug_info = {}
        self._load_atlas(atlas_file, label_file)
        self.n_regions = len(self.labels)
        logging.debug(f"[INIT] Loaded atlas '{self.atlas_name}' with {self.n_regions} regions")
        if self.func_dir and self.task_name:
            self._auto_find_bids_files()

    def _auto_find_bids_files(self):
        candidates = list(self.func_dir.glob(f"*task-{self.task_name.lower()}*bold.nii*"))
        if not candidates:
            candidates = list(self.func_dir.glob(f"*task-{self.task_name.upper()}*bold.nii*"))
        if not candidates:
            raise FileNotFoundError(f"[BIDSMapper] No fMRI file found for task={self.task_name} in {self.func_dir}")
        self.fmri_file = candidates[0]
        conf = list(self.func_dir.glob(f"*task-{self.task_name.lower()}*confounds*.tsv"))
        if conf:
            self.confounds_file = conf[0]
        logging.info(f"[BIDSMapper] Found fMRI: {self.fmri_file.name}")
        if self.confounds_file:
            logging.info(f"[BIDSMapper] Found confounds: {self.confounds_file.name}")

    def _load_atlas(self, atlas_file=None, label_file=None):
        atlas_file = Path(atlas_file) if atlas_file else atlas_file
        atlas_img = nib.load(atlas_file)
        if len(atlas_img.shape) > 3:
            atlas_img = nib.Nifti1Image(atlas_img.get_fdata()[..., 0], atlas_img.affine)
        self.atlas_img = atlas_img
        if self.atlas_name == 'schaefer':
            label_file = label_file or Path(atlas_file).parent / "schaefer200_mask_ready.json"
            with open(label_file, 'r', encoding='utf-8') as f:
                atlas_json = json.load(f)
            self.labels = list(atlas_json['regions'].keys())
        elif self.atlas_name == 'aal':
            label_file = label_file or Path(atlas_file).parent / "aal.xml"
            with open(label_file, 'r', encoding='ISO-8859-1') as f:
                root = ET.fromstring(f.read())
                self.labels = [label.find('name').text for label in root.findall('.//label')]
        else:
            raise ValueError("Unsupported atlas: choose 'schaefer' or 'aal'")
        logging.info(f"[Atlas] Loaded '{self.atlas_name}' with {len(self.labels)} regions.")

    def load_and_preprocess(self, fmri_file=None, confounds_file=None, standardize=True):
        fmri_file = fmri_file or self.fmri_file
        confounds_file = confounds_file or self.confounds_file
        if fmri_file is None:
            raise FileNotFoundError("No fMRI file provided or found")

        logging.info(f"[fMRI] Loading and preprocessing: {fmri_file}")
        preprocessor = FMRI_Preprocessor(tr=2.0, high_pass=0.01, low_pass=0.1, smoothing_fwhm=6.0)
        clean_ts = preprocessor.preprocess(fmri_file, confounds_file=confounds_file)
        clean_img = preprocessor.inverse_transform(clean_ts)

        try:
            self.atlas_img = image.resample_to_img(
                self.atlas_img, clean_img.slicer[..., 0], interpolation="nearest"
            )
        except TypeError:
            self.atlas_img = image.resample_to_img(
                self.atlas_img, clean_img.slicer[..., 0], interpolation="nearest", copy_header=True
            )

        masker = NiftiLabelsMasker(labels_img=self.atlas_img, standardize=standardize)
        ts = masker.fit_transform(clean_img).astype(np.float32)  # (T, N)
        self.data = ts.T  # (N, T)

        atlas_labels = np.unique(self.atlas_img.get_fdata())
        if 0 in atlas_labels:
            atlas_labels = atlas_labels[atlas_labels != 0]
        self.node_mask = np.isin(np.arange(1, len(self.labels) + 1), atlas_labels).astype(np.float32)

        tr = preprocessor.tr
        n_timepoints = self.data.shape[1]
        self.times = np.arange(n_timepoints) * tr
        self.debug_info["fMRI_time_steps"] = int(n_timepoints)
        self.debug_info["ROI_count"] = int(self.data.shape[0])
        self.debug_info["TR"] = tr
        logging.info(f"[fMRI] Extracted ROI time series: (nodes, time) = {self.data.shape}")
        logging.info(f"[fMRI] Active nodes: {int(np.sum(self.node_mask))}/{len(self.node_mask)}")
        logging.info(f"[fMRI] Time vector: {n_timepoints} points, TR={tr}s, total={n_timepoints*tr:.1f}s")

        try:
            fc = np.corrcoef(self.data)
            fc = fc.astype(np.float32)
            self.fc_matrix = fc
            n = fc.shape[0]
            if n <= 1:
                raise RuntimeError("Not enough nodes.")
            thr = np.percentile(np.abs(fc[np.triu_indices(n, k=1)]), 90)
            mask = (np.abs(fc) >= thr)
            np.fill_diagonal(mask, 0)
            edge_index = np.array(np.nonzero(mask))
            self.edge_index = edge_index
            self.edge_attr = np.abs(fc[edge_index[0], edge_index[1]]).astype(np.float32)
            logging.info(f"[Graph] Built {self.edge_index.shape[1]} edges (thr={thr:.4f})")
        except Exception as e:
            logging.warning(f"[Graph] Failed to build edge_index: {e}")
            self.edge_index = np.zeros((2, 0), dtype=int)
            self.edge_attr = None

    def attach_brain_atlas(self, brain_atlas):
        if not hasattr(brain_atlas, "regions"):
            raise ValueError("Invalid BrainAtlas object")
        region_ids, positions, functions = [], [], []
        matched = 0
        unmatched_labels = []
        node_labels = []
        for label in self.labels:
            match = None
            for rid, info in brain_atlas.regions.items():
                if label == rid or label == info.get("label_id") or label == info.get("function"):
                    match = rid
                    break
                if isinstance(info.get("label_id"), str) and label.lower() in info.get("label_id").lower():
                    match = rid
                    break
            region_ids.append(match)
            if match is not None:
                matched += 1
                positions.append(brain_atlas.regions[match]["position"])
                functions.append(brain_atlas.regions[match]["function"])
            else:
                positions.append(np.zeros(3))
                functions.append("unknown")
                if len(unmatched_labels) < 10:
                    unmatched_labels.append(label)
            node_labels.append(functions[-1])

        self.region_ids = region_ids
        self.region_positions = np.array(positions, dtype=np.float32)
        self.region_functions = functions

        unique_funcs = sorted(list(set(node_labels)))
        func2idx = {f: i for i, f in enumerate(unique_funcs)}
        labels = np.zeros((len(node_labels), len(unique_funcs)), dtype=np.float32)
        for i, f in enumerate(node_labels):
            labels[i, func2idx[f]] = 1.0
        self.node_labels = labels
        logging.info(f"[Atlas] Attached BrainAtlas metadata: {matched}/{len(region_ids)} matched")
        if unmatched_labels:
            logging.info(f"[Atlas] Some unmatched labels (up to 10): {unmatched_labels}")
        return labels

    def get_node_features(self, with_stats: bool = False) -> torch.Tensor:
        if self.data is None:
            raise RuntimeError("Call load_and_preprocess first.")
       
        x = torch.from_numpy(self.data).float()  # (N, T)
        x = x.unsqueeze(-1)  # (N, T, 1)
       
        if with_stats:
            mean = x.mean(dim=(0,1), keepdim=True)
            std = x.std(dim=(0,1), keepdim=True) + 1e-6
            x = (x - mean) / std
            logging.debug(f"[get_node_features] Applied z-score normalization")
        return x.detach()

    def to_pyg(self, node_type="fmri"):
        if self.edge_index is None or self.data is None:
            raise ValueError("Data not prepared. Run load_and_preprocess() first.")

        # 修复：分开获取
        x_seq_raw = self.get_node_features(with_stats=False)  # 原始
        x_seq_tensor = x_seq_raw.clone().detach()
        x_mean_tensor = x_seq_raw.mean(dim=1)  # 均值

        debug = {
            'x_seq_shape': tuple(x_seq_tensor.shape),
            'x_mean_shape': tuple(x_mean_tensor.shape),
            'with_stats': False
        }

        edge_index_tensor = torch.tensor(self.edge_index, dtype=torch.long) if self.edge_index.size > 0 else torch.zeros((2,0), dtype=torch.long)
        edge_attr_tensor = torch.tensor(self.edge_attr, dtype=torch.float32) if self.edge_attr is not None else None

        data = HeteroData()
        data[node_type].x = x_mean_tensor
        data[node_type].x_seq = x_seq_tensor
        data[node_type].node_mask = torch.tensor(self.node_mask, dtype=torch.float32)
        data[node_type].edge_index = edge_index_tensor
        if edge_attr_tensor is not None:
            data[node_type].edge_attr = edge_attr_tensor
        data[node_type].times = torch.tensor(self.times, dtype=torch.float32)

        pos_tensor = torch.tensor(self.region_positions, dtype=torch.float32) if self.region_positions is not None else torch.zeros((x_mean_tensor.shape[0], 3))
        n_nodes = x_mean_tensor.shape[0]
        if pos_tensor.shape[0] != n_nodes:
            if pos_tensor.shape[0] > n_nodes:
                pos_tensor = pos_tensor[:n_nodes]
            else:
                pad = torch.zeros((n_nodes - pos_tensor.shape[0], 3))
                pos_tensor = torch.cat([pos_tensor, pad], dim=0)
        if edge_index_tensor.numel() > 0:
            max_idx = edge_index_tensor.max().item()
            if max_idx >= n_nodes:
                mask = (edge_index_tensor[0] < n_nodes) & (edge_index_tensor[1] < n_nodes)
                edge_index_tensor = edge_index_tensor[:, mask]
                if edge_attr_tensor is not None:
                    edge_attr_tensor = edge_attr_tensor[mask]
        data[node_type].pos = pos_tensor

        if self.region_ids is not None:
            data[node_type].region_ids = torch.tensor([
                -1 if r is None else int(r) if isinstance(r, (int, np.integer)) else -1
                for r in self.region_ids
            ], dtype=torch.long)
        if self.node_labels is not None:
            data[node_type].y = torch.tensor(self.node_labels, dtype=torch.float32)
        data[node_type].debug_info = {**self.debug_info, **debug}
        data.mapper = self

        logging.info(f"[to_pyg] Added x_seq: {x_seq_tensor.shape}, x: {x_mean_tensor.shape}")
        return data

