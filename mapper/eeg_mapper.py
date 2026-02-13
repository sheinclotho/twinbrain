# mapper/eeg_mapper.py
import re
import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging
import mne
import numpy as np
import torch
from mne.time_frequency import psd_array_welch
from scipy.stats import skew, kurtosis
from tkinter import Tk, filedialog
from torch_geometric.data import Data
from torch_geometric.data import HeteroData

from preprocess.eeg_preprocessor import EEGPreprocessor

# Configure logger for EEGMapper
logger = logging.getLogger(__name__)


class EEGMapper:
    """
    EEG -> intermediate representation -> PyG Data converter.

    主要改进：
    - 将 epochs 正确拼接为连续时间序列 (n_channels, total_times)
    - 对时间轴使用分块 (max_time_chunk) 逐块计算特征，防止 OOM / 卡死
    - 更稳健的 PSD 参数自适应
    - 丰富 debug 输出，便于定位问题
    """

    def __init__(
        self,
        preprocessor: Optional[EEGPreprocessor] = None,
        epoch_length: float = 2.0,
        stc_cache_dir: Union[str, Path] = "cache/stc",
        debug: bool = True
    ):
        self.preprocessor = preprocessor if preprocessor else EEGPreprocessor()
        self.epoch_length = float(epoch_length)
        self.debug = debug

        self.raw: Optional[mne.io.BaseRaw] = None
        self.epochs: Optional[np.ndarray] = None  # 期望: (n_epochs, n_channels, n_times)
        self.fc_matrix: Optional[np.ndarray] = None
        self.node_mask: Optional[np.ndarray] = None
        self.channel_names: Optional[list] = None
        self.stc: Optional[mne.SourceEstimate] = None
        self.region_ids: Optional[np.ndarray] = None
        self.stc_cache_dir = Path(stc_cache_dir)
        self.stc_cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 文件查找与加载
    # -------------------------
    def _choose_directory(self, title: str) -> Path:
        try:
            root = Tk()
            root.withdraw()
            path = filedialog.askdirectory(title=title)
            root.destroy()
            if not path:
                raise FileNotFoundError("No directory selected.")
            return Path(path)
        except Exception:
            return Path.cwd()

    def _find_on_off_files(self, eeg_dir: Union[str, Path], task_name: Optional[str] = None) -> Union[Tuple[Path, Path], Dict[str, Dict[str, Path]]]:
        eeg_dir = Path(eeg_dir)
        eeg_files = list(eeg_dir.glob("sub-*_task-*_run-*_eeg.set"))
        task_map = {}
        for f in eeg_files:
            fname = f.name.lower()
            match = re.search(r"task-([a-z]+)(on|off)", fname)
            if match:
                base = match.group(1).upper()
                state = match.group(2).upper()
                task_map.setdefault(base, {})[state] = f
        if not task_map:
            raise FileNotFoundError(f"[EEGMapper] No task-XXX(ON|OFF) files found in {eeg_dir}")
        if task_name:
            task_name = task_name.upper()
            if task_name not in task_map or "ON" not in task_map[task_name] or "OFF" not in task_map[task_name]:
                raise FileNotFoundError(f"[EEGMapper] Cannot find ON/OFF for specified task={task_name} in {eeg_dir}")
            return task_map[task_name]["ON"], task_map[task_name]["OFF"]
        return task_map

    def load_task(self, task_name: Optional[str] = None, merge: bool = False, eeg_dir: Optional[Union[str, Path]] = None):
        if eeg_dir is None:
            eeg_dir = self._choose_directory(f"Select EEG directory for task {task_name or 'auto-detect'}")
        task_files = self._find_on_off_files(eeg_dir, task_name)
        if isinstance(task_files, dict):
            data = {}
            for t, files in task_files.items():
                on_file, off_file = files.get("ON"), files.get("OFF")
                if not (on_file and off_file):
                    logger.warning(f"Skipping incomplete task {t}")
                    continue
                if merge:
                    raw_on = self.preprocessor.preprocess(str(on_file))
                    raw_off = self.preprocessor.preprocess(str(off_file))
                    combined_raw = mne.concatenate_raws([raw_on, raw_off])
                    mapper = EEGMapper(preprocessor=self.preprocessor,
                                       epoch_length=self.epoch_length,
                                       stc_cache_dir=self.stc_cache_dir,
                                       debug=self.debug)
                    mapper.raw = combined_raw
                    mapper.channel_names = mapper.raw.ch_names
                    mapper.epochs = self.preprocessor.extract_epochs(mapper.raw, epoch_length=self.epoch_length)
                    mapper.node_mask = np.ones(len(mapper.channel_names), dtype=np.float32)
                    data[t] = mapper
                    logger.info(f"Loaded task {t}: merged {on_file.name} + {off_file.name}")
                else:
                    mapper_on = EEGMapper(preprocessor=self.preprocessor, epoch_length=self.epoch_length, stc_cache_dir=self.stc_cache_dir, debug=self.debug)
                    mapper_off = EEGMapper(preprocessor=self.preprocessor, epoch_length=self.epoch_length, stc_cache_dir=self.stc_cache_dir, debug=self.debug)
                    mapper_on.load_file(str(on_file))
                    mapper_off.load_file(str(off_file))
                    data[t] = {"on": mapper_on, "off": mapper_off}
                    logger.info(f"Loaded task {t}: ON={on_file.name}, OFF={off_file.name}")
            return data
        on_file, off_file = task_files
        if merge:
            raw_on = self.preprocessor.preprocess(str(on_file))
            raw_off = self.preprocessor.preprocess(str(off_file))
            self.raw = mne.concatenate_raws([raw_on, raw_off])
            self.channel_names = self.raw.ch_names
            self.epochs = self.preprocessor.extract_epochs(self.raw, epoch_length=self.epoch_length)
            self.node_mask = np.ones(len(self.channel_names), dtype=np.float32)
            return self
        else:
            mapper_on = EEGMapper(preprocessor=self.preprocessor, epoch_length=self.epoch_length, stc_cache_dir=self.stc_cache_dir, debug=self.debug)
            mapper_off = EEGMapper(preprocessor=self.preprocessor, epoch_length=self.epoch_length, stc_cache_dir=self.stc_cache_dir, debug=self.debug)
            mapper_on.load_file(str(on_file))
            mapper_off.load_file(str(off_file))
            return {"on": mapper_on, "off": mapper_off}

    def load_file(self, eeg_file: Optional[str] = None):
        if not eeg_file:
            raise FileNotFoundError("No EEG file provided or selected.")
        self.raw = self.preprocessor.preprocess(eeg_file)
        self.channel_names = self.raw.ch_names
        self.epochs = self.preprocessor.extract_epochs(self.raw, epoch_length=self.epoch_length)
        self.node_mask = np.ones(len(self.channel_names), dtype=np.float32)

        return self

    # -------------------------
    # 源定位（带缓存）
    # -------------------------
    def source_localization(self, method: str = "dSPM", force_rerun: bool = False) -> mne.SourceEstimate:
        if self.raw is None:
            raise ValueError("No raw EEG loaded.")
        ch_hash = "_".join(self.channel_names) if self.channel_names else "unknown_ch"
        cache_file = self.stc_cache_dir / f"{ch_hash}_{int(self.raw.info['sfreq'])}.stc"
        if cache_file.exists() and not force_rerun:
            try:
                stc = mne.read_source_estimate(str(cache_file))
                self.stc = stc
                return stc
            except Exception:
                pass
        try:
            self.raw.set_montage('standard_1005', on_missing='warn')
        except Exception as e:
            logger.warning(f"Montage setup failed: {e}")
        eeg_picks = mne.pick_types(self.raw.info, eeg=True, exclude=[])
        if len(eeg_picks) == 0:
            raise RuntimeError("No EEG channels.")
        info_for_forward = mne.pick_info(self.raw.info, eeg_picks)
        bem_sol = mne.make_sphere_model(r0='auto', head_radius=0.09, info=info_for_forward)
        origin = np.array([0.0, 0.0, 0.0])
        rr = origin.reshape(1, 3)
        nn = np.array([[0, 0, 1]])
        src = mne.setup_volume_source_space(
            pos=dict(
                rr=rr,
                nn=nn,
                origin=origin,
                int_rad=80.0,
                voxel_size=20.0
            ),
            bem=bem_sol,
            mri=None,
            add_interpolator=True
        )
        trans = mne.transforms.Transform("head", "mri", np.eye(4))
        fwd = mne.make_forward_solution(
            info_for_forward, trans=trans, src=src, bem=bem_sol,
            eeg=True, meg=False, verbose=False
        )
        noise_cov = mne.compute_raw_covariance(self.raw, tmin=0, tmax=None, rank='info')
        inv = mne.minimum_norm.make_inverse_operator(
            self.raw.info, fwd, noise_cov,
            loose=0.2, depth=None, verbose=False
        )
        stc = mne.minimum_norm.apply_inverse_raw(self.raw, inv, lambda2=1.0/9.0, method=method)
        try:
            stc.save(str(cache_file), overwrite=True)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
        self.stc = stc
        return stc

    # ------------------------
    # 生成标签
    # -------------------------
    def generate_node_labels(self, mode="function"):
        if mode == "function":
            self.node_labels = [self.channel_function[ch] for ch in self.channel_names]
        elif mode == "region":
            self.node_labels = [self.channel_region[ch] for ch in self.channel_names]
        else:
            self.node_labels = [0]*len(self.channel_names)  # fallback
        return self.node_labels


    # -------------------------
    # 特征计算（保留时间维），分块处理以防 OOM
    # -------------------------
    def get_node_features(self,
                          with_stats: bool = True,
                          auto_adjust_fft: bool = True,
                          max_time_chunk: int = 5000) -> np.ndarray:
        """
        返回: ndarray shape (n_channels, total_times, feat_dim)

        参数:
        - max_time_chunk: 单次处理的最大时间点数（避免一次性内存过大）
        """
        if self.epochs is None:
            raise ValueError("No epochs extracted. Call load_file or load_task first.")

        # === 1) 将 epochs 展平为连续时间序列 (n_channels, total_times) ===
        # self.epochs 常见格式: (n_epochs, n_channels, n_times_per_epoch)
        if isinstance(self.epochs, list):
            epochs_arr = np.stack(self.epochs, axis=0)
        else:
            epochs_arr = np.asarray(self.epochs)

        if epochs_arr.ndim != 3:
            raise ValueError(f"Unexpected epochs shape: {epochs_arr.shape}. Expect (n_epochs, n_channels, n_times).")

        n_epochs, n_channels, n_times_epoch = epochs_arr.shape
        total_times = n_epochs * n_times_epoch
        ts = epochs_arr.transpose(1, 0, 2).reshape(n_channels, total_times)  #(N, T)

        # 标准化（每通道）
        ts = (ts - ts.mean(axis=1, keepdims=True)) / (ts.std(axis=1, keepdims=True) + 1e-8)

        if self.debug:
            logger.debug(f"Feature source ts shape={ts.shape} (ch, time), epochs={n_epochs}, epoch_len={n_times_epoch}")
            logger.debug(f"max_time_chunk={max_time_chunk}")

        # === 2) 分块处理时间轴 ===
        feat_chunks = []
        time_slices = []
        start = 0
        while start < total_times:
            end = min(start + max_time_chunk, total_times)
            sub_ts = ts[:, start:end]  # (ch, chunk_time)
            if self.debug:
                logger.debug(f"Processing time chunk {start}:{end} -> shape {sub_ts.shape}")

            sub_feats = self._extract_feats_for_chunk(sub_ts,
                                                      with_stats=with_stats,
                                                      auto_adjust_fft=auto_adjust_fft)
            # sub_feats: (n_channels, chunk_time, feat_dim)
            feat_chunks.append(sub_feats)
            time_slices.append((start, end))
            start = end

        # === 3) 拼回完整时间轴 ===
        all_feats = np.concatenate(feat_chunks, axis=1) if feat_chunks else np.zeros((n_channels, 0, 0), dtype=np.float32)

        if self.debug:
            logger.debug(f"Feature shape={all_feats.shape} (ch, time, feat_dim)")
            if all_feats.size:
                logger.debug(f"Feature dim per node={all_feats.shape[-1]}")
                logger.debug(f"Mean amplitude per ch (first 5): {ts.mean(axis=1)[:5]}")

        return all_feats.astype(np.float32)

    def _extract_feats_for_chunk(self, sub_ts: np.ndarray, with_stats: bool, auto_adjust_fft: bool) -> np.ndarray:
        """
        sub_ts: (n_channels, chunk_time)
        返回: (n_channels, chunk_time, feat_dim)
        """
        n_channels, n_times = sub_ts.shape
        feats_list = []

        # 原始时序作为第一维特征 (逐时刻)
        feats_list.append(sub_ts[:, :, np.newaxis])  # (ch, t, 1)

        if with_stats:
            # 时域统计（per-channel）重复到每个时间点
            mean = sub_ts.mean(axis=1, keepdims=True)  # (ch,1)
            std = sub_ts.std(axis=1, keepdims=True)
            var = sub_ts.var(axis=1, keepdims=True)
            sks = np.array([skew(sub_ts[i]) for i in range(n_channels)])[:, np.newaxis]
            kts = np.array([kurtosis(sub_ts[i]) for i in range(n_channels)])[:, np.newaxis]
            stat_feats = np.concatenate([mean, std, var, sks, kts], axis=1)  # (ch, 5)
            stat_feats_exp = np.repeat(stat_feats[:, np.newaxis, :], n_times, axis=1)  # (ch, t, 5)
            feats_list.append(stat_feats_exp)

        # 频域特征：对 chunk 内每个通道计算 PSD 并重复到时间轴（节省内存）
        sfreq = float(self.raw.info["sfreq"])
        # 自适应 PSD 参数（保证 n_per_seg 合理且不会超过 chunk 长度）
        if auto_adjust_fft:
            n_per_seg = min(256, max(64, n_times // 4))
        else:
            n_per_seg = 256
        # n_fft 至少为 n_per_seg 的最小 2^k
        n_fft = 1 << (int(np.ceil(np.log2(n_per_seg))) if n_per_seg > 0 else 8)

        psd_list = []
        for i in range(n_channels):
            try:
                psd, freqs = psd_array_welch(
                    sub_ts[i], sfreq=sfreq, fmin=0.5, fmax=45.0,
                    n_fft=n_fft, n_per_seg=n_per_seg, average='mean'
                )
                delta = psd[(freqs >= 0.5) & (freqs < 4)].mean() if psd.size else 0.0
                theta = psd[(freqs >= 4) & (freqs < 8)].mean() if psd.size else 0.0
                alpha = psd[(freqs >= 8) & (freqs < 13)].mean() if psd.size else 0.0
                beta = psd[(freqs >= 13) & (freqs < 30)].mean() if psd.size else 0.0
                gamma = psd[(freqs >= 30) & (freqs < 45)].mean() if psd.size else 0.0
            except Exception as e:
                if self.debug:
                    logger.warning(f"PSD failed for channel {i} in chunk: {e}")
                delta = theta = alpha = beta = gamma = 0.0
            psd_list.append([delta, theta, alpha, beta, gamma])

        psd_arr = np.array(psd_list)  # (ch, 5)
        psd_exp = np.repeat(psd_arr[:, np.newaxis, :], n_times, axis=1)  # (ch, t, 5)
        feats_list.append(psd_exp)

        # 合并
        all_feats = np.concatenate(feats_list, axis=2)  # (ch, t, feat_dim)
        return all_feats.astype(np.float32)

    # -------------------------
    # 功能连接
    # -------------------------
    def compute_functional_connectivity(self, method: str = "correlation") -> np.ndarray:
        """
        计算 fc 基于连续时间序列（将 epochs 展平）。
        返回 (n_channels, n_channels) 的矩阵。
        """
        if self.epochs is None:
            raise ValueError("No epochs extracted. Call load_file or load_task first.")

        # 同上，把 epochs 展平为 (ch, total_times)
        if isinstance(self.epochs, list):
            epochs_arr = np.stack(self.epochs, axis=0)
        else:
            epochs_arr = np.asarray(self.epochs)
        if epochs_arr.ndim != 3:
            raise ValueError(f"Unexpected epochs shape: {epochs_arr.shape}.")
        n_epochs, n_channels, n_times_epoch = epochs_arr.shape
        ts = epochs_arr.transpose(1, 0, 2).reshape(n_channels, n_epochs * n_times_epoch)

        if method == "correlation":
            fc = np.corrcoef(ts)
        elif method == "covariance":
            fc = np.cov(ts)
        else:
            raise ValueError(f"Unsupported FC method: {method}")
        self.fc_matrix = fc.astype(np.float32)
        if self.debug:
            logger.debug(f"FC matrix computed, shape={self.fc_matrix.shape}")
        return self.fc_matrix

    # -------------------------
    # get_edge_index (保持原有接口)
    # -------------------------
    def get_edge_index(
            self,
            method: str = "threshold",
            threshold: Optional[float] = None,
            k: Optional[int] = None,
            template_path: Optional[str] = None,
        ) -> np.ndarray:
        """
        返回 edge_index: shape (2, n_edges)

        新增功能：
        - template_path 可选
        - 若模板存在 → 加载模板
        - 若模板不存在 → 生成并保存模板
        """
        # -----------------------------
        # 1. 若提供了模板路径且文件存在 → 直接加载
        # -----------------------------
        if template_path is not None and os.path.exists(template_path):
            ei = torch.load(template_path)
            if isinstance(ei, torch.Tensor):
                ei = ei.cpu().numpy()
            return ei

        # -----------------------------
        # 2. 正常流程：计算功能连接矩阵
        # -----------------------------
        if self.fc_matrix is None:
            self.compute_functional_connectivity()

        fc = np.abs(self.fc_matrix.copy())
        n = fc.shape[0]

        # -----------------------------
        # 3. 根据方法生成 mask（拓扑结构）
        # -----------------------------
        if method == "full":
            mask = np.ones_like(fc, dtype=bool)
            np.fill_diagonal(mask, 0)

        elif method == "threshold":
            if threshold is None:
                threshold = np.percentile(fc[np.triu_indices(n, k=1)], 90)
            mask = fc >= threshold
            np.fill_diagonal(mask, 0)

        elif method == "kNN":
            if k is None:
                raise ValueError("k must be provided for kNN method")
            mask = np.zeros_like(fc, dtype=bool)
            for i in range(n):
                inds = np.argsort(fc[i])[-k:]
                mask[i, inds] = True
            # 对称化
            mask = np.logical_or(mask, mask.T)
            np.fill_diagonal(mask, 0)

        else:
            raise ValueError(f"Unknown edge generation method: {method}")

        # -----------------------------
        # 4. 生成 edge_index
        # -----------------------------
        edge_index = np.array(np.nonzero(mask))

        # -----------------------------
        # 5. 如果要求，保存模板
        # -----------------------------
        if template_path is not None:
            # 保存为 torch.Tensor 以保持与 PyG 一致
            torch.save(torch.tensor(edge_index, dtype=torch.long), template_path)

        return edge_index


    # -------------------------
    # 转为 PyG Data
    # -------------------------
    def to_pyg(
        self,
        edge_index: Optional[Union[np.ndarray, torch.Tensor]] = None,
        edge_attr: Optional[Union[np.ndarray, torch.Tensor]] = None,
        edge_method: str = "threshold",
        threshold: Optional[float] = None,
        k: Optional[int] = None,
        node_type: str = "eeg",
        atlas_file: str = "F:/digital_twin_brain/schaefer200_mask_ready.json"
    ) -> HeteroData:
        from torch_geometric.data import HeteroData
        from scipy.spatial import cKDTree

        logger = logging.getLogger("EEGMapper")

        # --- 1. features (N, T, F) ---
        feats = self.get_node_features(with_stats=True)  # (N_eeg, T, F)
        feats = np.asarray(feats)
        if feats.ndim != 3:
            raise ValueError(f"EEG features must be 3D (N,T,F), got {feats.shape}")
        n_channels, total_times, feat_dim = feats.shape

        # --- 2. pos (N_eeg, 3) ---
        pos_list = []
        try:
            if getattr(self, "raw", None) is not None:
                montage = self.raw.get_montage()
                ch_pos_dict = montage.get_positions().get('ch_pos', {}) if montage else {}
                for ch in getattr(self, "channel_names", []):
                    p = ch_pos_dict.get(ch, [0.0, 0.0, 0.0])
                    pos_list.append([float(p[0]), float(p[1]), float(p[2])])
            else:
                pos_list = [[0.0, 0.0, 0.0]] * n_channels
        except Exception as e:
            logger.warning(f"[to_pyg] pos generation failed, using zeros: {e}")
            pos_list = [[0.0, 0.0, 0.0]] * n_channels
        pos = torch.tensor(pos_list, dtype=torch.float32)

        # --- 3. Load atlas centroids (200 ROIs) ---
        with open(atlas_file, "r") as f:
            atlas_json = json.load(f)

        if "regions" not in atlas_json:
            raise KeyError(f"[to_pyg] atlas JSON missing 'regions' field: {list(atlas_json.keys())}")

        regions = atlas_json["regions"]
        if not isinstance(regions, dict) or len(regions) == 0:
            raise ValueError("[to_pyg] 'regions' is empty or not a dict.")

        roi_names = list(regions.keys())
        atlas_centroids = []

        for name in roi_names:
            info = regions[name]
            if "position" not in info:
                raise KeyError(f"[to_pyg] ROI {name} missing 'position' field.")
            pos = info["position"]
            if not (isinstance(pos, list) and len(pos) == 3):
                raise ValueError(f"[to_pyg] ROI {name} has invalid 'position': {pos}")
            atlas_centroids.append(pos)

        atlas_centroids = np.asarray(atlas_centroids, dtype=float)  # (N_rois, 3)
        n_rois = atlas_centroids.shape[0]

        # --- 4. map EEG to ROIs using nearest-neighbor ---
        tree = cKDTree(pos_list)
        roi_features = np.zeros((n_rois, total_times, feat_dim), dtype=np.float32)
        for i, c in enumerate(atlas_centroids):
            dists, idxs = tree.query(c, k=n_channels)  # 查询所有电极
            # 使用所有电极加权平均（距离越近权重越大）
            weights = 1.0 / (dists + 1e-6)
            weights = weights / weights.sum()
            roi_features[i] = np.tensordot(weights, feats[idxs], axes=(0, 0))
        x_seq = torch.tensor(roi_features, dtype=torch.float32)          # (200, T, F)
        x = x_seq.mean(dim=1)                                           # (200, F)
        region_ids = torch.arange(n_rois, dtype=torch.long)             # [0..199]

        # --- 5. Edge index / attr ---
        if edge_index is None:
            if self.fc_matrix is None:
                self.compute_functional_connectivity()
            edge_index = self.get_edge_index(method="kNN", k=10, template_path="F:/digital_twin_brain/templates/eeg_200nodes_edgeindex.pt")
            logger.info(f"[to_pyg] Auto-generated edge_index: {edge_index.shape}")
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index).long()
        if edge_index.numel() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        if edge_attr is None:
            if self.fc_matrix is None:
                self.compute_functional_connectivity()
            if edge_index.numel() == 0:
                edge_attr = torch.empty((0,), dtype=torch.float32)
            else:
                ei_np = edge_index.cpu().numpy()
                weights = np.asarray(self.fc_matrix)[ei_np[0], ei_np[1]]
                weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                edge_attr = torch.tensor(weights, dtype=torch.float32).flatten()

        # --- 6. node_mask ---
        node_mask_arr = getattr(self, "node_mask", np.ones(n_rois, dtype=np.float32))
        node_mask = torch.tensor(np.asarray(node_mask_arr, dtype=bool), dtype=torch.bool)

        # --- 7. Build HeteroData ---
        data = HeteroData()
        data[node_type].x = x
        data[node_type].x_seq = x_seq
        data[node_type].node_mask = node_mask
        data[node_type].pos = torch.tensor(atlas_centroids, dtype=torch.float32)  # 用 ROI 质心作为 pos
        data[node_type].region_ids = region_ids

        if edge_index.numel() > 0:
            data[node_type, "connects", node_type].edge_index = edge_index
            if edge_attr is not None:
                data[node_type, "connects", node_type].edge_attr = edge_attr

        data.x_seq_dict = {node_type: data[node_type].x_seq}
        data.x_mean_dict = {node_type: data[node_type].x.mean(dim=1)}

        return data
