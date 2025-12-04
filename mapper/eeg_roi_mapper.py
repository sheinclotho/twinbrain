import numpy as np
import torch
import json
from pathlib import Path
import logging

logger = logging.getLogger("EEG-ROI-Mapper")

_atlas_centroids_cache = None

def _generate_atlas_centroids(atlas_file: str):
    """
    读取 Schaefer atlas JSON 文件并生成每个 ROI 的质心坐标
    atlas_file: JSON 文件路径，假设格式为 { "ROI_name": [[x,y,z], [x,y,z], ...], ... }
    返回: numpy array, shape=(num_rois, 3)
    """
    global _atlas_centroids_cache
    if _atlas_centroids_cache is not None:
        return _atlas_centroids_cache

    atlas_path = Path(atlas_file)
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_file}")

    with open(atlas_file, "r") as f:
        atlas_data = json.load(f)

    centroids = []
    for roi_name, vertices in atlas_data.items():
        vertices = np.array(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            logger.warning(f"[EEG-ROI-Mapper] ROI {roi_name} vertices shape invalid: {vertices.shape}, skipping")
            continue
        centroid = vertices.mean(axis=0)
        centroids.append(centroid)

    if len(centroids) == 0:
        raise ValueError("No valid ROI centroids generated from atlas")

    _atlas_centroids_cache = np.stack(centroids, axis=0)
    logger.info(f"[EEG-ROI-Mapper] Generated atlas centroids: {_atlas_centroids_cache.shape}")
    return _atlas_centroids_cache


def map_eeg_to_roi(eeg_pos: np.ndarray, atlas_file: "F:/digital_twin_brain/schaefer200_mask_ready.json"):
    """
    输入:
        eeg_pos: numpy array or torch.Tensor, shape=(N,3) 电极坐标
        atlas_file: schaefer atlas JSON 文件
    输出:
        region_ids: numpy array, shape=(N,) 每个电极映射到的 ROI 索引
    """
    if isinstance(eeg_pos, torch.Tensor):
        eeg_pos = eeg_pos.cpu().numpy()

    if eeg_pos.ndim != 2 or eeg_pos.shape[1] != 3:
        raise ValueError(f"eeg_pos must be (N,3), got {eeg_pos.shape}")

    try:
        atlas_centroids = _generate_atlas_centroids(atlas_file)

        # 欧氏距离最近 ROI
        dists = np.linalg.norm(eeg_pos[:, None, :] - atlas_centroids[None, :, :], axis=-1)
        region_ids = np.argmin(dists, axis=1)
        return region_ids.astype(np.int64)

    except Exception as e:
        logger.warning(f"[EEG-ROI-Mapper] mapping failed, returning -1: {e}")
        return np.full((eeg_pos.shape[0],), -1, dtype=np.int64)
