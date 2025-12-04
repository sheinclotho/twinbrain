import numpy as np
from typing import List, Optional, Literal
from meta_node import MetaNode
from mapper.atlas_mapper import BrainAtlas
from utils.utils import add_gaussian_noise
import nibabel as nib
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_atlas_img(atlas_img_path: Optional[str] = "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii", atlas_img: Optional[nib.Nifti1Image] = None):
    """加载 atlas 文件并缓存"""
    if atlas_img is not None:
        return {"img": atlas_img, "affine": atlas_img.affine, "shape": atlas_img.shape[:3]}

    if not hasattr(_get_atlas_img, "_cache"):
        import tkinter as tk
        from tkinter import filedialog
        if atlas_img_path is None:
            root = tk.Tk()
            root.withdraw()
            atlas_img_path = filedialog.askopenfilename(
                title="请选择 NIfTI atlas 文件",
                filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")]
            )
            if not atlas_img_path:
                raise ValueError("未选择 atlas 文件，无法生成 MNI 坐标。")
            root.destroy()

        atlas_img = nib.load(atlas_img_path)
        _get_atlas_img._cache = {
            "img": atlas_img,
            "affine": atlas_img.affine,
            "shape": atlas_img.shape[:3],
            "path": atlas_img_path
        }
        logger.info(f"使用 atlas 文件: {atlas_img_path}")
        logger.info(f"atlas image shape: {atlas_img.shape[:3]}, affine:\n{atlas_img.affine}")
    return _get_atlas_img._cache


def generate_nodes(
    meta: MetaNode,
    atlas: BrainAtlas,
    n: int,
    region_id: str,
    noise_std: float = 1.0,
    atlas_img_path: Optional[str] = "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii",
    atlas_img: Optional[nib.Nifti1Image] = None,
    mode: Literal["point", "mask"] = "mask",
) -> List[dict]:
    """生成单个 region 的节点"""
    cache = _get_atlas_img(atlas_img_path, atlas_img)
    atlas_data = cache["img"].get_fdata()
    affine = cache["affine"]
    shape = np.array(cache["shape"], dtype=float)

    if region_id not in atlas.regions:
        raise ValueError(f"Unknown region: {region_id}")

    nodes = []
    meta_template = meta.get_data()

    if mode == "point":
        raw_pos = np.array(atlas.regions[region_id]['position'], dtype=float)
        # 坐标标准化
        if np.all((raw_pos >= 0) & (raw_pos <= 1.0)):
            voxel_coord = raw_pos * (shape - 1.0)
        else:
            voxel_coord = raw_pos
        base_mni_coord = nib.affines.apply_affine(affine, voxel_coord)

        for i in range(n):
            mni_coord = add_gaussian_noise(base_mni_coord, std=noise_std)
            node_data = meta_template.copy()
            node_data.update({
                "node_id": f"{region_id}_node_{i}",
                "position": mni_coord.tolist(),
                "position_3d": mni_coord.tolist(),
                "region": region_id
            })
            nodes.append(node_data)

    elif mode == "mask":
        if "label_id" not in atlas.regions[region_id]:
            raise ValueError(f"region {region_id} 缺少 label_id，无法在 atlas mask 中采样。")

        label_id = atlas.regions[region_id]['label_id']
        voxel_coords = np.argwhere(atlas_data == label_id)
        if len(voxel_coords) == 0:
            raise ValueError(f"在 atlas 中找不到 region {region_id} (label={label_id}) 的体素。")

        if n > len(voxel_coords):
            logger.warning(f"节点数 n={n} 大于可用体素数 {len(voxel_coords)}，将重复采样")

        sampled_indices = np.random.choice(len(voxel_coords), n, replace=True)
        sampled_voxels = voxel_coords[sampled_indices]
        base_mni_coords = nib.affines.apply_affine(affine, sampled_voxels)

        for i in range(n):
            mni_coord = add_gaussian_noise(base_mni_coords[i], std=noise_std)
            node_data = meta_template.copy()
            node_data.update({
                "node_id": f"{region_id}_node_{i}",
                "position": mni_coord.tolist(),
                "position_3d": mni_coord.tolist(),
                "region": region_id
            })
            nodes.append(node_data)
    else:
        raise ValueError(f"未知 mode={mode}")

    return nodes


def generate_nodes_all_regions(
    meta: MetaNode,
    atlas: BrainAtlas,
    nodes_per_region: int,
    noise_std: float = 1.0,
    atlas_img_path: Optional[str] = "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii",
    atlas_img: Optional[nib.Nifti1Image] = None,
    mode: Literal["point", "mask"] = "mask"
) -> List[dict]:
    """为 atlas 中所有 region 生成节点，显示统一总进度条"""
    all_nodes = []
    region_ids = list(atlas.regions.keys())
    logger.info(f"开始生成 {len(region_ids)} 个脑区节点 (mode={mode})")

    for region_id in tqdm(region_ids, desc="Regions", unit="region", ncols=100):
        nodes = generate_nodes(
            meta,
            atlas,
            n=nodes_per_region,
            region_id=region_id,
            noise_std=noise_std,
            atlas_img_path=atlas_img_path,
            atlas_img=atlas_img,
            mode=mode
        )
        all_nodes.extend(nodes)
        logger.info(f"{region_id} 生成 {len(nodes)} 个节点")

    logger.info(f"所有脑区节点生成完成，总节点数: {len(all_nodes)}")
    return all_nodes
