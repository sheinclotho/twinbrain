# mapper/dti_mapper.py
"""
DTIMapper — 高级 DTI -> 区域级 Structural Connectome (SC) 生成模块

要点：
- 支持加载 tractography (.trk/.tck via nibabel.streamlines) 或 voxel FA/MD maps（nifti）
- 支持直接使用 precomputed connectome matrix（例如 mrtrix tck2connectome 输出）
- 支持 SIFT2 权重数组（streamline-wise）或在流迹级别计数
- 支持多种校正：length-correction, region-volume-correction, log1p, symmetric normalization
- 输出:
    - sc_matrix: np.ndarray (N,N) (float32)
    - edge_index: np.ndarray (2, E)
    - edge_attr: np.ndarray (E,) or (E, K) if multiple edge features requested
- 需要 atlas 对象或至少提供:
    - atlas.labels_data : 3D int array (voxel -> region index, 0 = background)
    - atlas.affine       : affine for voxel <-> world mm coordinates
    - atlas.n_regions    : number of regions (int)
    - atlas.region_volumes (optional): (N,) region volumes in mm^3 or voxel counts
    - atlas.region_centers (optional): (N,3) mm coordinates of ROI centers (if absent, computed)

设计原则：生产可用、参数可控、与 MultiModalMapper 无缝对接。
"""
from typing import Optional, Tuple, Dict, Any, Iterable
import os
import numpy as np
import nibabel as nib
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.stats import zscore
import warnings

try:
    # nibabel.streamlines available if nibabel >= 2.0
    import nibabel.streamlines as nbs
except Exception:
    nbs = None


class DTIMapper:
    def __init__(self, atlas: Any, debug: bool = False):
        """
        atlas: object with attributes
            - labels_data: ndarray (X,Y,Z) ints, region index in [0..N-1] or [1..N] (supports either)
            - affine: (4,4) affine mapping voxel->mm
            - n_regions: int N (optional, will infer)
            - region_volumes: optional (N,) in voxels or mm^3
            - region_centers: optional (N,3) mm coordinates
        """
        if not hasattr(atlas, "labels_data") or not hasattr(atlas, "affine"):
            raise ValueError("atlas must provide labels_data (3D int array) and affine.")
        self.atlas = atlas
        self.debug = debug

        # internal storage
        self.streamlines = None       # iterable of streamlines (list-like) (ndarray (L,3) per streamline)
        self.streamline_weights = None  # optional per-streamline weights (e.g. SIFT2)
        self.connectome_matrix = None   # optional precomputed connectome (N,N)
        self.fa_img = None             # voxel-wise fallback map
        self.sc_matrix = None          # final N x N np.float32 matrix

    # -------------------------
    # Loaders
    # -------------------------
    def load_streamlines(self, tract_path: str, weights: Optional[Iterable[float]] = None):
        """
        Load tractography (trk/tck) via nibabel.streamlines.
        Optionally supply per-streamline weights (SIFT2 output) as iterable.
        """
        if nbs is None:
            raise RuntimeError("nibabel.streamlines not available. Install nibabel>=2.x")

        if not os.path.exists(tract_path):
            raise FileNotFoundError(tract_path)
        sl_obj = nbs.load(tract_path)
        self.streamlines = list(sl_obj.streamlines)
        if weights is not None:
            self.streamline_weights = np.asarray(list(weights), dtype=np.float64)
            if len(self.streamline_weights) != len(self.streamlines):
                raise ValueError("streamline weights length mismatch.")
        else:
            self.streamline_weights = None

        if self.debug:
            print(f"[DTIMapper] Loaded {len(self.streamlines)} streamlines from {tract_path}")

    def load_connectome_matrix(self, mat: np.ndarray):
        """
        Directly provide a precomputed region x region connectome matrix (un-normalized).
        """
        mat = np.asarray(mat, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError("connectome matrix must be square (N,N).")
        self.connectome_matrix = mat.copy()
        if self.debug:
            print(f"[DTIMapper] Loaded precomputed connectome matrix shape {mat.shape}")

    def load_fallback_voxel_map(self, fa_path: str):
        """
        Load voxel-level scalar map (FA/MD) for fallback SC inference.
        """
        if not os.path.exists(fa_path):
            raise FileNotFoundError(fa_path)
        self.fa_img = nib.load(fa_path).get_fdata(dtype=np.float32)
        if self.debug:
            print(f"[DTIMapper] Loaded FA-like image shape {self.fa_img.shape}")

    # -------------------------
    # Core compute
    # -------------------------
    def compute_sc(
        self,
        method: str = "tract",  # tract | precomputed | voxel
        min_streamline_len: int = 6,
        length_correction: bool = True,
        volume_correction: bool = True,
        log_transform: bool = True,
        symmetric_normalize: bool = True,
        sparsify_pct: Optional[float] = None,
        threshold: Optional[float] = None,
        enforce_symmetric: bool = True
    ) -> np.ndarray:
        """
        Compute SC matrix (N, N) according to available inputs and method.
        Options:
            - method = 'tract' : use self.streamlines (preferred)
            - method = 'precomputed' : use self.connectome_matrix
            - method = 'voxel' : use self.fa_img fallback
        Post-processing:
            - length_correction: divide streamline counts by streamline length
            - volume_correction: divide by region volumes (atlas.region_volumes if available)
            - log_transform: apply log1p
            - symmetric_normalize: D^{-1/2} * SC * D^{-1/2}
            - sparsify_pct: keep top X percentile of positive weights (global)
            - threshold: absolute cutoff on weights (after transforms)
        Returns sc_matrix (float32)
        """
        if method == "precomputed":
            if self.connectome_matrix is None:
                raise ValueError("No precomputed connectome loaded.")
            sc = self.connectome_matrix.copy().astype(np.float64)
        elif method == "tract":
            if self.streamlines is None:
                raise ValueError("No streamlines loaded for tract method.")
            sc = self._sc_from_streamlines(min_len=min_streamline_len, length_corr=length_correction)
        elif method == "voxel":
            if self.fa_img is None:
                raise ValueError("No voxel map loaded for voxel fallback.")
            sc = self._sc_from_voxel_map()
        else:
            raise ValueError(f"Unknown method: {method}")

        # symmetry
        if enforce_symmetric:
            sc = 0.5 * (sc + sc.T)

        # optional volume correction
        if volume_correction:
            sc = self._apply_volume_correction(sc)

        # log transform (compress dynamic range)
        if log_transform:
            sc = np.log1p(sc)

        # sparsify (global percentile)
        if sparsify_pct is not None and 0.0 < sparsify_pct < 100.0:
            sc = self._sparsify_by_percentile(sc, sparsify_pct)

        # absolute threshold
        if threshold is not None:
            sc[sc < threshold] = 0.0

        # final normalization (symmetric)
        if symmetric_normalize:
            sc = self._symmetric_normalize(sc)

        # cast
        self.sc_matrix = sc.astype(np.float32)
        if self.debug:
            nnz = np.count_nonzero(self.sc_matrix)
            N = self.sc_matrix.shape[0]
            print(f"[DTIMapper] SC computed: shape={self.sc_matrix.shape}, nnz={nnz}, density={nnz/(N*N):.4f}")
        return self.sc_matrix

    # -------------------------
    # Streamline -> region mapping
    # -------------------------
    def _sc_from_streamlines(self, min_len: int = 6, length_corr: bool = True) -> np.ndarray:
        """
        Map streamline endpoints to atlas region indices; accumulate counts or weighted counts.
        Optionally apply length correction: weight = w_raw / length (mm)
        """
        labels = self.atlas.labels_data
        affine = self.atlas.affine
        N = getattr(self.atlas, "n_regions", None) or int(labels.max() + 1)
        sc = np.zeros((N, N), dtype=np.float64)

        # precompute region valid mask
        shape = labels.shape

        # iterate streamlines
        for idx, sl in enumerate(self.streamlines):
            if sl is None or len(sl) < min_len:
                continue
            # endpoints in mm (world)
            p0_mm = sl[0]
            p1_mm = sl[-1]

            # world->voxel (affine inverse)
            try:
                p0_vox = np.round(np.linalg.inv(affine).dot(np.append(p0_mm, 1)))[:3].astype(int)
                p1_vox = np.round(np.linalg.inv(affine).dot(np.append(p1_mm, 1)))[:3].astype(int)
            except Exception:
                continue

            # bounds check
            if (p0_vox < 0).any() or (p1_vox < 0).any():
                continue
            if (p0_vox >= shape).any() or (p1_vox >= shape).any():
                continue

            r0 = int(labels[tuple(p0_vox)])
            r1 = int(labels[tuple(p1_vox)])
            # skip unlabeled / same-region trivial
            if r0 <= 0 or r1 <= 0 or r0 == r1:
                continue
            # normalize to 0-based indexing if atlas is 1-based
            if labels.min() == 1:
                r0 -= 1
                r1 -= 1

            # compute base weight: either provided streamline weight or 1
            w = 1.0
            if self.streamline_weights is not None:
                w = float(self.streamline_weights[idx])
            # length correction: divide by streamline length in mm
            if length_corr:
                lengths = np.linalg.norm(sl[1:] - sl[:-1], axis=1)
                total_len = np.sum(lengths)
                if total_len > 0:
                    w = w / total_len

            sc[r0, r1] += w
            sc[r1, r0] += w

        return sc

    # -------------------------
    # Voxel fallback
    # -------------------------
    def _sc_from_voxel_map(self) -> np.ndarray:
        """
        Use voxel-level scalar map (FA) to synthesize inter-region weights.
        Simple rule: w_ij = (FA_i + FA_j) / (2 * dist_ij)
        """
        labels = self.atlas.labels_data
        N = getattr(self.atlas, "n_regions", None) or int(labels.max() + 1)
        centers = self._get_region_centers()
        region_fa = np.zeros(N, dtype=np.float64)
        for r in range(N):
            mask = labels == r if labels.min() == 0 else labels == (r + 1)
            if mask.sum() == 0:
                region_fa[r] = 0.0
            else:
                region_fa[r] = np.nanmean(self.fa_img[mask])
        # compute pairwise weights
        dists = cdist(centers, centers)
        eps = 1e-12
        sc = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                if dists[i, j] < eps:
                    w = 0.0
                else:
                    w = (region_fa[i] + region_fa[j]) / (2.0 * dists[i, j])
                sc[i, j] = w
                sc[j, i] = w
        return sc

    # -------------------------
    # Utilities: centers, normalization, sparsify
    # -------------------------
    def _get_region_centers(self) -> np.ndarray:
        """
        Return (N,3) centers in mm. Use atlas.region_centers if available, else compute.
        """
        if hasattr(self.atlas, "region_centers") and getattr(self.atlas, "region_centers") is not None:
            centers = np.asarray(self.atlas.region_centers, dtype=np.float32)
            return centers
        # compute
        labels = self.atlas.labels_data
        affine = self.atlas.affine
        N = getattr(self.atlas, "n_regions", None) or int(labels.max() + 1)
        centers = np.zeros((N, 3), dtype=np.float32)
        for r in range(N):
            mask = labels == r if labels.min() == 0 else labels == (r + 1)
            vox = np.argwhere(mask)
            if vox.shape[0] == 0:
                centers[r] = np.array([np.nan, np.nan, np.nan])
            else:
                mm = nib.affines.apply_affine(affine, vox)
                centers[r] = mm.mean(axis=0)
        return centers

    def _apply_volume_correction(self, sc: np.ndarray) -> np.ndarray:
        """
        Optionally divide weights by product of region volumes to reduce bias for large regions.
        atlas.region_volumes should be provided (voxels or mm^3). If absent, compute from labels.
        """
        N = sc.shape[0]
        if hasattr(self.atlas, "region_volumes") and self.atlas.region_volumes is not None:
            vols = np.asarray(self.atlas.region_volumes, dtype=np.float64)
            if len(vols) != N:
                vols = np.ones(N, dtype=np.float64)
        else:
            # compute counts
            labels = self.atlas.labels_data
            vols = np.zeros(N, dtype=np.float64)
            for r in range(N):
                mask = labels == r if labels.min() == 0 else labels == (r + 1)
                vols[r] = mask.sum() + 1e-12
        vol_mat = np.outer(vols, vols)
        # avoid division by zero
        sc_corr = sc / (vol_mat + 1e-12)
        return sc_corr

    def _sparsify_by_percentile(self, sc: np.ndarray, pct: float) -> np.ndarray:
        """
        Keep top pct percent of positive weights (global). pct in (0,100).
        """
        assert 0.0 < pct < 100.0
        vals = sc[np.triu_indices(sc.shape[0], k=1)]
        pos = vals[vals > 0]
        if pos.size == 0:
            return sc
        thr = np.percentile(pos, 100.0 - pct)
        sc_masked = sc.copy()
        sc_masked[sc_masked < thr] = 0.0
        return sc_masked

    def _symmetric_normalize(self, sc: np.ndarray) -> np.ndarray:
        """
        Symmetric normalization: D^{-1/2} * SC * D^{-1/2}
        """
        sc = sc.copy()
        sc[sc < 0] = 0.0
        deg = sc.sum(axis=1)
        deg_safe = np.where(deg <= 0, 1.0, deg)
        inv_sqrt = 1.0 / np.sqrt(deg_safe)
        Dinv = np.diag(inv_sqrt)
        sc_norm = Dinv @ sc @ Dinv
        return sc_norm

    # -------------------------
    # Export helpers
    # -------------------------
    def get_edges(self, threshold: float = 0.0, as_sparse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        From self.sc_matrix produce edge_index (2,E) and edge_attr (E,) or SciPy sparse if requested.
        threshold: cutoff on sc value (applied after compute_sc).
        as_sparse: if True, return scipy.sparse.coo_matrix
        """
        if self.sc_matrix is None:
            raise ValueError("SC matrix not computed. Call compute_sc() first.")
        sc = self.sc_matrix.copy()
        if threshold > 0:
            sc[sc < threshold] = 0.0
        # remove self edges
        np.fill_diagonal(sc, 0.0)
        rows, cols = np.nonzero(sc)
        edge_index = np.vstack([rows, cols]).astype(np.int64)
        edge_attr = sc[rows, cols].astype(np.float32)
        if as_sparse:
            coo = sparse.coo_matrix(sc)
            return coo, coo.data
        return edge_index, edge_attr

    def save_sc(self, out_path: str):
        """
        Save sc_matrix as npy
        """
        if self.sc_matrix is None:
            raise ValueError("SC not computed.")
        np.save(out_path, self.sc_matrix)
        if self.debug:
            print(f"[DTIMapper] SC saved to {out_path}")

    def load_sc(self, npy_path: str):
        """
        Load precomputed sc matrix from npy
        """
        if not os.path.exists(npy_path):
            raise FileNotFoundError(npy_path)
        sc = np.load(npy_path)
        if sc.ndim != 2 or sc.shape[0] != sc.shape[1]:
            raise ValueError("Invalid SC matrix in file.")
        self.sc_matrix = sc.astype(np.float32)
        if self.debug:
            print(f"[DTIMapper] SC loaded from {npy_path}, shape={self.sc_matrix.shape}")
        return self.sc_matrix
