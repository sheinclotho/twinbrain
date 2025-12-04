"""
train/embed_analysis.py

Load one or more embedding files produced by embed_utils.save_embeddings and provide:
 - NN identification (LOO) at sample or aggregated subject level
 - pairwise cosine similarity matrices and heatmap plotting
 - UMAP / t-SNE visualization of combined embeddings
 - simple clustering metrics (silhouette, db index) if sklearn available

Primary entrypoint:
  analyze_embedding_files(paths: List[str], out_dir: str, aggregate_by_subject: bool = True, vis_method='umap')

Produces:
 - JSON summary saved to out_dir/summary.json
 - Plots (umap.png, sim_heatmap.png)
 - Optionally returns Python dict of results
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# On Windows loky/joblib may try to probe physical cores using subprocesses and raise warnings.
# Set LOKY_MAX_CPU_COUNT before any joblib/loky/sklearn imports to silence the warning and
# provide a safe default that uses logical CPUs.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

def _load_npz(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out = {}
    for k in data.files:
        out[k] = data[k]
    return out

def _cosine_sim(X: np.ndarray) -> np.ndarray:
    # rows are samples
    Xc = X - X.mean(axis=0, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    return S

def aggregate_embeddings_from_files(paths: List[str], aggregate_by_subject: bool = True) -> Tuple[np.ndarray, List[Any], List[int]]:
    """
    Load embeddings from multiple npz files and aggregate into matrix X (n_samples, D).
    If aggregate_by_subject True, aggregates multiple samples per subject by mean and returns subject-level X.
    Returns:
      X, labels (subject ids), sample_indices (list mapping rows to source (file,index) tuple encoded as int)
    """
    recs = []
    metas = []
    for p in paths:
        d = _load_npz(p)
        emb = d.get("embeddings")
        sids = d.get("subject_ids")
        if emb is None or sids is None:
            continue
        # subject_ids may be stored as object dtype; ensure list
        if isinstance(sids, np.ndarray) and sids.dtype == object:
            sids_list = sids.tolist()
        else:
            sids_list = list(sids)
        for i in range(emb.shape[0]):
            recs.append({"emb": emb[i], "subject_id": sids_list[i] if i < len(sids_list) else None, "source": p, "index": i})
    if len(recs) == 0:
        return np.zeros((0,0)), [], []

    # aggregate by subject if requested
    if aggregate_by_subject:
        buckets = {}
        for r in recs:
            sid = r["subject_id"] if r["subject_id"] is not None else f"src:{os.path.basename(r['source'])}_idx{r['index']}"
            buckets.setdefault(sid, []).append(r["emb"])
        keys = list(buckets.keys())
        X = np.vstack([np.mean(np.vstack(buckets[k]), axis=0) for k in keys])
        labels = keys
        sample_idx = list(range(X.shape[0]))
    else:
        X = np.vstack([r["emb"] for r in recs])
        labels = [r["subject_id"] for r in recs]
        sample_idx = list(range(X.shape[0]))
    return X, labels, sample_idx

def nn_identification(X: np.ndarray, labels: List[Any]) -> Dict[str, Any]:
    """
    Leave-one-out nearest neighbor identification accuracy.
    """
    if X.shape[0] < 2:
        return {"accuracy": None}
    S = _cosine_sim(X)
    correct = []
    for i in range(S.shape[0]):
        sim = S[i].copy()
        sim[i] = -np.inf
        j = int(np.argmax(sim))
        correct.append(1 if labels[i] == labels[j] else 0)
    return {"accuracy": float(np.mean(correct)), "per_sample": correct}

def visualize_umap(X: np.ndarray, labels: List[Any], out_path: str, method: str = "umap", random_state: int = 42):
    try:
        import umap
        have_umap = True
    except Exception:
        have_umap = False

    n_samples = X.shape[0]
    # For very small n, UMAP/TSNE may be unstable; fallback to PCA visualization
    if n_samples < 6:
        # PCA fallback
        try:
            from sklearn.decomposition import PCA
            X2 = PCA(n_components=2).fit_transform(X)
        except Exception:
            # final fallback: use first two dims (if available) or zeros
            if X.shape[1] >= 2:
                X2 = X[:, :2]
            else:
                X2 = np.hstack([X, np.zeros((X.shape[0], 2 - X.shape[1]))])[:, :2]
    else:
        if method == "umap" and have_umap:
            reducer = umap.UMAP(n_neighbors=min(15, max(2, n_samples//3)), min_dist=0.1, random_state=random_state)
            X2 = reducer.fit_transform(X)
        else:
            from sklearn.manifold import TSNE
            # choose a safe perplexity: <= (n_samples-1)//3, at least 2
            max_perp = max(2, (n_samples - 1) // 3)
            perp = min(30, max_perp)
            ts = TSNE(n_components=2, random_state=random_state, init="pca", perplexity=perp)
            X2 = ts.fit_transform(X)

    plt.figure(figsize=(8,6))
    try:
        uniq, inv = np.unique(labels, return_inverse=True)
        sc = plt.scatter(X2[:,0], X2[:,1], c=inv, cmap="tab10", s=40)
    except Exception:
        plt.scatter(X2[:,0], X2[:,1], s=40)
    plt.title("Embedding visualization")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return X2

def plot_similarity_heatmap(S: np.ndarray, labels: List[Any], out_path: str):
    # order by label if categorical
    try:
        uniq, inv = np.unique(labels, return_inverse=True)
        order = np.argsort(inv * S.shape[0] + np.arange(S.shape[0]))
    except Exception:
        order = np.arange(S.shape[0])
    S_ord = S[order][:, order]
    plt.figure(figsize=(8,6))
    sns.heatmap(S_ord, cmap="vlag", center=0)
    plt.title("Pairwise cosine similarity (ordered)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def analyze_embedding_files(paths: List[str], out_dir: str, aggregate_by_subject: bool = True, vis_method: str = "umap") -> Dict[str, Any]:
    """
    High-level runner: loads multiple embedding files, aggregates, runs NN-id, visualizes and saves summary.
    Returns a dict with results and paths of saved artifacts.
    """
    os.makedirs(out_dir, exist_ok=True)
    X, labels, sample_idx = aggregate_embeddings_from_files(paths, aggregate_by_subject=aggregate_by_subject)
    res = {"n_rows": int(X.shape[0]), "aggregate_by_subject": bool(aggregate_by_subject), "files": paths}
    if X.size == 0:
        res["error"] = "no embeddings found"
        return res

    # add diagnostics about labels
    try:
        unique_labels = list(np.unique(labels))
        res["n_unique_labels"] = int(len(unique_labels))
        # count per-label occurrences (use python counts)
        from collections import Counter
        cnt = Counter(labels)
        res["label_counts_sample"] = {str(k): int(v) for k, v in cnt.items()}
    except Exception:
        res["n_unique_labels"] = None
        res["label_counts_sample"] = None

    # NN identification: only meaningful when some labels have >=2 samples
    do_nn = False
    if res["label_counts_sample"] is not None:
        # check if any label occurs more than once
        if any(v >= 2 for v in res["label_counts_sample"].values()):
            do_nn = True

    if do_nn:
        nn_res = nn_identification(X, labels)
        res["nn_ident"] = nn_res
    else:
        res["nn_ident"] = {"accuracy": None, "reason": "no label has >=2 samples; set aggregate_by_subject=False or provide repeated samples per subject"}

    # similarity matrix
    S = _cosine_sim(X)
    sim_path = os.path.join(out_dir, "similarity_heatmap.png")
    plot_similarity_heatmap(S, labels, sim_path)
    res["similarity_heatmap"] = sim_path

    # UMAP / TSNE (robust to small n)
    vis_path = os.path.join(out_dir, "embeddings_vis.png")
    try:
        visualize_umap(X, labels, vis_path, method=vis_method)
        res["vis"] = vis_path
    except Exception as e:
        res["vis_error"] = str(e)

    # clustering metrics (silhouette) if sklearn present
    try:
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        if Xs.shape[0] > 2:
            sil = float(silhouette_score(Xs, np.arange(Xs.shape[0]) % max(2, min(8, Xs.shape[0]))))
            db = float(davies_bouldin_score(Xs, np.arange(Xs.shape[0]) % max(2, min(8, Xs.shape[0]))))
            res["cluster_silhouette"] = sil
            res["cluster_davies_bouldin"] = db
    except Exception:
        pass

    # save summary
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(res, f, indent=2)

    return res