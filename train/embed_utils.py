"""
train/embed_utils.py

Utilities to extract and persist embeddings from DynamicHeteroTrainer / DynamicHeteroGNN.

Enhanced: save_embeddings now accepts `subject_ids_override` so you can supply
subject identifiers at save time (useful when graphs don't contain subject_id).
This avoids having subject_ids saved as `None` in the .npz files.

Primary function:
  save_embeddings(trainer, graphs, out_path, agg='global', node_type=None,
                  max_graphs=None, save_raw=False, subject_ids_override=None, subject_id_attr=None)

 - subject_ids_override: optional sequence with same length as graphs. If provided,
   these values will be saved as subject_ids (overrides any value found on graph).
 - subject_id_attr: optional str name to look up on each graph (e.g. 'subject_id', 'subj').
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import json
import numpy as np
import torch
import datetime

def _guess_subject_id(graph, subject_id_attr: Optional[str] = None) -> Optional[Any]:
    # Priority:
    # 1) if subject_id_attr provided, try it
    # 2) try common attribute names
    if subject_id_attr is not None and hasattr(graph, subject_id_attr):
        try:
            return getattr(graph, subject_id_attr)
        except Exception:
            pass
    for attr in ("subject_id", "subj", "sid", "y", "subject"):
        if hasattr(graph, attr):
            try:
                val = getattr(graph, attr)
                return int(val) if isinstance(val, (int, np.integer)) else val
            except Exception:
                return getattr(graph, attr)
    return None

def save_embeddings(
    trainer,
    graphs: Sequence[Any],
    out_path: str,
    agg: str = "global",
    node_type: Optional[str] = None,
    max_graphs: Optional[int] = None,
    save_raw: bool = False,
    device: Optional[torch.device] = None,
    subject_ids_override: Optional[Sequence[Any]] = None,
    subject_id_attr: Optional[str] = None,
):
    """
    Extract embeddings from trainer.model for each graph and save to a single .npz file.

    Args:
      trainer: DynamicHeteroTrainer instance (must have model, _prepare_x_dict)
      graphs: iterable of HeteroData (e.g., trainer.data_list or subset)
      out_path: path to output .npz (will overwrite)
      agg: which embedding to extract: "global" | "z_mean" | "proj_mean"
      node_type: when using z_mean or proj_mean, choose modality (default first available)
      max_graphs: limit how many graphs to process (None -> all)
      save_raw: if True, include raw proj_seq / z_dict entries (can be big)
      device: torch.device override (defaults to trainer.device)
      subject_ids_override: optional sequence (len == len(graphs) or <= max_graphs) to force subject ids
      subject_id_attr: optional attribute name to read subject id from each graph
    Result:
      Writes out_path (npz) containing:
        - embeddings: np.ndarray (n_samples, D)
        - subject_ids: object array
        - task_ids: object array
        - meta: JSON string with arguments and timestamp
        - optionally: raw_proj_seq_{i}_{nt}, raw_z_{i}_{nt}, raw_global_{i}
    """
    device = device or getattr(trainer, "device", torch.device("cpu"))
    model = trainer.model
    model.eval()

    graphs = list(graphs)
    if max_graphs is not None:
        graphs = graphs[:max_graphs]

    if subject_ids_override is not None:
        if len(subject_ids_override) < len(graphs):
            raise ValueError("subject_ids_override length < number of graphs to save")
        # keep a list aligned with graphs
        subj_over = list(subject_ids_override)
    else:
        subj_over = None

    embeddings: List[np.ndarray] = []
    subject_ids: List[Any] = []
    task_ids: List[Any] = []
    extra_raw: Dict[str, np.ndarray] = {}

    for i, g in enumerate(graphs):
        g = g.to(device)
        x_dict = trainer._prepare_x_dict(g)
        with torch.no_grad():
            outputs = model(x_dict, g.edge_index_dict)

        # Normalize expected unpacking
        if isinstance(outputs, tuple) and len(outputs) == 6:
            z_dict, gru_out, proj_seq_dict, recon_seq_dict, global_seq, attn = outputs
        elif isinstance(outputs, tuple) and len(outputs) == 5:
            z_dict, gru_out, proj_seq_dict, global_seq, attn = outputs
        else:
            raise RuntimeError("Unexpected model output format when extracting embeddings")

        # subject/task ids: override > graph attribute > None
        if subj_over is not None:
            sid = subj_over[i]
        else:
            sid = _guess_subject_id(g, subject_id_attr)
        subject_ids.append(sid)
        tid = getattr(g, "task_id", None)
        task_ids.append(tid)

        # choose embedding
        if agg == "global" and isinstance(global_seq, torch.Tensor):
            emb = global_seq.detach().cpu().numpy().reshape(-1)
        elif agg == "z_mean" and isinstance(z_dict, dict) and len(z_dict) > 0:
            nt_choice = node_type or next(iter(z_dict.keys()))
            z = z_dict.get(nt_choice)
            if z is None:
                raise RuntimeError(f"z_dict missing modality {nt_choice}")
            emb = z.mean(dim=0).detach().cpu().numpy().reshape(-1)
        elif agg == "proj_mean" and isinstance(proj_seq_dict, dict) and len(proj_seq_dict) > 0:
            nt_choice = node_type or next(iter(proj_seq_dict.keys()))
            p = proj_seq_dict.get(nt_choice)
            if p is None:
                raise RuntimeError(f"proj_seq_dict missing modality {nt_choice}")
            emb = p.mean(dim=(0, 1)).detach().cpu().numpy().reshape(-1)
        else:
            # fallback: zero vector sized hidden_dim
            D = getattr(model, "hidden_dim", 128)
            emb = np.zeros((D,), dtype=np.float32)

        embeddings.append(emb.astype(np.float32))

        # optionally store raw latents (useful for offline analysis)
        if save_raw:
            for nt, p in (proj_seq_dict.items() if isinstance(proj_seq_dict, dict) else []):
                key = f"raw_proj_{i}__{nt}"
                extra_raw[key] = p.detach().cpu().numpy()
            for nt, z in (z_dict.items() if isinstance(z_dict, dict) else []):
                key = f"raw_z_{i}__{nt}"
                extra_raw[key] = z.detach().cpu().numpy()
            if isinstance(global_seq, torch.Tensor):
                extra_raw[f"raw_global_{i}"] = global_seq.detach().cpu().numpy()

    embeddings_arr = np.vstack(embeddings) if len(embeddings) > 0 else np.zeros((0, 0), dtype=np.float32)

    meta = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "n_samples": len(embeddings),
        "agg": agg,
        "node_type": node_type,
        "save_raw": bool(save_raw),
        "subject_id_attr": subject_id_attr,
    }

    # ensure output dir
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # compose save dict
    save_dict = {
        "embeddings": embeddings_arr,
        "subject_ids": np.array(subject_ids, dtype=object),
        "task_ids": np.array(task_ids, dtype=object),
        "meta": json.dumps(meta),
    }
    # merge extra_raw (if any)
    save_dict.update(extra_raw)

    # save as npz (compressed)
    np.savez_compressed(out_path, **save_dict)
    return out_path