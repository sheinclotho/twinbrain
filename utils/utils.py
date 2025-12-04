import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric.data
from torch_geometric.data import Data, HeteroData
from mpl_toolkits.mplot3d import Axes3D
from nilearn import plotting
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity  
from meta_node import MetaNode
from torch_geometric.utils import to_networkx
import matplotlib as mpl
import logging
from pathlib import Path
import torch.nn.functional as F
import plotly.graph_objects as go
import nibabel as nib
from scipy.ndimage import center_of_mass, gaussian_filter1d
plt.rcParams['font.family'] = 'Arial'  
plt.rcParams['axes.unicode_minus'] = False   

def add_gaussian_noise(features: np.ndarray, std: float = 0.1) -> np.ndarray:
    """
    添加高斯噪声模拟生物变异。
    features: 节点特征向量 (e.g., [position, weight])
    """
    noise = np.random.normal(0, std, features.shape)
    return features + noise

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """numpy实现余弦相似度。"""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 * norm2 != 0 else 0.0

def compute_euclidean_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """欧氏距离。"""
    return np.linalg.norm(pos1 - pos2)


# -----------------------------
# NetworkX 2D 可视化
# -----------------------------
def visualize_graph(data, save_path=None):
    """
    通用异构图可视化。
    输入: HeteroData 对象
    输出: 网络结构图
    """
    if not isinstance(data, torch.nn.Module) and hasattr(data, "edge_index_dict"):
        # HeteroData 情况
        G = nx.Graph()
        for (src_type, rel_type, dst_type), edge_index in data.edge_index_dict.items():
            edges = edge_index.t().cpu().numpy()
            G.add_edges_from(
                [(f"{src_type}_{u}", f"{dst_type}_{v}") for u, v in edges],
                relation=rel_type
            )

        pos = nx.spring_layout(G, seed=0)
        edge_colors = [hash(d["relation"]) % 10 for _, _, d in G.edges(data=True)]
        nx.draw(G, pos,
                with_labels=False,
                node_size=60,
                edge_color=edge_colors,
                width=0.5,
                cmap=plt.cm.tab10)
    else:
        # 回退: 传统 networkx.Graph
        G = data
        pos = nx.spring_layout(G, seed=0)
        nx.draw(G, pos,
                with_labels=False,
                node_size=60,
                width=0.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



# -----------------------------
# NetworkX -> PyG Data
# -----------------------------
def nx_to_pyg_data(G: nx.Graph) -> HeteroData:
    data = HeteroData()
    for node_type, attrs in nx.get_node_attributes(G, "type").items():
        mask = [n for n, d in G.nodes(data=True) if d["type"] == node_type]
        data[node_type].x = torch.tensor(
            [G.nodes[n]["feature"] for n in mask], dtype=torch.float32
        )
    for edge_type in set(G.edges[e]["type"] for e in G.edges):
        edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == edge_type]
        if edges:
            data[edge_type].edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return data


# -----------------------------
# 3D 节点激活可视化（使用真实空间坐标）
# -----------------------------
def visualize_node_activations(data, predictions, save_path):
    """
    兼容 HeteroData 的节点激活可视化。
    data: HeteroData
    predictions: dict[str, np.ndarray] 或 np.ndarray
    save_path: 输出路径
    """
    # ======== Step 1. 兼容 HeteroData ========
    if hasattr(data, "edge_types"):  # torch_geometric.data.HeteroData
        G = nx.Graph()
        for ntype in data.node_types:
            num_nodes = data[ntype].num_nodes
            for i in range(num_nodes):
                G.add_node(f"{ntype}_{i}", ntype=ntype)

        for edge_type in data.edge_types:
            src, _, dst = edge_type
            edge_index = data[edge_type].edge_index.cpu().numpy()
            for s, t in edge_index.T:
                G.add_edge(f"{src}_{s}", f"{dst}_{t}", etype=edge_type[1])
    elif isinstance(data, nx.Graph):
        G = data
    else:
        raise TypeError(f"Unsupported type: {type(data)}")

    # ======== Step 2. 节点颜色映射 ========
    if isinstance(predictions, dict):
        vals = []
        for k, v in predictions.items():
            if isinstance(v, np.ndarray):
                v = v.flatten()
            vals.extend(v)
        node_colors = np.array(vals[:len(G.nodes())])
    else:
        node_colors = np.array(predictions).flatten()[:len(G.nodes())]

    if len(node_colors) < len(G.nodes()):
        pad = len(G.nodes()) - len(node_colors)
        node_colors = np.pad(node_colors, (0, pad), mode='constant')

    # ======== Step 3. 绘图 ========
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, cmap=plt.cm.coolwarm, node_size=80, ax=ax
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)

    sm = mpl.cm.ScalarMappable(
        cmap=plt.cm.coolwarm,
        norm=mpl.colors.Normalize(vmin=node_colors.min(), vmax=node_colors.max())
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Node activation")
    plt.title("Heterogeneous Graph Node Activations")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved node activation visualization to {save_path}")



def visualize_brain_connectome(
    hetero_or_matrix,
    predictions=None,
    edge_type=None,
    top_k=200,
    threshold=None,
    html_path=None,
    title=None,
    atlas_nii_path: str = None
):
    """
    交互式 connectome（HTML），支持 HeteroData / ndarray / tensor 输入，并可通过 Schaefer200 NIfTI atlas 文件计算 ROI 坐标。

    参数
    - hetero_or_matrix: HeteroData 或 np.ndarray（二阶连接矩阵）或 torch.Tensor
    - predictions: 可选，dict 或 np.ndarray，用于节点着色
    - edge_type: 可选，指定 edge type tuple, e.g. ('fmri','connects','fmri')
    - top_k: 显示权重最大的 top_k 条边
    - threshold: 若提供，用于过滤低于阈值的边
    - html_path: 输出 HTML 路径
    - title: 图标题
    - atlas_nii_path: 可选，Schaefer200 NIfTI 文件路径，用于计算 ROI 坐标

    返回
    - fig: plotly Figure（也会在 html_path 保存）
    """

    # ===== 1) 从 HeteroData 提取连接矩阵与节点信息 =====
    if isinstance(hetero_or_matrix, HeteroData):
        data = hetero_or_matrix
        if edge_type is None:
            candidates = [et for et in data.edge_types if ('connect' in et[1] or 'connect' in "".join(et))]
            if not candidates:
                raise ValueError("HeteroData 中未找到任何 'connects' / connect 类型的边。请指定 edge_type。")
            edge_type = candidates[0]
        src_type, rel, dst_type = edge_type
        e_index = data[edge_type].edge_index.cpu().numpy()
        e_attr = getattr(data[edge_type], 'edge_attr', None)
        if e_attr is not None:
            e_attr = e_attr.cpu().numpy().squeeze()
        n_nodes = data[src_type].num_nodes
        mat = np.zeros((n_nodes, n_nodes), dtype=float)
        for i in range(e_index.shape[1]):
            u, v = int(e_index[0, i]), int(e_index[1, i])
            w = float(e_attr[i]) if e_attr is not None else 1.0
            if 0 <= u < n_nodes and 0 <= v < n_nodes:
                mat[u, v] = w
                mat[v, u] = w
        connectome = mat
        coords = None
        if hasattr(data[src_type], 'pos') and data[src_type].pos is not None:
            coords = data[src_type].pos.cpu().numpy()
        elif any('position_3d' in k for k in data[src_type].keys()):
            try:
                coords = data[src_type].position_3d.cpu().numpy()
            except Exception:
                coords = None
        elif data[src_type].x is not None and data[src_type].x.shape[1] >= 3:
            coords = data[src_type].x.cpu().detach().numpy()[:, :3]
        node_prefix = src_type
    else:
        if isinstance(hetero_or_matrix, torch.Tensor):
            connectome = hetero_or_matrix.cpu().numpy()
        else:
            connectome = np.asarray(hetero_or_matrix, dtype=float)
        n_nodes = connectome.shape[0]
        coords = None
        node_prefix = "n"

    # ===== 2) threshold / top_k 筛选边 =====
    mat = np.array(connectome, dtype=float)
    np.fill_diagonal(mat, 0.0)
    weights = mat.copy()
    if threshold is not None:
        mat = np.where(np.abs(mat) >= threshold, mat, 0.0)
    iu, iv = np.where(np.triu(np.abs(mat) > 0, k=1))
    edge_list = [(a, b, mat[a, b]) for a, b in zip(iu, iv)]
    if len(edge_list) == 0 and np.any(np.abs(weights) > 0):
        iu2, iv2 = np.triu_indices_from(weights, k=1)
        vals = np.abs(weights[iu2, iv2])
        idx_sort = np.argsort(vals)[::-1][:min(top_k, vals.size)]
        edge_list = [(int(iu2[i]), int(iv2[i]), float(weights[iu2[i], iv2[i]])) for i in idx_sort]
    elif top_k is not None and len(edge_list) > top_k:
        edge_list = sorted(edge_list, key=lambda x: abs(x[2]), reverse=True)[:top_k]

    # ===== 3) node coords fallback / atlas =====
    if coords is None and atlas_nii_path is not None:
        atlas_path = Path(atlas_nii_path)
        if atlas_path.exists():
            img = nib.load(str(atlas_path))
            data_img = img.get_fdata()
            coords = np.zeros((int(np.max(data_img)), 3))
            for roi in range(1, int(np.max(data_img)) + 1):
                mask = data_img == roi
                if np.sum(mask) > 0:
                    coords[roi - 1] = center_of_mass(mask)
    if coords is None:
        G_tmp = nx.from_numpy_array(mat)
        pos2d = nx.spring_layout(G_tmp, seed=42)
        coords = np.zeros((mat.shape[0], 3), dtype=float)
        for i in range(mat.shape[0]):
            xy = pos2d.get(i, (np.random.rand(), np.random.rand()))
            coords[i, 0] = float(xy[0])
            coords[i, 1] = float(xy[1])
            coords[i, 2] = 0.0

    # ===== 4) node colors / labels =====
    node_vals = None
    if predictions is not None:
        if isinstance(predictions, dict):
            node_vals = np.asarray(predictions.get(node_prefix, next(iter(predictions.values())))).squeeze()
        else:
            node_vals = np.asarray(predictions).squeeze()
    if node_vals is None or node_vals.size == 0:
        deg = np.sum(np.abs(mat) > 0, axis=0)
        node_vals = deg.astype(float)
    if node_vals.shape[0] < mat.shape[0]:
        node_vals = np.pad(node_vals, (0, mat.shape[0] - node_vals.shape[0]), mode='constant')
    labels = [f"{node_prefix}_{i}" for i in range(mat.shape[0])]

    # ===== 5) build plotly traces =====
    edge_x, edge_y, edge_z = [], [], []
    for (u, v, w) in edge_list:
        x0, y0, z0 = coords[u]
        x1, y1, z1 = coords[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]
    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines',
                              line=dict(width=1, color='rgba(150,150,150,0.6)'), hoverinfo='none')
    node_trace = go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                              mode='markers',
                              marker=dict(size=6, color=node_vals, colorscale='Viridis',
                                          colorbar=dict(title='node val'), showscale=True),
                              text=[f"{labels[i]}<br>val={float(node_vals[i]):.4f}" for i in range(len(labels))],
                              hoverinfo='text')

    # edge midpoint hover
    mid_x, mid_y, mid_z, mid_text = [], [], [], []
    for (u, v, w) in edge_list:
        x0, y0, z0 = coords[u]; x1, y1, z1 = coords[v]
        mid_x.append((x0 + x1) / 2.0)
        mid_y.append((y0 + y1) / 2.0)
        mid_z.append((z0 + z1) / 2.0)
        mid_text.append(f"{u}-{v}: {w:.4f}")
    edge_label_trace = None
    if mid_x:
        edge_label_trace = go.Scatter3d(x=mid_x, y=mid_y, z=mid_z, mode='markers',
                                        marker=dict(size=1, color='rgba(0,0,0,0)'),
                                        text=mid_text, hoverinfo='text')

    layout = go.Layout(title=title or f"Connectome ({node_prefix})",
                       scene=dict(xaxis=dict(showbackground=False),
                                  yaxis=dict(showbackground=False),
                                  zaxis=dict(showbackground=False)),
                       margin=dict(b=0, l=0, r=0, t=30),
                       showlegend=False, hovermode='closest')
    data_traces = [edge_trace, node_trace]
    if edge_label_trace is not None:
        data_traces.append(edge_label_trace)
    fig = go.Figure(data=data_traces, layout=layout)
    if html_path is not None:
        fig.write_html(html_path)
        print(f"Saved interactive brain connectome (html) to {html_path}")
    return fig



def visualize_dynamic_evolution(preds, output_dir, task_name,
                                normalize: bool = True,
                                smooth_sigma: float = 2.0,
                                downsample: int = 4):
    """
    preds: list of dict，每个 dict 里有:
        - node_type_seq: (N, T, F)
        - global_state: (hidden_dim,) 可选
    normalize: 是否对每个曲线做 min-max 归一化到 [0,1]，便于不同模态比较
    smooth_sigma: 高斯平滑标准差，None 不平滑
    downsample: 下采样步长，None 不下采样
    """
    node_types = [k.replace("_seq", "") for k in preds[0].keys() if k.endswith("_seq")]
    dynamic_dict = {ntype: [] for ntype in node_types}
    global_dict = []

    for step_pred in preds:
        for ntype in node_types:
            seq = step_pred[f"{ntype}_seq"]  # (N, T, F)
            mean_over_nodes = seq.mean(axis=0)  # (T, F)
            mean_over_features = mean_over_nodes.mean(axis=1)  # (T,)
            dynamic_dict[ntype].append(mean_over_features)
        if "global_state" in step_pred:
            global_dict.append(step_pred["global_state"].mean())

    # 拼接时间序列
    for k in dynamic_dict:
        vals = np.concatenate(dynamic_dict[k])
        orig_vals = vals.copy()  # 保存原值用于差分指标
        if smooth_sigma is not None:
            vals = gaussian_filter1d(vals, sigma=smooth_sigma)
        if downsample is not None and downsample > 1:
            vals = vals[::downsample]
        if normalize:
            min_val, max_val = vals.min(), vals.max()
            vals = (vals - min_val) / (max_val - min_val + 1e-8)
        dynamic_dict[k] = vals
    if global_dict:
        global_dict = np.array(global_dict)
        orig_global = global_dict.copy()
        if smooth_sigma is not None:
            global_dict = gaussian_filter1d(global_dict, sigma=smooth_sigma)
        if downsample is not None and downsample > 1:
            global_dict = global_dict[::downsample]
        if normalize:
            min_val, max_val = global_dict.min(), global_dict.max()
            global_dict = (global_dict - min_val) / (max_val - min_val + 1e-8)
    else:
        global_dict = None
        orig_global = None

    # 绘图
    plt.figure(figsize=(10, 5))
    for ntype, vals in dynamic_dict.items():
        plt.plot(vals, label=f"{ntype} mean")
    if global_dict is not None:
        plt.plot(global_dict, "k--", label="global_state")
    plt.xlabel("Time step")
    plt.ylabel("Normalized activation" if normalize else "Activation (mean)")
    plt.title(f"Dynamic evolution - {task_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt_path = output_dir / f"dynamic_evolution_{task_name}.png"
    plt.savefig(plt_path)
    plt.close()

    # 差分指标使用原始数据，保证不受平滑/下采样影响
    diff_metric = {ntype: orig_vals[-1] - orig_vals[0] for ntype, orig_vals in dynamic_dict.items()}
    if global_dict is not None:
        diff_metric["global_state"] = orig_global[-1] - orig_global[0]

    return str(plt_path), diff_metric

# unsupervised_perf.py

# utils/evaluate_unsupervised_performance.py
import os
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData

logger = logging.getLogger("hetero_trainer")


def evaluate_unsupervised_performance(
    embeddings: Dict[str, np.ndarray],           # {node_type: (N, H)}
    graph: HeteroData,                           # 原始图，含 x_seq_dict
    model: torch.nn.Module,                      # DynamicHeteroGNN 实例
    output_dir: str | Path,
    smooth_window: int = 5,
    device: torch.device | None = None
) -> tuple[str, Dict[str, float]]:
    """
    无监督性能评估：使用 GRU 进行 next-step 预测 MSE

    Args:
        embeddings: 最后一帧的潜在表示（仅用于日志）
        graph: 原始 HeteroData，包含 x_seq_dict
        model: 已训练的 DynamicHeteroGNN（含 .grus）
        output_dir: 输出目录
        smooth_window: 滑动平均窗口
        device: 计算设备

    Returns:
        (png_path, mse_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    # 获取所有 node types
    node_types = [nt for nt in model.node_types if f"{nt}_seq" in graph]
    if not node_types:
        logger.warning("[Perf] No sequence data found in graph.x_seq_dict")
        dummy_png = output_dir / "unsupervised_performance_empty.png"
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No sequence data", ha='center', va='center')
        plt.axis('off')
        plt.savefig(dummy_png)
        plt.close()
        return str(dummy_png), {}

    mse_dict = {}
    plt.figure(figsize=(12, 6))

    for ntype in node_types:
        seq_key = f"{ntype}_seq"
        if seq_key not in graph:
            continue

        seq_np = graph[seq_key].cpu().numpy()  # (N, T, F_in)
        N, T, _ = seq_np.shape
        if T < 2:
            logger.warning(f"[Perf] Node type {ntype} has T<2, skipping.")
            continue

        # 转 tensor
        seq_tensor = torch.tensor(seq_np, dtype=torch.float32, device=device)  # (N, T, F)

        # 使用模型的 GRU 进行 next-step 预测
        gru = model.grus[ntype].to(device)
        with torch.no_grad():
            # 输入：[:, :-1, :] → 预测下一帧
            input_cur = seq_tensor[:, :-1, :].reshape(-1, seq_tensor.shape[-1])  # (N*(T-1), F)
            target_next = seq_tensor[:, 1:, :].reshape(-1, seq_tensor.shape[-1])

            # GRU forward (batch_first=True)
            # 需模拟 GRU 的前向过程
            hidden = None
            pred_list = []
            for t in range(T - 1):
                x_t = seq_tensor[:, t:t+1, :]  # (N, 1, F)
                _, hidden = gru(x_t, hidden)
                pred_list.append(hidden.squeeze(0))  # (N, H)
            pred_next = torch.stack(pred_list, dim=1).reshape(-1, model.hidden_dim)  # (N*(T-1), H)

            # 投影到输入空间（如果 F != H）
            if seq_tensor.shape[-1] != model.hidden_dim:
                proj = nn.Linear(model.hidden_dim, seq_tensor.shape[-1]).to(device)
                pred_next = proj(pred_next)

            mse = F.mse_loss(pred_next, target_next, reduction='none')  # (N*(T-1), F)
            mse_per_step = mse.mean(dim=1).reshape(N, T - 1).mean(dim=0).cpu().numpy()  # (T-1,)

        # 滑动平均
        if smooth_window > 1 and len(mse_per_step) >= smooth_window:
            cumsum = np.cumsum(np.insert(mse_per_step, 0, 0))
            mse_smooth = (cumsum[smooth_window:] - cumsum[:-smooth_window]) / smooth_window
        else:
            mse_smooth = mse_per_step

        # 绘图
        steps = np.arange(len(mse_smooth))
        plt.plot(steps, mse_smooth, label=f"{ntype} (N={N})", linewidth=2)

        mse_dict[ntype] = float(mse_smooth.mean())

    if mse_dict:
        plt.xlabel("Time Step (smoothed)")
        plt.ylabel("Next-step Prediction MSE")
        plt.title("Unsupervised Temporal Modeling Performance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        png_path = output_dir / "unsupervised_temporal_mse.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        logger.info(f"[Perf] Saved temporal MSE plot: {png_path}")
        logger.info(f"[Perf] MSE: { {k: f'{v:.6f}' for k, v in mse_dict.items()} }")
    else:
        png_path = output_dir / "unsupervised_temporal_mse_empty.png"
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No valid sequences for MSE", ha='center', va='center')
        plt.axis('off')
        plt.savefig(png_path)
        plt.close()
        mse_dict = {}

    return str(png_path), mse_dict