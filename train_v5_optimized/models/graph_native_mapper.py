"""
Graph-Native Brain Mapper
=========================

Complete reimagination of brain data mapping that:
1. Builds and MAINTAINS graph structure throughout pipeline
2. No graph → sequence → graph conversions
3. Native small-world network representation
4. Efficient spatial-temporal modeling on graphs

Philosophy:
- Brain = Graph (small-world network)
- Keep it as graph from data loading to training
- Leverage graph structure for interpretability
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops, to_undirected
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GraphNativeBrainMapper:
    """
    Unified brain data mapper that maintains graph structure.
    
    Key Principles:
    1. Build graph ONCE from anatomical/functional connectivity
    2. Keep node features as temporal signals ON the graph
    3. No unnecessary conversions
    4. Support both fMRI and EEG modalities
    """
    
    def __init__(
        self,
        atlas_name: str = 'schaefer200',
        preserve_temporal: bool = True,
        add_self_loops: bool = True,
        make_undirected: bool = True,
    ):
        """
        Initialize graph-native mapper.
        
        Args:
            atlas_name: Brain atlas for parcellation
            preserve_temporal: Keep temporal dimension in node features
            add_self_loops: Add self-connections to graph
            make_undirected: Symmetrize edge connections
        """
        self.atlas_name = atlas_name
        self.preserve_temporal = preserve_temporal
        self.add_self_loops_flag = add_self_loops
        self.make_undirected_flag = make_undirected
        
        # Graph structure (built once, reused)
        self.base_graph_structure = None
        self.node_positions = None  # 3D coordinates
        self.node_labels = None  # Region names
        
    def build_graph_structure(
        self,
        connectivity_matrix: Optional[np.ndarray] = None,
        threshold: float = 0.1,
        k_nearest: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build base graph structure from connectivity.
        
        This is the CORE graph that persists throughout pipeline.
        Built once from anatomical/functional connectivity.
        
        Args:
            connectivity_matrix: [N, N] connectivity weights
            threshold: Minimum weight to create edge
            k_nearest: If set, keep only k nearest neighbors per node
            
        Returns:
            edge_index: [2, E] graph edges
            edge_attr: [E, 1] edge weights
        """
        N = connectivity_matrix.shape[0]
        
        # Build adjacency
        if k_nearest is not None:
            # K-nearest neighbors (promotes small-world)
            edge_index_list = []
            edge_attr_list = []
            
            for i in range(N):
                # Get k strongest connections for node i
                weights = connectivity_matrix[i]
                top_k_indices = np.argsort(-weights)[:k_nearest]
                
                for j in top_k_indices:
                    if i != j and weights[j] > threshold:
                        edge_index_list.append([i, j])
                        edge_attr_list.append(weights[j])
            
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).unsqueeze(-1)
        
        else:
            # Threshold-based
            adj = connectivity_matrix > threshold
            np.fill_diagonal(adj, False)  # No self-loops initially
            
            edge_index = torch.tensor(np.array(np.where(adj)), dtype=torch.long)
            edge_attr = torch.tensor(
                connectivity_matrix[adj],
                dtype=torch.float32
            ).unsqueeze(-1)
        
        # Make undirected (brain connections are symmetric)
        if self.make_undirected_flag:
            edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce='mean')
        
        # Add self-loops (important for GNN message passing)
        if self.add_self_loops_flag:
            edge_index, edge_attr = add_self_loops(
                edge_index,
                edge_attr,
                fill_value=1.0,
                num_nodes=N
            )
        
        logger.info(
            f"Built graph: {N} nodes, {edge_index.shape[1]} edges "
            f"(avg degree: {edge_index.shape[1]/N:.1f})"
        )
        
        return edge_index, edge_attr
    
    def map_fmri_to_graph(
        self,
        timeseries: np.ndarray,
        connectivity_matrix: Optional[np.ndarray] = None,
        node_positions: Optional[np.ndarray] = None,
        node_labels: Optional[List[str]] = None,
    ) -> HeteroData:
        """
        Map fMRI timeseries to graph structure.
        
        KEEPS temporal dimension as node features!
        No flattening or unnecessary transformations.
        
        Args:
            timeseries: [N_rois, T_time] or [T_time, N_rois]
            connectivity_matrix: [N_rois, N_rois] functional connectivity
            node_positions: [N_rois, 3] brain coordinates
            node_labels: ROI names
            
        Returns:
            HeteroData with 'fmri' node type
        """
        # Ensure shape [N, T]
        if timeseries.ndim != 2:
            raise ValueError(f"Expected 2D timeseries, got shape {timeseries.shape}")
        
        if timeseries.shape[0] > timeseries.shape[1]:
            timeseries = timeseries.T  # Assume [T, N] -> [N, T]
        
        N_rois, T_time = timeseries.shape
        
        # Build graph structure (if not already built)
        if connectivity_matrix is None:
            # Use temporal correlation as connectivity
            connectivity_matrix = np.corrcoef(timeseries)
            connectivity_matrix = np.abs(connectivity_matrix)  # Use absolute correlation
        
        edge_index, edge_attr = self.build_graph_structure(
            connectivity_matrix,
            threshold=0.3,  # Keep strong connections
            k_nearest=20,  # Small-world: ~20 neighbors per node
        )
        
        # Node features: KEEP temporal dimension
        # Shape: [N, T, 1] - one feature channel (fMRI signal)
        x = torch.tensor(timeseries, dtype=torch.float32).unsqueeze(-1)  # [N, T, 1]
        
        # Create HeteroData
        data = HeteroData()
        
        # fMRI nodes with temporal features
        data['fmri'].x = x  # [N, T, 1]
        data['fmri'].num_nodes = N_rois
        
        # Graph structure
        data['fmri', 'connects', 'fmri'].edge_index = edge_index
        data['fmri', 'connects', 'fmri'].edge_attr = edge_attr
        
        # Metadata
        if node_positions is not None:
            data['fmri'].pos = torch.tensor(node_positions, dtype=torch.float32)
        
        if node_labels is not None:
            data['fmri'].labels = node_labels
        
        # Store temporal info
        data['fmri'].temporal_length = T_time
        data['fmri'].sampling_rate = 0.5  # Typical fMRI TR (2 seconds)
        
        logger.info(
            f"Mapped fMRI to graph: {N_rois} nodes, {T_time} timepoints, "
            f"{edge_index.shape[1]} edges"
        )
        
        return data
    
    def map_eeg_to_graph(
        self,
        timeseries: np.ndarray,
        channel_names: List[str],
        connectivity_matrix: Optional[np.ndarray] = None,
        channel_positions: Optional[np.ndarray] = None,
        atlas_mapping: Optional[Dict[str, int]] = None,
    ) -> HeteroData:
        """
        Map EEG timeseries to graph structure.
        
        Can map EEG channels to atlas ROIs if atlas_mapping provided,
        otherwise treats channels as graph nodes.
        
        Args:
            timeseries: [N_channels, T_time] EEG signals
            channel_names: Channel labels
            connectivity_matrix: [N_channels, N_channels] connectivity
            channel_positions: [N_channels, 3] electrode positions
            atlas_mapping: Map channels to atlas ROIs
            
        Returns:
            HeteroData with 'eeg' node type
        """
        # Ensure shape [N, T]
        if timeseries.ndim != 2:
            raise ValueError(f"Expected 2D timeseries, got shape {timeseries.shape}")
        
        if timeseries.shape[0] > timeseries.shape[1]:
            timeseries = timeseries.T
        
        N_channels, T_time = timeseries.shape
        
        # Build connectivity if not provided
        if connectivity_matrix is None:
            # Use coherence-based connectivity
            connectivity_matrix = self._compute_eeg_connectivity(timeseries)
        
        # Build graph structure
        edge_index, edge_attr = self.build_graph_structure(
            connectivity_matrix,
            threshold=0.2,
            k_nearest=10,  # EEG: fewer neighbors (more local)
        )
        
        # Node features: temporal EEG signals
        # Shape: [N, T, 1]
        x = torch.tensor(timeseries, dtype=torch.float32).unsqueeze(-1)
        
        # Create HeteroData
        data = HeteroData()
        
        # EEG nodes with temporal features
        data['eeg'].x = x  # [N, T, 1]
        data['eeg'].num_nodes = N_channels
        
        # Graph structure
        data['eeg', 'connects', 'eeg'].edge_index = edge_index
        data['eeg', 'connects', 'eeg'].edge_attr = edge_attr
        
        # Metadata
        if channel_positions is not None:
            data['eeg'].pos = torch.tensor(channel_positions, dtype=torch.float32)
        
        data['eeg'].labels = channel_names
        data['eeg'].temporal_length = T_time
        data['eeg'].sampling_rate = 250.0  # Typical EEG sampling rate
        
        # Atlas mapping if provided (for EEG-fMRI alignment)
        if atlas_mapping is not None:
            data['eeg'].atlas_mapping = atlas_mapping
        
        logger.info(
            f"Mapped EEG to graph: {N_channels} channels, {T_time} timepoints, "
            f"{edge_index.shape[1]} edges"
        )
        
        return data
    
    def _compute_eeg_connectivity(self, timeseries: np.ndarray) -> np.ndarray:
        """
        Compute EEG connectivity matrix using coherence.
        
        Args:
            timeseries: [N_channels, T_time]
            
        Returns:
            connectivity: [N_channels, N_channels]
        """
        N = timeseries.shape[0]
        
        # Use correlation as simple connectivity measure
        # For production, could use coherence in specific frequency bands
        connectivity = np.corrcoef(timeseries)
        connectivity = np.abs(connectivity)
        
        # Ensure valid values
        connectivity = np.nan_to_num(connectivity, nan=0.0)
        np.fill_diagonal(connectivity, 1.0)
        
        return connectivity
    
    def create_cross_modal_edges(
        self,
        data: HeteroData,
        eeg_to_fmri_mapping: Optional[Dict[int, int]] = None,
        distance_threshold: Optional[float] = None,
    ) -> HeteroData:
        """
        Add edges between EEG and fMRI nodes.
        
        Creates the cross-modal graph for joint modeling.
        
        Args:
            data: HeteroData with both 'eeg' and 'fmri' nodes
            eeg_to_fmri_mapping: Direct channel->ROI mapping
            distance_threshold: Max spatial distance for connections
            
        Returns:
            HeteroData with cross-modal edges added
        """
        if 'eeg' not in data.node_types or 'fmri' not in data.node_types:
            return data
        
        N_eeg = data['eeg'].num_nodes
        N_fmri = data['fmri'].num_nodes
        
        # Method 1: Direct mapping if provided
        if eeg_to_fmri_mapping is not None:
            edge_list = []
            for eeg_idx, fmri_idx in eeg_to_fmri_mapping.items():
                if 0 <= eeg_idx < N_eeg and 0 <= fmri_idx < N_fmri:
                    edge_list.append([eeg_idx, fmri_idx])
            
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                
                # Bidirectional connections
                data['eeg', 'projects_to', 'fmri'].edge_index = edge_index
                data['fmri', 'receives_from', 'eeg'].edge_index = edge_index.flip(0)
                
                logger.info(f"Added {len(edge_list)} cross-modal edges via mapping")
        
        # Method 2: Distance-based if positions available
        elif distance_threshold is not None:
            if hasattr(data['eeg'], 'pos') and hasattr(data['fmri'], 'pos'):
                eeg_pos = data['eeg'].pos  # [N_eeg, 3]
                fmri_pos = data['fmri'].pos  # [N_fmri, 3]
                
                # Compute pairwise distances
                dist = torch.cdist(eeg_pos, fmri_pos, p=2)  # [N_eeg, N_fmri]
                
                # Keep edges below threshold
                mask = dist < distance_threshold
                edge_index = mask.nonzero().t()  # [2, E]
                
                if edge_index.shape[1] > 0:
                    # Add edges
                    data['eeg', 'projects_to', 'fmri'].edge_index = edge_index
                    data['fmri', 'receives_from', 'eeg'].edge_index = edge_index.flip(0)
                    
                    # Edge attributes: inverse distance (closer = stronger)
                    edge_weights = 1.0 / (dist[mask] + 1e-6)
                    data['eeg', 'projects_to', 'fmri'].edge_attr = edge_weights.unsqueeze(-1)
                    data['fmri', 'receives_from', 'eeg'].edge_attr = edge_weights.unsqueeze(-1)
                    
                    logger.info(
                        f"Added {edge_index.shape[1]} cross-modal edges via distance "
                        f"(threshold={distance_threshold})"
                    )
        
        return data


class TemporalGraphFeatureExtractor(nn.Module):
    """
    Extract features from temporal signals ON the graph.
    
    No conversion to sequences - process directly on graph structure.
    Uses depthwise separable convolutions along temporal dimension.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        num_temporal_layers: int = 3,
        temporal_kernel_size: int = 7,
    ):
        """
        Initialize temporal feature extractor.
        
        Args:
            in_channels: Input feature channels (typically 1 for raw signals)
            hidden_channels: Hidden feature dimension
            num_temporal_layers: Number of temporal conv layers
            temporal_kernel_size: Kernel size for temporal convolution
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Temporal feature extraction (1D conv on time axis)
        layers = []
        
        # First layer: in_channels -> hidden_channels
        layers.append(nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2,
        ))
        layers.append(nn.BatchNorm1d(hidden_channels))
        layers.append(nn.ReLU())
        
        # Additional layers: hidden -> hidden
        for _ in range(num_temporal_layers - 1):
            # Depthwise separable: efficient and effective
            layers.append(nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=temporal_kernel_size,
                padding=temporal_kernel_size // 2,
                groups=hidden_channels,  # Depthwise
            ))
            layers.append(nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=1,  # Pointwise
            ))
            layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.ReLU())
        
        self.temporal_conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract temporal features.
        
        Args:
            x: Node features [N, T, C] where C=in_channels
            
        Returns:
            h: Temporal features [N, T, H] where H=hidden_channels
        """
        N, T, C = x.shape
        
        # Reshape for 1D conv: [N, T, C] -> [N, C, T]
        x = x.permute(0, 2, 1)
        
        # Apply temporal convolution
        h = self.temporal_conv(x)  # [N, H, T]
        
        # Reshape back: [N, H, T] -> [N, T, H]
        h = h.permute(0, 2, 1)
        
        return h
