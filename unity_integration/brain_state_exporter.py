"""
Brain State Exporter
====================

Export brain activity states to JSON format for Unity visualization.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class BrainStateExporter:
    """
    Export brain states to JSON format compatible with Unity frontend.
    
    The JSON format includes:
    - Region-level activity (fMRI, EEG)
    - Connections (structural and functional)
    - Global metrics
    - Predictions (if available)
    - Stimulation info (if applied)
    """
    
    def __init__(self, atlas_info: Optional[Dict[str, Any]] = None, model_version: str = "v4"):
        """
        Initialize exporter.
        
        Args:
            atlas_info: Dictionary containing atlas information
                       {'regions': {id: {'label': str, 'xyz': [x,y,z]}, ...}}
                       If None, default values will be used.
            model_version: Model version string
        """
        self.atlas_info = atlas_info or {}
        self.model_version = model_version
        self.regions_info = self.atlas_info.get('regions', {})
    
    def export_brain_state(
        self,
        brain_activity: Dict[str, torch.Tensor],
        connectivity: Optional[Dict[str, np.ndarray]] = None,
        time_point: int = 0,
        time_second: float = 0.0,
        subject_id: str = "unknown",
        predictions: Optional[Dict[str, Any]] = None,
        stimulation: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Export current brain state to JSON format.
        
        Args:
            brain_activity: Dictionary with modality -> tensor data
                           {'fmri': [N_regions, T, F], 'eeg': [N_regions, T, F]}
            connectivity: Optional connectivity matrices
                         {'structural': [N, N], 'functional': [N, N]}
            time_point: Current time point index
            time_second: Current time in seconds
            subject_id: Subject identifier
            predictions: Optional prediction results
            stimulation: Optional stimulation parameters
            output_path: If provided, save to file
        
        Returns:
            Dictionary containing brain state in JSON-compatible format
        """
        timestamp = datetime.now().isoformat()
        
        # Build JSON structure
        brain_state_json = {
            "version": "2.0",
            "timestamp": timestamp,
            "metadata": {
                "subject": subject_id,
                "atlas": self.atlas_info.get('name', 'Unknown'),
                "model_version": self.model_version,
                "time_point": time_point,
                "time_second": time_second
            },
            "brain_state": {
                "time_point": time_point,
                "time_second": time_second,
                "regions": self._export_regions(brain_activity, time_point, predictions),
                "connections": self._export_connections(connectivity),
                "networks": self._compute_networks(brain_activity, time_point),
                "global_metrics": self._compute_global_metrics(brain_activity, time_point)
            }
        }
        
        # Add stimulation info if applied
        if stimulation is not None:
            brain_state_json["stimulation"] = stimulation
        
        # Save to file if requested
        if output_path is not None:
            self.save_json(brain_state_json, output_path)
        
        return brain_state_json
    
    def _export_regions(
        self, 
        brain_activity: Dict[str, torch.Tensor],
        time_point: int,
        predictions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Export region-level information."""
        regions = []
        
        # Get number of regions from data
        fmri_data = brain_activity.get('fmri', None)
        eeg_data = brain_activity.get('eeg', None)
        
        if fmri_data is None and eeg_data is None:
            return regions
        
        n_regions = fmri_data.shape[0] if fmri_data is not None else eeg_data.shape[0]
        
        # Extract prediction data if available
        pred_data = None
        if predictions is not None and 'predicted_activity' in predictions:
            pred_data = predictions['predicted_activity']
            # Warn if prediction data size doesn't match number of regions
            if len(pred_data) != n_regions:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Prediction data size mismatch: got {len(pred_data)} predictions "
                    f"but have {n_regions} regions. Some regions may not have predictions."
                )
        
        for region_id in range(n_regions):
            region_info = self.regions_info.get(str(region_id + 1), {})
            
            region_dict = {
                "id": region_id,
                "label": region_info.get('label', f'Region_{region_id}'),
                "position": self._get_region_position(region_info),
                "activity": {}
            }
            
            # Add fMRI activity
            if fmri_data is not None:
                fmri_activity = self._compute_fmri_activity(
                    fmri_data[region_id], time_point
                )
                region_dict["activity"]["fmri"] = fmri_activity
            
            # Add EEG activity
            if eeg_data is not None:
                eeg_activity = self._compute_eeg_activity(
                    eeg_data[region_id], time_point
                )
                region_dict["activity"]["eeg"] = eeg_activity
            
            # Add prediction data if available
            if pred_data is not None:
                if region_id < len(pred_data):
                    pred_value = pred_data[region_id]
                    if isinstance(pred_value, torch.Tensor):
                        pred_value = pred_value.item()
                    # Normalize prediction value to [0, 1]
                    # Assumes model output is standardized to ~[-3, 3] range
                    # Maps: -3 → 0, 0 → 0.5, +3 → 1
                    pred_normalized = (pred_value + 3.0) / 6.0
                    pred_normalized = max(0.0, min(1.0, pred_normalized))
                    region_dict["activity"]["predictionValue"] = float(pred_normalized)
                    region_dict["activity"]["isPredicted"] = True
                else:
                    # Prediction data doesn't cover this region
                    region_dict["activity"]["isPredicted"] = False
            else:
                region_dict["activity"]["isPredicted"] = False
            
            regions.append(region_dict)
        
        return regions
    
    def _get_region_position(self, region_info: Dict[str, Any]) -> Dict[str, float]:
        """Get region 3D position, with safe fallback if xyz is missing."""
        xyz = region_info.get('xyz')
        
        # Safe fallback if xyz is missing
        if xyz is None or not isinstance(xyz, (list, tuple, np.ndarray)) or len(xyz) < 3:
            # Generate default position to ensure visualization works
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        
        return {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}
    
    def _compute_fmri_activity(
        self,
        region_data: torch.Tensor,
        time_point: int
    ) -> Dict[str, float]:
        """
        Compute fMRI activity metrics for a region with memory optimization.
        
        Uses CPU conversion and explicit cleanup to minimize memory usage.
        """
        # Ensure data is on CPU and detached to save memory
        if region_data.is_cuda:
            region_data = region_data.cpu()
        region_data = region_data.detach()
        
        # region_data shape: [T, F] or [T]
        if len(region_data.shape) == 1:
            # Single feature dimension
            activity = region_data[time_point].item() if time_point < len(region_data) else 0.0
        else:
            # Multiple features, take mean
            if time_point < region_data.shape[0]:
                activity = region_data[time_point].mean().item()
            else:
                activity = 0.0
        
        # Normalize to [0, 1] range (assuming data is roughly [-3, 3] after normalization)
        activity_normalized = (activity + 3.0) / 6.0
        activity_normalized = max(0.0, min(1.0, activity_normalized))
        
        return {
            "amplitude": float(activity_normalized),
            "raw_value": float(activity)
        }
    
    def _compute_eeg_activity(
        self,
        region_data: torch.Tensor,
        time_point: int
    ) -> Dict[str, Any]:
        """
        Compute EEG activity metrics for a region with memory optimization.
        """
        # Ensure data is on CPU and detached
        if region_data.is_cuda:
            region_data = region_data.cpu()
        region_data = region_data.detach()
        
        if len(region_data.shape) == 1:
            activity = region_data[time_point].item() if time_point < len(region_data) else 0.0
        else:
            if time_point < region_data.shape[0]:
                activity = region_data[time_point].mean().item()
            else:
                activity = 0.0
        
        # Normalize
        activity_normalized = (activity + 3.0) / 6.0
        activity_normalized = max(0.0, min(1.0, activity_normalized))
        
        return {
            "amplitude": float(activity_normalized),
            "raw_value": float(activity)
        }
    
    def _export_connections(
        self,
        connectivity: Optional[Dict[str, np.ndarray]]
    ) -> List[Dict[str, Any]]:
        """
        Export connectivity information with memory optimization.
        
        Uses sparse representation and thresholding to reduce output size.
        """
        if connectivity is None:
            return []
        
        connections = []
        max_connections = 10000  # Limit total connections for performance
        
        # Process structural connectivity
        if 'structural' in connectivity:
            struct_conn = connectivity['structural']
            threshold = 0.3  # Only include strong connections
            
            # Use numpy operations for efficiency
            i_indices, j_indices = np.triu_indices(struct_conn.shape[0], k=1)
            strengths = struct_conn[i_indices, j_indices]
            strong_mask = np.abs(strengths) > threshold
            
            # Filter to only strong connections
            strong_i = i_indices[strong_mask]
            strong_j = j_indices[strong_mask]
            strong_strengths = strengths[strong_mask]
            
            # Limit number of connections
            if len(strong_i) > max_connections // 2:
                # Sort by strength and keep top connections
                top_indices = np.argsort(np.abs(strong_strengths))[-max_connections // 2:]
                strong_i = strong_i[top_indices]
                strong_j = strong_j[top_indices]
                strong_strengths = strong_strengths[top_indices]
            
            for idx in range(len(strong_i)):
                connections.append({
                    "source": int(strong_i[idx]),
                    "target": int(strong_j[idx]),
                    "strength": float(np.abs(strong_strengths[idx])),
                    "type": "structural",
                    "bidirectional": True
                })
        
        # Process functional connectivity
        if 'functional' in connectivity:
            func_conn = connectivity['functional']
            threshold = 0.3
            
            # Use numpy operations for efficiency
            i_indices, j_indices = np.triu_indices(func_conn.shape[0], k=1)
            strengths = func_conn[i_indices, j_indices]
            strong_mask = np.abs(strengths) > threshold
            
            # Filter to only strong connections
            strong_i = i_indices[strong_mask]
            strong_j = j_indices[strong_mask]
            strong_strengths = strengths[strong_mask]
            
            # Limit number of connections
            remaining = max_connections - len(connections)
            if len(strong_i) > remaining:
                # Sort by strength and keep top connections
                top_indices = np.argsort(np.abs(strong_strengths))[-remaining:]
                strong_i = strong_i[top_indices]
                strong_j = strong_j[top_indices]
                strong_strengths = strong_strengths[top_indices]
            
            for idx in range(len(strong_i)):
                connections.append({
                    "source": int(strong_i[idx]),
                    "target": int(strong_j[idx]),
                    "strength": float(np.abs(strong_strengths[idx])),
                    "type": "functional",
                    "correlation": float(strong_strengths[idx])
                })
        
        return connections
    
    def _compute_networks(
        self,
        brain_activity: Dict[str, torch.Tensor],
        time_point: int
    ) -> Dict[str, Any]:
        """Compute network-level metrics."""
        networks = {}
        
        # Define common networks (simplified)
        network_definitions = {
            "visual": list(range(0, 20)),
            "motor": list(range(50, 70)),
            "default_mode": list(range(100, 120)),
        }
        
        fmri_data = brain_activity.get('fmri', None)
        if fmri_data is None:
            return networks
        
        for network_name, region_indices in network_definitions.items():
            # Filter valid indices
            valid_indices = [i for i in region_indices if i < fmri_data.shape[0]]
            
            if len(valid_indices) > 0:
                # Extract network activity
                network_data = fmri_data[valid_indices, time_point] if time_point < fmri_data.shape[1] else fmri_data[valid_indices, -1]
                
                networks[network_name] = {
                    "avg_activity": float(network_data.mean().item()),
                    "regions": valid_indices
                }
        
        return networks
    
    def _compute_global_metrics(
        self,
        brain_activity: Dict[str, torch.Tensor],
        time_point: int
    ) -> Dict[str, float]:
        """Compute global brain metrics."""
        fmri_data = brain_activity.get('fmri', None)
        
        if fmri_data is None:
            return {}
        
        # Get activity at current time point
        if time_point < fmri_data.shape[1]:
            current_activity = fmri_data[:, time_point]
        else:
            current_activity = fmri_data[:, -1]
        
        # Compute statistics
        if len(current_activity.shape) > 1:
            current_activity = current_activity.mean(dim=1)
        
        mean_activity = float(current_activity.mean().item())
        std_activity = float(current_activity.std().item())
        max_activity = float(current_activity.max().item())
        
        # Count active regions (above threshold)
        threshold = mean_activity
        active_regions = int((current_activity > threshold).sum().item())
        
        return {
            "mean_activity": mean_activity,
            "std_activity": std_activity,
            "max_activity": max_activity,
            "active_regions": active_regions
        }
    
    def export_sequence(
        self,
        brain_activity: Dict[str, torch.Tensor],
        output_dir: Path,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1,
        connectivity: Optional[Dict[str, np.ndarray]] = None,
        subject_id: str = "unknown"
    ):
        """
        Export a sequence of brain states.
        
        Args:
            brain_activity: Brain activity tensors
            output_dir: Output directory
            start: Start time point
            end: End time point (None = all)
            step: Time step
            connectivity: Optional connectivity matrices
            subject_id: Subject identifier
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fmri_data = brain_activity.get('fmri', None)
        if fmri_data is None:
            raise ValueError("No fMRI data available")
        
        T = fmri_data.shape[1]
        if end is None:
            end = T
        
        end = min(end, T)
        
        # Export each time point
        for t in range(start, end, step):
            brain_state = self.export_brain_state(
                brain_activity=brain_activity,
                connectivity=connectivity,
                time_point=t,
                time_second=float(t),  # Simplified
                subject_id=subject_id,
                output_path=output_dir / f"brain_state_{t:04d}.json"
            )
        
        # Also create an index file
        index = {
            "subject": subject_id,
            "start": start,
            "end": end,
            "step": step,
            "n_frames": len(range(start, end, step)),
            "files": [f"brain_state_{t:04d}.json" for t in range(start, end, step)]
        }
        
        with open(output_dir / "sequence_index.json", 'w') as f:
            json.dump(index, f, indent=2)
    
    @staticmethod
    def save_json(data: Dict[str, Any], output_path: Path):
        """Save dictionary to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_json(input_path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
