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
    
    def __init__(self, atlas_info: Dict[str, Any], model_version: str = "v4"):
        """
        Initialize exporter.
        
        Args:
            atlas_info: Dictionary containing atlas information
                       {'regions': {id: {'label': str, 'xyz': [x,y,z]}, ...}}
            model_version: Model version string
        """
        self.atlas_info = atlas_info
        self.model_version = model_version
        self.regions_info = atlas_info.get('regions', {})
    
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
                "regions": self._export_regions(brain_activity, time_point),
                "connections": self._export_connections(connectivity),
                "networks": self._compute_networks(brain_activity, time_point),
                "global_metrics": self._compute_global_metrics(brain_activity, time_point)
            }
        }
        
        # Add predictions if available
        if predictions is not None:
            brain_state_json["prediction"] = predictions
        
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
        time_point: int
    ) -> List[Dict[str, Any]]:
        """Export region-level information."""
        regions = []
        
        # Get number of regions from data
        fmri_data = brain_activity.get('fmri', None)
        eeg_data = brain_activity.get('eeg', None)
        
        if fmri_data is None and eeg_data is None:
            return regions
        
        n_regions = fmri_data.shape[0] if fmri_data is not None else eeg_data.shape[0]
        
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
        """Compute fMRI activity metrics for a region."""
        # region_data shape: [T, F]
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
        """Compute EEG activity metrics for a region."""
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
        """Export connectivity information."""
        if connectivity is None:
            return []
        
        connections = []
        
        # Process structural connectivity
        if 'structural' in connectivity:
            struct_conn = connectivity['structural']
            threshold = 0.3  # Only include strong connections
            
            for i in range(struct_conn.shape[0]):
                for j in range(i + 1, struct_conn.shape[1]):
                    strength = float(struct_conn[i, j])
                    if abs(strength) > threshold:
                        connections.append({
                            "source": int(i),
                            "target": int(j),
                            "strength": abs(strength),
                            "type": "structural",
                            "bidirectional": True
                        })
        
        # Process functional connectivity
        if 'functional' in connectivity:
            func_conn = connectivity['functional']
            threshold = 0.3
            
            for i in range(func_conn.shape[0]):
                for j in range(i + 1, func_conn.shape[1]):
                    strength = float(func_conn[i, j])
                    if abs(strength) > threshold:
                        connections.append({
                            "source": int(i),
                            "target": int(j),
                            "strength": abs(strength),
                            "type": "functional",
                            "correlation": strength
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
