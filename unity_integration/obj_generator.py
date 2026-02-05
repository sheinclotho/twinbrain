"""
OBJ 3D Model Generator
=====================

Generate detailed 3D brain models in OBJ format for Unity import.
Creates sphere meshes for brain regions with proper vertices, normals, and faces.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch
from datetime import datetime


class BrainOBJGenerator:
    """
    Generate OBJ 3D models for brain visualization.
    
    Creates sphere meshes for each brain region with proper geometry:
    - Vertices
    - Normal vectors
    - Texture coordinates
    - Faces (triangles)
    """
    
    def __init__(
        self,
        atlas_info: Optional[Dict[str, Any]] = None,
        sphere_resolution: int = 16
    ):
        """
        Initialize OBJ generator.
        
        Args:
            atlas_info: Brain atlas information (optional if providing coordinates directly)
            sphere_resolution: Number of segments for sphere (higher = smoother)
        """
        self.atlas_info = atlas_info or {}
        self.sphere_resolution = sphere_resolution
        self.regions_info = self.atlas_info.get('regions', {})
    
    def generate_sphere_vertices(
        self,
        center: np.ndarray,
        radius: float,
        resolution: int
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
        """
        Generate vertices, normals, and faces for a sphere.
        
        Args:
            center: Sphere center [x, y, z]
            radius: Sphere radius
            resolution: Number of segments
        
        Returns:
            vertices: Nx3 array of vertex positions
            normals: Nx3 array of normal vectors
            faces: List of triangular faces (vertex indices)
        """
        vertices = []
        normals = []
        faces = []
        
        # Generate vertices using spherical coordinates
        for i in range(resolution + 1):
            lat = np.pi * i / resolution  # Latitude
            
            for j in range(resolution + 1):
                lon = 2 * np.pi * j / resolution  # Longitude
                
                # Spherical to Cartesian conversion
                x = np.sin(lat) * np.cos(lon)
                y = np.sin(lat) * np.sin(lon)
                z = np.cos(lat)
                
                # Vertex position
                vertex = center + radius * np.array([x, y, z])
                vertices.append(vertex)
                
                # Normal (pointing outward from sphere center)
                normals.append([x, y, z])
        
        # Generate faces (triangles)
        for i in range(resolution):
            for j in range(resolution):
                # Calculate vertex indices
                v1 = i * (resolution + 1) + j
                v2 = v1 + 1
                v3 = (i + 1) * (resolution + 1) + j
                v4 = v3 + 1
                
                # Two triangles per quad
                faces.append((v1, v3, v2))
                faces.append((v2, v3, v4))
        
        return np.array(vertices), np.array(normals), faces
    
    def export_brain_model(
        self,
        output_path: Path,
        activity_data: Optional[torch.Tensor] = None,
        region_positions: Optional[Dict[int, np.ndarray]] = None,
        min_radius: float = 1.5,
        max_radius: float = 4.0,
        include_all_regions: bool = True
    ):
        """
        Export complete brain model to OBJ file.
        
        Args:
            output_path: Output OBJ file path
            activity_data: Optional activity values for each region [N_regions, ...]
            region_positions: Optional dict mapping region_id to [x,y,z] position (overrides atlas)
            min_radius: Minimum sphere radius
            max_radius: Maximum sphere radius (for high activity)
            include_all_regions: Include all regions regardless of activity
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Write header
            f.write("# TwinBrain 3D Brain Model\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Atlas: {self.atlas_info.get('name', 'Unknown')}\n")
            f.write(f"# Regions: {len(self.regions_info)}\n")
            f.write(f"# Resolution: {self.sphere_resolution}\n\n")
            
            vertex_offset = 1  # OBJ indices start at 1
            
            # Process each region
            # Use provided positions or fall back to atlas info
            if region_positions:
                regions_to_process = region_positions.items()
            else:
                regions_to_process = [(int(rid), info) for rid, info in self.regions_info.items()]
            
            for region_id, region_data in regions_to_process:
                if isinstance(region_data, dict):
                    # From atlas info
                    region_idx = region_id - 1
                    xyz = region_data.get('xyz', [0, 0, 0])
                    label = region_data.get('label', f'Region_{region_id}')
                else:
                    # Direct position array
                    region_idx = region_id - 1
                    xyz = region_data if isinstance(region_data, (list, np.ndarray)) else [0, 0, 0]
                    label = f'Region_{region_id}'
                
                center = np.array(xyz)
                
                # Calculate radius based on activity
                if activity_data is not None and region_idx < len(activity_data):
                    # Get activity value
                    if len(activity_data.shape) == 1:
                        activity = activity_data[region_idx].item()
                    else:
                        activity = activity_data[region_idx].mean().item()
                    
                    # Normalize activity to [0, 1]
                    activity_norm = (activity + 3.0) / 6.0
                    activity_norm = max(0.0, min(1.0, activity_norm))
                    
                    # Map to radius
                    radius = min_radius + (max_radius - min_radius) * activity_norm
                else:
                    radius = (min_radius + max_radius) / 2
                
                # Generate sphere geometry
                vertices, normals, faces = self.generate_sphere_vertices(
                    center, radius, self.sphere_resolution
                )
                
                # Write region comment
                f.write(f"# Region {region_id}: {label}\n")
                if activity_data is not None:
                    f.write(f"# Activity: {activity_norm:.3f}\n")
                f.write(f"# Position: {xyz}\n")
                f.write(f"# Radius: {radius:.2f}\n")
                f.write(f"g region_{region_id}\n")
                
                # Write vertices
                for vertex in vertices:
                    f.write(f"v {vertex[0]:.4f} {vertex[1]:.4f} {vertex[2]:.4f}\n")
                
                # Write normals
                for normal in normals:
                    f.write(f"vn {normal[0]:.4f} {normal[1]:.4f} {normal[2]:.4f}\n")
                
                # Write faces
                for face in faces:
                    v1, v2, v3 = face
                    # Add vertex offset and reference normals
                    f.write(f"f {v1+vertex_offset}//{v1+vertex_offset} ")
                    f.write(f"{v2+vertex_offset}//{v2+vertex_offset} ")
                    f.write(f"{v3+vertex_offset}//{v3+vertex_offset}\n")
                
                f.write("\n")
                
                # Update vertex offset for next region
                vertex_offset += len(vertices)
    
    def export_regions_separately(
        self,
        output_dir: Path,
        activity_data: Optional[torch.Tensor] = None,
        region_positions: Optional[Dict[int, np.ndarray]] = None,
        min_radius: float = 1.5,
        max_radius: float = 4.0,
        prefix: str = "region"
    ) -> List[Path]:
        """
        Export each brain region as a separate OBJ file.
        This is useful for brain membrane simulation where hundreds of separate
        objects need individual properties and scripts in Unity.
        
        Args:
            output_dir: Output directory for OBJ files
            activity_data: Optional activity values for each region [N_regions, ...]
            region_positions: Optional dict mapping region_id to [x,y,z] position
            min_radius: Minimum sphere radius
            max_radius: Maximum sphere radius (for high activity)
            prefix: Filename prefix for region files
        
        Returns:
            List of paths to exported OBJ files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        # Determine regions to process
        if region_positions:
            regions_to_process = region_positions.items()
        else:
            regions_to_process = [(int(rid), info) for rid, info in self.regions_info.items()]
        
        for region_id, region_data in regions_to_process:
            if isinstance(region_data, dict):
                # From atlas info
                region_idx = region_id - 1
                xyz = region_data.get('xyz', [0, 0, 0])
                label = region_data.get('label', f'Region_{region_id}')
            else:
                # Direct position array
                region_idx = region_id - 1
                xyz = region_data if isinstance(region_data, (list, np.ndarray)) else [0, 0, 0]
                label = f'Region_{region_id}'
            
            center = np.array(xyz)
            
            # Calculate radius based on activity
            if activity_data is not None and region_idx < len(activity_data):
                # Get activity value
                if len(activity_data.shape) == 1:
                    activity = activity_data[region_idx].item()
                else:
                    activity = activity_data[region_idx].mean().item()
                
                # Normalize activity to [0, 1]
                activity_norm = (activity + 3.0) / 6.0
                activity_norm = max(0.0, min(1.0, activity_norm))
                
                # Map to radius
                radius = min_radius + (max_radius - min_radius) * activity_norm
            else:
                radius = (min_radius + max_radius) / 2
            
            # Generate sphere geometry
            vertices, normals, faces = self.generate_sphere_vertices(
                center, radius, self.sphere_resolution
            )
            
            # Write individual OBJ file
            output_path = output_dir / f"{prefix}_{region_id:04d}.obj"
            with open(output_path, 'w') as f:
                # Write header
                f.write("# TwinBrain Brain Region (Individual Export)\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Region ID: {region_id}\n")
                f.write(f"# Label: {label}\n")
                f.write(f"# Position: {xyz}\n")
                f.write(f"# Radius: {radius:.2f}\n")
                if activity_data is not None:
                    f.write(f"# Activity: {activity_norm:.3f}\n")
                f.write(f"g region_{region_id}\n\n")
                
                # Write vertices
                for vertex in vertices:
                    f.write(f"v {vertex[0]:.4f} {vertex[1]:.4f} {vertex[2]:.4f}\n")
                
                # Write normals
                for normal in normals:
                    f.write(f"vn {normal[0]:.4f} {normal[1]:.4f} {normal[2]:.4f}\n")
                
                # Write faces (indices start at 1 for each file)
                for face in faces:
                    v1, v2, v3 = face
                    f.write(f"f {v1+1}//{v1+1} {v2+1}//{v2+1} {v3+1}//{v3+1}\n")
            
            exported_files.append(output_path)
        
        return exported_files
    
    def export_brain_sequence(
        self,
        output_dir: Path,
        activity_sequence: torch.Tensor,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1
    ):
        """
        Export a sequence of brain models over time.
        
        Args:
            output_dir: Output directory
            activity_sequence: Activity data [N_regions, T, Features]
            start: Start time point
            end: End time point
            step: Time step
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        n_regions, n_timepoints, _ = activity_sequence.shape
        if end is None:
            end = n_timepoints
        
        end = min(end, n_timepoints)
        
        # Export each time point
        for t in range(start, end, step):
            output_path = output_dir / f"brain_t{t:04d}.obj"
            activity = activity_sequence[:, t, :]
            
            self.export_brain_model(
                output_path=output_path,
                activity_data=activity
            )
    
    def export_connections(
        self,
        output_path: Path,
        connectivity_matrix: np.ndarray,
        threshold: float = 0.5,
        line_segments: int = 10
    ):
        """
        Export brain connections as lines in OBJ format.
        
        Args:
            output_path: Output OBJ file path
            connectivity_matrix: Connectivity matrix [N_regions, N_regions]
            threshold: Minimum connection strength to include
            line_segments: Number of segments per connection line
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# TwinBrain Brain Connections\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            
            vertex_offset = 1
            
            # Find strong connections
            n_regions = connectivity_matrix.shape[0]
            
            for i in range(n_regions):
                for j in range(i + 1, n_regions):
                    strength = connectivity_matrix[i, j]
                    
                    if abs(strength) >= threshold:
                        # Get region positions
                        region_i = self.regions_info.get(str(i + 1), {})
                        region_j = self.regions_info.get(str(j + 1), {})
                        
                        pos_i = np.array(region_i.get('xyz', [0, 0, 0]))
                        pos_j = np.array(region_j.get('xyz', [0, 0, 0]))
                        
                        # Create line segments
                        f.write(f"# Connection {i} -> {j}: strength={strength:.3f}\n")
                        f.write(f"g connection_{i}_{j}\n")
                        
                        # Generate line vertices
                        for seg in range(line_segments + 1):
                            t = seg / line_segments
                            pos = pos_i + t * (pos_j - pos_i)
                            f.write(f"v {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n")
                        
                        # Create line strip
                        f.write("l")
                        for seg in range(line_segments + 1):
                            f.write(f" {vertex_offset + seg}")
                        f.write("\n\n")
                        
                        vertex_offset += line_segments + 1
    
    def export_complete_scene(
        self,
        output_dir: Path,
        activity_data: Optional[torch.Tensor] = None,
        connectivity_matrix: Optional[np.ndarray] = None,
        export_connections: bool = True
    ):
        """
        Export complete brain scene (regions + connections).
        
        Args:
            output_dir: Output directory
            activity_data: Activity values for regions
            connectivity_matrix: Connection strengths
            export_connections: Whether to export connections
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export brain regions
        regions_path = output_dir / "brain_regions.obj"
        self.export_brain_model(
            output_path=regions_path,
            activity_data=activity_data
        )
        
        # Export connections if available
        if export_connections and connectivity_matrix is not None:
            connections_path = output_dir / "brain_connections.obj"
            self.export_connections(
                output_path=connections_path,
                connectivity_matrix=connectivity_matrix
            )
