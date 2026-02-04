"""
FreeSurfer Surface File Loader
================================

Load FreeSurfer surface files (.pial) and annotation files (.annot)
for use in Unity visualization workflow.

Supports:
- Loading left/right hemisphere surface meshes
- Loading parcellation annotations (e.g., Schaefer200)
- Extracting region centroids and labels
- Converting to TwinBrain atlas format
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging


class FreeSurferLoader:
    """
    Load and process FreeSurfer surface and annotation files.
    
    This class handles:
    - Reading .pial surface files (vertices and faces)
    - Reading .annot annotation files (region labels and colors)
    - Computing region centroids from surface parcellation
    - Converting to TwinBrain atlas format
    """
    
    def __init__(self):
        """Initialize FreeSurfer loader."""
        self.logger = logging.getLogger(__name__)
        self.surfaces = {}  # Store loaded surfaces
        self.annotations = {}  # Store loaded annotations
        self.atlas_info = None  # Store converted atlas info
    
    def load_surface(self, surface_path: Path, hemisphere: str = 'lh') -> Dict[str, np.ndarray]:
        """
        Load FreeSurfer surface file (.pial).
        
        Args:
            surface_path: Path to .pial file (e.g., lh.pial)
            hemisphere: 'lh' or 'rh'
        
        Returns:
            Dictionary with 'vertices' and 'faces' arrays
        """
        surface_path = Path(surface_path)
        if not surface_path.exists():
            raise FileNotFoundError(f"Surface file not found: {surface_path}")
        
        self.logger.info(f"Loading {hemisphere} surface from {surface_path}")
        
        # Load surface using nibabel
        vertices, faces = nib.freesurfer.read_geometry(str(surface_path))
        
        self.surfaces[hemisphere] = {
            'vertices': vertices,
            'faces': faces,
            'n_vertices': len(vertices),
            'n_faces': len(faces)
        }
        
        self.logger.info(f"  Loaded {len(vertices)} vertices, {len(faces)} faces")
        return self.surfaces[hemisphere]
    
    def load_annotation(self, annot_path: Path, hemisphere: str = 'lh') -> Dict[str, Any]:
        """
        Load FreeSurfer annotation file (.annot).
        
        Args:
            annot_path: Path to .annot file (e.g., lh.Schaefer2018_200Parcels_7Networks_order.annot)
            hemisphere: 'lh' or 'rh'
        
        Returns:
            Dictionary with annotation data
        """
        annot_path = Path(annot_path)
        if not annot_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annot_path}")
        
        self.logger.info(f"Loading {hemisphere} annotation from {annot_path}")
        
        # Load annotation using nibabel
        labels, ctab, names = nib.freesurfer.read_annot(str(annot_path))
        
        # Decode region names if they are bytes
        if len(names) > 0 and isinstance(names[0], bytes):
            names = [name.decode('utf-8') for name in names]
        
        self.annotations[hemisphere] = {
            'labels': labels,  # Per-vertex region labels
            'ctab': ctab,  # Color table
            'names': names,  # Region names
            'n_regions': len(names)
        }
        
        self.logger.info(f"  Loaded {len(names)} regions")
        return self.annotations[hemisphere]
    
    def compute_region_centroids(self, hemisphere: str = 'lh') -> np.ndarray:
        """
        Compute centroid position for each region.
        
        Args:
            hemisphere: 'lh' or 'rh'
        
        Returns:
            Array of centroids [n_regions, 3]
        """
        if hemisphere not in self.surfaces:
            raise ValueError(f"Surface not loaded for {hemisphere}")
        if hemisphere not in self.annotations:
            raise ValueError(f"Annotation not loaded for {hemisphere}")
        
        vertices = self.surfaces[hemisphere]['vertices']
        labels = self.annotations[hemisphere]['labels']
        names = self.annotations[hemisphere]['names']
        ctab = self.annotations[hemisphere]['ctab']
        
        centroids = []
        
        # Compute centroid for each region
        # FreeSurfer labels array contains color-table indices per vertex
        # ctab rows correspond to region indices
        for region_idx in range(len(names)):
            # Find vertices that belong to this region
            # The labels array contains indices into the color table
            region_mask = (labels == region_idx)
            region_vertices = vertices[region_mask]
            
            if len(region_vertices) > 0:
                # Compute centroid as mean of all vertices in region
                centroid = region_vertices.mean(axis=0)
                centroids.append(centroid)
            else:
                # If no vertices found, use origin (shouldn't happen normally)
                region_name = names[region_idx]
                self.logger.warning(f"No vertices found for region {region_name}")
                centroids.append(np.array([0, 0, 0]))
        
        return np.array(centroids)
    
    def load_bilateral_surfaces(
        self,
        lh_surface_path: Path,
        rh_surface_path: Path,
        lh_annot_path: Path,
        rh_annot_path: Path
    ) -> Dict[str, Any]:
        """
        Load both left and right hemisphere surfaces and annotations.
        
        Args:
            lh_surface_path: Path to lh.pial
            rh_surface_path: Path to rh.pial
            lh_annot_path: Path to lh annotation file
            rh_annot_path: Path to rh annotation file
        
        Returns:
            Combined data for both hemispheres
        """
        # Load left hemisphere
        self.load_surface(lh_surface_path, 'lh')
        self.load_annotation(lh_annot_path, 'lh')
        
        # Load right hemisphere
        self.load_surface(rh_surface_path, 'rh')
        self.load_annotation(rh_annot_path, 'rh')
        
        return {
            'lh': {
                'surface': self.surfaces['lh'],
                'annotation': self.annotations['lh']
            },
            'rh': {
                'surface': self.surfaces['rh'],
                'annotation': self.annotations['rh']
            }
        }
    
    def to_atlas_info(self, atlas_name: str = "FreeSurfer") -> Dict[str, Any]:
        """
        Convert loaded FreeSurfer data to TwinBrain atlas format.
        
        Args:
            atlas_name: Name for the atlas
        
        Returns:
            Atlas info dictionary compatible with TwinBrain exporters
        """
        if not self.surfaces or not self.annotations:
            raise ValueError("Must load surfaces and annotations first")
        
        atlas_info = {
            'name': atlas_name,
            'regions': {},
            'n_regions': 0
        }
        
        region_id = 1  # TwinBrain uses 1-based indexing
        
        # Process each hemisphere
        for hemisphere in ['lh', 'rh']:
            if hemisphere not in self.surfaces or hemisphere not in self.annotations:
                continue
            
            # Compute centroids for this hemisphere
            centroids = self.compute_region_centroids(hemisphere)
            names = self.annotations[hemisphere]['names']
            ctab = self.annotations[hemisphere]['ctab']
            
            # Add each region to atlas info
            for region_idx, (region_name, centroid) in enumerate(zip(names, centroids)):
                # Skip background/unknown regions (usually index 0)
                if region_name.lower() in ['unknown', 'corpus_callosum', 'corpuscallosum', '???']:
                    continue
                
                # Get color from color table
                if ctab is not None and region_idx < len(ctab):
                    color = ctab[region_idx, :3]  # RGB values
                else:
                    color = [128, 128, 128]  # Default gray
                
                # Determine network from region name (for Schaefer parcellation)
                network = self._extract_network_from_name(region_name)
                
                atlas_info['regions'][str(region_id)] = {
                    'label': region_name,
                    'hemisphere': hemisphere,
                    'xyz': centroid.tolist(),
                    'network': network,
                    'color': color.tolist() if isinstance(color, np.ndarray) else color
                }
                
                region_id += 1
        
        atlas_info['n_regions'] = len(atlas_info['regions'])
        self.atlas_info = atlas_info
        
        self.logger.info(f"Created atlas with {atlas_info['n_regions']} regions")
        return atlas_info
    
    def _extract_network_from_name(self, region_name: str) -> str:
        """
        Extract network name from region label.
        
        For Schaefer parcellation, region names typically contain network info.
        E.g., "7Networks_LH_Vis_1" -> "Visual"
        
        Args:
            region_name: Region label string
        
        Returns:
            Network name
        """
        # Common network abbreviations in Schaefer parcellation
        network_mapping = {
            'Vis': 'Visual',
            'SomMot': 'Somatomotor',
            'DorsAttn': 'Dorsal Attention',
            'SalVentAttn': 'Ventral Attention',
            'Limbic': 'Limbic',
            'Cont': 'Frontoparietal',
            'Default': 'Default Mode'
        }
        
        for abbrev, full_name in network_mapping.items():
            if abbrev in region_name:
                return full_name
        
        return "Unknown"
    
    def export_surfaces_as_obj(
        self,
        output_dir: Path,
        combine_hemispheres: bool = True
    ) -> List[Path]:
        """
        Export FreeSurfer surfaces as OBJ files for Unity.
        
        Args:
            output_dir: Output directory for OBJ files
            combine_hemispheres: If True, export as single file; if False, separate files
        
        Returns:
            List of exported OBJ file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        if combine_hemispheres:
            # Export both hemispheres in one OBJ file
            output_path = output_dir / "brain_surface_bilateral.obj"
            self._write_obj_file(output_path, ['lh', 'rh'])
            exported_files.append(output_path)
        else:
            # Export separate OBJ files for each hemisphere
            for hemisphere in ['lh', 'rh']:
                if hemisphere in self.surfaces:
                    output_path = output_dir / f"brain_surface_{hemisphere}.obj"
                    self._write_obj_file(output_path, [hemisphere])
                    exported_files.append(output_path)
        
        return exported_files
    
    def _write_obj_file(self, output_path: Path, hemispheres: List[str]):
        """
        Write surface data to OBJ file.
        
        Args:
            output_path: Output OBJ file path
            hemispheres: List of hemispheres to include
        """
        with open(output_path, 'w') as f:
            f.write("# TwinBrain FreeSurfer Surface Export\n")
            f.write(f"# File: {output_path.name}\n\n")
            
            vertex_offset = 1  # OBJ uses 1-based indexing
            
            for hemisphere in hemispheres:
                if hemisphere not in self.surfaces:
                    continue
                
                vertices = self.surfaces[hemisphere]['vertices']
                faces = self.surfaces[hemisphere]['faces']
                
                # Write hemisphere comment
                f.write(f"# Hemisphere: {hemisphere}\n")
                f.write(f"g {hemisphere}_surface\n")
                
                # Write vertices
                for vertex in vertices:
                    f.write(f"v {vertex[0]:.4f} {vertex[1]:.4f} {vertex[2]:.4f}\n")
                
                # Write faces (triangles)
                # FreeSurfer faces are 0-indexed, OBJ needs 1-indexed
                for face in faces:
                    v1, v2, v3 = face + vertex_offset
                    f.write(f"f {v1} {v2} {v3}\n")
                
                f.write("\n")
                vertex_offset += len(vertices)
        
        self.logger.info(f"Exported surface to {output_path}")


def load_freesurfer_data(
    lh_surface: str,
    rh_surface: str,
    lh_annot: str,
    rh_annot: str,
    atlas_name: str = "FreeSurfer_Schaefer200"
) -> Tuple[Dict[str, Any], FreeSurferLoader]:
    """
    Convenience function to load FreeSurfer data and convert to atlas format.
    
    Args:
        lh_surface: Path to lh.pial
        rh_surface: Path to rh.pial
        lh_annot: Path to lh annotation file
        rh_annot: Path to rh annotation file
        atlas_name: Name for the atlas
    
    Returns:
        Tuple of (atlas_info, loader)
    
    Example:
        >>> atlas_info, loader = load_freesurfer_data(
        ...     lh_surface="data/lh.pial",
        ...     rh_surface="data/rh.pial",
        ...     lh_annot="data/lh.Schaefer2018_200Parcels_7Networks_order.annot",
        ...     rh_annot="data/rh.Schaefer2018_200Parcels_7Networks_order.annot"
        ... )
        >>> print(f"Loaded {atlas_info['n_regions']} regions")
    """
    loader = FreeSurferLoader()
    
    # Load both hemispheres
    loader.load_bilateral_surfaces(
        lh_surface_path=Path(lh_surface),
        rh_surface_path=Path(rh_surface),
        lh_annot_path=Path(lh_annot),
        rh_annot_path=Path(rh_annot)
    )
    
    # Convert to atlas format
    atlas_info = loader.to_atlas_info(atlas_name=atlas_name)
    
    return atlas_info, loader
