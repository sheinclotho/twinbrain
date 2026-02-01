import json
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class BrainAtlas:
    """Brain atlas loader and manager for brain region information.
    
    Loads atlas data from JSON files containing region positions, functions,
    and connection hints between regions.
    """
    
    def __init__(self, file_path: str = None):
        """Initialize BrainAtlas.
        
        Args:
            file_path: Path to atlas JSON file. If None, raises ValueError.
        """
        self.regions = {}
        self.connection_hints = {}

        if file_path:
            self.load_from_path(file_path)
        else:
            raise ValueError("file_path is required. GUI file selection has been removed.")

    def load_from_path(self, file_path: str):
        """Load brain atlas from given file path.
        
        Args:
            file_path: Path to JSON atlas file.
            
        Raises:
            FileNotFoundError: If atlas file doesn't exist.
            ValueError: If JSON format is invalid.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Atlas file not found: {file_path}")
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            self._parse_data(data)
            logger.info(f"Loaded atlas from {file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in atlas file: {e}")
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading atlas file: {e}")
            raise

    def _parse_data(self, data: dict):
        """Parse atlas JSON data and populate regions and connection_hints.
        
        Args:
            data: Dictionary containing 'regions' and optionally 'connection_hints'.
            
        Raises:
            ValueError: If JSON structure is invalid.
        """
        # Parse regions
        if "regions" not in data or not isinstance(data["regions"], dict):
            raise ValueError("Invalid JSON: 'regions' must be a dictionary")

        for region_id, region_data in data["regions"].items():
            try:
                pos = np.array(region_data.get("position"), dtype=np.float32)
                if pos.shape != (3,):
                    raise ValueError(f"Position must be 3D coordinates, got shape {pos.shape}")
                
                self.regions[region_id] = {
                    "position": pos,
                    "function": region_data.get("function", "unknown"),
                    "label_id": region_data.get("label_id")
                }
            except (TypeError, ValueError) as e:
                logger.warning(f"Skipping region {region_id} due to invalid data: {e}")
                continue

        # Parse connection hints
        if "connection_hints" in data:
            for key, weight in data["connection_hints"].items():
                try:
                    src, tgt = key.split("-")
                    self.connection_hints[(src, tgt)] = float(weight)
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Skipping connection hint {key} due to invalid format: {e}")
                    continue

    def get_region_by_label(self, label):
        """Find brain region by label_id or function name (fuzzy match).
        
        Args:
            label: Label ID or function name to search for.
            
        Returns:
            Tuple of (region_id, region_data) or (None, None) if not found.
        """
        for rid, rdata in self.regions.items():
            label_str = str(label).lower()
            if (str(rdata.get("label_id")) == str(label) or 
                label_str in str(rdata.get("function", "")).lower()):
                return rid, rdata
        return None, None

    def get_region_by_position(self, xyz, tol=3.0):
        """Find closest brain region by 3D coordinates.
        
        Args:
            xyz: 3D coordinates (x, y, z) in mm.
            tol: Distance threshold in mm. Default: 3.0.
            
        Returns:
            Tuple of (region_id, region_data) or (None, None) if no region within tolerance.
        """
        xyz = np.array(xyz, dtype=np.float32)
        best_rid, best_dist = None, float("inf")
        
        for rid, rdata in self.regions.items():
            dist = np.linalg.norm(rdata["position"] - xyz)
            if dist < best_dist and dist <= tol:
                best_rid, best_dist = rid, dist
        
        if best_rid is None:
            return None, None
        return best_rid, self.regions[best_rid]

    def get_region_id_list(self):
        """Get list of all region IDs.
        
        Returns:
            List of region ID strings.
        """
        return list(self.regions.keys())
