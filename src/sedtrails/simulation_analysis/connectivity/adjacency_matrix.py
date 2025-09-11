"""
Adjacency Matrix Compiler for SedTrails Connectivity Analysis

This module compiles adjacency matrices from particle trajectory data with various
temporal aggregation options.
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Literal
from pathlib import Path

class AdjacencyMatrixCompiler:
    """
    Compiles adjacency matrices from SedTrails trajectory data.
    
    Supports multiple compilation modes:
    - 'final': Final position connectivity
    - 'all_time': All timesteps aggregated 
    - 'time_varying': 3D time-varying matrix
    """
    
    def __init__(self, trajectory_file: str):
        """
        Initialize the compiler with trajectory data.
        
        Parameters
        ----------
        trajectory_file : str
            Path to SedTrails NetCDF trajectory file
        """
        self.trajectory_file = Path(trajectory_file)
        self.trajectory_data = None
        self.adjacency_matrices = {}
        
    def load_trajectory_data(self) -> xr.Dataset:
        """Load trajectory data from NetCDF file."""
        if self.trajectory_data is None:
            self.trajectory_data = xr.open_dataset(self.trajectory_file)
        return self.trajectory_data
    
    def compile_adjacency_matrix(
        self,
        polygon_file: str,
        mode: Literal['final', 'all_time', 'time_varying'] = 'all_time',
        time_indices: Optional[List[int]] = None,
        min_particles_threshold: int = 1
    ) -> Dict:
        """
        Compile adjacency matrix from trajectory data.
        
        Parameters
        ----------
        polygon_file : str
            Path to connectivity polygons file
        mode : str
            Compilation mode ('final', 'all_time', 'time_varying')
        time_indices : List[int], optional
            Specific time indices to use (for time_varying mode)
        min_particles_threshold : int
            Minimum particles required for connection
            
        Returns
        -------
        Dict
            Dictionary containing adjacency matrix and metadata
        """
        # Load data
        ds = self.load_trajectory_data()
        polygons = self._load_polygons(polygon_file)
        
        # Extract trajectory data
        x_data = ds['x'].values  # (n_particles, n_timesteps)
        y_data = ds['y'].values
        time_data = ds['time'].values

        n_particles, n_timesteps = x_data.shape
        
        # Assign particles to source polygons
        source_assignments = self._assign_particles_to_polygons(
            x_data[:, 0], y_data[:, 0], polygons
        )
        
        if mode == 'final':
            # Use only final positions
            return self._compile_final_connectivity(
                x_data, y_data, source_assignments, polygons, 
                min_particles_threshold, ds.attrs
            )
            
        elif mode == 'all_time':
            # Aggregate all timesteps
            return self._compile_all_time_connectivity(
                x_data, y_data, source_assignments, polygons,
                min_particles_threshold, ds.attrs
            )
            
        elif mode == 'time_varying':
            # 3D time-varying matrix
            return self._compile_time_varying_connectivity(
                x_data, y_data, source_assignments, polygons,
                time_indices, min_particles_threshold, ds.attrs, time_data
            )
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _load_polygons(self, polygon_file: str) -> List[np.ndarray]:
        """Load connectivity polygons from file."""
        import pickle
        
        try:
            # Try to load as pickle file first (new format)
            with open(polygon_file, 'rb') as f:
                polygons = pickle.load(f)
            if isinstance(polygons, list):
                return polygons
            else:
                return [polygons]
        except (pickle.UnpicklingError, FileNotFoundError):
            try:
                # Fallback to NumPy file (old format)
                polygons = np.load(polygon_file, allow_pickle=True)
                if isinstance(polygons, np.ndarray) and polygons.dtype == object:
                    return list(polygons)
                else:
                    # Convert to list of arrays
                    return [polygons]
            except Exception:
                # Try other formats (CSV, shapefile, etc.)
                raise NotImplementedError("Polygon loading not fully implemented yet") from None
    
    def _assign_particles_to_polygons(
        self, 
        x_positions: np.ndarray, 
        y_positions: np.ndarray, 
        polygons: List[np.ndarray]
    ) -> np.ndarray:
        """
        Assign particles to source polygons based on initial positions.
        
        Returns
        -------
        np.ndarray
            Array of polygon indices for each particle (-1 if not in any polygon)
        """
        from matplotlib.path import Path as MplPath
        
        n_particles = len(x_positions)
        assignments = np.full(n_particles, -1, dtype=int)
        
        for poly_idx, polygon in enumerate(polygons):
            if polygon.shape[1] >= 2:  # Ensure we have x,y coordinates
                path = MplPath(polygon[:, :2])  # Use first two columns as x,y
                mask = path.contains_points(np.column_stack([x_positions, y_positions]))
                assignments[mask] = poly_idx
                
        return assignments
    
    def _compile_final_connectivity(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray, 
        source_assignments: np.ndarray,
        polygons: List[np.ndarray],
        min_threshold: int,
        metadata: Dict
    ) -> Dict:
        """Compile connectivity using final particle positions."""
        n_polygons = len(polygons)
        adjacency_matrix = np.zeros((n_polygons, n_polygons), dtype=int)
        
        # Get final positions (last valid timestep for each particle)
        for particle_idx in range(x_data.shape[0]):
            # Find last valid position
            valid_mask = ~(np.isnan(x_data[particle_idx, :]) | np.isnan(y_data[particle_idx, :]))
            if not np.any(valid_mask):
                continue
                
            valid_indices = np.where(valid_mask)[0]
            final_idx = valid_indices[-1]
            
            final_x = x_data[particle_idx, final_idx]
            final_y = y_data[particle_idx, final_idx]
            source_poly = source_assignments[particle_idx]
            
            if source_poly == -1:  # Particle not assigned to source polygon
                continue
                
            # Find destination polygon
            dest_poly = self._find_polygon_for_position(final_x, final_y, polygons)
            
            if dest_poly != -1:
                adjacency_matrix[source_poly, dest_poly] += 1
        
        # Apply threshold
        adjacency_matrix[adjacency_matrix < min_threshold] = 0
        
        return {
            'adjacency_matrix': adjacency_matrix,
            'mode': 'final',
            'n_polygons': n_polygons,
            'polygon_names': [f'Polygon_{i}' for i in range(n_polygons)],
            'metadata': metadata,
            'min_threshold': min_threshold
        }
    
    def _compile_all_time_connectivity(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        source_assignments: np.ndarray, 
        polygons: List[np.ndarray],
        min_threshold: int,
        metadata: Dict
    ) -> Dict:
        """Compile connectivity aggregating all timesteps."""
        n_polygons = len(polygons)
        adjacency_matrix = np.zeros((n_polygons, n_polygons), dtype=int)
        
        # Count connections at all timesteps
        for particle_idx in range(x_data.shape[0]):
            source_poly = source_assignments[particle_idx]
            if source_poly == -1:
                continue
                
            # Check all timesteps for this particle
            for time_idx in range(x_data.shape[1]):
                x_pos = x_data[particle_idx, time_idx] 
                y_pos = y_data[particle_idx, time_idx]
                
                if np.isnan(x_pos) or np.isnan(y_pos):
                    continue
                    
                dest_poly = self._find_polygon_for_position(x_pos, y_pos, polygons)
                if dest_poly != -1:
                    adjacency_matrix[source_poly, dest_poly] += 1
        
        # Apply threshold
        adjacency_matrix[adjacency_matrix < min_threshold] = 0
        
        return {
            'adjacency_matrix': adjacency_matrix,
            'mode': 'all_time', 
            'n_polygons': n_polygons,
            'polygon_names': [f'Polygon_{i}' for i in range(n_polygons)],
            'metadata': metadata,
            'min_threshold': min_threshold
        }
    
    def _compile_time_varying_connectivity(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        source_assignments: np.ndarray,
        polygons: List[np.ndarray], 
        time_indices: Optional[List[int]],
        min_threshold: int,
        metadata: Dict,
        time_data: np.ndarray
    ) -> Dict:
        """Compile 3D time-varying connectivity matrix."""
        n_polygons = len(polygons)
        
        if time_indices is None:
            time_indices = list(range(x_data.shape[1]))
            
        n_times = len(time_indices)
        adjacency_matrices = np.zeros((n_times, n_polygons, n_polygons), dtype=int)
        
        for t_idx, time_step in enumerate(time_indices):
            # Compile matrix for this timestep
            for particle_idx in range(x_data.shape[0]):
                source_poly = source_assignments[particle_idx]
                if source_poly == -1:
                    continue
                    
                x_pos = x_data[particle_idx, time_step]
                y_pos = y_data[particle_idx, time_step]
                
                if np.isnan(x_pos) or np.isnan(y_pos):
                    continue
                    
                dest_poly = self._find_polygon_for_position(x_pos, y_pos, polygons)
                if dest_poly != -1:
                    adjacency_matrices[t_idx, source_poly, dest_poly] += 1
        
        # Apply threshold to all matrices
        adjacency_matrices[adjacency_matrices < min_threshold] = 0
        
        return {
            'adjacency_matrices': adjacency_matrices,  # 3D array
            'mode': 'time_varying',
            'n_polygons': n_polygons,
            'n_timesteps': n_times,
            'time_indices': time_indices,
            'time_values': time_data[time_indices] if len(time_indices) <= len(time_data) else None,
            'polygon_names': [f'Polygon_{i}' for i in range(n_polygons)],
            'metadata': metadata,
            'min_threshold': min_threshold
        }
    
    def _find_polygon_for_position(
        self, 
        x: float, 
        y: float, 
        polygons: List[np.ndarray]
    ) -> int:
        """Find which polygon contains the given position."""
        from matplotlib.path import Path as MplPath
        
        for poly_idx, polygon in enumerate(polygons):
            if polygon.shape[1] >= 2:
                path = MplPath(polygon[:, :2])
                if path.contains_point((x, y)):
                    return poly_idx
        return -1
