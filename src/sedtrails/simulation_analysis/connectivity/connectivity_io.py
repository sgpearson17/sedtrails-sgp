"""
Connectivity I/O Module for SedTrails

This module handles reading and writing connectivity data in various formats,
including NetCDF files for adjacency matrices and network metadata.
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from datetime import datetime

class ConnectivityIO:
    """
    Handles input/output operations for connectivity data.
    
    Supports reading and writing adjacency matrices, polygon definitions,
    and network metadata in standard formats.
    """
    
    @staticmethod
    def save_adjacency_netcdf(
        adjacency_data: Dict,
        filename: str,
        compression: Optional[str] = 'zlib'
    ) -> None:
        """
        Save adjacency matrix data to NetCDF file.
        
        Parameters
        ----------
        adjacency_data : Dict
            Dictionary containing adjacency matrix and metadata
        filename : str
            Output filename
        compression : str, optional
            Compression method for NetCDF
        """
        output_path = Path(filename)
        
        # Create dataset based on matrix type
        if adjacency_data['mode'] == 'time_varying':
            adj_matrices = adjacency_data['adjacency_matrices']
            n_times, n_nodes, _ = adj_matrices.shape
            
            # Create coordinates
            coords = {
                'time_index': np.arange(n_times),
                'source_node': np.arange(n_nodes),
                'dest_node': np.arange(n_nodes)
            }
            
            # Add time values if available
            if 'time_values' in adjacency_data and adjacency_data['time_values'] is not None:
                coords['time_values'] = ('time_index', adjacency_data['time_values'])
            
            # Create data variables
            data_vars = {
                'adjacency_matrix': (
                    ['time_index', 'source_node', 'dest_node'], 
                    adj_matrices,
                    {'long_name': 'Time-varying adjacency matrix',
                     'units': 'particle_count'}
                )
            }
            
        else:
            # Static adjacency matrix
            adj_matrix = adjacency_data['adjacency_matrix']
            n_nodes = adj_matrix.shape[0]
            
            coords = {
                'source_node': np.arange(n_nodes),
                'dest_node': np.arange(n_nodes)
            }
            
            data_vars = {
                'adjacency_matrix': (
                    ['source_node', 'dest_node'],
                    adj_matrix,
                    {'long_name': f'{adjacency_data["mode"]} adjacency matrix',
                     'units': 'particle_count'}
                )
            }
        
        # Add polygon names if available
        polygon_names = adjacency_data.get('polygon_names', [])
        if polygon_names:
            # Convert strings to fixed-length character arrays for NetCDF
            max_len = max(len(name) for name in polygon_names) if polygon_names else 24
            name_array = np.array([name.ljust(max_len)[:max_len] for name in polygon_names])
            
            coords['polygon_name_chars'] = np.arange(max_len)
            data_vars['polygon_names'] = (
                ['source_node', 'polygon_name_chars'],
                np.array([list(name.encode('utf-8')[:max_len].ljust(max_len, b'\x00')) 
                         for name in polygon_names]),
                {'long_name': 'Polygon names'}
            )
        
        # Create dataset
        ds = xr.Dataset(data_vars, coords=coords)
        
        # Add global attributes
        ds.attrs.update({
            'title': 'SedTrails Connectivity Analysis - Adjacency Matrix',
            'institution': 'SedTrails Connectivity Analysis System',
            'created_on': datetime.now().isoformat(),
            'mode': adjacency_data['mode'],
            'n_polygons': adjacency_data['n_polygons'],
            'min_threshold': adjacency_data.get('min_threshold', 1)
        })
        
        # Add source metadata if available
        if 'metadata' in adjacency_data:
            source_metadata = adjacency_data['metadata']
            for key, value in source_metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    ds.attrs[f'source_{key}'] = value
        
        # Save with compression
        encoding = {}
        if compression:
            for var in data_vars.keys():
                encoding[var] = {'zlib': True, 'complevel': 6}
        
        try:
            # Try with netcdf4 backend and compression
            ds.to_netcdf(output_path, engine='netcdf4', encoding=encoding)
        except (ImportError, ValueError):
            # Fallback to scipy backend without compression
            if compression:
                print("Warning: Using scipy backend without compression")
            ds.to_netcdf(output_path, engine='scipy')
        
        print(f"Adjacency data saved to: {output_path}")
    
    @staticmethod
    def load_adjacency_netcdf(filename: str) -> Dict:
        """
        Load adjacency matrix data from NetCDF file.
        
        Parameters
        ----------
        filename : str
            Input filename
            
        Returns
        -------
        Dict
            Dictionary containing adjacency matrix and metadata
        """
        input_path = Path(filename)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # Try to load with netcdf4 engine first, fallback to scipy
        try:
            ds = xr.open_dataset(input_path, engine='netcdf4')
        except (ImportError, ValueError):
            try:
                ds = xr.open_dataset(input_path, engine='scipy')
            except Exception:
                # Final fallback - let xarray decide
                ds = xr.open_dataset(input_path)
        
        # Extract adjacency matrix
        adj_matrix = ds['adjacency_matrix'].values
        mode = ds.attrs.get('mode', 'unknown')
        
        # Build result dictionary
        result = {
            'mode': mode,
            'n_polygons': ds.attrs.get('n_polygons', adj_matrix.shape[-1]),
            'min_threshold': ds.attrs.get('min_threshold', 1)
        }
        
        if mode == 'time_varying':
            result['adjacency_matrices'] = adj_matrix
            result['n_timesteps'] = adj_matrix.shape[0]
            if 'time_values' in ds.coords:
                result['time_values'] = ds['time_values'].values
            if 'time_index' in ds.coords:
                result['time_indices'] = ds['time_index'].values
        else:
            result['adjacency_matrix'] = adj_matrix
        
        # Extract polygon names
        if 'polygon_names' in ds:
            names_array = ds['polygon_names'].values
            polygon_names = []
            for name_chars in names_array:
                name = ''.join([chr(c) for c in name_chars if c != 0]).strip()
                polygon_names.append(name)
            result['polygon_names'] = polygon_names
        else:
            result['polygon_names'] = [f'Polygon_{i}' for i in range(result['n_polygons'])]
        
        # Extract metadata
        metadata = {}
        for key, value in ds.attrs.items():
            if key.startswith('source_'):
                metadata[key[7:]] = value  # Remove 'source_' prefix
        result['metadata'] = metadata
        
        ds.close()
        return result
    
    @staticmethod
    def save_polygons_netcdf(
        polygons: List,
        polygon_assignments: Optional[np.ndarray],
        source_points: np.ndarray,
        filename: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save connectivity polygons to NetCDF file.
        
        Parameters
        ----------
        polygons : List
            List of polygon coordinate arrays
        polygon_assignments : np.ndarray, optional
            Source point assignments to polygons
        source_points : np.ndarray
            Source point coordinates
        filename : str
            Output filename
        metadata : Dict, optional
            Additional metadata
        """
        output_path = Path(filename)
        
        # Convert polygons to coordinate arrays
        max_vertices = max(len(poly) if hasattr(poly, '__len__') else 0 for poly in polygons)
        n_polygons = len(polygons)
        
        # Create polygon coordinate array
        poly_coords = np.full((n_polygons, max_vertices, 2), np.nan)
        poly_lengths = np.zeros(n_polygons, dtype=int)
        
        for i, poly in enumerate(polygons):
            if hasattr(poly, 'exterior'):  # Shapely polygon
                coords = np.array(poly.exterior.coords[:-1])  # Remove last duplicate point
            elif hasattr(poly, '__len__') and len(poly) > 0:
                coords = np.array(poly)
            else:
                coords = np.array([]).reshape(0, 2)
                
            if coords.size > 0:
                n_coords = min(len(coords), max_vertices)
                poly_coords[i, :n_coords, :] = coords[:n_coords]
                poly_lengths[i] = n_coords
        
        # Create coordinates
        coords = {
            'polygon_id': np.arange(n_polygons),
            'vertex_id': np.arange(max_vertices),
            'coord_dim': ['x', 'y'],
            'source_point_id': np.arange(len(source_points))
        }
        
        # Create data variables
        data_vars = {
            'polygon_coordinates': (
                ['polygon_id', 'vertex_id', 'coord_dim'],
                poly_coords,
                {'long_name': 'Polygon vertex coordinates',
                 'units': 'm'}
            ),
            'polygon_lengths': (
                ['polygon_id'],
                poly_lengths,
                {'long_name': 'Number of vertices per polygon'}
            ),
            'source_points': (
                ['source_point_id', 'coord_dim'],
                source_points,
                {'long_name': 'Source point coordinates',
                 'units': 'm'}
            )
        }
        
        if polygon_assignments is not None:
            data_vars['polygon_assignments'] = (
                ['source_point_id'],
                polygon_assignments,
                {'long_name': 'Polygon assignment for each source point',
                 'description': '-1 indicates no assignment'}
            )
        
        # Create dataset
        ds = xr.Dataset(data_vars, coords=coords)
        
        # Add global attributes
        ds.attrs.update({
            'title': 'SedTrails Connectivity Polygons',
            'institution': 'SedTrails Connectivity Analysis System',
            'created_on': datetime.now().isoformat(),
            'n_polygons': n_polygons,
            'n_source_points': len(source_points),
            'max_vertices_per_polygon': max_vertices
        })
        
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    ds.attrs[key] = value
        
        # Save to file
        try:
            # Try with netcdf4 backend and compression
            encoding = {
                'polygon_coordinates': {'zlib': True, 'complevel': 6},
                'source_points': {'zlib': True, 'complevel': 6}
            }
            ds.to_netcdf(output_path, engine='netcdf4', encoding=encoding)
        except (ImportError, ValueError):
            # Fallback to scipy backend without compression
            print("Warning: Using scipy backend without compression")
            ds.to_netcdf(output_path, engine='scipy')
        
        print(f"Polygon data saved to: {output_path}")
    
    @staticmethod
    def load_polygons_netcdf(filename: str) -> Dict:
        """
        Load connectivity polygons from NetCDF file.
        
        Parameters
        ----------
        filename : str
            Input filename
            
        Returns
        -------
        Dict
            Dictionary containing polygons and metadata
        """
        input_path = Path(filename)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # Try to load with netcdf4 engine first, fallback to scipy
        try:
            ds = xr.open_dataset(input_path, engine='netcdf4')
        except (ImportError, ValueError):
            try:
                ds = xr.open_dataset(input_path, engine='scipy')
            except Exception:
                # Final fallback - let xarray decide
                ds = xr.open_dataset(input_path)
        
        # Extract polygons
        poly_coords = ds['polygon_coordinates'].values
        poly_lengths = ds['polygon_lengths'].values
        
        polygons = []
        for i in range(len(poly_lengths)):
            length = poly_lengths[i]
            if length > 0:
                coords = poly_coords[i, :length, :]
                polygons.append(coords)
            else:
                polygons.append(np.array([]).reshape(0, 2))
        
        # Extract other data
        source_points = ds['source_points'].values
        
        result = {
            'polygons': polygons,
            'source_points': source_points,
            'n_polygons': ds.attrs.get('n_polygons', len(polygons)),
            'n_source_points': ds.attrs.get('n_source_points', len(source_points))
        }
        
        if 'polygon_assignments' in ds:
            result['polygon_assignments'] = ds['polygon_assignments'].values
        
        # Extract metadata
        metadata = {}
        for key, value in ds.attrs.items():
            if key not in ['title', 'institution', 'created_on', 'n_polygons', 
                          'n_source_points', 'max_vertices_per_polygon']:
                metadata[key] = value
        result['metadata'] = metadata
        
        ds.close()
        return result
    
    @staticmethod
    def export_network_metrics(
        metrics: Dict,
        filename: str,
        format: str = 'json'
    ) -> None:
        """
        Export network metrics to file.
        
        Parameters
        ---------- 
        metrics : Dict
            Network metrics dictionary
        filename : str
            Output filename
        format : str
            Output format ('json', 'csv')
        """
        output_path = Path(filename)
        
        if format == 'json':
            # Convert numpy types to Python types for JSON serialization
            json_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    json_metrics[key] = {k: float(v) if isinstance(v, np.floating) 
                                       else int(v) if isinstance(v, np.integer)
                                       else v for k, v in value.items()}
                elif isinstance(value, (np.floating, float)):
                    json_metrics[key] = float(value)
                elif isinstance(value, (np.integer, int)):
                    json_metrics[key] = int(value)
                else:
                    json_metrics[key] = value
            
            with open(output_path, 'w') as f:
                json.dump(json_metrics, f, indent=2)
                
        elif format == 'csv':
            import pandas as pd
            
            # Flatten nested dictionaries
            flat_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_metrics[f"{key}_{sub_key}"] = sub_value
                else:
                    flat_metrics[key] = value
            
            df = pd.DataFrame([flat_metrics])
            df.to_csv(output_path, index=False)
            
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Network metrics exported to: {output_path}")
