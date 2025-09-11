"""
Connectivity Polygon Generation

SedTrails

This module generates connectivity polygons for spatial aggregation of particles,
replacing the MATLAB Voronoi polygon functionality with more efficient Python methods.
"""

import numpy as np
from typing import List, Tuple, Optional, Literal, Dict
from scipy.spatial import Voronoi, ConvexHull
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class ConnectivityPolygonGenerator:
    """
    Generate connectivity polygons for spatial aggregation of particles.
    
    This class provides methods to create spatial polygons that define connectivity 
    regions for particle trajectory analysis.
    """
    
    def __init__(self, source_points: np.ndarray, boundary_polygon: Optional[np.ndarray] = None):
        """
        Initialize connectivity polygon generator.
        
        Parameters
        ----------
        source_points : np.ndarray
            Array of source coordinates, shape (n_sources, 2)
        boundary_polygon : np.ndarray, optional
            Boundary polygon vertices, shape (n_vertices, 2)
        """
        # Remove duplicate source points (within tolerance)
        source_points = np.array(source_points)
        if len(source_points) > 1:
            # Find unique points using a tolerance for floating point comparison
            tolerance = 1e-6
            unique_indices = []
            unique_points = []
            
            for i, point in enumerate(source_points):
                is_unique = True
                for existing_point in unique_points:
                    if np.linalg.norm(point - existing_point) < tolerance:
                        is_unique = False
                        break
                if is_unique:
                    unique_indices.append(i)
                    unique_points.append(point)
            
            self.source_points = np.array(unique_points)
            self.original_indices = unique_indices  # Track which original points were kept
            
            if len(unique_points) < len(source_points):
                print(f"Removed {len(source_points) - len(unique_points)} duplicate source points")
                print(f"Using {len(unique_points)} unique source locations")
        else:
            self.source_points = source_points
            self.original_indices = list(range(len(source_points)))
            
        self.boundary_polygon = boundary_polygon
        self.polygons = []
        self.polygon_assignments = None
        
    def generate_voronoi_polygons(
        self, 
        buffer_distance: float = 1000.0,
        clip_to_boundary: bool = True
    ) -> List[Polygon]:
        """
        Generate Voronoi polygons for each source point.
        
        Parameters
        ----------
        buffer_distance : float
            Buffer distance around source points for bounded Voronoi
        clip_to_boundary : bool
            Whether to clip polygons to boundary
            
        Returns
        -------
        List[Polygon]
            List of Shapely Polygon objects
        """
        if len(self.source_points) < 2:
            raise ValueError("Need at least 2 unique source points for Voronoi generation")
        
        # Create bounded Voronoi diagram
        buffer_box = self._create_buffer_box(buffer_distance)
        
        # Add buffer points to ensure finite regions
        buffered_points = np.vstack([self.source_points, buffer_box])
        
        # Generate Voronoi diagram
        vor = Voronoi(buffered_points)
        
        # Extract polygons for original points only
        polygons = []
        n_sources = len(self.source_points)
        
        for i in range(n_sources):
            region_index = vor.point_region[i]
            region = vor.regions[region_index]
            
            if len(region) > 0 and -1 not in region:
                # Valid finite region
                vertices = vor.vertices[region]
                polygon = Polygon(vertices)
                
                # Clip to boundary if specified
                if clip_to_boundary and self.boundary_polygon is not None:
                    boundary_poly = Polygon(self.boundary_polygon)
                    polygon = polygon.intersection(boundary_poly)
                
                polygons.append(polygon)
            else:
                # Create a small circular polygon as fallback
                center = self.source_points[i]
                polygon = Point(center).buffer(buffer_distance / 10)
                polygons.append(polygon)
        
        self.polygons = polygons
        return polygons
    
    def generate_clustered_polygons(
        self,
        n_clusters: int,
        method: str = 'kmeans',
        buffer_distance: float = 1000.0
    ) -> List[Polygon]:
        """
        Generate polygons by clustering source points.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters to create
        method : str
            Clustering method ('kmeans')
        buffer_distance : float
            Buffer distance for polygon creation
            
        Returns
        -------
        List[Polygon]
            List of clustered polygons
        """
        if n_clusters >= len(self.source_points):
            print(f"Warning: n_clusters ({n_clusters}) >= n_points ({len(self.source_points)})")
            n_clusters = max(1, len(self.source_points) - 1)
        
        # Perform clustering
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.source_points)
            self.polygon_assignments = labels
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Create polygons for each cluster
        polygons = []
        
        for cluster_id in range(n_clusters):
            cluster_points = self.source_points[labels == cluster_id]
            
            if len(cluster_points) >= 3:
                # Create convex hull
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]
                polygon = Polygon(hull_points).buffer(buffer_distance)
            elif len(cluster_points) == 2:
                # Create line buffer
                line_points = cluster_points
                polygon = Polygon(line_points).buffer(buffer_distance)
            else:
                # Single point - create circle
                center = cluster_points[0]
                polygon = Point(center).buffer(buffer_distance)
                
            polygons.append(polygon)
        
        self.polygons = polygons
        return polygons
    
    def _create_buffer_box(self, buffer_distance: float) -> np.ndarray:
        """Create buffer box around source points for bounded Voronoi."""
        min_x, min_y = self.source_points.min(axis=0)
        max_x, max_y = self.source_points.max(axis=0)
        
        # Extend by buffer distance
        buffer_box = np.array([
            [min_x - buffer_distance, min_y - buffer_distance],
            [max_x + buffer_distance, min_y - buffer_distance], 
            [max_x + buffer_distance, max_y + buffer_distance],
            [min_x - buffer_distance, max_y + buffer_distance]
        ])
        
        return buffer_box
    
    def save_polygons(self, filename: str, format: str = 'numpy'):
        """
        Save connectivity polygons to file.
        
        Parameters
        ----------
        filename : str
            Output filename
        format : str
            File format ('numpy', 'shapefile', 'geojson')
        """
        if not self.polygons:
            raise ValueError("No polygons generated yet")
            
        if format == 'numpy':
            # Convert to list of coordinate arrays
            poly_arrays = []
            for poly in self.polygons:
                if hasattr(poly, 'exterior'):
                    coords = np.array(poly.exterior.coords)
                    poly_arrays.append(coords)
                else:
                    # Empty polygon
                    poly_arrays.append(np.array([]))
            
            # Save as a pickled object to handle variable-length arrays
            import pickle
            with open(filename, 'wb') as f:
                pickle.dump(poly_arrays, f)
            
        elif format == 'shapefile':
            import geopandas as gpd
            gdf = gpd.GeoDataFrame({'geometry': self.polygons})
            gdf.to_file(filename)
            
        elif format == 'geojson':
            import geopandas as gpd
            gdf = gpd.GeoDataFrame({'geometry': self.polygons})
            gdf.to_file(filename, driver='GeoJSON')
            
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def plot_polygons(
        self, 
        show_points: bool = True,
        show_labels: bool = True,
        figsize: Tuple[float, float] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot connectivity polygons.
        
        Parameters
        ----------
        show_points : bool
            Whether to show source points
        show_labels : bool
            Whether to show polygon labels
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        if not self.polygons:
            raise ValueError("No polygons to plot")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot polygons
        for i, poly in enumerate(self.polygons):
            if hasattr(poly, 'exterior'):
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.3, facecolor=plt.cm.Set3(i % 12))
                ax.plot(x, y, 'k-', linewidth=1, alpha=0.8)
                
                # Add labels
                if show_labels:
                    centroid = poly.centroid
                    ax.text(centroid.x, centroid.y, str(i), 
                           fontsize=10, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Plot source points
        if show_points:
            ax.scatter(self.source_points[:, 0], self.source_points[:, 1], 
                      c='red', s=50, zorder=5, alpha=0.8, edgecolors='black')
        
        # Plot boundary if available
        if self.boundary_polygon is not None:
            bx, by = self.boundary_polygon[:, 0], self.boundary_polygon[:, 1]
            ax.plot(bx, by, 'k-', linewidth=2, label='Boundary')
            
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Connectivity Polygons (n={len(self.polygons)})')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig

    def plot_trajectories_on_polygons(
        self,
        trajectory_data: np.ndarray,
        max_trajectories: int = 50,
        figsize: Tuple[float, float] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot particle trajectories overlaid on connectivity polygons.
        
        Parameters
        ----------
        trajectory_data : np.ndarray
            Trajectory data with shape (n_particles, n_timesteps, 2)
        max_trajectories : int
            Maximum number of trajectories to plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        if not self.polygons:
            raise ValueError("No polygons to plot")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot polygons first
        for i, poly in enumerate(self.polygons):
            if hasattr(poly, 'exterior'):
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.2, facecolor=plt.cm.Set3(i % 12))
                ax.plot(x, y, 'k-', linewidth=1, alpha=0.6)
        
        # Plot subset of trajectories
        n_particles = trajectory_data.shape[0]
        indices = np.linspace(0, n_particles - 1, min(max_trajectories, n_particles), dtype=int)
        
        for idx in indices:
            traj = trajectory_data[idx]
            # Remove NaN values
            valid_mask = ~(np.isnan(traj[:, 0]) | np.isnan(traj[:, 1]))
            if np.any(valid_mask):
                valid_traj = traj[valid_mask]
                ax.plot(valid_traj[:, 0], valid_traj[:, 1], 
                       alpha=0.7, linewidth=0.8, color='blue')
                # Mark start point
                ax.scatter(valid_traj[0, 0], valid_traj[0, 1], 
                          c='red', s=20, zorder=5, alpha=0.8)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Particle Trajectories on Connectivity Polygons\n({len(indices)} trajectories shown)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
