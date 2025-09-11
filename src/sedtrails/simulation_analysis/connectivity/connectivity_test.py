"""
Connectivity Test Module for SedTrails

This module provides comprehensive testing and demonstration of connectivity
analysis functionality, designed for interactive use in Jupyter notebooks.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from .adjacency_matrix import AdjacencyMatrixCompiler
from .connectivity_polygons import ConnectivityPolygonGenerator
from .network_analysis import NetworkAnalyzer
from .connectivity_io import ConnectivityIO

class ConnectivityTester:
    """
    Comprehensive tester for connectivity analysis functionality.
    
    Provides methods for testing all aspects of the connectivity analysis
    pipeline with SedTrails data.
    """
    
    def __init__(self, trajectory_file: str, output_dir: str = "connectivity_output"):
        """
        Initialize connectivity tester.
        
        Parameters
        ----------
        trajectory_file : str
            Path to SedTrails trajectory NetCDF file
        output_dir : str
            Output directory for results
        """
        self.trajectory_file = Path(trajectory_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.compiler = None
        self.polygon_generator = None
        self.network_analyzer = None
        self.results = {}
        
    def test_polygon_generation(
        self,
        method: str = 'voronoi',
        n_clusters: Optional[int] = None,
        buffer_distance: float = 1000.0,
        show_plots: bool = True
    ) -> Dict:
        """
        Test connectivity polygon generation.
        
        Parameters
        ----------
        method : str
            Polygon generation method ('voronoi' or 'clustered')
        n_clusters : int, optional
            Number of clusters for clustered method
        buffer_distance : float
            Buffer distance for Voronoi polygons
        show_plots : bool
            Whether to show plots
            
        Returns
        -------
        Dict
            Test results and generated polygons
        """
        print(f"Testing polygon generation with method: {method}")
        
        # Load trajectory data to get source points
        self.compiler = AdjacencyMatrixCompiler(str(self.trajectory_file))
        ds = self.compiler.load_trajectory_data()
        
        # Extract initial positions as source points
        x_data = ds['x'].values
        y_data = ds['y'].values
        
        # Get first valid position for each particle
        source_points = []
        for i in range(x_data.shape[0]):
            valid_mask = ~(np.isnan(x_data[i, :]) | np.isnan(y_data[i, :]))
            if np.any(valid_mask):
                first_valid = np.where(valid_mask)[0][0]
                source_points.append([x_data[i, first_valid], y_data[i, first_valid]])
        
        source_points = np.array(source_points)
        
        # Initialize polygon generator
        self.polygon_generator = ConnectivityPolygonGenerator(source_points)
        
        # Generate polygons
        if method == 'voronoi':
            polygons = self.polygon_generator.generate_voronoi_polygons(
                buffer_distance=buffer_distance
            )
            test_name = "voronoi_polygons"
        elif method == 'clustered':
            if n_clusters is None:
                n_clusters = max(2, len(source_points) // 3)  # Default clustering
            polygons = self.polygon_generator.generate_clustered_polygons(
                n_clusters=n_clusters
            )
            test_name = f"clustered_polygons_n{n_clusters}"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Save polygons
        polygon_file = self.output_dir / f"{test_name}.npy"
        self.polygon_generator.save_polygons(str(polygon_file))
        
        # Save as NetCDF
        netcdf_file = self.output_dir / f"{test_name}.nc"
        ConnectivityIO.save_polygons_netcdf(
            polygons,
            getattr(self.polygon_generator, 'polygon_assignments', None),
            self.polygon_generator.source_points,  # Use filtered source points
            str(netcdf_file),
            metadata={'method': method, 'buffer_distance': buffer_distance}
        )
        
        # Plot results
        if show_plots:
            self.polygon_generator.plot_polygons(
                save_path=str(self.output_dir / f"{test_name}_plot.png")
            )
            plt.show()
            
            # If this is clustered polygons, also plot trajectories overlay
            if method == 'clustered':
                try:
                    # Load trajectory data for overlay
                    compiler = AdjacencyMatrixCompiler(str(self.trajectory_file))
                    ds = compiler.load_trajectory_data()
                    x_data = ds['x'].values
                    y_data = ds['y'].values
                    
                    # Create trajectory array (n_particles, n_timesteps, 2)
                    trajectory_data = np.stack([x_data, y_data], axis=2)
                    
                    # Plot trajectories on polygons
                    self.polygon_generator.plot_trajectories_on_polygons(
                        trajectory_data,
                        max_trajectories=30,
                        save_path=str(self.output_dir / f"{test_name}_trajectories.png")
                    )
                    plt.show()
                    
                    ds.close()
                except Exception as e:
                    print(f"    Warning: Could not plot trajectory overlay: {e}")
        
        results = {
            'method': method,
            'n_polygons': len(polygons),
            'n_source_points': len(source_points),
            'polygon_file': str(polygon_file),
            'netcdf_file': str(netcdf_file),
            'polygons': polygons
        }
        
        self.results[f'polygon_test_{method}'] = results
        print(f"✓ Generated {len(polygons)} polygons using {method} method")
        print(f"✓ Saved to: {polygon_file}")
        print(f"✓ Saved to NetCDF: {netcdf_file}")
        
        return results
    
    def test_adjacency_compilation(
        self,
        polygon_file: str,
        modes: Optional[List[str]] = None,
        min_threshold: int = 1,
        show_plots: bool = True
    ) -> Dict:
        """
        Test adjacency matrix compilation.
        
        Parameters
        ----------
        polygon_file : str
            Path to polygon file
        modes : List[str]
            List of compilation modes to test
        min_threshold : int
            Minimum particle threshold
        show_plots : bool
            Whether to show plots
            
        Returns
        -------
        Dict
            Test results for all modes
        """
        print(f"Testing adjacency matrix compilation with modes: {modes}")
        
        if modes is None:
            modes = ['final', 'all_time']
        
        if self.compiler is None:
            self.compiler = AdjacencyMatrixCompiler(str(self.trajectory_file))
        
        results = {}
        
        for mode in modes:
            print(f"\n  Testing mode: {mode}")
            
            # Compile adjacency matrix
            # Note: type checker doesn't recognize string literals at runtime
            adj_data = self.compiler.compile_adjacency_matrix(
                polygon_file=polygon_file,
                mode=mode,  # type: ignore
                min_particles_threshold=min_threshold
            )
            
            # Save to NetCDF
            output_file = self.output_dir / f"adjacency_matrix_{mode}.nc"
            ConnectivityIO.save_adjacency_netcdf(adj_data, str(output_file))
            
            # Test loading
            loaded_data = ConnectivityIO.load_adjacency_netcdf(str(output_file))
            
            # Verify data integrity
            if mode == 'time_varying':
                matrix_shape = adj_data['adjacency_matrices'].shape
                loaded_shape = loaded_data['adjacency_matrices'].shape
            else:
                matrix_shape = adj_data['adjacency_matrix'].shape
                loaded_shape = loaded_data['adjacency_matrix'].shape
            
            integrity_check = matrix_shape == loaded_shape
            
            results[mode] = {
                'adjacency_data': adj_data,
                'output_file': str(output_file),
                'matrix_shape': matrix_shape,
                'integrity_check': integrity_check,
                'n_polygons': adj_data['n_polygons']
            }
            
            print(f"    ✓ Matrix shape: {matrix_shape}")
            print(f"    ✓ Saved to: {output_file}")
            print(f"    ✓ Load/save integrity: {integrity_check}")
        
        self.results['adjacency_compilation'] = results
        
        # Plot adjacency matrices
        if show_plots:
            self._plot_adjacency_comparison(results)
        
        return results
    
    def test_network_analysis(
        self,
        adjacency_data: Dict,
        test_name: str = "network_test",
        show_plots: bool = True
    ) -> Dict:
        """
        Test network analysis functionality.
        
        Parameters
        ----------
        adjacency_data : Dict
            Adjacency matrix data
        test_name : str
            Name for this test
        show_plots : bool
            Whether to show plots
            
        Returns
        -------
        Dict
            Network analysis results
        """
        print(f"Testing network analysis: {test_name}")
        
        # Initialize network analyzer
        self.network_analyzer = NetworkAnalyzer(adjacency_data)
        
        # Create network
        G = self.network_analyzer.create_network(threshold=1, normalize=False)
        
        # Compute metrics
        metrics = self.network_analyzer.compute_network_metrics()
        
        # Export metrics
        metrics_file = self.output_dir / f"{test_name}_metrics.json"
        ConnectivityIO.export_network_metrics(metrics, str(metrics_file))
        
        # Get metrics summary
        try:
            summary_df = self.network_analyzer.get_metrics_summary()
            summary_file = self.output_dir / f"{test_name}_summary.csv"
            summary_df.to_csv(summary_file, index=False)
        except ImportError:
            print("    Warning: pandas not available, skipping CSV export")
            summary_df = None
            summary_file = None
        
        results = {
            'network': G,
            'metrics': metrics,
            'metrics_file': str(metrics_file),
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': metrics.get('density', 0)
        }
        
        if summary_df is not None and summary_file is not None:
            results['summary_file'] = str(summary_file)
        
        # Plot network
        if show_plots:
            # Plot network graph
            self.network_analyzer.plot_network(
                save_path=str(self.output_dir / f"{test_name}_network.png")
            )
            plt.show()
            
            # Plot adjacency matrix
            self.network_analyzer.plot_adjacency_matrix(
                save_path=str(self.output_dir / f"{test_name}_adjacency.png")
            )
            plt.show()
            
            # Plot geographic network if polygon positions are available
            try:
                # Get polygon centers as source points
                polygon_data = adjacency_data.get('polygon_info', {})
                if 'centroids' in polygon_data:
                    source_points = polygon_data['centroids']
                    self.network_analyzer.plot_geographic_network(
                        source_points=source_points,
                        save_path=str(self.output_dir / f"{test_name}_geographic.png")
                    )
                    plt.show()
                    print("    ✓ Geographic network plot saved")
                else:
                    print("    Warning: No polygon centroid data available for geographic plot")
            except Exception as e:
                print(f"    Warning: Could not create geographic network plot: {e}")
        
        self.results[f'network_analysis_{test_name}'] = results
        
        print(f"    ✓ Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"    ✓ Density: {metrics.get('density', 0):.3f}")
        print(f"    ✓ Metrics saved to: {metrics_file}")
        
        return results
    
    def run_full_test_suite(
        self,
        polygon_methods: Optional[List[str]] = None,
        adjacency_modes: Optional[List[str]] = None,
        n_clusters: int = 5,
        show_plots: bool = True
    ) -> Dict:
        """
        Run comprehensive test suite.
        
        Parameters
        ----------
        polygon_methods : List[str]
            Polygon generation methods to test
        adjacency_modes : List[str]
            Adjacency compilation modes to test
        n_clusters : int
            Number of clusters for clustered polygons
        show_plots : bool
            Whether to show plots
            
        Returns
        -------
        Dict
            Complete test results
        """
        print("="*60)
        print("SEDTRAILS CONNECTIVITY ANALYSIS - FULL TEST SUITE")
        print("="*60)
        
        if polygon_methods is None:
            polygon_methods = ['voronoi', 'clustered']
        if adjacency_modes is None:
            adjacency_modes = ['final', 'all_time']
        
        all_results = {}
        
        # Test polygon generation
        for method in polygon_methods:
            print(f"\n{'-'*40}")
            print(f"TESTING POLYGON GENERATION: {method.upper()}")
            print(f"{'-'*40}")
            
            poly_results = self.test_polygon_generation(
                method=method,
                n_clusters=n_clusters if method == 'clustered' else None,
                show_plots=show_plots
            )
            all_results[f'polygons_{method}'] = poly_results
            
            # Test adjacency compilation with these polygons
            print(f"\n{'-'*40}")
            print(f"TESTING ADJACENCY COMPILATION WITH {method.upper()} POLYGONS")
            print(f"{'-'*40}")
            
            adj_results = self.test_adjacency_compilation(
                polygon_file=poly_results['polygon_file'],
                modes=adjacency_modes,
                show_plots=show_plots
            )
            all_results[f'adjacency_{method}'] = adj_results
            
            # Test network analysis
            for mode in adjacency_modes:
                print(f"\n{'-'*40}")
                print(f"TESTING NETWORK ANALYSIS: {method.upper()} + {mode.upper()}")
                print(f"{'-'*40}")
                
                network_results = self.test_network_analysis(
                    adjacency_data=adj_results[mode]['adjacency_data'],
                    test_name=f"{method}_{mode}",
                    show_plots=show_plots
                )
                all_results[f'network_{method}_{mode}'] = network_results
        
        # Save complete results summary
        summary_file = self.output_dir / "test_suite_summary.txt"
        self._save_test_summary(all_results, summary_file)
        
        print(f"\n{'='*60}")
        print("TEST SUITE COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {self.output_dir}")
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*60}")
        
        return all_results
    
    def _plot_adjacency_comparison(self, results: Dict) -> None:
        """Plot comparison of adjacency matrices from different modes."""
        n_modes = len(results)
        if n_modes == 0:
            return
            
        fig, axes = plt.subplots(1, n_modes, figsize=(5*n_modes, 4))
        if n_modes == 1:
            axes = [axes]
        
        for i, (mode, data) in enumerate(results.items()):
            adj_data = data['adjacency_data']
            
            if mode == 'time_varying':
                # Show first timestep
                matrix = adj_data['adjacency_matrices'][0]
                title_suffix = " (t=0)"
            else:
                matrix = adj_data['adjacency_matrix']
                title_suffix = ""
            
            im = axes[i].imshow(matrix, cmap='viridis', aspect='auto')
            axes[i].set_title(f'{mode.replace("_", " ").title()}{title_suffix}')
            axes[i].set_xlabel('Destination')
            axes[i].set_ylabel('Source')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "adjacency_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_test_summary(self, results: Dict, filename: Path) -> None:
        """Save comprehensive test summary to file."""
        with open(filename, 'w') as f:
            f.write("SEDTRAILS CONNECTIVITY ANALYSIS - TEST SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Trajectory file: {self.trajectory_file}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            for test_name, test_results in results.items():
                f.write(f"\n{test_name.upper()}\n")
                f.write("-" * len(test_name) + "\n")
                
                if isinstance(test_results, dict):
                    for key, value in test_results.items():
                        if isinstance(value, (str, int, float, bool)):
                            f.write(f"  {key}: {value}\n")
                        elif hasattr(value, '__len__') and not isinstance(value, str):
                            f.write(f"  {key}: length={len(value)}\n")
                        else:
                            f.write(f"  {key}: {type(value).__name__}\n")
                
            f.write(f"\n\nTest completed: {__import__('datetime').datetime.now()}\n")
