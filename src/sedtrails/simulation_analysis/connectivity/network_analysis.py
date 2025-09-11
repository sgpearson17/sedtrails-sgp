"""
Network Analysis Module for SedTrails Connectivity

This module provides NetworkX-based network analysis tools for connectivity matrices,
replacing MATLAB Brain Connectivity Toolbox functionality.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd

class NetworkAnalyzer:
    """
    Analyzes connectivity networks using NetworkX.
    
    Provides network metrics, visualization, and analysis tools for
    particle connectivity data.
    """
    
    def __init__(self, adjacency_data: Dict):
        """
        Initialize network analyzer.
        
        Parameters
        ----------
        adjacency_data : Dict
            Dictionary containing adjacency matrix and metadata
        """
        self.adjacency_data = adjacency_data
        self.networks = {}
        self.metrics = {}
        
    def create_network(
        self, 
        threshold: Optional[float] = None,
        normalize: bool = False,
        time_index: Optional[int] = None
    ) -> nx.DiGraph:
        """
        Create NetworkX graph from adjacency matrix.
        
        Parameters
        ----------
        threshold : float, optional
            Minimum edge weight threshold
        normalize : bool
            Whether to normalize edge weights
        time_index : int, optional
            Time index for time-varying networks
            
        Returns
        -------
        nx.DiGraph
            NetworkX directed graph
        """
        # Get adjacency matrix
        if self.adjacency_data['mode'] == 'time_varying':
            if time_index is None:
                raise ValueError("time_index required for time-varying networks")
            adj_matrix = self.adjacency_data['adjacency_matrices'][time_index]
        else:
            adj_matrix = self.adjacency_data['adjacency_matrix']
            
        # Apply threshold
        if threshold is not None:
            adj_matrix = adj_matrix.copy()
            adj_matrix[adj_matrix < threshold] = 0
            
        # Normalize if requested
        if normalize:
            row_sums = adj_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            adj_matrix = adj_matrix / row_sums
            
        # Create NetworkX graph
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        
        # Add node attributes
        polygon_names = self.adjacency_data.get('polygon_names', [])
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['name'] = polygon_names[i] if i < len(polygon_names) else f'Node_{i}'
            G.nodes[node]['polygon_id'] = i
            
        # Store network
        key = f"threshold_{threshold}_norm_{normalize}"
        if time_index is not None:
            key += f"_time_{time_index}"
        self.networks[key] = G
        
        return G
    
    def compute_network_metrics(
        self, 
        network_key: Optional[str] = None,
        include_centrality: bool = True,
        include_clustering: bool = True,
        include_path_metrics: bool = True
    ) -> Dict:
        """
        Compute comprehensive network metrics.
        
        Parameters
        ----------
        network_key : str, optional
            Key of network to analyze (uses first if None)
        include_centrality : bool
            Whether to compute centrality metrics
        include_clustering : bool  
            Whether to compute clustering metrics
        include_path_metrics : bool
            Whether to compute path-based metrics
            
        Returns
        -------
        Dict
            Dictionary of network metrics
        """
        if not self.networks:
            G = self.create_network()
            network_key = list(self.networks.keys())[0]
        else:
            if network_key is None:
                network_key = list(self.networks.keys())[0]
            G = self.networks[network_key]
            
        metrics = {}
        
        # Basic metrics
        metrics['n_nodes'] = G.number_of_nodes()
        metrics['n_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        metrics['is_connected'] = nx.is_strongly_connected(G)
        
        # Convert to undirected for some metrics
        G_undirected = G.to_undirected()
        metrics['is_connected_undirected'] = nx.is_connected(G_undirected)
        
        # Centrality metrics
        if include_centrality:
            metrics['in_degree_centrality'] = nx.in_degree_centrality(G)
            metrics['out_degree_centrality'] = nx.out_degree_centrality(G)
            metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
            metrics['closeness_centrality'] = nx.closeness_centrality(G)
            metrics['eigenvector_centrality'] = self._safe_eigenvector_centrality(G)
            metrics['pagerank'] = nx.pagerank(G)
            
        # Clustering metrics  
        if include_clustering:
            metrics['clustering_coefficient'] = nx.average_clustering(G_undirected)
            metrics['transitivity'] = nx.transitivity(G_undirected)
            
        # Path metrics
        if include_path_metrics and nx.is_strongly_connected(G):
            metrics['average_shortest_path'] = nx.average_shortest_path_length(G)
            metrics['diameter'] = nx.diameter(G)
            
        # Community detection
        try:
            communities = nx.community.greedy_modularity_communities(G_undirected)
            metrics['n_communities'] = len(communities)
            metrics['modularity'] = nx.community.modularity(G_undirected, communities)
        except Exception:
            metrics['n_communities'] = None
            metrics['modularity'] = None
            
        self.metrics[network_key] = metrics
        return metrics
    
    def _safe_eigenvector_centrality(self, G: nx.DiGraph) -> Dict:
        """Compute eigenvector centrality with error handling."""
        try:
            return nx.eigenvector_centrality(G, max_iter=1000)
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            # Fallback to in-degree centrality
            return nx.in_degree_centrality(G)
    
    def plot_network(
        self,
        network_key: Optional[str] = None,
        node_positions: Optional[Dict] = None,
        node_size_metric: Optional[str] = 'degree',
        edge_width_metric: Optional[str] = 'weight',
        layout: str = 'spring',
        figsize: Tuple[float, float] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot network graph.
        
        Parameters
        ----------
        network_key : str, optional
            Network to plot
        node_positions : Dict, optional
            Custom node positions {node_id: (x, y)}
        node_size_metric : str, optional
            Metric for node sizing
        edge_width_metric : str, optional
            Metric for edge width
        layout : str
            Layout algorithm ('spring', 'circular', 'kamada_kawai')
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save plot
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        if network_key is None:
            if not self.networks:
                self.create_network()
            network_key = list(self.networks.keys())[0]
            
        G = self.networks[network_key]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get node positions
        if node_positions is not None:
            pos = node_positions
        else:
            if layout == 'spring':
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.spring_layout(G)
        
        # Node sizes
        if node_size_metric and node_size_metric in ['degree', 'in_degree', 'out_degree']:
            if node_size_metric == 'degree':
                node_sizes = [G.degree(node) * 100 + 200 for node in G.nodes()]
            elif node_size_metric == 'in_degree':
                node_sizes = [G.in_degree(node) * 100 + 200 for node in G.nodes()]
            elif node_size_metric == 'out_degree':
                node_sizes = [G.out_degree(node) * 100 + 200 for node in G.nodes()]
        else:
            node_sizes = 300
            
        # Edge widths
        if edge_width_metric == 'weight':
            edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
            if edge_weights:
                max_weight = max(edge_weights)
                edge_widths = [w / max_weight * 5 + 0.5 for w in edge_weights]
            else:
                edge_widths = 1
        else:
            edge_widths = 1
            
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, 
                              alpha=0.5, edge_color='gray', 
                              arrows=True, arrowsize=20, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_title(f'Connectivity Network ({network_key})')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_geographic_network(
        self,
        source_points: np.ndarray,
        network_key: Optional[str] = None,
        node_size_metric: Optional[str] = 'degree',
        edge_width_metric: Optional[str] = 'weight',
        figsize: Tuple[float, float] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot network with nodes at their geographic source positions.
        
        Parameters
        ----------
        source_points : np.ndarray
            Source point coordinates, shape (n_points, 2)
        network_key : str, optional
            Network to plot
        node_size_metric : str, optional
            Metric for node sizing
        edge_width_metric : str, optional
            Metric for edge width
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save plot
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        if network_key is None:
            if not self.networks:
                self.create_network()
            network_key = list(self.networks.keys())[0]
            
        G = self.networks[network_key]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create geographic positions dict
        pos = {}
        for i, node in enumerate(G.nodes()):
            if i < len(source_points):
                pos[node] = (source_points[i, 0], source_points[i, 1])
            else:
                # Fallback for extra nodes
                pos[node] = (0, 0)
        
        # Node sizes based on degree
        if node_size_metric and node_size_metric in ['degree', 'in_degree', 'out_degree']:
            if node_size_metric == 'degree':
                node_sizes = [G.degree(node) * 100 + 200 for node in G.nodes()]
            elif node_size_metric == 'in_degree':
                node_sizes = [G.in_degree(node) * 100 + 200 for node in G.nodes()]
            elif node_size_metric == 'out_degree':
                node_sizes = [G.out_degree(node) * 100 + 200 for node in G.nodes()]
        else:
            node_sizes = 300
            
        # Edge widths based on connectivity strength
        if edge_width_metric == 'weight':
            edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
            if edge_weights:
                max_weight = max(edge_weights)
                edge_widths = [w / max_weight * 5 + 0.5 for w in edge_weights]
            else:
                edge_widths = 1
        else:
            edge_widths = 1
            
        # Draw network at geographic positions
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, 
                              alpha=0.5, edge_color='gray', 
                              arrows=True, arrowsize=20, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Geographic Connectivity Network ({network_key})')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_adjacency_matrix(
        self,
        time_index: Optional[int] = None,
        normalize_rows: bool = False,
        figsize: Tuple[float, float] = (10, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot adjacency matrix as heatmap.
        
        Parameters
        ----------
        time_index : int, optional
            Time index for time-varying matrices
        normalize_rows : bool
            Whether to normalize rows to [0,1]
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save plot
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        # Get adjacency matrix
        if self.adjacency_data['mode'] == 'time_varying':
            if time_index is None:
                time_index = 0
            adj_matrix = self.adjacency_data['adjacency_matrices'][time_index]
            title_suffix = f" (t={time_index})"
        else:
            adj_matrix = self.adjacency_data['adjacency_matrix']
            title_suffix = ""
            
        if normalize_rows:
            row_sums = adj_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            adj_matrix = adj_matrix / row_sums
            
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(adj_matrix, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Connection Strength')
        
        # Labels
        polygon_names = self.adjacency_data.get('polygon_names', [])
        if polygon_names:
            ax.set_xticks(range(len(polygon_names)))
            ax.set_yticks(range(len(polygon_names)))
            ax.set_xticklabels(polygon_names, rotation=45, ha='right')
            ax.set_yticklabels(polygon_names)
        
        ax.set_xlabel('Destination Polygon')
        ax.set_ylabel('Source Polygon')
        ax.set_title(f'Adjacency Matrix{title_suffix}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def get_metrics_summary(self, network_key: Optional[str] = None) -> pd.DataFrame:
        """
        Get summary of network metrics as DataFrame.
        
        Parameters
        ----------
        network_key : str, optional
            Network key to summarize
            
        Returns
        -------
        pd.DataFrame
            Summary of network metrics
        """
        if network_key is None:
            if not self.metrics:
                self.compute_network_metrics()
            network_key = list(self.metrics.keys())[0]
            
        metrics = self.metrics[network_key]
        
        # Extract scalar metrics
        scalar_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, bool)) and value is not None:
                scalar_metrics[key] = value
                
        return pd.DataFrame([scalar_metrics])
