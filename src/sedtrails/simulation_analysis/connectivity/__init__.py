"""Connectivity analysis utilities for SedTRAILS trajectory outputs."""

from .adjacency import compile_adjacency_matrix
from .io import read_adjacency_netcdf, write_adjacency_netcdf
from .network import (
    adjacency_to_digraph,
    compute_network_metrics,
    compute_node_metrics,
    detect_communities,
    node_metrics_to_dataframe,
)
from .plotting import (
    plot_adjacency_matrix,
    plot_community_map,
    plot_network_geographic,
    plot_network_layout,
    plot_node_metric_bars,
    plot_node_metric_map,
)
from .polygons import ConnectivityPolygons, generate_connectivity_polygons, group_sources_by_position
from .workflow import ConnectivityAdjacencySummary, compile_adjacency_from_results

__all__ = [
    'ConnectivityPolygons',
    'ConnectivityAdjacencySummary',
    'adjacency_to_digraph',
    'compile_adjacency_from_results',
    'compile_adjacency_matrix',
    'compute_network_metrics',
    'compute_node_metrics',
    'detect_communities',
    'generate_connectivity_polygons',
    'group_sources_by_position',
    'node_metrics_to_dataframe',
    'plot_adjacency_matrix',
    'plot_community_map',
    'plot_network_geographic',
    'plot_network_layout',
    'plot_node_metric_bars',
    'plot_node_metric_map',
    'read_adjacency_netcdf',
    'write_adjacency_netcdf',
]
