"""NetworkX conversion and metrics for SedTRAILS connectivity matrices."""

from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd
import xarray as xr


def adjacency_to_digraph(adjacency: xr.Dataset | xr.DataArray | np.ndarray, *, timestep: int | None = None) -> nx.DiGraph:
    """Convert a SedTRAILS adjacency matrix to a weighted directed graph."""

    matrix, node_x, node_y, node_id = _matrix_and_nodes(adjacency, timestep)
    graph = nx.DiGraph()

    n_nodes = matrix.shape[0]
    for node in range(n_nodes):
        attrs = {}
        if node_x is not None and node_y is not None:
            attrs['x'] = float(node_x[node])
            attrs['y'] = float(node_y[node])
        if node_id is not None:
            attrs['label'] = str(node_id[node])
        graph.add_node(node, **attrs)

    for src in range(n_nodes):
        for sink in range(n_nodes):
            weight = float(matrix[src, sink])
            if weight != 0.0 and np.isfinite(weight):
                graph.add_edge(src, sink, weight=weight)

    return graph


def compute_network_metrics(graph: nx.DiGraph) -> dict[str, float | int]:
    """Compute common whole-network metrics for a directed weighted graph."""

    metrics: dict[str, float | int] = {
        'n_nodes': graph.number_of_nodes(),
        'n_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'reciprocity': nx.reciprocity(graph) or 0.0,
    }
    if graph.number_of_nodes() > 0:
        undirected = graph.to_undirected()
        metrics['average_clustering'] = nx.average_clustering(undirected, weight='weight')
        metrics['n_weak_components'] = nx.number_weakly_connected_components(graph)
    else:
        metrics['average_clustering'] = 0.0
        metrics['n_weak_components'] = 0
    return metrics


def compute_node_metrics(graph: nx.DiGraph) -> dict[str, dict[int, float]]:
    """Compute common node-level directed weighted network metrics."""

    return {
        'in_degree': dict(graph.in_degree()),
        'out_degree': dict(graph.out_degree()),
        'weighted_in_degree': dict(graph.in_degree(weight='weight')),
        'weighted_out_degree': dict(graph.out_degree(weight='weight')),
        'betweenness': nx.betweenness_centrality(graph, weight='weight', normalized=True),
        'pagerank': nx.pagerank(graph, weight='weight') if graph.number_of_nodes() else {},
    }


def node_metrics_to_dataframe(node_metrics: dict[str, dict[int, float]]) -> pd.DataFrame:
    """Convert node metric dictionaries to a node-indexed DataFrame."""

    return pd.DataFrame(node_metrics).sort_index().rename_axis('node')


def detect_communities(graph: nx.DiGraph) -> tuple[dict[int, int], float]:
    """Detect weighted graph communities and return node labels plus modularity."""

    if graph.number_of_nodes() == 0:
        return {}, 0.0

    undirected = graph.to_undirected()
    if undirected.number_of_edges() == 0:
        labels = {node: index for index, node in enumerate(undirected.nodes())}
        return labels, 0.0

    communities = list(nx.community.greedy_modularity_communities(undirected, weight='weight'))
    labels = {}
    for community_id, community in enumerate(communities):
        for node in community:
            labels[node] = community_id
    modularity = nx.community.modularity(undirected, communities, weight='weight')
    return labels, float(modularity)


def _matrix_and_nodes(
    adjacency: xr.Dataset | xr.DataArray | np.ndarray,
    timestep: int | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    node_x = node_y = node_id = None
    if isinstance(adjacency, xr.Dataset):
        data = adjacency['adjacency']
        node_x = np.asarray(adjacency['node_x'].values) if 'node_x' in adjacency else None
        node_y = np.asarray(adjacency['node_y'].values) if 'node_y' in adjacency else None
        node_id = np.asarray(adjacency['node_id'].values) if 'node_id' in adjacency else None
    elif isinstance(adjacency, xr.DataArray):
        data = adjacency
    else:
        matrix = np.asarray(adjacency, dtype=float)
        return _select_timestep(matrix, timestep), node_x, node_y, node_id

    matrix = np.asarray(data.values, dtype=float)
    return _select_timestep(matrix, timestep), node_x, node_y, node_id


def _select_timestep(matrix: np.ndarray, timestep: int | None) -> np.ndarray:
    if matrix.ndim == 2:
        return matrix
    if matrix.ndim != 3:
        raise ValueError('Adjacency matrix must be 2D or 3D.')
    if timestep is None:
        raise ValueError('timestep is required for time-varying adjacency matrices.')
    return matrix[int(timestep)]
