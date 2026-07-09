"""Plotting helpers for SedTRAILS connectivity networks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import xarray as xr


def plot_adjacency_matrix(
    adjacency: xr.Dataset | xr.DataArray | np.ndarray,
    *,
    log_scale: bool = False,
    ax: plt.Axes | None = None,
    cmap: str = 'viridis',
) -> plt.Axes:
    """Plot a 2D connectivity adjacency matrix."""

    matrix = _matrix(adjacency)
    if matrix.ndim != 2:
        raise ValueError('plot_adjacency_matrix expects a 2D adjacency matrix.')
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    values = np.log10(matrix, where=matrix > 0, out=np.full_like(matrix, np.nan)) if log_scale else matrix
    image = ax.imshow(values, origin='lower', cmap=cmap)
    label = 'Connectivity [log10(count)]' if log_scale else 'Connectivity [count]'
    plt.colorbar(image, ax=ax, label=label)
    ax.set_xlabel('Sink node')
    ax.set_ylabel('Source node')
    ax.set_title('Connectivity adjacency matrix')
    return ax


def plot_network_geographic(
    graph: nx.DiGraph,
    *,
    ax: plt.Axes | None = None,
    width_scale: float = 2.0,
    node_size: float = 120.0,
) -> plt.Axes:
    """Plot a NetworkX graph using node ``x``/``y`` attributes."""

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True) if 'x' in data and 'y' in data}
    if len(pos) != graph.number_of_nodes():
        raise ValueError("All graph nodes must have 'x' and 'y' attributes for geographic plotting.")
    _draw_graph(graph, pos, ax=ax, width_scale=width_scale, node_size=node_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Connectivity network')
    return ax


def plot_network_layout(
    graph: nx.DiGraph,
    *,
    layout: str = 'spring',
    ax: plt.Axes | None = None,
    width_scale: float = 2.0,
    node_size: float = 120.0,
) -> plt.Axes:
    """Plot a graph with a standard NetworkX layout."""

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    if layout == 'spring':
        pos = nx.spring_layout(graph, weight='weight', seed=0)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(graph, weight='weight')
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    else:
        raise ValueError("layout must be one of {'spring', 'kamada_kawai', 'circular'}.")
    _draw_graph(graph, pos, ax=ax, width_scale=width_scale, node_size=node_size)
    ax.set_axis_off()
    ax.set_title(f'Connectivity network ({layout})')
    return ax


def _draw_graph(
    graph: nx.DiGraph,
    pos: dict,
    *,
    ax: plt.Axes,
    width_scale: float,
    node_size: float,
) -> None:
    weights = np.asarray([data.get('weight', 1.0) for _, _, data in graph.edges(data=True)], dtype=float)
    if weights.size:
        widths = width_scale * weights / np.nanmax(weights)
    else:
        widths = 1.0
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color='white', edgecolors='black', ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos,
        width=widths,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=12,
        edge_color='0.25',
        alpha=0.75,
        ax=ax,
    )


def _matrix(adjacency: xr.Dataset | xr.DataArray | np.ndarray) -> np.ndarray:
    if isinstance(adjacency, xr.Dataset):
        return np.asarray(adjacency['adjacency'].values, dtype=float)
    if isinstance(adjacency, xr.DataArray):
        return np.asarray(adjacency.values, dtype=float)
    return np.asarray(adjacency, dtype=float)
