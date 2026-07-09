"""Compile connectivity adjacency matrices from SedTRAILS trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import xarray as xr
from shapely import Point

from sedtrails.pathway_visualizer.sedtrails_plotting import TrajectoryArrays

from .polygons import ConnectivityPolygons, source_positions_from_arrays


@dataclass(frozen=True)
class AdjacencyOptions:
    """Options used to compile a connectivity adjacency matrix."""

    mode: str = 'all'
    count_repeated_visits: bool = True
    weight: str = 'raw_counts'
    include_self_links: bool = True


def compile_adjacency_matrix(
    trajectories: TrajectoryArrays,
    connectivity: ConnectivityPolygons,
    *,
    mode: str = 'all',
    count_repeated_visits: bool = True,
    weight: str = 'raw_counts',
    include_self_links: bool = True,
    source_node: np.ndarray | None = None,
    representative_volume: np.ndarray | None = None,
) -> xr.Dataset:
    """Compile source-to-sink connectivity from particle trajectories.

    Parameters
    ----------
    trajectories
        Particle trajectory arrays in SedTRAILS' normalized particle-major
        layout.
    connectivity
        Connectivity polygons and default source-node assignments.
    mode
        ``"final"`` counts only each particle's final valid position.
        ``"all"`` counts all valid saved positions. ``"time"`` returns a
        time-varying matrix with dimensions ``time, source_node, sink_node``.
    count_repeated_visits
        If true, repeated visits by the same particle to a sink are counted
        each time. If false, each particle contributes at most once per sink
        for ``mode="all"``.
    weight
        ``"raw_counts"`` for particle-position counts, ``"probability"`` for
        row-normalized counts, or ``"representative_volume"`` for volume-
        weighted counts.
    include_self_links
        If false, source-to-same-sink diagonal entries are zeroed.
    source_node
        Optional source-node label per particle. Defaults to
        ``connectivity.source_node`` when lengths match, otherwise inferred
        from initial particle position.
    representative_volume
        Optional per-particle representative volume for volume-weighted counts.
    """

    options = AdjacencyOptions(
        mode=mode,
        count_repeated_visits=count_repeated_visits,
        weight=weight,
        include_self_links=include_self_links,
    )
    _validate_options(options)

    x = np.asarray(trajectories.x, dtype=float)
    y = np.asarray(trajectories.y, dtype=float)
    valid = trajectories.valid_mask()
    source_node = _source_node(trajectories, connectivity, source_node)
    sink_node = _sink_membership(x, y, connectivity)

    n_particles, n_timesteps = x.shape
    n_nodes = len(connectivity)
    counts = _empty_counts(mode, n_timesteps, n_nodes)
    particle_weight = _particle_weight(weight, representative_volume, n_particles)

    if mode == 'final':
        final_idx = _last_valid_index(valid)
        for particle in range(n_particles):
            if final_idx[particle] < 0:
                continue
            src = source_node[particle]
            sink = sink_node[particle, final_idx[particle]]
            if src >= 0 and sink >= 0:
                counts[src, sink] += particle_weight[particle]
    elif mode == 'all':
        for particle in range(n_particles):
            src = source_node[particle]
            if src < 0:
                continue
            sinks = sink_node[particle, valid[particle]]
            sinks = sinks[sinks >= 0]
            if not count_repeated_visits:
                sinks = np.unique(sinks)
            for sink in sinks:
                counts[src, sink] += particle_weight[particle]
    else:
        for timestep in range(n_timesteps):
            for particle in np.flatnonzero(valid[:, timestep]):
                src = source_node[particle]
                sink = sink_node[particle, timestep]
                if src >= 0 and sink >= 0:
                    counts[timestep, src, sink] += particle_weight[particle]

    if not include_self_links:
        _zero_self_links(counts)

    matrix = _normalize_if_requested(counts, weight)
    return _to_dataset(matrix, trajectories, connectivity, options)


def _validate_options(options: AdjacencyOptions) -> None:
    if options.mode not in {'final', 'all', 'time'}:
        raise ValueError("mode must be one of {'final', 'all', 'time'}.")
    if options.weight not in {'raw_counts', 'probability', 'representative_volume'}:
        raise ValueError("weight must be one of {'raw_counts', 'probability', 'representative_volume'}.")


def _source_node(
    trajectories: TrajectoryArrays,
    connectivity: ConnectivityPolygons,
    source_node: np.ndarray | None,
) -> np.ndarray:
    n_particles = trajectories.x.shape[0]
    if source_node is None and len(connectivity.source_node) == n_particles:
        return np.asarray(connectivity.source_node, dtype=int)
    if source_node is not None:
        source_node = np.asarray(source_node, dtype=int)
        if source_node.shape != (n_particles,):
            raise ValueError('source_node must contain one label per particle.')
        return source_node

    source_xy = source_positions_from_arrays(trajectories.x, trajectories.y, trajectories.valid_mask())
    labels = np.full(n_particles, -1, dtype=int)
    for particle, xy in enumerate(source_xy):
        if np.isfinite(xy).all():
            labels[particle] = _point_membership(float(xy[0]), float(xy[1]), connectivity)
    return labels


def _sink_membership(x: np.ndarray, y: np.ndarray, connectivity: ConnectivityPolygons) -> np.ndarray:
    membership = np.full(x.shape, -1, dtype=int)
    for index in np.ndindex(x.shape):
        if np.isfinite(x[index]) and np.isfinite(y[index]):
            membership[index] = _point_membership(float(x[index]), float(y[index]), connectivity)
    return membership


def _point_membership(x: float, y: float, connectivity: ConnectivityPolygons) -> int:
    point = Point(x, y)
    for node, polygon in enumerate(connectivity.polygons):
        if polygon.covers(point):
            return node
    return -1


def _empty_counts(mode: str, n_timesteps: int, n_nodes: int) -> np.ndarray:
    if mode == 'time':
        return np.zeros((n_timesteps, n_nodes, n_nodes), dtype=float)
    return np.zeros((n_nodes, n_nodes), dtype=float)


def _particle_weight(weight: str, representative_volume: np.ndarray | None, n_particles: int) -> np.ndarray:
    if weight != 'representative_volume':
        return np.ones(n_particles, dtype=float)
    if representative_volume is None:
        raise ValueError("representative_volume is required when weight='representative_volume'.")
    representative_volume = np.asarray(representative_volume, dtype=float)
    if representative_volume.shape != (n_particles,):
        raise ValueError('representative_volume must contain one value per particle.')
    return representative_volume


def _last_valid_index(valid: np.ndarray) -> np.ndarray:
    last = np.where(valid, np.arange(valid.shape[1]), -1).max(axis=1)
    return last.astype(int)


def _zero_self_links(counts: np.ndarray) -> None:
    if counts.ndim == 2:
        np.fill_diagonal(counts, 0.0)
    else:
        for timestep in range(counts.shape[0]):
            np.fill_diagonal(counts[timestep], 0.0)


def _normalize_if_requested(counts: np.ndarray, weight: str) -> np.ndarray:
    if weight != 'probability':
        return counts
    totals = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)


def _to_dataset(
    matrix: np.ndarray,
    trajectories: TrajectoryArrays,
    connectivity: ConnectivityPolygons,
    options: AdjacencyOptions,
) -> xr.Dataset:
    node_coords = {
        'source_node': np.arange(len(connectivity), dtype=int),
        'sink_node': np.arange(len(connectivity), dtype=int),
    }

    if matrix.ndim == 3:
        time_values = np.asarray(trajectories.time[0], dtype=float)
        dims = ('time', 'source_node', 'sink_node')
        coords = {'time': time_values, **node_coords}
    else:
        dims = ('source_node', 'sink_node')
        coords = node_coords

    ds = xr.Dataset(
        data_vars={
            'adjacency': (dims, matrix),
            'node_x': (('source_node',), connectivity.node_x),
            'node_y': (('source_node',), connectivity.node_y),
            'node_id': (('source_node',), np.asarray(connectivity.node_id, dtype=str)),
        },
        coords=coords,
        attrs={
            'title': 'SedTRAILS Connectivity Adjacency Matrix',
            'sedtrails_file_kind': 'connectivity_adjacency',
            'sedtrails_connectivity_schema': 'adjacency_v1',
            **{f'compile_{key}': _netcdf_safe_attr(value) for key, value in asdict(options).items()},
        },
    )
    ds['adjacency'].attrs.update({
        'long_name': 'Source-to-sink particle connectivity',
        'weight': options.weight,
    })
    return ds


def _netcdf_safe_attr(value):
    if isinstance(value, bool):
        return int(value)
    return value
