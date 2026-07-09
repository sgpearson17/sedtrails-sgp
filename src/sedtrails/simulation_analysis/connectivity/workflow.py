"""High-level workflows for SedTRAILS connectivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sedtrails.pathway_visualizer.sedtrails_plotting import load_from_xarray
from sedtrails.pathway_visualizer.trajectories import read_netcdf

from .adjacency import compile_adjacency_matrix
from .io import write_adjacency_netcdf
from .polygons import generate_connectivity_polygons, group_sources_by_position, source_positions_from_arrays


@dataclass(frozen=True)
class ConnectivityAdjacencySummary:
    """Summary of a compiled connectivity adjacency NetCDF file."""

    output_file: Path
    n_particles: int
    n_nodes: int
    mode: str
    weight: str
    count_repeated_visits: bool
    include_self_links: bool


def compile_adjacency_from_results(
    results_file: str | Path,
    output_file: str | Path,
    *,
    mode: str = 'all',
    polygon_mode: str = 'per_source',
    n_cells: int | None = None,
    group_sources_by_initial_position: bool = True,
    source_group_tolerance: float = 0.0,
    count_repeated_visits: bool = True,
    weight: str = 'raw_counts',
    include_self_links: bool = True,
) -> ConnectivityAdjacencySummary:
    """Read trajectory NetCDF output and write a connectivity adjacency NetCDF file."""

    results_file = Path(results_file)
    output_file = Path(output_file)
    ds = read_netcdf(results_file)
    try:
        trajectories = load_from_xarray(ds)
    finally:
        ds.close()

    source_xy = source_positions_from_arrays(trajectories.x, trajectories.y, trajectories.valid_mask())
    finite_sources = np.isfinite(source_xy).all(axis=1)
    if not finite_sources.all():
        raise ValueError('Connectivity analysis requires finite initial positions for all particles.')

    source_group = None
    if group_sources_by_initial_position and polygon_mode in {'per_source', 'grouped'}:
        source_group = group_sources_by_position(source_xy, tolerance=source_group_tolerance)

    connectivity = generate_connectivity_polygons(
        source_xy,
        mode=polygon_mode,
        n_cells=n_cells,
        source_group=source_group,
    )
    source_node = connectivity.source_node if len(connectivity.source_node) == trajectories.x.shape[0] else None
    adjacency_ds = compile_adjacency_matrix(
        trajectories,
        connectivity,
        mode=mode,
        count_repeated_visits=count_repeated_visits,
        weight=weight,
        include_self_links=include_self_links,
        source_node=source_node,
    )
    adjacency_ds.attrs['source_results_file'] = str(results_file)
    adjacency_ds.attrs['polygon_mode'] = polygon_mode
    adjacency_ds.attrs['group_sources_by_initial_position'] = int(group_sources_by_initial_position)
    adjacency_ds.attrs['source_group_tolerance'] = float(source_group_tolerance)

    path = write_adjacency_netcdf(adjacency_ds, output_file)
    return ConnectivityAdjacencySummary(
        output_file=path,
        n_particles=int(trajectories.x.shape[0]),
        n_nodes=len(connectivity),
        mode=mode,
        weight=weight,
        count_repeated_visits=count_repeated_visits,
        include_self_links=include_self_links,
    )
