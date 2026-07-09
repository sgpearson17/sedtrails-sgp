"""Connectivity polygon generation and source assignment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.cluster.vq import kmeans2
from shapely import MultiPoint, Point, Polygon, voronoi_polygons
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class ConnectivityPolygons:
    """Polygons and source-to-node assignments used for connectivity analysis."""

    polygons: tuple[BaseGeometry, ...]
    source_node: np.ndarray
    node_x: np.ndarray
    node_y: np.ndarray
    node_id: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.polygons)


def source_positions_from_arrays(x: np.ndarray, y: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Return initial valid particle positions as ``(n_particles, 2)``."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError('x and y must be 2D arrays with matching shape.')

    if valid_mask is None:
        valid_mask = np.isfinite(x) & np.isfinite(y)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(x) & np.isfinite(y)

    first_valid = np.argmax(valid_mask, axis=1)
    no_valid = ~valid_mask.any(axis=1)
    source_xy = np.column_stack((x[np.arange(x.shape[0]), first_valid], y[np.arange(y.shape[0]), first_valid]))
    source_xy[no_valid, :] = np.nan
    return source_xy


def generate_connectivity_polygons(
    source_xy: np.ndarray,
    *,
    mode: str = 'per_source',
    n_cells: int | None = None,
    source_group: Sequence[int] | None = None,
    domain_polygon: BaseGeometry | np.ndarray | None = None,
    buffer_distance: float = 0.0,
) -> ConnectivityPolygons:
    """Generate connectivity polygons from source coordinates.

    Parameters
    ----------
    source_xy
        Source coordinates as ``(n_sources, 2)``.
    mode
        ``"per_source"`` creates one Voronoi cell per source. ``"n_cells"``
        clusters sources into ``n_cells`` cells. ``"grouped"`` uses explicit
        ``source_group`` labels.
    n_cells
        Number of aggregated cells for ``mode="n_cells"``.
    source_group
        Optional explicit source-to-node labels. Useful when several particles
        should be treated as one source.
    domain_polygon
        Optional clipping polygon. If omitted, the source convex hull buffered
        by ``buffer_distance`` is used.
    buffer_distance
        Buffer applied to the inferred source domain.
    """

    source_xy = np.asarray(source_xy, dtype=float)
    if source_xy.ndim != 2 or source_xy.shape[1] != 2:
        raise ValueError('source_xy must have shape (n_sources, 2).')
    if not np.isfinite(source_xy).all():
        raise ValueError('source_xy must contain only finite coordinates.')

    n_sources = source_xy.shape[0]
    if n_sources == 0:
        raise ValueError('At least one source point is required.')

    clip_polygon = _domain_polygon(source_xy, domain_polygon, buffer_distance)

    if source_group is not None:
        labels = _dense_labels(np.asarray(source_group))
    elif mode == 'per_source':
        labels = np.arange(n_sources, dtype=int)
    elif mode == 'n_cells':
        if n_cells is None:
            raise ValueError("n_cells is required when mode='n_cells'.")
        if n_cells < 1 or n_cells > n_sources:
            raise ValueError('n_cells must be between 1 and the number of sources.')
        _, labels = kmeans2(source_xy, int(n_cells), minit='points', seed=0)
        labels = _dense_labels(labels)
    elif mode == 'grouped':
        raise ValueError("source_group is required when mode='grouped'.")
    else:
        raise ValueError("mode must be one of {'per_source', 'n_cells', 'grouped'}.")

    node_seed_xy = _node_seed_positions(source_xy, labels)
    node_voronoi = _source_voronoi_cells(node_seed_xy, clip_polygon)
    polygons = []
    node_x = []
    node_y = []
    node_ids = []

    for node in range(int(labels.max()) + 1):
        merged = node_voronoi[node].intersection(clip_polygon)
        polygons.append(merged)
        centroid = merged.centroid
        node_x.append(float(centroid.x))
        node_y.append(float(centroid.y))
        node_ids.append(f'node_{node}')

    return ConnectivityPolygons(
        polygons=tuple(polygons),
        source_node=labels.astype(int),
        node_x=np.asarray(node_x, dtype=float),
        node_y=np.asarray(node_y, dtype=float),
        node_id=tuple(node_ids),
    )


def group_sources_by_position(source_xy: np.ndarray, *, tolerance: float = 0.0) -> np.ndarray:
    """Return dense source-group labels for identical or near-identical source positions."""

    source_xy = np.asarray(source_xy, dtype=float)
    if source_xy.ndim != 2 or source_xy.shape[1] != 2:
        raise ValueError('source_xy must have shape (n_sources, 2).')
    if tolerance < 0:
        raise ValueError('tolerance must be non-negative.')
    if tolerance == 0:
        keys = source_xy
    else:
        keys = np.round(source_xy / float(tolerance)).astype(np.int64)
    _, labels = np.unique(keys, axis=0, return_inverse=True)
    return labels.astype(int)


def _dense_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    _, dense = np.unique(labels, return_inverse=True)
    return dense.astype(int)


def _node_seed_positions(source_xy: np.ndarray, labels: np.ndarray) -> np.ndarray:
    seeds = []
    for node in range(int(labels.max()) + 1):
        seeds.append(np.nanmean(source_xy[labels == node], axis=0))
    return np.asarray(seeds, dtype=float)


def _domain_polygon(
    source_xy: np.ndarray,
    domain_polygon: BaseGeometry | np.ndarray | None,
    buffer_distance: float,
) -> BaseGeometry:
    if domain_polygon is not None:
        if isinstance(domain_polygon, np.ndarray):
            domain_polygon = Polygon(domain_polygon)
        return domain_polygon.buffer(0)

    points = MultiPoint(source_xy)
    if len(source_xy) == 1:
        span = max(float(buffer_distance), 1.0)
        return Point(source_xy[0]).buffer(span)

    hull = points.convex_hull
    span = max(np.ptp(source_xy[:, 0]), np.ptp(source_xy[:, 1]), 1.0)
    inferred_buffer = float(buffer_distance) if buffer_distance > 0 else span * 0.05
    return hull.buffer(inferred_buffer)


def _source_voronoi_cells(source_xy: np.ndarray, clip_polygon: BaseGeometry) -> tuple[BaseGeometry, ...]:
    if source_xy.shape[0] == 1:
        return (clip_polygon,)

    try:
        cells = voronoi_polygons(MultiPoint(source_xy), extend_to=clip_polygon.envelope, ordered=True)
        geoms = list(cells.geoms)
    except Exception:
        geoms = []
    if len(geoms) != source_xy.shape[0]:
        # Fallback for GEOS builds where ordered output is unavailable or altered.
        geoms = []
        unordered = list(voronoi_polygons(MultiPoint(source_xy), extend_to=clip_polygon.envelope).geoms)
        for xy in source_xy:
            point = Point(float(xy[0]), float(xy[1]))
            matches = [poly for poly in unordered if poly.buffer(1e-9).contains(point)]
            geoms.append(matches[0] if matches else min(unordered, key=lambda poly: poly.distance(point)))

    return tuple(poly.intersection(clip_polygon).buffer(0) for poly in geoms)
