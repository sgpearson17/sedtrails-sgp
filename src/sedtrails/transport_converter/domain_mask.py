"""Domain masking helpers for polygon-defined inner boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import Delaunay

from sedtrails.transport_converter.tekal import read_tekal_polygons


@dataclass(frozen=True)
class ConnectivityMaskResult:
    """Filtered connectivity and diagnostics for a polygon mask operation."""

    connectivity: np.ndarray
    active_mask: np.ndarray
    removed_count: int


def inner_boundary_files_from_config(domain_config: Mapping[str, Any] | None) -> list[str]:
    """Return configured inner-boundary Tekal files as strings."""

    if not domain_config:
        return []

    pol_files = domain_config.get('inner_boundary_pol_files', [])
    if pol_files is None:
        return []
    if isinstance(pol_files, (str, Path)):
        return [str(pol_files)]
    return [str(pol_file) for pol_file in pol_files]


def load_inner_boundary_polygons(domain_config: Mapping[str, Any] | None) -> list[np.ndarray]:
    """Load inner-boundary polygons from a domain configuration mapping."""

    return read_tekal_polygons(inner_boundary_files_from_config(domain_config))


def delaunay_connectivity(node_x: np.ndarray, node_y: np.ndarray) -> np.ndarray:
    """Build triangular candidate connectivity from node coordinates."""

    x = np.asarray(node_x, dtype=float).ravel()
    y = np.asarray(node_y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f'node_x and node_y must have the same shape, got {x.shape} and {y.shape}')
    if x.size < 3:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(Delaunay(np.column_stack((x, y))).simplices, dtype=np.int64)


def filter_connectivity_by_inner_polygons(
    node_x: np.ndarray,
    node_y: np.ndarray,
    connectivity: np.ndarray,
    polygons: Iterable[np.ndarray],
) -> ConnectivityMaskResult:
    """Remove faces or triangles whose centroid falls inside any polygon."""

    faces = np.asarray(connectivity, dtype=np.int64)
    if faces.size == 0:
        empty_mask = np.zeros(faces.shape[0], dtype=bool)
        return ConnectivityMaskResult(connectivity=faces, active_mask=empty_mask, removed_count=0)

    polygon_list = [np.asarray(polygon, dtype=float) for polygon in polygons if np.asarray(polygon).shape[0] >= 3]
    if not polygon_list:
        return ConnectivityMaskResult(
            connectivity=faces.copy(),
            active_mask=np.ones(faces.shape[0], dtype=bool),
            removed_count=0,
        )

    centroids = face_centroids(node_x, node_y, faces)
    inside = points_inside_any_polygon(centroids, polygon_list)
    active_mask = ~inside
    return ConnectivityMaskResult(
        connectivity=faces[active_mask],
        active_mask=active_mask,
        removed_count=int(np.count_nonzero(inside)),
    )


def triangulate_face_connectivity(connectivity: np.ndarray) -> np.ndarray:
    """Convert padded polygon face-node connectivity to triangles by fan split."""

    faces = np.asarray(connectivity, dtype=np.int64)
    triangles: list[list[int]] = []
    for face in faces:
        valid = face[face >= 0]
        if valid.size < 3:
            continue
        if valid.size == 3:
            triangles.append([int(valid[0]), int(valid[1]), int(valid[2])])
            continue
        for index in range(1, valid.size - 1):
            triangles.append([int(valid[0]), int(valid[index]), int(valid[index + 1])])

    if not triangles:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(triangles, dtype=np.int64)


def face_centroids(node_x: np.ndarray, node_y: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    """Compute centroids for padded face-node connectivity."""

    x = np.asarray(node_x, dtype=float).ravel()
    y = np.asarray(node_y, dtype=float).ravel()
    faces = np.asarray(connectivity, dtype=np.int64)
    centroids = np.full((faces.shape[0], 2), np.nan, dtype=float)

    for index, face in enumerate(faces):
        valid = face[(face >= 0) & (face < x.size) & (face < y.size)]
        if valid.size:
            centroids[index, 0] = float(np.mean(x[valid]))
            centroids[index, 1] = float(np.mean(y[valid]))

    return centroids


def points_inside_any_polygon(points: np.ndarray, polygons: Iterable[np.ndarray]) -> np.ndarray:
    """Return a boolean mask for points inside at least one polygon."""

    points_array = np.asarray(points, dtype=float)
    inside = np.zeros(points_array.shape[0], dtype=bool)
    finite = np.isfinite(points_array).all(axis=1)
    if not np.any(finite):
        return inside

    finite_points = points_array[finite]
    finite_inside = np.zeros(finite_points.shape[0], dtype=bool)
    for polygon in polygons:
        polygon_array = np.asarray(polygon, dtype=float)
        if polygon_array.shape[0] < 3:
            continue
        finite_inside |= MplPath(polygon_array[:, :2]).contains_points(finite_points)

    inside[finite] = finite_inside
    return inside
