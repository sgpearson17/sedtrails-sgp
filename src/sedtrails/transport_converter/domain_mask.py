"""Domain masking and boundary classification helpers."""

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


@dataclass(frozen=True)
class BoundaryEdgeClassification:
    """Boundary edge classes and diagnostics from polygon overrides."""

    edges: np.ndarray
    midpoints: np.ndarray
    classes: np.ndarray
    class_sources: tuple[tuple[str, ...], ...]
    class_counts: dict[str, int]
    polygon_counts: dict[str, int]
    class_files: dict[str, list[str]]

    def to_metadata(self) -> dict[str, Any]:
        """Return a metadata-friendly boundary edge classification table."""
        return {
            'edge_nodes': self.edges.tolist(),
            'edge_midpoints': self.midpoints.tolist(),
            'edge_classes': self.classes.tolist(),
            'edge_class_sources': [list(sources) for sources in self.class_sources],
            'class_counts': dict(self.class_counts),
            'polygon_counts': dict(self.polygon_counts),
            'class_pol_files': {name: list(files) for name, files in self.class_files.items()},
        }


BOUNDARY_CLASS_NAMES = ('open', 'land')


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


def boundary_class_files_from_config(domain_config: Mapping[str, Any] | None) -> dict[str, list[str]]:
    """Return configured boundary-class Tekal files by class."""

    class_files = {class_name: [] for class_name in BOUNDARY_CLASS_NAMES}
    if not domain_config:
        return class_files

    configured = domain_config.get('boundary_class_pol_files', {}) or {}
    for class_name in BOUNDARY_CLASS_NAMES:
        pol_files = configured.get(class_name, [])
        if pol_files is None:
            continue
        if isinstance(pol_files, (str, Path)):
            class_files[class_name] = [str(pol_files)]
        else:
            class_files[class_name] = [str(pol_file) for pol_file in pol_files]
    return class_files


def load_boundary_class_polygons(domain_config: Mapping[str, Any] | None) -> dict[str, list[np.ndarray]]:
    """Load boundary-class override polygons by class from domain configuration."""

    return {
        class_name: read_tekal_polygons(pol_files)
        for class_name, pol_files in boundary_class_files_from_config(domain_config).items()
    }


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


def extract_boundary_edges(connectivity: np.ndarray) -> np.ndarray:
    """Return edges used by exactly one face/triangle in a connectivity table."""

    faces = np.asarray(connectivity, dtype=np.int64)
    edge_counts: dict[tuple[int, int], int] = {}
    oriented_edges: dict[tuple[int, int], tuple[int, int]] = {}

    for face in faces:
        valid = [int(index) for index in face if index >= 0]
        if len(valid) < 2:
            continue
        for index, start in enumerate(valid):
            end = valid[(index + 1) % len(valid)]
            if start == end:
                continue
            key = tuple(sorted((start, end)))
            edge_counts[key] = edge_counts.get(key, 0) + 1
            oriented_edges.setdefault(key, (start, end))

    boundary_edges = [oriented_edges[key] for key, count in edge_counts.items() if count == 1]
    if not boundary_edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(boundary_edges, dtype=np.int64)


def classify_boundary_edges_from_config(
    node_x: np.ndarray,
    node_y: np.ndarray,
    connectivity: np.ndarray,
    domain_config: Mapping[str, Any] | None,
) -> BoundaryEdgeClassification | None:
    """Classify active boundary edges using configured open/land polygon overrides."""

    class_files = boundary_class_files_from_config(domain_config)
    if not any(class_files.values()):
        return None
    class_polygons = load_boundary_class_polygons(domain_config)
    return classify_boundary_edges(node_x, node_y, connectivity, class_polygons, class_files=class_files)


def classify_boundary_edges(
    node_x: np.ndarray,
    node_y: np.ndarray,
    connectivity: np.ndarray,
    class_polygons: Mapping[str, Iterable[np.ndarray]],
    class_files: Mapping[str, list[str]] | None = None,
) -> BoundaryEdgeClassification:
    """Classify active boundary edges by testing edge midpoints against class polygons."""

    x = np.asarray(node_x, dtype=float).ravel()
    y = np.asarray(node_y, dtype=float).ravel()
    edges = extract_boundary_edges(connectivity)
    midpoints = np.full((edges.shape[0], 2), np.nan, dtype=float)
    if edges.size:
        valid = (edges >= 0) & (edges < x.size) & (edges < y.size)
        valid_edges = valid.all(axis=1)
        midpoints[valid_edges, 0] = np.mean(x[edges[valid_edges]], axis=1)
        midpoints[valid_edges, 1] = np.mean(y[edges[valid_edges]], axis=1)

    matches_by_class = {
        class_name: points_inside_any_polygon(midpoints, polygons)
        for class_name, polygons in class_polygons.items()
        if class_name in BOUNDARY_CLASS_NAMES
    }

    edge_classes: list[str] = []
    class_sources: list[tuple[str, ...]] = []
    for edge_index in range(edges.shape[0]):
        matches = tuple(
            class_name
            for class_name in BOUNDARY_CLASS_NAMES
            if matches_by_class.get(class_name, np.zeros(edges.shape[0], dtype=bool))[edge_index]
        )
        class_sources.append(matches)
        if not matches:
            edge_classes.append('unclassified')
        elif 'land' in matches:
            edge_classes.append('land')
        elif len(matches) == 1:
            edge_classes.append(matches[0])
        else:
            edge_classes.append('ambiguous')

    class_array = np.asarray(edge_classes, dtype=object)
    class_counts = {class_name: int(np.count_nonzero(class_array == class_name)) for class_name in BOUNDARY_CLASS_NAMES}
    class_counts['unclassified'] = int(np.count_nonzero(class_array == 'unclassified'))
    class_counts['ambiguous'] = int(np.count_nonzero(class_array == 'ambiguous'))
    polygon_counts = {
        class_name: len([polygon for polygon in polygons if np.asarray(polygon).shape[0] >= 3])
        for class_name, polygons in class_polygons.items()
        if class_name in BOUNDARY_CLASS_NAMES
    }
    for class_name in BOUNDARY_CLASS_NAMES:
        polygon_counts.setdefault(class_name, 0)

    return BoundaryEdgeClassification(
        edges=edges,
        midpoints=midpoints,
        classes=class_array,
        class_sources=tuple(class_sources),
        class_counts=class_counts,
        polygon_counts=polygon_counts,
        class_files={class_name: list(files) for class_name, files in (class_files or {}).items()},
    )


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
