import copy
import hashlib
import sys
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import count
from typing import Any, ClassVar, Dict

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from sedtrails.particle_tracer.coordinate_transform import (
    DEFAULT_EARTH_RADIUS_M,
    build_coordinate_transform,
    coordinate_system_from_metadata,
    metric_crs_from_metadata,
    source_crs_from_metadata,
)
from sedtrails.particle_tracer.geodetic_geometry import lonlat_to_ecef, normalize_longitude
from sedtrails.transport_converter.domain_mask import (
    DEFAULT_MAX_FALLBACK_TRIANGULATION_POINTS,
)
from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata


_forcing_generation_counter = count(1)
_forcing_generation_lock = threading.Lock()


def _next_forcing_generation() -> int:
    """Return a process-unique monotonically increasing forcing generation."""
    with _forcing_generation_lock:
        return next(_forcing_generation_counter)


@dataclass
class SedtrailsData:
    """
    A data class for internally structuring SedTrails data.

    This class holds data for multiple time steps with time as the first dimension
    for time-dependent variables.

    Attributes:
    -----------

    times: np.ndarray
        Array of time values in seconds since reference_date
    reference_date: np.datetime64
        Reference date for the time values
    x: np.ndarray
        X-coordinates of the grid cells
    y: np.ndarray
        Y-coordinates of the grid cells
    bed_level: np.ndarray
        Bed level in meters (typically time-independent)
    depth_avg_flow_velocity: Dict[str, np.ndarray]
        Depth-averaged flow velocity components in m/s
        (keys: 'x', 'y', 'magnitude', each with time as first dimension)
    fractions: int
        Number of sediment fractions
    bed_load_transport: Dict[str, np.ndarray]
        Bed load sediment transport in kg/m/s
        (keys: 'x', 'y', 'magnitude', each with time as first dimension)
        Shape: (time, fractions, spatial) or (time, spatial)
    suspended_transport: Dict[str, np.ndarray]
        Suspended sediment transport in kg/m/s
        (keys: 'x', 'y', 'magnitude', each with time as first dimension)
        Shape: (time, fractions, spatial) or (time, spatial)
    water_depth: np.ndarray
        Water depth in meters (with time as first dimension)
    mean_bed_shear_stress: np.ndarray
        Mean bed shear stress in pascal (with time as first dimension)
    max_bed_shear_stress: np.ndarray
        Max bed shear stress in pascal (with time as first dimension)
    sediment_concentration: np.ndarray
        Suspended sediment concentration in kg/m^3 (with time as first dimension)
        Shape: (time, fractions, spatial) or (time, spatial)
    nonlinear_wave_velocity: Dict[str, np.ndarray]
        Nonlinear wave velocity in m/s
        (keys: 'x', 'y', 'magnitude', each with time as first dimension)
    metadata: SedtrailsMetadata
        Additional metadata for this dataset

    # === PHYSICS FIELDS (added by PhysicsConverter) ===
    # Note: All physics fields match the structure of transport data
    # Shape: (time, fractions, spatial) if fractions exist, else (time, spatial)
    shields_number: np.ndarray = None
        Shields parameter (dimensionless bed shear stress) [-]
    bed_load_layer_thickness: np.ndarray = None
        Representative thickness of bed load transport layer [m]
    suspended_layer_thickness: np.ndarray = None
        Representative thickness of suspended transport layer [m]
    mixing_layer_thickness: np.ndarray = None
        Mixing layer thickness [m]
    bed_load_velocity: Dict[str, np.ndarray] = None
        Bed load velocity components in m/s
        (keys: 'x', 'y', 'magnitude', each matching transport data structure)
    suspended_velocity: Dict[str, np.ndarray] = None
        Suspended sediment velocity components in m/s
        (keys: 'x', 'y', 'magnitude', each matching transport data structure)
    """

    times: np.ndarray
    reference_date: np.datetime64
    x: np.ndarray
    y: np.ndarray
    bed_level: np.ndarray
    depth_avg_flow_velocity: Dict[str, np.ndarray]
    fractions: int
    bed_load_transport: Dict[str, np.ndarray]
    suspended_transport: Dict[str, np.ndarray]
    water_depth: np.ndarray
    mean_bed_shear_stress: np.ndarray
    max_bed_shear_stress: np.ndarray
    sediment_concentration: np.ndarray
    nonlinear_wave_velocity: Dict[str, np.ndarray]
    metadata: SedtrailsMetadata
    node_x: np.ndarray | None = None
    node_y: np.ndarray | None = None
    face_node_connectivity: np.ndarray | None = None
    particle_face_connectivity: np.ndarray | None = None
    particle_triangle_neighbors: np.ndarray | None = None
    face_node_fill_value: int = -1
    forcing_generation: int = field(
        init=False,
        default_factory=_next_forcing_generation,
    )

    _GRID_METADATA_CACHE_MAX_ENTRIES: ClassVar[int] = 8
    _GRID_METADATA_CACHE_MAX_BYTES: ClassVar[int] = 16 * 1024 * 1024
    _GRID_METADATA_CHUNK_SIZE: ClassVar[int] = 65_536
    _GRID_METADATA_FALLBACK_MAX_POINTS: ClassVar[int] = (
        DEFAULT_MAX_FALLBACK_TRIANGULATION_POINTS
    )
    _grid_metadata_cache: ClassVar[OrderedDict] = OrderedDict()
    _grid_metadata_cache_bytes: ClassVar[int] = 0
    _grid_metadata_cache_lock: ClassVar[threading.RLock] = threading.RLock()

    def __post_init__(self):
        """Initialize container for dynamic physics fields and validate metadata."""

        self._normalize_optional_mesh_geometry()
        # Validate metadata field
        self._validate_metadata()
        # TODO: do we also need to check that min max values are sensible? i.e. min <= max
        self._calculate_timestep()
        self._compute_grid_metadata()
        self._physics_fields: Dict[str, np.ndarray | Dict[str, np.ndarray]] = {}

    def _normalize_optional_mesh_geometry(self) -> None:
        """Normalize optional UGRID-style mesh geometry for dashboard consumers."""
        if self.node_x is not None:
            self.node_x = np.asarray(self.node_x)
        if self.node_y is not None:
            self.node_y = np.asarray(self.node_y)
        if self.face_node_connectivity is not None:
            self.face_node_connectivity = self._normalize_connectivity_dtype(self.face_node_connectivity)
        if self.particle_face_connectivity is not None:
            self.particle_face_connectivity = self._normalize_connectivity_dtype(
                self.particle_face_connectivity
            )
        if self.particle_triangle_neighbors is not None:
            self.particle_triangle_neighbors = self._normalize_connectivity_dtype(
                self.particle_triangle_neighbors
            )

    @staticmethod
    def _normalize_connectivity_dtype(connectivity: np.ndarray) -> np.ndarray:
        """Use compact connectivity indices when their range is int32-safe."""
        array = np.asarray(connectivity)
        if array.dtype == np.dtype(np.int32):
            return array
        if np.issubdtype(array.dtype, np.integer):
            if array.size == 0 or (
                np.min(array) >= np.iinfo(np.int32).min
                and np.max(array) <= np.iinfo(np.int32).max
            ):
                return np.asarray(array, dtype=np.int32)
        return np.asarray(array, dtype=np.int64)

    def mesh_geometry(self) -> Dict[str, Any] | None:
        """
        Return optional face-node mesh geometry for visualization code.

        Returns
        -------
        Dict[str, Any] | None
            Dictionary containing the requested values.
        """
        if self.node_x is None or self.node_y is None or self.face_node_connectivity is None:
            return None

        return {
            'x': self.x,
            'y': self.y,
            'node_x': self.node_x,
            'node_y': self.node_y,
            'face_node_connectivity': self.face_node_connectivity,
            'face_node_fill_value': self.face_node_fill_value,
        }

    def _calculate_timestep(self):
        """Calculate median timestep and add to metadata."""

        if len(self.times) < 2:
            # Cannot calculate timestep with fewer than 2 time points
            timestep = None
        else:
            # Calculate median timestep, this helps ignore the weird startup timesteps
            all_timesteps = np.diff(self.times)
            timestep = float(np.median(np.diff(self.times)))

            # Optional: Add validation
            if timestep <= 0:
                warnings.warn(f'Calculated timestep is non-positive: {timestep}', stacklevel=1)

            # Check if we have timesteps deviating from the median
            tolerance = 1e-6
            deviations = np.abs(all_timesteps - timestep)
            deviating_indices = np.where(deviations > tolerance)[0]

            if len(deviating_indices) > 0:
                warnings.warn(
                    f'Found {len(deviating_indices)} timesteps deviating from median ({timestep:.6f}s)', stacklevel=2
                )

        self.metadata.add('timestep', timestep)

    def _compute_grid_metadata(self):
        """
        Compute grid metadata and add to metadata: minimum resolution and outer envelope.

        Computes:
        - min_resolution: minimum positive connected-edge distance when
          authoritative topology is available, otherwise the nearest-point
          distance for a safety-bounded small grid
        - outer_envelope: seam-aware coordinate bounds for authoritative
          topology, otherwise the small-grid convex hull
        """
        cache_key = self._grid_metadata_cache_key()
        cache_class = type(self)
        with cache_class._grid_metadata_cache_lock:
            cached = cache_class._grid_metadata_cache.get(cache_key)
            if cached is None:
                values = self._calculate_grid_metadata()
                cache_class._store_grid_metadata(cache_key, values)
            else:
                cache_class._grid_metadata_cache.move_to_end(cache_key)
                values = cached[0]

        self.metadata.update(copy.deepcopy(values))

    def _calculate_grid_metadata(self) -> Dict[str, Any]:
        """Calculate grid metadata values without modifying instance metadata."""
        flat_x = np.asarray(self.x).reshape(-1)
        flat_y = np.asarray(self.y).reshape(-1)
        if flat_x.shape != flat_y.shape:
            raise ValueError(
                f'x and y coordinates must have the same size, got {flat_x.size} and {flat_y.size}'
            )

        coordinate_stats = self._bounded_coordinate_stats(flat_x, flat_y)
        if coordinate_stats['count'] < 2 or not self._stats_have_distinct_points(
            coordinate_stats
        ):
            return {
                'min_resolution': None,
                'min_resolution_m': None,
                'outer_envelope': [],
            }

        connectivity = self._grid_point_connectivity(flat_x.size)
        if connectivity is None:
            if coordinate_stats['count'] > self._GRID_METADATA_FALLBACK_MAX_POINTS:
                raise ValueError(
                    'Ocean-scale grid metadata requires particle_face_connectivity '
                    '(or face_node_connectivity that indexes x/y). Provide authoritative '
                    'triangular topology; the topology-free KDTree/ConvexHull fallback is '
                    f'limited to {self._GRID_METADATA_FALLBACK_MAX_POINTS} finite points.'
                )
            coordinates = self._collect_finite_coordinates(
                flat_x,
                flat_y,
                coordinate_stats['count'],
            )
            return self._calculate_small_grid_metadata(coordinates)

        transform = self._bounded_coordinate_transform(coordinate_stats)
        min_resolution_m = self._minimum_topology_edge_resolution(
            flat_x,
            flat_y,
            connectivity,
            transform,
        )
        outer_envelope = self._bounded_outer_envelope(
            flat_x,
            flat_y,
            coordinate_stats,
            transform.is_geographic,
        )
        values = {
            'min_resolution': min_resolution_m,
            'min_resolution_m': min_resolution_m,
            'outer_envelope': outer_envelope,
        }
        values.update(transform.metadata())
        return values

    def _calculate_small_grid_metadata(self, coordinates: np.ndarray) -> Dict[str, Any]:
        """Use the bounded legacy geometry fallback for a small point set."""
        unique_coords = np.unique(coordinates, axis=0)
        if unique_coords.shape[0] < 2:
            return {
                'min_resolution': None,
                'min_resolution_m': None,
                'outer_envelope': [],
            }

        coordinate_system = coordinate_system_from_metadata(self.metadata)
        transform = build_coordinate_transform(
            unique_coords[:, 0],
            unique_coords[:, 1],
            coordinate_system,
            source_crs=source_crs_from_metadata(self.metadata),
            metric_crs=metric_crs_from_metadata(self.metadata),
            runtime_geometry=self.metadata.get('runtime_geometry', 'planar'),
            surface_model=self.metadata.get('surface_model', 'sphere'),
            earth_radius_m=self.metadata.get('earth_radius_m', DEFAULT_EARTH_RADIUS_M),
            longitude_wrap=self.metadata.get('longitude_wrap', 'auto'),
            velocity_basis=self.metadata.get('velocity_basis', 'auto'),
        )
        if transform.is_geodetic:
            unit_ecef = lonlat_to_ecef(
                unique_coords[:, 0],
                unique_coords[:, 1],
                radius=1.0,
            )
            tree = cKDTree(unit_ecef)
            chord_distances, _ = tree.query(unit_ecef, k=2)
            chord = np.clip(chord_distances[:, 1], 0.0, 2.0)
            nearest_distances = (
                2.0
                * transform.earth_radius_m
                * np.arcsin(0.5 * chord)
            )
        else:
            metric_x, metric_y = transform.source_to_metric(unique_coords[:, 0], unique_coords[:, 1])
            unique_metric_coords = np.column_stack((metric_x, metric_y))
            tree = cKDTree(unique_metric_coords)
            distances, _ = tree.query(unique_metric_coords, k=2)
            nearest_distances = distances[:, 1]
        positive_distances = nearest_distances[nearest_distances > 0.0]

        if positive_distances.size == 0:
            min_resolution_m = None
        else:
            min_resolution_m = float(np.min(positive_distances))

        envelope_coords = unique_coords
        if transform.is_geodetic:
            longitude = unique_coords[:, 0]
            radians = np.deg2rad(longitude)
            centre = np.rad2deg(
                np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
            )
            envelope_coords = unique_coords.copy()
            envelope_coords[:, 0] = centre + normalize_longitude(longitude - centre)

        min_x = float(np.min(envelope_coords[:, 0]))
        max_x = float(np.max(envelope_coords[:, 0]))
        min_y = float(np.min(unique_coords[:, 1]))
        max_y = float(np.max(unique_coords[:, 1]))

        # Compute outer envelope using convex hull; degenerate grids have no 2D
        # hull, so use the bounding box directly instead of warning on expected input.
        if envelope_coords.shape[0] < 3 or np.linalg.matrix_rank(
            envelope_coords - envelope_coords.mean(axis=0)
        ) < 2:
            outer_envelope = [[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]
        else:
            try:
                hull = ConvexHull(envelope_coords)
                outer_envelope = envelope_coords[hull.vertices].tolist()
            except Exception as e:
                warnings.warn(f'Convex hull failed ({e}); using bounding box instead.', stacklevel=1)
                outer_envelope = [[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]

        # For projected grids min_resolution preserves its historical meaning;
        # for geographic grids it is the metric CFL spacing.
        values = {
            'min_resolution': min_resolution_m,
            'min_resolution_m': min_resolution_m,
            'outer_envelope': outer_envelope,
        }
        values.update(transform.metadata())
        return values

    def _bounded_coordinate_stats(
        self,
        flat_x: np.ndarray,
        flat_y: np.ndarray,
    ) -> Dict[str, float | int]:
        """Return finite coordinate statistics using bounded temporary arrays."""
        count = 0
        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf
        sum_x = 0.0
        sum_y = 0.0
        longitude_sin_sum = 0.0
        longitude_cos_sum = 0.0
        first_x = np.nan
        for start in range(0, flat_x.size, self._GRID_METADATA_CHUNK_SIZE):
            stop = min(start + self._GRID_METADATA_CHUNK_SIZE, flat_x.size)
            chunk_x = np.asarray(flat_x[start:stop], dtype=np.float64)
            chunk_y = np.asarray(flat_y[start:stop], dtype=np.float64)
            finite = np.isfinite(chunk_x) & np.isfinite(chunk_y)
            if not np.any(finite):
                continue
            valid_x = chunk_x[finite]
            valid_y = chunk_y[finite]
            if count == 0:
                first_x = float(valid_x[0])
            count += valid_x.size
            min_x = min(min_x, float(np.min(valid_x)))
            max_x = max(max_x, float(np.max(valid_x)))
            min_y = min(min_y, float(np.min(valid_y)))
            max_y = max(max_y, float(np.max(valid_y)))
            sum_x += float(np.sum(valid_x, dtype=np.float64))
            sum_y += float(np.sum(valid_y, dtype=np.float64))
            longitude_radians = np.deg2rad(valid_x)
            longitude_sin_sum += float(np.sum(np.sin(longitude_radians)))
            longitude_cos_sum += float(np.sum(np.cos(longitude_radians)))
        return {
            'count': count,
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'sum_x': sum_x,
            'sum_y': sum_y,
            'longitude_sin_sum': longitude_sin_sum,
            'longitude_cos_sum': longitude_cos_sum,
            'first_x': first_x,
        }

    @staticmethod
    def _stats_have_distinct_points(stats: Dict[str, float | int]) -> bool:
        """Return whether bounded coordinate extrema contain two locations."""
        return bool(stats['min_x'] != stats['max_x'] or stats['min_y'] != stats['max_y'])

    def _collect_finite_coordinates(
        self,
        flat_x: np.ndarray,
        flat_y: np.ndarray,
        finite_count: int,
    ) -> np.ndarray:
        """Collect a safety-bounded point set for the legacy fallback."""
        coordinates = np.empty((finite_count, 2), dtype=np.float64)
        output_start = 0
        for start in range(0, flat_x.size, self._GRID_METADATA_CHUNK_SIZE):
            stop = min(start + self._GRID_METADATA_CHUNK_SIZE, flat_x.size)
            chunk_x = np.asarray(flat_x[start:stop], dtype=np.float64)
            chunk_y = np.asarray(flat_y[start:stop], dtype=np.float64)
            finite = np.isfinite(chunk_x) & np.isfinite(chunk_y)
            output_stop = output_start + int(np.count_nonzero(finite))
            coordinates[output_start:output_stop, 0] = chunk_x[finite]
            coordinates[output_start:output_stop, 1] = chunk_y[finite]
            output_start = output_stop
        return coordinates

    def _grid_point_connectivity(self, point_count: int) -> np.ndarray | None:
        """Return authoritative connectivity that indexes the field x/y arrays."""
        connectivity = self.particle_face_connectivity
        if connectivity is None and self.face_node_connectivity is not None:
            node_coordinates_are_field_coordinates = self.node_x is None and self.node_y is None
            if self.node_x is not None and self.node_y is not None:
                node_x = np.asarray(self.node_x)
                node_y = np.asarray(self.node_y)
                node_coordinates_are_field_coordinates = (
                    node_x.size == point_count
                    and node_y.size == point_count
                    and np.shares_memory(node_x, np.asarray(self.x))
                    and np.shares_memory(node_y, np.asarray(self.y))
                )
            if node_coordinates_are_field_coordinates:
                connectivity = self.face_node_connectivity
        if connectivity is None:
            return None

        array = np.asarray(connectivity)
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError('Grid connectivity must have shape (n_faces, n_vertices>=2).')
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError('Grid connectivity must use an integer dtype.')
        for start in range(0, array.shape[0], self._GRID_METADATA_CHUNK_SIZE):
            stop = min(start + self._GRID_METADATA_CHUNK_SIZE, array.shape[0])
            block = array[start:stop]
            if block.size and int(np.max(block)) >= point_count:
                raise ValueError(
                    f'Grid connectivity contains an index outside the {point_count} x/y points.'
                )
        return array

    def _bounded_coordinate_transform(self, stats: Dict[str, float | int]):
        """Build coordinate metadata without gathering all finite points."""
        coordinate_system = coordinate_system_from_metadata(self.metadata)
        mean_x = float(stats['sum_x']) / int(stats['count'])
        mean_y = float(stats['sum_y']) / int(stats['count'])
        sample_x = np.asarray([mean_x])
        sample_y = np.asarray([mean_y])
        runtime_geometry = str(self.metadata.get('runtime_geometry', 'planar'))
        metric_crs = metric_crs_from_metadata(self.metadata)
        metric_crs_text = '' if metric_crs is None else str(metric_crs).strip().lower()
        if coordinate_system == 'geographic' and runtime_geometry.strip().lower() in {
            'planar',
            'local-projected',
            'projected',
            'auto',
        } and metric_crs_text in {'', 'auto', 'auto_utm', 'auto-utm', 'none'}:
            try:
                metric_crs = self._bounded_auto_utm_crs(stats)
            except ValueError:
                if runtime_geometry.strip().lower() != 'auto':
                    raise
                runtime_geometry = 'geodetic'
                metric_crs = None
        return build_coordinate_transform(
            sample_x,
            sample_y,
            coordinate_system,
            source_crs=source_crs_from_metadata(self.metadata),
            metric_crs=metric_crs,
            runtime_geometry=runtime_geometry,
            surface_model=self.metadata.get('surface_model', 'sphere'),
            earth_radius_m=self.metadata.get('earth_radius_m', DEFAULT_EARTH_RADIUS_M),
            longitude_wrap=self.metadata.get('longitude_wrap', 'auto'),
            velocity_basis=self.metadata.get('velocity_basis', 'auto'),
        )

    @staticmethod
    def _bounded_auto_utm_crs(stats: Dict[str, float | int]) -> str:
        """Resolve auto UTM from exact statistics accumulated in bounded passes."""
        min_y = float(stats['min_y'])
        max_y = float(stats['max_y'])
        if min_y < -80.0 or max_y > 84.0:
            raise ValueError(
                'Cannot infer UTM CRS for latitudes outside [-80.0, 84.0] degrees. '
                'Configure general.input_model.metric_crs explicitly.'
            )
        min_x = float(stats['min_x'])
        max_x = float(stats['max_x'])
        if max_x - min_x > 180.0:
            raise ValueError(
                'Cannot infer auto_utm CRS for antimeridian-crossing grids. '
                'Configure metric_crs explicitly.'
            )
        mean_x = float(stats['sum_x']) / int(stats['count'])
        mean_y = float(stats['sum_y']) / int(stats['count'])
        zone = max(1, min(60, int(np.floor((mean_x + 180.0) / 6.0)) + 1))
        central_meridian = -183.0 + 6.0 * zone
        max_offset = max(abs(min_x - central_meridian), abs(max_x - central_meridian))
        if max_offset > 3.0:
            raise ValueError(
                f'Grid spans beyond UTM zone {zone} bounds for auto_utm '
                f'(max longitude offset {max_offset:.3f} deg). '
                'Configure general.input_model.metric_crs explicitly.'
            )
        epsg = (32600 if mean_y >= 0.0 else 32700) + zone
        return f'EPSG:{epsg}'

    def _minimum_topology_edge_resolution(
        self,
        flat_x: np.ndarray,
        flat_y: np.ndarray,
        connectivity: np.ndarray,
        transform,
    ) -> float | None:
        """Return the exact minimum positive connected-edge distance in metres."""
        minimum = np.inf
        width = connectivity.shape[1]
        for start in range(0, connectivity.shape[0], self._GRID_METADATA_CHUNK_SIZE):
            stop = min(start + self._GRID_METADATA_CHUNK_SIZE, connectivity.shape[0])
            block = connectivity[start:stop]
            counts = np.sum(block >= 0, axis=1)
            row_indices = np.arange(block.shape[0])
            for vertex in range(width):
                active = counts > vertex
                if not np.any(active):
                    continue
                active_rows = row_indices[active]
                start_nodes = block[active_rows, vertex]
                next_vertices = (vertex + 1) % counts[active]
                end_nodes = block[active_rows, next_vertices]
                valid = (
                    (start_nodes >= 0)
                    & (end_nodes >= 0)
                    & (start_nodes < flat_x.size)
                    & (end_nodes < flat_x.size)
                )
                if not np.any(valid):
                    continue
                distances = self._topology_edge_distances(
                    flat_x,
                    flat_y,
                    start_nodes[valid],
                    end_nodes[valid],
                    transform,
                )
                positive = distances[np.isfinite(distances) & (distances > 0.0)]
                if positive.size:
                    minimum = min(minimum, float(np.min(positive)))
        return None if not np.isfinite(minimum) else minimum

    @staticmethod
    def _topology_edge_distances(
        flat_x: np.ndarray,
        flat_y: np.ndarray,
        start_nodes: np.ndarray,
        end_nodes: np.ndarray,
        transform,
    ) -> np.ndarray:
        """Return one bounded batch of topology-edge distances in metres."""
        x0 = np.asarray(flat_x[start_nodes], dtype=np.float64)
        y0 = np.asarray(flat_y[start_nodes], dtype=np.float64)
        x1 = np.asarray(flat_x[end_nodes], dtype=np.float64)
        y1 = np.asarray(flat_y[end_nodes], dtype=np.float64)
        if transform.is_geodetic:
            lon0 = np.deg2rad(x0)
            lat0 = np.deg2rad(y0)
            lon1 = np.deg2rad(x1)
            lat1 = np.deg2rad(y1)
            delta_lon = lon1 - lon0
            delta_lat = lat1 - lat0
            haversine = (
                np.sin(0.5 * delta_lat) ** 2
                + np.cos(lat0) * np.cos(lat1) * np.sin(0.5 * delta_lon) ** 2
            )
            haversine = np.clip(haversine, 0.0, 1.0)
            return 2.0 * transform.earth_radius_m * np.arctan2(
                np.sqrt(haversine),
                np.sqrt(1.0 - haversine),
            )
        metric_x0, metric_y0 = transform.source_to_metric(x0, y0)
        metric_x1, metric_y1 = transform.source_to_metric(x1, y1)
        return np.hypot(metric_x1 - metric_x0, metric_y1 - metric_y0)

    def _bounded_outer_envelope(
        self,
        flat_x: np.ndarray,
        flat_y: np.ndarray,
        stats: Dict[str, float | int],
        is_geographic: bool,
    ) -> list[list[float]]:
        """Return a seam-aware bounding envelope using bounded coordinate passes."""
        min_x = float(stats['min_x'])
        max_x = float(stats['max_x'])
        if is_geographic:
            sine = float(stats['longitude_sin_sum'])
            cosine = float(stats['longitude_cos_sum'])
            if abs(sine) + abs(cosine) > np.finfo(np.float64).eps:
                centre = float(np.rad2deg(np.arctan2(sine, cosine)))
            else:
                centre = float(stats['first_x'])
            min_x = np.inf
            max_x = -np.inf
            for start in range(0, flat_x.size, self._GRID_METADATA_CHUNK_SIZE):
                stop = min(start + self._GRID_METADATA_CHUNK_SIZE, flat_x.size)
                chunk_x = np.asarray(flat_x[start:stop], dtype=np.float64)
                chunk_y = np.asarray(flat_y[start:stop], dtype=np.float64)
                finite = np.isfinite(chunk_x) & np.isfinite(chunk_y)
                if not np.any(finite):
                    continue
                unwrapped = centre + normalize_longitude(chunk_x[finite] - centre)
                min_x = min(min_x, float(np.min(unwrapped)))
                max_x = max(max_x, float(np.max(unwrapped)))
        min_y = float(stats['min_y'])
        max_y = float(stats['max_y'])
        return [[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]

    def _grid_metadata_cache_key(self) -> bytes:
        """Return a content-safe key for geometry-derived metadata."""
        digest = hashlib.sha256()
        self._update_grid_digest(digest, 'x', self.x)
        self._update_grid_digest(digest, 'y', self.y)
        connectivity = self._grid_point_connectivity(np.asarray(self.x).size)
        if connectivity is None:
            digest.update(b'grid_point_connectivity:none\0')
        else:
            self._update_grid_digest(
                digest,
                'grid_point_connectivity',
                connectivity,
            )

        settings = (
            ('coordinate_system', coordinate_system_from_metadata(self.metadata)),
            ('source_crs', source_crs_from_metadata(self.metadata)),
            ('metric_crs', metric_crs_from_metadata(self.metadata)),
            ('runtime_geometry', self.metadata.get('runtime_geometry', 'planar')),
            ('surface_model', self.metadata.get('surface_model', 'sphere')),
            ('earth_radius_m', self.metadata.get('earth_radius_m', DEFAULT_EARTH_RADIUS_M)),
            ('longitude_wrap', self.metadata.get('longitude_wrap', 'auto')),
            ('velocity_basis', self.metadata.get('velocity_basis', 'auto')),
        )
        for name, value in settings:
            digest.update(name.encode('ascii'))
            digest.update(repr(value).encode('utf-8'))
            digest.update(b'\0')
        return digest.digest()

    @staticmethod
    def _update_grid_digest(digest, name: str, values: np.ndarray) -> None:
        """Add an array's type, shape, and complete contents to a digest."""
        array = np.asarray(values)
        if array.dtype.hasobject:
            raise TypeError(f'{name} coordinates must use a numeric dtype')

        digest.update(name.encode('ascii'))
        digest.update(array.dtype.str.encode('ascii'))
        digest.update(repr(array.shape).encode('ascii'))
        digest.update(b'\0')
        if array.size == 0:
            return
        if array.flags.c_contiguous:
            digest.update(memoryview(array).cast('B'))
            return

        iterator = np.nditer(
            array,
            flags=['external_loop', 'buffered', 'zerosize_ok'],
            op_flags=['readonly'],
            order='C',
            buffersize=65_536,
        )
        for chunk in iterator:
            digest.update(memoryview(np.ascontiguousarray(chunk)).cast('B'))

    @classmethod
    def _store_grid_metadata(cls, cache_key: bytes, values: Dict[str, Any]) -> None:
        """Store one metadata result while enforcing count and byte limits."""
        stored_values = copy.deepcopy(values)
        entry_bytes = cls._estimate_cache_bytes(stored_values)
        if entry_bytes > cls._GRID_METADATA_CACHE_MAX_BYTES:
            return

        previous = cls._grid_metadata_cache.pop(cache_key, None)
        if previous is not None:
            cls._grid_metadata_cache_bytes -= previous[1]
        cls._grid_metadata_cache[cache_key] = (stored_values, entry_bytes)
        cls._grid_metadata_cache_bytes += entry_bytes

        while (
            len(cls._grid_metadata_cache) > cls._GRID_METADATA_CACHE_MAX_ENTRIES
            or cls._grid_metadata_cache_bytes > cls._GRID_METADATA_CACHE_MAX_BYTES
        ):
            _, (_, removed_bytes) = cls._grid_metadata_cache.popitem(last=False)
            cls._grid_metadata_cache_bytes -= removed_bytes

    @staticmethod
    def _estimate_cache_bytes(value: Any) -> int:
        """Estimate retained bytes for nested metadata containers."""
        seen = set()
        pending = [value]
        total = 0
        while pending:
            current = pending.pop()
            object_id = id(current)
            if object_id in seen:
                continue
            seen.add(object_id)
            total += sys.getsizeof(current)
            if isinstance(current, dict):
                pending.extend(current.keys())
                pending.extend(current.values())
            elif isinstance(current, (list, tuple, set, frozenset)):
                pending.extend(current)
            elif isinstance(current, np.ndarray):
                total += current.nbytes
        return total

    @classmethod
    def _clear_grid_metadata_cache(cls) -> None:
        """Clear reusable metadata state, primarily for isolated tests."""
        with cls._grid_metadata_cache_lock:
            cls._grid_metadata_cache.clear()
            cls._grid_metadata_cache_bytes = 0

    def _validate_metadata(self):
        """Validate that metadata field exists and is the correct type."""

        # Only check that metadata is the right type
        if not isinstance(self.metadata, SedtrailsMetadata):
            raise TypeError(f'metadata must be an instance of SedtrailsMetadata, got {type(self.metadata).__name__}')

    def add_physics_field(self, name: str, data):
        """
        Add a physics field to the data structure.

        Parameters
        ----------

        name : str
            Name of the physics field
        data : np.ndarray or dict
            Physics data (scalar array or dict with 'x', 'y', 'magnitude' for vectors)
        """

        self._physics_fields[name] = data
        setattr(self, name, data)

    def has_physics_field(self, name: str) -> bool:
        """
        Check if a specific physics field exists.

        Parameters
        ----------
        name : str
            Name of the requested object.

        Returns
        -------
        bool
            Boolean result of the check.
        """

        return name in self._physics_fields

    def get_physics_fields(self) -> list:
        """
        Get list of available physics field names.

        Returns
        -------
        list
            Computed value returned by the function.
        """

        return list(self._physics_fields.keys())

    def has_physics_data(self) -> bool:
        """
        Check if any physics fields have been added.

        Returns
        -------
        bool
            Boolean result of the check.
        """

        return len(self._physics_fields) > 0

    # ------------------------------------------------------------------
    # Time slicing
    # ------------------------------------------------------------------
    def __getitem__(self, time_index: int) -> Dict:
        """
        Get data for a specific time index.

        Parameters
        ----------

        time_index : int
            Time index to extract

        Returns
        -------

        Dict
            Dictionary containing all data for the specified time index
        """

        if time_index < 0 or time_index >= len(self.times):
            raise IndexError(f'Time index {time_index} out of bounds (0-{len(self.times) - 1})')

        # Core fields (only include optional fields when present)
        data = {
            'time': self.times[time_index],
            'reference_date': self.reference_date,
            'x': self.x,
            'y': self.y,
            'bed_level': self._get_time_slice_or_static(self.bed_level, time_index),
            'fractions': self.fractions,
        }

        if self.water_depth is not None:
            data['water_depth'] = self.water_depth[time_index]

        if self.mean_bed_shear_stress is not None:
            data['mean_bed_shear_stress'] = self.mean_bed_shear_stress[time_index]

        if self.max_bed_shear_stress is not None:
            data['max_bed_shear_stress'] = self.max_bed_shear_stress[time_index]

        if self.sediment_concentration is not None:
            data['sediment_concentration'] = self.sediment_concentration[time_index]

        if self.depth_avg_flow_velocity is not None:
            data['depth_avg_flow_velocity'] = {
                'x': self.depth_avg_flow_velocity['x'][time_index],
                'y': self.depth_avg_flow_velocity['y'][time_index],
                'magnitude': self.depth_avg_flow_velocity['magnitude'][time_index],
            }

        if self.bed_load_transport is not None:
            data['bed_load_transport'] = {
                'x': self.bed_load_transport['x'][time_index],
                'y': self.bed_load_transport['y'][time_index],
                'magnitude': self.bed_load_transport['magnitude'][time_index],
            }

        if self.suspended_transport is not None:
            data['suspended_transport'] = {
                'x': self.suspended_transport['x'][time_index],
                'y': self.suspended_transport['y'][time_index],
                'magnitude': self.suspended_transport['magnitude'][time_index],
            }

        if self.nonlinear_wave_velocity is not None:
            data['nonlinear_wave_velocity'] = {
                'x': self.nonlinear_wave_velocity['x'][time_index],
                'y': self.nonlinear_wave_velocity['y'][time_index],
                'magnitude': self.nonlinear_wave_velocity['magnitude'][time_index],
            }

        # Dynamic physics fields
        for name, value in self._physics_fields.items():
            if isinstance(value, dict):  # vector field
                data[name] = {
                    'x': value['x'][time_index],
                    'y': value['y'][time_index],
                    'magnitude': value['magnitude'][time_index],
                }
            else:  # scalar field
                data[name] = value[time_index]

        return data

    def _get_time_slice_or_static(self, value: np.ndarray, time_index: int) -> np.ndarray:
        """Return a time slice when the first axis matches times, otherwise static data."""
        if value is None:
            return value
        array = np.asarray(value)
        if array.ndim > 1 and array.shape[0] == len(self.times):
            return array[time_index]
        return value
