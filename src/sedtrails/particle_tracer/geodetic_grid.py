"""Geodetic particle geometry for ocean-scale spherical meshes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from sedtrails.particle_tracer.coordinate_transform import CoordinateTransform
from sedtrails.particle_tracer.geodetic_geometry import (
    EARTH_MEAN_RADIUS_M,
    east_north_basis,
    ecef_to_lonlat,
    lonlat_to_ecef,
    move_lonlat,
    normalize_longitude,
)

TRIANGLE_TOLERANCE = 1.0e-10
MAX_FACE_WALK_STEPS = 128
DEFAULT_VELOCITY_CACHE_BYTES = 512 * 1024**2
BOUNDARY_CLASS_UNCLASSIFIED = np.int8(0)
BOUNDARY_CLASS_OPEN = np.int8(1)
BOUNDARY_CLASS_LAND = np.int8(2)


def _forcing_cache_generation(cache_key) -> int | None:
    """Extract the monotonic forcing generation from a slice cache key."""
    if (
        isinstance(cache_key, tuple)
        and cache_key
        and isinstance(cache_key[0], (int, np.integer))
    ):
        return int(cache_key[0])
    return None


@dataclass
class GeodeticGridGeometry:
    """Cached projection-free geometry for a spherical triangular mesh.

    Particle coordinates exposed through this compatibility backend are
    longitude and latitude in degrees. All interpolation, velocity, distance,
    and movement calculations are performed with ECEF vectors and metres.
    """

    grid_x: np.ndarray
    grid_y: np.ndarray
    coordinate_transform: CoordinateTransform
    triangles: np.ndarray
    triangle_neighbors: np.ndarray
    node_unit_ecef: np.ndarray
    face_centres: np.ndarray
    face_east: np.ndarray
    face_north: np.ndarray
    face_vertex_x: np.ndarray
    face_vertex_y: np.ndarray
    inv00: np.ndarray
    inv01: np.ndarray
    inv10: np.ndarray
    inv11: np.ndarray
    triangle_edge_class_codes: np.ndarray
    outer_envelope: np.ndarray
    triangle_finder: Any = None
    triangulation: Any = None
    boundary_edges: np.ndarray | None = None
    boundary_edge_class_codes: np.ndarray | None = None
    metric_grid_x: np.ndarray = field(init=False)
    metric_grid_y: np.ndarray = field(init=False)
    p0_x: np.ndarray = field(init=False)
    p0_y: np.ndarray = field(init=False)
    _face_tree: cKDTree = field(init=False, repr=False)
    _velocity_cache: OrderedDict[tuple, np.ndarray] = field(
        init=False,
        default_factory=OrderedDict,
        repr=False,
    )
    _velocity_cache_bytes: int = field(init=False, default=0, repr=False)
    _velocity_cache_max_bytes: int = field(
        init=False,
        default=DEFAULT_VELOCITY_CACHE_BYTES,
        repr=False,
    )
    _velocity_cache_generation: int | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.metric_grid_x = self.grid_x
        self.metric_grid_y = self.grid_y
        self.p0_x = self.face_vertex_x[:, 0]
        self.p0_y = self.face_vertex_y[:, 0]
        self._face_tree = cKDTree(self.face_centres)

    @classmethod
    def from_points(
        cls,
        grid_x,
        grid_y,
        *,
        triangles,
        boundary_edge_classification=None,
        source_crs='EPSG:4326',
        surface_model='sphere',
        earth_radius_m=EARTH_MEAN_RADIUS_M,
        longitude_wrap='auto',
        velocity_basis='east_north',
    ) -> 'GeodeticGridGeometry':
        """Build a geodetic grid from longitude/latitude and connectivity.

        Parameters
        ----------
        grid_x, grid_y : array-like
            Longitude and latitude node coordinates in degrees.
        triangles : array-like
            Authoritative triangular connectivity with shape ``(n, 3)``.
        boundary_edge_classification : mapping, optional
            Boundary edge nodes and ``open``/``land`` labels.
        source_crs : str, default='EPSG:4326'
            Geographic source CRS.
        surface_model : str, default='sphere'
            Surface model. The operational backend currently supports a
            sphere; ellipsoidal input is rejected rather than approximated.
        earth_radius_m : float, default=6371008.8
            Sphere radius in metres.
        longitude_wrap : str, default='auto'
            Output longitude wrapping convention.
        velocity_basis : str, default='east_north'
            Input horizontal vector basis.

        Returns
        -------
        GeodeticGridGeometry
            Cached spherical mesh geometry.
        """
        longitude = np.asarray(grid_x, dtype=np.float64).ravel()
        latitude = np.asarray(grid_y, dtype=np.float64).ravel()
        if longitude.shape != latitude.shape:
            raise ValueError('grid_x and grid_y must have the same shape.')
        if not np.all(np.isfinite(longitude)) or not np.all(np.isfinite(latitude)):
            raise ValueError('Geodetic grid coordinates must be finite.')
        if np.any((latitude < -90.0) | (latitude > 90.0)):
            raise ValueError('Geodetic grid latitude must be within [-90, 90] degrees.')

        index_dtype = np.int32 if longitude.size <= np.iinfo(np.int32).max else np.int64
        triangle_array = np.asarray(triangles, dtype=index_dtype)
        if triangle_array.ndim != 2 or triangle_array.shape[1] != 3 or triangle_array.shape[0] == 0:
            raise ValueError(
                'Geodetic runtime geometry requires authoritative triangular connectivity with shape (n, 3).'
            )
        if np.any(triangle_array < 0) or np.any(triangle_array >= longitude.size):
            raise ValueError('Triangle connectivity contains an out-of-range node index.')

        transform = CoordinateTransform(
            'geographic',
            source_crs,
            None,
            runtime_geometry='geodetic',
            surface_model=surface_model,
            earth_radius_m=earth_radius_m,
            longitude_wrap=longitude_wrap,
            velocity_basis=velocity_basis,
        )
        if transform.surface_model != 'sphere':
            raise NotImplementedError(
                'The geodetic particle backend currently supports surface_model="sphere" only.'
            )
        if transform.velocity_basis not in {'auto', 'east_north'}:
            raise ValueError(
                'Geodetic runtime geometry requires east/north velocity components; '
                f'got velocity_basis={transform.velocity_basis!r}.'
            )

        node_unit = lonlat_to_ecef(longitude, latitude, radius=1.0)
        (
            face_centres,
            face_east,
            face_north,
            face_vertex_x,
            face_vertex_y,
            inv00,
            inv01,
            inv10,
            inv11,
        ) = _build_face_charts(node_unit, triangle_array, transform.earth_radius_m)
        neighbors = _compute_triangle_neighbors(triangle_array)
        boundary_edges, boundary_codes = _parse_boundary_edge_classification(
            boundary_edge_classification,
            longitude.size,
        )
        triangle_edge_codes = _build_triangle_edge_class_codes(
            triangle_array,
            boundary_edges,
            boundary_codes,
        )

        centre_longitude = _circular_mean_longitude(longitude)
        unwrapped = centre_longitude + normalize_longitude(longitude - centre_longitude)
        envelope = np.array(
            [
                [np.min(unwrapped), np.min(latitude)],
                [np.min(unwrapped), np.max(latitude)],
                [np.max(unwrapped), np.max(latitude)],
                [np.max(unwrapped), np.min(latitude)],
            ],
            dtype=np.float64,
        )
        return cls(
            grid_x=longitude,
            grid_y=latitude,
            coordinate_transform=transform,
            triangles=triangle_array,
            triangle_neighbors=neighbors,
            node_unit_ecef=node_unit,
            face_centres=face_centres,
            face_east=face_east,
            face_north=face_north,
            face_vertex_x=face_vertex_x,
            face_vertex_y=face_vertex_y,
            inv00=inv00,
            inv01=inv01,
            inv10=inv10,
            inv11=inv11,
            triangle_edge_class_codes=triangle_edge_codes,
            outer_envelope=envelope,
            boundary_edges=boundary_edges,
            boundary_edge_class_codes=boundary_codes,
        )

    @property
    def is_geodetic(self) -> bool:
        """Return ``True`` for backend dispatch."""
        return True

    def find_triangle(self, x, y) -> int:
        """Return the containing spherical triangle, or ``-1``."""
        return int(self.locate_points(np.asarray([x]), np.asarray([y]))[0])

    def locate_points(self, x_points, y_points, start_simplices=None) -> np.ndarray:
        """Locate longitude/latitude points using cached faces then an ECEF tree."""
        longitude = np.asarray(x_points, dtype=np.float64).ravel()
        latitude = np.asarray(y_points, dtype=np.float64).ravel()
        if longitude.shape != latitude.shape:
            raise ValueError('x_points and y_points must have the same shape.')
        if longitude.size == 0:
            return np.empty(0, dtype=np.int64)
        points = lonlat_to_ecef(longitude, latitude, radius=1.0)
        return self._locate_unit_ecef_points(points, start_simplices)

    def _locate_unit_ecef_points(self, points, start_simplices=None) -> np.ndarray:
        """Locate unit ECEF points using compiled face walks and tree fallback."""
        unit_points = np.ascontiguousarray(points, dtype=np.float64)
        if unit_points.ndim != 2 or unit_points.shape[1] != 3:
            raise ValueError('ECEF point array must have shape (n, 3).')
        result = np.full(unit_points.shape[0], -1, dtype=np.int64)

        if start_simplices is not None:
            starts = np.asarray(start_simplices, dtype=np.int64).ravel()
            if starts.shape == result.shape:
                result = _walk_points_ecef_numba(
                    unit_points,
                    starts,
                    self.triangles,
                    self.triangle_neighbors,
                    self.node_unit_ecef,
                    self.face_centres,
                    MAX_FACE_WALK_STEPS,
                    TRIANGLE_TOLERANCE,
                )

        missing = np.flatnonzero(result < 0)
        if missing.size == 0:
            return result
        candidate_count = min(32, self.triangles.shape[0])
        _, candidates = self._face_tree.query(unit_points[missing], k=candidate_count)
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.ndim == 1:
            candidates = candidates[:, np.newaxis]
        result[missing] = _locate_candidates_ecef_numba(
            unit_points[missing],
            np.ascontiguousarray(candidates),
            self.triangles,
            self.node_unit_ecef,
            self.face_centres,
            TRIANGLE_TOLERANCE,
        )
        return result

    def interpolate_field(self, field_values, x_points, y_points):
        """Interpolate one nodal scalar field on the spherical mesh."""
        values, _ = self.interpolate_fields_with_simplex(
            (field_values,),
            x_points,
            y_points,
        )
        return values[0]

    def interpolate_fields(self, fields, x_points, y_points):
        """Interpolate multiple nodal scalar fields with one point-location pass."""
        values, _ = self.interpolate_fields_with_simplex(fields, x_points, y_points)
        return values

    def interpolate_fields_with_simplex(
        self,
        fields,
        x_points,
        y_points,
        simplex_ids=None,
    ):
        """Interpolate fields and return containing face IDs."""
        longitude = np.asarray(x_points, dtype=np.float64)
        latitude = np.asarray(y_points, dtype=np.float64)
        shape = longitude.shape
        faces = self.locate_points(longitude, latitude, simplex_ids)
        weights = self._weights_for_lonlat(longitude.ravel(), latitude.ravel(), faces)
        valid = faces >= 0
        outputs = []
        for field_values in fields:
            field_array = np.asarray(field_values).ravel()
            if field_array.size != self.grid_x.size:
                raise ValueError(
                    f'Field size {field_array.size} does not match geodetic node count {self.grid_x.size}.'
                )
            output = np.full(faces.shape, np.nan, dtype=np.result_type(field_array.dtype, np.float64))
            if np.any(valid):
                nodes = self.triangles[faces[valid]]
                output[valid] = np.sum(field_array[nodes] * weights[valid], axis=1)
            outputs.append(output.reshape(shape))
        return tuple(outputs), faces

    def interpolate_vector(self, grid_u, grid_v, x_points, y_points):
        """Interpolate east/north vectors through their ECEF representation."""
        longitude = np.asarray(x_points, dtype=np.float64)
        latitude = np.asarray(y_points, dtype=np.float64)
        faces = self.locate_points(longitude, latitude)
        unit_points = lonlat_to_ecef(longitude.ravel(), latitude.ravel(), radius=1.0)
        vectors = self._interpolate_ecef_velocity(
            self.prepare_vector_field(grid_u, grid_v),
            unit_points,
            faces,
        )
        east, north = east_north_basis(longitude.ravel(), latitude.ravel())
        return (
            np.sum(vectors * east, axis=1).reshape(longitude.shape),
            np.sum(vectors * north, axis=1).reshape(longitude.shape),
        )

    def interpolate_temporal_vector(
        self,
        lower_u,
        lower_v,
        upper_u,
        upper_v,
        weight,
        x_points,
        y_points,
    ):
        """Interpolate and blend two east/north vector slices."""
        lower_x, lower_y = self.interpolate_vector(lower_u, lower_v, x_points, y_points)
        if weight <= 0.0:
            return lower_x, lower_y
        upper_x, upper_y = self.interpolate_vector(upper_u, upper_v, x_points, y_points)
        return (
            lower_x + float(weight) * (upper_x - lower_x),
            lower_y + float(weight) * (upper_y - lower_y),
        )

    def prepare_vector_field(self, grid_u, grid_v, *, cache_key=None) -> np.ndarray:
        """Convert one east/north node field to a bounded cached ECEF field.

        Parameters
        ----------
        grid_u, grid_v : array-like
            Eastward and northward nodal components.
        cache_key : hashable, optional
            Explicit forcing-slice generation key. Writable arrays are cached
            only when this key is supplied; callers must use a new key after
            mutating a reused input buffer.

        Returns
        -------
        numpy.ndarray
            Contiguous ECEF tangent vectors with shape ``(n_nodes, 3)``.
        """
        u = np.asarray(grid_u).ravel()
        v = np.asarray(grid_v).ravel()
        if u.size != self.grid_x.size or v.size != self.grid_x.size:
            raise ValueError('Vector field size must match the geodetic node count.')
        cache_allowed = cache_key is not None
        if cache_key is not None:
            try:
                hash(cache_key)
            except TypeError as exc:
                raise TypeError('cache_key must be hashable.') from exc
        key = cache_key
        if cache_allowed:
            generation = _forcing_cache_generation(cache_key)
            if (
                generation is not None
                and generation != self._velocity_cache_generation
            ):
                self.clear_velocity_cache()
                self._velocity_cache_generation = generation
            cached = self._velocity_cache.get(key)
            if cached is not None:
                self._velocity_cache.move_to_end(key)
                return cached
        vectors = _east_north_to_ecef_numba(
            np.ascontiguousarray(u),
            np.ascontiguousarray(v),
            self.grid_x,
            self.grid_y,
        )
        if (
            cache_allowed
            and vectors.nbytes <= self._velocity_cache_max_bytes
        ):
            while (
                self._velocity_cache
                and self._velocity_cache_bytes + vectors.nbytes > self._velocity_cache_max_bytes
            ):
                _, evicted = self._velocity_cache.popitem(last=False)
                self._velocity_cache_bytes -= evicted.nbytes
            self._velocity_cache[key] = vectors
            self._velocity_cache_bytes += vectors.nbytes
        return vectors

    def clear_velocity_cache(self) -> None:
        """Release all prepared ECEF velocity slices held by this geometry."""
        self._velocity_cache.clear()
        self._velocity_cache_bytes = 0

    def update_particles_with_simplex(
        self,
        x0,
        y0,
        grid_u,
        grid_v,
        dt,
        simplex_ids=None,
        igeo=0,
    ):
        """Advance particles and return updated spherical face IDs."""
        x, y, faces, _ = self.update_particles_with_boundary_class(
            x0,
            y0,
            grid_u,
            grid_v,
            dt,
            simplex_ids=simplex_ids,
            igeo=igeo,
        )
        return x, y, faces

    def update_particles_temporal_with_simplex(
        self,
        x0,
        y0,
        lower_u,
        lower_v,
        upper_u,
        upper_v,
        weight,
        dt,
        simplex_ids=None,
        igeo=0,
    ):
        """Advance particles with temporal field interpolation."""
        x, y, faces, _ = self.update_particles_temporal_with_boundary_class(
            x0,
            y0,
            lower_u,
            lower_v,
            upper_u,
            upper_v,
            weight,
            dt,
            simplex_ids=simplex_ids,
            igeo=igeo,
        )
        return x, y, faces

    def update_particles_with_boundary_class(
        self,
        x0,
        y0,
        grid_u,
        grid_v,
        dt,
        simplex_ids=None,
        igeo=0,
    ):
        """Advance particles and return crossed boundary codes."""
        return self.update_particles_temporal_with_boundary_class(
            x0,
            y0,
            grid_u,
            grid_v,
            grid_u,
            grid_v,
            0.0,
            dt,
            simplex_ids=simplex_ids,
            igeo=igeo,
        )

    def update_particles_temporal_with_boundary_class(
        self,
        x0,
        y0,
        lower_u,
        lower_v,
        upper_u,
        upper_v,
        weight,
        dt,
        simplex_ids=None,
        igeo=0,
    ):
        """Advance particles with normalized ECEF RK4 and face walking."""
        lower = self.prepare_vector_field(lower_u, lower_v)
        upper = lower if weight <= 0.0 else self.prepare_vector_field(upper_u, upper_v)
        return self.update_particles_prepared_temporal_with_boundary_class(
            x0,
            y0,
            lower,
            upper,
            weight,
            dt,
            simplex_ids=simplex_ids,
            igeo=igeo,
        )

    def update_particles_prepared_temporal_with_boundary_class(
        self,
        x0,
        y0,
        lower_ecef,
        upper_ecef,
        weight,
        dt,
        simplex_ids=None,
        igeo=0,
    ):
        """Advance particles with already prepared ECEF vector slices.

        Parameters
        ----------
        x0, y0 : array-like
            Initial longitude and latitude coordinates in degrees.
        lower_ecef, upper_ecef : array-like
            Prepared nodal ECEF velocities with shape ``(n_nodes, 3)``.
        weight : float
            Temporal interpolation weight.
        dt : float
            Integration timestep in seconds.
        simplex_ids : array-like, optional
            Cached starting face IDs.
        igeo : int, default=0
            Compatibility flag. Value 1 is invalid for geodetic geometry.

        Returns
        -------
        tuple
            New longitude, latitude, face IDs, and boundary class codes.
        """
        if int(igeo) == 1:
            raise ValueError('igeo=1 is not used by geodetic runtime geometry.')
        longitude = np.asarray(x0, dtype=np.float64)
        latitude = np.asarray(y0, dtype=np.float64)
        shape = longitude.shape
        if longitude.size == 0:
            return (
                longitude.copy(),
                latitude.copy(),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int8),
            )
        starts = self.locate_points(longitude, latitude, simplex_ids)
        lower = _validate_prepared_vector_field(lower_ecef, self.grid_x.size)
        upper = lower if weight <= 0.0 else _validate_prepared_vector_field(
            upper_ecef,
            self.grid_x.size,
        )
        destination = self._rk4_ecef(
            longitude.ravel(),
            latitude.ravel(),
            starts,
            lower,
            upper,
            float(weight),
            float(dt),
        )
        new_longitude, new_latitude = ecef_to_lonlat(destination)
        new_longitude = self._wrap_longitude(new_longitude)
        unit_destination = destination / self.coordinate_transform.earth_radius_m
        new_faces = self._locate_unit_ecef_points(unit_destination, starts)
        boundary_codes = self._boundary_codes_for_exits(
            destination,
            starts,
            new_faces,
        )
        return (
            new_longitude.reshape(shape),
            new_latitude.reshape(shape),
            new_faces,
            boundary_codes.reshape(shape),
        )

    def update_particles(self, x0, y0, grid_u, grid_v, dt, igeo=0):
        """Advance particles without exposing face IDs."""
        x, y, _ = self.update_particles_with_simplex(x0, y0, grid_u, grid_v, dt, igeo=igeo)
        return x, y

    def update_particles_temporal(
        self,
        x0,
        y0,
        lower_u,
        lower_v,
        upper_u,
        upper_v,
        weight,
        dt,
        igeo=0,
    ):
        """Advance particles with temporal interpolation without face IDs."""
        x, y, _ = self.update_particles_temporal_with_simplex(
            x0,
            y0,
            lower_u,
            lower_v,
            upper_u,
            upper_v,
            weight,
            dt,
            igeo=igeo,
        )
        return x, y

    def apply_diffusion(self, longitude, latitude, east_m, north_m):
        """Apply tangent-plane diffusion displacements in metres."""
        moved_longitude, moved_latitude = move_lonlat(
            longitude,
            latitude,
            east_m,
            north_m,
            radius=self.coordinate_transform.earth_radius_m,
        )
        return self._wrap_longitude(moved_longitude), moved_latitude

    def classify_boundary_crossings(self, x0, y0, x1, y1) -> np.ndarray:
        """Return topological boundary labels for segments leaving the mesh."""
        starts = self.locate_points(x0, y0)
        destinations = lonlat_to_ecef(x1, y1, radius=1.0)
        new_faces = self.locate_points(x1, y1, starts)
        codes = self._boundary_codes_for_exits(destinations, starts, new_faces)
        labels = np.full(codes.shape, 'unclassified', dtype=object)
        labels[codes == BOUNDARY_CLASS_OPEN] = 'open'
        labels[codes == BOUNDARY_CLASS_LAND] = 'land'
        return labels.reshape(np.asarray(x0).shape)

    def _rk4_ecef(self, longitude, latitude, starts, lower, upper, weight, dt):
        radius = self.coordinate_transform.earth_radius_m
        unit0 = lonlat_to_ecef(longitude, latitude, radius=1.0)
        r0 = radius * unit0
        k1 = self._velocity_at_ecef(unit0, starts, lower, upper, weight)
        r2 = _normalize_to_radius(r0 + 0.5 * dt * k1, radius)
        unit2 = r2 / radius
        f2 = self._locate_unit_ecef_points(unit2, starts)
        k2 = self._velocity_at_ecef(unit2, f2, lower, upper, weight)
        r3 = _normalize_to_radius(r0 + 0.5 * dt * k2, radius)
        unit3 = r3 / radius
        f3 = self._locate_unit_ecef_points(unit3, f2)
        k3 = self._velocity_at_ecef(unit3, f3, lower, upper, weight)
        r4 = _normalize_to_radius(r0 + dt * k3, radius)
        unit4 = r4 / radius
        f4 = self._locate_unit_ecef_points(unit4, f3)
        k4 = self._velocity_at_ecef(unit4, f4, lower, upper, weight)
        return _normalize_to_radius(r0 + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, radius)

    def _velocity_at_ecef(self, unit_positions, faces, lower, upper, weight):
        lower_values = self._interpolate_ecef_velocity(lower, unit_positions, faces)
        if weight <= 0.0:
            values = lower_values
        else:
            upper_values = self._interpolate_ecef_velocity(upper, unit_positions, faces)
            values = lower_values + weight * (upper_values - lower_values)
        return values - np.sum(values * unit_positions, axis=1)[:, np.newaxis] * unit_positions

    def _interpolate_ecef_velocity(self, field, unit_points, faces):
        weights = self._weights_for_ecef(unit_points, faces)
        result = np.zeros((len(faces), 3), dtype=np.float64)
        valid = faces >= 0
        if np.any(valid):
            nodes = self.triangles[faces[valid]]
            result[valid] = np.sum(field[nodes] * weights[valid, :, np.newaxis], axis=1)
        return result

    def _weights_for_lonlat(self, longitude, latitude, faces):
        points = lonlat_to_ecef(longitude, latitude, radius=1.0)
        return self._weights_for_ecef(points, faces)

    def _weights_for_ecef(self, points, faces):
        """Return face-local interpolation weights for unit ECEF points."""
        weights = np.full((len(faces), 3), np.nan, dtype=np.float64)
        valid = faces >= 0
        if not np.any(valid):
            return weights
        face_ids = faces[valid]
        denominator = np.sum(points[valid] * self.face_centres[face_ids], axis=1)
        local_x = (
            self.coordinate_transform.earth_radius_m
            * np.sum(points[valid] * self.face_east[face_ids], axis=1)
            / denominator
        )
        local_y = (
            self.coordinate_transform.earth_radius_m
            * np.sum(points[valid] * self.face_north[face_ids], axis=1)
            / denominator
        )
        dx = local_x - self.face_vertex_x[face_ids, 0]
        dy = local_y - self.face_vertex_y[face_ids, 0]
        w1 = self.inv00[face_ids] * dx + self.inv01[face_ids] * dy
        w2 = self.inv10[face_ids] * dx + self.inv11[face_ids] * dy
        weights[valid, 0] = 1.0 - w1 - w2
        weights[valid, 1] = w1
        weights[valid, 2] = w2
        return weights

    def _boundary_codes_for_exits(self, destinations, starts, new_faces):
        return _boundary_codes_for_exits_numba(
            np.ascontiguousarray(destinations, dtype=np.float64),
            np.ascontiguousarray(starts, dtype=np.int64),
            np.ascontiguousarray(new_faces, dtype=np.int64),
            self.triangles,
            self.triangle_neighbors,
            self.node_unit_ecef,
            self.face_centres,
            self.triangle_edge_class_codes,
            MAX_FACE_WALK_STEPS,
            TRIANGLE_TOLERANCE,
        )

    def _wrap_longitude(self, longitude):
        """Apply the configured public longitude convention."""
        wrap = self.coordinate_transform.longitude_wrap
        if wrap == '0_360' or (wrap == 'auto' and np.all(self.grid_x >= 0.0)):
            return np.mod(np.asarray(longitude, dtype=float), 360.0)
        return normalize_longitude(longitude)


def _build_face_charts(node_unit, triangles, radius):
    face_count = triangles.shape[0]
    face_centres = np.empty((face_count, 3), dtype=np.float64)
    face_east = np.empty((face_count, 3), dtype=np.float64)
    face_north = np.empty((face_count, 3), dtype=np.float64)
    x = np.empty((face_count, 3), dtype=np.float64)
    y = np.empty((face_count, 3), dtype=np.float64)
    inv00 = np.empty(face_count, dtype=np.float64)
    inv01 = np.empty(face_count, dtype=np.float64)
    inv10 = np.empty(face_count, dtype=np.float64)
    inv11 = np.empty(face_count, dtype=np.float64)
    chunk_size = 262_144
    for start in range(0, face_count, chunk_size):
        stop = min(start + chunk_size, face_count)
        vertices = node_unit[triangles[start:stop]]
        centres = np.sum(vertices, axis=1)
        norms = np.linalg.norm(centres, axis=1)
        if np.any(norms <= 1.0e-15):
            raise ValueError('Geodetic triangles must define convex regions smaller than a hemisphere.')
        centres /= norms[:, np.newaxis]
        centre_lon, centre_lat = ecef_to_lonlat(centres)
        east, north = east_north_basis(centre_lon, centre_lat)
        denominator = np.sum(vertices * centres[:, np.newaxis, :], axis=2)
        if np.any(denominator <= 1.0e-12):
            raise ValueError('A geodetic triangle is too large for a stable face-local chart.')
        local_x = radius * np.sum(vertices * east[:, np.newaxis, :], axis=2) / denominator
        local_y = radius * np.sum(vertices * north[:, np.newaxis, :], axis=2) / denominator
        m00 = local_x[:, 1] - local_x[:, 0]
        m01 = local_x[:, 2] - local_x[:, 0]
        m10 = local_y[:, 1] - local_y[:, 0]
        m11 = local_y[:, 2] - local_y[:, 0]
        determinant = m00 * m11 - m01 * m10
        if np.any(np.abs(determinant) <= 1.0e-12):
            raise ValueError('Geodetic triangle has a degenerate face-local chart.')
        face_centres[start:stop] = centres
        face_east[start:stop] = east
        face_north[start:stop] = north
        x[start:stop] = local_x
        y[start:stop] = local_y
        inv00[start:stop] = m11 / determinant
        inv01[start:stop] = -m01 / determinant
        inv10[start:stop] = -m10 / determinant
        inv11[start:stop] = m00 / determinant
    return (
        face_centres,
        face_east,
        face_north,
        x,
        y,
        inv00,
        inv01,
        inv10,
        inv11,
    )


def _compute_triangle_neighbors(triangles):
    index_dtype = triangles.dtype if np.issubdtype(triangles.dtype, np.integer) else np.int64
    neighbors = np.full((triangles.shape[0], 3), -1, dtype=index_dtype)
    if triangles.shape[0] == 0:
        return neighbors
    edges = np.stack(
        (
            triangles[:, [1, 2]],
            triangles[:, [0, 2]],
            triangles[:, [0, 1]],
        ),
        axis=1,
    ).reshape(-1, 2)
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    group_start = np.r_[0, 1 + np.flatnonzero(np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1))]
    group_end = np.r_[group_start[1:], sorted_edges.shape[0]]
    counts = group_end - group_start
    if np.any(counts > 2):
        raise ValueError('Geodetic triangle connectivity contains a non-manifold edge.')
    paired = group_start[counts == 2]
    first = order[paired]
    second = order[paired + 1]
    neighbor_flat = neighbors.ravel()
    neighbor_flat[first] = second // 3
    neighbor_flat[second] = first // 3
    return neighbors


def _parse_boundary_edge_classification(classification, node_count):
    if not classification:
        return None, None
    edges = np.asarray(classification.get('edge_nodes'), dtype=np.int64)
    labels = np.asarray(classification.get('edge_classes'), dtype=object)
    if edges.ndim != 2 or edges.shape[1] != 2 or labels.shape != (edges.shape[0],):
        return None, None
    valid = ((edges >= 0) & (edges < node_count)).all(axis=1)
    edges = edges[valid]
    labels = labels[valid]
    codes = np.zeros(labels.shape, dtype=np.int8)
    codes[labels == 'open'] = BOUNDARY_CLASS_OPEN
    codes[labels == 'land'] = BOUNDARY_CLASS_LAND
    return edges, codes


def _build_triangle_edge_class_codes(triangles, boundary_edges, boundary_codes):
    codes = np.zeros((triangles.shape[0], 3), dtype=np.int8)
    if boundary_edges is None:
        return codes
    lookup = {
        tuple(sorted((int(edge[0]), int(edge[1])))): int(code)
        for edge, code in zip(boundary_edges, boundary_codes, strict=True)
    }
    opposite_edges = ((1, 2), (0, 2), (0, 1))
    for face, triangle in enumerate(triangles):
        for edge_index, edge_vertices in enumerate(opposite_edges):
            edge = tuple(sorted((int(triangle[edge_vertices[0]]), int(triangle[edge_vertices[1]]))))
            codes[face, edge_index] = lookup.get(edge, 0)
    return codes


def _normalize_to_radius(vectors, radius):
    norms = np.linalg.norm(vectors, axis=-1)
    if np.any(norms <= 0.0):
        raise ValueError('Cannot normalize a zero-length ECEF vector.')
    return radius * vectors / norms[:, np.newaxis]


def _validate_prepared_vector_field(values, node_count):
    """Return a validated contiguous prepared ECEF vector field."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (node_count, 3):
        raise ValueError(
            f'Prepared vector field must have shape ({node_count}, 3), got {array.shape}.'
        )
    return np.ascontiguousarray(array)


def _circular_mean_longitude(longitude):
    radians = np.deg2rad(longitude)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))))


@njit(cache=True, parallel=True)
def _east_north_to_ecef_numba(u, v, longitude, latitude):
    """Convert east/north nodal vectors to ECEF without basis temporaries."""
    vectors = np.empty((u.shape[0], 3), dtype=np.float64)
    degrees_to_radians = np.pi / 180.0
    for index in prange(u.shape[0]):
        lon = longitude[index] * degrees_to_radians
        lat = latitude[index] * degrees_to_radians
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        vectors[index, 0] = -u[index] * sin_lon - v[index] * sin_lat * cos_lon
        vectors[index, 1] = u[index] * cos_lon - v[index] * sin_lat * sin_lon
        vectors[index, 2] = v[index] * cos_lat
    return vectors


@njit(cache=True, inline='always')
def _face_sides_ecef_numba(point, face, triangles, node_unit_ecef, face_centres):
    """Return inward half-space values for one point and spherical face."""
    i0 = triangles[face, 0]
    i1 = triangles[face, 1]
    i2 = triangles[face, 2]

    v0x = node_unit_ecef[i0, 0]
    v0y = node_unit_ecef[i0, 1]
    v0z = node_unit_ecef[i0, 2]
    v1x = node_unit_ecef[i1, 0]
    v1y = node_unit_ecef[i1, 1]
    v1z = node_unit_ecef[i1, 2]
    v2x = node_unit_ecef[i2, 0]
    v2y = node_unit_ecef[i2, 1]
    v2z = node_unit_ecef[i2, 2]

    n0x = v1y * v2z - v1z * v2y
    n0y = v1z * v2x - v1x * v2z
    n0z = v1x * v2y - v1y * v2x
    norm0 = np.sqrt(n0x * n0x + n0y * n0y + n0z * n0z)
    n0x /= norm0
    n0y /= norm0
    n0z /= norm0

    n1x = v2y * v0z - v2z * v0y
    n1y = v2z * v0x - v2x * v0z
    n1z = v2x * v0y - v2y * v0x
    norm1 = np.sqrt(n1x * n1x + n1y * n1y + n1z * n1z)
    n1x /= norm1
    n1y /= norm1
    n1z /= norm1

    n2x = v0y * v1z - v0z * v1y
    n2y = v0z * v1x - v0x * v1z
    n2z = v0x * v1y - v0y * v1x
    norm2 = np.sqrt(n2x * n2x + n2y * n2y + n2z * n2z)
    n2x /= norm2
    n2y /= norm2
    n2z /= norm2

    centre_x = face_centres[face, 0]
    centre_y = face_centres[face, 1]
    centre_z = face_centres[face, 2]
    if n0x * centre_x + n0y * centre_y + n0z * centre_z < 0.0:
        n0x = -n0x
        n0y = -n0y
        n0z = -n0z
    if n1x * centre_x + n1y * centre_y + n1z * centre_z < 0.0:
        n1x = -n1x
        n1y = -n1y
        n1z = -n1z
    if n2x * centre_x + n2y * centre_y + n2z * centre_z < 0.0:
        n2x = -n2x
        n2y = -n2y
        n2z = -n2z

    point_x = point[0]
    point_y = point[1]
    point_z = point[2]
    return (
        n0x * point_x + n0y * point_y + n0z * point_z,
        n1x * point_x + n1y * point_y + n1z * point_z,
        n2x * point_x + n2y * point_y + n2z * point_z,
    )


@njit(cache=True, inline='always')
def _walk_point_ecef_numba(
    point,
    start_face,
    triangles,
    triangle_neighbors,
    node_unit_ecef,
    face_centres,
    max_steps,
    tolerance,
):
    """Walk neighboring spherical faces for one unit ECEF point."""
    face = start_face
    previous_face = -1
    for _ in range(max_steps):
        if face < 0 or face >= triangles.shape[0]:
            return -1
        side0, side1, side2 = _face_sides_ecef_numba(
            point,
            face,
            triangles,
            node_unit_ecef,
            face_centres,
        )
        edge = 0
        minimum = side0
        if side1 < minimum:
            edge = 1
            minimum = side1
        if side2 < minimum:
            edge = 2
            minimum = side2
        if minimum >= -tolerance:
            return face
        next_face = triangle_neighbors[face, edge]
        if next_face < 0 or next_face == face or next_face == previous_face:
            return -1
        previous_face = face
        face = next_face
    return -1


@njit(cache=True, parallel=True)
def _walk_points_ecef_numba(
    points,
    start_faces,
    triangles,
    triangle_neighbors,
    node_unit_ecef,
    face_centres,
    max_steps,
    tolerance,
):
    """Walk cached spherical faces for an array of unit ECEF points."""
    result = np.empty(start_faces.shape[0], dtype=np.int64)
    for index in prange(start_faces.shape[0]):
        result[index] = _walk_point_ecef_numba(
            points[index],
            start_faces[index],
            triangles,
            triangle_neighbors,
            node_unit_ecef,
            face_centres,
            max_steps,
            tolerance,
        )
    return result


@njit(cache=True, parallel=True)
def _locate_candidates_ecef_numba(
    points,
    candidates,
    triangles,
    node_unit_ecef,
    face_centres,
    tolerance,
):
    """Select the first containing candidate face for each unit ECEF point."""
    result = np.full(points.shape[0], -1, dtype=np.int64)
    for index in prange(points.shape[0]):
        for candidate_index in range(candidates.shape[1]):
            face = candidates[index, candidate_index]
            side0, side1, side2 = _face_sides_ecef_numba(
                points[index],
                face,
                triangles,
                node_unit_ecef,
                face_centres,
            )
            if side0 >= -tolerance and side1 >= -tolerance and side2 >= -tolerance:
                result[index] = face
                break
    return result


@njit(cache=True, inline='always')
def _walk_exit_code_ecef_numba(
    point,
    start_face,
    triangles,
    triangle_neighbors,
    node_unit_ecef,
    face_centres,
    triangle_edge_class_codes,
    max_steps,
    tolerance,
):
    """Walk to the first outer edge crossed by one ECEF destination."""
    point_norm = np.sqrt(
        point[0] * point[0]
        + point[1] * point[1]
        + point[2] * point[2]
    )
    if point_norm <= 0.0:
        return BOUNDARY_CLASS_UNCLASSIFIED
    scaled_tolerance = tolerance * point_norm
    face = start_face
    previous_face = -1
    for _ in range(max_steps):
        if face < 0 or face >= triangles.shape[0]:
            return BOUNDARY_CLASS_UNCLASSIFIED
        side0, side1, side2 = _face_sides_ecef_numba(
            point,
            face,
            triangles,
            node_unit_ecef,
            face_centres,
        )
        edge = 0
        minimum = side0
        if side1 < minimum:
            edge = 1
            minimum = side1
        if side2 < minimum:
            edge = 2
            minimum = side2
        if minimum >= -scaled_tolerance:
            return BOUNDARY_CLASS_UNCLASSIFIED
        next_face = triangle_neighbors[face, edge]
        if next_face < 0:
            return triangle_edge_class_codes[face, edge]
        if next_face == face or next_face == previous_face:
            return BOUNDARY_CLASS_UNCLASSIFIED
        previous_face = face
        face = next_face
    return BOUNDARY_CLASS_UNCLASSIFIED


@njit(cache=True, parallel=True)
def _boundary_codes_for_exits_numba(
    destinations,
    start_faces,
    new_faces,
    triangles,
    triangle_neighbors,
    node_unit_ecef,
    face_centres,
    triangle_edge_class_codes,
    max_steps,
    tolerance,
):
    """Classify all exiting particle segments with compiled face walks."""
    codes = np.zeros(start_faces.shape[0], dtype=np.int8)
    for index in prange(start_faces.shape[0]):
        if start_faces[index] >= 0 and new_faces[index] < 0:
            codes[index] = _walk_exit_code_ecef_numba(
                destinations[index],
                start_faces[index],
                triangles,
                triangle_neighbors,
                node_unit_ecef,
                face_centres,
                triangle_edge_class_codes,
                max_steps,
                tolerance,
            )
    return codes
