"""Geodetic particle geometry for ocean-scale spherical meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
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
BOUNDARY_CLASS_UNCLASSIFIED = np.int8(0)
BOUNDARY_CLASS_OPEN = np.int8(1)
BOUNDARY_CLASS_LAND = np.int8(2)


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
    _velocity_cache: dict[tuple, np.ndarray] = field(init=False, default_factory=dict, repr=False)

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
        result = np.full(longitude.size, -1, dtype=np.int64)

        if start_simplices is not None:
            starts = np.asarray(start_simplices, dtype=np.int64).ravel()
            if starts.shape == result.shape:
                valid = (starts >= 0) & (starts < self.triangles.shape[0])
                for index in np.flatnonzero(valid):
                    face = self._walk_to_containing_face(points[index], int(starts[index]))
                    if face >= 0:
                        result[index] = face

        missing = np.flatnonzero(result < 0)
        if missing.size == 0:
            return result
        candidate_count = min(32, self.triangles.shape[0])
        _, candidates = self._face_tree.query(points[missing], k=candidate_count)
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.ndim == 1:
            candidates = candidates[:, np.newaxis]
        for row, point_index in enumerate(missing):
            for face in candidates[row]:
                if self._point_in_face(points[point_index], int(face)):
                    result[point_index] = int(face)
                    break
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
        vectors = self._interpolate_ecef_velocity(
            self.prepare_vector_field(grid_u, grid_v),
            longitude.ravel(),
            latitude.ravel(),
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

    def prepare_vector_field(self, grid_u, grid_v) -> np.ndarray:
        """Convert one east/north node field to cached ECEF tangent vectors."""
        u = np.asarray(grid_u).ravel()
        v = np.asarray(grid_v).ravel()
        if u.size != self.grid_x.size or v.size != self.grid_x.size:
            raise ValueError('Vector field size must match the geodetic node count.')
        cache_allowed = not u.flags.writeable and not v.flags.writeable
        key = (_array_identity(u), _array_identity(v))
        if cache_allowed:
            cached = self._velocity_cache.get(key)
            if cached is not None:
                return cached
        east, north = east_north_basis(self.grid_x, self.grid_y)
        vectors = (
            u.astype(np.float64, copy=False)[:, np.newaxis] * east
            + v.astype(np.float64, copy=False)[:, np.newaxis] * north
        )
        vectors = np.ascontiguousarray(vectors)
        if cache_allowed:
            if len(self._velocity_cache) >= 8:
                self._velocity_cache.pop(next(iter(self._velocity_cache)))
            self._velocity_cache[key] = vectors
        return vectors

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
        lower = self.prepare_vector_field(lower_u, lower_v)
        upper = lower if weight <= 0.0 else self.prepare_vector_field(upper_u, upper_v)
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
        new_faces = self.locate_points(new_longitude, new_latitude, starts)
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
        r0 = lonlat_to_ecef(longitude, latitude, radius=radius)
        k1 = self._velocity_at_ecef(r0, starts, lower, upper, weight)
        r2 = _normalize_to_radius(r0 + 0.5 * dt * k1, radius)
        f2 = self.locate_points(*ecef_to_lonlat(r2), starts)
        k2 = self._velocity_at_ecef(r2, f2, lower, upper, weight)
        r3 = _normalize_to_radius(r0 + 0.5 * dt * k2, radius)
        f3 = self.locate_points(*ecef_to_lonlat(r3), f2)
        k3 = self._velocity_at_ecef(r3, f3, lower, upper, weight)
        r4 = _normalize_to_radius(r0 + dt * k3, radius)
        f4 = self.locate_points(*ecef_to_lonlat(r4), f3)
        k4 = self._velocity_at_ecef(r4, f4, lower, upper, weight)
        return _normalize_to_radius(r0 + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, radius)

    def _velocity_at_ecef(self, positions, faces, lower, upper, weight):
        longitude, latitude = ecef_to_lonlat(positions)
        lower_values = self._interpolate_ecef_velocity(lower, longitude, latitude, faces)
        if weight <= 0.0:
            values = lower_values
        else:
            upper_values = self._interpolate_ecef_velocity(upper, longitude, latitude, faces)
            values = lower_values + weight * (upper_values - lower_values)
        unit = _normalize_to_radius(positions, 1.0)
        return values - np.sum(values * unit, axis=1)[:, np.newaxis] * unit

    def _interpolate_ecef_velocity(self, field, longitude, latitude, faces):
        weights = self._weights_for_lonlat(longitude, latitude, faces)
        result = np.zeros((len(faces), 3), dtype=np.float64)
        valid = faces >= 0
        if np.any(valid):
            nodes = self.triangles[faces[valid]]
            result[valid] = np.sum(field[nodes] * weights[valid, :, np.newaxis], axis=1)
        return result

    def _weights_for_lonlat(self, longitude, latitude, faces):
        points = lonlat_to_ecef(longitude, latitude, radius=1.0)
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

    def _walk_to_containing_face(self, point, start_face):
        face = start_face
        visited = set()
        for _ in range(128):
            if face < 0 or face in visited:
                return -1
            visited.add(face)
            sides = self._face_sides(point, face)
            edge = int(np.argmin(sides))
            if sides[edge] >= -TRIANGLE_TOLERANCE:
                return face
            face = int(self.triangle_neighbors[face, edge])
        return -1

    def _point_in_face(self, point, face):
        return bool(np.all(self._face_sides(point, face) >= -TRIANGLE_TOLERANCE))

    def _face_sides(self, point, face):
        vertices = self.node_unit_ecef[self.triangles[face]]
        edge_normals = np.cross(
            vertices[[1, 2, 0]],
            vertices[[2, 0, 1]],
        )
        signs = edge_normals @ self.face_centres[face]
        edge_normals *= np.where(signs < 0.0, -1.0, 1.0)[:, np.newaxis]
        return edge_normals @ point

    def _boundary_codes_for_exits(self, destinations, starts, new_faces):
        codes = np.zeros(len(starts), dtype=np.int8)
        exits = (starts >= 0) & (new_faces < 0)
        for index in np.flatnonzero(exits):
            point = destinations[index] / np.linalg.norm(destinations[index])
            codes[index] = self._walk_exit_code(point, int(starts[index]))
        return codes

    def _walk_exit_code(self, point, start_face):
        """Walk toward a destination and return the crossed outer-edge class."""
        face = int(start_face)
        visited = set()
        for _ in range(128):
            if face < 0 or face in visited:
                return BOUNDARY_CLASS_UNCLASSIFIED
            visited.add(face)
            sides = self._face_sides(point, face)
            edge = int(np.argmin(sides))
            if sides[edge] >= -TRIANGLE_TOLERANCE:
                return BOUNDARY_CLASS_UNCLASSIFIED
            neighbor = int(self.triangle_neighbors[face, edge])
            if neighbor < 0:
                return self.triangle_edge_class_codes[face, edge]
            face = neighbor
        return BOUNDARY_CLASS_UNCLASSIFIED

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


def _circular_mean_longitude(longitude):
    radians = np.deg2rad(longitude)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))))


def _array_identity(array):
    interface = array.__array_interface__
    return (
        int(interface['data'][0]),
        array.shape,
        array.strides,
        array.dtype.str,
    )
