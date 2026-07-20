"""Projection-free geometry on a spherical Earth.

The functions in this module use geocentric Cartesian coordinates for
topology and movement. Longitude wrapping is therefore limited to input and
output, and does not create a seam in geometric operations.

Cartesian arrays use ``(..., 3)`` shape with the component axis last. Public
functions return ``float64`` NumPy arrays so their results can be passed
directly to Numba kernels without Python objects.
"""

from __future__ import annotations

import numpy as np

EARTH_MEAN_RADIUS_M = 6_371_008.8
"""IUGG mean Earth radius in metres."""

_DEGREES_TO_RADIANS = np.pi / 180.0
_RADIANS_TO_DEGREES = 180.0 / np.pi
_MIN_VECTOR_NORM = 1.0e-15


def normalize_longitude(longitude) -> np.ndarray:
    """Wrap longitude to the half-open interval ``[-180, 180)``.

    Parameters
    ----------
    longitude : array-like
        Longitude in degrees.

    Returns
    -------
    numpy.ndarray
        Wrapped ``float64`` longitude with the input shape.

    Raises
    ------
    ValueError
        If any longitude is not finite.
    """
    longitude_array = np.asarray(longitude, dtype=np.float64)
    if not np.all(np.isfinite(longitude_array)):
        raise ValueError('longitude must contain only finite values')
    return (longitude_array + 180.0) % 360.0 - 180.0


def lonlat_to_ecef(
    longitude,
    latitude,
    radius: float = EARTH_MEAN_RADIUS_M,
) -> np.ndarray:
    """Convert spherical longitude/latitude to ECEF coordinates.

    Parameters
    ----------
    longitude, latitude : array-like
        Longitude and latitude in degrees. Inputs must be broadcastable to a
        common shape. Longitude may use any wrapping convention.
    radius : float, default=EARTH_MEAN_RADIUS_M
        Sphere radius in metres.

    Returns
    -------
    numpy.ndarray
        ECEF coordinates in metres with shape ``broadcast_shape + (3,)``.

    Raises
    ------
    ValueError
        If coordinates are not finite, latitude is outside ``[-90, 90]``, or
        ``radius`` is not positive and finite.
    """
    radius_value = _validate_radius(radius)
    lon_array, lat_array = _broadcast_lonlat(longitude, latitude)
    lon_radians = lon_array * _DEGREES_TO_RADIANS
    lat_radians = lat_array * _DEGREES_TO_RADIANS

    cos_latitude = np.cos(lat_radians)
    result = np.empty(lon_array.shape + (3,), dtype=np.float64)
    result[..., 0] = radius_value * cos_latitude * np.cos(lon_radians)
    result[..., 1] = radius_value * cos_latitude * np.sin(lon_radians)
    result[..., 2] = radius_value * np.sin(lat_radians)
    return result


def ecef_to_lonlat(ecef) -> tuple[np.ndarray, np.ndarray]:
    """Convert nonzero ECEF vectors to spherical longitude/latitude.

    The input vectors need not have a particular radius. Longitude is wrapped
    to ``[-180, 180)``. At either exact pole, longitude follows ``atan2(y, x)``
    and is therefore a deterministic coordinate convention rather than a
    physical direction.

    Parameters
    ----------
    ecef : array-like
        Cartesian coordinates with shape ``(..., 3)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Longitude and latitude arrays in degrees with shape ``ecef.shape[:-1]``.

    Raises
    ------
    ValueError
        If the input has the wrong shape, contains non-finite values, or
        contains a zero-length vector.
    """
    vectors, norms = _normalized_vectors(ecef, name='ecef')
    longitude = np.arctan2(vectors[..., 1], vectors[..., 0]) * _RADIANS_TO_DEGREES
    latitude = np.arctan2(
        vectors[..., 2],
        np.hypot(vectors[..., 0], vectors[..., 1]),
    ) * _RADIANS_TO_DEGREES
    del norms
    return normalize_longitude(longitude), latitude


def east_north_basis(longitude, latitude) -> tuple[np.ndarray, np.ndarray]:
    """Return orthonormal east and north unit vectors in ECEF coordinates.

    Parameters
    ----------
    longitude, latitude : array-like
        Longitude and latitude in degrees. Inputs must be broadcastable to a
        common shape.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        East and north arrays with shape ``broadcast_shape + (3,)``.

    Notes
    -----
    Geographic east is undefined at an exact pole. This function uses the
    supplied longitude to select a deterministic limiting basis there. The
    basis remains finite, orthonormal, and right-handed at both poles.
    """
    lon_array, lat_array = _broadcast_lonlat(longitude, latitude)
    lon_radians = lon_array * _DEGREES_TO_RADIANS
    lat_radians = lat_array * _DEGREES_TO_RADIANS

    sin_longitude = np.sin(lon_radians)
    cos_longitude = np.cos(lon_radians)
    sin_latitude = np.sin(lat_radians)
    cos_latitude = np.cos(lat_radians)

    east = np.empty(lon_array.shape + (3,), dtype=np.float64)
    east[..., 0] = -sin_longitude
    east[..., 1] = cos_longitude
    east[..., 2] = 0.0

    north = np.empty(lon_array.shape + (3,), dtype=np.float64)
    north[..., 0] = -sin_latitude * cos_longitude
    north[..., 1] = -sin_latitude * sin_longitude
    north[..., 2] = cos_latitude
    return east, north


def spherical_exponential_map(
    position_ecef,
    tangent_displacement_m,
    radius: float = EARTH_MEAN_RADIUS_M,
) -> np.ndarray:
    """Move ECEF positions by the exact exponential map on a sphere.

    Parameters
    ----------
    position_ecef : array-like
        Starting ECEF positions with shape ``(..., 3)``. Inputs must lie on
        the sphere to normal floating-point tolerance.
    tangent_displacement_m : array-like
        Tangent displacements in metres with shape broadcastable to
        ``position_ecef``.
    radius : float, default=EARTH_MEAN_RADIUS_M
        Sphere radius in metres.

    Returns
    -------
    numpy.ndarray
        Moved ECEF positions in metres with the broadcast shape.

    Raises
    ------
    ValueError
        If inputs are invalid, positions are not on the requested sphere, or
        displacement vectors have a material radial component.

    Notes
    -----
    This operation follows the oriented great circle for the requested
    displacement. Distances greater than half a circumference are supported
    and are not reduced to the shortest path.
    """
    radius_value = _validate_radius(radius)
    position = _as_cartesian_vectors(position_ecef, name='position_ecef')
    displacement = _as_cartesian_vectors(
        tangent_displacement_m,
        name='tangent_displacement_m',
    )
    try:
        position, displacement = np.broadcast_arrays(position, displacement)
    except ValueError as exc:
        raise ValueError(
            'position_ecef and tangent_displacement_m must be broadcastable'
        ) from exc

    position_norm = np.linalg.norm(position, axis=-1)
    sphere_tolerance = max(1.0e-8 * radius_value, 1.0e-6)
    if np.any(np.abs(position_norm - radius_value) > sphere_tolerance):
        raise ValueError('position_ecef must lie on the requested sphere')

    unit_position = position / position_norm[..., np.newaxis]
    radial_component = np.sum(displacement * unit_position, axis=-1)
    displacement_norm = np.linalg.norm(displacement, axis=-1)
    tangent_tolerance = 1.0e-12 * np.maximum(displacement_norm, 1.0)
    if np.any(np.abs(radial_component) > tangent_tolerance):
        raise ValueError('tangent_displacement_m must be tangent to the sphere')

    tangent = displacement - radial_component[..., np.newaxis] * unit_position
    arc_length = np.linalg.norm(tangent, axis=-1)
    unit_tangent = np.zeros_like(tangent)
    np.divide(
        tangent,
        arc_length[..., np.newaxis],
        out=unit_tangent,
        where=arc_length[..., np.newaxis] > 0.0,
    )

    angle = arc_length / radius_value
    result = radius_value * (
        np.cos(angle)[..., np.newaxis] * unit_position
        + np.sin(angle)[..., np.newaxis] * unit_tangent
    )
    zero_displacement = arc_length == 0.0
    return np.where(zero_displacement[..., np.newaxis], position, result)


def move_lonlat(
    longitude,
    latitude,
    east_displacement_m,
    north_displacement_m,
    radius: float = EARTH_MEAN_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Move longitude/latitude by east and north tangent displacements.

    Parameters
    ----------
    longitude, latitude : array-like
        Starting longitude and latitude in degrees.
    east_displacement_m, north_displacement_m : array-like
        Local tangent-plane displacements in metres. All coordinate and
        displacement inputs must broadcast to a common shape.
    radius : float, default=EARTH_MEAN_RADIUS_M
        Sphere radius in metres.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Destination longitude and latitude in degrees. Longitude is wrapped to
        ``[-180, 180)``.
    """
    lon_array, lat_array, east_m, north_m = np.broadcast_arrays(
        np.asarray(longitude, dtype=np.float64),
        np.asarray(latitude, dtype=np.float64),
        np.asarray(east_displacement_m, dtype=np.float64),
        np.asarray(north_displacement_m, dtype=np.float64),
    )
    _validate_lonlat_values(lon_array, lat_array)
    if not np.all(np.isfinite(east_m)) or not np.all(np.isfinite(north_m)):
        raise ValueError('displacements must contain only finite values')

    position = lonlat_to_ecef(lon_array, lat_array, radius=radius)
    east, north = east_north_basis(lon_array, lat_array)
    displacement = east_m[..., np.newaxis] * east + north_m[..., np.newaxis] * north
    destination = spherical_exponential_map(position, displacement, radius=radius)
    return ecef_to_lonlat(destination)


def spherical_distance(
    longitude_a,
    latitude_a,
    longitude_b,
    latitude_b,
    radius: float = EARTH_MEAN_RADIUS_M,
) -> np.ndarray:
    """Return shortest great-circle distance in metres.

    The ECEF ``atan2(cross, dot)`` formulation is stable for small distances,
    antimeridian crossings, polar points, and antipodal points.

    Parameters
    ----------
    longitude_a, latitude_a, longitude_b, latitude_b : array-like
        Endpoint coordinates in degrees. Inputs must broadcast to a common
        shape.
    radius : float, default=EARTH_MEAN_RADIUS_M
        Sphere radius in metres.

    Returns
    -------
    numpy.ndarray
        Shortest surface distance in metres with the broadcast input shape.
    """
    radius_value = _validate_radius(radius)
    lon_a, lat_a, lon_b, lat_b = np.broadcast_arrays(
        np.asarray(longitude_a, dtype=np.float64),
        np.asarray(latitude_a, dtype=np.float64),
        np.asarray(longitude_b, dtype=np.float64),
        np.asarray(latitude_b, dtype=np.float64),
    )
    _validate_lonlat_values(lon_a, lat_a)
    _validate_lonlat_values(lon_b, lat_b)

    point_a = lonlat_to_ecef(lon_a, lat_a, radius=1.0)
    point_b = lonlat_to_ecef(lon_b, lat_b, radius=1.0)
    cross_norm = np.linalg.norm(np.cross(point_a, point_b), axis=-1)
    dot_product = np.sum(point_a * point_b, axis=-1)
    angle = np.arctan2(cross_norm, np.clip(dot_product, -1.0, 1.0))
    return radius_value * angle


def spherical_triangle_edge_normals(triangle_ecef) -> np.ndarray:
    """Return inward unit normals for a convex spherical triangle.

    Parameters
    ----------
    triangle_ecef : array-like
        Three nonzero ECEF vertex vectors with shape ``(3, 3)``. Triangle
        edges are the minor great-circle arcs between consecutive vertices.

    Returns
    -------
    numpy.ndarray
        Inward unit great-circle normals with shape ``(3, 3)``. Row ``i``
        corresponds to the edge from vertex ``i`` to vertex ``(i + 1) % 3``.

    Raises
    ------
    ValueError
        If the triangle is degenerate, contains an antipodal edge, or does not
        define an unambiguous convex region smaller than a hemisphere.
    """
    triangle = _as_cartesian_vectors(triangle_ecef, name='triangle_ecef')
    if triangle.shape != (3, 3):
        raise ValueError(f'triangle_ecef must have shape (3, 3), got {triangle.shape}')
    vertices, _ = _normalized_vectors(triangle, name='triangle_ecef')

    reference = np.sum(vertices, axis=0)
    reference_norm = np.linalg.norm(reference)
    if reference_norm <= _MIN_VECTOR_NORM:
        raise ValueError('triangle does not define an unambiguous convex region')
    reference /= reference_norm

    edge_normals = np.cross(vertices, np.roll(vertices, -1, axis=0))
    edge_norms = np.linalg.norm(edge_normals, axis=1)
    if np.any(edge_norms <= _MIN_VECTOR_NORM):
        raise ValueError('triangle has a degenerate or antipodal edge')
    edge_normals /= edge_norms[:, np.newaxis]

    reference_side = edge_normals @ reference
    if np.any(np.abs(reference_side) <= _MIN_VECTOR_NORM):
        raise ValueError('triangle is degenerate or spans a hemisphere')
    edge_normals *= np.where(reference_side < 0.0, -1.0, 1.0)[:, np.newaxis]

    vertex_side = np.sum(edge_normals * np.roll(vertices, 1, axis=0), axis=1)
    if np.any(vertex_side <= _MIN_VECTOR_NORM):
        raise ValueError('triangle does not define a convex minor-arc region')
    return np.ascontiguousarray(edge_normals, dtype=np.float64)


def spherical_triangle_contains(
    points_ecef,
    triangle_ecef,
    *,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Test whether ECEF points lie in a convex spherical triangle.

    Parameters
    ----------
    points_ecef : array-like
        One point with shape ``(3,)`` or points with shape ``(..., 3)``.
        Radial distance is ignored.
    triangle_ecef : array-like
        Triangle vertices with shape ``(3, 3)``.
    tolerance : float, default=1.0e-12
        Nonnegative dimensionless tolerance for inward half-space tests.

    Returns
    -------
    numpy.ndarray
        Boolean result with shape ``points_ecef.shape[:-1]``. Boundary points
        are included.

    Raises
    ------
    ValueError
        If inputs are invalid or ``tolerance`` is negative or non-finite.
    """
    tolerance_value = float(tolerance)
    if not np.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError('tolerance must be nonnegative and finite')
    points, _ = _normalized_vectors(points_ecef, name='points_ecef')
    edge_normals = spherical_triangle_edge_normals(triangle_ecef)
    signed_sides = points @ edge_normals.T
    return np.all(signed_sides >= -tolerance_value, axis=-1)


def _broadcast_lonlat(longitude, latitude) -> tuple[np.ndarray, np.ndarray]:
    """Return validated longitude and latitude arrays with a common shape."""
    lon_array, lat_array = np.broadcast_arrays(
        np.asarray(longitude, dtype=np.float64),
        np.asarray(latitude, dtype=np.float64),
    )
    _validate_lonlat_values(lon_array, lat_array)
    return lon_array, lat_array


def _validate_lonlat_values(longitude: np.ndarray, latitude: np.ndarray) -> None:
    """Validate finite longitude and bounded latitude arrays."""
    if not np.all(np.isfinite(longitude)) or not np.all(np.isfinite(latitude)):
        raise ValueError('longitude and latitude must contain only finite values')
    if np.any(latitude < -90.0) or np.any(latitude > 90.0):
        raise ValueError('latitude must be within [-90, 90] degrees')


def _validate_radius(radius: float) -> float:
    """Return a positive finite scalar radius."""
    radius_value = float(radius)
    if not np.isfinite(radius_value) or radius_value <= 0.0:
        raise ValueError('radius must be positive and finite')
    return radius_value


def _as_cartesian_vectors(values, *, name: str) -> np.ndarray:
    """Return finite Cartesian vectors as a float64 array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f'{name} must have shape (..., 3), got {array.shape}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain only finite values')
    return array


def _normalized_vectors(values, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return unit vectors and original vector norms."""
    vectors = _as_cartesian_vectors(values, name=name)
    norms = np.linalg.norm(vectors, axis=-1)
    if np.any(norms <= _MIN_VECTOR_NORM):
        raise ValueError(f'{name} must not contain zero-length vectors')
    return vectors / norms[..., np.newaxis], norms
