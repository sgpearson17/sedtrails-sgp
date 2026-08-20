import numpy as np
import pytest

from sedtrails.particle_tracer.geodetic_geometry import (
    EARTH_MEAN_RADIUS_M,
    east_north_basis,
    ecef_to_lonlat,
    lonlat_to_ecef,
    move_lonlat,
    normalize_longitude,
    spherical_distance,
    spherical_exponential_map,
    spherical_triangle_contains,
    spherical_triangle_edge_normals,
)


def test_lonlat_ecef_roundtrip_is_vectorized_and_wraps_longitude():
    """ECEF conversion should round-trip broadcast coordinates without a seam."""
    longitude = np.array([0.0, 180.0, 181.0, 540.0])
    latitude = np.array([[0.0], [45.0], [-60.0]])

    ecef = lonlat_to_ecef(longitude, latitude)
    result_longitude, result_latitude = ecef_to_lonlat(ecef)

    assert ecef.shape == (3, 4, 3)
    assert ecef.dtype == np.float64
    expected_longitude = np.broadcast_to(normalize_longitude(longitude), (3, 4))
    expected_latitude = np.broadcast_to(latitude, (3, 4))
    np.testing.assert_allclose(result_longitude, expected_longitude, atol=1.0e-12)
    np.testing.assert_allclose(result_latitude, expected_latitude, atol=1.0e-12)
    np.testing.assert_allclose(
        np.linalg.norm(ecef, axis=-1),
        EARTH_MEAN_RADIUS_M,
        rtol=0.0,
        atol=1.0e-9,
    )


@pytest.mark.parametrize('latitude', [-90.0, -45.0, 0.0, 45.0, 90.0])
def test_east_north_basis_is_orthonormal_and_right_handed(latitude):
    """The tangent basis should remain valid at ordinary points and poles."""
    longitude = np.array([-180.0, -30.0, 0.0, 127.0])
    latitudes = np.full(longitude.shape, latitude)

    east, north = east_north_basis(longitude, latitudes)
    up = lonlat_to_ecef(longitude, latitudes, radius=1.0)

    np.testing.assert_allclose(np.linalg.norm(east, axis=-1), 1.0, atol=1.0e-15)
    np.testing.assert_allclose(np.linalg.norm(north, axis=-1), 1.0, atol=1.0e-15)
    np.testing.assert_allclose(np.sum(east * north, axis=-1), 0.0, atol=1.0e-15)
    np.testing.assert_allclose(np.cross(east, north), up, atol=1.0e-15)


def test_spherical_exponential_map_follows_exact_quarter_great_circle():
    """A quarter-circumference tangent move should reach the expected axis."""
    radius = 10.0
    position = np.array([radius, 0.0, 0.0])
    displacement = np.array([0.0, 0.5 * np.pi * radius, 0.0])

    destination = spherical_exponential_map(position, displacement, radius=radius)

    np.testing.assert_allclose(destination, [0.0, radius, 0.0], atol=1.0e-14)
    np.testing.assert_allclose(np.linalg.norm(destination), radius, atol=1.0e-14)


def test_move_lonlat_uses_metres_and_crosses_antimeridian():
    """Eastward movement should cross the longitude seam without discontinuity."""
    distance_m = 30_000.0

    longitude, latitude = move_lonlat(179.9, 0.0, distance_m, 0.0)

    assert -180.0 <= longitude < -179.0
    np.testing.assert_allclose(latitude, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        spherical_distance(179.9, 0.0, longitude, latitude),
        distance_m,
        rtol=0.0,
        atol=1.0e-8,
    )


def test_move_lonlat_reaches_pole_by_exact_meridional_distance():
    """Northward exponential movement should remain finite at the pole."""
    distance_m = 0.5 * np.pi * EARTH_MEAN_RADIUS_M

    longitude, latitude = move_lonlat(37.0, 0.0, 0.0, distance_m)

    assert np.isfinite(longitude)
    np.testing.assert_allclose(latitude, 90.0, atol=1.0e-12)


def test_spherical_distance_handles_seam_poles_and_antipodes():
    """Distance should be correct at the main longitude singularities."""
    radius = 2.0

    seam = spherical_distance(179.9, 0.0, -179.9, 0.0, radius=radius)
    polar = spherical_distance(0.0, 89.0, 180.0, 89.0, radius=radius)
    antipodal = spherical_distance(0.0, 0.0, 180.0, 0.0, radius=radius)

    np.testing.assert_allclose(seam, radius * np.deg2rad(0.2), atol=1.0e-15)
    np.testing.assert_allclose(polar, radius * np.deg2rad(2.0), atol=1.0e-15)
    np.testing.assert_allclose(antipodal, radius * np.pi, atol=1.0e-15)


def test_spherical_triangle_contains_points_across_antimeridian():
    """Triangle containment should not see a seam at longitude 180 degrees."""
    triangle = lonlat_to_ecef(
        np.array([179.0, -179.0, 180.0]),
        np.array([-1.0, -1.0, 2.0]),
        radius=1.0,
    )
    points = lonlat_to_ecef(
        np.array([180.0, 179.5, -179.5, 170.0]),
        np.array([0.0, -0.5, -0.5, 0.0]),
        radius=1.0,
    )

    result = spherical_triangle_contains(points, triangle)

    np.testing.assert_array_equal(result, [True, True, True, False])


def test_spherical_triangle_contains_north_pole_and_boundary():
    """A polar triangle should include its pole, edges, and vertices."""
    triangle = lonlat_to_ecef(
        np.array([0.0, 120.0, -120.0]),
        np.array([80.0, 80.0, 80.0]),
        radius=1.0,
    )
    edge_midpoint = triangle[0] + triangle[1]
    points = np.vstack(
        (
            lonlat_to_ecef(0.0, 90.0, radius=1.0),
            triangle[0],
            edge_midpoint,
            lonlat_to_ecef(0.0, 0.0, radius=1.0),
        )
    )

    result = spherical_triangle_contains(points, triangle)
    normals = spherical_triangle_edge_normals(triangle)

    np.testing.assert_array_equal(result, [True, True, True, False])
    assert normals.flags.c_contiguous
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-15)


@pytest.mark.parametrize(
    ('call', 'message'),
    [
        (lambda: lonlat_to_ecef(0.0, 91.0), 'latitude'),
        (lambda: lonlat_to_ecef(0.0, 0.0, radius=0.0), 'radius'),
        (lambda: ecef_to_lonlat([0.0, 0.0, 0.0]), 'zero-length'),
        (
            lambda: spherical_exponential_map(
                [EARTH_MEAN_RADIUS_M, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ),
            'tangent',
        ),
        (
            lambda: spherical_triangle_edge_normals(
                lonlat_to_ecef([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], radius=1.0)
            ),
            'degenerate',
        ),
    ],
)
def test_invalid_geometry_inputs_fail_explicitly(call, message):
    """Invalid geometry should fail before producing misleading coordinates."""
    with pytest.raises(ValueError, match=message):
        call()
