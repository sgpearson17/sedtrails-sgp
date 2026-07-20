"""Integration tests for ocean-scale geodetic grid geometry."""

import numpy as np
import pytest

from sedtrails.particle_tracer.geodetic_geometry import spherical_distance
from sedtrails.particle_tracer.position_calculator_numba import create_grid_geometry


def _dateline_geometry(boundary_edge_classification=None, longitude_wrap='auto'):
    longitude = np.array([179.0, -179.0, 180.0])
    latitude = np.array([-1.0, -1.0, 2.0])
    return create_grid_geometry(
        longitude,
        latitude,
        triangles=np.array([[0, 1, 2]], dtype=np.int64),
        boundary_edge_classification=boundary_edge_classification,
        coordinate_system='geographic',
        runtime_geometry='geodetic',
        surface_model='sphere',
        longitude_wrap=longitude_wrap,
        velocity_basis='east_north',
    )


def test_geodetic_grid_locates_and_interpolates_across_antimeridian():
    """Face location and scalar interpolation should have no longitude seam."""
    geometry = _dateline_geometry()

    faces = geometry.locate_points(
        np.array([179.5, -179.5, 170.0]),
        np.array([-0.5, -0.5, 0.0]),
    )
    values = geometry.interpolate_field(
        np.array([7.0, 7.0, 7.0]),
        np.array([179.5, -179.5]),
        np.array([-0.5, -0.5]),
    )

    np.testing.assert_array_equal(faces, [0, 0, -1])
    np.testing.assert_allclose(values, 7.0)


def test_geodetic_grid_moves_in_metres_across_antimeridian():
    """An eastward node field should move particles by physical metres."""
    geometry = _dateline_geometry()
    start_longitude = np.array([179.9])
    start_latitude = np.array([0.0])

    longitude, latitude, faces = geometry.update_particles_with_simplex(
        start_longitude,
        start_latitude,
        np.ones(3),
        np.zeros(3),
        20_000.0,
        simplex_ids=np.array([0]),
    )

    assert longitude[0] < -179.0
    assert faces[0] == 0
    np.testing.assert_allclose(
        spherical_distance(
            start_longitude,
            start_latitude,
            longitude,
            latitude,
        ),
        20_000.0,
        rtol=2.0e-4,
        atol=0.1,
    )


def test_geodetic_grid_reports_topological_open_boundary():
    """The first crossed classified edge should define the exit class."""
    geometry = _dateline_geometry(
        {
            'edge_nodes': [[0, 1], [1, 2], [0, 2]],
            'edge_classes': ['land', 'open', 'land'],
        }
    )

    _, _, faces, boundary_codes = geometry.update_particles_with_boundary_class(
        np.array([-179.5]),
        np.array([-0.5]),
        np.full(3, 200.0),
        np.zeros(3),
        20_000.0,
        simplex_ids=np.array([0]),
    )

    assert faces[0] == -1
    assert boundary_codes[0] in (1, 2)


def test_geodetic_grid_vector_cache_is_shared_for_identical_slices():
    """One immutable vector slice should be converted only once."""
    geometry = _dateline_geometry()
    u = np.ones(3)
    v = np.zeros(3)
    u.flags.writeable = False
    v.flags.writeable = False

    first = geometry.prepare_vector_field(u, v)
    second = geometry.prepare_vector_field(u, v)

    assert first is second
    assert len(geometry._velocity_cache) == 1


def test_geodetic_grid_recomputes_reused_writable_vector_buffers():
    """A mutable forcing buffer must not return stale cached velocities."""
    geometry = _dateline_geometry()
    u = np.ones(3)
    v = np.zeros(3)

    first = geometry.prepare_vector_field(u, v)
    u[:] = 2.0
    second = geometry.prepare_vector_field(u, v)

    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(second, axis=1), 2.0)
    assert len(geometry._velocity_cache) == 0


def test_geodetic_grid_applies_zero_to_360_longitude_wrap():
    """Movement across the antimeridian preserves configured public wrapping."""
    geometry = _dateline_geometry(longitude_wrap='0_360')

    moved_x, moved_y = geometry.apply_diffusion(
        np.array([179.9]),
        np.array([0.2]),
        np.array([20_000.0]),
        np.array([0.0]),
    )

    assert 180.0 < moved_x[0] < 181.0
    assert moved_y[0] == pytest.approx(0.2, abs=1.0e-3)
