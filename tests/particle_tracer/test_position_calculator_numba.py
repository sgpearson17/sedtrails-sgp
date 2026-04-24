import numpy as np

from sedtrails.particle_tracer.position_calculator_numba import create_grid_geometry, create_numba_particle_calculator


def square_grid():
    grid_x = np.array([0.0, 1.0, 1.0, 0.0])
    grid_y = np.array([0.0, 0.0, 1.0, 1.0])
    return grid_x, grid_y


def test_cached_geometry_reused_by_calculator():
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)

    calculator = create_numba_particle_calculator(grid_x, grid_y, grid_geometry=geometry)

    assert calculator['geometry'] is geometry
    assert calculator['triangles'].shape[1] == 3


def test_linear_field_interpolation_uses_cached_delaunay_locator():
    grid_x, grid_y = square_grid()
    calculator = create_numba_particle_calculator(grid_x, grid_y)
    field = 2.0 * grid_x + 3.0 * grid_y

    x_points = np.array([0.25, 0.75, 1.0])
    y_points = np.array([0.25, 0.75, 0.5])
    expected = 2.0 * x_points + 3.0 * y_points

    interpolated = calculator['interpolate_field'](field, x_points, y_points)

    np.testing.assert_allclose(interpolated, expected, rtol=1e-12, atol=1e-12)


def test_explicit_triangle_connectivity_is_preserved():
    grid_x, grid_y = square_grid()
    triangles = np.array([[0, 1, 2], [0, 2, 3]])

    calculator = create_numba_particle_calculator(grid_x, grid_y, triangles=triangles)

    np.testing.assert_array_equal(calculator['triangles'], triangles)


def test_rk4_update_with_constant_velocity():
    grid_x, grid_y = square_grid()
    calculator = create_numba_particle_calculator(grid_x, grid_y)
    grid_u = np.ones_like(grid_x)
    grid_v = np.full_like(grid_y, 0.5)

    x0 = np.array([0.2, 0.4])
    y0 = np.array([0.2, 0.3])
    dt = 0.1

    x_new, y_new = calculator['update_particles'](x0, y0, grid_u, grid_v, dt)

    np.testing.assert_allclose(x_new, x0 + dt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y_new, y0 + 0.5 * dt, rtol=1e-12, atol=1e-12)


def test_temporal_rk4_update_matches_preblended_grid():
    grid_x, grid_y = square_grid()
    calculator = create_numba_particle_calculator(grid_x, grid_y)
    lower_u = np.ones_like(grid_x)
    lower_v = np.full_like(grid_y, 0.5)
    upper_u = np.full_like(grid_x, 3.0)
    upper_v = np.full_like(grid_y, 1.5)
    weight = 0.25

    x0 = np.array([0.2, 0.4])
    y0 = np.array([0.2, 0.3])
    dt = 0.1

    x_temporal, y_temporal = calculator['update_particles_temporal'](
        x0,
        y0,
        lower_u,
        lower_v,
        upper_u,
        upper_v,
        weight,
        dt,
    )
    x_blended, y_blended = calculator['update_particles'](
        x0,
        y0,
        lower_u + weight * (upper_u - lower_u),
        lower_v + weight * (upper_v - lower_v),
        dt,
    )

    np.testing.assert_allclose(x_temporal, x_blended, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y_temporal, y_blended, rtol=1e-12, atol=1e-12)
