import numpy as np
import pytest
from sedtrails.interpolate.particlepositioncalculator import ParticlePositionCalculator

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def simple_interpolator():
    """
    Create a simple ParticlePositionCalculator instance based on a single triangle
    with a constant velocity field.
    """
    # Define a simple nondegenerate triangle.
    grid_x = np.array([0.0, 1.0, 0.0])
    grid_y = np.array([0.0, 0.0, 1.0])
    # Constant velocity field.
    grid_u = np.array([1.0, 1.0, 1.0])
    grid_v = np.array([0.0, 0.0, 0.0])

    return ParticlePositionCalculator(grid_x, grid_y, grid_u, grid_v, igeo=0)

@pytest.fixture
def mixed_network_interpolator():
    # Define grid nodes.
    grid_x = np.array([0, 5, 2,
                       0, 5, 2,
                       0, 5, 2,
                       0, 5, 2], dtype=np.float64)
    grid_y = np.array([0, 0, 0,
                       1, 1, 1,
                       2, 2, 2,
                       3, 3, 3], dtype=np.float64)
       
    # Define constant velocity fields on grid nodes.
    grid_u = np.full_like(grid_x, 1.0, dtype=np.float64)   # constant u = 1.0
    grid_v = np.full_like(grid_y, 0.5, dtype=np.float64)     # constant v = 0.5

    return ParticlePositionCalculator(grid_x, grid_y, grid_u, grid_v, igeo=0)


# -----------------------------------------------------------------------------
# Tests for field interpolation
# -----------------------------------------------------------------------------

def test_interpolate_field_constant(simple_interpolator):
    """
    When the grid field is constant, the interpolated value at any particle
    position inside the triangle should equal that constant.
    """
    constant_field = np.full_like(simple_interpolator.grid_x, 5.0)
    part_x = np.array([0.2, 0.4, 0.1])
    part_y = np.array([0.2, 0.1, 0.4])
    interp_vals = simple_interpolator.interpolate_field(constant_field, part_x, part_y)
    np.testing.assert_allclose(interp_vals, 5.0)


# -----------------------------------------------------------------------------
# Tests for particle update without diffusion
# -----------------------------------------------------------------------------

def test_update_particles_no_velocity(simple_interpolator):
    """
    If grid velocities are zero, the particle positions should remain unchanged.
    """
    simple_interpolator.grid_u = np.zeros_like(simple_interpolator.grid_u)
    simple_interpolator.grid_v = np.zeros_like(simple_interpolator.grid_v)
    part_x = np.array([0.2, 0.4, 0.1])
    part_y = np.array([0.2, 0.1, 0.4])
    dt = 0.1
    rndfac = 0.0  # disable diffusion
    x_new, y_new, xdiff, ydiff = simple_interpolator.update_particles(
        part_x, part_y, dt, rndfac, parallel=False)
    np.testing.assert_allclose(x_new, part_x)
    np.testing.assert_allclose(y_new, part_y)
    np.testing.assert_allclose(xdiff, 0.0)
    np.testing.assert_allclose(ydiff, 0.0)


# -----------------------------------------------------------------------------
# Tests for particle update with diffusion
# -----------------------------------------------------------------------------

def test_update_particles_with_diffusion(simple_interpolator):
    """
    When a nonzero diffusion factor is used, diffusion increments should be
    added to the updated particle positions.
    """
    part_x = np.array([0.2, 0.4, 0.1])
    part_y = np.array([0.2, 0.1, 0.4])
    dt = 0.1
    rndfac = 0.5
    np.random.seed(42)
    x_new, y_new, xdiff, ydiff = simple_interpolator.update_particles(
        part_x, part_y, dt, rndfac, parallel=False)
    assert np.any(np.abs(xdiff) > 0)
    assert np.any(np.abs(ydiff) > 0)


# -----------------------------------------------------------------------------
# Tests for consistency between parallel and serial execution
# -----------------------------------------------------------------------------

def test_parallel_vs_serial(simple_interpolator):
    """
    With diffusion disabled, the serial and parallel versions should produce
    nearly identical particle updates.
    """
    part_x = np.linspace(0.1, 0.9, 50)
    part_y = np.linspace(0.1, 0.9, 50)
    dt = 0.1
    rndfac = 0.0
    x_serial, y_serial, _, _ = simple_interpolator.update_particles(
        part_x, part_y, dt, rndfac, parallel=False)
    x_parallel, y_parallel, _, _ = simple_interpolator.update_particles(
        part_x, part_y, dt, rndfac, parallel=True, num_workers=2)
    np.testing.assert_allclose(x_serial, x_parallel)
    np.testing.assert_allclose(y_serial, y_parallel)


# -----------------------------------------------------------------------------
# Test for geographic adjustment (igeo==1)
# -----------------------------------------------------------------------------

def test_geographic_adjustment():
    """
    For geographic coordinates (igeo==1), grid_u values should be scaled by
    1/(cos(latitude)*geofac) and grid_v by 1/geofac. For a constant field,
    the RK4 update should yield x_new = x0 + dt * adjusted_u.
    """
    # Define a nondegenerate triangle with constant latitude.
    grid_x = np.array([0.0, 1.0, 0.0],dtype=np.float64)
    grid_y = np.array([45.0, 45.0, 46.0],dtype=np.float64)
    grid_u = np.array([1.0, 1.0, 1.0],dtype=np.float64)
    grid_v = np.array([0.0, 0.0, 0.0],dtype=np.float64)
    
    interpolator = ParticlePositionCalculator(grid_x, grid_y, grid_u, grid_v, igeo=1)
    dt = 1.0
    rndfac = 0.0
    part_x = np.array([0.2],dtype=np.float64)
    part_y = np.array([0.2],dtype=np.float64)
    expected_u = 1.0 / (np.cos(np.deg2rad(45.0)) * interpolator.geofac)
    expected_x_new = part_x + dt * expected_u
    x_new, y_new, _, _ = interpolator.update_particles(
        part_x, part_y, dt, rndfac, parallel=False)
    np.testing.assert_allclose(x_new, expected_x_new, rtol=1e-5)
    np.testing.assert_allclose(y_new, part_y, rtol=1e-5)


# -----------------------------------------------------------------------------
# Mixed network interpolation tests
# -----------------------------------------------------------------------------
def test_mixed_network_linear_interpolation(mixed_network_interpolator):
    """
    Verify that when the grid field is a linear function f(x,y) = 2*x + 3*y,
    barycentric interpolation recovers the exact value.
    """
    linear_func = lambda x, y: 2 * x + 3 * y
    field = linear_func(mixed_network_interpolator.grid_x,
                        mixed_network_interpolator.grid_y)
    
    # Generate particle positions strictly within the convex hull.
    np.random.seed(42)
    part_x = np.random.uniform(0., 1., 20)
    part_y = np.random.uniform(0., 3., 20)
    
    interp_vals = mixed_network_interpolator.interpolate_field(field, part_x, part_y,
                                                               parallel=False)
    expected_vals = linear_func(part_x, part_y)
    np.testing.assert_allclose(interp_vals, expected_vals, rtol=1e-5)

def test_mixed_network_update_particles(mixed_network_interpolator):
    """
    Test that updating particles with constant grid velocities produces a simple shift.
    With constant u = 1.0 and v = 0.5, the RK4 integration should yield:
         x_new = x0 + dt * 1.0   and   y_new = y0 + dt * 0.5.
    """
    np.random.seed(42)
    # Use particles strictly inside the domain to avoid boundary issues.
    part_x = np.random.uniform(0.1, 1.9, 2)
    part_y = np.random.uniform(0.1, 2.9, 2)
    dt = 0.2
    rndfac = 0.0

    expected_x = part_x + dt * 1.0
    expected_y = part_y + dt * 0.5

    x_new, y_new, xdiff, ydiff = mixed_network_interpolator.update_particles(
        part_x, part_y, dt, rndfac, parallel=False)

    np.testing.assert_allclose(x_new, expected_x, rtol=1e-5)
    np.testing.assert_allclose(y_new, expected_y, rtol=1e-5)
    np.testing.assert_allclose(xdiff, 0.0, atol=1e-8)
    np.testing.assert_allclose(ydiff, 0.0, atol=1e-8)

def test_mixed_network_parallel_interpolation(mixed_network_interpolator):
    """
    Ensure that interpolation on the mixed network is consistent between serial and
    parallel execution.
    """
    linear_func = lambda x, y: 2 * x + 3 * y
    field = linear_func(mixed_network_interpolator.grid_x,
                        mixed_network_interpolator.grid_y)
    
    np.random.seed(42)
    part_x = np.random.uniform(0.1, 1.9, 25)
    part_y = np.random.uniform(0.1, 2.9, 25)
    
    interp_serial = mixed_network_interpolator.interpolate_field(
        field, part_x, part_y, parallel=False)
    interp_parallel = mixed_network_interpolator.interpolate_field(
        field, part_x, part_y, parallel=True, num_workers=2)
    
    np.testing.assert_allclose(interp_serial, interp_parallel, rtol=1e-5)