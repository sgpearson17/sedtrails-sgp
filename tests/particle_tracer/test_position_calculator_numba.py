import numpy as np
import pytest

import sedtrails.particle_tracer.position_calculator_numba as position_module
from sedtrails.particle_tracer.position_calculator_numba import (
    BOUNDARY_CLASS_LAND,
    BOUNDARY_CLASS_OPEN,
    create_grid_geometry,
    create_numba_particle_calculator,
)


def square_grid():
    grid_x = np.array([0.0, 1.0, 1.0, 0.0])
    grid_y = np.array([0.0, 0.0, 1.0, 1.0])
    return grid_x, grid_y


def test_create_grid_geometry_preserves_preexisting_positional_arguments():
    """Native neighbors must remain an opt-in keyword-only argument."""
    grid_x, grid_y = square_grid()

    geometry = create_grid_geometry(grid_x, grid_y, None, None, 'projected')

    assert geometry.triangles.shape[1] == 3


def test_grid_geometry_from_points_preserves_positional_boundary_arguments():
    """Existing positional boundary arguments must keep their meaning."""
    grid_x, grid_y = square_grid()
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    classification = {
        'edge_nodes': np.array([[0, 1]], dtype=np.int32),
        'edge_classes': ['open'],
    }

    geometry = position_module.GridGeometry.from_points(
        grid_x,
        grid_y,
        triangles,
        classification,
        'projected',
    )

    np.testing.assert_array_equal(
        geometry.boundary_edges,
        np.array([[0, 1]], dtype=np.int32),
    )


def test_numba_calculator_uses_supplied_native_geodetic_neighbors(monkeypatch):
    """The calculator factory must forward authoritative geodetic topology."""
    import sedtrails.particle_tracer.geodetic_grid as geodetic_grid_module

    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    native_neighbors = np.array([[1, -1, -1], [-1, 0, -1]], dtype=np.int32)
    grid_x = np.array([0.0, 1.0, 0.0, 1.0])
    grid_y = np.array([0.0, 0.0, 1.0, 1.0])

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("The fallback neighbor construction must not run.")

    monkeypatch.setattr(
        geodetic_grid_module,
        "_compute_triangle_neighbors",
        unexpected_fallback,
    )
    calculator = create_numba_particle_calculator(
        grid_x,
        grid_y,
        triangles=triangles,
        coordinate_system='geographic',
        runtime_geometry='geodetic',
        surface_model='sphere',
        velocity_basis='east_north',
        triangle_neighbors=native_neighbors,
    )

    np.testing.assert_array_equal(
        calculator['geometry'].triangle_neighbors,
        native_neighbors,
    )


def test_planar_geometry_uses_authoritative_triangle_neighbors():
    """Planar geometry should retain valid source neighbor topology."""
    grid_x, grid_y = square_grid()
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    native_neighbors = np.array([[-1, 1, -1], [-1, -1, 0]], dtype=np.int32)

    geometry = create_grid_geometry(
        grid_x,
        grid_y,
        triangles=triangles,
        triangle_neighbors=native_neighbors,
    )

    np.testing.assert_array_equal(geometry.triangle_neighbors, native_neighbors)
    assert geometry.triangle_neighbors.dtype == np.int32


def test_planar_prepared_geographic_velocity_is_cached_by_generation():
    """Projected velocity conversion should occur once per forcing key."""
    grid_x = np.array([4.0, 4.01, 4.0])
    grid_y = np.array([52.0, 52.0, 52.01])
    geometry = create_grid_geometry(
        grid_x,
        grid_y,
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
        coordinate_system='geographic',
        source_crs='EPSG:4326',
        metric_crs='EPSG:32631',
    )
    u_values = np.ones(3)
    v_values = np.zeros(3)

    first = geometry.prepare_vector_field(
        u_values,
        v_values,
        cache_key=(7, 'lower'),
    )
    second = geometry.prepare_vector_field(
        u_values,
        v_values,
        cache_key=(7, 'lower'),
    )

    assert first is second
    assert first[0].shape == (3,)
    assert first[1].shape == (3,)


def test_planar_prepared_projected_float32_fields_remain_views():
    """Preparation must not copy already contiguous projected fields."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)
    u_values = np.ones(4, dtype=np.float32)
    v_values = np.zeros(4, dtype=np.float32)

    prepared_u, prepared_v = geometry.prepare_vector_field(u_values, v_values)

    assert np.shares_memory(prepared_u, u_values)
    assert np.shares_memory(prepared_v, v_values)
    assert prepared_u.dtype == np.float32
    assert prepared_v.dtype == np.float32


def test_planar_prepared_update_matches_regular_temporal_update():
    """Prepared fields must preserve planar RK4 and boundary results."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)
    x0 = np.array([0.25, 0.75])
    y0 = np.array([0.25, 0.75])
    lower_u = np.full(4, 0.1)
    lower_v = np.zeros(4)
    upper_u = np.full(4, 0.2)
    upper_v = np.zeros(4)
    starts = geometry.locate_points(x0, y0)

    expected = geometry.update_particles_temporal_with_boundary_class(
        x0,
        y0,
        lower_u,
        lower_v,
        upper_u,
        upper_v,
        0.25,
        0.1,
        simplex_ids=starts,
    )
    actual = geometry.update_particles_prepared_temporal_with_boundary_class(
        x0,
        y0,
        geometry.prepare_vector_field(lower_u, lower_v),
        geometry.prepare_vector_field(upper_u, upper_v),
        0.25,
        0.1,
        simplex_ids=starts,
    )

    for actual_values, expected_values in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_values, expected_values)


def test_planar_location_bounds_direct_api_temporary_batches(monkeypatch):
    """Direct geometry calls should not materialize all query points at once."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)
    observed_sizes = []
    original = geometry._locate_points_chunk

    def small_chunks(size):
        for start in range(0, size, 2):
            yield slice(start, min(start + 2, size))

    def record_chunk(x_values, y_values, starts=None):
        observed_sizes.append(len(x_values))
        return original(x_values, y_values, starts)

    monkeypatch.setattr(position_module, 'particle_chunk_slices', small_chunks)
    monkeypatch.setattr(geometry, '_locate_points_chunk', record_chunk)

    simplices = geometry.locate_points(
        np.linspace(0.1, 0.9, 5),
        np.full(5, 0.5),
    )

    assert observed_sizes == [2, 2, 1]
    assert np.all(simplices >= 0)


def test_cached_multi_field_interpolation_does_not_stack_nodal_fields(monkeypatch):
    """Each nodal field should remain a view instead of joining a mesh copy."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)
    x_points = np.array([0.25, 0.75])
    y_points = np.array([0.25, 0.25])

    monkeypatch.setattr(
        position_module.np,
        'vstack',
        lambda *_args, **_kwargs: pytest.fail('nodal fields must not be stacked'),
    )
    (values_a, values_b), simplices = geometry.interpolate_fields_with_simplex(
        (grid_x + grid_y, 2.0 * grid_x - grid_y),
        x_points,
        y_points,
    )

    np.testing.assert_allclose(values_a, [0.5, 1.0])
    np.testing.assert_allclose(values_b, [0.25, 1.25])
    assert np.all(simplices >= 0)


def test_cached_geometry_reused_by_calculator():
    """Ensures the calculator reuses a provided GridGeometry instance."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)

    calculator = create_numba_particle_calculator(grid_x, grid_y, grid_geometry=geometry)

    assert calculator['geometry'] is geometry
    assert calculator['triangles'].shape[1] == 3
    assert calculator['interpolate_fields_with_simplex'].__self__ is geometry


def test_linear_field_interpolation_uses_cached_delaunay_locator():
    """Checks linear nodal fields interpolate to exact values at query points."""
    grid_x, grid_y = square_grid()
    calculator = create_numba_particle_calculator(grid_x, grid_y)
    field = 2.0 * grid_x + 3.0 * grid_y

    x_points = np.array([0.25, 0.75, 1.0])
    y_points = np.array([0.25, 0.75, 0.5])
    expected = 2.0 * x_points + 3.0 * y_points

    interpolated = calculator['interpolate_field'](field, x_points, y_points)

    np.testing.assert_allclose(interpolated, expected, rtol=1e-12, atol=1e-12)


def test_multi_field_interpolation_reuses_single_point_location():
    """Verifies multiple fields share one point-location pass."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)
    call_count = 0
    original_locate_points = geometry.locate_points

    def counted_locate_points(x_points, y_points, start_simplices=None):
        nonlocal call_count
        call_count += 1
        return original_locate_points(x_points, y_points, start_simplices)

    geometry.locate_points = counted_locate_points
    field_a = grid_x + grid_y
    field_b = 2.0 * grid_x - grid_y

    values_a, values_b = geometry.interpolate_fields(
        (field_a, field_b),
        np.array([0.25, 0.75]),
        np.array([0.25, 0.25]),
    )

    assert call_count == 1
    np.testing.assert_allclose(values_a, [0.5, 1.0])
    np.testing.assert_allclose(values_b, [0.25, 1.25])


def test_multi_field_interpolation_with_simplex_matches_global_lookup():
    """Cached-simplex field interpolation should match regular interpolation."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)
    x_points = np.array([0.25, 0.75])
    y_points = np.array([0.25, 0.25])
    starts = geometry.locate_points(x_points, y_points)
    field_a = grid_x + grid_y
    field_b = 2.0 * grid_x - grid_y

    expected_a, expected_b = geometry.interpolate_fields((field_a, field_b), x_points, y_points)
    (values_a, values_b), simplices = geometry.interpolate_fields_with_simplex(
        (field_a, field_b),
        x_points,
        y_points,
        simplex_ids=starts,
    )

    np.testing.assert_allclose(values_a, expected_a)
    np.testing.assert_allclose(values_b, expected_b)
    np.testing.assert_array_equal(simplices, starts)


def test_multi_field_interpolation_with_simplex_returns_flat_values():
    """Cached interpolation should match the flat output shape of regular interpolation."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)
    x_points = np.array([[0.25, 0.75]])
    y_points = np.array([[0.25, 0.25]])
    starts = geometry.locate_points(x_points, y_points)

    (values,), simplices = geometry.interpolate_fields_with_simplex(
        (grid_x + grid_y,),
        x_points,
        y_points,
        simplex_ids=starts,
    )

    assert values.shape == (2,)
    assert simplices.shape == (2,)
    np.testing.assert_allclose(values, [0.5, 1.0])


def test_multi_field_interpolation_with_simplex_rejects_bad_field_length():
    """Invalid nodal field lengths should fail before entering the fused kernel."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)
    x_points = np.array([0.25])
    y_points = np.array([0.25])
    starts = geometry.locate_points(x_points, y_points)

    with pytest.raises(ValueError, match='field arrays must have 4 values, got 3'):
        geometry.interpolate_fields_with_simplex(
            (np.array([1.0, 2.0, 3.0]),),
            x_points,
            y_points,
            simplex_ids=starts,
        )


def test_multi_field_interpolation_with_simplex_refreshes_bad_cache():
    """Bad simplex seeds should fall back to a valid containing triangle."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)
    x_points = np.array([0.25, 0.75])
    y_points = np.array([0.25, 0.25])
    bad_starts = np.full(x_points.shape, -1, dtype=np.int64)

    (values,), simplices = geometry.interpolate_fields_with_simplex(
        (grid_x + grid_y,),
        x_points,
        y_points,
        simplex_ids=bad_starts,
    )

    np.testing.assert_allclose(values, [0.5, 1.0])
    np.testing.assert_array_equal(simplices, geometry.locate_points(x_points, y_points))


def test_multi_field_interpolation_with_simplex_marks_moved_outside_particle():
    """Valid simplex seeds should refresh to -1 for points that moved outside."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)
    starts = geometry.locate_points(np.array([0.25]), np.array([0.25]))

    (values,), simplices = geometry.interpolate_fields_with_simplex(
        (grid_x + grid_y,),
        np.array([2.0]),
        np.array([0.5]),
        simplex_ids=starts,
    )

    assert np.isnan(values[0])
    np.testing.assert_array_equal(simplices, np.array([-1]))


def test_interpolation_outside_domain_returns_nan():
    """Outside-domain particle queries should not be silently converted to zero."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)

    values = geometry.interpolate_field(grid_x + grid_y, np.array([2.0]), np.array([0.5]))

    assert np.isnan(values[0])


def test_explicit_triangle_connectivity_is_preserved():
    """Confirms user-supplied triangle connectivity is used unchanged."""
    grid_x, grid_y = square_grid()
    triangles = np.array([[0, 1, 2], [0, 2, 3]])

    calculator = create_numba_particle_calculator(grid_x, grid_y, triangles=triangles)

    np.testing.assert_array_equal(calculator['triangles'], triangles)


def test_authoritative_planar_geometry_avoids_global_hull_and_preserves_int32(monkeypatch):
    """Explicit topology bypasses point deduplication/hull and retains compact indices."""
    grid_x, grid_y = square_grid()
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    monkeypatch.setattr(
        position_module.np,
        'unique',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('authoritative topology must not deduplicate all points')
        ),
    )
    monkeypatch.setattr(
        position_module,
        'ConvexHull',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('authoritative topology must not build a convex hull')
        ),
    )

    geometry = create_grid_geometry(grid_x, grid_y, triangles=triangles)

    assert geometry.triangles is triangles
    assert geometry.triangles.dtype == np.int32
    assert geometry.triangle_neighbors.dtype == np.int32
    assert geometry.outer_envelope.shape == (4, 2)


def test_explicit_empty_triangle_connectivity_disables_delaunay_fallback():
    """Explicit empty connectivity should leave all particle locations outside."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y, triangles=np.empty((0, 3), dtype=np.int64))

    np.testing.assert_array_equal(geometry.triangles, np.empty((0, 3), dtype=np.int64))
    np.testing.assert_array_equal(geometry.locate_points(np.array([0.5]), np.array([0.5])), np.array([-1]))
    values = geometry.interpolate_field(grid_x + grid_y, np.array([0.5]), np.array([0.5]))
    assert np.isnan(values[0])


def test_rk4_update_with_constant_velocity():
    """Validates RK4 advection matches constant-velocity analytical motion."""
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


def test_geographic_update_uses_metric_geometry_and_returns_metric():
    """Geographic grids should project once and advect in runtime metres."""
    grid_x = np.array([4.0, 4.001, 4.001, 4.0])
    grid_y = np.array([52.0, 52.0, 52.001, 52.001])
    geometry = create_grid_geometry(
        grid_x,
        grid_y,
        coordinate_system='geographic',
        metric_crs='EPSG:3857',
    )
    x0, y0 = geometry.coordinate_transform.source_to_metric(
        np.array([4.0002]),
        np.array([52.0002]),
    )
    grid_u = np.ones_like(grid_x)
    grid_v = np.zeros_like(grid_y)
    dt = 1.0e-3

    u_metric, v_metric = geometry.interpolate_vector(grid_u, grid_v, x0, y0)
    x_new, y_new = geometry.update_particles(x0, y0, grid_u, grid_v, dt, igeo=0)
    expected_x = x0 + u_metric * dt
    expected_y = y0 + v_metric * dt

    assert not np.allclose(u_metric, np.ones_like(u_metric), rtol=0.0, atol=1.0e-3)
    np.testing.assert_allclose(x_new, expected_x, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(y_new, expected_y, rtol=1e-9, atol=1e-9)


def test_geographic_velocity_arrays_project_east_north_components():
    """East/north source velocities should be mapped into the projected metric basis."""
    grid_x = np.array([4.0, 4.001, 4.001, 4.0])
    grid_y = np.array([52.0, 52.0, 52.001, 52.001])
    geometry = create_grid_geometry(
        grid_x,
        grid_y,
        coordinate_system='geographic',
        metric_crs='EPSG:3857',
    )

    u_metric, v_metric = geometry._velocity_arrays(np.ones_like(grid_x), np.zeros_like(grid_y), igeo=0)

    np.testing.assert_allclose(u_metric, geometry.velocity_east_x, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(v_metric, geometry.velocity_east_y, rtol=0.0, atol=0.0)
    assert not np.allclose(u_metric, np.ones_like(u_metric), rtol=0.0, atol=1.0e-3)


def test_geographic_update_does_not_call_pyproj_during_movement(monkeypatch):
    """Per-timestep movement should use precomputed projected geometry and basis arrays."""
    grid_x = np.array([4.0, 4.001, 4.001, 4.0])
    grid_y = np.array([52.0, 52.0, 52.001, 52.001])
    geometry = create_grid_geometry(grid_x, grid_y, coordinate_system='geographic')
    x0, y0 = geometry.coordinate_transform.source_to_metric(
        np.array([4.0002]),
        np.array([52.0002]),
    )

    def fail_transform(*args, **kwargs):
        raise AssertionError('pyproj transform should not run during particle movement')

    monkeypatch.setattr(geometry.coordinate_transform, 'source_to_metric', fail_transform)
    monkeypatch.setattr(geometry.coordinate_transform, 'metric_to_source', fail_transform)

    x_new, y_new = geometry.update_particles(
        x0,
        y0,
        np.ones_like(grid_x),
        np.zeros_like(grid_y),
        1.0,
    )

    assert np.isfinite(x_new).all()
    assert np.isfinite(y_new).all()


def test_geographic_cached_simplex_interpolation_uses_metric_coordinates():
    """Cached-simplex field interpolation should use projected runtime coordinates."""
    grid_x = np.array([4.0, 4.001, 4.001, 4.0])
    grid_y = np.array([52.0, 52.0, 52.001, 52.001])
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    geometry = create_grid_geometry(grid_x, grid_y, triangles=triangles, coordinate_system='geographic')
    x0, y0 = geometry.coordinate_transform.source_to_metric(
        np.array([4.0002]),
        np.array([52.0002]),
    )
    field = geometry.metric_grid_x + 2.0 * geometry.metric_grid_y

    (values,), simplices = geometry.interpolate_fields_with_simplex((field,), x0, y0)
    (cached_values,), refreshed = geometry.interpolate_fields_with_simplex((field,), x0, y0, simplices)

    expected = x0 + 2.0 * y0
    np.testing.assert_allclose(values, expected, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(cached_values, expected, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_array_equal(refreshed, simplices)


def test_legacy_igeo_scaling_is_rejected():
    """The old per-step spherical scaling mode must not bypass pyproj geometry."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)

    with pytest.raises(ValueError, match='igeo=1'):
        geometry.update_particles(
            np.array([0.2]),
            np.array([0.2]),
            np.ones_like(grid_x),
            np.zeros_like(grid_y),
            1.0,
            igeo=1,
        )


def test_temporal_rk4_update_matches_preblended_grid():
    """Ensures temporal RK4 equals RK4 on a preblended velocity field."""
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


def test_temporal_rk4_uses_one_point_location_pass_per_update():
    """Checks temporal updates call point-location only once per update."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)
    call_count = 0
    original_locate_points = geometry.locate_points

    def counted_locate_points(x_points, y_points, start_simplices=None):
        nonlocal call_count
        call_count += 1
        return original_locate_points(x_points, y_points, start_simplices)

    geometry.locate_points = counted_locate_points

    geometry.update_particles_temporal(
        np.array([0.2, 0.4]),
        np.array([0.2, 0.3]),
        np.ones_like(grid_x),
        np.full_like(grid_y, 0.5),
        np.full_like(grid_x, 3.0),
        np.full_like(grid_y, 1.5),
        0.25,
        0.1,
    )

    assert call_count == 1


def test_temporal_update_returns_reusable_simplex_ids():
    """Confirms temporal updates return simplex ids usable as next-step seeds."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)
    starts = geometry.locate_points(np.array([0.2, 0.4]), np.array([0.2, 0.3]))

    x_new, y_new, new_simplices = geometry.update_particles_temporal_with_simplex(
        np.array([0.2, 0.4]),
        np.array([0.2, 0.3]),
        np.ones_like(grid_x),
        np.full_like(grid_y, 0.5),
        np.full_like(grid_x, 3.0),
        np.full_like(grid_y, 1.5),
        0.25,
        0.1,
        simplex_ids=starts,
    )

    np.testing.assert_allclose(x_new, [0.35, 0.55])
    np.testing.assert_allclose(y_new, [0.275, 0.375])
    np.testing.assert_array_equal(new_simplices, geometry.locate_points(x_new, y_new, new_simplices))


def test_position_update_preserves_particle_array_shape():
    """Ensures particle update keeps input array dimensionality intact."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)
    x0 = np.array([[0.2, 0.4]])
    y0 = np.array([[0.2, 0.3]])

    x_new, y_new = geometry.update_particles(
        x0,
        y0,
        np.ones_like(grid_x),
        np.full_like(grid_y, 0.5),
        0.1,
    )

    assert x_new.shape == x0.shape
    assert y_new.shape == y0.shape
    np.testing.assert_allclose(x_new, x0 + 0.1)
    np.testing.assert_allclose(y_new, y0 + 0.05)


def test_velocity_arrays_do_not_copy_non_geographic_float32_fields():
    """Verifies non-geographic velocity arrays are returned without copies."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(grid_x, grid_y)
    grid_u = np.ones_like(grid_x, dtype=np.float32)
    grid_v = np.ones_like(grid_y, dtype=np.float32)

    grid_u_adj, grid_v_adj = geometry._velocity_arrays(grid_u, grid_v, igeo=0)

    assert np.shares_memory(grid_u_adj, grid_u)
    assert np.shares_memory(grid_v_adj, grid_v)


def test_boundary_crossing_classification_uses_encoded_edge_classes():
    """Boundary crossing classes should use encoded edge classes and preserve labels."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(
        grid_x,
        grid_y,
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        boundary_edge_classification={
            'edge_nodes': [[0, 1], [1, 2], [2, 3], [3, 0]],
            'edge_classes': ['open', 'land', 'unclassified', 'land'],
        },
    )

    classes = geometry.classify_boundary_crossings(
        np.array([0.5, 0.8, 0.5]),
        np.array([0.2, 0.5, 0.8]),
        np.array([0.5, 1.3, 0.5]),
        np.array([-0.3, 0.5, 1.3]),
    )

    assert geometry.boundary_edge_class_codes.dtype == np.int8
    assert geometry.triangle_edge_class_codes.dtype == np.int8
    assert BOUNDARY_CLASS_OPEN in geometry.triangle_edge_class_codes
    assert BOUNDARY_CLASS_LAND in geometry.triangle_edge_class_codes
    np.testing.assert_array_equal(classes, np.array(['open', 'land', 'unclassified'], dtype=object))


def test_boundary_aware_update_returns_exit_class_codes():
    """Particle updates should report the boundary class crossed by the final simplex walk."""
    grid_x, grid_y = square_grid()
    calculator = create_numba_particle_calculator(
        grid_x,
        grid_y,
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        boundary_edge_classification={
            'edge_nodes': [[0, 1], [1, 2], [2, 3], [3, 0]],
            'edge_classes': ['open', 'land', 'unclassified', 'land'],
        },
    )
    geometry = calculator['geometry']

    x_new, y_new, new_simplices, boundary_class_codes = geometry.update_particles_with_boundary_class(
        np.array([0.5, 0.8]),
        np.array([0.2, 0.5]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0, 0.0]),
        0.5,
        simplex_ids=geometry.locate_points(np.array([0.5, 0.8]), np.array([0.2, 0.5])),
    )

    np.testing.assert_allclose(x_new, np.array([0.5, 0.8]))
    assert y_new[0] < 0.0
    assert new_simplices[0] == -1
    assert boundary_class_codes[0] == BOUNDARY_CLASS_OPEN


def test_boundary_crossing_classification_rejects_mismatched_segments():
    """Boundary crossing classification should validate segment array lengths before numba execution."""
    grid_x, grid_y = square_grid()
    geometry = create_grid_geometry(
        grid_x,
        grid_y,
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        boundary_edge_classification={
            'edge_nodes': [[0, 1]],
            'edge_classes': ['open'],
        },
    )

    with pytest.raises(ValueError, match='start and end coordinates'):
        geometry.classify_boundary_crossings(
            np.array([0.5, 0.6]),
            np.array([0.2, 0.2]),
            np.array([0.5]),
            np.array([-0.3]),
        )
