"""Integration tests for ocean-scale geodetic grid geometry."""

import time

import numpy as np
import pytest

import sedtrails.particle_tracer.geodetic_grid as geodetic_grid_module
import sedtrails.particle_tracer.particle_seeder as particle_seeder_module
from sedtrails.particle_tracer.geodetic_geometry import spherical_distance
from sedtrails.particle_tracer.particle_seeder import ParticlePopulation
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


def _structured_geometry(nx=41, ny=21):
    """Build a classified triangular geographic grid for scale regressions."""
    longitude_values = np.linspace(0.0, 0.04, nx)
    latitude_values = np.linspace(30.0, 30.02, ny)
    longitude, latitude = np.meshgrid(longitude_values, latitude_values)
    cell = np.arange((nx - 1) * (ny - 1), dtype=np.int64)
    lower_left = (cell // (nx - 1)) * nx + np.remainder(cell, nx - 1)
    triangles = np.empty((2 * cell.size, 3), dtype=np.int64)
    triangles[0::2] = np.column_stack(
        (lower_left, lower_left + 1, lower_left + nx)
    )
    triangles[1::2] = np.column_stack(
        (lower_left + 1, lower_left + nx + 1, lower_left + nx)
    )

    south = np.column_stack((np.arange(nx - 1), np.arange(1, nx)))
    north_start = (ny - 1) * nx
    north = north_start + south
    west_start = np.arange(ny - 1) * nx
    west = np.column_stack((west_start, west_start + nx))
    east = west + (nx - 1)
    boundary_edges = np.vstack((south, north, west, east))
    boundary_classes = np.full(boundary_edges.shape[0], 'land', dtype=object)
    boundary_classes[-east.shape[0] :] = 'open'

    return create_grid_geometry(
        longitude.ravel(),
        latitude.ravel(),
        triangles=triangles,
        boundary_edge_classification={
            'edge_nodes': boundary_edges,
            'edge_classes': boundary_classes,
        },
        coordinate_system='geographic',
        runtime_geometry='geodetic',
        surface_model='sphere',
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


def test_geodetic_grid_does_not_implicitly_cache_by_allocation_identity():
    """Arrays without a forcing generation must never enter the shared cache."""
    geometry = _dateline_geometry()
    u = np.ones(3)
    v = np.zeros(3)
    u.flags.writeable = False
    v.flags.writeable = False

    first = geometry.prepare_vector_field(u, v)
    second = geometry.prepare_vector_field(u, v)

    assert first is not second
    assert len(geometry._velocity_cache) == 0


def test_geodetic_grid_explicit_generation_caches_writable_forcing():
    """Writable production slices should cache only with explicit generations."""
    geometry = _dateline_geometry()
    u = np.ones(3)
    v = np.zeros(3)

    first = geometry.prepare_vector_field(u, v, cache_key=('window', 0))
    second = geometry.prepare_vector_field(u, v, cache_key=('window', 0))
    u[:] = 2.0
    third = geometry.prepare_vector_field(u, v, cache_key=('window', 1))

    assert first is second
    assert third is not first
    np.testing.assert_allclose(np.linalg.norm(third, axis=1), 2.0)


def test_geodetic_grid_vector_cache_enforces_byte_limit():
    """Prepared forcing cache should evict old slices before exceeding its cap."""
    geometry = _dateline_geometry()
    u = np.ones(3)
    v = np.zeros(3)
    field_bytes = geometry.prepare_vector_field(u, v).nbytes
    geometry.clear_velocity_cache()
    geometry._velocity_cache_max_bytes = field_bytes

    geometry.prepare_vector_field(u, v, cache_key=('window', 0))
    geometry.prepare_vector_field(u, v, cache_key=('window', 1))

    assert len(geometry._velocity_cache) == 1
    assert geometry._velocity_cache_bytes <= geometry._velocity_cache_max_bytes


def test_geodetic_grid_purges_previous_forcing_generation():
    """A new input window must release all prepared vectors from the old one."""
    geometry = _dateline_geometry()
    u = np.ones(3)
    v = np.zeros(3)

    geometry.prepare_vector_field(
        u,
        v,
        cache_key=(7, ('plan',), 'velocity', 0, 0),
    )
    geometry.prepare_vector_field(
        u,
        v,
        cache_key=(7, ('plan',), 'velocity', 0, 1),
    )
    assert len(geometry._velocity_cache) == 2

    geometry.prepare_vector_field(
        u,
        v,
        cache_key=(8, ('plan',), 'velocity', 0, 0),
    )

    assert len(geometry._velocity_cache) == 1
    assert geometry._velocity_cache_generation == 8
    assert all(key[0] == 8 for key in geometry._velocity_cache)


def test_geodetic_grid_purges_before_oversized_generation_conversion():
    """Oversized new vectors must still evict cache entries from old windows."""
    geometry = _dateline_geometry()
    u = np.ones(3)
    v = np.zeros(3)
    prepared = geometry.prepare_vector_field(
        u,
        v,
        cache_key=(7, ('plan',), 'velocity', 0, 0),
    )
    assert len(geometry._velocity_cache) == 1
    geometry._velocity_cache_max_bytes = prepared.nbytes - 1

    geometry.prepare_vector_field(
        u,
        v,
        cache_key=(8, ('plan',), 'velocity', 0, 0),
    )

    assert len(geometry._velocity_cache) == 0
    assert geometry._velocity_cache_generation == 8


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


def test_geodetic_grid_location_uses_compiled_array_kernels():
    """Cached and cold point location should not enter Python particle loops."""
    geometry = _dateline_geometry()

    cached = geometry.locate_points(
        np.array([179.5, -179.5]),
        np.array([-0.5, -0.5]),
        np.array([0, 0]),
    )
    cold = geometry.locate_points(
        np.array([179.5, -179.5]),
        np.array([-0.5, -0.5]),
    )

    np.testing.assert_array_equal(cached, [0, 0])
    np.testing.assert_array_equal(cold, [0, 0])
    assert geodetic_grid_module._walk_points_ecef_numba.nopython_signatures
    assert geodetic_grid_module._locate_candidates_ecef_numba.nopython_signatures


def test_geodetic_face_walk_normalizes_tolerance_for_eleven_metre_faces():
    """A wrong cached face on a small mesh must walk to the actual neighbor."""
    longitude, latitude = np.meshgrid(
        np.array([0.0, 0.0001, 0.0002]),
        np.array([30.0, 30.0001]),
    )
    triangles = np.array(
        [
            [0, 1, 3],
            [1, 4, 3],
            [1, 2, 4],
            [2, 5, 4],
        ],
        dtype=np.int64,
    )
    geometry = create_grid_geometry(
        longitude.ravel(),
        latitude.ravel(),
        triangles=triangles,
        coordinate_system='geographic',
        runtime_geometry='geodetic',
        surface_model='sphere',
        velocity_basis='east_north',
    )
    x = np.array([0.000175])
    y = np.array([30.000025])

    cold = geometry.locate_points(x, y)
    walked = geometry.locate_points(x, y, start_simplices=np.array([0]))

    assert cold[0] in (2, 3)
    np.testing.assert_array_equal(walked, cold)


def test_geodetic_rk4_avoids_intermediate_lonlat_roundtrips(monkeypatch):
    """RK4 should convert ECEF to public longitude/latitude only at output."""
    geometry = _dateline_geometry()
    original = geodetic_grid_module.ecef_to_lonlat
    calls = 0

    def counted_conversion(values):
        nonlocal calls
        calls += 1
        return original(values)

    monkeypatch.setattr(geodetic_grid_module, 'ecef_to_lonlat', counted_conversion)
    geometry.update_particles_with_simplex(
        np.array([179.5, -179.5]),
        np.array([-0.5, -0.5]),
        np.ones(3),
        np.zeros(3),
        10.0,
        simplex_ids=np.array([0, 0]),
    )

    assert calls == 1


def test_geodetic_forcing_is_prepared_once_across_chunks_and_populations(monkeypatch):
    """Equivalent plan arrays should share one prepared forcing generation."""
    geometry = _dateline_geometry()
    conversion_calls = 0
    original = geodetic_grid_module._east_north_to_ecef_numba
    original_flatnonzero = np.flatnonzero
    flatnonzero_input_sizes = []

    def counted_conversion(*args):
        nonlocal conversion_calls
        conversion_calls += 1
        return original(*args)

    monkeypatch.setattr(
        geodetic_grid_module,
        '_east_north_to_ecef_numba',
        counted_conversion,
    )

    def bounded_flatnonzero(values):
        flatnonzero_input_sizes.append(np.asarray(values).size)
        return original_flatnonzero(values)

    monkeypatch.setattr(
        particle_seeder_module.np,
        'flatnonzero',
        bounded_flatnonzero,
    )
    plan_identity = ('passive_tracer', ('velocity_basis', 'east_north'))
    flow_fields = [
        {
            'lower': {'u': np.ones(3), 'v': np.zeros(3)},
            'upper': {'u': 2.0 * np.ones(3), 'v': np.zeros(3)},
            'weight': 0.5,
            'cache_generation': {
                'lower': (7, plan_identity, 'depth_avg_flow_velocity', 0, 0),
                'upper': (7, plan_identity, 'depth_avg_flow_velocity', 0, 1),
            },
        }
        for _ in range(2)
    ]
    assert flow_fields[0]['lower']['u'] is not flow_fields[1]['lower']['u']
    prepared_by_population = []
    for flow_field in flow_fields:
        population = object.__new__(ParticlePopulation)
        particle_count = 65_537
        population.grid_geometry = geometry
        population.particles = {
            'x': np.full(particle_count, 179.5),
            'y': np.full(particle_count, -0.5),
            'status_mobile': np.ones(particle_count, dtype=bool),
        }
        population._particle_simplices = np.zeros(particle_count, dtype=np.int64)
        chunk_fields = []

        def capture_chunk(
            supplied_flow_field,
            current_timestep,
            particle_indices,
            *,
            prepared_flow_field,
            _chunk_fields=chunk_fields,
        ):
            _chunk_fields.append(prepared_flow_field)

        population._update_position_chunk = capture_chunk
        population.update_position(flow_field, 10.0)
        assert len(chunk_fields) == 2
        assert chunk_fields[0] is chunk_fields[1]
        prepared_by_population.append(chunk_fields[0])

    assert conversion_calls == 2
    assert max(flatnonzero_input_sizes) <= 65_536
    assert prepared_by_population[0]['lower'] is prepared_by_population[1]['lower']
    assert prepared_by_population[0]['upper'] is prepared_by_population[1]['upper']


def test_geodetic_structured_crossing_and_exit_throughput_regression():
    """Multi-face RK4 and compiled exits should retain bounded throughput."""
    geometry = _structured_geometry()
    particle_count = 5_000
    rng = np.random.default_rng(42)
    longitude = rng.uniform(0.002, 0.035, particle_count)
    latitude = rng.uniform(30.002, 30.018, particle_count)
    exit_count = 500
    longitude[-exit_count:] = 0.0399
    starts = geometry.locate_points(longitude, latitude)
    u = 2.0 * np.ones(geometry.grid_x.size)
    v = np.zeros(geometry.grid_x.size)

    geometry.update_particles_with_boundary_class(
        longitude[:4],
        latitude[:4],
        u,
        v,
        150.0,
        simplex_ids=starts[:4],
    )

    durations = []
    result = None
    for _ in range(3):
        start_time = time.perf_counter()
        result = geometry.update_particles_with_boundary_class(
            longitude,
            latitude,
            u,
            v,
            150.0,
            simplex_ids=starts,
        )
        durations.append(time.perf_counter() - start_time)

    new_longitude, new_latitude, new_faces, boundary_codes = result
    outside = new_faces < 0
    inside = ~outside
    assert np.count_nonzero(outside) >= exit_count
    np.testing.assert_array_equal(boundary_codes[outside], np.ones(np.count_nonzero(outside)))
    assert np.count_nonzero(new_faces[inside] != starts[inside]) > particle_count // 2
    np.testing.assert_array_equal(
        new_faces[inside],
        geometry.locate_points(new_longitude[inside], new_latitude[inside]),
    )
    assert geodetic_grid_module._boundary_codes_for_exits_numba.nopython_signatures

    assert particle_count / np.median(durations) >= 10_000.0


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
