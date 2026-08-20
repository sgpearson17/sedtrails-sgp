import numpy as np

from sedtrails.particle_tracer.data_retriever import FieldDataRetriever
from sedtrails.particle_tracer.position_calculator_numba import create_grid_geometry
from sedtrails.transport_converter.sedtrails_data import SedtrailsData
from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata


def build_sedtrails_data():
    """Builds a minimal two-time-slice SedtrailsData fixture for retriever tests."""
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    lower_u = np.array([1.0, 1.0, 1.0, 1.0])
    upper_u = np.array([3.0, 3.0, 3.0, 3.0])
    lower_v = np.array([0.5, 0.5, 0.5, 0.5])
    upper_v = np.array([1.5, 1.5, 1.5, 1.5])
    lower_mag = np.array([1.0, 2.0, 1.0, 2.0])
    upper_mag = np.array([4.0, 1.0, 4.0, 1.0])
    scalar = np.array([[0.0, 1.0, 2.0, 1.0], [2.0, 3.0, 4.0, 3.0]])
    zeros_time_space = np.zeros((2, x.size), dtype=float)
    zero_vector = {
        'x': zeros_time_space.copy(),
        'y': zeros_time_space.copy(),
        'magnitude': zeros_time_space.copy(),
    }
    metadata = SedtrailsMetadata(
        flowfield_domain={
            'x_min': float(np.min(x)),
            'x_max': float(np.max(x)),
            'y_min': float(np.min(y)),
            'y_max': float(np.max(y)),
        }
    )

    # Assemble the minimal model input used by the tests: one flow field, one scalar field,
    # and two timesteps to exercise lower/upper bounds and temporal weighting logic.
    return SedtrailsData(
        times=np.array([0.0, 10.0], dtype=float),
        reference_date=np.datetime64('1970-01-01T00:00:00'),
        x=x,
        y=y,
        bed_level=scalar[0],
        depth_avg_flow_velocity={
            'x': np.vstack([lower_u, upper_u]),
            'y': np.vstack([lower_v, upper_v]),
            'magnitude': np.vstack([lower_mag, upper_mag]),
        },
        fractions=1,
        bed_load_transport=zero_vector,
        suspended_transport=zero_vector,
        water_depth=scalar,
        mean_bed_shear_stress=zeros_time_space.copy(),
        max_bed_shear_stress=zeros_time_space.copy(),
        sediment_concentration=zeros_time_space.copy(),
        nonlinear_wave_velocity=zero_vector,
        metadata=metadata,
    )


def test_flow_field_bounds_defer_full_grid_interpolation():
    """Checks flow bounds return lower/upper slices and interpolation weight."""
    retriever = FieldDataRetriever(build_sedtrails_data())

    bounds = retriever.get_flow_field_bounds(2.5, 'depth_avg_flow_velocity')

    assert bounds['weight'] == 0.25
    np.testing.assert_allclose(bounds['lower']['u'], 1.0)
    np.testing.assert_allclose(bounds['upper']['u'], 3.0)


def test_flow_field_bounds_expose_stable_cache_generations():
    """Forcing bounds should identify slices consistently across retrievers."""
    data = build_sedtrails_data()
    first = FieldDataRetriever(data).get_flow_field_bounds(
        2.5,
        'depth_avg_flow_velocity',
    )
    second = FieldDataRetriever(data).get_flow_field_bounds(
        2.5,
        'depth_avg_flow_velocity',
    )

    assert first['cache_generation'] == second['cache_generation']
    assert first['cache_generation']['lower'] != first['cache_generation']['upper']
    hash(first['cache_generation']['lower'])
    hash(first['cache_generation']['upper'])


def test_new_forcing_window_invalidates_reused_array_allocation():
    """A new data window must not reuse prepared vectors from the same buffer."""
    first_data = build_sedtrails_data()
    reused_u = first_data.depth_avg_flow_velocity['x']
    first_bounds = FieldDataRetriever(first_data).get_flow_field_bounds(
        2.5,
        'depth_avg_flow_velocity',
    )
    geometry = create_grid_geometry(
        first_data.x,
        first_data.y,
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        coordinate_system='geographic',
        runtime_geometry='geodetic',
        surface_model='sphere',
        velocity_basis='east_north',
    )
    first = geometry.prepare_vector_field(
        first_bounds['lower']['u'],
        first_bounds['lower']['v'],
        cache_key=first_bounds['cache_generation']['lower'],
    )

    reused_u[:] = 9.0
    second_data = build_sedtrails_data()
    second_data.depth_avg_flow_velocity['x'] = reused_u
    second_bounds = FieldDataRetriever(second_data).get_flow_field_bounds(
        2.5,
        'depth_avg_flow_velocity',
    )
    second = geometry.prepare_vector_field(
        second_bounds['lower']['u'],
        second_bounds['lower']['v'],
        cache_key=second_bounds['cache_generation']['lower'],
    )

    assert second_data.forcing_generation > first_data.forcing_generation
    assert (
        first_bounds['cache_generation']['lower']
        != second_bounds['cache_generation']['lower']
    )
    assert second is not first
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.hypot(1.0, 0.5))
    np.testing.assert_allclose(np.linalg.norm(second, axis=1), np.hypot(9.0, 0.5))


def test_scalar_field_bounds_defer_full_grid_interpolation():
    """Checks scalar bounds return lower/upper slices and interpolation weight."""
    retriever = FieldDataRetriever(build_sedtrails_data())

    bounds = retriever.get_scalar_field_bounds(2.5, 'water_depth')

    assert bounds['weight'] == 0.25
    np.testing.assert_allclose(bounds['lower'], [0.0, 1.0, 2.0, 1.0])
    np.testing.assert_allclose(bounds['upper'], [2.0, 3.0, 4.0, 3.0])


def test_scalar_field_bounds_time_independent_field():
    """Checks that a time-independent field (bed_level) returns the full spatial array, not a scalar."""
    retriever = FieldDataRetriever(build_sedtrails_data())

    bounds = retriever.get_scalar_field_bounds(2.5, 'bed_level')

    # bed_level is time-independent (shape (4,)); both bounds should be the full spatial array
    assert bounds['lower'].ndim == 1
    assert len(bounds['lower']) == 4
    np.testing.assert_allclose(bounds['lower'], [0.0, 1.0, 2.0, 1.0])
    np.testing.assert_allclose(bounds['upper'], [0.0, 1.0, 2.0, 1.0])


def test_flow_max_velocity_bound_is_conservative():
    """Verifies max-velocity bound is conservative relative to exact interpolation."""
    retriever = FieldDataRetriever(build_sedtrails_data())
    exact = retriever.get_flow_field(2.5, 'depth_avg_flow_velocity')['magnitude']

    bound = retriever.get_flow_max_velocity_bound(2.5, 'depth_avg_flow_velocity')

    assert bound >= np.nanmax(exact)
    assert bound == 2.5
