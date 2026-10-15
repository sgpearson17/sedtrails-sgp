"""Tests for the XBeach format plugin."""

import numpy as np
import pytest
import xarray as xr

from sedtrails.exceptions.exceptions import ConfigurationError
from sedtrails.particle_tracer.data_retriever import FieldDataRetriever
from sedtrails.particle_tracer.particle_seeder import ParticleSeeder
from sedtrails.transport_converter.format_converter import FormatConverter
from sedtrails.transport_converter.plugins.format import xbeach


def _existing_input_path():
    """Returns a guaranteed-existing file path for plugin constructor inputs."""
    return __file__


def _native_wave_dataset():
    dataset = xr.Dataset(
        data_vars={
            'ua_mean': (('meantime', 'ny', 'nx'), np.full((4, 2, 2), 0.1)),
            'thetamean_mean': (('meantime', 'ny', 'nx'), np.full((4, 2, 2), 180.0)),
            'thetamean': (('globaltime', 'ny', 'nx'), np.full((9, 2, 2), 270.0)),
            'hh': (('globaltime', 'ny', 'nx'), np.ones((9, 2, 2))),
            '_parameters': xr.DataArray(0, attrs={'rotate': 1, 'eps': 0.005, 'tstart': 0.0}),
        },
        coords={
            'globalx': (('ny', 'nx'), [[0.0, 1.0], [0.0, 1.0]]),
            'globaly': (('ny', 'nx'), [[0.0, 0.0], [1.0, 1.0]]),
            'meantime': ('meantime', [10.0, 20.0, 30.0, 40.0]),
            'globaltime': ('globaltime', np.arange(0.0, 41.0, 5.0)),
        },
    )
    dataset.meantime.attrs['units'] = 's'
    dataset.globaltime.attrs['units'] = 's'
    dataset.thetamean.attrs['units'] = 'rad'
    return dataset


def test_native_wave_velocity_uses_snapshot_nautical_directions_not_inconsistent_mean_angle():
    plugin = xbeach.FormatPlugin(_existing_input_path())
    plugin.input_data = _native_wave_dataset()
    converted = plugin.convert(required_fields=('nonlinear_wave_velocity',))
    np.testing.assert_allclose(converted.nonlinear_wave_velocity['x'], 0.1)
    np.testing.assert_allclose(converted.nonlinear_wave_velocity['y'], 0.0, atol=1e-15)
    np.testing.assert_allclose(converted.nonlinear_wave_velocity['magnitude'], 0.1)


def test_native_wave_velocity_converts_all_four_nautical_directions():
    plugin = xbeach.FormatPlugin(_existing_input_path())
    plugin.input_data = _native_wave_dataset()
    plugin.input_data.thetamean.values[:] = [[270.0, 180.0], [90.0, 0.0]]
    result = plugin.convert(required_fields=('nonlinear_wave_velocity',))
    np.testing.assert_allclose(result.nonlinear_wave_velocity['x'], [[0.1, 0, -0.1, 0]] * 4, atol=1e-15)
    np.testing.assert_allclose(result.nonlinear_wave_velocity['y'], [[0, 0.1, 0, -0.1]] * 4, atol=1e-15)
    assert result.metadata.nonlinear_wave_velocity_source == 'ua_mean'
    assert 'approximate' in result.metadata.nonlinear_wave_velocity_averaging


def test_native_wave_direction_average_is_circular_and_time_weighted():
    plugin = xbeach.FormatPlugin(_existing_input_path())
    dataset = _native_wave_dataset()
    dataset.thetamean.values[:3] = np.array([350.0, 0.0, 10.0])[:, None, None]
    plugin.input_data = dataset
    result = plugin.convert(required_fields=('nonlinear_wave_velocity',))
    np.testing.assert_allclose(result.nonlinear_wave_velocity['x'][0], 0, atol=1e-15)
    np.testing.assert_allclose(result.nonlinear_wave_velocity['y'][0], -0.1, atol=1e-15)

    dataset = _native_wave_dataset().assign_coords(globaltime=[0, 1, 10, 11, 20, 21, 30, 31, 40])
    dataset.thetamean.values[1:] = 180
    plugin.input_data = dataset
    result = plugin.convert(required_fields=('nonlinear_wave_velocity',))
    norm = np.hypot(0.5, 9.5)
    np.testing.assert_allclose(result.nonlinear_wave_velocity['x'][0], 0.1 * 0.5 / norm)
    np.testing.assert_allclose(result.nonlinear_wave_velocity['y'][0], 0.1 * 9.5 / norm)


def test_snapshot_average_weights_clip_intervals_and_support_initial_instant():
    times = np.array([0.0, 5.0, 10.0])
    np.testing.assert_allclose(xbeach.FormatPlugin._snapshot_average_weights(times, 2, 8), [0.9, 4.2, 0.9])
    np.testing.assert_allclose(xbeach.FormatPlugin._snapshot_average_weights(times[:2], 2, 2), [0.6, 0.4])
    np.testing.assert_array_equal(xbeach.FormatPlugin._snapshot_average_weights(times[:1], 0, 0), [1])


def test_native_wave_velocity_is_consistent_across_windows_and_morfac():
    plugin = xbeach.FormatPlugin(_existing_input_path()), xbeach.FormatPlugin(_existing_input_path(), morfac=3)
    for reader in plugin:
        reader.input_data = _native_wave_dataset()
        reader.input_data.thetamean.values[:] = np.linspace(245, 295, 9)[:, None, None]
    original = plugin[0].convert(required_fields=('nonlinear_wave_velocity',))
    full = plugin[1].convert(required_fields=('nonlinear_wave_velocity',))
    window = plugin[1].convert(
        current_time=float(full.times[-2]), reading_interval=1, required_fields=('nonlinear_wave_velocity',)
    )
    assert len(window.times) < len(full.times)
    selected = np.searchsorted(full.times, window.times)
    for name in ('x', 'y', 'magnitude'):
        np.testing.assert_allclose(full.nonlinear_wave_velocity[name], original.nonlinear_wave_velocity[name])
        np.testing.assert_allclose(window.nonlinear_wave_velocity[name], full.nonlinear_wave_velocity[name][selected])


def test_native_wave_velocity_skips_fill_mean_times_and_masks_dry_snapshots():
    plugin = xbeach.FormatPlugin(_existing_input_path())
    dataset = _native_wave_dataset()
    dataset = dataset.assign_coords(meantime=[10.0, 20.0, 30.0, 9.96920997e36])
    dataset.hh.values[:, 0, 0] = 0.005
    dataset.hh.values[:5, 0, 1] = np.nan
    dataset.thetamean.values[:5, 0, 1] = np.nan
    dataset.hh.encoding['_FillValue'] = -999.0
    plugin.input_data = dataset
    result = plugin.convert(required_fields=('nonlinear_wave_velocity',))
    np.testing.assert_array_equal(result.times, [10, 20, 30])
    np.testing.assert_array_equal(result.nonlinear_wave_velocity['magnitude'][:, 0], 0)
    np.testing.assert_array_equal(result.nonlinear_wave_velocity['magnitude'][:2, 1], 0)
    np.testing.assert_allclose(result.nonlinear_wave_velocity['x'][2, 1], 0.1)


@pytest.mark.parametrize('missing', ['ua_mean', 'thetamean', 'globaltime', 'hh', '_parameters'])
def test_explicit_wave_request_rejects_missing_source_instead_of_zero_fallback(missing):
    plugin = xbeach.FormatPlugin(_existing_input_path())
    plugin.input_data = _native_wave_dataset().drop_vars(missing)
    with pytest.raises(KeyError, match=missing):
        plugin.convert(required_fields=('nonlinear_wave_velocity',))


@pytest.mark.parametrize('name,value', [('rotate', 0), ('eps', -0.1), ('eps', np.nan), ('tstart', np.nan)])
def test_native_wave_velocity_rejects_invalid_source_parameters(name, value):
    plugin = xbeach.FormatPlugin(_existing_input_path())
    plugin.input_data = _native_wave_dataset()
    plugin.input_data['_parameters'].attrs[name] = value
    with pytest.raises(ValueError, match='rotate|eps|tstart'):
        plugin.convert(required_fields=('nonlinear_wave_velocity',))


@pytest.mark.parametrize('field', ['ua_mean', 'thetamean', 'hh'])
def test_native_wave_velocity_rejects_unencoded_invalid_wet_data(field):
    plugin = xbeach.FormatPlugin(_existing_input_path())
    plugin.input_data = _native_wave_dataset()
    plugin.input_data[field].values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite|Non-finite'):
        plugin.convert(required_fields=('nonlinear_wave_velocity',))


def test_native_wave_velocity_rejects_unbracketed_intervals_and_cancelled_directions():
    plugin = xbeach.FormatPlugin(_existing_input_path())
    plugin.input_data = _native_wave_dataset().assign_coords(globaltime=np.arange(1.0, 42.0, 5.0))
    with pytest.raises(ValueError, match='bracket the full mean interval'):
        plugin.convert(required_fields=('nonlinear_wave_velocity',))
    plugin.input_data = _native_wave_dataset()
    plugin.input_data.thetamean.values[1] = 90
    with pytest.raises(ValueError, match='No well-defined wet snapshot direction'):
        plugin.convert(required_fields=('nonlinear_wave_velocity',))
    plugin.input_data.ua_mean.values[:] = 0
    result = plugin.convert(required_fields=('nonlinear_wave_velocity',))
    np.testing.assert_array_equal(result.nonlinear_wave_velocity['magnitude'], 0)


def test_native_wave_velocity_applies_cutout_mask_to_snapshot_planes():
    plugin = xbeach.FormatPlugin(_existing_input_path())
    plugin.input_data = _native_wave_dataset()
    plugin.input_data.globalx.values[1, 1] = np.nan
    plugin.input_data.hh.values[:, 1, 1] = np.nan
    result = plugin.convert(required_fields=('nonlinear_wave_velocity',))
    assert result.nonlinear_wave_velocity['x'].shape == (4, 3)
    np.testing.assert_allclose(result.nonlinear_wave_velocity['x'], 0.1)


def test_native_wave_velocity_handles_different_cf_time_units():
    plugin = xbeach.FormatPlugin(_existing_input_path())
    dataset = _native_wave_dataset()
    dataset = dataset.assign_coords(globaltime=dataset.globaltime / 3600)
    dataset.globaltime.attrs['units'] = 'hours since 1970-01-01 00:00:00'
    dataset.meantime.attrs['units'] = 'seconds since 1970-01-01 00:00:00'
    plugin.input_data = dataset
    result = plugin.convert(required_fields=('nonlinear_wave_velocity',))
    np.testing.assert_allclose(result.nonlinear_wave_velocity['x'], 0.1)
    np.testing.assert_allclose(result.nonlinear_wave_velocity['y'], 0, atol=1e-15)

def test_xbeach_unstructured_fallback_uses_shared_bounded_helper():
    """One-dimensional XBeach points use the runtime-compatible shared fallback."""
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    plugin = xbeach.FormatPlugin(_existing_input_path())
    plugin.input_data = xr.Dataset(
        coords={
            'globalx': (('point',), x),
            'globaly': (('point',), y),
        }
    )

    triangles = plugin._particle_face_connectivity(x, y)

    assert triangles.shape == (1, 3)
    assert set(triangles[0]) == {0, 1, 2}


def test_xbeach_public_source_plane_estimator_uses_reader_float64_shape():
    """Public byte estimate uses metadata and the reader's float64 outputs."""
    plugin = xbeach.FormatPlugin(_existing_input_path())
    plugin.input_data = xr.Dataset(
        coords={
            'globalx': (('ny', 'nx'), np.zeros((2, 2))),
            'globaly': (('ny', 'nx'), np.zeros((2, 2))),
        }
    )

    estimated = plugin.estimate_source_bytes_per_time_plane(
        ('bed_level', 'depth_avg_flow_velocity')
    )

    assert estimated == 4 * 4 * np.dtype(np.float64).itemsize


def test_xbeach_convert_uses_mean_variables_and_flattens_spatial_dims(monkeypatch):
    """Checks XBeach mean output maps to SedTRAILS dimensions."""
    globalx = np.array([[0.0, 1.0], [0.0, 1.0]])
    globaly = np.array([[0.0, 0.0], [1.0, 1.0]])
    scalar = np.arange(8, dtype=float).reshape(2, 2, 2)
    frac_x = np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    frac_y = frac_x + 10.0

    ds = xr.Dataset(
        data_vars={
            'zb_mean': (('meantime', 'ny', 'nx'), scalar),
            'hh_mean': (('meantime', 'ny', 'nx'), scalar + 1.0),
            'ue_mean': (('meantime', 'ny', 'nx'), np.ones((2, 2, 2))),
            've_mean': (('meantime', 'ny', 'nx'), np.full((2, 2, 2), 2.0)),
            'taubx_mean': (('meantime', 'ny', 'nx'), np.full((2, 2, 2), 3.0)),
            'tauby_mean': (('meantime', 'ny', 'nx'), np.full((2, 2, 2), 4.0)),
            'Subg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), frac_x),
            'Svbg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), frac_y),
            'Susg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), frac_x + 20.0),
            'Svsg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), frac_y + 20.0),
            'cctot_mean': (('meantime', 'ny', 'nx'), scalar + 2.0),
            'ua_mean': (('meantime', 'ny', 'nx'), scalar + 3.0),
            'thetamean_mean': (('meantime', 'ny', 'nx'), np.zeros((2, 2, 2))),
            'thetamean': (('globaltime', 'ny', 'nx'), np.full((3, 2, 2), 270.0)),
            'hh': (('globaltime', 'ny', 'nx'), np.ones((3, 2, 2))),
            '_parameters': xr.DataArray(0, attrs={'rotate': 1, 'eps': 0.005, 'tstart': 0.0}),
        },
        coords={
            'globalx': (('ny', 'nx'), globalx),
            'globaly': (('ny', 'nx'), globaly),
            'meantime': (('meantime',), np.array([10.0, 20.0])),
            'globaltime': (('globaltime',), np.array([0.0, 10.0, 20.0])),
        },
    )
    ds['meantime'].attrs['units'] = 's'
    ds['globalx'].attrs['units'] = 'degrees_east'
    ds['globalx'].attrs['standard_name'] = 'longitude'
    ds['globaly'].attrs['units'] = 'degrees_north'
    ds['globaly'].attrs['standard_name'] = 'latitude'

    def fake_load(self):
        """Injects synthetic XBeach mean-output dataset."""
        self.input_data = ds
        return ds

    def fail_delaunay(*_args, **_kwargs):
        """Fails if structured XBeach grids fall back to Delaunay connectivity."""
        raise AssertionError('Structured XBeach grids should not use Delaunay connectivity')

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)
    monkeypatch.setattr(xbeach, 'delaunay_connectivity', fail_delaunay)

    plugin = xbeach.FormatPlugin(_existing_input_path())
    sedtrails_data = plugin.convert(reference_date=np.datetime64('1970-01-01T00:00:00'))

    expected_connectivity = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int64)
    np.testing.assert_array_equal(sedtrails_data.times, np.array([10.0, 20.0]))
    np.testing.assert_array_equal(sedtrails_data.x, np.array([0.0, 1.0, 0.0, 1.0]))
    assert sedtrails_data.bed_level.shape == (2, 4)
    assert sedtrails_data.water_depth.shape == (2, 4)
    assert sedtrails_data.bed_load_transport['x'].shape == (2, 4)
    assert sedtrails_data.suspended_transport['y'].shape == (2, 4)
    assert sedtrails_data.fractions == 1
    assert sedtrails_data.metadata.source_sediment_classes == 2
    assert sedtrails_data.metadata.coordinate_system == 'geographic'
    assert sedtrails_data.metadata.sediment_transport_fraction_handling == 'sum_over_source_sediment_classes'
    np.testing.assert_array_equal(sedtrails_data.face_node_connectivity, expected_connectivity)
    np.testing.assert_array_equal(sedtrails_data.bed_load_transport['x'], np.array([[4.0, 6.0, 8.0, 10.0], [20.0, 22.0, 24.0, 26.0]]))
    np.testing.assert_array_equal(sedtrails_data.bed_load_transport['y'], np.array([[24.0, 26.0, 28.0, 30.0], [40.0, 42.0, 44.0, 46.0]]))
    np.testing.assert_allclose(sedtrails_data.mean_bed_shear_stress, 5.0)
    np.testing.assert_allclose(sedtrails_data.max_bed_shear_stress, 5.0)
    np.testing.assert_array_equal(sedtrails_data.nonlinear_wave_velocity['x'], (scalar + 3.0).reshape(2, 4))


def test_xbeach_convert_skips_numeric_fill_meantime_rows(monkeypatch):
    """Checks XBeach fill values in ``meantime`` do not enter conversion."""
    globalx = np.array([[0.0, 1.0], [0.0, 1.0]])
    globaly = np.array([[0.0, 0.0], [1.0, 1.0]])
    num_valid_times = 13
    num_raw_times = num_valid_times + 1
    scalar = np.arange(num_raw_times * 4, dtype=float).reshape(num_raw_times, 2, 2)
    transport = np.ones((num_raw_times, 1, 2, 2), dtype=float)

    ds = xr.Dataset(
        data_vars={
            'zb_mean': (('meantime', 'ny', 'nx'), scalar),
            'hh_mean': (('meantime', 'ny', 'nx'), scalar + 1.0),
            'ue_mean': (('meantime', 'ny', 'nx'), scalar + 2.0),
            've_mean': (('meantime', 'ny', 'nx'), scalar + 3.0),
            'taubx_mean': (('meantime', 'ny', 'nx'), np.full((num_raw_times, 2, 2), 3.0)),
            'tauby_mean': (('meantime', 'ny', 'nx'), np.full((num_raw_times, 2, 2), 4.0)),
            'Subg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), transport),
            'Svbg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), transport + 1.0),
            'Susg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), transport + 2.0),
            'Svsg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), transport + 3.0),
            'cctot_mean': (('meantime', 'ny', 'nx'), scalar + 4.0),
            'ua_mean': (('meantime', 'ny', 'nx'), scalar + 5.0),
            'thetamean_mean': (('meantime', 'ny', 'nx'), np.zeros((num_raw_times, 2, 2))),
        },
        coords={
            'globalx': (('ny', 'nx'), globalx),
            'globaly': (('ny', 'nx'), globaly),
            'meantime': (('meantime',), np.append(np.arange(num_valid_times, dtype=float) * 10.0, 9.96920997e36)),
        },
    )
    ds['meantime'].attrs['units'] = 's'

    def fake_load(self):
        """Injects synthetic XBeach mean-output dataset with a fill time row."""
        self.input_data = ds
        return ds

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)

    plugin = xbeach.FormatPlugin(_existing_input_path())
    assert plugin.get_time_bounds(reference_date=np.datetime64('1970-01-01T00:00:00')) == (0.0, 120.0)

    sedtrails_data = plugin.convert(reference_date=np.datetime64('1970-01-01T00:00:00'))

    np.testing.assert_array_equal(sedtrails_data.times, np.arange(num_valid_times, dtype=float) * 10.0)
    np.testing.assert_array_equal(sedtrails_data.bed_level, scalar[:num_valid_times].reshape(num_valid_times, 4))
    assert sedtrails_data.bed_level.shape == (num_valid_times, 4)

    chunked_data = plugin.convert(
        current_time=100.0,
        reading_interval=10.0,
        reference_date=np.datetime64('1970-01-01T00:00:00'),
    )

    np.testing.assert_array_equal(chunked_data.times, np.array([100.0, 110.0]))
    np.testing.assert_array_equal(chunked_data.bed_level, scalar[10:12].reshape(2, 4))


def test_xbeach_get_time_bounds_skips_declared_fill_values(monkeypatch):
    """Checks declared time fill values are not treated as valid mean times."""
    ds = xr.Dataset(coords={'meantime': (('meantime',), np.array([-999.0, 0.0, 10.0]))})
    ds['meantime'].attrs['units'] = 's'
    ds['meantime'].attrs['missing_value'] = -999.0

    def fake_load(self):
        """Injects synthetic XBeach mean-time metadata."""
        self.input_data = ds
        return ds

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)

    plugin = xbeach.FormatPlugin(_existing_input_path())

    assert plugin.get_time_bounds(reference_date=np.datetime64('1970-01-01T00:00:00')) == (0.0, 10.0)


def test_xbeach_get_time_bounds_rejects_all_invalid_meantime(monkeypatch):
    """Checks all-invalid XBeach mean times fail with a clear error."""
    ds = xr.Dataset(coords={'meantime': (('meantime',), np.array([9.96920997e36]))})
    ds['meantime'].attrs['units'] = 's'

    def fake_load(self):
        """Injects synthetic XBeach mean-time fill data."""
        self.input_data = ds
        return ds

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)

    plugin = xbeach.FormatPlugin(_existing_input_path())

    with pytest.raises(ValueError, match='no valid mean time values'):
        plugin.get_time_bounds(reference_date=np.datetime64('1970-01-01T00:00:00'))


def test_xbeach_cutout_coordinates_filter_fields_and_connectivity(monkeypatch):
    """Checks XBeach cutout points are removed from fields and mesh connectivity."""
    globalx = np.array([[0.0, np.nan, 2.0], [0.0, 1.0, 2.0]])
    globaly = np.array([[0.0, np.nan, 0.0], [1.0, 1.0, 1.0]])
    scalar = np.arange(6, dtype=float).reshape(1, 2, 3)
    transport = np.ones((1, 1, 2, 3), dtype=float)

    ds = xr.Dataset(
        data_vars={
            'zb_mean': (('meantime', 'ny', 'nx'), scalar),
            'hh_mean': (('meantime', 'ny', 'nx'), scalar + 1.0),
            'ue_mean': (('meantime', 'ny', 'nx'), scalar + 2.0),
            've_mean': (('meantime', 'ny', 'nx'), scalar + 3.0),
            'taubx_mean': (('meantime', 'ny', 'nx'), np.full((1, 2, 3), 3.0)),
            'tauby_mean': (('meantime', 'ny', 'nx'), np.full((1, 2, 3), 4.0)),
            'Subg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), transport),
            'Svbg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), transport + 1.0),
            'Susg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), transport + 2.0),
            'Svsg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), transport + 3.0),
            'cctot_mean': (('meantime', 'ny', 'nx'), scalar + 4.0),
        },
        coords={
            'globalx': (('ny', 'nx'), globalx),
            'globaly': (('ny', 'nx'), globaly),
            'meantime': (('meantime',), np.array([0.0])),
        },
    )

    def fake_load(self):
        """Injects synthetic XBeach grid coordinates with a cutout point."""
        self.input_data = ds
        return ds

    def fail_delaunay(*_args, **_kwargs):
        """Fails if structured XBeach grids fall back to Delaunay connectivity."""
        raise AssertionError('Structured XBeach grids should not use Delaunay connectivity')

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)
    monkeypatch.setattr(xbeach, 'delaunay_connectivity', fail_delaunay)

    plugin = xbeach.FormatPlugin(_existing_input_path())
    sedtrails_data = plugin.convert()
    field_data = plugin.get_seeding_field_data()

    expected_x = np.array([0.0, 2.0, 0.0, 1.0, 2.0])
    expected_y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    expected_connectivity = np.array([[0, 3, 2]], dtype=np.int64)

    np.testing.assert_array_equal(sedtrails_data.x, expected_x)
    np.testing.assert_array_equal(sedtrails_data.y, expected_y)
    np.testing.assert_array_equal(sedtrails_data.bed_level, np.array([[0.0, 2.0, 3.0, 4.0, 5.0]]))
    np.testing.assert_array_equal(sedtrails_data.depth_avg_flow_velocity['x'], np.array([[2.0, 4.0, 5.0, 6.0, 7.0]]))
    np.testing.assert_array_equal(sedtrails_data.face_node_connectivity, expected_connectivity)
    np.testing.assert_array_equal(field_data.x, expected_x)
    np.testing.assert_array_equal(field_data.y, expected_y)
    np.testing.assert_array_equal(field_data.face_node_connectivity, expected_connectivity)
    assert np.isfinite(field_data.x).all()
    assert np.isfinite(field_data.y).all()


def test_xbeach_empty_cutout_connectivity_does_not_fall_back_to_delaunay(monkeypatch):
    """Checks cutout-only cells stay outside instead of being Delaunay-filled."""
    ds = xr.Dataset(
        coords={
            'globalx': (('ny', 'nx'), np.array([[0.0, 1.0], [0.0, np.nan]])),
            'globaly': (('ny', 'nx'), np.array([[0.0, 0.0], [1.0, np.nan]])),
        },
    )

    def fake_load(self):
        """Injects a grid where no complete structured triangle survives."""
        self.input_data = ds
        return ds

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)

    field_data = xbeach.FormatPlugin(_existing_input_path()).get_seeding_field_data()

    np.testing.assert_array_equal(field_data.particle_face_connectivity, np.empty((0, 3), dtype=np.int64))
    config = {
        'name': 'cutout-only population',
        'particle_type': 'sand',
        'transport_probability': 'no_probability',
        'seeding': {
            'strategy': {'point': {'locations': ['0.5,0.5']}},
            'quantity': 1,
            'release_start': '0',
            'burial_depth': {'constant': 0.0},
        },
    }
    with pytest.raises(ConfigurationError, match='outside the input field domain'):
        ParticleSeeder([config]).seed(field_data)


def test_format_converter_get_seeding_field_data_uses_xbeach_geometry_reader(monkeypatch):
    """Checks FormatConverter uses the XBeach geometry fast path."""
    ds = xr.Dataset(
        coords={
            'globalx': (('ny', 'nx'), np.array([[0.0, 1.0], [0.0, 1.0]])),
            'globaly': (('ny', 'nx'), np.array([[0.0, 0.0], [1.0, 1.0]])),
        },
    )

    def fake_load(self):
        """Injects synthetic XBeach grid coordinates."""
        self.input_data = ds
        return ds

    def fail_convert(self, *_args, **_kwargs):
        """Fails if seeding falls back to full conversion."""
        raise RuntimeError('XBeach seeding should use get_seeding_coordinates')

    def fail_delaunay(*_args, **_kwargs):
        """Fails if structured XBeach grids fall back to Delaunay connectivity."""
        raise AssertionError('Structured XBeach grids should not use Delaunay connectivity')

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)
    monkeypatch.setattr(xbeach.FormatPlugin, 'convert', fail_convert)
    monkeypatch.setattr(xbeach, 'delaunay_connectivity', fail_delaunay)

    converter = FormatConverter(
        {'input_file': _existing_input_path(), 'input_format': 'xbeach', 'reference_date': '2000-01-01'}
    )
    field_data = converter.get_seeding_field_data()

    expected_connectivity = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int64)
    np.testing.assert_array_equal(field_data.x, np.array([0.0, 1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(field_data.y, np.array([0.0, 0.0, 1.0, 1.0]))
    np.testing.assert_array_equal(field_data.face_node_connectivity, expected_connectivity)
    np.testing.assert_array_equal(field_data.particle_face_connectivity, expected_connectivity)
    assert field_data.reference_date == np.datetime64('2000-01-01')


def test_xbeach_get_scalar_field_slices_time_dependent_bed_level(monkeypatch):
    """Verifies dynamic XBeach bed levels are exposed as time slices."""
    ds = xr.Dataset(
        data_vars={
            'zb_mean': (('meantime', 'ny', 'nx'), np.arange(8, dtype=float).reshape(2, 2, 2)),
            'hh_mean': (('meantime', 'ny', 'nx'), np.ones((2, 2, 2))),
            'ue_mean': (('meantime', 'ny', 'nx'), np.ones((2, 2, 2))),
            've_mean': (('meantime', 'ny', 'nx'), np.zeros((2, 2, 2))),
            'taubx_mean': (('meantime', 'ny', 'nx'), np.ones((2, 2, 2))),
            'tauby_mean': (('meantime', 'ny', 'nx'), np.zeros((2, 2, 2))),
            'Subg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), np.ones((2, 1, 2, 2))),
            'Svbg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), np.zeros((2, 1, 2, 2))),
            'Susg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), np.ones((2, 1, 2, 2))),
            'Svsg_mean': (('meantime', 'sediment_classes', 'ny', 'nx'), np.zeros((2, 1, 2, 2))),
            'cctot_mean': (('meantime', 'ny', 'nx'), np.ones((2, 2, 2))),
        },
        coords={
            'globalx': (('ny', 'nx'), np.array([[0.0, 1.0], [0.0, 1.0]])),
            'globaly': (('ny', 'nx'), np.array([[0.0, 0.0], [1.0, 1.0]])),
            'meantime': (('meantime',), np.array([0.0, 10.0])),
        },
    )

    def fake_load(self):
        """Injects synthetic XBeach dataset with changing bed levels."""
        self.input_data = ds
        return ds

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)

    sedtrails_data = xbeach.FormatPlugin(_existing_input_path()).convert()
    retriever = FieldDataRetriever(sedtrails_data)

    field = retriever.get_scalar_field(10.0, 'bed_level')

    np.testing.assert_array_equal(field['magnitude'], np.array([4.0, 5.0, 6.0, 7.0]))


def test_xbeach_transport_component_without_fraction_dim_is_not_summed(monkeypatch):
    """Checks transport data shaped (time, y, x) is flattened, not summed over y."""
    transport = np.arange(8, dtype=float).reshape(2, 2, 2)
    ds = xr.Dataset(
        data_vars={'Subg_mean': (('meantime', 'ny', 'nx'), transport)},
        coords={'meantime': (('meantime',), np.array([0.0, 10.0]))},
    )

    def fake_load(self):
        """Injects a synthetic transport dataset without a fraction dimension."""
        self.input_data = ds
        return ds

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)

    plugin = xbeach.FormatPlugin(_existing_input_path())
    plugin.load()
    total_transport, source_sediment_classes = plugin._mean_transport_component(
        'Subg_mean',
        slice(None),
        num_times=2,
        grid_size=4,
    )

    assert source_sediment_classes == 1
    np.testing.assert_array_equal(total_transport, transport.reshape(2, 4))


def test_selective_xbeach_window_is_byte_bounded_and_bracketed(monkeypatch):
    """Selected XBeach fields should fit a bounded two-plane time window."""
    bed_level = np.arange(24, dtype=float).reshape(6, 2, 2)
    dataset = xr.Dataset(
        data_vars={
            'zb_mean': (('meantime', 'ny', 'nx'), bed_level),
        },
        coords={
            'globalx': (('ny', 'nx'), np.array([[0.0, 1.0], [0.0, 1.0]])),
            'globaly': (('ny', 'nx'), np.array([[0.0, 0.0], [1.0, 1.0]])),
            'meantime': (('meantime',), np.arange(0.0, 60.0, 10.0)),
        },
    )

    def fake_load(self):
        self.input_data = dataset
        return dataset

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)
    data = xbeach.FormatPlugin(_existing_input_path()).convert(
        current_time=25.0,
        reading_interval=100.0,
        required_fields=('bed_level', 'derived_runtime_field'),
        max_memory_bytes=64,
    )

    np.testing.assert_array_equal(data.times, np.array([20.0, 30.0]))
    np.testing.assert_array_equal(data.bed_level, bed_level[2:4].reshape(2, 4))
    assert data.depth_avg_flow_velocity is None
    assert data.bed_load_transport is None
    assert data.suspended_transport is None
    assert data.water_depth is None
    assert data.mean_bed_shear_stress is None
    assert data.max_bed_shear_stress is None
    assert data.sediment_concentration is None
    assert data.nonlinear_wave_velocity is None


def test_xbeach_memory_error_precedes_field_materialization(monkeypatch):
    """XBeach should reject a budget that cannot hold two required planes."""
    dataset = xr.Dataset(
        coords={
            'globalx': (('ny', 'nx'), np.array([[0.0, 1.0], [0.0, 1.0]])),
            'globaly': (('ny', 'nx'), np.array([[0.0, 0.0], [1.0, 1.0]])),
            'meantime': (('meantime',), np.arange(0.0, 40.0, 10.0)),
        },
    )

    def fake_load(self):
        self.input_data = dataset
        return dataset

    def fail_mapping(*_args, **_kwargs):
        raise AssertionError('field materialization must not start')

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)
    monkeypatch.setattr(xbeach.FormatPlugin, '_map_xbeach_variables', fail_mapping)

    with pytest.raises(MemoryError, match=r'64 bytes.*max_memory_bytes=63'):
        xbeach.FormatPlugin(_existing_input_path()).convert(
            current_time=15.0,
            reading_interval=10.0,
            required_fields=('bed_level',),
            max_memory_bytes=63,
        )


def test_xbeach_structured_topology_is_compact_reused_and_not_filtered(monkeypatch):
    """XBeach should retain the all-active int32 candidate table by identity."""
    dataset = xr.Dataset(
        coords={
            'globalx': (('ny', 'nx'), np.array([[0.0, 1.0], [0.0, 1.0]])),
            'globaly': (('ny', 'nx'), np.array([[0.0, 0.0], [1.0, 1.0]])),
        }
    )

    def fake_load(self):
        self.input_data = dataset
        return dataset

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)
    plugin = xbeach.FormatPlugin(_existing_input_path())
    x, y = plugin.get_seeding_coordinates()
    original_builder = plugin._structured_grid_connectivity
    build_count = 0

    def counted_builder():
        nonlocal build_count
        build_count += 1
        return original_builder()

    monkeypatch.setattr(plugin, '_structured_grid_connectivity', counted_builder)
    first = plugin._particle_face_connectivity(x, y)
    second = plugin._particle_face_connectivity(x.copy(), y.copy())

    assert first is second
    assert first.dtype == np.int32
    assert build_count == 1
    assert plugin._filter_active_triangles(first) is first
