"""Tests for the XBeach format plugin."""

import numpy as np
import xarray as xr

from sedtrails.particle_tracer.data_retriever import FieldDataRetriever
from sedtrails.transport_converter.plugins.format import xbeach


def _existing_input_path():
    """Returns a guaranteed-existing file path for plugin constructor inputs."""
    return __file__


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
        },
        coords={
            'globalx': (('ny', 'nx'), globalx),
            'globaly': (('ny', 'nx'), globaly),
            'meantime': (('meantime',), np.array([10.0, 20.0])),
            'globaltime': (('globaltime',), np.array([0.0, 10.0, 20.0])),
        },
    )
    ds['meantime'].attrs['units'] = 's'

    def fake_load(self):
        """Injects synthetic XBeach mean-output dataset."""
        self.input_data = ds
        return ds

    monkeypatch.setattr(xbeach.FormatPlugin, 'load', fake_load)

    plugin = xbeach.FormatPlugin(_existing_input_path())
    sedtrails_data = plugin.convert(reference_date=np.datetime64('1970-01-01T00:00:00'))

    np.testing.assert_array_equal(sedtrails_data.times, np.array([10.0, 20.0]))
    np.testing.assert_array_equal(sedtrails_data.x, np.array([0.0, 1.0, 0.0, 1.0]))
    assert sedtrails_data.bed_level.shape == (2, 4)
    assert sedtrails_data.water_depth.shape == (2, 4)
    assert sedtrails_data.bed_load_transport['x'].shape == (2, 4)
    assert sedtrails_data.suspended_transport['y'].shape == (2, 4)
    assert sedtrails_data.fractions == 1
    assert sedtrails_data.metadata.source_sediment_classes == 2
    assert sedtrails_data.metadata.sediment_transport_fraction_handling == 'sum_over_source_sediment_classes'
    np.testing.assert_array_equal(sedtrails_data.bed_load_transport['x'], np.array([[4.0, 6.0, 8.0, 10.0], [20.0, 22.0, 24.0, 26.0]]))
    np.testing.assert_array_equal(sedtrails_data.bed_load_transport['y'], np.array([[24.0, 26.0, 28.0, 30.0], [40.0, 42.0, 44.0, 46.0]]))
    np.testing.assert_allclose(sedtrails_data.mean_bed_shear_stress, 5.0)
    np.testing.assert_allclose(sedtrails_data.max_bed_shear_stress, 5.0)
    np.testing.assert_array_equal(sedtrails_data.nonlinear_wave_velocity['x'], (scalar + 3.0).reshape(2, 4))


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
