"""Tests for the format converter."""

import numpy as np
import pytest
import xarray as xr

from sedtrails.transport_converter.format_converter import FormatConverter
from sedtrails.transport_converter.plugins.format import sfincs


class _PluginWithCoordinateReader:
    def __init__(self):
        self.convert_called = False

    def get_seeding_coordinates(self):
        return [1.0, 2.0], [3.0, 4.0]

    def convert(self, *_args, **_kwargs):
        self.convert_called = True
        raise RuntimeError('convert should not be called when get_seeding_coordinates is available')


class _PluginWithConvertOnly:
    class _SedtrailsLike:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def convert(self, *_args, **_kwargs):
        return self._SedtrailsLike(x=(10.0, 20.0), y=(30.0, 40.0))


def test_get_seeding_field_data_prefers_coordinate_reader():
    converter = FormatConverter({'input_file': 'dummy.nc', 'input_format': 'dummy'})
    plugin = _PluginWithCoordinateReader()
    converter._format_plugin = plugin

    field_data = converter.get_seeding_field_data()

    np.testing.assert_array_equal(field_data.x, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(field_data.y, np.array([3.0, 4.0]))
    assert not plugin.convert_called


def test_get_seeding_field_data_falls_back_to_convert():
    converter = FormatConverter({'input_file': 'dummy.nc', 'input_format': 'dummy'})
    converter._format_plugin = _PluginWithConvertOnly()

    field_data = converter.get_seeding_field_data()

    np.testing.assert_array_equal(field_data.x, np.array([10.0, 20.0]))
    np.testing.assert_array_equal(field_data.y, np.array([30.0, 40.0]))


def test_sfincs_get_seeding_coordinates_uses_ugrid_face_coordinates(monkeypatch):
    class FakeUgrid2d:
        def __init__(self):
            self.face_x = np.array([1.0, 2.0, 3.0])
            self.face_y = np.array([4.0, 5.0, 6.0])

    class FakeUgridDataset:
        def __init__(self):
            self.grid = FakeUgrid2d()

    def fake_load(self):
        self.input_data = FakeUgridDataset()

    monkeypatch.setattr(sfincs.xu, 'UgridDataset', FakeUgridDataset)
    monkeypatch.setattr(sfincs.xu, 'Ugrid2d', FakeUgrid2d)
    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin('dummy.nc')
    x, y = plugin.get_seeding_coordinates()

    np.testing.assert_array_equal(x, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(y, np.array([4.0, 5.0, 6.0]))


def test_sfincs_get_seeding_coordinates_fallback_computes_centroids(monkeypatch):
    ds = xr.Dataset(
        data_vars={
            'mesh2d_node_x': (('node',), np.array([0.0, 2.0, 2.0, 0.0])),
            'mesh2d_node_y': (('node',), np.array([0.0, 0.0, 2.0, 2.0])),
            'mesh2d_face_nodes': (('face', 'nmax'), np.array([[1, 2, 3, 4]], dtype=np.int64)),
        }
    )
    ds['mesh2d_face_nodes'].attrs['start_index'] = 1
    ds['mesh2d_face_nodes'].encoding['_FillValue'] = -999

    def fake_load(self):
        self.input_data = ds

    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin('dummy.nc')
    x, y = plugin.get_seeding_coordinates()

    np.testing.assert_allclose(x, np.array([1.0]))
    np.testing.assert_allclose(y, np.array([1.0]))


def test_sfincs_convert_stores_face_node_mesh_geometry(monkeypatch):
    ds = xr.Dataset(
        data_vars={
            'mesh2d_node_x': (('node',), np.array([0.0, 2.0, 2.0, 0.0])),
            'mesh2d_node_y': (('node',), np.array([0.0, 0.0, 2.0, 2.0])),
            'mesh2d_face_nodes': (('face', 'nmax'), np.array([[1, 2, 3, 4]], dtype=np.int64)),
            'zb': (('face',), np.array([-1.0])),
            'h': (('time', 'face'), np.array([[1.0], [2.0]])),
            'u': (('time', 'face'), np.array([[0.1], [0.2]])),
            'v': (('time', 'face'), np.array([[0.0], [0.0]])),
        },
        coords={'time': np.array(['2024-01-01T00:00:00', '2024-01-01T00:01:00'], dtype='datetime64[ns]')},
    )
    ds['mesh2d_face_nodes'].attrs['start_index'] = 1
    ds['mesh2d_face_nodes'].encoding['_FillValue'] = -999

    def fake_load(self):
        self.input_data = ds

    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin('dummy.nc')
    sedtrails_data = plugin.convert(reference_date=np.datetime64('2024-01-01T00:00:00'))

    np.testing.assert_array_equal(sedtrails_data.node_x, np.array([0.0, 2.0, 2.0, 0.0]))
    np.testing.assert_array_equal(sedtrails_data.node_y, np.array([0.0, 0.0, 2.0, 2.0]))
    np.testing.assert_array_equal(sedtrails_data.face_node_connectivity, np.array([[0, 1, 2, 3]]))
    assert sedtrails_data.face_node_fill_value == -1
    assert sedtrails_data.mesh_geometry()['face_node_connectivity'].shape == (1, 4)


def test_sfincs_get_seeding_coordinates_raises_on_non_ugrid2d(monkeypatch):
    class FakeNonUgrid2d:
        pass

    class FakeUgridDataset:
        def __init__(self):
            self.grid = FakeNonUgrid2d()

    def fake_load(self):
        self.input_data = FakeUgridDataset()

    monkeypatch.setattr(sfincs.xu, 'UgridDataset', FakeUgridDataset)
    monkeypatch.setattr(sfincs.xu, 'Ugrid2d', type('OtherGrid', (), {}))
    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin('dummy.nc')

    with pytest.raises(TypeError, match='Expected Ugrid2d'):
        plugin.get_seeding_coordinates()
