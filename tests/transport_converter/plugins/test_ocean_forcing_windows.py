"""Tests for selective, byte-bounded FM and SFINCS forcing windows."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sedtrails.transport_converter.plugins.format import fm_netcdf, sfincs


def _forcing_times() -> np.ndarray:
    """Return six ten-second forcing timestamps."""
    return (
        np.arange(0, 60, 10).astype('timedelta64[s]')
        + np.datetime64('1970-01-01')
    )


def _fm_dataset() -> xr.Dataset:
    """Return a small FM forcing dataset with explicit triangular topology."""
    water_depth = np.arange(24, dtype=float).reshape(6, 4)
    dataset = xr.Dataset(
        data_vars={
            'net_xcc': (('face',), np.array([0.0, 1.0, 1.0, 0.0])),
            'net_ycc': (('face',), np.array([0.0, 0.0, 1.0, 1.0])),
            'NetElemNode': (
                ('element', 'nmax'),
                np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            ),
            'waterdepth': (('time', 'face'), water_depth),
            'bedlevel': (('time', 'face'), -water_depth),
        },
        coords={'time': _forcing_times()},
    )
    dataset['NetElemNode'].attrs['start_index'] = 0
    return dataset


def _sfincs_dataset() -> xr.Dataset:
    """Return a small SFINCS forcing dataset with shared-node topology."""
    water_depth = np.arange(24, dtype=float).reshape(6, 4)
    dataset = xr.Dataset(
        data_vars={
            'mesh2d_node_x': (
                ('node',),
                np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            ),
            'mesh2d_node_y': (
                ('node',),
                np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
            ),
            'mesh2d_face_nodes': (
                ('face', 'nmax'),
                np.array(
                    [
                        [0, 1, 4, 3],
                        [1, 2, 5, 4],
                        [3, 4, 7, 6],
                        [4, 5, 8, 7],
                    ],
                    dtype=np.int32,
                ),
            ),
            'h': (('time', 'face'), water_depth),
            'zb': (('time', 'face'), -water_depth),
        },
        coords={'time': _forcing_times()},
    )
    dataset['mesh2d_face_nodes'].attrs['start_index'] = 0
    dataset['mesh2d_face_nodes'].encoding['_FillValue'] = -1
    return dataset


@pytest.mark.parametrize(
    ('plugin_module', 'dataset_factory', 'water_variable'),
    [
        (fm_netcdf, _fm_dataset, 'waterdepth'),
        (sfincs, _sfincs_dataset, 'h'),
    ],
)
def test_selective_window_keeps_two_plane_bracket(
    tmp_path: Path,
    plugin_module,
    dataset_factory,
    water_variable,
) -> None:
    """Only selected water depth is retained inside a two-plane byte budget."""
    input_file = tmp_path / f'{plugin_module.__name__.split(".")[-1]}.nc'
    input_file.touch()
    dataset = dataset_factory()
    plugin = plugin_module.FormatPlugin(str(input_file))
    plugin.input_data = dataset

    data = plugin.convert(
        current_time=25.0,
        reading_interval=100.0,
        required_fields=('water_depth', 'derived_runtime_field'),
        max_memory_bytes=64,
    )

    np.testing.assert_array_equal(data.times, np.array([20.0, 30.0]))
    np.testing.assert_array_equal(
        data.water_depth,
        dataset[water_variable].values[2:4],
    )
    assert data.bed_level is None
    assert data.depth_avg_flow_velocity is None
    assert data.bed_load_transport is None
    assert data.suspended_transport is None
    assert data.mean_bed_shear_stress is None
    assert data.max_bed_shear_stress is None
    assert data.sediment_concentration is None
    assert data.nonlinear_wave_velocity is None


@pytest.mark.parametrize(
    ('plugin_module', 'dataset_factory', 'mapper_name'),
    [
        (fm_netcdf, _fm_dataset, '_map_dfm_variables'),
        (sfincs, _sfincs_dataset, '_map_sfincs_variables'),
    ],
)
def test_memory_error_precedes_forcing_materialization(
    tmp_path: Path,
    monkeypatch,
    plugin_module,
    dataset_factory,
    mapper_name,
) -> None:
    """A budget below two selected planes fails before mapping starts."""
    input_file = tmp_path / f'{plugin_module.__name__.split(".")[-1]}_small.nc'
    input_file.touch()
    plugin = plugin_module.FormatPlugin(str(input_file))
    plugin.input_data = dataset_factory()

    def fail_mapping(*_args, **_kwargs):
        raise AssertionError('forcing materialization must not start')

    monkeypatch.setattr(plugin, mapper_name, fail_mapping)

    with pytest.raises(MemoryError, match=r'64 bytes.*max_memory_bytes=63'):
        plugin.convert(
            current_time=25.0,
            reading_interval=10.0,
            required_fields=('water_depth',),
            max_memory_bytes=63,
        )


@pytest.mark.parametrize(
    ('plugin_module', 'dataset_factory', 'x_name', 'y_name'),
    [
        (
            fm_netcdf,
            _fm_dataset,
            'sea_water_x_velocity',
            'sea_water_y_velocity',
        ),
        (sfincs, _sfincs_dataset, 'u', 'v'),
    ],
)
def test_integer_vector_estimator_includes_hypot_promotion(
    tmp_path: Path,
    plugin_module,
    dataset_factory,
    x_name,
    y_name,
) -> None:
    """Int16 components retain a float32 magnitude in estimates and caps."""
    input_file = tmp_path / f'{plugin_module.__name__.split(".")[-1]}_int16.nc'
    input_file.touch()
    dataset = dataset_factory()
    vector = np.arange(24, dtype=np.int16).reshape(6, 4)
    dataset[x_name] = (('time', 'face'), vector)
    dataset[y_name] = (('time', 'face'), vector)
    plugin = plugin_module.FormatPlugin(str(input_file))
    plugin.input_data = dataset

    estimated = plugin.estimate_source_bytes_per_time_plane(
        ('depth_avg_flow_velocity',)
    )

    assert estimated == 32
    with pytest.raises(MemoryError, match=r'64 bytes.*max_memory_bytes=63'):
        plugin.convert(
            current_time=25.0,
            reading_interval=10.0,
            required_fields=('depth_avg_flow_velocity',),
            max_memory_bytes=63,
        )


def test_sfincs_load_uses_lazy_xugrid_open(tmp_path: Path, monkeypatch) -> None:
    """SFINCS opens UGRID lazily instead of loading every array eagerly."""
    input_file = tmp_path / 'lazy_sfincs.nc'
    input_file.touch()
    sentinel = object()
    calls = []

    def fake_open(path, **kwargs):
        calls.append((path, kwargs))
        return sentinel

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError('xu.load_dataset must not be used')

    monkeypatch.setattr(sfincs.xu, 'open_dataset', fake_open)
    monkeypatch.setattr(sfincs.xu, 'load_dataset', unexpected_load)

    plugin = sfincs.FormatPlugin(str(input_file))
    plugin.load()

    assert plugin.input_data is sentinel
    assert calls[0][0] == input_file
    assert calls[0][1]['decode_times'] is True
