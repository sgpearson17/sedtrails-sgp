"""Tests for the Delft3D4 NetCDF format converter."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sedtrails.transport_converter.plugins.format import d3d4_netcdf


SAMPLE_FILE = Path('sample-data/trim-inlet.nc')


def _plot_if_requested(
    data,
    output_dir: Path,
    time_idx: int,
    mask: np.ndarray | None,
) -> None:
    """Write optional diagnostic plots when SEDTRAILS_PLOT_DIR is configured."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.asarray(data.x)
    y = np.asarray(data.y)
    scalar_x, scalar_y = _resolve_scalar_coordinates(data)
    time_label = _format_time_label(data, time_idx)

    if x.ndim == 2 and y.ndim == 2:
        invalid_mask = (x == 0) & (y == 0)
        if mask is not None:
            invalid_mask = invalid_mask | mask
        stride_x = max(1, x.shape[0] // 25)
        stride_y = max(1, x.shape[1] // 25)
        xs = x[::stride_x, ::stride_y]
        ys = y[::stride_x, ::stride_y]
        u = data.depth_avg_flow_velocity['x'][time_idx][::stride_x, ::stride_y]
        v = data.depth_avg_flow_velocity['y'][time_idx][::stride_x, ::stride_y]
        subsampled_mask = invalid_mask[::stride_x, ::stride_y]
        u = np.where(subsampled_mask, np.nan, u)
        v = np.where(subsampled_mask, np.nan, v)
        plt.figure(figsize=(7, 6))
        plt.quiver(xs, ys, u, v, scale=10)
        plt.title(f'Depth-averaged flow velocity ({time_label})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(output_dir / 'flow_velocity_vectors.png', dpi=150)
        plt.close()
    elif x.ndim == 1 and y.ndim == 1:
        u = data.depth_avg_flow_velocity['x'][time_idx]
        v = data.depth_avg_flow_velocity['y'][time_idx]
        stride = max(1, x.size // 2500)
        xs = x[::stride]
        ys = y[::stride]
        u = u[::stride]
        v = v[::stride]
        plt.figure(figsize=(7, 6))
        plt.quiver(xs, ys, u, v, scale=10)
        plt.title(f'Depth-averaged flow velocity ({time_label})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(output_dir / 'flow_velocity_vectors.png', dpi=150)
        plt.close()

    _plot_scalar_field(
        scalar_x,
        scalar_y,
        data.mean_bed_shear_stress[time_idx],
        output_dir / 'mean_bed_shear_stress.png',
        title=f'Mean bed shear stress ({time_label})',
        colorbar_label='Pa',
        mask=mask,
    )

    _plot_scalar_field(
        scalar_x,
        scalar_y,
        data.max_bed_shear_stress[time_idx],
        output_dir / 'max_bed_shear_stress.png',
        title=f'Max bed shear stress ({time_label})',
        colorbar_label='Pa',
        mask=mask,
    )

    if data.bed_load_transport is not None:
        _plot_scalar_field(
            scalar_x,
            scalar_y,
            data.bed_load_transport['magnitude'][time_idx],
            output_dir / 'bed_load_transport.png',
            title=f'Bed load transport magnitude ({time_label})',
            colorbar_label='m3/(s m)',
            mask=mask,
        )

    if data.suspended_transport is not None:
        _plot_scalar_field(
            scalar_x,
            scalar_y,
            data.suspended_transport['magnitude'][time_idx],
            output_dir / 'suspended_transport.png',
            title=f'Suspended transport magnitude ({time_label})',
            colorbar_label='m3/(s m)',
            mask=mask,
        )


def _plot_scalar_field(
    x,
    y,
    values,
    output_path: Path,
    title: str,
    colorbar_label: str,
    mask: np.ndarray | None = None,
) -> None:
    """Plot a scalar field using provided x/y coordinates when available."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    if np.asarray(x).ndim == 2 and np.asarray(y).ndim == 2:
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        invalid_mask = (x_arr == 0) & (y_arr == 0)
        if mask is not None:
            invalid_mask = invalid_mask | mask
        masked_values = np.where(invalid_mask, np.nan, values)
        valid = np.isfinite(masked_values)
        plt.scatter(
            x_arr[valid],
            y_arr[valid],
            c=masked_values[valid],
            s=36,
            marker='s',
            linewidths=0,
        )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')
    elif np.asarray(x).ndim == 1 and np.asarray(y).ndim == 1:
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        invalid_mask = (x_arr == 0) & (y_arr == 0)
        if mask is not None:
            invalid_mask = invalid_mask | mask
        masked_values = np.where(invalid_mask, np.nan, values)
        valid = np.isfinite(masked_values)
        plt.scatter(
            x_arr[valid],
            y_arr[valid],
            c=masked_values[valid],
            s=36,
            marker='s',
            linewidths=0,
        )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')
    else:
        plt.imshow(values, origin='lower')
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _format_time_label(data, time_idx: int) -> str:
    """Return a timestamp label for plot titles."""
    try:
        timestamp = data.reference_date + np.timedelta64(int(data.times[time_idx]), 's')
        return str(np.datetime_as_string(timestamp, unit='s'))
    except Exception:
        return f'index {time_idx}'


def _resolve_plot_time_index(data) -> int:
    """Resolve plot time index from environment, with safe fallback."""
    raw_value = os.environ.get('SEDTRAILS_PLOT_TIME_INDEX')
    if raw_value is None:
        return 0
    try:
        idx = int(raw_value)
    except ValueError:
        return 0
    if idx < 0:
        idx = 0
    if idx >= data.times.size:
        idx = data.times.size - 1
    return idx


def _resolve_scalar_coordinates(data) -> tuple[np.ndarray, np.ndarray]:
    """Return XZ/YZ coordinates for scalar plotting when available."""
    import xarray as xr

    if np.asarray(data.x).ndim == 1:
        return np.asarray(data.x), np.asarray(data.y)

    if SAMPLE_FILE.exists():
        try:
            ds = xr.open_dataset(SAMPLE_FILE, decode_times=False, decode_timedelta=False)
        except Exception:
            ds = None
        else:
            if 'XZ' in ds and 'YZ' in ds:
                return np.asarray(ds['XZ'].values), np.asarray(ds['YZ'].values)

    return np.asarray(data.x), np.asarray(data.y)


def _resolve_plot_mask() -> str | None:
    """Return optional plot mask selector from environment."""
    value = os.environ.get('SEDTRAILS_PLOT_MASK')
    if value is None:
        return None
    value = value.strip().lower()
    return value if value else None


def _build_kcs_mask() -> np.ndarray | None:
    """Build a mask from KCS values in the sample dataset when available."""
    import xarray as xr

    if not SAMPLE_FILE.exists():
        return None

    try:
        ds = xr.open_dataset(SAMPLE_FILE, decode_times=False, decode_timedelta=False)
    except Exception:
        return None

    if 'KCS' not in ds:
        return None

    kcs = np.asarray(ds['KCS'].values)
    return kcs == 0


@pytest.mark.integration
@pytest.mark.skipif(not SAMPLE_FILE.exists(), reason='Sample Delft3D4 NetCDF file not available')
def test_delft3d4_netcdf_conversion() -> None:
    """Check Delft3D4 NetCDF conversion produces a valid SedtrailsData object."""
    plugin = d3d4_netcdf.FormatPlugin(str(SAMPLE_FILE))
    data = plugin.convert(reference_date=np.datetime64('1970-01-01T00:00:00'))

    assert data.times.size > 0
    assert data.x.shape == data.y.shape
    assert data.bed_level.shape == data.x.shape

    assert data.depth_avg_flow_velocity is not None
    assert data.depth_avg_flow_velocity['x'].shape[0] == data.times.size
    assert data.depth_avg_flow_velocity['y'].shape == data.depth_avg_flow_velocity['x'].shape

    assert data.water_depth.shape[0] == data.times.size
    assert data.mean_bed_shear_stress.shape == data.max_bed_shear_stress.shape

    assert np.isfinite(data.x).any()
    assert np.isfinite(data.y).any()

    domain = data.metadata.flowfield_domain
    for key in ['x_min', 'x_max', 'y_min', 'y_max']:
        assert key in domain

    plot_dir = os.environ.get('SEDTRAILS_PLOT_DIR')
    if plot_dir:
        time_idx = _resolve_plot_time_index(data)
        mask_type = _resolve_plot_mask()
        mask = _build_kcs_mask() if mask_type == 'kcs' else None
        _plot_if_requested(data, Path(plot_dir), time_idx, mask)


def test_selects_configured_fraction_index_from_lsed_dimension(tmp_path: Path) -> None:
    """The plugin should select the configured sediment fraction index for LSED data."""
    input_file = tmp_path / 'dummy.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.sediment_fraction_index = 2

    var = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=('LSED', 'M'),
        coords={'LSED': ['sediment100_nat', 'sediment200_nat', 'sediment_mud']},
    )

    selected = plugin._select_first_dims(var)
    np.testing.assert_array_equal(selected.values, var.values[2])


def test_selects_configured_fraction_name_from_lsed_dimension(tmp_path: Path) -> None:
    """The plugin should resolve sediment fraction by coordinate name when configured."""
    input_file = tmp_path / 'dummy.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.sediment_fraction_index = 0
    plugin.sediment_fraction_name = 'sediment300_nat'

    var = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=('LSED', 'M'),
        coords={'LSED': ['sediment100_nat', 'sediment200_nat', 'sediment300_nat']},
    )

    selected = plugin._select_first_dims(var)
    np.testing.assert_array_equal(selected.values, var.values[2])


def test_converts_namcon_labels_into_metadata(tmp_path: Path) -> None:
    """The converter should expose Delft3D-4 NAMCON labels as sediment fraction metadata."""
    input_file = tmp_path / 'namcon.nc'

    labels = ['sediment100_nat', 'sediment200_nat', 'sediment300_nat']
    namcon = np.full((len(labels), 20), b' ', dtype='S1')
    for row_index, label in enumerate(labels):
        encoded = np.array(list(label.ljust(20)), dtype='S1')
        namcon[row_index, :] = encoded

    x_coords = np.array([[0.0, 1.0], [0.0, 1.0]])
    y_coords = np.array([[0.0, 0.0], [1.0, 1.0]])

    dataset = xr.Dataset(
        data_vars={
            'NAMCON': (('LSTSCI', 'strlen20'), namcon),
            'XCOR': (('y', 'x'), x_coords),
            'YCOR': (('y', 'x'), y_coords),
            'DP0': (('y', 'x'), np.array([[1.0, 1.0], [1.0, 1.0]])),
            'time': (('time',), np.array(['2010-01-01T00:00:00'], dtype='datetime64[ns]')),
            'U1': (('time', 'y', 'x'), np.array([[[1.0, 1.0], [1.0, 1.0]]])),
            'V1': (('time', 'y', 'x'), np.array([[[0.0, 0.0], [0.0, 0.0]]])),
            'TAUKSI': (('time', 'y', 'x'), np.array([[[0.1, 0.1], [0.1, 0.1]]])),
            'TAUETA': (('time', 'y', 'x'), np.array([[[0.2, 0.2], [0.2, 0.2]]])),
            'TAUMAX': (('time', 'y', 'x'), np.array([[[0.3, 0.3], [0.3, 0.3]]])),
            'SBUU': (('time', 'LSED', 'y', 'x'), np.ones((1, 3, 2, 2))),
            'SBVV': (('time', 'LSED', 'y', 'x'), np.ones((1, 3, 2, 2))),
            'SSUU': (('time', 'LSED', 'y', 'x'), np.ones((1, 3, 2, 2))),
            'SSVV': (('time', 'LSED', 'y', 'x'), np.ones((1, 3, 2, 2))),
            'R1': (('time', 'LSED', 'y', 'x'), np.ones((1, 3, 2, 2))),
        }
    )
    dataset.to_netcdf(input_file)

    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    data = plugin.convert(reference_date=np.datetime64('1970-01-01T00:00:00'))

    assert data.metadata.get('sediment_fraction_labels') == labels


def test_raises_for_out_of_bounds_fraction_index(tmp_path: Path) -> None:
    """An invalid fraction index should fail fast with a clear error."""
    input_file = tmp_path / 'dummy.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))
    plugin.sediment_fraction_index = 99

    var = xr.DataArray(np.arange(8).reshape(2, 4), dims=('LSED', 'M'))

    with pytest.raises(ValueError, match='out of bounds'):
        plugin._select_first_dims(var)


def test_preserves_fraction_dimension_when_requested(tmp_path: Path) -> None:
    """Sediment variables should keep LSED when selection is deferred to population level."""
    input_file = tmp_path / 'dummy.nc'
    input_file.write_text('')
    plugin = d3d4_netcdf.FormatPlugin(str(input_file))

    var = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=('LSED', 'M'),
    )

    selected = plugin._select_first_dims(var, select_fraction_dims=False)
    assert selected.dims == ('LSED', 'M')
    np.testing.assert_array_equal(selected.values, var.values)
