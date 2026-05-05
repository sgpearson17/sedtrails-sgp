"""Tests for the Delft3D4 NetCDF format converter."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sedtrails.transport_converter.plugins.format import delft3d4_netcdf


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

    if SAMPLE_FILE.exists():
        try:
            ds = xr.open_dataset(SAMPLE_FILE, decode_times=False, decode_timedelta=False)
        except Exception:
            ds = None
        else:
            if 'XZ' in ds and 'YZ' in ds:
                return np.asarray(ds['XZ'].values), np.asarray(ds['YZ'].values)

    return np.asarray(data.x), np.asarray(data.y)


def _compute_cell_edges(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute cell edge coordinates from cell centers for curvilinear grids."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError('x and y must be 2D arrays for edge computation')

    m, n = x.shape
    x_edges = np.empty((m + 1, n + 1), dtype=float)
    y_edges = np.empty((m + 1, n + 1), dtype=float)

    x_edges[1:-1, 1:-1] = 0.25 * (x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:])
    y_edges[1:-1, 1:-1] = 0.25 * (y[:-1, :-1] + y[1:, :-1] + y[:-1, 1:] + y[1:, 1:])

    x_edges[0, 1:-1] = x_edges[1, 1:-1] - (x_edges[2, 1:-1] - x_edges[1, 1:-1])
    x_edges[-1, 1:-1] = x_edges[-2, 1:-1] + (x_edges[-2, 1:-1] - x_edges[-3, 1:-1])
    x_edges[1:-1, 0] = x_edges[1:-1, 1] - (x_edges[1:-1, 2] - x_edges[1:-1, 1])
    x_edges[1:-1, -1] = x_edges[1:-1, -2] + (x_edges[1:-1, -2] - x_edges[1:-1, -3])

    y_edges[0, 1:-1] = y_edges[1, 1:-1] - (y_edges[2, 1:-1] - y_edges[1, 1:-1])
    y_edges[-1, 1:-1] = y_edges[-2, 1:-1] + (y_edges[-2, 1:-1] - y_edges[-3, 1:-1])
    y_edges[1:-1, 0] = y_edges[1:-1, 1] - (y_edges[1:-1, 2] - y_edges[1:-1, 1])
    y_edges[1:-1, -1] = y_edges[1:-1, -2] + (y_edges[1:-1, -2] - y_edges[1:-1, -3])

    x_edges[0, 0] = x_edges[0, 1] + x_edges[1, 0] - x_edges[1, 1]
    x_edges[0, -1] = x_edges[0, -2] + x_edges[1, -1] - x_edges[1, -2]
    x_edges[-1, 0] = x_edges[-1, 1] + x_edges[-2, 0] - x_edges[-2, 1]
    x_edges[-1, -1] = x_edges[-1, -2] + x_edges[-2, -1] - x_edges[-2, -2]

    y_edges[0, 0] = y_edges[0, 1] + y_edges[1, 0] - y_edges[1, 1]
    y_edges[0, -1] = y_edges[0, -2] + y_edges[1, -1] - y_edges[1, -2]
    y_edges[-1, 0] = y_edges[-1, 1] + y_edges[-2, 0] - y_edges[-2, 1]
    y_edges[-1, -1] = y_edges[-1, -2] + y_edges[-2, -1] - y_edges[-2, -2]

    return x_edges, y_edges


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
    plugin = delft3d4_netcdf.FormatPlugin(str(SAMPLE_FILE))
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
