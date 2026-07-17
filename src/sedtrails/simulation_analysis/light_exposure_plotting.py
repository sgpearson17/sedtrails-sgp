"""Plotting helpers for particle light-exposure diagnostics."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.typing import ArrayLike

from sedtrails.simulation_analysis.light_exposure import ExposureResult


def _elapsed_hours(time: ArrayLike) -> np.ndarray:
    values = np.asarray(time)
    if values.ndim != 1:
        raise ValueError('time must be one-dimensional')
    if np.issubdtype(values.dtype, np.datetime64):
        seconds = (values.astype('datetime64[ns]') - values[0].astype('datetime64[ns]')).astype('timedelta64[s]')
        return seconds.astype(float) / 3600.0
    if np.issubdtype(values.dtype, np.timedelta64):
        seconds = (values.astype('timedelta64[s]') - values[0].astype('timedelta64[s]')).astype(float)
        return seconds / 3600.0
    return (values.astype(float) - float(values[0])) / 3600.0


def _time_labels(hours: np.ndarray) -> list[str]:
    return [f'{hour / 24.0:.2f}' for hour in hours]


def _as_1d_float(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    return array


def _as_1d_bool(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    return array


def _true_intervals(mask: np.ndarray, x: np.ndarray) -> list[tuple[float, float]]:
    transitions = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    intervals: list[tuple[float, float]] = []
    for start, stop in zip(starts, stops):
        right = x[max(stop - 1, start)]
        if right == x[start] and start + 1 < x.size:
            right = x[start + 1]
        intervals.append((float(x[start]), float(right)))
    return intervals


def _light_penetration_grid(
    water_depth: np.ndarray,
    attenuation_coefficient: np.ndarray,
    surface_light: np.ndarray,
    z_grid: np.ndarray,
) -> np.ndarray:
    depth_below_surface = water_depth[:, None] - z_grid[None, :]
    in_water = (z_grid[None, :] >= 0.0) & (depth_below_surface >= 0.0)
    with np.errstate(over='ignore', invalid='ignore'):
        light = surface_light[:, None] * np.exp(-attenuation_coefficient[:, None] * np.maximum(depth_below_surface, 0.0))
    return np.where(in_water, light, np.nan)


def plot_particle_bleaching_potential(
    time: ArrayLike,
    water_depth: ArrayLike,
    particle_elevation_above_bed: ArrayLike,
    concentration_series: dict[str, ArrayLike] | ArrayLike,
    surface_light: ArrayLike,
    attenuation_coefficient: ArrayLike,
    exposure: ExposureResult,
    *,
    title_prefix: str = '',
    concentration_unit: str = 'kg m$^{-3}$',
    light_unit: str = '$\\mu$mol s$^{-1}$ m$^{-2}$',
    colormap: str = 'viridis',
    n_depth_bins: int = 160,
    burial_mask: ArrayLike | None = None,
    max_x_ticks: int = 9,
):
    """Create a MATLAB-style seven-panel particle light-exposure diagnostic plot.

    Parameters
    ----------
    time : array-like, shape (time,)
        Saved trajectory times.
    water_depth : array-like, shape (time,)
        Water level/depth above the bed.
    particle_elevation_above_bed : array-like, shape (time,)
        Particle elevation above the bed, from saved ``z`` or estimated ``z_c``.
    concentration_series : dict or array-like
        One or more concentration time series. Dict keys are used as legend labels.
    surface_light : array-like, shape (time,)
        Incoming surface radiation.
    attenuation_coefficient : array-like, shape (time,)
        Light attenuation coefficient.
    exposure : ExposureResult
        Output from ``compute_light_exposure`` for the same particle/time series.

    Returns
    -------
    tuple
        ``(fig, axes)`` from Matplotlib.
    """

    time_hours = _elapsed_hours(time)
    water_depth = _as_1d_float(water_depth, 'water_depth')
    particle_z = _as_1d_float(particle_elevation_above_bed, 'particle_elevation_above_bed')
    surface_light = _as_1d_float(surface_light, 'surface_light')
    kd = _as_1d_float(attenuation_coefficient, 'attenuation_coefficient')

    if isinstance(concentration_series, dict):
        concentration_items = [(str(label), _as_1d_float(values, f'concentration {label}')) for label, values in concentration_series.items()]
    else:
        concentration_items = [('Concentration', _as_1d_float(concentration_series, 'concentration_series'))]

    instantaneous = np.asarray(exposure.light_intensity, dtype=float).reshape(len(time_hours), -1)[:, 0]
    cumulative = np.asarray(exposure.cumulative_light_dose, dtype=float).reshape(len(time_hours), -1)[:, 0]
    bleaching = np.asarray(exposure.bleaching_probability, dtype=float).reshape(len(time_hours), -1)[:, 0]
    osl_reduction = np.asarray(exposure.osl_signal_reduction, dtype=float).reshape(len(time_hours), -1)[:, 0]
    buried = None if burial_mask is None else _as_1d_bool(burial_mask, 'burial_mask')

    z_max = max(float(np.nanmax(water_depth)), float(np.nanmax(particle_z)), 1.0)
    z_min = min(0.0, float(np.nanmin(particle_z)))
    if z_min < 0.0:
        # Keep below-bed values visible even when burial depth is very small.
        z_min = min(z_min * 1.15, -0.1 * z_max)
    z_grid = np.linspace(0.0, z_max, int(n_depth_bins))
    light_grid = _light_penetration_grid(water_depth, kd, surface_light, z_grid)

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(14, 12.0),
        sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.0, 1.0, 1.15, 1.0, 1.0, 1.0], 'hspace': 0.5},
    )

    panel_title_prefix = f'{title_prefix} ' if title_prefix else ''

    if buried is not None:
        if buried.size != time_hours.size:
            raise ValueError('burial_mask length must match time length')
        for left, right in _true_intervals(buried, time_hours):
            for ax in axes:
                ax.axvspan(left, right, color='0.85', alpha=0.4, lw=0, zorder=0)

    axes[0].plot(time_hours, water_depth, color='blue', lw=1.2, label='Surface')
    axes[0].plot(time_hours, particle_z, color='black', marker='.', ms=2.5, lw=0.9, label='Particle')
    axes[0].set_ylabel('h [m]')
    axes[0].set_ylim(z_min, z_max)
    axes[0].set_title(f'{panel_title_prefix}(a) Particle elevation', fontweight='bold')
    axes[0].legend(loc='upper right', frameon=True, fancybox=False, edgecolor='0.3')

    default_colors = ['blue', 'red', 'black', 'tab:green']
    for index, (label, values) in enumerate(concentration_items):
        axes[1].plot(time_hours, values, lw=1.1, color=default_colors[index % len(default_colors)], label=label)
    axes[1].set_ylabel(f'C [{concentration_unit}]')
    axes[1].set_title('(b) Concentration', fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True, fancybox=False, edgecolor='0.3')

    axes[2].plot(
        time_hours,
        surface_light,
        color='tab:orange',
        marker='o',
        mfc='white',
        mec='tab:orange',
        ms=3.5,
        lw=1.1,
    )
    axes[2].set_ylabel(f'I$_0$ [{light_unit}]')
    axes[2].set_title('(c) Incoming Solar Radiation', fontweight='bold')

    mesh = axes[3].pcolormesh(time_hours, z_grid, light_grid.T, shading='auto', cmap=colormap)
    axes[3].plot(time_hours, particle_z, color='white', marker='.', ms=3, lw=0.9)
    axes[3].set_ylabel('h [m]')
    axes[3].set_ylim(z_min, z_max)
    axes[3].set_title('(d) Light Penetration', fontweight='bold')
    # Keep panel width aligned with other subplots by using an inset colorbar axis.
    cax = inset_axes(
        axes[3],
        width='0.8%',
        height='100%',
        loc='lower left',
        bbox_to_anchor=(1.01, 0.0, 1, 1),
        bbox_transform=axes[3].transAxes,
        borderpad=0,
    )
    fig.colorbar(mesh, cax=cax, label=f'I [{light_unit}]')

    exposure_axis = axes[4]
    cumulative_axis = exposure_axis.twinx()
    exposure_axis.plot(time_hours, instantaneous, color='tab:blue', lw=1.1)
    cumulative_axis.plot(time_hours, cumulative, color='orangered', lw=1.1)
    exposure_axis.set_ylabel(f'I$_D$ [{light_unit}]', color='tab:blue')
    cumulative_axis.set_ylabel(f'$\\Sigma$ Exp [{light_unit}]', color='orangered')
    exposure_axis.tick_params(axis='y', colors='tab:blue')
    cumulative_axis.tick_params(axis='y', colors='orangered')
    exposure_axis.set_title('(e) Particle Exposure to Light', fontweight='bold')

    axes[5].plot(time_hours, bleaching, color='tab:purple', lw=1.1)
    axes[5].set_ylabel('P$_{bleach}$ [-]')
    axes[5].set_ylim(-0.05, 1.05)
    axes[5].set_title('(f) Bleaching Probability (placeholder)', fontweight='bold')

    axes[6].plot(time_hours, osl_reduction, color='tab:brown', lw=1.1)
    axes[6].set_ylabel('OSL signal [-]')
    axes[6].set_title('(g) OSL Signal Reduction (placeholder)', fontweight='bold')

    x_min = float(np.nanmin(time_hours))
    x_max = float(np.nanmax(time_hours))
    n_ticks = int(max(3, max_x_ticks))
    ticks = np.linspace(x_min, x_max, n_ticks)
    axes[-1].xaxis.set_major_locator(ticker.FixedLocator(ticks))
    axes[-1].set_xticklabels(_time_labels(ticks))
    axes[-1].set_xlabel('time since release [days]')
    cumulative_axis.set_xlim(x_min, x_max)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.tick_params(direction='in', top=True, right=True)
        ax.set_xlim(x_min, x_max)

    return fig, axes
