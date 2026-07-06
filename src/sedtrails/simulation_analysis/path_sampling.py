"""Sample Eulerian fields along saved particle trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay


@dataclass(frozen=True)
class SampledField:
    """Field values sampled at particle trajectory positions."""

    values: np.ndarray
    lower_time_index: np.ndarray
    upper_time_index: np.ndarray
    time_weight: np.ndarray


def _as_1d_array(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    if array.size == 0:
        raise ValueError(f'{name} must not be empty')
    return array


def _as_float_array(value: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        raise ValueError(f'{name} must not be empty')
    return array


def _time_to_float(value: np.ndarray) -> np.ndarray:
    """Convert numeric or datetime-like times to monotonically comparable floats."""
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.datetime64):
        return array.astype('datetime64[ns]').astype(np.int64).astype(float)
    if np.issubdtype(array.dtype, np.timedelta64):
        return array.astype('timedelta64[ns]').astype(np.int64).astype(float)
    return array.astype(float)


def interpolation_indices(field_time: ArrayLike, target_time: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower/upper field-time indices and linear interpolation weights."""

    source_time = _as_1d_array(field_time, 'field_time')
    target = np.asarray(target_time)
    source_float = _time_to_float(source_time)
    target_float = _time_to_float(target)

    if np.any(np.diff(source_float) < 0):
        raise ValueError('field_time must be monotonically increasing')

    lower = np.searchsorted(source_float, target_float, side='right') - 1
    lower = np.clip(lower, 0, source_float.size - 1)
    upper = np.clip(lower + 1, 0, source_float.size - 1)

    denominator = source_float[upper] - source_float[lower]
    weight = np.zeros_like(target_float, dtype=float)
    valid = denominator != 0
    np.divide(target_float - source_float[lower], denominator, out=weight, where=valid)
    weight = np.clip(weight, 0.0, 1.0)

    before = target_float <= source_float[0]
    after = target_float >= source_float[-1]
    lower = np.where(before, 0, lower)
    upper = np.where(before, 0, upper)
    weight = np.where(before, 0.0, weight)
    lower = np.where(after, source_float.size - 1, lower)
    upper = np.where(after, source_float.size - 1, upper)
    weight = np.where(after, 0.0, weight)

    return lower.astype(np.int64), upper.astype(np.int64), weight


def _values_for_fraction(field_values: ArrayLike, sediment_fraction: int | None) -> np.ndarray:
    values = _as_float_array(field_values, 'field_values')
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        if sediment_fraction is None:
            raise ValueError('sediment_fraction is required for fields with a fraction dimension')
        return values[:, int(sediment_fraction), :]
    raise ValueError('field_values must have shape (time, nodes) or (time, fractions, nodes)')


def _spatial_sample(
    triangulation: Delaunay,
    nearest: NearestNDInterpolator | None,
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    nearest_fallback: bool,
) -> np.ndarray:
    interpolator = LinearNDInterpolator(triangulation, field, fill_value=np.nan)
    sampled = np.asarray(interpolator(x, y), dtype=float)
    if nearest_fallback and nearest is not None:
        missing = ~np.isfinite(sampled)
        if np.any(missing):
            sampled[missing] = nearest(x[missing], y[missing])
    return sampled


def sample_field_at_trajectories(
    field_time: ArrayLike,
    grid_x: ArrayLike,
    grid_y: ArrayLike,
    field_values: ArrayLike,
    trajectory_time: ArrayLike,
    trajectory_x: ArrayLike,
    trajectory_y: ArrayLike,
    *,
    sediment_fraction: int | None = None,
    nearest_fallback: bool = False,
) -> SampledField:
    """Sample a time-varying spatial field at saved particle positions.

    Parameters
    ----------
    field_time : array-like, shape (n_time,)
        Time coordinate of the Eulerian field. Numeric and datetime64 values are supported.
    grid_x, grid_y : array-like, shape (n_nodes,)
        Spatial coordinates of the Eulerian field nodes.
    field_values : array-like, shape (n_time, n_nodes) or (n_time, n_fraction, n_nodes)
        Field values to sample.
    trajectory_time : array-like, shape (n_samples,)
        Saved trajectory times.
    trajectory_x, trajectory_y : array-like, shape (n_samples, n_particles)
        Particle coordinates at saved trajectory times.
    sediment_fraction : int or None
        Fraction index for fields with a sediment-fraction dimension.
    nearest_fallback : bool
        Fill spatially outside-triangulation values using nearest-neighbor sampling.

    Returns
    -------
    SampledField
        Sampled values and the temporal interpolation metadata.
    """

    source_time = _as_1d_array(field_time, 'field_time')
    x_grid = _as_1d_array(grid_x, 'grid_x').astype(float)
    y_grid = _as_1d_array(grid_y, 'grid_y').astype(float)
    values = _values_for_fraction(field_values, sediment_fraction)
    target_time = _as_1d_array(trajectory_time, 'trajectory_time')
    x_path = _as_float_array(trajectory_x, 'trajectory_x')
    y_path = _as_float_array(trajectory_y, 'trajectory_y')

    if x_grid.shape != y_grid.shape:
        raise ValueError('grid_x and grid_y must have the same shape')
    if values.shape[0] != source_time.size:
        raise ValueError('field_values first dimension must match field_time length')
    if values.shape[-1] != x_grid.size:
        raise ValueError('field_values node dimension must match grid_x/grid_y length')
    if x_path.shape != y_path.shape:
        raise ValueError('trajectory_x and trajectory_y must have the same shape')
    if x_path.shape[0] != target_time.size:
        raise ValueError('trajectory path first dimension must match trajectory_time length')

    lower, upper, weight = interpolation_indices(source_time, target_time)
    out = np.full(x_path.shape, np.nan, dtype=float)

    valid_nodes = np.isfinite(x_grid) & np.isfinite(y_grid)
    if np.count_nonzero(valid_nodes) < 3:
        raise ValueError('At least three finite grid nodes are required for spatial interpolation')
    points = np.column_stack((x_grid[valid_nodes], y_grid[valid_nodes]))
    triangulation = Delaunay(points)

    valid_query = np.isfinite(x_path) & np.isfinite(y_path)
    for sample_index in range(target_time.size):
        query_mask = valid_query[sample_index]
        if not np.any(query_mask):
            continue

        lo = lower[sample_index]
        hi = upper[sample_index]
        w = weight[sample_index]

        lower_field = values[lo, valid_nodes]
        upper_field = values[hi, valid_nodes]
        lower_nearest = NearestNDInterpolator(points, lower_field) if nearest_fallback else None
        upper_nearest = NearestNDInterpolator(points, upper_field) if nearest_fallback else None

        lower_sample = _spatial_sample(
            triangulation,
            lower_nearest,
            lower_field,
            x_path[sample_index, query_mask],
            y_path[sample_index, query_mask],
            nearest_fallback=nearest_fallback,
        )
        if lo == hi:
            sampled = lower_sample
        else:
            upper_sample = _spatial_sample(
                triangulation,
                upper_nearest,
                upper_field,
                x_path[sample_index, query_mask],
                y_path[sample_index, query_mask],
                nearest_fallback=nearest_fallback,
            )
            sampled = (1.0 - w) * lower_sample + w * upper_sample

        out[sample_index, query_mask] = sampled

    return SampledField(values=out, lower_time_index=lower, upper_time_index=upper, time_weight=weight)
