"""
Module for visualizing particle trajectories from SedTRAILS NetCDF output files.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sedtrails.particle_tracer.geodetic_geometry import (
    EARTH_MEAN_RADIUS_M,
    spherical_distance,
)
from matplotlib.collections import LineCollection


DEFAULT_MAX_PLOT_PARTICLES = 10_000
DEFAULT_MAX_PLOT_POINTS = 2_000_000
DEFAULT_RENDER_PARTICLE_CHUNK = 256


def _decode_netcdf_name(raw_value) -> str:
    """Decode a NetCDF string value stored as bytes, fixed-width bytes, or char arrays."""
    if raw_value is None:
        return ''

    if isinstance(raw_value, str):
        return raw_value.strip()

    if isinstance(raw_value, (bytes, np.bytes_)):
        return raw_value.decode('utf-8', errors='ignore').replace('\x00', '').strip()

    arr = np.asarray(raw_value)
    if arr.ndim == 0:
        return _decode_netcdf_name(arr.item())


    # Scalar numpy bytes/str (0-dim array or item)
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, (bytes, np.bytes_)):
            return item.decode('utf-8', errors='ignore').strip('\x00').strip()
        return str(item).strip()

    # Char-array representation (e.g., dtype='|S1' with trailing nulls)
    if arr.ndim > 0 and arr.size > 0 and arr.dtype.kind in ('S', 'U'):
        flattened = arr.ravel().tolist()
        chars = []
        for item in flattened:
            if isinstance(item, (bytes, np.bytes_)):
                text = item.decode('utf-8', errors='ignore')
            else:
                text = str(item)
            text = text.replace('\x00', '')
            if text:
                chars.append(text)
        return ''.join(chars).strip()

    return str(raw_value).strip()


def read_netcdf(results_file_path: Path) -> xr.Dataset:
    """Read a SedTRAILS NetCDF file and return the xarray Dataset.

    Parameters
    ----------
    results_file_path : pathlib.Path
        Path to the SedTRAILS NetCDF file.

    Returns
    -------
    xr.Dataset
        The xarray Dataset containing the SedTRAILS results.

    """
    results_file_path = Path(results_file_path)
    if not results_file_path.exists():
        raise FileNotFoundError(f"NetCDF file '{results_file_path}' not found.")
    ds = xr.open_dataset(results_file_path, engine='netcdf4', decode_times=False)
    return ds


def _reference_datetime64(ds: xr.Dataset, time_var: xr.DataArray) -> np.datetime64 | None:
    for value in (
        time_var.attrs.get('reference_date'),
        ds.attrs.get('reference_date'),
    ):
        if value is not None:
            return np.datetime64(str(value), 'ns')

    for units in (
        time_var.attrs.get('units'),
        time_var.encoding.get('units'),
        ds.attrs.get('time_units'),
        ds.attrs.get('units'),
    ):
        match = re.match(r'^\s*seconds\s+since\s+(.+?)\s*$', str(units), flags=re.IGNORECASE)
        if match:
            return np.datetime64(match.group(1), 'ns')

    return None


def _time_values_as_seconds(ds: xr.Dataset, time_var: xr.DataArray) -> np.ndarray:
    values = np.asarray(time_var.values)
    if np.issubdtype(values.dtype, np.datetime64):
        values_ns = values.astype('datetime64[ns]')
        valid = ~np.isnat(values_ns)
        reference = _reference_datetime64(ds, time_var)
        if reference is None and np.any(valid):
            reference = values_ns[valid].min()
        if reference is None:
            return np.full(values_ns.shape, np.nan, dtype=float)

        seconds = np.full(values_ns.shape, np.nan, dtype=float)
        seconds[valid] = (values_ns[valid] - reference) / np.timedelta64(1, 's')
        return seconds

    return np.asarray(values, dtype=float)


def _infer_particle_dim(ds: xr.Dataset) -> str | None:
    """Infer the particle dimension used by the trajectory arrays."""
    if 'x' not in ds:
        return None

    x_var = ds['x']
    if x_var.ndim == 1:
        return x_var.dims[0]
    if x_var.ndim == 2 and 'time' in ds and ds['time'].ndim == 1 and x_var.dims[0] == ds['time'].dims[0]:
        return x_var.dims[1]
    if 'n_particles' in ds.sizes:
        return 'n_particles'
    if 'population_id' in ds and ds['population_id'].ndim == 1:
        return ds['population_id'].dims[0]
    return None


def _particle_count(ds: xr.Dataset) -> int:
    particle_dim = _infer_particle_dim(ds)
    if particle_dim is None:
        return 0
    return int(ds.sizes[particle_dim])


def _sample_target(
    n_particles: int,
    *,
    max_particles: int | None,
    sample_fraction: float | None,
) -> int:
    """Validate sampling options and return the selected particle count."""
    if max_particles is not None and max_particles < 1:
        raise ValueError("'max_particles' must be at least 1.")
    if sample_fraction is not None and not (0.0 < sample_fraction <= 1.0):
        raise ValueError("'sample_fraction' must be greater than 0 and less than or equal to 1.")
    if max_particles is not None and sample_fraction is not None and max_particles != DEFAULT_MAX_PLOT_PARTICLES:
        raise ValueError("'max_particles' and 'sample_fraction' are mutually exclusive.")

    if n_particles <= 0:
        return 0
    if sample_fraction is not None:
        return max(1, min(int(np.ceil(n_particles * sample_fraction)), n_particles))
    if max_particles is None:
        return n_particles
    return min(int(max_particles), n_particles)


def _population_sample_counts(population_sizes: np.ndarray, target: int) -> np.ndarray:
    """Allocate a deterministic stratified sample budget across populations."""
    population_sizes = np.asarray(population_sizes, dtype=np.int64)
    counts = np.zeros(population_sizes.size, dtype=int)
    active = population_sizes > 0
    if target <= 0 or not np.any(active):
        return counts

    ideal = population_sizes[active] * target / int(population_sizes.sum())
    active_counts = np.floor(ideal).astype(int)
    if target >= int(np.count_nonzero(active)):
        active_counts = np.maximum(active_counts, 1)
    active_counts = np.minimum(active_counts, population_sizes[active])

    while active_counts.sum() > target:
        removable = np.flatnonzero(active_counts > (1 if target >= active_counts.size else 0))
        if removable.size == 0:
            break
        candidate = removable[np.argmin(ideal[removable] - active_counts[removable])]
        active_counts[candidate] -= 1

    fractional = ideal - np.floor(ideal)
    while active_counts.sum() < target:
        capacity = population_sizes[active] - active_counts
        candidates = np.flatnonzero(capacity > 0)
        if candidates.size == 0:
            break
        candidate = candidates[np.argmax(fractional[candidates])]
        active_counts[candidate] += 1
        fractional[candidate] = -1.0

    counts[active] = active_counts
    return counts


def _population_ranges_from_metadata(ds: xr.Dataset, n_particles: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return validated contiguous population ranges from static output metadata."""
    if 'population_start_idx' not in ds or 'population_count' not in ds:
        return None

    starts = np.asarray(ds['population_start_idx'].values, dtype=np.int64)
    counts = np.asarray(ds['population_count'].values, dtype=np.int64)
    if starts.ndim != 1 or counts.ndim != 1 or starts.shape != counts.shape:
        return None
    if np.any(starts < 0) or np.any(counts < 0) or int(counts.sum()) != n_particles:
        return None

    ends = starts + counts
    order = np.argsort(starts, kind='stable')
    if np.any(ends > n_particles) or np.any(starts[order][1:] < ends[order][:-1]):
        return None
    return starts, counts


def _stratified_indices_from_ranges(
    starts: np.ndarray,
    counts: np.ndarray,
    target: int,
    sample_seed: int,
) -> np.ndarray:
    """Sample bounded per-population index arrays from contiguous metadata ranges."""
    selected_counts = _population_sample_counts(counts, target)
    rng = np.random.default_rng(sample_seed)
    selected = [
        int(start) + rng.choice(int(count), size=int(selected_count), replace=False)
        for start, count, selected_count in zip(starts, counts, selected_counts, strict=True)
        if selected_count > 0
    ]
    if not selected:
        return np.empty(0, dtype=int)
    return np.sort(np.concatenate(selected).astype(int, copy=False))


def _trajectory_time_dimension(ds: xr.Dataset) -> str | None:
    """Return the one-dimensional trajectory time dimension when present."""
    if 'time' not in ds or ds['time'].ndim != 1:
        return None
    return ds['time'].dims[0]


def _time_sample_indices(
    n_timesteps: int,
    n_particles: int,
    max_plot_points: int | None,
) -> np.ndarray | None:
    """Return evenly spaced time indices that keep the rendering point budget bounded."""
    if max_plot_points is None or n_timesteps <= 1:
        return None
    if max_plot_points < 1:
        raise ValueError("'max_plot_points' must be at least 1 or None.")

    target = min(n_timesteps, max(1, int(max_plot_points) // max(1, n_particles)))
    if target >= n_timesteps:
        return None
    if target == 1:
        return np.array([0], dtype=int)
    return np.linspace(0, n_timesteps - 1, num=target, dtype=int)


def _trajectory_shape(ds: xr.Dataset) -> tuple[int, int]:
    """Return plotting-shape dimensions without loading coordinate values."""
    x_var = ds['x']
    y_var = ds['y']
    if x_var.ndim == 1 and y_var.ndim == 1 and x_var.shape == y_var.shape:
        return int(x_var.shape[0]), 1
    if (
        x_var.ndim == 2
        and y_var.ndim == 2
        and 'time' in ds
        and ds['time'].ndim == 1
        and x_var.dims[0] == ds['time'].dims[0]
        and y_var.dims[0] == ds['time'].dims[0]
        and x_var.shape == y_var.shape
    ):
        return int(x_var.shape[1]), int(x_var.shape[0])
    raise ValueError(
        'Expected SedTRAILS time-major trajectory arrays shaped as '
        '(n_timesteps, n_particles), or 1D checkpoint arrays.'
    )


def _trajectory_time_values(ds: xr.Dataset, n_timesteps: int) -> np.ndarray:
    """Return one shared time vector in seconds without particle-sized broadcasting."""
    if 'time' not in ds:
        return np.arange(n_timesteps, dtype=float)
    if ds['time'].ndim == 0:
        return np.full(n_timesteps, float(_time_values_as_seconds(ds, ds['time'])), dtype=float)
    if ds['time'].ndim == 1:
        time_values = _time_values_as_seconds(ds, ds['time'])
        if time_values.size == n_timesteps:
            return time_values
        raise ValueError("'time' length does not match the timestep dimension.")
    raise ValueError("Unsupported 'time' variable shape for trajectory plotting.")
def _select_sample_indices(
    n_particles: int,
    population_ids: np.ndarray | None = None,
    *,
    max_particles: int | None = None,
    sample_fraction: float | None = None,
    sample_seed: int = 0,
) -> np.ndarray:
    """Return deterministic particle indices for plotting."""
    if max_particles is not None and sample_fraction is not None:
        raise ValueError("'max_particles' and 'sample_fraction' are mutually exclusive.")
    if max_particles is not None and max_particles < 1:
        raise ValueError("'max_particles' must be at least 1.")
    if sample_fraction is not None and not (0.0 < sample_fraction <= 1.0):
        raise ValueError("'sample_fraction' must be greater than 0 and less than or equal to 1.")

    if n_particles <= 0:
        return np.array([], dtype=int)
    if max_particles is None and sample_fraction is None:
        return np.arange(n_particles, dtype=int)

    if max_particles is not None:
        target = min(int(max_particles), n_particles)
    else:
        target = int(np.ceil(n_particles * float(sample_fraction)))
        target = max(1, min(target, n_particles))

    if target >= n_particles:
        return np.arange(n_particles, dtype=int)

    rng = np.random.default_rng(sample_seed)
    if population_ids is None or len(population_ids) != n_particles:
        return np.sort(rng.choice(n_particles, size=target, replace=False))

    selected: list[np.ndarray] = []
    unique_populations = np.unique(population_ids)
    population_indices = [np.flatnonzero(population_ids == pop_id) for pop_id in unique_populations]
    ideal_counts = np.array([len(indices) * target / n_particles for indices in population_indices], dtype=float)
    counts = np.floor(ideal_counts).astype(int)

    if target >= len(population_indices):
        counts = np.maximum(counts, 1)

    overflow = int(counts.sum() - target)
    if overflow > 0:
        removable_order = np.argsort(ideal_counts - counts)
        for idx in removable_order:
            if overflow == 0:
                break
            minimum = 1 if target >= len(population_indices) else 0
            if counts[idx] > minimum:
                counts[idx] -= 1
                overflow -= 1

    remainder = int(target - counts.sum())
    if remainder > 0:
        fractional_order = np.argsort(-(ideal_counts - np.floor(ideal_counts)))
        for idx in fractional_order:
            if remainder == 0:
                break
            capacity = len(population_indices[idx]) - counts[idx]
            if capacity > 0:
                counts[idx] += 1
                remainder -= 1

    for indices, count in zip(population_indices, counts, strict=True):
        if count > 0:
            selected.append(rng.choice(indices, size=min(count, len(indices)), replace=False))

    if not selected:
        return np.array([], dtype=int)

    selected_indices = np.unique(np.concatenate(selected).astype(int))
    if selected_indices.size < target:
        remaining_pool = np.setdiff1d(np.arange(n_particles, dtype=int), selected_indices, assume_unique=True)
        if remaining_pool.size:
            extra = rng.choice(
                remaining_pool, size=min(target - selected_indices.size, remaining_pool.size), replace=False
            )
            selected_indices = np.concatenate((selected_indices, extra))
    elif selected_indices.size > target:
        selected_indices = rng.choice(selected_indices, size=target, replace=False)

    return np.sort(selected_indices)


def _sample_dataset(
    ds: xr.Dataset,
    *,
    max_particles: int | None = DEFAULT_MAX_PLOT_PARTICLES,
    sample_fraction: float | None = None,
    sample_seed: int = 0,
    max_plot_points: int | None = DEFAULT_MAX_PLOT_POINTS,
) -> tuple[xr.Dataset, int, int]:
    """Sample particles and timesteps before loading trajectory coordinates.

    ``max_particles=None`` disables particle sampling. ``max_plot_points=None``
    disables time decimation; both opt-outs are explicit because they can create
    a large Matplotlib workload.
    """
    n_total = _particle_count(ds)
    particle_dim = _infer_particle_dim(ds)
    if particle_dim is None:
        return ds, n_total, n_total

    _, n_timesteps = _trajectory_shape(ds)
    target = _sample_target(
        n_total,
        max_particles=max_particles,
        sample_fraction=sample_fraction,
    )
    if max_plot_points is not None:
        if max_plot_points < 1:
            raise ValueError("'max_plot_points' must be at least 1 or None.")
        minimum_points_per_particle = 2 if n_timesteps > 1 else 1
        target = min(target, max(1, int(max_plot_points) // minimum_points_per_particle))

    selection = {}
    if target < n_total:
        ranges = _population_ranges_from_metadata(ds, n_total)
        if ranges is None:
            rng = np.random.default_rng(sample_seed)
            particle_indices = np.sort(rng.choice(n_total, size=target, replace=False))
        else:
            particle_indices = _stratified_indices_from_ranges(
                ranges[0],
                ranges[1],
                target,
                sample_seed,
            )
        selection[particle_dim] = particle_indices

    time_dim = _trajectory_time_dimension(ds)
    time_indices = _time_sample_indices(n_timesteps, target, max_plot_points)
    if time_indices is not None and time_dim is not None:
        selection[time_dim] = time_indices

    sampled_ds = ds.isel(selection) if selection else ds
    return sampled_ds, n_total, target


def _trajectory_arrays(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x/y/time arrays in plotting shape: (n_particles, n_timesteps)."""
    x_var = ds['x']
    y_var = ds['y']

    if x_var.ndim == 1 and y_var.ndim == 1:
        x_data = np.asarray(x_var.values, dtype=float)[:, np.newaxis]
        y_data = np.asarray(y_var.values, dtype=float)[:, np.newaxis]
    elif (
        x_var.ndim == 2
        and y_var.ndim == 2
        and 'time' in ds
        and ds['time'].ndim == 1
        and x_var.dims[0] == ds['time'].dims[0]
        and y_var.dims[0] == ds['time'].dims[0]
        and x_var.shape == y_var.shape
    ):
        x_data = np.asarray(x_var.values, dtype=float).T
        y_data = np.asarray(y_var.values, dtype=float).T
    else:
        raise ValueError(
            'Expected SedTRAILS time-major trajectory arrays shaped as '
            '(n_timesteps, n_particles), or 1D checkpoint arrays.'
        )

    n_particles, n_timesteps = x_data.shape
    if 'time' not in ds:
        time_data = np.broadcast_to(np.arange(n_timesteps, dtype=float), (n_particles, n_timesteps))
    elif ds['time'].ndim == 0:
        time_value = _time_values_as_seconds(ds, ds['time'])
        time_data = np.full((n_particles, n_timesteps), float(time_value))
    elif ds['time'].ndim == 1:
        time_values = _time_values_as_seconds(ds, ds['time'])
        if time_values.size == n_timesteps:
            time_data = np.broadcast_to(time_values, (n_particles, n_timesteps))
        else:
            raise ValueError("'time' length does not match the timestep dimension.")
    else:
        raise ValueError("Unsupported 'time' variable shape for trajectory plotting.")

    return x_data, y_data, time_data


def _iter_trajectory_chunks(ds: xr.Dataset, chunk_size: int = DEFAULT_RENDER_PARTICLE_CHUNK):
    """Yield bounded plotting arrays for consecutive selected particle chunks."""
    if chunk_size < 1:
        raise ValueError("'chunk_size' must be at least 1.")

    n_particles, _ = _trajectory_shape(ds)
    particle_dim = _infer_particle_dim(ds)
    if particle_dim is None:
        return

    for start in range(0, n_particles, chunk_size):
        stop = min(start + chunk_size, n_particles)
        chunk_ds = ds.isel({particle_dim: slice(start, stop)})
        x_data, y_data, time_data = _trajectory_arrays(chunk_ds)
        yield start, x_data, y_data, time_data


def _decode_population_names(ds: xr.Dataset, n_populations: int) -> list[str]:
    population_names = []
    if 'population_name' in ds:
        population_var = ds['population_name']
        n_available_names = int(population_var.sizes.get('n_populations', population_var.shape[0]))
        n_to_decode = min(n_populations, n_available_names)

        for i in range(n_to_decode):
            if population_var.ndim == 1:
                raw_name = population_var[i].values
            else:
                raw_name = population_var[i, :].values
            decoded = _decode_netcdf_name(raw_name)
            population_names.append(decoded or f'Population {i}')

        if len(population_names) < n_populations:
            population_names.extend([f'Population {i}' for i in range(len(population_names), n_populations)])
    else:
        population_names = [f'Population {i}' for i in range(n_populations)]

    return population_names


def _population_colors(n_populations: int):
    try:
        pop_cmap = plt.get_cmap('Set1')
        return pop_cmap(np.linspace(0, 1, n_populations))
    except (AttributeError, ValueError):
        base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        return [base_colors[i % len(base_colors)] for i in range(n_populations)]


def _particle_colors(n_particles: int):
    try:
        cmap = plt.get_cmap('viridis')
        return cmap(np.linspace(0, 1, n_particles))
    except (AttributeError, ValueError):
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        return [color_cycle[i % len(color_cycle)] for i in range(n_particles)]


def _line_segments(
    x_data: np.ndarray,
    y_data: np.ndarray,
    *,
    geographic: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    segments = []
    plotted_indices = []
    endpoint_indices = []
    starts = []
    ends = []

    for i in range(x_data.shape[0]):
        mask = ~(np.isnan(x_data[i, :]) | np.isnan(y_data[i, :]))
        if np.any(mask):
            x_traj = x_data[i, mask]
            y_traj = y_data[i, mask]
            points = np.column_stack((x_traj, y_traj))
            if len(points) >= 2:
                if geographic:
                    split_indices = np.flatnonzero(np.abs(np.diff(x_traj)) > 180.0) + 1
                    for segment in np.split(points, split_indices):
                        if len(segment) >= 2:
                            segments.append(segment)
                            plotted_indices.append(i)
                else:
                    segments.append(points)
                    plotted_indices.append(i)
            endpoint_indices.append(i)
            starts.append(points[0])
            ends.append(points[-1])

    return (
        segments,
        np.asarray(plotted_indices, dtype=int),
        np.asarray(endpoint_indices, dtype=int),
        np.asarray(starts, dtype=float) if starts else np.empty((0, 2), dtype=float),
        np.asarray(ends, dtype=float) if ends else np.empty((0, 2), dtype=float),
    )


def _distance_line_segments(
    x_data: np.ndarray,
    y_data: np.ndarray,
    time_data: np.ndarray,
    min_time: float,
    *,
    geographic: bool = False,
    earth_radius_m: float = EARTH_MEAN_RADIUS_M,
) -> tuple[list[np.ndarray], np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    distance_segments = []
    distance_indices = []
    particle_distances = {}

    for i in range(x_data.shape[0]):
        mask = ~(np.isnan(x_data[i, :]) | np.isnan(y_data[i, :]))
        if np.any(mask):
            x_traj = x_data[i, mask]
            y_traj = y_data[i, mask]
            time_traj = time_data[i, mask]
            time_hours = (time_traj - min_time) / 3600.0
            if geographic:
                distances = spherical_distance(
                    x_traj[0],
                    y_traj[0],
                    x_traj,
                    y_traj,
                    radius=earth_radius_m,
                )
            else:
                distances = np.sqrt((x_traj - x_traj[0]) ** 2 + (y_traj - y_traj[0]) ** 2)
            particle_distances[i] = (time_hours, distances)
            if len(time_hours) >= 2:
                distance_segments.append(np.column_stack((time_hours, distances)))
                distance_indices.append(i)

    return distance_segments, np.asarray(distance_indices, dtype=int), particle_distances


def _add_endpoint_markers(ax, starts, ends, *, colors, markers: str, marker_size: float):
    if markers == 'none':
        return
    if markers == 'start-end' and len(starts):
        ax.scatter(starts[:, 0], starts[:, 1], color=colors, marker='x', s=marker_size, linewidth=1, zorder=5)
    if markers in {'end', 'start-end'} and len(ends):
        ax.scatter(
            ends[:, 0],
            ends[:, 1],
            color=colors,
            marker='o',
            s=marker_size,
            edgecolor='black',
            linewidth=0.5,
            zorder=5,
        )


def _resolve_output_file(ds: xr.Dataset, output: str | Path | None) -> Path:
    """Resolve a trajectory-plot output path.

    ``None`` selects the NetCDF source directory. Any supplied path is used
    literally, so ``'.'`` selects the current working directory.
    """
    default_name = 'particle_trajectories.png'

    if output is not None:
        output_path = Path(output)
        if output_path.is_dir():
            return output_path / default_name
        return output_path

    source_file = ds.encoding.get('source', '.')
    source_dir = Path(source_file).parent
    return source_dir / default_name


def _normalize_panels(panels: str | list[str] | tuple[str, ...]) -> list[str]:
    valid_panels = ('spatial', 'distance', 'population', 'population-distance')

    if isinstance(panels, str):
        requested = [panel.strip().lower() for panel in panels.split(',') if panel.strip()]
    else:
        requested = [str(panel).strip().lower() for panel in panels if str(panel).strip()]

    if not requested:
        requested = ['spatial']
    if 'all' in requested:
        return list(valid_panels)

    invalid = sorted(set(requested) - set(valid_panels))
    if invalid:
        raise ValueError(
            f"'panels' contains invalid values: {', '.join(invalid)}. Valid values are: all, {', '.join(valid_panels)}."
        )

    ordered = [panel for panel in valid_panels if panel in requested]
    if 'spatial' not in ordered:
        ordered.insert(0, 'spatial')
    return ordered


def _create_panel_axes(panels: list[str]):
    if len(panels) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        return fig, {panels[0]: ax}

    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 8 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    axes_by_panel = dict(zip(panels, axes_flat, strict=True))
    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)
    return fig, axes_by_panel


def plot_trajectories(
    ds,
    output=None,
    max_particles: int | None = DEFAULT_MAX_PLOT_PARTICLES,
    sample_fraction: float | None = None,
    sample_seed: int = 0,
    markers: str = 'start-end',
    marker_size: float = 12.0,
    panels: str | list[str] | tuple[str, ...] = 'spatial',
    show: bool | None = None,
    max_plot_points: int | None = DEFAULT_MAX_PLOT_POINTS,
):
    """Plot particle trajectories from a SedTRAILS dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing particle trajectory variables, including ``x``,
        ``y``, and ``time``.
    output : str or pathlib.Path, optional
        Output target for the figure. If this is an existing directory, the
        plot is saved as ``particle_trajectories.png`` inside that directory.
        Otherwise, it is treated as an exact output filename. When omitted,
        the default output is ``particle_trajectories.png`` in the source
        dataset directory when available.
    max_particles : int, optional
        Maximum number of particles to plot. Defaults to 10,000. Sampling is
        deterministic and stratified when static population metadata is available.
        Set to ``None`` to retain all particles.
    sample_fraction : float, optional
        Fraction of particles to plot. This overrides the default particle cap.
    sample_seed : int, optional
        Seed used for deterministic particle sampling.
    max_plot_points : int, optional
        Maximum selected particle-time coordinates to render. Defaults to
        2,000,000 and evenly decimates time before coordinate loading. Set to
        ``None`` to retain every selected timestep.
    markers : {'none', 'end', 'start-end'}, optional
        Which endpoint markers to draw.
    marker_size : float, optional
        Marker size for start/end points.
    panels : str or sequence of str, optional
        Panels to draw. ``spatial`` is the fast default. Use ``all`` for the
        previous four-panel figure, or a comma-separated subset of ``spatial``,
        ``distance``, ``population``, and ``population-distance``.
    show : bool or None, optional
        Whether to display the figure. By default, figures are not shown.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the selected panels.
    axes_by_panel : dict[str, matplotlib.axes.Axes]
        Axes keyed by panel name.

    Notes
    -----
    This function creates a four-panel matplotlib figure. It returns the figure
    and axes to support testing or further customization.
    """
    if markers not in {'none', 'end', 'start-end'}:
        raise ValueError("'markers' must be one of: 'none', 'end', or 'start-end'.")
    if marker_size <= 0:
        raise ValueError("'marker_size' must be greater than 0.")
    selected_panels = _normalize_panels(panels)

    output_path = _resolve_output_file(ds, output)
    if show is None:
        show = False

    sampled_ds, n_total_particles, n_sampled_particles = _sample_dataset(
        ds,
        max_particles=max_particles,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        max_plot_points=max_plot_points,
    )
    n_particles, n_timesteps = _trajectory_shape(sampled_ds)
    time_values = _trajectory_time_values(sampled_ds, n_timesteps)
    geographic = str(sampled_ds.attrs.get('coordinate_system', 'projected')).lower() in {
        'geographic',
        'spherical',
        'lonlat',
        'longlat',
        'latitude_longitude',
    }
    earth_radius_m = float(sampled_ds.attrs.get('earth_radius_m', EARTH_MEAN_RADIUS_M))

    if n_sampled_particles != n_total_particles:
        print(
            f'\nPlotting trajectories for {n_sampled_particles} of {n_total_particles} '
            f'particles over {n_timesteps} timesteps...'
        )
    else:
        print(f'\nPlotting trajectories for {n_particles} particles over {n_timesteps} timesteps...')

    fig, axes_by_panel = _create_panel_axes(selected_panels)
    population_ids = (
        np.asarray(sampled_ds['population_id'].values, dtype=int)
        if 'population_id' in sampled_ds
        else np.zeros(n_particles, dtype=int)
    )
    n_populations = int(sampled_ds.sizes['n_populations']) if 'n_populations' in sampled_ds.sizes else 1
    population_names = _decode_population_names(sampled_ds, n_populations)
    pop_colors = _population_colors(n_populations)
    colors = np.asarray(_particle_colors(n_particles))

    ax1 = axes_by_panel['spatial']
    ax1.set_title(f'(a) Particle Trajectories - Individual Colors (n={n_particles})')
    ax1.set_xlabel('Longitude [degrees east]' if geographic else 'X [m]')
    ax1.set_ylabel('Latitude [degrees north]' if geographic else 'Y [m]')

    ax2 = axes_by_panel.get('distance')
    if ax2 is not None:
        ax2.set_title('(b) Distance from Initial Position vs Time')
        ax2.set_xlabel('Time [hours]')
        ax2.set_ylabel('Distance from Initial Position [m]')

    ax3 = axes_by_panel.get('population')
    if ax3 is not None:
        ax3.set_title('(c) Particle Trajectories - Colored by Population')
        ax3.set_xlabel('Longitude [degrees east]' if geographic else 'X [m]')
        ax3.set_ylabel('Latitude [degrees north]' if geographic else 'Y [m]')

    ax4 = axes_by_panel.get('population-distance')
    if ax4 is not None:
        ax4.set_title('(d) Distance from Initial Position by Population')
        ax4.set_xlabel('Time [hours]')
        ax4.set_ylabel('Distance from Initial Position [m]')

    spatial_has_segments = False
    population_has_segments = np.zeros(n_populations, dtype=bool)
    population_time_bounds = np.full((n_populations, 2), np.nan, dtype=float)
    for particle_start, x_data, y_data, _time_data in _iter_trajectory_chunks(sampled_ds):
        spatial_segments, spatial_indices, endpoint_indices, starts, ends = _line_segments(
            x_data,
            y_data,
            geographic=geographic,
        )
        if spatial_segments:
            ax1.add_collection(
                LineCollection(
                    spatial_segments,
                    colors=colors[particle_start + spatial_indices],
                    alpha=0.7,
                    linewidths=1,
                )
            )
            spatial_has_segments = True
        if len(starts):
            _add_endpoint_markers(
                ax1,
                starts,
                ends,
                colors=colors[particle_start + endpoint_indices],
                markers=markers,
                marker_size=marker_size,
            )

        chunk_population_ids = population_ids[particle_start:particle_start + x_data.shape[0]]
        valid_positions = np.isfinite(x_data) & np.isfinite(y_data)
        if ax3 is not None:
            for pop_idx in range(n_populations):
                in_population = chunk_population_ids == pop_idx
                if not np.any(in_population):
                    continue
                pop_segments, _, _, pop_starts, pop_ends = _line_segments(
                    x_data[in_population],
                    y_data[in_population],
                    geographic=geographic,
                )
                if pop_segments:
                    ax3.add_collection(
                        LineCollection(pop_segments, colors=[pop_colors[pop_idx]], alpha=0.7, linewidths=1)
                    )
                    population_has_segments[pop_idx] = True
                if len(pop_starts):
                    _add_endpoint_markers(
                        ax3,
                        pop_starts,
                        pop_ends,
                        colors=[pop_colors[pop_idx]] * len(pop_starts),
                        markers=markers,
                        marker_size=marker_size,
                    )

        if ax4 is not None:
            for pop_idx in range(n_populations):
                in_population = chunk_population_ids == pop_idx
                if not np.any(in_population):
                    continue
                active_times = np.any(valid_positions[in_population], axis=0)
                if not np.any(active_times):
                    continue
                first_time = float(time_values[np.flatnonzero(active_times)[0]])
                last_time = float(time_values[np.flatnonzero(active_times)[-1]])
                if np.isnan(population_time_bounds[pop_idx, 0]):
                    population_time_bounds[pop_idx] = (first_time, last_time)
                else:
                    population_time_bounds[pop_idx, 0] = min(population_time_bounds[pop_idx, 0], first_time)
                    population_time_bounds[pop_idx, 1] = max(population_time_bounds[pop_idx, 1], last_time)

    if spatial_has_segments:
        ax1.autoscale()
    if n_particles <= 10:
        for particle_idx, color in enumerate(colors):
            ax1.plot([], [], color=color, alpha=0.7, linewidth=1, label=f'Particle {particle_idx}')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    if not geographic:
        ax1.set_aspect('equal', adjustable='box')

    if ax3 is not None:
        if np.any(population_has_segments):
            for pop_idx in np.flatnonzero(population_has_segments):
                ax3.plot([], [], color=pop_colors[pop_idx], alpha=0.7, linewidth=1, label=population_names[pop_idx])
            ax3.legend()
            ax3.autoscale()
        ax3.grid(True, alpha=0.3)
        if not geographic:
            ax3.set_aspect('equal', adjustable='box')

    if ax2 is not None or ax4 is not None:
        min_time = float(np.nanmin(time_values)) if np.any(np.isfinite(time_values)) else 0.0
        common_times = {}
        distance_sums = {}
        distance_sums_sq = {}
        distance_counts = {}
        if ax4 is not None:
            for pop_idx in range(n_populations):
                start_time, end_time = population_time_bounds[pop_idx]
                if not (np.isfinite(start_time) and np.isfinite(end_time)):
                    continue
                common_times[pop_idx] = np.linspace(
                    (start_time - min_time) / 3600.0,
                    (end_time - min_time) / 3600.0,
                    100,
                )
                distance_sums[pop_idx] = np.zeros(100, dtype=float)
                distance_sums_sq[pop_idx] = np.zeros(100, dtype=float)
                distance_counts[pop_idx] = 0

        for particle_start, x_data, y_data, time_data in _iter_trajectory_chunks(sampled_ds):
            distance_segments, distance_indices, particle_distances = _distance_line_segments(
                x_data,
                y_data,
                time_data,
                min_time,
                geographic=geographic,
                earth_radius_m=earth_radius_m,
            )
            if ax2 is not None and distance_segments:
                ax2.add_collection(
                    LineCollection(
                        distance_segments,
                        colors=colors[particle_start + distance_indices],
                        alpha=0.7,
                        linewidths=1,
                    )
                )

            if ax4 is not None:
                per_population_segments = {pop_idx: [] for pop_idx in range(n_populations)}
                for local_idx, (time_hours, distances) in particle_distances.items():
                    pop_idx = int(population_ids[particle_start + local_idx])
                    if pop_idx not in common_times or time_hours.size < 2:
                        continue
                    per_population_segments[pop_idx].append(np.column_stack((time_hours, distances)))
                    interpolated = np.interp(common_times[pop_idx], time_hours, distances)
                    distance_sums[pop_idx] += interpolated
                    distance_sums_sq[pop_idx] += interpolated * interpolated
                    distance_counts[pop_idx] += 1
                for pop_idx, segments in per_population_segments.items():
                    if segments:
                        ax4.add_collection(
                            LineCollection(segments, colors=[pop_colors[pop_idx]], alpha=0.3, linewidths=0.8)
                        )

        if ax2 is not None:
            ax2.autoscale()
            ax2.grid(True, alpha=0.3)
        if ax4 is not None:
            for pop_idx, common_time in common_times.items():
                count = distance_counts[pop_idx]
                if count <= 0:
                    continue
                mean_distances = distance_sums[pop_idx] / count
                variance = np.maximum(distance_sums_sq[pop_idx] / count - mean_distances**2, 0.0)
                std_distances = np.sqrt(variance)
                ax4.plot(
                    common_time,
                    mean_distances,
                    color=pop_colors[pop_idx],
                    linewidth=3,
                    label=f'{population_names[pop_idx]} (mean)',
                )
                ax4.fill_between(
                    common_time,
                    mean_distances - std_distances,
                    mean_distances + std_distances,
                    color=pop_colors[pop_idx],
                    alpha=0.2,
                    label=f'{population_names[pop_idx]} (+/-1 std)',
                )
            if common_times:
                ax4.legend()
                ax4.autoscale()
            ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save plot if requested
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to: {output_path}')

    if show:
        plt.show()
    elif output_path is not None:
        plt.close(fig)

    return fig, axes_by_panel
