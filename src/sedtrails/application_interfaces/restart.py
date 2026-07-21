"""Utilities to build restart inputs from a SedTRAILS NetCDF output file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import os
import re
from typing import Any

import numpy as np
import xarray as xr
import yaml
from pyproj import CRS

from sedtrails.application_interfaces.validator import SedtrailsYamlLoader
from sedtrails.particle_tracer.timer import convert_duration_string_to_seconds
from sedtrails.transport_converter.format_converter import FormatConverter


@dataclass
class RestartSummary:
    """Summary of artifacts generated for a restart configuration."""

    output_config: Path
    seed_files: dict[str, Path]
    restart_time: str
    retained_particles: int


@dataclass
class RestartParticleState:
    """Particle state extracted from a SedTRAILS output file for restart seeding."""

    x: np.ndarray
    y: np.ndarray
    pop_ids: np.ndarray
    alive_mask: np.ndarray
    in_domain_mask: np.ndarray
    restart_seconds: float | None


RESTART_SEED_WRITE_CHUNK_SIZE = 65_536


@dataclass(frozen=True)
class _RestartStreamContext:
    """Describe the final particle state that is streamed for a restart."""

    particle_dim: str
    time_dim: str | None
    time_index: int | None
    n_particles: int
    restart_seconds: float | None


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if value is None:
        raise ValueError('Missing time.start in base configuration.')
    text = str(value).strip()
    if text.endswith('Z'):
        text = text[:-1]
    return datetime.fromisoformat(text)


def _format_datetime(value: datetime) -> str:
    return value.strftime('%Y-%m-%d %H:%M:%S')


def _config_reference_date(config: dict[str, Any], default: str | None = None) -> datetime | None:
    reference_date_value = config.get('general', {}).get('input_model', {}).get('reference_date', default)
    if reference_date_value is None:
        return None
    return _to_datetime(reference_date_value)


def _parse_time_units_reference_date(units: Any) -> datetime | None:
    if units is None:
        return None

    match = re.match(r'^\s*seconds\s+since\s+(.+?)\s*$', str(units), flags=re.IGNORECASE)
    if not match:
        return None

    return _to_datetime(match.group(1))


def _dataset_reference_date(ds: xr.Dataset, config: dict[str, Any]) -> datetime | None:
    """Return the reference date used by output time values."""
    for attr_name in ('reference_date', 'time_reference_date'):
        if attr_name in ds.attrs:
            return _to_datetime(ds.attrs[attr_name])

    for attr_name in ('time_units', 'units'):
        reference_date = _parse_time_units_reference_date(ds.attrs.get(attr_name))
        if reference_date is not None:
            return reference_date

    if 'time' in ds:
        for attr_name in ('time_units', 'units'):
            reference_date = _parse_time_units_reference_date(ds['time'].attrs.get(attr_name))
            if reference_date is not None:
                return reference_date

    return _config_reference_date(config)


def _infer_restart_datetime(
    config: dict[str, Any],
    restart_seconds: float | None,
    dataset_reference_date: datetime | None = None,
) -> datetime:
    """Infer restart datetime from NetCDF time semantics and config."""
    base_start = _to_datetime(config.get('time', {}).get('start'))
    if restart_seconds is None:
        return base_start

    if dataset_reference_date is not None:
        return dataset_reference_date + timedelta(seconds=restart_seconds)

    reference_date = _config_reference_date(config)
    if reference_date is not None:
        return reference_date + timedelta(seconds=restart_seconds)

    # Heuristic fallback: very large values usually indicate absolute epoch seconds.
    if restart_seconds > 1.0e8:
        return datetime.utcfromtimestamp(restart_seconds)

    return base_start + timedelta(seconds=restart_seconds)


def _restart_format_config(config: dict[str, Any], config_path: Path) -> dict[str, Any] | None:
    input_model = config.get('general', {}).get('input_model', {})
    inputs = config.get('inputs', {})
    input_file = inputs.get('data')
    input_format = input_model.get('format')
    if not input_file or not input_format:
        return None

    input_path = Path(str(input_file))
    if not input_path.is_absolute():
        input_path = config_path.parent / input_path

    format_config = {
        'input_file': str(input_path),
        'input_format': input_format,
        'reference_date': input_model.get('reference_date', '1970-01-01'),
        'morfac': input_model.get('morfac', 1.0),
        'sediment_fraction_index': input_model.get('sediment_fraction_index', 0),
        'sediment_fraction_name': input_model.get('sediment_fraction_name'),
    }
    for name in (
        'coordinate_system',
        'source_crs',
        'metric_crs',
        'runtime_geometry',
        'surface_model',
        'earth_radius_m',
        'longitude_wrap',
        'velocity_basis',
    ):
        if name in input_model:
            format_config[name] = input_model[name]
    return format_config


def _validate_restart_coordinate_compatibility(ds: xr.Dataset, config: dict[str, Any]) -> None:
    """Reject checkpoint and base configurations with incompatible geometry."""
    dataset_system = ds.attrs.get('coordinate_system')
    if dataset_system is None:
        return

    input_model = config.get('general', {}).get('input_model', {})
    config_system = input_model.get('coordinate_system')
    if config_system not in (None, 'auto'):
        dataset_normalized = _normalized_coordinate_label(dataset_system)
        config_normalized = _normalized_coordinate_label(config_system)
        if dataset_normalized != config_normalized:
            raise ValueError(
                'Restart coordinate_system does not match the base configuration: '
                f'{dataset_system!r} versus {config_system!r}.'
            )

    for name in ('runtime_geometry', 'surface_model', 'velocity_basis'):
        dataset_value = ds.attrs.get(name)
        config_value = input_model.get(name)
        if dataset_value is None or config_value in (None, 'auto'):
            continue
        if str(dataset_value).strip().lower() != str(config_value).strip().lower():
            raise ValueError(
                f'Restart {name} does not match the base configuration: '
                f'{dataset_value!r} versus {config_value!r}.'
            )

    for name in ('source_crs', 'metric_crs'):
        dataset_value = ds.attrs.get(name)
        config_value = input_model.get(name)
        if dataset_value in (None, '') or config_value in (None, '', 'auto', 'auto_utm'):
            continue
        try:
            compatible = CRS.from_user_input(dataset_value) == CRS.from_user_input(config_value)
        except Exception:
            compatible = str(dataset_value).strip() == str(config_value).strip()
        if not compatible:
            raise ValueError(
                f'Restart {name} does not match the base configuration: '
                f'{dataset_value!r} versus {config_value!r}.'
            )

    dataset_radius = ds.attrs.get('earth_radius_m')
    config_radius = input_model.get('earth_radius_m')
    if dataset_radius is not None and config_radius is not None and not np.isclose(
        float(dataset_radius),
        float(config_radius),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            'Restart earth_radius_m does not match the base configuration: '
            f'{dataset_radius!r} versus {config_radius!r}.'
        )


def _normalized_coordinate_label(value: Any) -> str:
    """Normalize public projected/geographic coordinate labels."""
    normalized = str(value).strip().lower().replace('_', '-')
    if normalized in {
        'geo',
        'geographic',
        'spherical',
        'lon-lat',
        'lonlat',
        'longlat',
        'latitude-longitude',
    }:
        return 'geographic'
    if normalized in {'projected', 'cartesian'}:
        return 'projected'
    return normalized


def _validate_restart_time_matches_input(
    config: dict[str, Any],
    config_path: Path,
    restart_dt: datetime,
) -> None:
    """Validate that the generated restart start is covered by the original forcing."""
    format_config = _restart_format_config(config, config_path)
    if format_config is None:
        return

    reference_date = _config_reference_date(config, default='1970-01-01')
    if reference_date is None:
        return

    forcing_start, forcing_end = FormatConverter(format_config).get_time_bounds()
    restart_seconds = (restart_dt - reference_date).total_seconds()
    if forcing_start <= restart_seconds <= forcing_end:
        return

    restart_time = _format_datetime(restart_dt)
    raise ValueError(
        f'Restart time {restart_time} ({restart_seconds:.3f}s since reference_date '
        f'{_format_datetime(reference_date)}) is outside the original input forcing window '
        f'[{forcing_start:.3f}, {forcing_end:.3f}]s. '
        'The generated restart would start before or after available forcing data.'
    )


def _status_values_to_mask(values: Any, default: bool = True) -> np.ndarray:
    """Convert status values to a boolean mask while treating NaN fill values as false."""
    arr = np.asarray(values)
    if arr.dtype.kind in ('f', 'c'):
        return np.isfinite(arr) & (arr != 0)
    return arr.astype(bool)


def _population_ids(ds: xr.Dataset, n_particles: int) -> np.ndarray:
    if 'population_id' not in ds:
        return np.zeros(n_particles, dtype=int)

    pop_ids = np.asarray(ds['population_id'].values).astype(int)
    if pop_ids.shape[0] != n_particles:
        raise ValueError("'population_id' length does not match number of particles.")
    return pop_ids


def _last_written_time_index(ds: xr.Dataset) -> int:
    """Return the last written v2 timestep without reading the full trajectory arrays."""
    n_timesteps = int(ds.sizes.get('n_timesteps', ds.sizes.get('time', 0)))
    if n_timesteps <= 0:
        raise ValueError('NetCDF output contains no timesteps.')

    written_slots = ds.attrs.get('written_slots')
    if written_slots is not None:
        try:
            written_slots = int(written_slots)
        except (TypeError, ValueError):
            written_slots = 0
        if written_slots > 0:
            return min(written_slots, n_timesteps) - 1

    if 'time' in ds and ds['time'].ndim == 1:
        time_values = np.asarray(ds['time'].values, dtype=float)
        finite = np.flatnonzero(np.isfinite(time_values))
        if finite.size:
            return int(finite[-1])

    return n_timesteps - 1


def _extract_checkpoint_state(ds: xr.Dataset) -> RestartParticleState:
    x_data = np.asarray(ds['x'].values, dtype=float)
    y_data = np.asarray(ds['y'].values, dtype=float)
    if x_data.ndim != 1 or y_data.ndim != 1:
        raise ValueError("Checkpoint output must contain 1D 'x' and 'y' particle arrays.")

    n_particles = x_data.shape[0]
    alive_mask = np.ones(n_particles, dtype=bool)
    if 'status_alive' in ds:
        alive_mask = _status_values_to_mask(ds['status_alive'].values)

    in_domain_mask = np.ones(n_particles, dtype=bool)
    if 'status_domain' in ds:
        in_domain_mask = _status_values_to_mask(ds['status_domain'].values)

    restart_seconds = None
    if 'time' in ds:
        time_values = np.asarray(ds['time'].values, dtype=float)
        finite_times = time_values[np.isfinite(time_values)]
        if finite_times.size:
            restart_seconds = float(np.max(finite_times))

    return RestartParticleState(
        x=x_data,
        y=y_data,
        pop_ids=_population_ids(ds, n_particles),
        alive_mask=np.asarray(alive_mask, dtype=bool),
        in_domain_mask=np.asarray(in_domain_mask, dtype=bool),
        restart_seconds=restart_seconds,
    )


def _is_time_particle_layout(ds: xr.Dataset) -> bool:
    if 'time' not in ds or ds['time'].ndim != 1:
        return False
    if ds['x'].ndim != 2 or ds['y'].ndim != 2:
        return False
    return ds['x'].dims[0] == ds['time'].dims[0] and ds['y'].dims[0] == ds['time'].dims[0]


def _extract_time_particle_state(ds: xr.Dataset) -> RestartParticleState:
    slot_idx = _last_written_time_index(ds)
    time_dim = ds['time'].dims[0]

    x_data = np.asarray(ds['x'].isel({time_dim: slot_idx}).values, dtype=float)
    y_data = np.asarray(ds['y'].isel({time_dim: slot_idx}).values, dtype=float)
    if x_data.ndim != 1 or y_data.ndim != 1:
        raise ValueError("Expected time-major 'x' and 'y' arrays with particle slices.")

    n_particles = x_data.shape[0]
    alive_mask = np.ones(n_particles, dtype=bool)
    if 'status_alive' in ds:
        alive_mask = _status_values_to_mask(ds['status_alive'].isel({time_dim: slot_idx}).values)

    in_domain_mask = np.ones(n_particles, dtype=bool)
    if 'status_domain' in ds:
        in_domain_mask = _status_values_to_mask(ds['status_domain'].isel({time_dim: slot_idx}).values)

    restart_seconds = None
    time_values = np.asarray(ds['time'].values, dtype=float)
    if 0 <= slot_idx < time_values.size and np.isfinite(time_values[slot_idx]):
        restart_seconds = float(time_values[slot_idx])

    return RestartParticleState(
        x=x_data,
        y=y_data,
        pop_ids=_population_ids(ds, n_particles),
        alive_mask=np.asarray(alive_mask, dtype=bool),
        in_domain_mask=np.asarray(in_domain_mask, dtype=bool),
        restart_seconds=restart_seconds,
    )


def _extract_restart_state(ds: xr.Dataset) -> RestartParticleState:
    """Extract restart particle state from v2 trajectory output or a checkpoint."""
    if ds.attrs.get('sedtrails_file_kind') == 'checkpoint' or (ds['x'].ndim == 1 and ds['y'].ndim == 1):
        return _extract_checkpoint_state(ds)
    if _is_time_particle_layout(ds):
        return _extract_time_particle_state(ds)

    raise ValueError(
        'Restart generation requires SedTRAILS time-major trajectory output '
        "('x'/'y' shaped as n_timesteps x n_particles with 1D 'time') "
        'or sedtrails_checkpoint.nc. Legacy particle-major trajectory files are not supported.'
    )


def _checkpoint_restart_seconds(ds: xr.Dataset) -> float | None:
    """Return the latest finite checkpoint time without loading a long time axis."""
    if 'time' not in ds:
        return None

    time_var = ds['time']
    if time_var.ndim == 0:
        value = float(np.asarray(time_var.values))
        return value if np.isfinite(value) else None
    if time_var.ndim != 1:
        return None

    latest = -np.inf
    time_dim = time_var.dims[0]
    for start in range(0, int(time_var.size), RESTART_SEED_WRITE_CHUNK_SIZE):
        stop = min(start + RESTART_SEED_WRITE_CHUNK_SIZE, int(time_var.size))
        values = np.asarray(time_var.isel({time_dim: slice(start, stop)}).values, dtype=float)
        if np.any(np.isfinite(values)):
            latest = max(latest, float(np.nanmax(values)))
    return None if not np.isfinite(latest) else latest


def _restart_stream_context(ds: xr.Dataset) -> _RestartStreamContext:
    """Describe the final state to stream for restart generation."""
    is_checkpoint = (
        ds.attrs.get('sedtrails_file_kind') == 'checkpoint'
        or (ds['x'].ndim == 1 and ds['y'].ndim == 1)
    )
    if is_checkpoint:
        if ds['x'].ndim != 1 or ds['y'].ndim != 1 or ds['x'].shape != ds['y'].shape:
            raise ValueError("Checkpoint output must contain matching 1D 'x' and 'y' particle arrays.")
        return _RestartStreamContext(
            particle_dim=ds['x'].dims[0],
            time_dim=None,
            time_index=None,
            n_particles=int(ds['x'].shape[0]),
            restart_seconds=_checkpoint_restart_seconds(ds),
        )

    if _is_time_particle_layout(ds):
        time_dim = ds['time'].dims[0]
        particle_dim = ds['x'].dims[1]
        if ds['x'].dims[1] != ds['y'].dims[1] or ds['x'].shape != ds['y'].shape:
            raise ValueError("Expected matching time-major 'x' and 'y' particle arrays.")
        time_index = _last_written_time_index(ds)
        time_value = float(np.asarray(ds['time'].isel({time_dim: time_index}).values))
        return _RestartStreamContext(
            particle_dim=particle_dim,
            time_dim=time_dim,
            time_index=time_index,
            n_particles=int(ds['x'].shape[1]),
            restart_seconds=time_value if np.isfinite(time_value) else None,
        )

    raise ValueError(
        'Restart generation requires SedTRAILS time-major trajectory output '
        "('x'/'y' shaped as n_timesteps x n_particles with 1D 'time') "
        'or sedtrails_checkpoint.nc. Legacy particle-major trajectory files are not supported.'
    )


def _restart_chunk_variable(
    ds: xr.Dataset,
    name: str,
    context: _RestartStreamContext,
    particle_slice: slice,
) -> np.ndarray:
    """Read one particle-sized field chunk at the restart timestep."""
    variable = ds[name]
    if context.particle_dim not in variable.dims:
        raise ValueError(f"Restart field '{name}' does not use the particle dimension.")
    indexers = {context.particle_dim: particle_slice}
    if context.time_dim is not None and context.time_dim in variable.dims:
        indexers[context.time_dim] = context.time_index
    values = np.asarray(variable.isel(indexers).values)
    expected_shape = (particle_slice.stop - particle_slice.start,)
    if values.shape != expected_shape:
        raise ValueError(f"Restart field '{name}' does not resolve to a particle vector.")
    return values


def _stream_restart_chunks(ds: xr.Dataset, context: _RestartStreamContext):
    """Yield valid-mask restart chunks without materializing all particle arrays."""
    for start in range(0, context.n_particles, RESTART_SEED_WRITE_CHUNK_SIZE):
        particle_slice = slice(start, min(start + RESTART_SEED_WRITE_CHUNK_SIZE, context.n_particles))
        x_values = np.asarray(_restart_chunk_variable(ds, 'x', context, particle_slice), dtype=float)
        y_values = np.asarray(_restart_chunk_variable(ds, 'y', context, particle_slice), dtype=float)
        if 'population_id' in ds:
            pop_ids = np.asarray(
                _restart_chunk_variable(ds, 'population_id', context, particle_slice),
                dtype=int,
            )
        else:
            pop_ids = np.zeros(x_values.size, dtype=int)

        alive_mask = np.ones(x_values.size, dtype=bool)
        if 'status_alive' in ds:
            alive_mask = _status_values_to_mask(
                _restart_chunk_variable(ds, 'status_alive', context, particle_slice)
            )
        in_domain_mask = np.ones(x_values.size, dtype=bool)
        if 'status_domain' in ds:
            in_domain_mask = _status_values_to_mask(
                _restart_chunk_variable(ds, 'status_domain', context, particle_slice)
            )
        if alive_mask.shape != x_values.shape or in_domain_mask.shape != x_values.shape:
            raise ValueError('Restart status fields must resolve to particle vectors.')

        keep_mask = np.isfinite(x_values) & np.isfinite(y_values) & alive_mask & in_domain_mask
        yield x_values, y_values, pop_ids, keep_mask


def _write_streamed_restart_seed_files(
    ds: xr.Dataset,
    context: _RestartStreamContext,
    populations: list[dict[str, Any]],
    seeds_dir: Path,
) -> tuple[dict[int, Path], np.ndarray, int]:
    """Stream final-state chunks once into temporary per-population restart CSVs."""
    seed_files: dict[int, Path] = {}
    temporary_files: dict[int, Path] = {}
    handles: dict[int, Any] = {}
    population_counts = np.zeros(len(populations), dtype=np.int64)
    retained_particles = 0
    completed = False
    try:
        for pop_idx, population in enumerate(populations):
            pop_name = str(population.get('name', f'population_{pop_idx + 1}'))
            points_file = seeds_dir / f'{pop_name}.restart_points.csv'
            temporary_file = points_file.with_name(f'{points_file.name}.tmp')
            temporary_file.unlink(missing_ok=True)
            handle = open(temporary_file, 'w', encoding='utf-8')
            handle.write('x,y\n')
            seed_files[pop_idx] = points_file
            temporary_files[pop_idx] = temporary_file
            handles[pop_idx] = handle

        for x_values, y_values, pop_ids, keep_mask in _stream_restart_chunks(ds, context):
            retained_particles += int(np.count_nonzero(keep_mask))
            configured = keep_mask & (pop_ids >= 0) & (pop_ids < len(populations))
            if not np.any(configured):
                continue
            population_counts += np.bincount(
                pop_ids[configured],
                minlength=len(populations),
            )[:len(populations)]
            for pop_idx, handle in handles.items():
                selected = configured & (pop_ids == pop_idx)
                if np.any(selected):
                    np.savetxt(
                        handle,
                        np.column_stack((x_values[selected], y_values[selected])),
                        fmt='%.8f,%.8f',
                    )

        for handle in handles.values():
            handle.close()
        handles.clear()
        for pop_idx, temporary_file in temporary_files.items():
            if population_counts[pop_idx] <= 0:
                temporary_file.unlink(missing_ok=True)
                seed_files.pop(pop_idx)
            else:
                temporary_file.replace(seed_files[pop_idx])
        completed = True
        return seed_files, population_counts, retained_particles
    finally:
        for handle in handles.values():
            handle.close()
        if not completed:
            for temporary_file in temporary_files.values():
                temporary_file.unlink(missing_ok=True)

def _path_relative_to_cwd(path: Path) -> str:
    """Return path as POSIX text relative to current working directory when possible."""
    try:
        relative = Path(os.path.relpath(path.resolve(), Path.cwd().resolve()))
    except ValueError:
        # Different drives on Windows cannot be relativized.
        return str(path.resolve()).replace('\\', '/')

    text = str(relative).replace('\\', '/')
    if not relative.is_absolute() and not text.startswith(('.', '..')):
        text = f'./{text}'
    return text


def _seconds_to_duration_string(total_seconds: int) -> str:
    """Convert seconds to a compact SedTRAILS duration string (e.g., 1D2H30M)."""
    if total_seconds <= 0:
        return '0S'

    remaining = int(total_seconds)
    days, remaining = divmod(remaining, 86400)
    hours, remaining = divmod(remaining, 3600)
    minutes, seconds = divmod(remaining, 60)

    parts = []
    if days:
        parts.append(f'{days}D')
    if hours:
        parts.append(f'{hours}H')
    if minutes:
        parts.append(f'{minutes}M')
    if seconds:
        parts.append(f'{seconds}S')

    return ''.join(parts) if parts else '0S'


def _open_restart_dataset(netcdf_path: Path) -> xr.Dataset:
    """Open a SedTRAILS result NetCDF without xarray backend plugin discovery."""
    return xr.open_dataset(netcdf_path, engine='netcdf4', decode_times=False)


def _validate_restart_dataset_schema(ds: xr.Dataset, netcdf_path: Path) -> None:
    """Validate that the input file has the particle trajectory fields needed for restart."""
    missing = [name for name in ('x', 'y') if name not in ds]
    if not missing:
        return

    available = ', '.join(map(str, list(ds.data_vars)[:10])) or 'none'
    if len(ds.data_vars) > 10:
        available = f'{available}, ...'
    raise ValueError(
        f"'{netcdf_path}' is not a SedTRAILS trajectory output file. "
        "Restart generation expects the NetCDF file written by `sedtrails run` "
        "(usually `results/sedtrails_results.nc`) passed with `--file`. "
        f"Missing required particle variable(s): {', '.join(missing)}. "
        f'Available data variables: {available}. '
        'Eulerian forcing files belong in `inputs.data` of the base config, not in `--file`.'
    )


def create_restart_from_netcdf(
    netcdf_file: str,
    base_config_file: str,
    output_config_file: str = 'sedtrails-restart.yaml',
    seed_points_dir: str | None = None,
) -> RestartSummary:
    """
    Create a restart YAML and per-population seed files from NetCDF output.

    The generated configuration keeps existing settings from ``base_config_file`` but
    replaces each population seeding strategy with ``file_points`` using particle
    coordinates from the last valid timestep for each particle in ``netcdf_file``.

    Parameters
    ----------
    netcdf_file : str
        Path to the NetCDF results file.
    base_config_file : str
        Path to the base configuration file.
    output_config_file : str
        Path where the restart configuration file is written.
    seed_points_dir : str | None
        Directory containing restart seed-point files.

    Returns
    -------
    RestartSummary
        Computed value returned by the function.
    """

    netcdf_path = Path(netcdf_file)
    config_path = Path(base_config_file)
    output_path = Path(output_config_file)

    if not netcdf_path.is_file():
        raise FileNotFoundError(f'NetCDF file not found: {netcdf_path}')
    if not config_path.is_file():
        raise FileNotFoundError(f'Base configuration file not found: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as handle:
        config = yaml.load(handle, Loader=SedtrailsYamlLoader)

    if not isinstance(config, dict):
        raise ValueError('Base configuration must parse to a dictionary.')

    populations = config.get('particles', {}).get('populations', [])
    if not populations:
        raise ValueError('Base configuration has no particles.populations entries.')

    ds = _open_restart_dataset(netcdf_path)
    try:
        _validate_restart_dataset_schema(ds, netcdf_path)
        _validate_restart_coordinate_compatibility(ds, config)

        restart_context = _restart_stream_context(ds)
        restart_dt = _infer_restart_datetime(
            config,
            restart_context.restart_seconds,
            _dataset_reference_date(ds, config),
        )
        _validate_restart_time_matches_input(config, config_path, restart_dt)
        restart_time = _format_datetime(restart_dt)

        original_duration = config.get('time', {}).get('duration')
        if original_duration is None:
            raise ValueError('Missing time.duration in base configuration.')
        original_duration_seconds = convert_duration_string_to_seconds(str(original_duration))

        original_start = _to_datetime(config.get('time', {}).get('start'))
        elapsed_seconds = max(0, int((restart_dt - original_start).total_seconds()))
        remaining_seconds = max(0, original_duration_seconds - elapsed_seconds)
        if remaining_seconds <= 0:
            raise ValueError('No remaining duration left in original run. Restart is not needed.')
        remaining_duration = _seconds_to_duration_string(remaining_seconds)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if seed_points_dir is None:
            seeds_dir = output_path.parent / f'{output_path.stem}_seeds'
        else:
            seeds_dir = Path(seed_points_dir)
        seeds_dir.mkdir(parents=True, exist_ok=True)

        streamed_seed_files, population_counts, retained_particles = _write_streamed_restart_seed_files(
            ds,
            restart_context,
            populations,
            seeds_dir,
        )
        if retained_particles <= 0:
            raise ValueError('No valid particle positions available to seed restart.')
        if not streamed_seed_files:
            raise ValueError('No populations had active particles to include in restart files.')

        for pop_idx, population in enumerate(populations):
            population.setdefault('seeding', {})
            population['seeding']['release_start'] = restart_time
            if population_counts[pop_idx] <= 0:
                # Disable seeding for this population on restart (avoid reseeding new particles).
                population['seeding']['quantity'] = 0
                continue

            population['seeding']['quantity'] = 1
            population['seeding']['strategy'] = {
                'file_points': {
                    'path': _path_relative_to_cwd(streamed_seed_files[pop_idx]),
                    'x_col': 'x',
                    'y_col': 'y',
                    'has_header': True,
                    'deduplicate': False,
                }
            }

        seed_files = {
            str(population.get('name', f'population_{pop_idx + 1}')): streamed_seed_files[pop_idx]
            for pop_idx, population in enumerate(populations)
            if pop_idx in streamed_seed_files
        }
        config.setdefault('time', {})
        config['time']['start'] = restart_time
        config['time']['duration'] = remaining_duration

        with open(output_path, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        return RestartSummary(
            output_config=output_path,
            seed_files=seed_files,
            restart_time=restart_time,
            retained_particles=retained_particles,
        )
    finally:
        ds.close()
