"""Utilities to build restart inputs from a SedTRAILS NetCDF output file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import os
from typing import Any

import numpy as np
import xarray as xr
import yaml

from sedtrails.application_interfaces.validator import SedtrailsYamlLoader
from sedtrails.particle_tracer.timer import convert_duration_string_to_seconds


@dataclass
class RestartSummary:
    """Summary of artifacts generated for a restart configuration."""

    output_config: Path
    seed_files: dict[str, Path]
    restart_time: str
    retained_particles: int


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if value is None:
        raise ValueError('Missing time.start in base configuration.')
    return datetime.fromisoformat(str(value))


def _format_datetime(value: datetime) -> str:
    return value.strftime('%Y-%m-%d %H:%M:%S')


def _infer_restart_datetime(config: dict[str, Any], restart_seconds: float | None) -> datetime:
    """Infer restart datetime from NetCDF time semantics and config."""
    base_start = _to_datetime(config.get('time', {}).get('start'))
    if restart_seconds is None:
        return base_start

    reference_date_value = config.get('general', {}).get('input_model', {}).get('reference_date')
    if reference_date_value is not None:
        reference_date = _to_datetime(reference_date_value)
        return reference_date + timedelta(seconds=restart_seconds)

    # Heuristic fallback: very large values usually indicate absolute epoch seconds.
    if restart_seconds > 1.0e8:
        return datetime.utcfromtimestamp(restart_seconds)

    return base_start + timedelta(seconds=restart_seconds)


def _last_valid_index_per_particle(x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    valid_xy = np.isfinite(x_data) & np.isfinite(y_data)
    has_valid = valid_xy.any(axis=1)
    indices = np.full(x_data.shape[0], -1, dtype=int)
    if np.any(has_valid):
        rev_idx = np.argmax(valid_xy[:, ::-1], axis=1)
        indices[has_valid] = x_data.shape[1] - 1 - rev_idx[has_valid]
    return indices


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


def create_restart_from_netcdf(
    netcdf_file: str,
    base_config_file: str,
    output_config_file: str = 'sedtrails-restart.yaml',
    seed_points_dir: str | None = None,
) -> RestartSummary:
    """Create a restart YAML and per-population seed files from NetCDF output.

    The generated configuration keeps existing settings from ``base_config_file`` but
    replaces each population seeding strategy with ``file_points`` using particle
    coordinates from the last valid timestep for each particle in ``netcdf_file``.
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

    ds = xr.open_dataset(netcdf_path)
    try:
        if 'x' not in ds or 'y' not in ds:
            raise ValueError("NetCDF output must contain 'x' and 'y' variables.")

        x_data = np.asarray(ds['x'].values, dtype=float)
        y_data = np.asarray(ds['y'].values, dtype=float)
        if x_data.ndim != 2 or y_data.ndim != 2:
            raise ValueError("Expected 'x' and 'y' to be 2D arrays (n_particles, n_timesteps).")

        n_particles = x_data.shape[0]

        if 'population_id' in ds:
            pop_ids = np.asarray(ds['population_id'].values).astype(int)
        else:
            pop_ids = np.zeros(n_particles, dtype=int)

        if pop_ids.shape[0] != n_particles:
            raise ValueError("'population_id' length does not match number of particles.")

        last_indices = _last_valid_index_per_particle(x_data, y_data)

        alive_mask = np.ones(n_particles, dtype=bool)
        if 'status_alive' in ds:
            alive = np.asarray(ds['status_alive'].values)
            for particle_idx, timestep_idx in enumerate(last_indices):
                if timestep_idx >= 0:
                    alive_mask[particle_idx] = bool(alive[particle_idx, timestep_idx])

        in_domain_mask = np.ones(n_particles, dtype=bool)
        if 'status_domain' in ds:
            status_domain = np.asarray(ds['status_domain'].values)
            for particle_idx, timestep_idx in enumerate(last_indices):
                if timestep_idx >= 0:
                    in_domain_mask[particle_idx] = bool(status_domain[particle_idx, timestep_idx])

        keep_mask = (last_indices >= 0) & alive_mask & in_domain_mask
        if not np.any(keep_mask):
            raise ValueError('No valid particle positions available to seed restart.')

        time_values = None
        if 'time' in ds:
            time_values = np.asarray(ds['time'].values, dtype=float)

        restart_seconds = None
        if time_values is not None and time_values.ndim == 2:
            selected_times = np.array(
                [time_values[i, last_indices[i]] for i in range(n_particles) if keep_mask[i]],
                dtype=float,
            )
            finite_times = selected_times[np.isfinite(selected_times)]
            if finite_times.size:
                restart_seconds = float(np.max(finite_times))

        restart_dt = _infer_restart_datetime(config, restart_seconds)
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

        seed_files: dict[str, Path] = {}

        for pop_idx, population in enumerate(populations):
            pop_name = str(population.get('name', f'population_{pop_idx + 1}'))

            selected = [
                (float(x_data[i, last_indices[i]]), float(y_data[i, last_indices[i]]))
                for i in range(n_particles)
                if keep_mask[i] and int(pop_ids[i]) == pop_idx
            ]

            population.setdefault('seeding', {})
            population['seeding']['release_start'] = restart_time

            if not selected:
                # Disable seeding for this population on restart (avoid reseeding new particles).
                population['seeding']['quantity'] = 0
                continue

            points_file = seeds_dir / f'{pop_name}.restart_points.csv'
            with open(points_file, 'w', encoding='utf-8') as handle:
                handle.write('x,y\n')
                for x_coord, y_coord in selected:
                    handle.write(f'{x_coord:.8f},{y_coord:.8f}\n')

            population['seeding']['quantity'] = 1
            population['seeding']['strategy'] = {
                'file_points': {
                    'path': _path_relative_to_cwd(points_file),
                    'x_col': 'x',
                    'y_col': 'y',
                    'has_header': True,
                    'deduplicate': False,
                }
            }

            seed_files[pop_name] = points_file

        if not seed_files:
            raise ValueError('No populations had active particles to include in restart files.')

        config.setdefault('time', {})
        config['time']['start'] = restart_time
        config['time']['duration'] = remaining_duration

        with open(output_path, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        retained_particles = int(np.count_nonzero(keep_mask))
        return RestartSummary(
            output_config=output_path,
            seed_files=seed_files,
            restart_time=restart_time,
            retained_particles=retained_particles,
        )
    finally:
        ds.close()
