"""Mangrove windage support utilities."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sedtrails.particle_tracer.timer import convert_duration_string_to_seconds


def wind_components_from_speed_direction(
    speed: Any,
    direction_degrees: Any,
    convention: str = 'from_meteorological',
) -> tuple[np.ndarray, np.ndarray]:
    """Convert speed/direction wind input to u/v components."""
    speed_array = np.asarray(speed, dtype=float)
    direction_radians = np.deg2rad(np.asarray(direction_degrees, dtype=float))
    if convention == 'from_meteorological':
        return -speed_array * np.sin(direction_radians), -speed_array * np.cos(direction_radians)
    if convention == 'to_oceanographic':
        return speed_array * np.sin(direction_radians), speed_array * np.cos(direction_radians)
    raise ValueError(f'Unsupported wind direction convention: {convention!r}')


def combine_velocity_bounds(base_flow_bounds: Mapping[str, Any], wind_bounds: Mapping[str, Any], coefficient: float) -> dict:
    """Return hydrodynamic velocity bounds plus a scaled windage contribution."""
    lower_u = np.asarray(base_flow_bounds['lower']['u']) + float(coefficient) * np.asarray(wind_bounds['lower']['u'])
    lower_v = np.asarray(base_flow_bounds['lower']['v']) + float(coefficient) * np.asarray(wind_bounds['lower']['v'])
    upper_u = np.asarray(base_flow_bounds['upper']['u']) + float(coefficient) * np.asarray(wind_bounds['upper']['u'])
    upper_v = np.asarray(base_flow_bounds['upper']['v']) + float(coefficient) * np.asarray(wind_bounds['upper']['v'])
    return {
        'x': base_flow_bounds['x'],
        'y': base_flow_bounds['y'],
        'lower': {
            'u': lower_u,
            'v': lower_v,
            'magnitude': np.sqrt(lower_u**2 + lower_v**2),
        },
        'upper': {
            'u': upper_u,
            'v': upper_v,
            'magnitude': np.sqrt(upper_u**2 + upper_v**2),
        },
        'weight': float(base_flow_bounds.get('weight', 0.0)),
        'lower_index': base_flow_bounds.get('lower_index'),
        'upper_index': base_flow_bounds.get('upper_index'),
    }


def get_wind_bounds(time_seconds: float, forcing_config: Mapping[str, Any], retriever, *, direction_convention: str) -> dict:
    """Return wind bounds normalized to the tracer runtime shape."""
    forcing_type = forcing_config.get('type', 'constant')
    if forcing_type == 'constant':
        return _constant_wind_bounds(retriever, forcing_config, direction_convention=direction_convention)
    if forcing_type == 'timeseries_point':
        return _timeseries_point_wind_bounds(time_seconds, retriever, forcing_config, direction_convention=direction_convention)
    if forcing_type == 'field':
        return _field_wind_bounds(time_seconds, retriever, forcing_config)
    raise ValueError(f'Unsupported mangrove wind forcing type: {forcing_type!r}')


def _constant_wind_bounds(retriever, forcing_config: Mapping[str, Any], *, direction_convention: str) -> dict:
    base = _reference_grid(retriever)
    lower_u, lower_v = _forcing_uv(forcing_config, direction_convention=direction_convention)
    lower_u = np.asarray(lower_u, dtype=float)
    lower_v = np.asarray(lower_v, dtype=float)
    return {
        'x': base['x'],
        'y': base['y'],
        'lower': {'u': lower_u, 'v': lower_v, 'magnitude': np.sqrt(lower_u**2 + lower_v**2)},
        'upper': {'u': lower_u, 'v': lower_v, 'magnitude': np.sqrt(lower_u**2 + lower_v**2)},
        'weight': 0.0,
    }


def _timeseries_point_wind_bounds(
    time_seconds: float,
    retriever,
    forcing_config: Mapping[str, Any],
    *,
    direction_convention: str,
) -> dict:
    times = np.asarray([_time_value_seconds(value) for value in forcing_config.get('times', ())], dtype=float)
    if times.size == 0:
        raise ValueError('timeseries_point wind forcing requires a non-empty times array.')

    if time_seconds <= times[0]:
        lower_index = upper_index = 0
        weight = 0.0
    elif time_seconds >= times[-1]:
        lower_index = upper_index = times.size - 1
        weight = 0.0
    else:
        lower_index = int(np.searchsorted(times, time_seconds, side='right') - 1)
        upper_index = lower_index + 1
        interval = times[upper_index] - times[lower_index]
        weight = 0.0 if interval == 0.0 else float((time_seconds - times[lower_index]) / interval)

    lower_u, lower_v = _timeseries_uv_at_index(forcing_config, lower_index, direction_convention=direction_convention)
    upper_u, upper_v = _timeseries_uv_at_index(forcing_config, upper_index, direction_convention=direction_convention)
    base = _reference_grid(retriever)
    lower_u = np.asarray(lower_u, dtype=float)
    lower_v = np.asarray(lower_v, dtype=float)
    upper_u = np.asarray(upper_u, dtype=float)
    upper_v = np.asarray(upper_v, dtype=float)
    return {
        'x': base['x'],
        'y': base['y'],
        'lower': {'u': lower_u, 'v': lower_v, 'magnitude': np.sqrt(lower_u**2 + lower_v**2)},
        'upper': {'u': upper_u, 'v': upper_v, 'magnitude': np.sqrt(upper_u**2 + upper_v**2)},
        'weight': weight,
        'lower_index': lower_index,
        'upper_index': upper_index,
    }


def _field_wind_bounds(time_seconds: float, retriever, forcing_config: Mapping[str, Any]) -> dict:
    if forcing_config.get('flow_field_name'):
        return retriever.get_flow_field_bounds(time_seconds, forcing_config['flow_field_name'])

    u_field_name = forcing_config.get('u_field_name')
    v_field_name = forcing_config.get('v_field_name')
    if not u_field_name or not v_field_name:
        raise ValueError('field wind forcing requires flow_field_name or both u_field_name and v_field_name.')

    u_bounds = retriever.get_scalar_field_bounds(time_seconds, u_field_name)
    v_bounds = retriever.get_scalar_field_bounds(time_seconds, v_field_name)
    lower_u = np.asarray(u_bounds['lower'], dtype=float)
    lower_v = np.asarray(v_bounds['lower'], dtype=float)
    upper_u = np.asarray(u_bounds['upper'], dtype=float)
    upper_v = np.asarray(v_bounds['upper'], dtype=float)
    return {
        'x': u_bounds['x'],
        'y': u_bounds['y'],
        'lower': {'u': lower_u, 'v': lower_v, 'magnitude': np.sqrt(lower_u**2 + lower_v**2)},
        'upper': {'u': upper_u, 'v': upper_v, 'magnitude': np.sqrt(upper_u**2 + upper_v**2)},
        'weight': float(u_bounds.get('weight', 0.0)),
        'lower_index': u_bounds.get('lower_index'),
        'upper_index': u_bounds.get('upper_index'),
    }


def _forcing_uv(forcing_config: Mapping[str, Any], *, direction_convention: str) -> tuple[np.ndarray, np.ndarray]:
    if 'u' in forcing_config or 'v' in forcing_config:
        return np.asarray(forcing_config.get('u', 0.0), dtype=float), np.asarray(forcing_config.get('v', 0.0), dtype=float)
    return wind_components_from_speed_direction(
        forcing_config.get('speed', 0.0),
        forcing_config.get('direction', 0.0),
        convention=direction_convention,
    )


def _timeseries_uv_at_index(forcing_config: Mapping[str, Any], index: int, *, direction_convention: str) -> tuple[float, float]:
    if 'u_values' in forcing_config or 'v_values' in forcing_config:
        u_values = forcing_config.get('u_values', [])
        v_values = forcing_config.get('v_values', [])
        return float(u_values[index]), float(v_values[index])
    speed_values = forcing_config.get('speed_values', [])
    direction_values = forcing_config.get('direction_values', [])
    u_value, v_value = wind_components_from_speed_direction(
        speed_values[index],
        direction_values[index],
        convention=direction_convention,
    )
    return float(np.asarray(u_value)), float(np.asarray(v_value))


def _time_value_seconds(value: Any) -> float:
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return float(convert_duration_string_to_seconds(str(value)))


def _reference_grid(retriever) -> dict:
    return {
        'x': retriever.sedtrails_data.x,
        'y': retriever.sedtrails_data.y,
    }
