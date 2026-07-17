"""Offline light-exposure calculations for particle trajectories.

The functions in this module are intentionally array-based. They can be used
with values sampled from a SedTRAILS trajectory file, from legacy MATLAB output,
or from a future online simulation callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


AttenuationStrategy = Literal['mud', 'sand', 'max', 'mean', 'sum', 'additive_excess']
ZConvention = Literal['depth_below_surface', 'height_above_bed', 'elevation']


@dataclass(frozen=True)
class StorlazziAttenuationCoefficients:
    """Linear attenuation coefficients digitized from Storlazzi et al. (2015)."""

    mud_slope: float = 0.0157
    mud_intercept: float = 1.2651
    sand_slope: float = 0.0036
    sand_intercept: float = 1.5352


@dataclass(frozen=True)
class ExposureResult:
    """Container for per-timestep and cumulative exposure diagnostics."""

    light_intensity: np.ndarray
    cumulative_light_dose: np.ndarray
    equivalent_sunlight_hours: np.ndarray
    time_above_threshold: np.ndarray
    bleaching_probability: np.ndarray
    osl_signal_reduction: np.ndarray
    attenuation_coefficient: np.ndarray
    particle_depth_below_surface: np.ndarray


DEFAULT_STORLAZZI_COEFFICIENTS = StorlazziAttenuationCoefficients()


def _as_float_array(value: ArrayLike | None, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        raise ValueError(f'{name} must not be empty')
    return array


def _as_bool_array(value: ArrayLike | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=bool)


def _reshape_time_vector_like(value: np.ndarray, target_ndim: int) -> np.ndarray:
    """Reshape a time vector for broadcasting against time-major particle arrays."""
    if value.ndim == 1 and target_ndim > 1:
        return value.reshape((value.shape[0],) + (1,) * (target_ndim - 1))
    return value


def _time_to_elapsed_seconds(value: ArrayLike) -> np.ndarray:
    """Convert numeric or datetime-like time vectors to elapsed seconds."""
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError('time must be one-dimensional')
    if array.size == 0:
        raise ValueError('time must not be empty')
    if np.issubdtype(array.dtype, np.datetime64):
        seconds = array.astype('datetime64[ns]').astype(np.int64).astype(float) / 1e9
    elif np.issubdtype(array.dtype, np.timedelta64):
        seconds = array.astype('timedelta64[ns]').astype(np.int64).astype(float) / 1e9
    else:
        seconds = array.astype(float)
    return seconds - seconds[0]


def attenuation_from_storlazzi(
    sed_conc_mud: ArrayLike | None = None,
    sed_conc_sand: ArrayLike | None = None,
    *,
    coefficients: StorlazziAttenuationCoefficients = DEFAULT_STORLAZZI_COEFFICIENTS,
) -> dict[str, np.ndarray | None]:
    """Compute mud and sand attenuation components from concentration in kg/m3.

    Parameters
    ----------
    sed_conc_mud, sed_conc_sand : array-like or None
        Suspended sediment concentration arrays in kg/m3.
    coefficients : StorlazziAttenuationCoefficients
        Linear coefficients for each sediment class.

    Returns
    -------
    dict
        Keys ``mud`` and ``sand``. Missing inputs return ``None`` values.
    """

    mud = _as_float_array(sed_conc_mud, 'sed_conc_mud')
    sand = _as_float_array(sed_conc_sand, 'sed_conc_sand')

    return {
        'mud': None if mud is None else coefficients.mud_slope * mud + coefficients.mud_intercept,
        'sand': None if sand is None else coefficients.sand_slope * sand + coefficients.sand_intercept,
    }


def combine_attenuation_components(
    mud: ArrayLike | None,
    sand: ArrayLike | None,
    *,
    strategy: AttenuationStrategy = 'additive_excess',
    coefficients: StorlazziAttenuationCoefficients = DEFAULT_STORLAZZI_COEFFICIENTS,
) -> np.ndarray:
    """Combine mud and sand attenuation components into one coefficient.

    ``mud`` and ``sand`` should usually be the outputs of
    :func:`attenuation_from_storlazzi`. For separate sand-only and mud-only
    analyses, use ``strategy='sand'`` or ``strategy='mud'``.

    ``additive_excess`` keeps the lower clear-water intercept once and adds the
    concentration-dependent excess from each available component.
    """

    mud_array = _as_float_array(mud, 'mud')
    sand_array = _as_float_array(sand, 'sand')

    if strategy == 'mud':
        if mud_array is None:
            raise ValueError("strategy='mud' requires a mud attenuation component")
        return mud_array

    if strategy == 'sand':
        if sand_array is None:
            raise ValueError("strategy='sand' requires a sand attenuation component")
        return sand_array

    if mud_array is None and sand_array is None:
        raise ValueError('At least one attenuation component is required')

    if mud_array is None:
        return np.asarray(sand_array, dtype=float)
    if sand_array is None:
        return np.asarray(mud_array, dtype=float)

    mud_array, sand_array = np.broadcast_arrays(mud_array, sand_array)

    if strategy == 'max':
        return np.maximum(mud_array, sand_array)
    if strategy == 'mean':
        return 0.5 * (mud_array + sand_array)
    if strategy == 'sum':
        return mud_array + sand_array
    if strategy == 'additive_excess':
        baseline = min(coefficients.mud_intercept, coefficients.sand_intercept)
        mud_excess = mud_array - coefficients.mud_intercept
        sand_excess = sand_array - coefficients.sand_intercept
        return baseline + mud_excess + sand_excess

    raise ValueError(f'Unsupported attenuation strategy: {strategy}')


def rouse_centroid_depth(
    water_depth: ArrayLike,
    settling_velocity: ArrayLike,
    shear_velocity: ArrayLike,
    *,
    von_karman: float = 0.41,
) -> np.ndarray:
    """Estimate particle depth below the water surface using MacDonald et al. (2006).

    Implements the equation supplied in the legacy workflow:
    ``z_c = depth * 0.0398 * 10 ** (-1.08 * tanh(1.2 * log(ws / (kappa * ustar)) - 0.4))``.
    Returned values are clipped to ``[0, water_depth]``.
    """

    depth = _as_float_array(water_depth, 'water_depth')
    ws = _as_float_array(settling_velocity, 'settling_velocity')
    ustar = _as_float_array(shear_velocity, 'shear_velocity')
    depth, ws, ustar = np.broadcast_arrays(depth, ws, ustar)

    denominator = von_karman * ustar
    ratio = np.full_like(depth, np.nan, dtype=float)
    valid = np.isfinite(ws) & np.isfinite(denominator) & (ws > 0.0) & (denominator > 0.0)
    np.divide(ws, denominator, out=ratio, where=valid)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        zc = depth * 0.0398 * 10.0 ** (-1.08 * np.tanh(1.2 * np.log(ratio) - 0.4))

    return np.clip(zc, 0.0, depth)


def particle_depth_below_surface(
    water_depth: ArrayLike,
    *,
    z: ArrayLike | None = None,
    z_c: ArrayLike | None = None,
    bed_level: ArrayLike | None = None,
    z_convention: ZConvention = 'depth_below_surface',
) -> np.ndarray:
    """Resolve particle depth below the water surface.

    Parameters
    ----------
    water_depth : array-like
        Local water depth in meters.
    z : array-like or None
        Preferred particle vertical coordinate.
    z_c : array-like or None
        Fallback centroid depth when ``z`` is unavailable.
    bed_level : array-like or None
        Required only when ``z_convention='elevation'``.
    z_convention : {'depth_below_surface', 'height_above_bed', 'elevation'}
        Interpretation of ``z``.
    """

    depth = _as_float_array(water_depth, 'water_depth')
    z_value = _as_float_array(z, 'z')
    zc_value = _as_float_array(z_c, 'z_c')

    if z_value is None:
        if zc_value is None:
            raise ValueError('Either z or z_c must be provided')
        particle_depth = zc_value
    elif z_convention == 'depth_below_surface':
        particle_depth = z_value
    elif z_convention == 'height_above_bed':
        particle_depth = depth - z_value
    elif z_convention == 'elevation':
        bed = _as_float_array(bed_level, 'bed_level')
        if bed is None:
            raise ValueError("bed_level is required when z_convention='elevation'")
        water_surface = bed + depth
        particle_depth = water_surface - z_value
    else:
        raise ValueError(f'Unsupported z convention: {z_convention}')

    depth, particle_depth = np.broadcast_arrays(depth, particle_depth)
    return np.clip(particle_depth, 0.0, np.maximum(depth, 0.0))


def surface_light_series(
    time: ArrayLike,
    *,
    constant: float = 1.0,
    daylight_mask: ArrayLike | None = None,
    series_time: ArrayLike | None = None,
    series_values: ArrayLike | None = None,
) -> np.ndarray:
    """Return surface light for each output time.

    A constant value is used unless ``series_time`` and ``series_values`` are
    provided. ``daylight_mask=False`` forces light to zero at those samples.
    """

    time_array = _as_float_array(time, 'time')
    if series_time is None and series_values is None:
        light = np.full(time_array.shape, float(constant), dtype=float)
    elif series_time is not None and series_values is not None:
        source_time = _as_float_array(series_time, 'series_time')
        source_values = _as_float_array(series_values, 'series_values')
        if source_time.ndim != 1 or source_values.ndim != 1:
            raise ValueError('series_time and series_values must be one-dimensional')
        if source_time.size != source_values.size:
            raise ValueError('series_time and series_values must have the same length')
        light = np.interp(time_array, source_time, source_values, left=source_values[0], right=source_values[-1])
    else:
        raise ValueError('series_time and series_values must be provided together')

    daylight = _as_bool_array(daylight_mask)
    if daylight is not None:
        light, daylight = np.broadcast_arrays(light, daylight)
        light = np.where(daylight, light, 0.0)

    return light


def light_intensity_at_particle(
    surface_light: ArrayLike,
    attenuation_coefficient: ArrayLike,
    particle_depth: ArrayLike,
    *,
    burial_depth: ArrayLike | None = None,
    burial_depth_threshold: float = 0.0,
    is_buried: ArrayLike | None = None,
    in_domain: ArrayLike | None = None,
    is_beached: ArrayLike | None = None,
) -> np.ndarray:
    """Compute light intensity at the particle for each sampled timestep.

    Buried particles receive zero light. ``burial_depth_threshold`` can be used
    to ignore numerical burial-depth noise. Beached particles receive full
    surface light unless buried. Particles outside the domain receive zero
    additional exposure so cumulative diagnostics remain at their last valid
    value.
    """

    surface = _as_float_array(surface_light, 'surface_light')
    kd = _as_float_array(attenuation_coefficient, 'attenuation_coefficient')
    depth = _as_float_array(particle_depth, 'particle_depth')
    target_ndim = max(surface.ndim, kd.ndim, depth.ndim)
    surface = _reshape_time_vector_like(surface, target_ndim)
    kd = _reshape_time_vector_like(kd, target_ndim)
    depth = _reshape_time_vector_like(depth, target_ndim)
    surface, kd, depth = np.broadcast_arrays(surface, kd, depth)

    with np.errstate(over='ignore', invalid='ignore'):
        intensity = surface * np.exp(-kd * np.maximum(depth, 0.0))

    beached = _as_bool_array(is_beached)
    if beached is not None:
        intensity, surface, beached = np.broadcast_arrays(intensity, surface, beached)
        intensity = np.where(beached, surface, intensity)

    buried = _as_bool_array(is_buried)
    burial = _as_float_array(burial_depth, 'burial_depth')
    if burial is not None:
        burial = np.broadcast_to(burial, intensity.shape)
        burial_buried = np.isfinite(burial) & (burial > float(burial_depth_threshold))
        buried = burial_buried if buried is None else np.broadcast_to(buried, intensity.shape) | burial_buried
    if buried is not None:
        intensity = np.where(np.broadcast_to(buried, intensity.shape), 0.0, intensity)

    domain = _as_bool_array(in_domain)
    if domain is not None:
        intensity = np.where(np.broadcast_to(domain, intensity.shape), intensity, 0.0)

    return np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)


def integrate_exposure(
    time: ArrayLike,
    light_intensity: ArrayLike,
    *,
    light_threshold: float = 0.0,
    reference_surface_light: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate light intensity over saved output timesteps.

    The first timestep has ``dt=0``. Each later sample represents exposure over
    the interval since the previous saved time.
    """

    elapsed_seconds = _time_to_elapsed_seconds(time)
    light = _as_float_array(light_intensity, 'light_intensity')
    if light.shape[0] != elapsed_seconds.size:
        raise ValueError('light_intensity first dimension must match time length')
    if np.any(np.diff(elapsed_seconds) < 0):
        raise ValueError('time must be monotonically increasing')

    dt = np.diff(elapsed_seconds, prepend=elapsed_seconds[0])
    reshape = (elapsed_seconds.size,) + (1,) * (light.ndim - 1)
    dt = dt.reshape(reshape)

    cumulative_dose = np.cumsum(light * dt, axis=0)
    time_above_threshold = np.cumsum((light > light_threshold) * dt, axis=0)
    equivalent_sunlight_hours = cumulative_dose / float(reference_surface_light) / 3600.0

    return cumulative_dose, equivalent_sunlight_hours, time_above_threshold


def bleaching_probability(
    cumulative_light_dose: ArrayLike,
    *,
    bleaching_coefficient: float = 1.0,
    stored_lum_signal: float = 1e6,
) -> np.ndarray:
    """Placeholder bleaching probability from cumulative light dose.

    Formula:
    ``1 - (stored_lum_signal - cumulative_light_dose * bleaching_coefficient) / stored_lum_signal``

    Placeholder clipping behavior (as requested):
    - if ``stored_lum_signal - cumulative_light_dose * bleaching_coefficient < 0`` then probability = 0
    - if ``stored_lum_signal - cumulative_light_dose * bleaching_coefficient > 1`` then probability = 1
    """

    if stored_lum_signal <= 0.0:
        raise ValueError('stored_lum_signal must be positive')

    cumulative = _as_float_array(cumulative_light_dose, 'cumulative_light_dose')
    remaining_signal = float(stored_lum_signal) - cumulative * float(bleaching_coefficient)
    probability = 1.0 - (remaining_signal / float(stored_lum_signal))
    probability = np.where(remaining_signal < 0.0, 0.0, probability)
    probability = np.where(remaining_signal > 1.0, 1.0, probability)
    return np.asarray(probability, dtype=float)


def osl_signal_reduction(
    cumulative_light_dose: ArrayLike,
    *,
    bleaching_coefficient: float = 1.0,
    stored_lum_signal: float = 1e6,
) -> np.ndarray:
    """Placeholder remaining OSL signal after cumulative light exposure."""

    if stored_lum_signal <= 0.0:
        raise ValueError('stored_lum_signal must be positive')

    cumulative = _as_float_array(cumulative_light_dose, 'cumulative_light_dose')
    remaining_signal = float(stored_lum_signal) - cumulative * float(bleaching_coefficient)
    return np.clip(np.asarray(remaining_signal, dtype=float), 0.0, float(stored_lum_signal))


def compute_light_exposure(
    time: ArrayLike,
    water_depth: ArrayLike,
    attenuation_coefficient: ArrayLike,
    *,
    z: ArrayLike | None = None,
    z_c: ArrayLike | None = None,
    bed_level: ArrayLike | None = None,
    z_convention: ZConvention = 'depth_below_surface',
    surface_light: ArrayLike | None = None,
    burial_depth: ArrayLike | None = None,
    burial_depth_threshold: float = 0.0,
    is_buried: ArrayLike | None = None,
    in_domain: ArrayLike | None = None,
    is_beached: ArrayLike | None = None,
    light_threshold: float = 0.0,
    reference_surface_light: float = 1.0,
    bleaching_coefficient: float = 1.0,
    stored_lum_signal: float = 1e6,
) -> ExposureResult:
    """Compute per-sample and cumulative light exposure diagnostics."""

    time_values = np.asarray(time)
    particle_depth = particle_depth_below_surface(
        water_depth,
        z=z,
        z_c=z_c,
        bed_level=bed_level,
        z_convention=z_convention,
    )
    kd = _as_float_array(attenuation_coefficient, 'attenuation_coefficient')
    if surface_light is None:
        surface = surface_light_series(time_values, constant=reference_surface_light)
    else:
        surface = _as_float_array(surface_light, 'surface_light')

    target_ndim = max(surface.ndim, kd.ndim, particle_depth.ndim)
    surface = _reshape_time_vector_like(surface, target_ndim)
    kd = _reshape_time_vector_like(kd, target_ndim)
    particle_depth = _reshape_time_vector_like(particle_depth, target_ndim)

    intensity = light_intensity_at_particle(
        surface,
        kd,
        particle_depth,
        burial_depth=burial_depth,
        burial_depth_threshold=burial_depth_threshold,
        is_buried=is_buried,
        in_domain=in_domain,
        is_beached=is_beached,
    )
    cumulative, equivalent_hours, above_threshold = integrate_exposure(
        time_values,
        intensity,
        light_threshold=light_threshold,
        reference_surface_light=reference_surface_light,
    )
    bleaching = bleaching_probability(
        cumulative,
        bleaching_coefficient=bleaching_coefficient,
        stored_lum_signal=stored_lum_signal,
    )
    osl_reduction = osl_signal_reduction(
        cumulative,
        bleaching_coefficient=bleaching_coefficient,
        stored_lum_signal=stored_lum_signal,
    )
    return ExposureResult(
        light_intensity=intensity,
        cumulative_light_dose=cumulative,
        equivalent_sunlight_hours=equivalent_hours,
        time_above_threshold=above_threshold,
        bleaching_probability=bleaching,
        osl_signal_reduction=osl_reduction,
        attenuation_coefficient=np.broadcast_to(kd, intensity.shape),
        particle_depth_below_surface=np.broadcast_to(particle_depth, intensity.shape),
    )
