"""Solar radiation helpers for offline light-exposure workflows.

This module currently provides a lightweight clear-sky radiation estimate for
offline diagnostics. The implementation is intentionally simple and fast, and
is designed to produce realistic day-night and seasonal variability without a
full atmospheric radiation model.

Method summary
--------------
1. Solar geometry from day-of-year and time-of-day:
     - fractional year angle (``gamma``)
     - solar declination and equation of time (NOAA-style trigonometric series)
     - hour angle from local clock time corrected by equation of time, longitude,
         and UTC offset
2. Top-of-atmosphere horizontal irradiance:
     - ``I_toa = S0 * E0 * max(cos(theta_z), 0)``
     - where ``E0`` is the Earth-Sun distance correction factor.
3. Clear-sky attenuation with one bulk transmissivity parameter and
     Kasten-Young relative air mass.
4. Optional conversion from W m-2 to PAR units using a fixed factor.

Key assumptions and limitations
-------------------------------
- Clear-sky only: no explicit clouds, aerosols, humidity, or ozone.
- Single bulk transmissivity: atmospheric effects are collapsed to one scalar
    parameter (``atmospheric_transmissivity``), constant in time.
- No terrain/horizon masking or local shading.
- Timestamps are interpreted as local civil time using
    ``utc_offset_hours``/``longitude_deg`` for solar-time correction.
- The trigonometric seasonal series uses a 365-day year approximation.
- PAR conversion uses a fixed scalar (default 2.1 umol J-1), which is a common
    broadband approximation and not spectrum-resolving.

This is appropriate for first-order offline sensitivity analyses. For
site-specific absolute irradiance studies, users should replace or calibrate
this forcing with measured radiation or a dedicated meteorological product.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_datetime64_ns(time: ArrayLike) -> np.ndarray:
    values = np.asarray(time)
    if values.ndim != 1:
        raise ValueError('time must be one-dimensional')
    if values.size == 0:
        raise ValueError('time must not be empty')
    if not np.issubdtype(values.dtype, np.datetime64):
        raise ValueError('time must be datetime64 for realistic solar radiation calculations')
    return values.astype('datetime64[ns]')


def compute_clear_sky_surface_light(
    time: ArrayLike,
    *,
    latitude_deg: float,
    longitude_deg: float = 0.0,
    utc_offset_hours: float = 0.0,
    atmospheric_transmissivity: float = 0.75,
    solar_constant_w_m2: float = 1361.0,
    par_conversion_umol_per_joule: float = 2.1,
    output_unit: str = 'umol_photons_m2_s',
) -> np.ndarray:
    """Estimate clear-sky incoming shortwave/PAR from time and location.

    Formulation notes
    -----------------
    This function uses common closed-form solar-geometry approximations similar
    to NOAA solar-calculator equations for declination and equation of time,
    with Kasten-Young air mass for clear-sky attenuation. The returned signal
    is therefore a physically plausible clear-sky estimate, not an
    observation-constrained meteorological reconstruction.

    Parameters
    ----------
    time : array-like of datetime64
        Timestamps for which to compute incoming radiation.
    latitude_deg : float
        Latitude in decimal degrees (north positive).
    longitude_deg : float
        Longitude in decimal degrees (east positive), used for solar-time correction.
    utc_offset_hours : float
        Local time zone offset from UTC for timestamps.
    atmospheric_transmissivity : float
        Bulk clear-sky transmissivity at unit air mass; typical values are 0.7-0.8.
    solar_constant_w_m2 : float
        Solar constant at top of atmosphere.
    par_conversion_umol_per_joule : float
        Approximate conversion from W m-2 to umol photons m-2 s-1.
    output_unit : {'w_m2', 'umol_photons_m2_s'}
        Output units for the returned timeseries.

        Returns
        -------
        np.ndarray
                One-dimensional incoming surface radiation timeseries in the requested
                unit.

        Notes
        -----
        Recommended starting ranges:
        - ``atmospheric_transmissivity``: about 0.70 to 0.80 for clear-sky marine
            conditions.
        - ``par_conversion_umol_per_joule``: around 2.0 to 2.2 depending on
            assumed spectral composition.
    """

    if not (-90.0 <= float(latitude_deg) <= 90.0):
        raise ValueError('latitude_deg must be within [-90, 90]')
    if not (0.0 < float(atmospheric_transmissivity) <= 1.0):
        raise ValueError('atmospheric_transmissivity must be in (0, 1]')

    t = _as_datetime64_ns(time)
    start_of_year = t.astype('datetime64[Y]').astype('datetime64[D]')
    day_number = (t.astype('datetime64[D]') - start_of_year).astype(int) + 1
    seconds_of_day = (t - t.astype('datetime64[D]')).astype('timedelta64[s]').astype(float)

    fractional_day_of_year = day_number + seconds_of_day / 86400.0
    gamma = 2.0 * np.pi / 365.0 * (fractional_day_of_year - 1.0)

    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )

    equation_of_time_min = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )

    local_clock_min = seconds_of_day / 60.0
    solar_time_min = local_clock_min + equation_of_time_min + 4.0 * float(longitude_deg) - 60.0 * float(utc_offset_hours)
    hour_angle = np.deg2rad(solar_time_min / 4.0 - 180.0)

    latitude = np.deg2rad(float(latitude_deg))
    cos_zenith = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    cos_zenith_positive = np.maximum(cos_zenith, 0.0)

    day_angle = 2.0 * np.pi * (day_number - 1.0) / 365.0
    eccentricity = (
        1.00011
        + 0.034221 * np.cos(day_angle)
        + 0.00128 * np.sin(day_angle)
        + 0.000719 * np.cos(2.0 * day_angle)
        + 0.000077 * np.sin(2.0 * day_angle)
    )

    toa_horizontal = float(solar_constant_w_m2) * eccentricity * cos_zenith_positive

    zenith_deg = np.rad2deg(np.arccos(cos_zenith))
    air_mass = np.full_like(toa_horizontal, np.nan, dtype=float)
    valid = cos_zenith_positive > 0.0
    air_mass[valid] = 1.0 / (
        cos_zenith_positive[valid] + 0.50572 * (96.07995 - zenith_deg[valid]) ** -1.6364
    )

    transmittance = np.zeros_like(toa_horizontal)
    transmittance[valid] = float(atmospheric_transmissivity) ** air_mass[valid]
    surface_w_m2 = toa_horizontal * transmittance
    surface_w_m2 = np.nan_to_num(surface_w_m2, nan=0.0, posinf=0.0, neginf=0.0)

    if output_unit == 'w_m2':
        return surface_w_m2
    if output_unit == 'umol_photons_m2_s':
        return surface_w_m2 * float(par_conversion_umol_per_joule)
    raise ValueError("output_unit must be 'w_m2' or 'umol_photons_m2_s'")
