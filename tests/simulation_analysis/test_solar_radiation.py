import numpy as np
import pytest

from sedtrails.simulation_analysis.solar_radiation import compute_clear_sky_surface_light


def test_clear_sky_surface_light_is_zero_at_midnight_and_positive_at_noon():
    time = np.array(
        [
            '2020-06-21T00:00:00',
            '2020-06-21T12:00:00',
        ],
        dtype='datetime64[s]',
    )

    light = compute_clear_sky_surface_light(
        time,
        latitude_deg=53.45,
        longitude_deg=0.0,
        utc_offset_hours=0.0,
    )

    assert light[0] == 0.0
    assert light[1] > 0.0


def test_clear_sky_surface_light_has_reasonable_diurnal_cycle():
    time = np.array(
        [
            '2020-06-21T03:00:00',
            '2020-06-21T09:00:00',
            '2020-06-21T12:00:00',
            '2020-06-21T15:00:00',
            '2020-06-21T21:00:00',
        ],
        dtype='datetime64[s]',
    )

    light = compute_clear_sky_surface_light(time, latitude_deg=53.45)

    assert light[2] >= light[1]
    assert light[2] >= light[3]
    assert light[0] < light[2]
    assert light[-1] <= light[1]


def test_clear_sky_surface_light_supports_w_m2_output():
    time = np.array(['2020-06-21T12:00:00'], dtype='datetime64[s]')

    light_w = compute_clear_sky_surface_light(time, latitude_deg=53.45, output_unit='w_m2')
    light_par = compute_clear_sky_surface_light(time, latitude_deg=53.45, output_unit='umol_photons_m2_s')

    np.testing.assert_allclose(light_par, light_w * 2.1)


def test_clear_sky_surface_light_rejects_non_datetime_time():
    with pytest.raises(ValueError, match='datetime64'):
        compute_clear_sky_surface_light(np.array([0.0, 3600.0]), latitude_deg=53.45)


def test_clear_sky_surface_light_transmissivity_controls_amplitude():
    time = np.array(['2020-06-21T12:00:00'], dtype='datetime64[s]')

    light_low = compute_clear_sky_surface_light(
        time,
        latitude_deg=53.45,
        atmospheric_transmissivity=0.65,
    )
    light_high = compute_clear_sky_surface_light(
        time,
        latitude_deg=53.45,
        atmospheric_transmissivity=0.85,
    )

    assert light_high[0] > light_low[0]


def test_clear_sky_surface_light_consistent_for_equivalent_solar_time_offset():
    time = np.array(['2020-06-21T12:00:00'], dtype='datetime64[s]')

    baseline = compute_clear_sky_surface_light(
        time,
        latitude_deg=53.45,
        longitude_deg=0.0,
        utc_offset_hours=0.0,
    )
    shifted = compute_clear_sky_surface_light(
        time,
        latitude_deg=53.45,
        longitude_deg=15.0,
        utc_offset_hours=1.0,
    )

    np.testing.assert_allclose(shifted, baseline, rtol=1e-7, atol=1e-9)
