import numpy as np
import pytest

from sedtrails.simulation_analysis.light_exposure import (
    attenuation_from_storlazzi,
    combine_attenuation_components,
    compute_light_exposure,
    integrate_exposure,
    light_intensity_at_particle,
    particle_depth_below_surface,
    rouse_centroid_depth,
    surface_light_series,
)


def test_storlazzi_attenuation_components_use_digitized_linear_relations():
    result = attenuation_from_storlazzi(
        sed_conc_mud=np.array([0.0, 10.0]),
        sed_conc_sand=np.array([0.0, 10.0]),
    )

    np.testing.assert_allclose(result['mud'], [1.2651, 1.4221])
    np.testing.assert_allclose(result['sand'], [1.5352, 1.5712])


def test_combine_attenuation_can_run_sand_only_or_mud_only():
    components = attenuation_from_storlazzi(
        sed_conc_mud=np.array([2.0]),
        sed_conc_sand=np.array([3.0]),
    )

    np.testing.assert_allclose(
        combine_attenuation_components(components['mud'], components['sand'], strategy='mud'),
        components['mud'],
    )
    np.testing.assert_allclose(
        combine_attenuation_components(components['mud'], components['sand'], strategy='sand'),
        components['sand'],
    )


def test_rouse_centroid_depth_is_clipped_to_water_column():
    zc = rouse_centroid_depth(
        water_depth=np.array([2.0, 2.0]),
        settling_velocity=np.array([0.01, 0.01]),
        shear_velocity=np.array([0.0, 0.1]),
    )

    assert np.isnan(zc[0])
    assert 0.0 <= zc[1] <= 2.0


def test_particle_depth_supports_height_above_bed_convention():
    depth = particle_depth_below_surface(
        water_depth=np.array([2.0, 3.0]),
        z=np.array([0.5, 1.0]),
        z_convention='height_above_bed',
    )

    np.testing.assert_allclose(depth, [1.5, 2.0])


def test_surface_light_series_applies_daylight_mask():
    light = surface_light_series(
        np.array([0.0, 1.0, 2.0]),
        constant=10.0,
        daylight_mask=np.array([True, False, True]),
    )

    np.testing.assert_allclose(light, [10.0, 0.0, 10.0])


def test_light_intensity_zero_when_buried_and_full_when_beached():
    light = light_intensity_at_particle(
        surface_light=np.array([10.0, 10.0, 10.0]),
        attenuation_coefficient=np.array([2.0, 2.0, 2.0]),
        particle_depth=np.array([1.0, 1.0, 1.0]),
        burial_depth=np.array([0.0, 0.1, 0.0]),
        is_beached=np.array([False, False, True]),
    )

    assert light[0] == pytest.approx(10.0 * np.exp(-2.0))
    assert light[1] == 0.0
    assert light[2] == 10.0


def test_light_intensity_ignores_burial_depth_below_threshold():
    light = light_intensity_at_particle(
        surface_light=np.array([10.0, 10.0]),
        attenuation_coefficient=np.array([0.0, 0.0]),
        particle_depth=np.array([0.0, 0.0]),
        burial_depth=np.array([1e-13, 1e-3]),
        burial_depth_threshold=1e-6,
    )

    np.testing.assert_allclose(light, [10.0, 0.0])


def test_light_intensity_broadcasts_time_series_over_particles():
    light = light_intensity_at_particle(
        surface_light=np.array([10.0, 20.0]),
        attenuation_coefficient=np.ones((2, 3)),
        particle_depth=np.zeros((2, 3)),
    )

    np.testing.assert_allclose(light, [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])


def test_integrate_exposure_uses_saved_timestep_intervals():
    cumulative, equivalent_hours, above = integrate_exposure(
        time=np.array([0.0, 10.0, 20.0]),
        light_intensity=np.array([[1.0], [2.0], [3.0]]),
        light_threshold=2.5,
        reference_surface_light=1.0,
    )

    np.testing.assert_allclose(cumulative[:, 0], [0.0, 20.0, 50.0])
    np.testing.assert_allclose(equivalent_hours[:, 0], [0.0, 20.0 / 3600.0, 50.0 / 3600.0])
    np.testing.assert_allclose(above[:, 0], [0.0, 0.0, 10.0])


def test_integrate_exposure_converts_datetime_to_seconds():
    cumulative, equivalent_hours, _ = integrate_exposure(
        time=np.array(['2020-01-01T00:00:00', '2020-01-01T01:00:00'], dtype='datetime64[s]'),
        light_intensity=np.array([[1.0], [1.0]]),
        reference_surface_light=1.0,
    )

    np.testing.assert_allclose(cumulative[:, 0], [0.0, 3600.0])
    np.testing.assert_allclose(equivalent_hours[:, 0], [0.0, 1.0])


def test_compute_light_exposure_returns_placeholders_for_future_osl_models():
    result = compute_light_exposure(
        time=np.array([0.0, 10.0]),
        water_depth=np.array([[2.0], [2.0]]),
        attenuation_coefficient=np.array([[1.0], [1.0]]),
        z=np.array([[0.0], [1.0]]),
        surface_light=np.array([10.0, 10.0]),
        burial_depth=np.array([[0.0], [0.0]]),
    )

    assert result.light_intensity.shape == (2, 1)
    assert result.cumulative_light_dose[-1, 0] == pytest.approx(10.0 * np.exp(-1.0) * 10.0)
    assert np.isnan(result.bleaching_probability).all()
    assert np.isnan(result.osl_signal_reduction).all()
