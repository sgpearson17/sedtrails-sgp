import warnings

import numpy as np
import pytest

from sedtrails.transport_converter import physics_lib
from sedtrails.transport_converter.physics_converter import PhysicsConfig
from sedtrails.transport_converter.plugins.physics.soulsby import PhysicsPlugin, _safe_velocity_direction


class SedtrailsDataStub:
    def __init__(self):
        self.depth_avg_flow_velocity = {
            'x': np.array([[0.0, 1.0], [0.0, 0.0]]),
            'y': np.array([[0.0, 0.0], [0.0, 2.0]]),
            'magnitude': np.array([[0.0, 1.0], [0.0, 2.0]]),
        }
        self.mean_bed_shear_stress = np.ones((2, 2))
        self.max_bed_shear_stress = np.ones((2, 2))
        self.nonlinear_wave_velocity = None

    def add_physics_field(self, name, data):
        setattr(self, name, data)


def _soulsby_config():
    return PhysicsConfig.from_dict(
        {'tracer_method': 'soulsby'},
        {
            'soulsby': {
                'tracer_grain_size': 0.0002,
                'background_grain_size': 0.0002,
                'soulsby_b_e': 1.7e-7,
                'soulsby_theta_s': 0.1,
                'soulsby_gamma_e': 0.1,
                'soulsby_mu_d': 0.5,
            }
        },
    )


def test_safe_velocity_direction_uses_zero_direction_for_zero_speed():
    direction_x, direction_y = _safe_velocity_direction(
        np.array([[0.0, 3.0]]),
        np.array([[0.0, 4.0]]),
        np.array([[0.0, 5.0]]),
    )

    np.testing.assert_array_equal(direction_x, np.array([[0.0, 0.6]]))
    np.testing.assert_array_equal(direction_y, np.array([[0.0, 0.8]]))


def test_soulsby_zero_flow_magnitude_does_not_warn_or_emit_nan():
    config = _soulsby_config()
    plugin = PhysicsPlugin(config, tracer_config={})
    sedtrails_data = SedtrailsDataStub()
    grain_properties = physics_lib.compute_grain_properties(
        config.tracer_grain_size,
        config.gravity,
        config.particle_density,
        config.water_density,
        config.kinematic_viscosity,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        plugin.add_physics(sedtrails_data, grain_properties, transport_probability_method='no_probability')

    grain_velocity = sedtrails_data.grain_velocity
    assert np.isfinite(grain_velocity['magnitude']).all()
    assert np.isfinite(grain_velocity['x']).all()
    assert np.isfinite(grain_velocity['y']).all()
    assert grain_velocity['magnitude'][0, 0] == 0.0
    assert grain_velocity['x'][0, 0] == 0.0
    assert grain_velocity['y'][0, 0] == 0.0


def test_vectorized_soulsby_fields_match_scalar_equations():
    config = _soulsby_config()
    data = SedtrailsDataStub()
    data.depth_avg_flow_velocity = {
        'x': np.array([[0.0, 0.01], [0.03, 0.1]]),
        'y': np.array([[0.0, 0.0], [0.04, -0.05]]),
    }
    data.depth_avg_flow_velocity['magnitude'] = np.hypot(
        data.depth_avg_flow_velocity['x'], data.depth_avg_flow_velocity['y']
    )
    data.mean_bed_shear_stress = np.array([[0.0, 0.005], [0.03, 0.2]])
    data.max_bed_shear_stress = np.array([[0.0, 0.01], [0.1, 1.0]])
    properties = physics_lib.compute_grain_properties(
        config.tracer_grain_size,
        config.gravity,
        config.particle_density,
        config.water_density,
        config.kinematic_viscosity,
    )
    PhysicsPlugin(config, tracer_config={}).add_physics(data, properties, 'stochastic_transport')

    g = config.gravity
    rho_s, rho_w = config.particle_density, config.water_density
    grain_size = config.tracer_grain_size
    background_grain_size = config.background_grain_size
    grain_size_ratio = grain_size / background_grain_size
    dimensionless_grain_size = properties['dimensionless_grain_size']
    theta_cr = 0.3 / (1 + 1.2 * dimensionless_grain_size) + 0.055 * (
        1 - np.exp(-0.020 * dimensionless_grain_size)
    )
    theta_cr_exp = (
        theta_cr
        * np.sqrt(8 / (3 * grain_size_ratio**2 + 6 * grain_size_ratio - 1))
        * (3.2260 * grain_size_ratio)
        / (
            4 * grain_size_ratio
            - 2 * (grain_size_ratio + 1 - np.sqrt(grain_size_ratio**2 + 2 * grain_size_ratio - 1 / 3))
        )
    )
    background_dimensionless_size = (
        g * (rho_s / rho_w - 1) / config.kinematic_viscosity**2
    ) ** (1 / 3) * background_grain_size
    background_theta_cr = 0.3 / (1 + 1.2 * background_dimensionless_size) + 0.055 * (
        1 - np.exp(-0.020 * background_dimensionless_size)
    )
    denominator = g * (rho_s - rho_w) * grain_size
    theta_max = np.abs(data.max_bed_shear_stress) / denominator
    background_theta_max = np.abs(data.max_bed_shear_stress) / (
        g * (rho_s - rho_w) * background_grain_size
    )
    bed_load_velocity = physics_lib.compute_bed_load_velocity(
        theta_max,
        properties['critical_shields'],
        physics_lib.compute_shear_velocity(data.mean_bed_shear_stress, rho_w),
    )
    max_shear_velocity = physics_lib.compute_shear_velocity(data.max_bed_shear_stress, rho_w)
    settling_velocity = properties['settling_velocity']
    rouse_number = np.full_like(max_shear_velocity, np.inf)
    np.divide(
        settling_velocity,
        0.4 * max_shear_velocity,
        out=rouse_number,
        where=max_shear_velocity > 0,
    )

    expected_b = np.zeros_like(theta_max)
    expected_p = np.zeros_like(theta_max)
    expected_rb = np.zeros_like(theta_max)
    expected_rs = np.zeros_like(theta_max)
    expected_r = np.zeros_like(theta_max)
    expected_grain_magnitude = np.zeros_like(theta_max)
    expected_grain_x = np.zeros_like(theta_max)
    expected_grain_y = np.zeros_like(theta_max)
    for i in range(theta_max.shape[0]):
        for j in range(theta_max.shape[1]):
            if background_theta_max[i, j] > background_theta_cr:
                expected_b[i, j] = 1.7e-7 * (
                    1 - np.exp(-(background_theta_max[i, j] - background_theta_cr) / 0.1)
                )
            if theta_max[i, j] > theta_cr_exp:
                expected_p[i, j] = (
                    1 + ((np.pi / (6 * config.soulsby_mu_d)) / (theta_max[i, j] - theta_cr_exp)) ** 4
                ) ** (-1 / 4)
                flow_speed = data.depth_avg_flow_velocity['magnitude'][i, j]
                if np.isfinite(flow_speed) and flow_speed > 0:
                    expected_rb[i, j] = bed_load_velocity[i, j] / flow_speed
                    if expected_rb[i, j] > 1:
                        expected_rb[i, j] = 1
                    elif not np.isfinite(expected_rb[i, j]):
                        expected_rb[i, j] = 0
            if expected_rb[i, j] != 0:
                with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                    rb = expected_rb[i, j]
                    rouse = rouse_number[i, j]
                    expected_rs[i, j] = (
                        rb * (1 - rouse) / (8 / 7 - rouse)
                        * ((8 / 7 * rb) ** (8 - 7 * rouse) - 1)
                        / ((8 / 7 * rb) ** (7 - 7 * rouse) - 1)
                    )
                if expected_rs[i, j] > 1:
                    expected_rs[i, j] = 1
                elif np.isnan(expected_rs[i, j]):
                    expected_rs[i, j] = 0
            expected_r[i, j] = expected_rs[i, j] if rouse_number[i, j] < 2.5 else expected_rb[i, j]
            expected_grain_magnitude[i, j] = expected_p[i, j] * expected_r[i, j] * (
                data.depth_avg_flow_velocity['magnitude'][i, j]
            )
            if data.depth_avg_flow_velocity['magnitude'][i, j] > 0:
                expected_grain_x[i, j] = (
                    data.depth_avg_flow_velocity['x'][i, j]
                    / data.depth_avg_flow_velocity['magnitude'][i, j]
                    * expected_grain_magnitude[i, j]
                )
                expected_grain_y[i, j] = (
                    data.depth_avg_flow_velocity['y'][i, j]
                    / data.depth_avg_flow_velocity['magnitude'][i, j]
                    * expected_grain_magnitude[i, j]
                )

    expected_a = config.soulsby_gamma_e * expected_b / (1 - config.soulsby_gamma_e)
    np.testing.assert_allclose(data.soulsby_b, expected_b, rtol=1e-13, atol=0)
    np.testing.assert_allclose(data.soulsby_a, expected_a, rtol=1e-13, atol=0)
    np.testing.assert_allclose(data.grain_velocity['magnitude'], expected_grain_magnitude, rtol=1e-13, atol=0)
    np.testing.assert_allclose(data.grain_velocity['x'], expected_grain_x, rtol=1e-13, atol=0)
    np.testing.assert_allclose(data.grain_velocity['y'], expected_grain_y, rtol=1e-13, atol=0)


def _apply_soulsby(config, data):
    properties = physics_lib.compute_grain_properties(
        config.tracer_grain_size,
        config.gravity,
        config.particle_density,
        config.water_density,
        config.kinematic_viscosity,
    )
    PhysicsPlugin(config, tracer_config={}).add_physics(data, properties, 'no_probability')


@pytest.mark.parametrize('wave_x,wave_y', [(0.0, 1.0), (3.0, 4.0), (-1.0, 0.0)])
@pytest.mark.parametrize('facua', [0.0, 0.5, 1.0, 2.0])
def test_optional_wave_coupling_matches_explicit_effective_flow_without_mutating_inputs(wave_x, wave_y, facua):
    config = _soulsby_config()
    config.include_nonlinear_wave_velocity = True
    config.facua = facua
    data = SedtrailsDataStub()
    original = {name: values.copy() for name, values in data.depth_avg_flow_velocity.items()}
    data.nonlinear_wave_velocity = {
        'x': np.full((2, 2), wave_x),
        'y': np.full((2, 2), wave_y),
        'magnitude': np.full((2, 2), 999.0),
    }
    reference = SedtrailsDataStub()
    x, y = original['x'] + facua * wave_x, original['y'] + facua * wave_y
    reference.depth_avg_flow_velocity = {'x': x, 'y': y, 'magnitude': np.hypot(x, y)}
    _apply_soulsby(_soulsby_config(), reference)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        _apply_soulsby(config, data)
    for name in ('x', 'y', 'magnitude'):
        np.testing.assert_allclose(data.grain_velocity[name], reference.grain_velocity[name])
        np.testing.assert_array_equal(data.depth_avg_flow_velocity[name], original[name])
    np.testing.assert_array_equal(data.nonlinear_wave_velocity['x'], np.full((2, 2), wave_x))
    np.testing.assert_array_equal(data.nonlinear_wave_velocity['y'], np.full((2, 2), wave_y))


def test_wave_coupling_is_disabled_by_default_even_with_wave_input():
    config = _soulsby_config()
    assert config.include_nonlinear_wave_velocity is False
    assert config.facua == 1.0
    config.facua = 2.0
    data, reference = SedtrailsDataStub(), SedtrailsDataStub()
    data.nonlinear_wave_velocity = {'x': np.full((2, 2), 10.0), 'y': np.full((2, 2), 20.0)}
    _apply_soulsby(config, data)
    _apply_soulsby(config, reference)
    for name in ('x', 'y', 'magnitude'):
        np.testing.assert_array_equal(data.grain_velocity[name], reference.grain_velocity[name])


@pytest.mark.parametrize(
    'wave',
    [
        None,
        {'x': np.zeros((2, 2))},
        {'x': np.zeros((1, 2)), 'y': np.zeros((2, 2))},
        {'x': np.full((2, 2), np.nan), 'y': np.zeros((2, 2))},
        {'x': np.zeros((2, 2)), 'y': np.full((2, 2), np.inf)},
    ],
)
def test_enabled_wave_coupling_rejects_missing_or_invalid_vectors(wave):
    config = _soulsby_config()
    config.include_nonlinear_wave_velocity = True
    data = SedtrailsDataStub()
    data.nonlinear_wave_velocity = wave
    with pytest.raises(ValueError, match='wave|effective flow'):
        _apply_soulsby(config, data)


@pytest.mark.parametrize('value', ['false', 1, None])
def test_wave_coupling_requires_a_boolean(value):
    config = _soulsby_config()
    config.include_nonlinear_wave_velocity = value
    with pytest.raises(ValueError, match='must be true or false'):
        PhysicsPlugin(config, tracer_config={})


@pytest.mark.parametrize('value', [-0.1, np.nan, np.inf, -np.inf, '1', True, None])
def test_wave_calibration_requires_a_finite_non_negative_number(value):
    config = _soulsby_config()
    config.facua = value
    with pytest.raises(ValueError, match='facua'):
        PhysicsPlugin(config, tracer_config={})
