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
def test_optional_wave_coupling_matches_explicit_effective_flow_without_mutating_inputs(wave_x, wave_y):
    config = _soulsby_config()
    config.include_nonlinear_wave_velocity = True
    data = SedtrailsDataStub()
    original = {name: values.copy() for name, values in data.depth_avg_flow_velocity.items()}
    data.nonlinear_wave_velocity = {
        'x': np.full((2, 2), wave_x),
        'y': np.full((2, 2), wave_y),
        'magnitude': np.full((2, 2), 999.0),
    }
    reference = SedtrailsDataStub()
    x, y = original['x'] + wave_x, original['y'] + wave_y
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
