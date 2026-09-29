from types import SimpleNamespace

import numpy as np

from sedtrails.particle_tracer.windage import (
    combine_velocity_bounds,
    get_wind_bounds,
    wind_components_from_speed_direction,
)


class _RetrieverStub:
    def __init__(self):
        self.sedtrails_data = SimpleNamespace(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]))

    def get_flow_field_bounds(self, time_seconds, flow_field_name):
        return {
            'x': self.sedtrails_data.x,
            'y': self.sedtrails_data.y,
            'lower': {'u': np.array([1.0, 2.0]), 'v': np.array([3.0, 4.0]), 'magnitude': np.array([0.0, 0.0])},
            'upper': {'u': np.array([5.0, 6.0]), 'v': np.array([7.0, 8.0]), 'magnitude': np.array([0.0, 0.0])},
            'weight': 0.25,
            'lower_index': 0,
            'upper_index': 1,
        }

    def get_scalar_field_bounds(self, time_seconds, scalar_field_name):
        fields = {
            'wind_u': {'lower': np.array([1.0, 2.0]), 'upper': np.array([2.0, 3.0])},
            'wind_v': {'lower': np.array([4.0, 5.0]), 'upper': np.array([5.0, 6.0])},
        }
        field = fields[scalar_field_name]
        return {
            'x': self.sedtrails_data.x,
            'y': self.sedtrails_data.y,
            'lower': field['lower'],
            'upper': field['upper'],
            'weight': 0.5,
            'lower_index': 0,
            'upper_index': 1,
        }


def test_meteorological_wind_direction_converts_correctly():
    u, v = wind_components_from_speed_direction(4.4, 292.5, convention='from_meteorological')

    np.testing.assert_allclose(u, 4.065069943154143)
    np.testing.assert_allclose(v, -1.683432513190568)


def test_constant_timeseries_and_field_wind_providers_share_runtime_shape():
    retriever = _RetrieverStub()

    constant = get_wind_bounds(
        0.0,
        {'type': 'constant', 'u': 1.0, 'v': 2.0},
        retriever,
        direction_convention='from_meteorological',
    )
    timeseries = get_wind_bounds(
        1800.0,
        {'type': 'timeseries_point', 'times': [0.0, 3600.0], 'u_values': [1.0, 3.0], 'v_values': [2.0, 4.0]},
        retriever,
        direction_convention='from_meteorological',
    )
    field = get_wind_bounds(
        1800.0,
        {'type': 'field', 'u_field_name': 'wind_u', 'v_field_name': 'wind_v'},
        retriever,
        direction_convention='from_meteorological',
    )

    for bounds in (constant, timeseries, field):
        assert set(bounds) >= {'x', 'y', 'lower', 'upper', 'weight'}
        assert set(bounds['lower']) == {'u', 'v', 'magnitude'}
        assert set(bounds['upper']) == {'u', 'v', 'magnitude'}


def test_windage_modifies_velocity_by_scaled_wind_components():
    base_flow = {
        'x': np.array([0.0]),
        'y': np.array([0.0]),
        'lower': {'u': np.array([0.5]), 'v': np.array([1.0]), 'magnitude': np.array([0.0])},
        'upper': {'u': np.array([0.5]), 'v': np.array([1.0]), 'magnitude': np.array([0.0])},
        'weight': 0.0,
    }
    wind = {
        'x': np.array([0.0]),
        'y': np.array([0.0]),
        'lower': {'u': np.array([4.0]), 'v': np.array([-2.0]), 'magnitude': np.array([0.0])},
        'upper': {'u': np.array([4.0]), 'v': np.array([-2.0]), 'magnitude': np.array([0.0])},
        'weight': 0.0,
    }

    effective = combine_velocity_bounds(base_flow, wind, 0.02)

    np.testing.assert_allclose(effective['lower']['u'], np.array([0.58]))
    np.testing.assert_allclose(effective['lower']['v'], np.array([0.96]))
