"""Tests for population tracer runtime planning."""

import pytest

from sedtrails.exceptions.exceptions import ConfigurationError
from sedtrails.simulation_orchestrator.runtime_plan import (
    DEFAULT_TRANSPORT_PROBABILITY_METHOD,
    build_population_runtime_plans,
    required_physics_fields,
    unique_flow_field_names,
)


def _population_config(
    tracer_methods,
    *,
    characteristics=None,
    transport_probability=DEFAULT_TRANSPORT_PROBABILITY_METHOD,
):
    return {
        'name': 'sand',
        'particle_type': 'sand',
        'characteristics': characteristics or {'density': 2650.0, 'grain_size': 0.00025},
        'tracer_methods': tracer_methods,
        'transport_probability': transport_probability,
    }


def test_vanwesten_population_creates_population_scoped_converter():
    population_config = _population_config(
        {'vanwesten': {'flow_field_name': ['bed_load_velocity', 'suspended_velocity'], 'beta': 0.3}},
        transport_probability='stochastic_transport',
    )

    runtime_plan = build_population_runtime_plans([population_config], [object()], {'gravity': 9.8})[0]

    assert runtime_plan.population_index == 0
    assert runtime_plan.tracer.method_name == 'vanwesten'
    assert runtime_plan.tracer.flow_field_names == ('bed_load_velocity', 'suspended_velocity')
    assert runtime_plan.tracer.transport_probability_method == 'stochastic_transport'
    assert runtime_plan.tracer.converter.config.tracer_method == 'vanwesten'
    assert runtime_plan.tracer.converter.config.gravity == 9.8
    assert runtime_plan.tracer.converter.config.beta == 0.3
    assert runtime_plan.tracer.converter.config.grain_diameter == 0.00025


def test_soulsby_population_creates_population_scoped_converter():
    population_config = _population_config(
        {
            'soulsby': {
                'flow_field_name': ['grain_velocity'],
                'tracer_grain_size': 0.0001,
                'background_grain_size': 0.0002,
                'soulsby_mu_d': 0.6,
            }
        },
        characteristics={'density': 2600.0, 'grain_size': 0.0004},
    )

    runtime_plan = build_population_runtime_plans([population_config], [object()], {'water_density': 1025.0})[0]

    assert runtime_plan.tracer.method_name == 'soulsby'
    assert runtime_plan.tracer.flow_field_names == ('grain_velocity',)
    assert runtime_plan.tracer.converter.config.tracer_method == 'soulsby'
    assert runtime_plan.tracer.converter.config.water_density == 1025.0
    assert runtime_plan.tracer.converter.config.particle_density == 2600.0
    assert runtime_plan.tracer.converter.config.tracer_grain_size == 0.0001
    assert runtime_plan.tracer.converter.config.background_grain_size == 0.0002
    assert runtime_plan.tracer.converter.config.soulsby_mu_d == 0.6


def test_mixed_populations_preserve_each_population_method_and_flow_fields():
    vanwesten_config = _population_config({'vanwesten': {'flow_field_name': ['bed_load_velocity']}})
    soulsby_config = _population_config({'soulsby': {'flow_field_name': ['grain_velocity']}})

    runtime_plans = build_population_runtime_plans(
        [vanwesten_config, soulsby_config],
        [object(), object()],
        {},
    )

    assert [runtime_plan.tracer.method_name for runtime_plan in runtime_plans] == ['vanwesten', 'soulsby']
    assert [runtime_plan.tracer.flow_field_names for runtime_plan in runtime_plans] == [
        ('bed_load_velocity',),
        ('grain_velocity',),
    ]
    assert unique_flow_field_names(runtime_plans) == ['bed_load_velocity', 'grain_velocity']


def test_multiple_methods_in_one_population_raises_configuration_error():
    population_config = _population_config(
        {
            'vanwesten': {'flow_field_name': ['bed_load_velocity']},
            'soulsby': {'flow_field_name': ['grain_velocity']},
        }
    )

    with pytest.raises(ConfigurationError, match='multiple tracer methods'):
        build_population_runtime_plans([population_config], [object()], {})


def test_unknown_method_raises_configuration_error():
    population_config = _population_config({'unknown': {'flow_field_name': ['some_velocity']}})

    with pytest.raises(ConfigurationError, match='unsupported tracer method'):
        build_population_runtime_plans([population_config], [object()], {})


def test_missing_flow_field_name_raises_configuration_error():
    population_config = _population_config({'vanwesten': {'beta': 0.3}})

    with pytest.raises(ConfigurationError, match='flow_field_name'):
        build_population_runtime_plans([population_config], [object()], {})


def test_population_count_mismatch_raises_configuration_error():
    population_config = _population_config({'vanwesten': {'flow_field_name': ['bed_load_velocity']}})

    with pytest.raises(ConfigurationError, match='does not match seeded population count'):
        build_population_runtime_plans([population_config], [], {})


def test_required_physics_fields_are_method_specific_and_unique():
    assert required_physics_fields('vanwesten', ['bed_load_velocity', 'bed_load_velocity']) == (
        'bed_load_velocity',
        'mixing_layer_thickness',
        'bed_load_probability',
    )
    assert required_physics_fields('soulsby', ['grain_velocity']) == (
        'grain_velocity',
        'mixing_layer_thickness',
        'soulsby_a',
        'soulsby_b',
    )
