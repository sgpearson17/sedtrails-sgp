"""Population schema regression tests."""

import yaml
import pytest

from sedtrails.application_interfaces.validator import YAMLConfigValidator
from sedtrails.exceptions import YamlValidationError


def test_population_schema_rejects_multiple_tracer_methods(tmp_path):
    config = _base_config()
    config['particles']['populations'][0]['tracer_methods'] = {
        'vanwesten': {'flow_field_name': ['bed_load_velocity']},
        'soulsby': {'flow_field_name': ['grain_velocity']},
    }

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


def test_population_schema_requires_flow_field_name(tmp_path):
    config = _base_config()
    config['particles']['populations'][0]['tracer_methods'] = {'vanwesten': {'beta': 0.2}}

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


def test_population_schema_accepts_one_method_with_flow_fields(tmp_path):
    config = _base_config()

    validated = _validate_config(tmp_path, config)

    tracer_methods = validated['particles']['populations'][0]['tracer_methods']
    assert tracer_methods == {'vanwesten': {'flow_field_name': ['bed_load_velocity']}}


def _validate_config(tmp_path, config):
    config_file = tmp_path / 'sedtrails.yml'
    config_file.write_text(yaml.dump(config))
    return YAMLConfigValidator().validate_yaml(str(config_file))


def _base_config():
    return {
        'particles': {
            'populations': [
                {
                    'name': 'sand',
                    'particle_type': 'sand',
                    'characteristics': {
                        'density': 2650.0,
                        'grain_size': 0.00025,
                    },
                    'tracer_methods': {
                        'vanwesten': {
                            'flow_field_name': ['bed_load_velocity'],
                        },
                    },
                    'seeding': {
                        'burial_depth': {'constant': 0.0},
                        'release_start': '2016-09-21 19:30:00',
                        'quantity': 1,
                        'strategy': {
                            'point': {
                                'locations': ['0.0,0.0'],
                            },
                        },
                    },
                }
            ]
        }
    }
