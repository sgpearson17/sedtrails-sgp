"""Coordinate configuration schema regression tests."""

import pytest
import yaml

from sedtrails.application_interfaces.validator import YAMLConfigValidator
from sedtrails.exceptions import YamlValidationError


EXPECTED_COORDINATE_DEFAULTS = {
    'coordinate_system': 'auto',
    'source_crs': 'EPSG:4326',
    'runtime_geometry': 'auto',
    'metric_crs': 'auto_utm',
    'surface_model': 'sphere',
    'earth_radius_m': 6_371_008.8,
    'longitude_wrap': 'auto',
    'velocity_basis': 'auto',
}


def test_coordinate_schema_applies_backward_compatible_defaults(tmp_path):
    """Apply coordinate defaults without changing existing projected configurations."""
    validated = _validate_config(tmp_path, _base_config())

    input_model = validated['general']['input_model']
    for key, value in EXPECTED_COORDINATE_DEFAULTS.items():
        assert input_model[key] == value


def test_coordinate_schema_accepts_spherical_configuration(tmp_path):
    """Accept the complete explicit configuration for global spherical geometry."""
    config = _base_config()
    config['general']['input_model'].update(
        {
            'coordinate_system': 'spherical',
            'source_crs': 'EPSG:4326',
            'runtime_geometry': 'geodetic',
            'metric_crs': None,
            'surface_model': 'sphere',
            'earth_radius_m': 6_371_000.0,
            'longitude_wrap': '0_360',
            'velocity_basis': 'east_north',
        }
    )

    validated = _validate_config(tmp_path, config)

    input_model = validated['general']['input_model']
    for key in EXPECTED_COORDINATE_DEFAULTS:
        assert input_model[key] == config['general']['input_model'][key]


def test_coordinate_schema_accepts_explicit_local_projection(tmp_path):
    """Accept an explicit metre projection for regional geographic geometry."""
    config = _base_config()
    config['general']['input_model'].update(
        {
            'coordinate_system': 'geographic',
            'runtime_geometry': 'planar',
            'metric_crs': 'EPSG:32631',
            'velocity_basis': 'source_xy',
        }
    )

    validated = _validate_config(tmp_path, config)

    assert validated['general']['input_model']['runtime_geometry'] == 'planar'
    assert validated['general']['input_model']['metric_crs'] == 'EPSG:32631'
    assert validated['general']['input_model']['velocity_basis'] == 'source_xy'


@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('coordinate_system', 'degrees'),
        ('source_crs', ''),
        ('runtime_geometry', 'global'),
        ('metric_crs', ''),
        ('surface_model', 'flat'),
        ('earth_radius_m', 0),
        ('earth_radius_m', -1),
        ('longitude_wrap', 'signed'),
        ('velocity_basis', 'map'),
    ],
)
def test_coordinate_schema_rejects_invalid_values(tmp_path, key, value):
    """Reject invalid coordinate enum values, empty CRS labels, and radii."""
    config = _base_config()
    config['general']['input_model'][key] = value

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


def test_coordinate_schema_rejects_spherical_runtime_for_projected_source(tmp_path):
    """Reject a spherical backend when source coordinates are explicitly projected."""
    config = _base_config()
    config['general']['input_model'].update(
        {
            'coordinate_system': 'projected',
            'runtime_geometry': 'geodetic',
        }
    )

    with pytest.raises(YamlValidationError, match='YAML config validation error'):
        _validate_config(tmp_path, config)


def test_generated_template_includes_coordinate_defaults(tmp_path):
    """Expose the full coordinate contract in generated configuration templates."""
    output_file = tmp_path / 'sedtrails-template.yml'

    YAMLConfigValidator().create_config_template(str(output_file))

    template = yaml.safe_load(output_file.read_text(encoding='utf-8'))
    input_model = template['general']['input_model']
    for key, value in EXPECTED_COORDINATE_DEFAULTS.items():
        assert input_model[key] == value


def _validate_config(tmp_path, config):
    """Write and validate one temporary SedTRAILS configuration."""
    config_file = tmp_path / 'sedtrails.yml'
    config_file.write_text(yaml.safe_dump(config), encoding='utf-8')
    return YAMLConfigValidator().validate_yaml(str(config_file))


def _base_config():
    """Return a minimal configuration with the historical input-model fields."""
    return {
        'general': {
            'input_model': {
                'format': 'fm_netcdf',
                'reference_date': '1970-01-01',
            }
        },
        'inputs': {'data': 'dummy.nc'},
    }
