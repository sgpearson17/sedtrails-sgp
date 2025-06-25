"""
Unit tests for the YAMLConfigValidator class in the sedtrails.configuration_interface.validator module.
"""

import tempfile
import pytest
import yaml
import json
from sedtrails.configuration_interface.validator import YAMLConfigValidator
from sedtrails.exceptions import YamlValidationError


@pytest.fixture
def validator():
    # You can use any valid schema file path, but for testing _apply_defaults, schema_content is enough
    dummy_schema = {}
    dummy_schema_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yml')
    yaml.dump(dummy_schema, dummy_schema_file)
    dummy_schema_file.close()

    DummyValidator = YAMLConfigValidator(schema_filepath=dummy_schema_file.name)

    return DummyValidator


class TestYAMLConfigValidator:
    """
    Test suite for validating YAML configuration files using YAMLConfigValidator.
    """

    def test_apply_defaults_object(self, validator):
        schema = {
            'type': 'object',
            'properties': {
                'a': {'type': 'string', 'default': 'foo'},
                'b': {'type': 'number', 'default': 42},
            },
        }
        config = {}
        result = validator._apply_defaults(schema, config)
        assert result == {'a': 'foo', 'b': 42}

    def test_apply_defaults_partial_object(self, validator):
        schema = {
            'type': 'object',
            'properties': {
                'a': {'type': 'string', 'default': 'foo'},
                'b': {'type': 'number', 'default': 42},
            },
        }
        config = {'a': 'bar'}
        result = validator._apply_defaults(schema, config)
        assert result == {'a': 'bar', 'b': 42}

    def test_apply_defaults_nested_object(self, validator):
        schema = {
            'type': 'object',
            'properties': {'outer': {'type': 'object', 'properties': {'inner': {'type': 'string', 'default': 'baz'}}}},
        }
        config = {'outer': {}}
        result = validator._apply_defaults(schema, config)
        assert result == {'outer': {'inner': 'baz'}}

    def test_apply_defaults_array(self, validator):
        schema = {'type': 'array', 'items': {'type': 'object', 'properties': {'x': {'type': 'integer', 'default': 1}}}}
        config = [{'x': 2}, {}]
        result = validator._apply_defaults(schema, config)
        assert result == [{'x': 2}, {'x': 1}]

    # -----------------------------
    # Tests for validate_yaml
    # -----------------------------
    def test_validate_yaml_success(self, tmp_path):
        """
        Test YAML file validation when a YAML file is created successfully
        """
        # Define a schema that requires "path" and provides defaults for "name" and "folder"
        schema = {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'type': 'object',
            'properties': {
                'path': {'type': 'string'},
                'name': {'type': 'string', 'default': 'default_name'},
            },
            'required': ['path'],
        }

        # Create a temporary schema file with the desired schema.
        schema_file = tmp_path / 'schema.json'
        schema_file.write_text(json.dumps(schema))
        # Create a temporary YAML configuration file containing only "path"
        config_data = {'path': '/a/b/c.txt'}
        config_file = tmp_path / 'valid_config.yml'
        config_file.write_text(yaml.dump(config_data))

        # Instantiate the validator with the schema file.
        validator = YAMLConfigValidator(schema_file)
        result = validator.validate_yaml(str(config_file))

        assert result['path'] == '/a/b/c.txt'
        assert result['name'] == 'default_name'  # Default applied

    def test_validate_yaml_validation_error(self, tmp_path):
        """
        Test YAML file validation error is generated
        """
        # Schema requires "path", so if it's missing, validation should error.
        schema = {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'type': 'object',
            'properties': {'path': {'type': 'string'}},
            'required': ['path'],
        }
        # Create a temporary schema file.
        schema_file = tmp_path / 'schema.json'
        schema_file.write_text(json.dumps(schema))

        # Create a YAML file missing the required "path" property.
        config_data = {'name': 'some_name'}
        config_file = tmp_path / 'config_invalid.yml'
        config_file.write_text(yaml.dump(config_data))
        validator_instance = YAMLConfigValidator(schema_file)
        with pytest.raises(YamlValidationError, match='YAML config validation error'):
            validator_instance.validate_yaml(str(config_file))

    # -----------------------------
    # Tests for export_schema_to_yaml: file contents
    # -----------------------------
    def test_export_schema_to_yaml(self, tmp_path):
        """
        Test yaml file contents to see if a yaml file with the correct input schema is saved to the file system
        """
        schema = {'test': 'value'}
        schema_file = tmp_path / 'schema.json'
        schema_file.write_text(json.dumps(schema))
        validator = YAMLConfigValidator(schema_file)

        # Test export without writing to a file.
        yaml_str = validator.export_schema_to_yaml()

        loaded_schema = yaml.safe_load(str(yaml_str))
        assert loaded_schema == schema

        # Test export with writing to a file.
        output_file = tmp_path / 'schema_output.yml'
        validator.export_schema_to_yaml(str(output_file))
        saved_file_content = output_file.read_text()
        loaded_schema_file = yaml.safe_load(saved_file_content)

        assert loaded_schema_file == schema

    # -----------------------------
    # Tests for export_schema_to_yaml: file creation
    # -----------------------------
    def test_yaml_file_creation(self, tmp_path, validator):
        """
        Test if a yaml file is saved to the file saved
        """
        # Define the data to be written to the YAML file
        schema = {'test': 'value'}
        validator.schema_content = schema

        # Define the path for the temporary YAML file
        output_file = tmp_path / 'schema_output.yml'

        # Write the data to the YAML file
        _ = validator.export_schema_to_yaml(str(output_file))

        # Check if the YAML file has been created
        assert output_file.exists()
