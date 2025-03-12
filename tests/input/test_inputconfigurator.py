import os
import tempfile

import pytest
import yaml

from sedtrails.configuration_interface.validator import YAMLConfigValidator


class TestYAMLConfigValidator:
    def setup_method(self):
        # Create a temporary dummy schema file for tests that do not require a real schema.
        self.dummy_schema = {}
        self.dummy_schema_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yml"
        )
        yaml.dump(self.dummy_schema, self.dummy_schema_file)
        self.dummy_schema_file.close()
        self.validator = YAMLConfigValidator(self.dummy_schema_file.name)

    def teardown_method(self):
        # Remove the temporary dummy schema file.
        os.remove(self.dummy_schema_file.name)

    # -----------------------------
    # Tests for _resolve_json_pointer
    # -----------------------------
    def test_resolve_json_pointer_success(self):
        data = {"a": {"b": "value"}}
        pointer = "#/a/b"
        result = self.validator._resolve_json_pointer(pointer, data)
        assert result == "value"

    def test_resolve_json_pointer_with_properties(self):
        data = {"a": {"b": "value"}}
        # Pointer includes "properties" tokens that should be removed.
        pointer = "#/properties/a/properties/b"
        result = self.validator._resolve_json_pointer(pointer, data)
        assert result == "value"

    def test_resolve_json_pointer_failure(self):
        data = {"a": {"b": "value"}}
        pointer = "#/a/c"
        with pytest.raises(ValueError):
            self.validator._resolve_json_pointer(pointer, data)

    # -----------------------------
    # Tests for _resolve_default_directive
    # -----------------------------
    def test_resolve_default_directive(self):
        # Setup root data with a key "path" holding a file path.
        root_data = {"path": "/a/b/c.txt"}
        directive = {
            "$ref": "#/path",
            "transform": "dirname",
            "prefix": "pre_",
            "suffix": "_suf",
        }
        result = self.validator._resolve_default_directive(directive, root_data)
        expected = os.path.normpath("pre_" + os.path.dirname("/a/b/c.txt") + "_suf")
        assert result == expected

    def test_resolve_default_directive_missing_ref(self):
        root_data = {"path": "/a/b/c.txt"}
        directive = {"transform": "dirname"}
        with pytest.raises(ValueError):
            self.validator._resolve_default_directive(directive, root_data)

    def test_resolve_default_directive_non_string_ref(self):
        root_data = {"path": 123}
        directive = {"$ref": "#/path", "transform": "dirname"}
        with pytest.raises(ValueError):
            self.validator._resolve_default_directive(directive, root_data)

    def test_resolve_default_directive_unsupported_transform(self):
        root_data = {"path": "/a/b/c.txt"}
        directive = {"$ref": "#/path", "transform": "unsupported"}
        with pytest.raises(NotImplementedError):
            self.validator._resolve_default_directive(directive, root_data)

    # -----------------------------
    # Tests for _apply_defaults
    # -----------------------------
    def test_apply_defaults(self):
        # Define a schema with defaults (including a directive for "folder")
        schema = {
            "properties": {
                "folder": {
                    "type": "string",
                    "default": {
                        "$ref": "#/path",
                        "transform": "dirname",
                        "prefix": "pre_",
                        "suffix": "_suf",
                    },
                },
                "name": {"type": "string", "default": "default_name"},
                "nested": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "default": "nested_default"}
                    },
                },
            }
        }
        # Data missing "folder", "name", and nested.value.
        data = {"path": "/a/b/c.txt", "nested": {}}
        self.validator._apply_defaults(schema, data, data)
        expected_folder = os.path.normpath(
            "pre_" + os.path.dirname("/a/b/c.txt") + "_suf"
        )
        assert data["folder"] == expected_folder
        assert data["name"] == "default_name"
        assert data["nested"]["value"] == "nested_default"

    # -----------------------------
    # Tests for validate_yaml
    # -----------------------------
    def test_validate_yaml_success(self, tmp_path):
        # Define a schema that requires "path" and provides defaults for "name" and "folder"
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "name": {"type": "string", "default": "default_name"},
                "folder": {
                    "type": "string",
                    "default": {
                        "$ref": "#/path",
                        "transform": "dirname",
                        "prefix": "pre_",
                        "suffix": "_suf",
                    },
                },
            },
            "required": ["path"],
        }
        # Create a temporary schema file with the desired schema.
        schema_file = tmp_path / "schema.yml"
        schema_file.write_text(yaml.dump(schema))
        # Create a temporary YAML configuration file containing only "path"
        config_data = {"path": "/a/b/c.txt"}
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump(config_data))
        # Instantiate the validator with the schema file.
        validator_instance = YAMLConfigValidator(str(schema_file))
        config = validator_instance.validate_yaml(str(config_file))
        expected_folder = os.path.normpath(
            "pre_" + os.path.dirname("/a/b/c.txt") + "_suf"
        )
        assert config["path"] == "/a/b/c.txt"
        assert config["name"] == "default_name"
        assert config["folder"] == expected_folder

    def test_validate_yaml_validation_error(self, tmp_path):
        # Schema requires "path", so if it's missing, validation should error.
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        # Create a temporary schema file.
        schema_file = tmp_path / "schema.yml"
        schema_file.write_text(yaml.dump(schema))
        # Create a YAML file missing the required "path" property.
        config_data = {"name": "some_name"}
        config_file = tmp_path / "config_invalid.yml"
        config_file.write_text(yaml.dump(config_data))
        validator_instance = YAMLConfigValidator(str(schema_file))
        with pytest.raises(ValueError) as excinfo:
            validator_instance.validate_yaml(str(config_file))
        assert "YAML file validation error" in str(excinfo.value)

    # -----------------------------
    # Tests for export_schema_to_yaml: file contents
    # -----------------------------
    def test_export_schema_to_yaml(self, tmp_path):
        schema = {"test": "value"}
        self.validator.schema = schema
        # Test export without writing to a file.
        yaml_str = self.validator.export_schema_to_yaml()
        loaded_schema = yaml.safe_load(yaml_str)
        assert loaded_schema == schema

        # Test export with writing to a file.
        output_file = tmp_path / "schema_output.yml"
        yaml_str_2 = self.validator.export_schema_to_yaml(str(output_file))
        file_content = output_file.read_text()
        loaded_schema_file = yaml.safe_load(file_content)
        assert loaded_schema_file == schema
        assert yaml_str_2 == file_content

    # -----------------------------
    # Tests for export_schema_to_yaml: file creation
    # -----------------------------
    def test_yaml_file_creation(self, tmp_path):
        # Define the data to be written to the YAML file
        schema = {"test": "value"}
        self.validator.schema = schema

        # Define the path for the temporary YAML file
        output_file = tmp_path / "schema_output.yml"

        # Write the data to the YAML file
        yaml_str = self.validator.export_schema_to_yaml(str(output_file))

        # Check if the YAML file has been created
        assert output_file.exists()