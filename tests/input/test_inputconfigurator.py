import os
import yaml
import pytest
from sedtrails.converters.inputconfiguration import YAMLConfigValidator


class TestYAMLConfigValidator:
    def setup_method(self):
        # Create an instance with a dummy file path (not used for direct method tests)
        self.validator = YAMLConfigValidator("dummy_path")

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
            "suffix": "_suf"
        }
        result = self.validator._resolve_default_directive(directive, root_data)
        # Expected steps:
        #   1. Resolve "$ref" → "/a/b/c.txt"
        #   2. Transform "dirname" → os.path.dirname("/a/b/c.txt") gives "/a/b"
        #   3. Apply prefix → "pre_/a/b"
        #   4. Append suffix and normalize → os.path.normpath("pre_/a/b_suf")
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
                        "suffix": "_suf"
                    }
                },
                "name": {
                    "type": "string",
                    "default": "default_name"
                },
                "nested": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "string",
                            "default": "nested_default"
                        }
                    }
                }
            }
        }
        # Data missing "folder", "name", and nested.default.
        data = {"path": "/a/b/c.txt", "nested": {}}
        self.validator._apply_defaults(schema, data, data)
        expected_folder = os.path.normpath("pre_" + os.path.dirname("/a/b/c.txt") + "_suf")
        assert data["folder"] == expected_folder
        assert data["name"] == "default_name"
        assert data["nested"]["value"] == "nested_default"

    # -----------------------------
    # Tests for load_and_validate
    # -----------------------------
    def test_load_and_validate_success(self, tmp_path):
        # Define a schema that requires "path" and provides defaults for "name" and "folder"
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema",
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
                        "suffix": "_suf"
                    }
                }
            },
            "required": ["path"]
        }
        # Create a temporary YAML file containing only "path"
        config_data = {"path": "/a/b/c.txt"}
        file_path = tmp_path / "config.yml"
        file_path.write_text(yaml.dump(config_data))
        # Instantiate the validator, set the schema, and load the config.
        validator_instance = YAMLConfigValidator(str(file_path))
        validator_instance.schema = schema
        config = validator_instance.load_and_validate()

        expected_folder = os.path.normpath("pre_" + os.path.dirname("/a/b/c.txt") + "_suf")
        assert config["path"] == "/a/b/c.txt"
        assert config["name"] == "default_name"
        assert config["folder"] == expected_folder

    def test_load_and_validate_validation_error(self, tmp_path):
        # Schema requires "path", so if it's missing the validator should error.
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema",
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
        }
        # Create a YAML file missing the required "path" property.
        config_data = {"name": "some_name"}
        file_path = tmp_path / "config_invalid.yml"
        file_path.write_text(yaml.dump(config_data))
        validator_instance = YAMLConfigValidator(str(file_path))
        validator_instance.schema = schema

        with pytest.raises(ValueError) as excinfo:
            validator_instance.load_and_validate()
        assert "YAML file validation error" in str(excinfo.value)

    # -----------------------------
    # Tests for export_schema_to_yaml
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
        # Read file and compare.
        file_content = output_file.read_text()
        loaded_schema_file = yaml.safe_load(file_content)
        assert loaded_schema_file == schema
        assert yaml_str_2 == file_content
