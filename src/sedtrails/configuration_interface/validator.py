import jsonschema
import yaml
import json
from typing import Any, Dict, Optional
from sedtrails.exceptions import YamlParsingError, YamlOutputError, YamlValidationError
from pathlib import Path


class YAMLConfigValidator:
    """
    A class to load, validate, and process a YAML configuration file based on a JSON Schema.

    The class is initialized with the path to a valid JSON schema (in YAML or JSON format).
    Later, you can use the `validate_yaml` method to check if a YAML configuration adheres
    to the loaded schema. Default directives for folder names (using "$ref", "transform", etc.)
    are supported.

    Attributes
    ----------
    schema_content : Dict[str, Any]
        The loaded JSON schema content as a dictionary.
    schema_path : Path
        The resolved path to the JSON schema file.
    config : Dict[str, Any]
        The validated configuration data as a nested dictionary.
    """

    def __init__(self, schema_filepath: str) -> None:
        """
        Initialize the YAML configuration validator by loading the schema from an external file.

        Parameters
        ----------
        schema_filepath : str
            The path to the JSON schema file (YAML or JSON format).
        """

        self.schema_content: Dict[str, Any] = self.load_schema_from_file(schema_filepath)
        self.schema_path = Path(schema_filepath).resolve()
        self.config: Dict[str, Any] = {}

    def load_schema_from_file(self, schema_file: str) -> Dict[str, Any]:
        """
        Loads the JSON schema from an external file and fills self.schema with its content.

        Parameters
        ----------
        schema_file : str
            The path to JSON schema file.

        Returns
        -------
        Dict[str, Any]
            The loaded schema as a dictionary.

        Raises
        ------
        json.JSONDecodeError
            If the schema file is not a valid JSON file.
        """

        try:
            with open(schema_file, 'r') as f:
                schema_data: Any = json.load(f)
        except json.JSONDecodeError as err:
            raise err
        else:
            return schema_data

    def _apply_defaults(self, schema_content: Dict[str, Any], config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively applies default values from the schema to the data dictionary.
        If a default is specified as a dictionary with "$ref" and transformation keys,
        it is computed accordingly.

        Parameters
        ----------
        schema_content : dict
            The JSON schema (or subschema).
        config_data : dict
            The configuration data to which defaults should be applied.

        Returns
        -------
        dict
            The configuration data with defaults applied.

        Raises
        ------
        ValueError
            If the default value cannot be resolved.
        """

        schema_type = schema_content.get('type')

        if schema_type == 'object' and isinstance(config_data, dict):
            properties = schema_content.get('properties', {})
            for key, prop_schema in properties.items():
                if key not in config_data:
                    # Create missing property with default value
                    if 'default' in prop_schema:
                        config_data[key] = prop_schema['default']
                    elif prop_schema.get('type') == 'object':
                        # Create empty object and apply defaults recursively
                        config_data[key] = {}
                        config_data[key] = self._apply_defaults(prop_schema, config_data[key])
                else:
                    # Property exists, apply defaults recursively
                    config_data[key] = self._apply_defaults(prop_schema, config_data[key])

        elif schema_type == 'array' and isinstance(config_data, list):
            item_schema = schema_content.get('items', {})
            for i, item in enumerate(config_data):
                config_data[i] = self._apply_defaults(item_schema, item)

        return config_data

    def validate_yaml(self, yml_filepath: str) -> Dict[str, Any]:
        """
        Loads the YAML file, validates it against the SedTRAILS JSON schema,
        applies default values (including processing default folder directives), and
        returns the resulting configuration as a nested dictionary.

        Parameters
        ----------
        yml_filepath : str
            The path to the YAML configuration file to validate.

        Returns
        -------
        dict
            The validated and default-populated configuration as a nested dictionary.

        Raises
        ------
        YamlParsingError
            If the YAML file cannot be read
        YamlValidationError
            If the YAML configuration is invalid.
        ValueError
            If there is an error applying default values from the schema.
        """

        # loading the YAML file
        try:
            with open(yml_filepath, 'r') as f:
                yaml_data: Dict[str, Any] = yaml.safe_load(f)
        except Exception as e:
            raise YamlParsingError(f'Error reading YAML file: {e}') from e

        # Resolver to handle references in different files
        resolver = jsonschema.RefResolver(base_uri=f'file://{self.schema_path.parent}/', referrer=self.schema_content)

        # Validate the YAML data against the schema
        try:
            jsonschema.validate(instance=yaml_data, schema=self.schema_content, resolver=resolver)

        except jsonschema.ValidationError as e:
            raise YamlValidationError(f'YAML config validation error: {e.message}') from e

        # apply default values from the schema
        try:
            config_with_defaults = self._apply_defaults(self.schema_content, config_data=yaml_data)
        except Exception as e:
            raise ValueError(f'Error applying defaults: {e}') from e
        else:
            self.config = config_with_defaults
            return self.config

    def export_schema_to_yaml(self, output_file: Optional[str] = None) -> str | None:
        """
        Exports the JSON schema as a valid YAML string.

        Parameters
        ----------
        output_file : str, optional
            If provided, the YAML schema is written to this file.

        Returns
        -------
        str
            The schema as a YAML formatted string, if no output file is specified.
        If an output file is specified, the schema is written to that file.

        Raises
        ------
        YamlOutputError
            If there is an error while writing the schema to the file.
        """
        schema_yaml: str = yaml.dump(self.schema_content, sort_keys=False)
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(schema_yaml)
            except Exception as e:
                raise YamlOutputError(f'Error writing schema to file: {e}') from e
        else:
            return schema_yaml

    def export_config_to_yaml(self, output_file: str = './example.sedtrails.yaml') -> None:
        """
        Exports the validated configuration as a valid YAML string.

        Parameters
        ----------
        output_file : str
            path to file where the YAML configuration is written.

        Returns
        -------
        The configuration is written to that file.

        Raises
        ------
        YamlOutputError
            If there is an error while writing the configuration to the file.
        """

        self._apply_defaults(self.schema_content, self.config)
        print(f'Applied defaults to config: {self.config}')

        config_yaml: str = yaml.dump(self.config, sort_keys=False)

        print(f'config yaml: {config_yaml}')
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(config_yaml)
            except Exception as e:
                raise YamlOutputError(f'Error writing config to file: {e}') from e

    def create_config_template(self, output_file: str = './sedtrails.template.yaml') -> None:
        """
        Creates a configuration template with all schema defaults applied.

        Parameters
        ----------
        output_file : str
            Path to file where the YAML configuration template is written.

        Raises
        ------
        YamlOutputError
            If there is an error while writing the configuration to the file.
        """

        try:
            # Create empty config and apply all schema defaults
            empty_config = {}
            config_with_all_defaults = self._apply_defaults(self.schema_content, empty_config)

            # Create output directory if needed
            from pathlib import Path

            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to YAML
            config_yaml = yaml.dump(
                config_with_all_defaults, default_flow_style=False, sort_keys=False, indent=2, allow_unicode=True
            )

            # Write to file with header
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('# SedTRAILS Configuration Template\n')
                f.write('# Generated using defaults for configuration properties\n\n')
                f.write(config_yaml)

            print(f'Configuration template exported to: {output_path.absolute()}')

        except Exception as e:
            raise YamlOutputError(f'Error writing config template to file: {e}') from e
