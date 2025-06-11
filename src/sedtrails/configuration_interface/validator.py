import os
import jsonschema
import jsonschema.validators
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

    def _resolve_default_directive(self, directive: Dict[str, Any], root_data: Dict[str, Any]) -> str:
        """
        #TODO: DEPRICATE THIS FUNCTION
        Resolves a default directive specified as a dictionary with a "$ref" and optional
        transformation instructions.

        Parameters
        ----------
        directive : dict
            The default directive dictionary.
        root_data : dict
            The full configuration data (used to resolve the $ref).

        Returns
        -------
        str
            The computed default value as a string.

        Raises
        ------
        ValueError
            If the directive does not contain a '$ref
        TypeError:
            If the referenced value is not a string.
        NotImplementedError
            If the specified transformation is not supported.
        """
        if '$ref' not in directive:
            raise ValueError("Directive must contain a '$ref' key.")
        ref_value: Any = self._resolve_json_pointer(directive['$ref'], root_data)
        if not isinstance(ref_value, str):
            raise TypeError('Referenced value must be a string for folder name transformations.')
        result: str = ref_value
        # Apply transformation if specified.
        if 'transform' in directive:
            match directive['transform']:
                case 'dirname':
                    result = os.path.dirname(result)
                case _:
                    raise NotImplementedError(f"Transform '{directive['transform']}' is not supported.")
        # Apply prefix if specified.
        if 'prefix' in directive:
            result = directive['prefix'] + result
        if 'suffix' in directive:
            result = os.path.normpath(result + directive['suffix'])
        return result

    def _apply_defaults(self, schema: Dict[str, Any], data: Dict[str, Any], root_data: Dict[str, Any]) -> None:
        """
        #TODO: MODIFY THIS FUNCTION TO APPLY DEFAULTS ON THE SCHEMA
        Recursively applies default values from the schema to the data dictionary.
        If a default is specified as a dictionary with "$ref" and transformation keys,
        it is computed accordingly.

        Parameters
        ----------
        schema : dict
            The JSON schema (or subschema).
        data : dict
            The portion of configuration data to update.
        root_data : dict
            The full configuration data (for resolving references).

        Raises
        ------
        ValueError
            If the default value cannot be resolved.
        """
        if not isinstance(data, dict):
            return
        for key, subschema in schema.get('properties', {}).items():
            if key not in data and 'default' in subschema:
                default_val: Any = subschema['default']
                if isinstance(default_val, dict) and '$ref' in default_val:
                    try:
                        data[key] = self._resolve_default_directive(default_val, root_data)
                    except Exception as e:
                        raise ValueError(f"Error resolving default for '{key}': {e}") from e
                else:
                    data[key] = default_val
            if key in data and subschema.get('type') == 'object':
                self._apply_defaults(subschema, data[key], root_data)

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
            print(f'Validation error: {e.message}')

        # self._apply_defaults(self.schema, data, data)
        self.config = yaml_data
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
