import os
from typing import Any, Dict, Optional

import jsonschema
import yaml


class YAMLConfigValidator:
    """
    A class to load, validate, and process a YAML configuration file based on a JSON Schema.

    The class is initialized with the path to a valid JSON schema (in YAML or JSON format).
    Later, you can use the `validate_yaml` method to check if a YAML configuration adheres
    to the loaded schema. Default directives for folder names (using "$ref", "transform", etc.)
    are supported.
    """

    def __init__(self, schema_filepath: str) -> None:
        """
        Initialize the YAML configuration validator by loading the schema from an external file.

        Parameters
        ----------
        schema_filepath : str
            The path to the JSON schema file (YAML or JSON format).
        """
        self.schema: Dict[str, Any] = self.load_schema_from_file(schema_filepath)
        self.config: Dict[str, Any] = {}

    def load_schema_from_file(self, schema_file: str) -> Dict[str, Any]:
        """
        Loads the JSON schema from an external file and fills self.schema with its content.

        Parameters
        ----------
        schema_file : str
            The path to the external YAML or JSON schema file.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the schema file cannot be read or if the loaded schema is not a valid dictionary.
        """
        try:
            with open(schema_file, "r") as f:
                schema_data: Any = yaml.safe_load(f)
            if not isinstance(schema_data, dict):
                raise ValueError("Loaded schema is not a valid dictionary.")
            return schema_data
        except Exception as e:
            raise ValueError(f"Error reading schema file: {e}")

    def _resolve_json_pointer(self, pointer: str, data: Dict[str, Any]) -> Any:
        """
        Resolves a JSON pointer to a value in the provided data dictionary.
        This function also removes any "properties" segments so that a pointer like
        "#/properties/input_output_directories/properties/inputDir" maps to the
        data key path "input_output_directories/inputDir".

        Parameters
        ----------
        pointer : str
            The JSON pointer string.
        data : dict
            The root data dictionary.

        Returns
        -------
        Any
            The value pointed to in the data.

        Raises
        ------
        ValueError
            If the pointer cannot be fully resolved.
        """
        if pointer.startswith("#/"):
            pointer = pointer[2:]
        tokens = pointer.split("/")
        tokens = [token for token in tokens if token != "properties"]
        value: Any = data
        for token in tokens:
            if isinstance(value, dict) and token in value:
                value = value[token]
            else:
                raise ValueError(
                    f"Could not resolve pointer '{pointer}' in configuration data."
                )
        return value

    def _resolve_default_directive(
        self, directive: Dict[str, Any], root_data: Dict[str, Any]
    ) -> str:
        """
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
            If the directive does not contain a '$ref' or if the referenced value is not a string.
        NotImplementedError
            If the specified transformation is not supported.
        """
        if "$ref" not in directive:
            raise ValueError("Directive must contain a '$ref' key.")
        ref_value: Any = self._resolve_json_pointer(directive["$ref"], root_data)
        if not isinstance(ref_value, str):
            raise ValueError(
                "Referenced value must be a string for folder name transformations."
            )
        result: str = ref_value
        # Apply transformation if specified.
        if "transform" in directive:
            match directive["transform"]:
                case "dirname":
                    result = os.path.dirname(result)
                case _:
                    raise NotImplementedError(
                        f"Transform '{directive['transform']}' is not supported."
                    )
        # Apply prefix if specified.
        if "prefix" in directive:
            result = directive["prefix"] + result
        if "suffix" in directive:
            result = os.path.normpath(result + directive["suffix"])
        return result

    def _apply_defaults(
        self, schema: Dict[str, Any], data: Dict[str, Any], root_data: Dict[str, Any]
    ) -> None:
        """
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
        """
        if not isinstance(data, dict):
            return
        for key, subschema in schema.get("properties", {}).items():
            if key not in data and "default" in subschema:
                default_val: Any = subschema["default"]
                if isinstance(default_val, dict) and "$ref" in default_val:
                    try:
                        data[key] = self._resolve_default_directive(
                            default_val, root_data
                        )
                    except Exception as e:
                        raise ValueError(f"Error resolving default for '{key}': {e}")
                else:
                    data[key] = default_val
            if key in data and subschema.get("type") == "object":
                self._apply_defaults(subschema, data[key], root_data)

    def validate_yaml(self, yml_filepath: str) -> Dict[str, Any]:
        """
        Loads the YAML file, validates it against the loaded JSON schema,
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
        ValueError
            If the YAML file cannot be read or if the configuration is invalid.
        """
        try:
            with open(yml_filepath, "r") as f:
                data: Dict[str, Any] = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Error reading YAML file: {e}")

        ValidatorClass = jsonschema.validators.validator_for(self.schema)
        validator = ValidatorClass(self.schema)
        errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
        if errors:
            raise ValueError(f"YAML file validation error: {errors[0].message}")

        self._apply_defaults(self.schema, data, data)
        self.config = data
        return self.config

    def export_schema_to_yaml(self, output_file: Optional[str] = None) -> str:
        """
        Exports the JSON schema as a valid YAML string.

        Parameters
        ----------
        output_file : str, optional
            If provided, the YAML schema is written to this file.

        Returns
        -------
        str
            The schema as a YAML formatted string.

        Raises
        ------
        ValueError
            If there is an error writing the schema to the file.
        """
        schema_yaml: str = yaml.dump(self.schema, sort_keys=False)
        if output_file:
            try:
                with open(output_file, "w") as f:
                    f.write(schema_yaml)
            except Exception as e:
                raise ValueError(f"Error writing schema to file: {e}")
        return schema_yaml
