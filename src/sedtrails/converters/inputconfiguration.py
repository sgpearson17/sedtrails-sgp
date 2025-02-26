import os
import yaml
import jsonschema
from typing import Any, Dict, Optional

class YAMLConfigValidator:
    """
    A class to load, validate, and process a YAML configuration file based on a JSON Schema.

    This updated version uses the draft 2020-12 standard and supports default directives
    for folder names. For any property whose default is given as a dictionary with a "$ref",
    a "transform" (currently supporting "dirname"), and optional "prefix" or "suffix", the directive
    is resolved using the referenced value from the root configuration.
    """

    def __init__(self, yml_filepath: str) -> None:
        """
        Initialize the YAML configuration validator.

        Parameters
        ----------
        yml_filepath : str
            The path to the YAML configuration file.
        """
        self.yml_filepath: str = yml_filepath
        self.config: Dict[str, Any] = {}

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
        tokens = pointer.split('/')
        # Remove "properties" tokens to convert schema path to data path.
        tokens = [token for token in tokens if token != "properties"]
        value: Any = data
        for token in tokens:
            if isinstance(value, dict) and token in value:
                value = value[token]
            else:
                raise ValueError(f"Could not resolve pointer '{pointer}' in configuration data.")
        return value

    def _resolve_default_directive(self, directive: Dict[str, Any], root_data: Dict[str, Any]) -> str:
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
        # Resolve the reference.
        if "$ref" not in directive:
            raise ValueError("Directive must contain a '$ref' key.")
        ref_value: Any = self._resolve_json_pointer(directive["$ref"], root_data)
        if not isinstance(ref_value, str):
            raise ValueError("Referenced value must be a string for folder name transformations.")
        result: str = ref_value
        # Apply transformation if specified.
        if "transform" in directive:
            match directive["transform"]:
               case "dirname":
                   result = os.path.dirname(result)
               case _:
                   raise NotImplementedError(f"Transform '{directive['transform']}' is not supported.")
        # Apply prefix if specified.
        if "prefix" in directive:
            result = directive["prefix"] + result
        # Apply suffix if specified.
        if "suffix" in directive:
            # Using os.path.normpath to normalize the resulting path.
            result = os.path.normpath(result + directive["suffix"])
        return result

    def _apply_defaults(self, schema: Dict[str, Any], data: Dict[str, Any], root_data: Dict[str, Any]) -> None:
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
                # Check if the default value is a directive.
                if isinstance(default_val, dict) and "$ref" in default_val:
                    try:
                        data[key] = self._resolve_default_directive(default_val, root_data)
                    except Exception as e:
                        raise ValueError(f"Error resolving default for '{key}': {e}")
                else:
                    data[key] = default_val
            # Recurse if the property is an object.
            if key in data and subschema.get("type") == "object":
                self._apply_defaults(subschema, data[key], root_data)

    def load_and_validate(self) -> Dict[str, Any]:
        """
        Loads the YAML file, validates it against the JSON schema (draft 2020-12),
        applies default values (including processing default folder directives), and
        returns the resulting configuration as a nested dictionary.

        Returns
        -------
        dict
            The validated and default-populated configuration as a nested dictionary.

        Raises
        ------
        ValueError
            If the YAML file cannot be read or if the configuration is invalid.
        """
        # Load YAML file.
        try:
            with open(self.yml_filepath, 'r') as f:
                data: Dict[str, Any] = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Error reading YAML file: {e}")

        # Determine and use the appropriate validator for the schema.
        ValidatorClass = jsonschema.validators.validator_for(self.schema)
        validator = ValidatorClass(self.schema)
        # Validate the loaded data.
        errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
        if errors:
            raise ValueError(f"YAML file validation error: {errors[0].message}")

        # Apply defaults. Pass the full data as root_data for resolving references.
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
                with open(output_file, 'w') as f:
                    f.write(schema_yaml)
            except Exception as e:
                raise ValueError(f"Error writing schema to file: {e}")
        return schema_yaml
