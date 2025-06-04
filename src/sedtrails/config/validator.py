# schema_to_yaml_example.py
import json
import yaml
from pathlib import Path
import jsonschema

# TODO: implement this validator into validator component


def validate_sedtrails_config(config_file: str, schema_file: str) -> bool:
    """
    Validate the SedTrails configuration file against the schema.
    :param config_file: Path to the configuration file.
    :param schema_file: Path to the JSON schema file.

    :return: True if valid, False otherwise.

    """

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    schema_path = Path(schema_file).resolve()
    with open(schema_path, 'r') as f:
        schema_content = json.load(f)

    resolver = jsonschema.RefResolver(base_uri=f'file://{schema_path.parent}/', referrer=schema_content)

    try:
        jsonschema.validate(instance=config, schema=schema_content, resolver=resolver)
    except jsonschema.ValidationError as e:
        print(f'Validation error: {e.message}')
        return False
    else:
        return True


if __name__ == '__main__':
    example = 'examples/config.example.yaml'

    schema = 'src/sedtrails/config/main.schema.json'

    validate_sedtrails_config(example, schema)
