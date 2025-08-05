"""
A custom YAML loader is necessary because default loader always converts datetime strings to datetime objects
We want to keey datetime as strings, for convenience convertions are handled internally
by the SedTRAILS configuration interface
"""

import yaml


class SedtrailsYamlLoader(yaml.SafeLoader):
    """Custom YAML loader that avoids converting datetime strings to datetime objects."""

    pass


# Add a constructor that treats timestamps as strings instead of datetime objects
def construct_timestamp_as_string(loader, node):
    """Construct timestamp nodes as strings instead of datetime objects."""
    return loader.construct_scalar(node)


SedtrailsYamlLoader.add_constructor('tag:yaml.org,2002:timestamp', construct_timestamp_as_string)
