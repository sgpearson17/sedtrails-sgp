"""
Configuration Controller
========================

Reads simulation configuration files, applies default configuration values,
and provides configurations to other components.
"""

import os
from abc import ABC, abstractmethod

from sedtrails.configuration_interface.validator import YAMLConfigValidator


class Controller(ABC):
    """
    Abstract base class for configuration controllers.
    """

    # TODO: Something to keep in mind in this class is that
    # if we can and it is convenient to have separted methods to retrieve parts of the configuration
    # that go to the particle tracer and the transport converter. For example, if it is convenient to
    # have a method that returns the configuration values that are only relevant for the particle tracer.

    @abstractmethod
    def read_config(self, config_file):
        """
        Reads the configuration file and applies default values.
        """
        pass

    @abstractmethod
    def get_config(self):
        """
        Returns the current configuration.
        """
        pass


class ConfigurationController(Controller):
    """
    A controller for managing the configuration of the simulation.
    """

    def __init__(self):
        self.config_data = {}

    def read_config(self, config_file: str) -> None:
        """
        Reads the configuration file, validated its contents and set it.

        Parameters
        ----------
        config_file : str
            The path to the configuration file to read.
        """

        valid_json = os.path.join(r'/config_schema.json')  # not happy with hardcoded path

        if not os.path.exists(valid_json):
            raise FileNotFoundError(
                f'SedTRAILS validation schema file not found: {valid_json}. \
                Input cannot be parsed.'
            )

        validator = YAMLConfigValidator(valid_json)
        self.config_data = validator.validate_yaml(config_file)

        return None

    def get_config(self) -> dict:
        """
        Returns the current configuration.

        Returns
        -------
        dict
            The current configuration.
        """

        return self.config_data
