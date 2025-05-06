"""
Configuration Controller
========================

Reads simulation configuration files, applies default configuration values,
and provides configurations to other components.
"""

from abc import ABC, abstractmethod


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
        self.config = {}

    def read_config(self, config_file: str) -> None:
        """
        Reads the configuration file, validated its contents and set it.

        Parameters
        ----------
        config_file : str
            The path to the configuration file to read.
        """

        #TODO: Implement reading from a configuration file
        # read the configuration file
        # validate the configuration

        self.config  = {}  # Placeholder for the actual configuration data

        return None

    def get_config(self) -> dict:
        """
        Returns the current configuration.

        Returns
        -------
        dict
            The current configuration.
        """

        return self.config
    

    def _validate_config(self) -> bool:
        """
        Validates the configuration.

        Parameters
        ----------
        config : dict
            The configuration to validate.

        Returns
        -------
        bool
            True if the configuration is valid, False otherwise.
        """
        # TODO: this shouold use the validator.py to validate the configuration
        
        pass
    

