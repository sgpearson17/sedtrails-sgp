import yaml
from typing import str, Dict
from sedtrails.converters.inputconfiguration import YAMLConfigValidator

class SedtrailsModel():
    def __init__(self):
         self.config: Dict = None

    def load_and_validate(self, config_file: str) -> Dict:
       validator = YAMLConfigValidator(config_file)
       self.config = validator.load_and_validate()
       return self.config

    def export_config_to_yaml(self, output_file: str = None) -> str:
        """
        Exports the configuration dictionary as a valid YAML string. This can be used to 
        save the config to a file, or to a logging class.

        Parameters
        ----------
        output_file : str, optional
            If provided, the configuration is written to this file.

        Returns
        -------
        str
            The configuration as a YAML formatted string.

        Raises
        ------
        ValueError
            If the configuration is not available or if there is an error writing to the file.
        """
        if not self.config:
            raise ValueError("Configuration has not been loaded. Please call load_and_validate() first.")
        config_yaml = yaml.dump(self.config, sort_keys=False)
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(config_yaml)
            except Exception as e:
                raise ValueError(f"Error writing configuration to file: {e}")
        return config_yaml