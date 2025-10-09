"""
Application Interface (API)
===========================

This module provides the main Application Interface (Python API) for SedTRAILS.
It exposes functions and classes to configure and run SedTRAILS simulations
programmatically, and provides access to visualization and data analysis
functionalities.

This is the primary interface for Python users who want to use SedTRAILS
programmatically rather than through the CLI.
"""

from typing import Any, Dict, Optional
import logging


# ============================================================================
# Simulation Functions
# ============================================================================


def run_simulation(
    config_file: str,
    output_file: str = 'sedtrails.nc',
    validate: bool = True,
    verbose: bool = False,
) -> str:
    """
    Run a SedTRAILS simulation from a configuration file.

    This is the main high-level function for running simulations. It handles
    configuration loading, validation, execution, and output generation.

    Parameters
    ----------
    config_file : str
        Path to the SedTRAILS configuration YAML file.
    output_file : str, optional
        Path to the output NetCDF file. Default is 'sedtrails.nc'.
    validate : bool, optional
        Whether to validate the configuration before running. Default is True.
    verbose : bool, optional
        Enable verbose logging output. Default is False.

    Returns
    -------
    str
        Path to the generated output file.

    Raises
    ------
    ConfigurationError
        If the configuration file is invalid.
    FileNotFoundError
        If the configuration file does not exist.

    Examples
    --------
    >>> import sedtrails
    >>> output = sedtrails.run_simulation('config.yml', output='results.nc')
    >>> print(f"Results saved to: {output}")

    >>> # Run with validation disabled
    >>> sedtrails.run_simulation('config.yml', validate=False)
    """
    from sedtrails.simulation import Simulation
    from sedtrails.exceptions.exceptions import ConfigurationError

    # Set up logging if verbose
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Create simulation instance
    try:
        simulation = Simulation(config_file)
    except Exception as e:
        raise ConfigurationError(f'Failed to initialize simulation: {e}') from e

    # Validate configuration if requested
    if validate:
        try:
            simulation.validate_config()
        except Exception as e:
            raise ConfigurationError(f'Configuration validation failed: {e}') from e

    # Run the simulation
    simulation.run()

    # Return the output file path
    # Note: The actual output path is determined by the simulation
    # This will need adjustment based on how simulation.run() handles output
    return output_file


def create_simulation_runner(config_file: str) -> 'SimulationRunner':
    """
    Create a SimulationRunner instance for advanced control over simulation execution.

    This function provides access to a more detailed interface for users who need
    fine-grained control over the simulation process, such as step-by-step execution,
    inspection of intermediate states, or custom callbacks.

    Parameters
    ----------
    config_file : str
        Path to the SedTRAILS configuration YAML file.

    Returns
    -------
    SimulationRunner
        A SimulationRunner instance configured with the given configuration file.

    Examples
    --------
    >>> import sedtrails
    >>> runner = sedtrails.create_simulation_runner('config.yml')
    >>> runner.validate()
    >>> runner.run(enable_dashboard=True)
    >>> runner.save_results('output.nc')
    """
    return SimulationRunner(config_file)


# ============================================================================
# Configuration Functions
# ============================================================================


def load_configuration(config_file: str) -> Dict[str, Any]:
    """
    Load and validate a SedTRAILS configuration file.

    This function reads a YAML configuration file, validates it against the
    SedTRAILS schema, and returns the configuration as a dictionary.

    Parameters
    ----------
    config_file : str
        Path to the SedTRAILS configuration YAML file.

    Returns
    -------
    dict
        The validated configuration dictionary.

    Raises
    ------
    ConfigurationError
        If the configuration file is invalid or cannot be loaded.
    FileNotFoundError
        If the configuration file does not exist.

    Examples
    --------
    >>> import sedtrails
    >>> config = sedtrails.load_configuration('config.yml')
    >>> print(config['particles']['populations'])
    """
    from sedtrails.application_interfaces.configuration_controller import ConfigurationController
    from sedtrails.exceptions.exceptions import ConfigurationError

    try:
        controller = ConfigurationController(config_file)
        controller.load_config(config_file)
        return controller.get_config()
    except Exception as e:
        raise ConfigurationError(f'Failed to load configuration: {e}') from e


def validate_configuration(config_file: str) -> bool:
    """
    Validate a SedTRAILS configuration file without loading the full simulation.

    Parameters
    ----------
    config_file : str
        Path to the SedTRAILS configuration YAML file.

    Returns
    -------
    bool
        True if the configuration is valid, False otherwise.

    Examples
    --------
    >>> import sedtrails
    >>> if sedtrails.validate_configuration('config.yml'):
    ...     print("Configuration is valid!")
    ... else:
    ...     print("Configuration has errors")
    """
    try:
        load_configuration(config_file)
        return True
    except Exception:
        return False


def create_config_template(output_file: str = './sedtrails-template.yml') -> None:
    """
    Create a SedTRAILS configuration template file.

    This function generates a YAML configuration file with default values and
    documentation for all available configuration options.

    Parameters
    ----------
    output_file : str, optional
        Path where the template file will be created.
        Default is './sedtrails-template.yml'.

    Examples
    --------
    >>> import sedtrails
    >>> sedtrails.create_config_template('my_template.yml')
    """
    from sedtrails.application_interfaces.validator import YAMLConfigValidator

    validator = YAMLConfigValidator()
    validator.create_config_template(output_file)


# ============================================================================
# Visualization Functions
# ============================================================================


def plot_trajectories(
    results_file: str,
    save: bool = False,
    output_dir: str = '.',
) -> None:
    """
    Plot particle trajectories from a SedTRAILS NetCDF results file.

    Parameters
    ----------
    results_file : str
        Path to the SedTRAILS NetCDF results file.
    save : bool, optional
        Whether to save the plot as a PNG file. Default is False (display only).
    output_dir : str, optional
        Directory where the plot will be saved if save=True.
        Default is current directory.

    Examples
    --------
    >>> import sedtrails
    >>> sedtrails.plot_trajectories('results.nc', save=True)

    >>> # Display without saving
    >>> sedtrails.plot_trajectories('results.nc')
    """
    from sedtrails.pathway_visualizer import plot_trajectories as _plot, read_netcdf

    ds = read_netcdf(results_file)
    _plot(ds, save_plot=save, output_dir=output_dir)


def inspect_netcdf(results_file: str) -> 'NetCDFInspector':
    """
    Create an inspector for examining SedTRAILS NetCDF results metadata.

    Parameters
    ----------
    results_file : str
        Path to the SedTRAILS NetCDF results file.

    Returns
    -------
    NetCDFInspector
        An inspector instance that can be used to examine the file metadata.

    Examples
    --------
    >>> import sedtrails
    >>> inspector = sedtrails.inspect_netcdf('results.nc')
    >>> inspector.print_metadata()
    >>> inspector.inspect_populations()
    """
    from sedtrails.nc_inspector import NetCDFInspector

    return NetCDFInspector(results_file)


# ============================================================================
# Analysis Functions (Placeholder for future implementation)
# ============================================================================


def analyze_simulation(
    input_file: str,
    output_file: str = 'analysis.nc',
) -> str:
    """
    Perform statistical analysis on simulation results.

    Note: This function is not yet fully implemented.

    Parameters
    ----------
    input_file : str
        Path to the SedTRAILS NetCDF results file.
    output_file : str, optional
        Path to save the analysis results. Default is 'analysis.nc'.

    Returns
    -------
    str
        Path to the analysis output file.

    Raises
    ------
    NotImplementedError
        This functionality is not yet implemented.
    """
    raise NotImplementedError(
        'Statistical analysis functionality is not yet fully implemented. This feature is planned for a future release.'
    )


def analyze_network(
    input_file: str,
    output_file: str = 'network_analysis.nc',
) -> str:
    """
    Perform network connectivity analysis on simulation results.

    Note: This function is not yet fully implemented.

    Parameters
    ----------
    input_file : str
        Path to the SedTRAILS NetCDF results file.
    output_file : str, optional
        Path to save the network analysis results. Default is 'network_analysis.nc'.

    Returns
    -------
    str
        Path to the network analysis output file.

    Raises
    ------
    NotImplementedError
        This functionality is not yet implemented.
    """
    raise NotImplementedError(
        'Network analysis functionality is not yet fully implemented. This feature is planned for a future release.'
    )


# ============================================================================
# Advanced API Classes
# ============================================================================


class SimulationRunner:
    """
    Advanced interface for controlling SedTRAILS simulations programmatically.

    This class provides fine-grained control over the simulation process,
    allowing users to validate, configure, and execute simulations step-by-step.

    Parameters
    ----------
    config_file : str
        Path to the SedTRAILS configuration YAML file.

    Attributes
    ----------
    config_file : str
        Path to the configuration file.
    simulation : Simulation or None
        The underlying Simulation object (created after validate() is called).

    Examples
    --------
    >>> runner = SimulationRunner('config.yml')
    >>> runner.validate()
    >>> runner.run(enable_dashboard=True)
    >>> runner.save_results('output.nc')
    """

    def __init__(self, config_file: str):
        """
        Initialize the SimulationRunner.

        Parameters
        ----------
        config_file : str
            Path to the SedTRAILS configuration YAML file.
        """
        self.config_file = config_file
        self.simulation = None
        self._config = None

    def validate(self) -> bool:
        """
        Validate the configuration file.

        Returns
        -------
        bool
            True if validation succeeds.

        Raises
        ------
        ConfigurationError
            If validation fails.
        """
        from sedtrails.simulation import Simulation

        try:
            self.simulation = Simulation(self.config_file)
            self.simulation.validate_config()
            return True
        except Exception as e:
            from sedtrails.exceptions.exceptions import ConfigurationError

            raise ConfigurationError(f'Validation failed: {e}') from e

    def run(self, enable_dashboard: Optional[bool] = None, verbose: bool = False) -> None:
        """
        Run the simulation.

        Parameters
        ----------
        enable_dashboard : bool, optional
            Override the dashboard setting from the configuration.
            If None, uses the configuration value.
        verbose : bool, optional
            Enable verbose logging output. Default is False.

        Raises
        ------
        RuntimeError
            If validate() has not been called first.
        """
        if self.simulation is None:
            raise RuntimeError('Must call validate() before run()')

        if verbose:
            logging.basicConfig(level=logging.INFO)

        # TODO: Implement dashboard override when supported by Simulation class
        if enable_dashboard is not None:
            logging.warning('Dashboard override not yet implemented, using config value')

        self.simulation.run()

    def get_config(self) -> Dict[str, Any]:
        """
        Get the loaded configuration.

        Returns
        -------
        dict
            The configuration dictionary.

        Raises
        ------
        RuntimeError
            If validate() has not been called first.
        """
        if self.simulation is None:
            raise RuntimeError('Must call validate() before get_config()')

        return self.simulation.config

    def save_results(self, output_file: str) -> str:
        """
        Save simulation results to a file.

        Note: This method is a placeholder. The current Simulation class
        handles output automatically during run().

        Parameters
        ----------
        output_file : str
            Path to save the results.

        Returns
        -------
        str
            Path to the saved file.
        """
        # TODO: This will need to be implemented when Simulation class
        # supports separate save functionality
        logging.info(f'Results will be saved to {output_file}')
        logging.warning('save_results() is not fully implemented yet')
        return output_file


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # Simulation
    'run_simulation',
    'create_simulation_runner',
    'SimulationRunner',
    # Configuration
    'load_configuration',
    'validate_configuration',
    'create_config_template',
    # Visualization
    'plot_trajectories',
    'inspect_netcdf',
    # Analysis (not yet implemented)
    'analyze_simulation',
    'analyze_network',
]
