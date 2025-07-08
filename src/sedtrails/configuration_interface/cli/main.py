"""
SedTRAILS CLI interface.
"""

import typer
from pathlib import Path
from sedtrails.configuration_interface.configuration_controller import ConfigurationController


app = typer.Typer(help='Sedtrails: Configure, run, and analyze sediment particle tracking.')


# Subcommand to load and validate a YAML configuration file.
@app.command('load-config')
def load_config(
    config_file: Path = typer.Option(
        'sedtrails.yml',
        '--config',
        '-c',
        help='Path to the SedTRAILS configuration file.',
    ),
) -> dict:
    """
    Load and validate a YAML configuration file using an existing validator.
    Returns a dictionary with the valid configuration settings.

    Parameters
    ----------
      config_file : Path
         Path to the YAML configuration file.

    Returns
    -------
      dict
         A dictionary with the validated configuration settings.

    """

    controller = ConfigurationController()

    try:
        typer.echo(f"Loading and validating configuration from '{config_file}'...")
        # loads file and validates it
        controller.load_config(str(config_file))
        config = controller.get_config()
        # config = {'setting1': 'value1', 'setting2': 'value2'}
        typer.echo('Configuration validated successfully:')
        typer.echo(str(config))
        return config
    except Exception as e:
        typer.echo(f'Error loading configuration: {e}')
        raise typer.Exit(code=1) from e


# Subcommand to run a simulation; it also validates the configuration.
@app.command('run-simulation')
def run_simulation(
    config_file: Path = typer.Option(
        'sedtrails.yml',
        '--config',
        '-c',
        help='Path to the SedTRAILS configuration file.',
    ),
    output_file: Path = typer.Option(
        'sedtrails.nc',
        '--output',
        '-o',
        help='Path to the output SedTRAILS netCDF file.',
    ),
):
    """
    Validate the configuration and run a simulation based on it.
    The simulation results are written to a netCDF file.

    Parameters
    ----------
      config_file : Path
         Path to the SedTRAILS configuration file.
      output_file : Path
         Path to the output SedTRAILS netCDF file.

    Example
    -------
      sedtrails run-simulation --config my_config.yml --output results.nc

    """
    # First, load and validate the configuration.
    try:
        typer.echo(f"Validating configuration from '{config_file}'...")
        # from config_validator import ConfigValidator
        # config = ConfigValidator.validate(config_file)
        typer.echo('Configuration validated successfully.')
    except Exception as e:
        typer.echo(f'Error validating configuration: {e}')
        raise typer.Exit(code=1) from e

    # Run the simulation with the validated configuration.
    try:
        typer.echo('Running simulation...')
        pass
        typer.echo(f"Simulation complete. Output saved to '{output_file}'.")
    except Exception as e:
        typer.echo(f'Error running simulation: {e}')
        raise typer.Exit(code=1) from e


# Subcommand to perform statistical analysis on the simulation results.
@app.command('analyze')
def analyze(
    input_file: Path = typer.Option(
        'sedtrails.nc',
        '--input',
        '-i',
        help='Input SedTRAILS netCDF file containing particle tracks.',
    ),
    output_file: Path = typer.Option(
        'analysis.nc',
        '--output',
        '-o',
        help='Output SedTRAILS netCDF file containing statistical and connectivity results.',
    ),
):
    """
    Read the simulation netCDF file (if it exists) and perform a statistical analysis on the results.
    The analysis is saved to a netCDF file.

    Parameters
    ----------
      input_file : Path
         Path to the input SedTRAILS netCDF file containing particle tracks.
      output_file : Path
         Path to the output SedTRAILS netCDF file containing statistical and connectivity results.
    """
    try:
        typer.echo(f"Performing statistical analysis on '{input_file}'...")
        pass
        typer.echo(f"Analysis complete. Results saved to '{output_file}'.")
    except Exception as e:
        typer.echo(f'Error performing analysis: {e}')
        raise typer.Exit(code=1) from e


# Subcommand to perform network analysis on the simulation results.
@app.command('network-analysis')
def network_analysis(
    input_file: Path = typer.Option(
        'sedtrails.nc',
        '--input',
        '-i',
        help='Input netCDF file containing particle tracking results.',
    ),
    output_file: Path = typer.Option(
        'analysis.nc',
        '--output',
        '-o',
        help='Path to the output SedTRAILS netCDF file containing statistical and connectivity results.',
    ),
):
    # [Aysun] The original script had the same file path (sedtrails.nc) for both the input
    # and the output, based on the help text I assume the output should be the analysis.nc
    # instead? Please check the function below to make sure it is correct
    """
    Perform a network analysis on the simulation results.
    The network analysis results are written to a netCDF file.

    Parameters
    ----------
      input_file : Path
         Path to the input netCDF file containing particle tracking results.
      output_file : Path
         Path to the output netCDF file containing statistical and connectivity results.
    """
    try:
        typer.echo(f"Performing network analysis on '{input_file}'...")
        pass
        typer.echo(f"Network analysis complete. Results saved to '{output_file}'.")
    except Exception as e:
        typer.echo(f'Error performing network analysis: {e}')
        raise typer.Exit(code=1) from e


if __name__ == '__main__':
    app()
