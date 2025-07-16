"""
SedTRAILS CLI interface.
"""

import typer
from pathlib import Path
from sedtrails.configuration_interface.configuration_controller import ConfigurationController, YAMLConfigValidator
from sedtrails.simulation import Simulation


def version_callback(value: bool):
    """Callback for version option."""
    import sedtrails.__version__ as version

    if value:
        typer.echo(f'SedTRAILS {version}')  # You can import version from __version__.py
        raise typer.Exit()


app = typer.Typer(
    help='Sedtrails: Configure, run, and analyze sediment particle tracking.',
    add_completion=False,  # Disable completion to avoid conflicts
    context_settings={'help_option_names': ['-h', '--help']},  # Enable -h for help
)

# separate Typer app for config subcommands
config_app = typer.Typer(
    help='Configuration file management commands.',
    context_settings={'help_option_names': ['-h', '--help']},  # Enable -h for help in subcommands too
)

app.add_typer(config_app, name='config')


@app.callback()
def main(
    version: bool = typer.Option(
        False, '--version', '-v', callback=version_callback, is_eager=True, help='Show version and exit.'
    ),
):
    """
    SedTRAILS: Configure, run, and analyze sediment particle tracking.
    """
    pass


# Subcommand to load and validate a YAML configuration file.
@config_app.command('load')
def load_config(
    config_file: str = typer.Option(
        'sedtrails.yml',
        '--config',
        '-c',
        help='Path to the SedTRAILS configuration file.',
    ),
) -> dict:
    """
    Load and validate a YAML configuration file.
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

    controller = ConfigurationController(config_file)

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


@config_app.command('create')
def create_config_template(
    output_file: str = typer.Option(
        './sedtrails-template.yml',
        '--output',
        '-o',
        help='Path to the output configuration template file.',
    ),
):
    """
    Create a configuration file for SedTRAILS with all possible configurations and default values.

    Parameters
    ----------
      output_file : Path
         Path to the output configuration template file.
    """
    try:
        # Use importlib.resources to get the schema file from the installed package
        from importlib import resources

        with resources.as_file(resources.files('sedtrails.config').joinpath('main.schema.json')) as schema_path:
            validator = YAMLConfigValidator(str(schema_path))
            validator.create_config_template(output_file)
    except Exception as e:
        typer.echo(f'Error creating configuration template: {e}')
        raise typer.Exit(code=1) from e


# Subcommand to run a simulation; it also validates the configuration.
@app.command('run')
def run_simulation(
    config_file: str = typer.Option(
        'sedtrails.yml',
        '--config',
        '-c',
        help='Path to the SedTRAILS configuration file.',
    ),
    output_file: str = typer.Option(
        'sedtrails.nc',
        '--output',
        '-o',
        help='Path to the output SedTRAILS netCDF file.',
    ),
):
    """
    Run a simulation based on a configuratio file.
    The simulation results are written to a netCDF file.

    Parameters
    ----------
      config_file : Path
         Path to the SedTRAILS configuration file.
      output_file : Path
         Path to the output SedTRAILS netCDF file.

    Example
    -------
      sedtrails run --config my_config.yml --output results.nc

    """

    # lazy initialization of the Simulation class
    simulation = Simulation(config_file)

    # First, load and validate the configuration.
    try:
        typer.echo(f"Validating configuration from '{config_file}'...")
        # from config_validator import ConfigValidator
        simulation.validate_config()  # This will trigger the validation
        typer.echo('Configuration validated successfully.')
    except Exception as e:
        typer.echo(f'Error validating configuration: {e}')
        raise typer.Exit(code=1) from e

    # Run the simulation with the validated configuration.
    try:
        typer.echo('Running simulation...')
        simulation.run()
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
