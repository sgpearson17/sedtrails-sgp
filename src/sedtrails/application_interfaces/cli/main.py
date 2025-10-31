"""
SedTRAILS CLI interface.
"""

import typer
from pathlib import Path


def version_callback(value: bool):
    """Callback for version option."""
    import sedtrails.__version__ as version

    if value:
        typer.echo(f'SedTRAILS {version}')  # import version from __version__.py
        raise typer.Exit()


app = typer.Typer(
    help='Sedtrails: Configure, run, and analyze sediment particle tracking.',
    add_completion=False,  # Disable completion to avoid conflicts
    context_settings={'help_option_names': ['-h', '--help']},  # Enable -h for help
)


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


# Subcommand to run a simulation; it also validates the configuration.
@app.command('run')
def run_simulation_cmd(
    config_file: str = typer.Option(
        'sedtrails.yml',
        '--config',
        '-c',
        help='Path to the SedTRAILS configuration file.',
    ),
):
    """
    Run a simulation based on a configuration file.
    The simulation results are written to a netCDF file.
    The output location is determined by the configuration file.

    Example: sedtrails run --config my_config.yml

    """
    from sedtrails.application_interfaces.api import run_simulation

    try:
        typer.echo(f"Starting simulation from '{config_file}'...")
        output_dir = run_simulation(config_file=config_file, verbose=True)
        typer.echo(f"Simulation complete. Results saved to '{output_dir}'.")
    except Exception as e:
        typer.echo(f'Error running simulation: {e}')
        raise typer.Exit(code=1) from e


@app.command('inspect')
def inspect_metadata(
    results_file: str = typer.Option(
        'sedtrails_results.nc',
        '--file',
        '-f',
        help='Path to the SedTRAILS netCDF file to inspect. By default, it expects an "sedtrails_results.nc" file in the current directory.',
    ),
    populations: bool = typer.Option(
        False,
        '--populations',
        '-p',
        help='Inspect and print detailed information about particle populations in the file.',
    ),
):
    """
    Print metadata information about a SedTRAILS netCDF results file.
    """
    from sedtrails.application_interfaces.api import inspect_netcdf

    try:
        inspector = inspect_netcdf(results_file)
        inspector.print_metadata()  # print general metadata
        if populations:
            inspector.inspect_populations()  # print particle population info

    except Exception as e:
        typer.echo(f'Error inspecting metadata: {e}')
        raise typer.Exit(code=1) from e


######################################################################################################
# CONFIG subcommands
######################################################################################################
config_app = typer.Typer(
    help='Commands for managing Configuration files.',
    context_settings={'help_option_names': ['-h', '--help']},
)
app.add_typer(config_app, name='config')


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
    Checks if a YAML configuration file is a valid SedTRAILS configuration.
    Returns a dictionary with the valid configuration settings.
    """
    from sedtrails.application_interfaces.api import load_configuration

    try:
        typer.echo(f"Loading and validating configuration from '{config_file}'...")
        config = load_configuration(config_file)
        typer.echo('Configuration validated successfully:')
        typer.echo(str(config))
        return config
    except Exception as e:
        typer.echo(f'Error loading configuration: {e}')
        raise typer.Exit(code=1) from e


@config_app.command('create')
def create_config_template_cmd(
    output_file: str = typer.Option(
        './sedtrails-template.yml',
        '--output',
        '-o',
        help='Path to the output configuration template file.',
    ),
):
    """
    Create a configuration file for simulations in SedTRAILS.
    The file contains most possible configurations items with default values.
    """
    from sedtrails.application_interfaces.api import create_config_template

    try:
        create_config_template(output_file)
        typer.echo(f"Configuration template created at '{output_file}'")
    except Exception as e:
        typer.echo(f'Error creating configuration template: {e}')
        raise typer.Exit(code=1) from e


######################################################################################################
# ANALYZER subcommands
######################################################################################################
analyzer_app = typer.Typer(
    help='Commands for analyzing simulation results. NOT IMPLEMENTED.',
    context_settings={'help_option_names': ['-h', '--help']},
)
app.add_typer(analyzer_app, name='analyzer')


# Subcommand to perform statistical analysis on the simulation results.
@analyzer_app.command('analyze')
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
        typer.echo('THIS IS HAS NOT BEEN IMPLEMENTED YET.')
    except Exception as e:
        typer.echo(f'Error performing analysis: {e}')
        raise typer.Exit(code=1) from e


######################################################################################################
# NETWORK subcommands
######################################################################################################
network_app = typer.Typer(
    help='Commands to perform network analysi on simulation results. NOT IMPLEMENTED.',
    context_settings={'help_option_names': ['-h', '--help']},
)
app.add_typer(network_app, name='network')


# Subcommand to perform network analysis on the simulation results.
@network_app.command('analysis')
def analysis(
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
        typer.echo('THIS IS HAS NOT BEEN IMPLEMENTED YET.')

    except Exception as e:
        typer.echo(f'Error performing network analysis: {e}')
        raise typer.Exit(code=1) from e


######################################################################################################
# VIZ subcommands
######################################################################################################
vizualizer_app = typer.Typer(
    help='Commands for visualizing simulation results.',
    context_settings={'help_option_names': ['-h', '--help']},
)
app.add_typer(vizualizer_app, name='viz')


@vizualizer_app.command('trajectories')
def plot_trajectories_cmd(
    results_file: str = typer.Option(
        'sedtrails_results.nc',
        '--file',
        '-f',
        help='Path to the SedTRAILS netCDF file to visualize. By default, it expects an "sedtrails_results.nc" file in the current directory.',
    ),
    save_fig: bool = typer.Option(
        False,
        '--save',
        '-s',
        help='Save plot as a PNG file. Creates a "particle_trajectories.png" file',
    ),
    output_dir: str = typer.Option(
        '.',
        '--output-dir',
        '-o',
        help='Directory to save plot if --save is used. Default is the current directory.',
    ),
):
    """
    Plot particle trajectories from a SedTRAILS netCDF results file.
    """
    from sedtrails.application_interfaces.api import plot_trajectories

    try:
        plot_trajectories(results_file, save=save_fig, output_dir=output_dir)
        if not save_fig:
            typer.echo('Plot displayed successfully')
    except Exception as e:
        typer.echo(f'Error plotting trajectories: {e}')
        raise typer.Exit(code=1) from e


if __name__ == '__main__':
    app()
