"""
SedTRAILS CLI interface.
"""

import typer
from pathlib import Path


def version_callback(value: bool):
    """
    Callback for version option.

    Parameters
    ----------
    value : bool
        Value to assign or validate.
    """
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

    Parameters
    ----------
    version : bool
        The version value.
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
    report_domain_exits: bool = typer.Option(
        True,
        '--report-domain-exits/--no-report-domain-exits',
        help='Report particles that leave the model domain or beach on land during and after the run.',
    ),
):
    """
    Run a simulation based on a configuration file.
    The simulation results are written to a netCDF file.
    The output location is determined by the configuration file.

    Example: sedtrails run --config my_config.yml

    Parameters
    ----------
    config_file : str
        Path to the SedTRAILS configuration file.
    """
    from sedtrails.application_interfaces.api import run_simulation

    try:
        typer.echo(f"Starting simulation from '{config_file}'...")
        output_dir = run_simulation(
            config_file=config_file,
            verbose=True,
            report_domain_exits=report_domain_exits,
        )
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

    Parameters
    ----------
    results_file : str
        Path to the SedTRAILS NetCDF results file.
    populations : bool
        Particle populations to process.
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

    Parameters
    ----------
    config_file : str
        Path to the SedTRAILS configuration file.

    Returns
    -------
    dict
        Dictionary containing the requested values.
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

    Parameters
    ----------
    output_file : str
        Path where output is written.
    """
    from sedtrails.application_interfaces.api import create_config_template

    try:
        create_config_template(output_file)
        typer.echo(f"Configuration template created at '{output_file}'")
    except Exception as e:
        typer.echo(f'Error creating configuration template: {e}')
        raise typer.Exit(code=1) from e


@config_app.command('gui')
def seeding_gui_cmd(
    config_file: str = typer.Option(
        'sedtrails.yml',
        '--config',
        '-c',
        help='Path to the source SedTRAILS configuration file.',
    ),
    output_file: str | None = typer.Option(
        None,
        '--output',
        '-o',
        help='Path to the copied configuration file to write from the GUI. Defaults beside --config.',
    ),
    points_output_path: str | None = typer.Option(
        None,
        '--points-output',
        help='Path to the generated x/y seed-point text file. Defaults next to --output.',
    ),
    population: str | None = typer.Option(
        None,
        '--population',
        '-p',
        help='Population name to update. Defaults to the first configured population.',
    ),
    input_format: str | None = typer.Option(
        None,
        '--format',
        help="Override general.input_model.format. Supported GUI formats: 'fm_netcdf' and 'xbeach'.",
    ),
    variable: str | None = typer.Option(
        None,
        '--variable',
        help="Bathymetry variable to display. Defaults to 'bedlevel' or 'bed_level'.",
    ),
):
    """
    Open a small GUI for choosing seed points and writing a copied config.

    Parameters
    ----------
    config_file : str
        Path to the SedTRAILS configuration file.
    output_file : str | None
        Path where output is written.
    points_output_path : str | None
        Path where selected seed points are written.
    population : str | None
        Particle population to process.
    input_format : str | None
        Input model format identifier.
    variable : str | None
        Name of the variable to inspect or sample.
    """
    from sedtrails.application_interfaces.seeding_gui import SeedingGuiError, launch_seeding_gui

    try:
        launch_seeding_gui(
            config_path=config_file,
            output_path=output_file,
            points_output_path=points_output_path,
            population_name=population,
            format_override=input_format,
            variable=variable,
        )
    except SeedingGuiError as e:
        typer.echo(f'Error opening seeding GUI: {e}')
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f'Unexpected error opening seeding GUI: {e}')


@config_app.command('restart')
def create_restart_config_cmd(
    results_file: str = typer.Option(
        'sedtrails_results.nc',
        '--file',
        '-f',
        help='Path to the SedTRAILS netCDF results file.',
    ),
    base_config_file: str = typer.Option(
        'sedtrails.yml',
        '--config',
        '-c',
        help='Path to the original SedTRAILS configuration file.',
    ),
    output_config_file: str = typer.Option(
        'sedtrails-restart.yaml',
        '--output',
        '-o',
        help='Path to write the generated restart YAML file.',
    ),
    seed_points_dir: str = typer.Option(
        None,
        '--seed-dir',
        help='Optional directory for generated restart seed point files.',
    ),
):
    """
    Create a restart YAML and point files from a partial/full NetCDF output.

    Parameters
    ----------
    results_file : str
        Path to the SedTRAILS NetCDF results file.
    base_config_file : str
        Path to the base configuration file.
    output_config_file : str
        Path where the restart configuration file is written.
    seed_points_dir : str
        Directory containing restart seed-point files.
    """
    from sedtrails.application_interfaces.api import create_restart_config

    try:
        summary = create_restart_config(
            results_file=results_file,
            base_config_file=base_config_file,
            output_config_file=output_config_file,
            seed_points_dir=seed_points_dir,
        )
        typer.echo(f"Restart config written to '{summary.output_config}'")
        typer.echo(f"Restart time set to '{summary.restart_time}'")
        typer.echo(f"Retained particles: {summary.retained_particles}")
        typer.echo('Generated seed point files:')
        for population_name, path in summary.seed_files.items():
            typer.echo(f'  - {population_name}: {path}')
    except Exception as e:
        typer.echo(f'Error creating restart config: {e}')
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
    help='Commands to compile and analyze connectivity networks.',
    context_settings={'help_option_names': ['-h', '--help']},
)
app.add_typer(network_app, name='network')


def _compile_connectivity_adjacency_cli(
    input_file: Path = typer.Option(
        'sedtrails_results.nc',
        '--input',
        '-i',
        help='Input SedTRAILS trajectory NetCDF file.',
    ),
    output_file: Path = typer.Option(
        'sedtrails_connectivity_adjacency.nc',
        '--output',
        '-o',
        help='Output NetCDF file for the connectivity adjacency matrix.',
    ),
    mode: str = typer.Option(
        'all',
        '--mode',
        help="Adjacency mode: 'all', 'final', or 'time'.",
    ),
    polygon_mode: str = typer.Option(
        'per_source',
        '--polygon-mode',
        help="Connectivity polygon mode: 'per_source' or 'n_cells'.",
    ),
    n_cells: int | None = typer.Option(
        None,
        '--n-cells',
        help="Number of cells when --polygon-mode is 'n_cells'.",
    ),
    group_sources: bool = typer.Option(
        True,
        '--group-sources/--no-group-sources',
        help='Group particles with identical initial positions into one source node.',
    ),
    source_group_tolerance: float = typer.Option(
        0.0,
        '--source-group-tolerance',
        help='Coordinate tolerance for grouping initial source positions.',
    ),
    count_repeated_visits: bool = typer.Option(
        True,
        '--count-repeated-visits/--unique-visits',
        help='Count repeated particle-position visits, or count each particle once per sink.',
    ),
    weight: str = typer.Option(
        'raw_counts',
        '--weight',
        help="Weighting mode: 'raw_counts', 'probability', or 'representative_volume'.",
    ),
    include_self_links: bool = typer.Option(
        True,
        '--include-self-links/--zero-self-links',
        help='Include or zero source-to-same-sink links.',
    ),
):
    """
    Compile a connectivity adjacency matrix from SedTRAILS particle tracks.

    Parameters
    ----------
      input_file : Path
         Path to the input NetCDF file containing particle tracking results.
      output_file : Path
         Path to the output NetCDF file containing connectivity adjacency results.
    """
    from sedtrails.application_interfaces.api import compile_connectivity_adjacency

    try:
        typer.echo(f"Compiling connectivity adjacency from '{input_file}'...")
        summary = compile_connectivity_adjacency(
            input_file=str(input_file),
            output_file=str(output_file),
            mode=mode,
            polygon_mode=polygon_mode,
            n_cells=n_cells,
            group_sources_by_initial_position=group_sources,
            source_group_tolerance=source_group_tolerance,
            count_repeated_visits=count_repeated_visits,
            weight=weight,
            include_self_links=include_self_links,
        )
        typer.echo(f"Adjacency matrix written to '{summary.output_file}'.")
        typer.echo(f'Particles: {summary.n_particles}')
        typer.echo(f'Connectivity nodes: {summary.n_nodes}')
        typer.echo(f'Mode: {summary.mode}; weight: {summary.weight}')

    except Exception as e:
        typer.echo(f'Error compiling connectivity adjacency: {e}')
        raise typer.Exit(code=1) from e


network_app.command('adjacency')(_compile_connectivity_adjacency_cli)
network_app.command('analysis')(_compile_connectivity_adjacency_cli)


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

    Parameters
    ----------
    results_file : str
        Path to the SedTRAILS NetCDF results file.
    save_fig : bool
        The save fig value.
    output_dir : str
        Directory where output files are written.
    """
    from sedtrails.application_interfaces.api import plot_trajectories

    try:
        plot_trajectories(results_file, save=save_fig, output_dir=output_dir)
        if save_fig:
            typer.echo(f"Plot saved to '{output_dir}/particle_trajectories.png'")
        else:
            typer.echo('Plot displayed successfully')
    except Exception as e:
        typer.echo(f'Error plotting trajectories: {e}')
        raise typer.Exit(code=1) from e


if __name__ == '__main__':
    app()
