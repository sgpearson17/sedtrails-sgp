:py:mod:`sedtrails.configuration_interface.cli.main`
====================================================

.. py:module:: sedtrails.configuration_interface.cli.main

.. autodoc2-docstring:: sedtrails.configuration_interface.cli.main
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`version_callback <sedtrails.configuration_interface.cli.main.version_callback>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.version_callback
          :summary:
   * - :py:obj:`main <sedtrails.configuration_interface.cli.main.main>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.main
          :summary:
   * - :py:obj:`run_simulation <sedtrails.configuration_interface.cli.main.run_simulation>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.run_simulation
          :summary:
   * - :py:obj:`inspect_metadata <sedtrails.configuration_interface.cli.main.inspect_metadata>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.inspect_metadata
          :summary:
   * - :py:obj:`load_config <sedtrails.configuration_interface.cli.main.load_config>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.load_config
          :summary:
   * - :py:obj:`create_config_template <sedtrails.configuration_interface.cli.main.create_config_template>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.create_config_template
          :summary:
   * - :py:obj:`analyze <sedtrails.configuration_interface.cli.main.analyze>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.analyze
          :summary:
   * - :py:obj:`analysis <sedtrails.configuration_interface.cli.main.analysis>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.analysis
          :summary:
   * - :py:obj:`plot_trajectories <sedtrails.configuration_interface.cli.main.plot_trajectories>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.plot_trajectories
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`app <sedtrails.configuration_interface.cli.main.app>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.app
          :summary:
   * - :py:obj:`config_app <sedtrails.configuration_interface.cli.main.config_app>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.config_app
          :summary:
   * - :py:obj:`analyzer_app <sedtrails.configuration_interface.cli.main.analyzer_app>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.analyzer_app
          :summary:
   * - :py:obj:`network_app <sedtrails.configuration_interface.cli.main.network_app>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.network_app
          :summary:
   * - :py:obj:`vizualizer_app <sedtrails.configuration_interface.cli.main.vizualizer_app>`
     - .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.vizualizer_app
          :summary:

API
~~~

.. py:function:: version_callback(value: bool)
   :canonical: sedtrails.configuration_interface.cli.main.version_callback

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.version_callback

.. py:data:: app
   :canonical: sedtrails.configuration_interface.cli.main.app
   :value: 'Typer(...)'

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.app

.. py:function:: main(version: bool = typer.Option(False, '--version', '-v', callback=version_callback, is_eager=True, help='Show version and exit.'))
   :canonical: sedtrails.configuration_interface.cli.main.main

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.main

.. py:function:: run_simulation(config_file: str = typer.Option('sedtrails.yml', '--config', '-c', help='Path to the SedTRAILS configuration file.'), output_file: str = typer.Option('sedtrails.nc', '--output', '-o', help='Path to the output SedTRAILS netCDF file.'))
   :canonical: sedtrails.configuration_interface.cli.main.run_simulation

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.run_simulation

.. py:function:: inspect_metadata(results_file: str = typer.Option('sedtrails_results.nc', '--file', '-f', help='Path to the SedTRAILS netCDF file to inspect. By default, it expects an "sedtrails_results.nc" file in the current directory.'), populations: bool = typer.Option(False, '--populations', '-p', help='Inspect and print detailed information about particle populations in the file.'))
   :canonical: sedtrails.configuration_interface.cli.main.inspect_metadata

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.inspect_metadata

.. py:data:: config_app
   :canonical: sedtrails.configuration_interface.cli.main.config_app
   :value: 'Typer(...)'

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.config_app

.. py:function:: load_config(config_file: str = typer.Option('sedtrails.yml', '--config', '-c', help='Path to the SedTRAILS configuration file.')) -> dict
   :canonical: sedtrails.configuration_interface.cli.main.load_config

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.load_config

.. py:function:: create_config_template(output_file: str = typer.Option('./sedtrails-template.yml', '--output', '-o', help='Path to the output configuration template file.'))
   :canonical: sedtrails.configuration_interface.cli.main.create_config_template

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.create_config_template

.. py:data:: analyzer_app
   :canonical: sedtrails.configuration_interface.cli.main.analyzer_app
   :value: 'Typer(...)'

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.analyzer_app

.. py:function:: analyze(input_file: pathlib.Path = typer.Option('sedtrails.nc', '--input', '-i', help='Input SedTRAILS netCDF file containing particle tracks.'), output_file: pathlib.Path = typer.Option('analysis.nc', '--output', '-o', help='Output SedTRAILS netCDF file containing statistical and connectivity results.'))
   :canonical: sedtrails.configuration_interface.cli.main.analyze

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.analyze

.. py:data:: network_app
   :canonical: sedtrails.configuration_interface.cli.main.network_app
   :value: 'Typer(...)'

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.network_app

.. py:function:: analysis(input_file: pathlib.Path = typer.Option('sedtrails.nc', '--input', '-i', help='Input netCDF file containing particle tracking results.'), output_file: pathlib.Path = typer.Option('analysis.nc', '--output', '-o', help='Path to the output SedTRAILS netCDF file containing statistical and connectivity results.'))
   :canonical: sedtrails.configuration_interface.cli.main.analysis

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.analysis

.. py:data:: vizualizer_app
   :canonical: sedtrails.configuration_interface.cli.main.vizualizer_app
   :value: 'Typer(...)'

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.vizualizer_app

.. py:function:: plot_trajectories(results_file: str = typer.Option('sedtrails_results.nc', '--file', '-f', help='Path to the SedTRAILS netCDF file to visualize. By default, it expects an "sedtrails_results.nc" file in the current directory.'), save_fig: bool = typer.Option(False, '--save', '-s', help='Save plot as a PNG file. Creates a "particle_trajectories.png" file'), output_dir: str = typer.Option('.', '--output-dir', '-o', help='Directory to save plot if --save is used. Default is the current directory.'))
   :canonical: sedtrails.configuration_interface.cli.main.plot_trajectories

   .. autodoc2-docstring:: sedtrails.configuration_interface.cli.main.plot_trajectories
