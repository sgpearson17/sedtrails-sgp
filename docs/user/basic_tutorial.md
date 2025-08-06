# Basic Tutorial
Basic tutorial about how to run SedTRAILS using the command line, after SedTRAILS has been installed.

Help:
sedtrails =h

Usage: sedtrails [OPTIONS] COMMAND [ARGS]...

 Sedtrails: Configure, run, and analyze sediment particle tracking.

Options 
--version  -v        Show version and exit.                                                                                                                                                                                             
--help     -h        Show this message and exit.                                                                                                                                                                                           

Commands 
run                Run a simulation based on a configuration file. The simulation results are written to a netCDF file.
network-analysis   Perform a network analysis on the simulation results. The network analysis results are written to a netCDF file.
config             Configuration file management commands.       

sedtrails run -h

 Usage: sedtrails run [OPTIONS]

 Run a simulation based on a configuration file. The simulation results are written to a netCDF file.

 Example: sedtrails run --config my_config.yml --output results.nc

Options 
--config  -c      TEXT  Path to the SedTRAILS configuration file. [default: sedtrails.yml]
--output  -o      TEXT  Path to the output SedTRAILS netCDF file. [default: sedtrails.nc]
--help    -h            Show this message and exit.             


sedtrails run --config examples/config.example_002.yaml

within your config file, make sure that your filepaths are correct:
folder_settings:
  input_data: C:/sample-data/inlet_sedtrails.nc
  output_dir: C:/sample-data/output