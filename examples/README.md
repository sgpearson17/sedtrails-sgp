# SedTRAILS Examples

This directory contains tracked example inputs and scripts for local testing and exploration.

## Simulation Configurations

- `sedtrails-example.yaml`: Basic D-Flow FM/van Westen simulation using randomly seeded sand populations.
- `sedtrails-example-multisource.yaml`: D-Flow FM/van Westen simulation using seed points from `sources_xy_inlet.txt`.
- `config.example_soulsby.yaml`: D-Flow FM/Soulsby simulation with two sand populations.
- `config.example_sfincs.yaml`: SFINCS/passive-tracer simulation.

The NetCDF forcing files referenced by these examples are not stored in this directory. Download or create the required files locally and update each configuration's `inputs.data` path before running.

## Python Examples

- `reading_flowfields_example.py`: Opens and inspects flow-field input data.
- `seeder_example.py`: Demonstrates particle seeding helpers.
- `particle_simulation_example.py`: Basic particle simulation script.
- `particle_simulation_numba_example.py`: Particle simulation script using the Numba path.
- `numba_example.py`: Small Numba-oriented example.
- `visualization_utils.py`: Visualization helper example code.

## Data Files

- `sources_xy_inlet.txt`: Example x/y release points for `sedtrails-example-multisource.yaml`.
- `log_example.txt`: Example simulation log output.
