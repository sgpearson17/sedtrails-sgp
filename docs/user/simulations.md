# Simulations

::: warning
Explain how a user configures and runs a simulation
Outputs are explain ina separate section
:::

## Configuring a Simulation

Simulations are configured using YAML files. These files define the parameters and settings for running SedTRAILS simulations, including input data, particle properties, and output options.

For a detailed reference of all available parameters, please refer to the [Simulation Parameters Reference](../references/simulation-params.md).


### Example Configuration File

```yaml

general:
  input_model: 
    format: fm_netcdf
    reference_date: 1970-01-01  # Default reference date for the input model
    morfac: 1  # Morphological acceleration factor for time decompression
inputs:
  data: ./sample-data/inlet_sedtrails.nc
  read_interval: 10D  # Time chunk size for reading input data
time:
  start:  2016-09-21 19:20:00
  timestep: 60S
  duration: 1D
  cfl_condition: 0.7  # CFL condition for adaptive timestep (0 = disabled)
particles:
  populations:
    - name: populaton_1
      particle_type: sand
      characteristics:
        grain_size: 0.00025 
        density: 2650.0  
      tracer_methods:
        vanwesten:
          flow_field_name: 
            - bed_load_velocity
            - suspended_velocity
      transport_probability: stochastic_transport  # Options: no_probability, stochastic_transport, reduced_velocity
      seeding:
        burial_depth: 
          constant: 0
        release_start: 2016-09-21 19:30:00
        quantity: 1
        strategy: 
          random:
            bbox: "39400,16800 40600,17800"
            seed: 42
            nlocations: 10
    - name: population_2
      particle_type: sand
      characteristics:
        grain_size: 0.00035 
        density: 2650.0  
      tracer_methods:
        vanwesten:
          flow_field_name: 
            - bed_load_velocity
            - suspended_velocity
      transport_probability: stochastic_transport  # Options: no_probability, stochastic_transport, reduced_velocity
      seeding:
        burial_depth: 
          constant: 0
        release_start: 2016-09-21 19:30:00
        quantity: 1
        strategy: 
          random:
            bbox: "39400,16800 40600,17800"
            seed: 42
            nlocations: 5
outputs:
  directory: ./results
  store_tracks: true
  save_interval: 1H
visualization:
  dashboard:
    enable: true
    update_interval: 1H
```

## Running a Simulation

The  following steps will guide you through running a simple SedTRAILS simulation using the example configuration file. Configuration files are files describing the parameters and settings for running the SedTRAILS simulations.

:::important
Make sure you have SedTRAILS installed. If you haven't installed it yet, please refer to the [Installation Guide](./installation.md).

<a href="../_static/downloads/config-example.yaml" download>Download the example configuration file</a> to your computer and save it in a directory where you want to run the simulation. For example `./examples/config-example.yaml`.
:::

1. Download the dataset file named `inlet_sedtrails.nc` from [this link](https://surfdrive.surf.nl/files/index.php/s/VUGKZm7QexAXuD9?path=%2Fdfm), and save it to your directory.


2. Update the `input_data` parameter in configuration file and save the changes. You can use any text-editor to open and update the file. This parameter must pint to the location of the `inlet_sedtrails.nc` dataset you downloaded earlier.
For example: 
```yaml
input_data: ./inlet_sedtrails.nc
```

1. Using the terminal, go to the directory containing the dataset and configuration files:
```bash
cd ./<path-to-you-simulation-directory>/
```

4. Run the model using the following command:
```bash
sedtrails run -c ./config-example.yaml
```

::: note
The simulation will start running, and a dashboard will open to show the progress. Close the dashboard window to get back to the terminal and see the simulation results.
:::

## Input Models

SedTRAILS computes particle pathways based on pre-existing hydrodynamic and/or sediment transport model output. At present we support output from several models:
1. D-Flow FM
2. XBeach
3. SFINCS
4. Delft3D-4

In the sections below we provide information about the output settings required in these models in order to be used in SedTRAILS.

### 1. D-Flow FM
The map output from D-Flow FM is written to binary `*.nc` files which can be directly read by SedTRAILS using the `fm_netcdf.py` format converter plugin.

### 2. XBeach
The map output from D-Flow FM is written to binary `*.nc` files which can  be directly read by SedTRAILS using the `xbeach.py` format converter plugin. 

::: note
NOTE: JOHAN- MERGE BRANCH, IT IS NOT IN DEV
:::

### 3. SFINCS
The map output from D-Flow FM is written to binary `*.nc` files which can  be directlyread by SedTRAILS using the `sfincs.py` format converter plugin.

### 4. Delft3D-4
By default, the map output from Delft3D-4 is written to binary `trim-*.dat` files. At present it is not possible to directly read these files in with Python, so it is better to instead write the map output as `*.nc` files. To enable `*.nc` output in Delft3D-4, add the following lines to the `*.mdf` file:
```
FlNcdf= #maphis#
ncFormat=4
```
The resulting  `*.nc` file can then be directly read by SedTRAILS using the `delft3d4_netcdf.py` format converter plugin.

Note: the format converter interpolates data at all U/V points to cell centers.

### Other models
To add other models as input, you need to construct a format converter plugin as per the
[Plugin Guidelines](../docs/developer/plugins.md).
