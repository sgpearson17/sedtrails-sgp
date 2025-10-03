# Simulations

::: warning
Explain how a user condigures and runs a simulation
Outputs are explain ina separate section
:::

## Configuring a Simulation

::: Provide a how to guide for setting up a simulation. Use the current example to briefly explain the structure of a configuraiton file. Refrence the simulation parameters section in the apendices 
:::

This is how you refer to an appendix: [appendix](#apendix-parameters)

## Running a Simulation

The  following steps will guide you through running a simple SedTRAILS simulation using the example configuration file. Configuration files are files describing the parameters and settings for running the SedTRAILS simulations.

:::important
Make sure you have SedTRAILS installed. If you haven't installed it yet, please refer to the [Installation Guide](./installation.md).

<a href="../_static/downloads/config-example.yaml" download>Download the example configuration file</a> to your computer and save it in a directory where you want to run the simulation. For example `./examples/config-example.yaml`.
:::

1. Download the dataset file named `inlet_sedtrails.n` from [this link](https://surfdrive.surf.nl/files/index.php/s/VUGKZm7QexAXuD9?path=%2Fdfm), and save it to your directory.


2. Update the `input_data` parameter in configuration file and save the changes. You can use any text-editor to open and update the file. This parameter must pint to the location of the `inlet_sedtrails.nz` dataset you downloaded earlier.
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

