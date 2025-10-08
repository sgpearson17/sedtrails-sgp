# SedTRAILS Plugins

SedTRAILS supports a plugin system that allows users to extend its functionality by 
writing plugins for the *Transport Converter*. 
Plugins can be writen for the Format and Physics Converters. 

This document provides an overview of how to create and integrate plugins into SedTRAILS. We use the case of creating a plugin for the Physics Converter as an example.

## Creating a Plugin

We assume you have set up a development environment for SedTRAILS. If you haven't done this yet, please refer to the [Developer Guide](./developer/dev-environment.md).
In essence, a plugin for the  Physics Converter must implement a class that inherits from `BasePhysicsPlugin` and implements the `add_physics` method.
The `add_physics` method takes as input a `SedtrailsData` object, which contains data to perform the physics calculations, and add the results (physics conversiont) to the `SedtrailsData` object itself.
To **create a physics plugin for SedTRAILS**, follow these steps:

1. Create a new Python file for your plugin in `src/sedtrails/transport_converter/plugins/physics/`. For example, `myplugin.py`.

2. Define a class for your plugin that inherits from the appropriate base class. For a Physics Converter plugin, inherit from `BasePhysicsPlugin`.

    ```python
    # myplugin.py
    from sedtrails.transport_converter.plugins import BasePhysicsPlugin
    from sedtrails.transport_converter import SedtrailsData # Import SedtrailsData 

    class PhysicsPlugin(BasePhysicsPlugin):
    """
    Plugin for sediment transport physics calculations.
    This plugin implements the physics calculations as described in XXXX et al. (2023).
    """

    def __init__(
        self,
        config
    ):  
        super().__init__()
        self.config = config

    def add_physics(
        self,
        sedtrails_data: SedtrailsData,
    ):
        """
        Add physics using XX approach.
        """

        print('Using XX compute transport velocities and add to SedTRAILS data...')

        ## Your code goes here
    ```





## Registering a Plugin

