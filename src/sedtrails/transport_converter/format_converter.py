"""
Format Converter: converts input data formats into SedtrailsData format.

This module reads various input data formats (e.g., NetCDF files from different
hydrodynamic models) and converts them into the SedtrailsData structure for
use in the SedTRAILS particle tracking system.
"""

import numpy as np
import xarray as xr
import xugrid as xu
from typing import Union, Dict
from sedtrails.transport_converter.sedtrails_data import SedtrailsData


class FormatConverter:
    """
    A class to convert various input data formats to the SedtrailsData format.

    This class provides methods to read data from different file formats and
    convert them to the SedtrailsData format for use in the SedTRAILS particle
    tracking system.
    """

    def __init__(self, config: Dict):
        """
        Initialize the FormatConverter.

        Parameters:
        -----------
        config : dict
            Configuration dictionary containing settings for the converter.
            Must include 'input_file', 'input_format', optionally 'reference_date' (default
            "1970-01-01" (Unix epoch))
        """
        self.config = config
        self._reference_date: Union[str, None] = None
        self.input_data = None
        self._format_plugin = None
        self._input_format: Union[str, None] = None
        self._input_file: Union[str, None] = None

    def __post_init__(self):
        """
        Config validation and initialization.
        """

        if not isinstance(self.config, dict):
            raise TypeError(f'Config must be a dictionary, got {type(self.config)}')

    @property
    def input_file(self):
        """Get the input file path."""
        if self._input_file is None:
            self._input_file = self.config.get('input_file')
            if not self._input_file:
                raise ValueError('Input file path must be provided in the configuration')
        return self._input_file

    @property
    def input_format(self) -> str | None:
        """Get the format to convert to."""
        if self._input_format is None:
            self._input_format = self.config.get('input_format')
            if not self._input_format:
                raise ValueError('Input format must be specified in the configuration')
        return self._input_format

    @property
    def reference_date(self) -> str:
        """Get the reference date as a numpy datetime64 object."""

        if self._reference_date is None:
            self._reference_date = self.config.get('reference_date', '1970-01-01')
        return self.reference_date  # Default to Unix epoch

    @property
    def format_plugin(self):
        """
        Get the format plugin instance based on the specified format.
        """

        import importlib  # lazy import for performance

        if self._format_plugin is None:
            # Dynamically import the format plugin based on the input type
            plugin_module_name = f'sedtrails.transport_converter.plugins.format.{self.input_format}'
            try:
                plugin_module = importlib.import_module(plugin_module_name)
            except ImportError as e:
                raise ImportError(
                    f'Failed to import format plugin module: {plugin_module_name} '
                    f'Ensure the module exists and is correctly named.'
                ) from e
            else:
                # Initialize the format plugin with the input file and type
                self._format_plugin = plugin_module.FormatPlugin(self.input_file)

        return self._format_plugin

    def read_data(self) -> None:
        """
        Read the input data based on the input type.
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f'Input file not found: {self.input_file}')

        if self.input_type == InputType.NETCDF_DFM:
            self._read_netcdf_dfm()
        elif self.input_type == InputType.TRIM_D3D4:
            # Placeholder for future implementation
            raise NotImplementedError('TRIM_D3D4 format not implemented yet')
        else:
            raise NotImplementedError(f'Input type not implemented: {self.input_type}')

    def _read_netcdf_dfm(self) -> None:
        """
        Read a Delft3D Flexible Mesh NetCDF file using xugrid.
        """
        try:
            # First try using xugrid's open_dataset which handles UGRID conventions
            self.input_data = xu.open_dataset(self.input_file, decode_timedelta=True)
        except Exception as e:
            print(f'Could not open file with xugrid: {e}')
            # Fallback to regular xarray
            try:
                self.input_data = xr.open_dataset(self.input_file, decode_timedelta=True)
            except Exception as e:
                raise IOError(f'Failed to open NetCDF file: {e}') from e

        print(f'Successfully loaded {self.input_file}')
        print(f'Variables in dataset: {list(self.input_data.data_vars)}')

    def get_time_info(self) -> Dict:
        """
        Get time information from the dataset.

        Returns:
        --------
        Dict
            Dictionary containing time values, start time, end time,
            and time in seconds since reference date
        """
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call read_data() first.')

        # Get the time variable
        time_var = self.input_data['time']
        time_values = time_var.values
        time_start = time_values[0]
        time_end = time_values[-1]

        # Get original time units and calendar from the attributes
        orig_units = getattr(time_var, 'units', None)
        orig_calendar = getattr(time_var, 'calendar', 'standard')

        # Convert time values to seconds since reference_date
        seconds_since_ref = np.array([float((t - self.reference_date) / np.timedelta64(1, 's')) for t in time_values])

        return {
            'time_values': time_values,
            'time_start': time_start,
            'time_end': time_end,
            'original_units': orig_units,
            'original_calendar': orig_calendar,
            'seconds_since_reference': seconds_since_ref,
            'reference_date': self.reference_date,
            'num_times': len(time_values),
        }

    def _map_dfm_variables(self) -> Dict:
        """
        Map Delft3D Flexible Mesh variables to SedtrailsData structure.

        This function processes all time steps at once.

        Returns:
        --------
        Dict
            Dictionary with mapped variables
        """
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call read_data() first.')

        # Get time information
        time_info = self.get_time_info()
        num_times = time_info['num_times']

        # Variable mapping for DFM files
        variable_map = {
            'x': 'net_xcc',  # X-coordinates
            'y': 'net_ycc',  # Y-coordinates
            'bed_level': 'bedlevel',  # Bed level
            'water_depth': 'waterdepth',  # Water depth
            'flow_velocity_x': 'sea_water_x_velocity',  # X-component of flow velocity
            'flow_velocity_y': 'sea_water_y_velocity',  # Y-component of flow velocity
            'mean_bed_shear_stress': 'mean_bss_magnitude',  # Mean bed shear stress
            'max_bed_shear_stress': 'max_bss_magnitude',  # Max bed shear stress
            'bed_load_transport_x': 'bedload_x_comp',  # X-component of bed load sediment transport
            'bed_load_transport_y': 'bedload_y_comp',  # Y-component of bed load sediment transport
            'suspended_transport_x': 'susload_x_comp',  # X-component of suspended sediment transport
            'suspended_transport_y': 'susload_y_comp',  # Y-component of suspended sediment transport
            'sediment_concentration': 'suspended_sed_conc',  # Suspended sediment concentration
        }

        # Extract data from dataset
        data = {}

        # First, get spatial coordinates (typically not time-dependent)
        for key in ['x', 'y']:
            var_name = variable_map[key]
            if var_name in self.input_data:
                data[key] = self.input_data[var_name].values
            else:
                raise KeyError(f"Required variable '{var_name}' not found in dataset")

        # Determine the spatial grid dimensions
        grid_shape = data['x'].shape

        # Get bed level (typically not time-dependent)
        var_name = variable_map['bed_level']
        if var_name in self.input_data:
            bed_level_var = self.input_data[var_name]
            if 'time' in bed_level_var.dims:
                # If bed level has a time dimension, take the first time step
                data['bed_level'] = bed_level_var.isel(time=0).values
            else:
                data['bed_level'] = bed_level_var.values
        else:
            # Default to zeros if not found
            data['bed_level'] = np.zeros(grid_shape)
            print(f"Warning: Variable '{var_name}' not found, using zeros")

        # Extract time-dependent variables
        time_dependent_vars = [
            'water_depth',
            'mean_bed_shear_stress',
            'max_bed_shear_stress',
            'sediment_concentration',
            'flow_velocity_x',
            'flow_velocity_y',
            'bed_load_transport_x',
            'bed_load_transport_y',
            'suspended_transport_x',
            'suspended_transport_y',
        ]

        for key in time_dependent_vars:
            var_name = variable_map[key]
            if var_name in self.input_data:
                var = self.input_data[var_name]

                # Check if variable has time dimension
                if 'time' in var.dims:
                    # Check if variable has layer dimension
                    if 'layer' in var.dims:
                        # For variables with time and layer, select layer 0
                        data[key] = var.isel(layer=0).values
                    else:
                        # For variables with time but no layer
                        data[key] = var.values
                else:
                    # For variables without time dimension, broadcast to all time steps
                    data[key] = np.broadcast_to(var.values, (num_times, *var.shape))
            else:
                # Default to zeros if not found
                data[key] = np.zeros((num_times, *grid_shape))
                print(f"Warning: Variable '{var_name}' not found, using zeros")

        return data

    def convert_to_sedtrails_data(self) -> SedtrailsData:
        """
        Converts dataset to SedtrailsData format for all time steps.

        Returns:
        --------
        SedtrailsData
            Data in SedtrailsData format with time as the first dimension for
            time-dependent variables, with time in seconds since reference_date
        """

        if self._format_plugin is None:
            plugin = self.format_plugin
        else:
            plugin = self._format_plugin

        print(f'Using {plugin.__class__.__name__} to convert data to SedtrailsData format...')

        sedtrails_data = plugin.convert()
        print('Successfully converted data to SedtrailsData format.')
        return sedtrails_data


if __name__ == '__main__':
    print('Please see the examples directory for usage examples.')

    conf = {
        'input_file': 'sedtrails/sample-data/inlet_sedtrails.nc',
        'input_format': 'fm_netcdf',
        'reference_date': '1970-01-01',
    }

    converter = FormatConverter(conf)
    converter.convert_to_sedtrails_data()
