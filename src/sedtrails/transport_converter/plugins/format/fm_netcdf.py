"""A plugin for converting Delft3D Flexible Mesh NetCDF to SedTRAILS format."""

import xugrid as xu
import xarray as xr
import numpy as np
from sedtrails.transport_converter.plugins import BaseFormatPlugin
from sedtrails.transport_converter.sedtrails_data import SedtrailsData
from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata
from pathlib import Path
from typing import Dict, Any, List, Union


class FormatPlugin(BaseFormatPlugin):
    """
    Plugin for converting Delft3D Flexible Mesh NetCDF to SedTRAILS format.
    """

    def __init__(self, input_file: str):
        """
        Initialize the plugin with the input file.

        Parameters:
        -----------
        input_file : str
            Path to the Delft3D Flexible Mesh NetCDF file.
        """
        super().__init__()
        self.input_file = Path(input_file)
        self.input_data = None  # holds Dataset after reading
        self._input_variables: List[str] = []

    def __post_init__(self):
        # Check if the input file exists
        if not self.input_file.exists():
            raise FileNotFoundError(f'Input file not found: {self.input_file}')

    @property
    def variables(self) -> List[str]:
        """
        Get the variables in the input dataset.

        Returns:
        --------
        List
            List of variable names in the input dataset.
        """

        if self.input_data is None:  # Ensure input_data is loaded
            self.load()

        if not self._input_variables:  # Ensures variables are only loaded once
            try:
                self._input_variables = list(self.input_data.data_vars)
            except AttributeError as e:
                raise ValueError('Input data could not retrieve variables') from e
            else:
                print(f'Variables in {self.input_file}:')

        return self._input_variables

    def convert(self) -> SedtrailsData:
        """
        Delft3D from Flexible Mesh NetCDF.

        Returns:
        --------
        SedtrailsData
            The converted SedtrailsData object.
        """

        # Read the NetCDF file
        self.load()

        time_info = self._get_time_info(self.input_data, reference_date=np.datetime64('1970-01-01T00:00:00'))

        # Map the variables to SedtrailsData structure
        mapped_data = self._map_dfm_variables(time_info)
        seconds_since_ref = time_info['seconds_since_reference']
        self.reference_date = time_info['reference_date']

        # Calculate magnitudes for vector quantities
        # Flow velocity magnitude
        depth_avg_velocity_magnitude = np.sqrt(
            mapped_data['flow_velocity_x'] ** 2 + mapped_data['flow_velocity_y'] ** 2
        )

        # Bed load magnitude
        bed_load_magnitude = np.sqrt(
            mapped_data['bed_load_transport_x'] ** 2 + mapped_data['bed_load_transport_y'] ** 2
        )

        # Suspended sediment magnitude
        suspended_transport_magnitude = np.sqrt(
            mapped_data['suspended_transport_x'] ** 2 + mapped_data['suspended_transport_y'] ** 2
        )

        # Create dictionaries for vector quantities
        depth_avg_flow_velocity = {
            'x': mapped_data['flow_velocity_x'],
            'y': mapped_data['flow_velocity_y'],
            'magnitude': depth_avg_velocity_magnitude,
        }

        bed_load_transport = {
            'x': mapped_data['bed_load_transport_x'],
            'y': mapped_data['bed_load_transport_y'],
            'magnitude': bed_load_magnitude,
        }

        suspended_transport = {
            'x': mapped_data['suspended_transport_x'],
            'y': mapped_data['suspended_transport_y'],
            'magnitude': suspended_transport_magnitude,
        }

        # Create nonlinear wave velocity dictionary with zeros
        # Using the same shape as other vector quantities
        nonlinear_wave_velocity = {
            'x': np.zeros_like(mapped_data['flow_velocity_x']),
            'y': np.zeros_like(mapped_data['flow_velocity_y']),
            'magnitude': np.zeros_like(depth_avg_velocity_magnitude),
        }

        # Create SedtrailsMetadata object
        metadata = SedtrailsMetadata(
            flowfield_domain={
                'x_min': np.min(mapped_data['x']),
                'x_max': np.max(mapped_data['x']),
                'y_min': np.min(mapped_data['y']),
                'y_max': np.max(mapped_data['y']),
            }
        )

        # Create SedtrailsData object
        sedtrails_data = SedtrailsData(
            times=seconds_since_ref,
            reference_date=self.reference_date,
            x=mapped_data['x'],
            y=mapped_data['y'],
            bed_level=mapped_data['bed_level'],
            depth_avg_flow_velocity=depth_avg_flow_velocity,
            fractions=1,  # Default to 1 fraction
            bed_load_transport=bed_load_transport,
            suspended_transport=suspended_transport,
            water_depth=mapped_data['water_depth'],
            mean_bed_shear_stress=mapped_data['mean_bed_shear_stress'],
            max_bed_shear_stress=mapped_data['max_bed_shear_stress'],
            sediment_concentration=mapped_data['sediment_concentration'],
            nonlinear_wave_velocity=nonlinear_wave_velocity,
            metadata=metadata,
        )

        return sedtrails_data

    def load(self) -> Any:
        """
        Reads and loads a Delft3D Flexible Mesh NetCDF file using xugrid.
        """

        if self.input_data is None:
            try:
                # First try using xugrid's open_dataset which handles UGRID conventions
                self.input_data = xu.open_dataset(self.input_file, decode_timedelta=True)
            except Exception as e:
                print(f'Could not open file with xugrid: {e} \n Trying with Xarray...')
                # Fallback to regular xarray
                try:
                    self.input_data = xr.open_dataset(self.input_file, decode_timedelta=True)
                except Exception as e:
                    raise IOError(f'Failed to open NetCDF file: {e}') from e

                else:
                    print(f'Sucessfully loaded (Xarray): {self.input_file}')
            else:
                print('Successfully loaded (Xugrid)', self.input_file)

    def _get_time_info(self, input_data: Union[xu.UgridDataset, xr.Dataset], reference_date: np.datetime64) -> Dict:
        """
        Get and transforms time information of a dataset.

        Parameters:
        -----------
        input_data : xu.UgridDataset
            The input dataset containing time information.

        reference_data : np.datetime64
            The reference date to calculate time in seconds.

        Returns:
        --------
        Dict
            Dictionary containing time values, start time, end time,
            and time in seconds since reference date
        """

        # check reference_date is a numpy datetime64
        if not isinstance(reference_date, np.datetime64):
            raise TypeError('reference_date must be a numpy datetime64 object')

        if input_data is None:
            raise ValueError('Dataset not loaded. Call read_netcdf_dfm() first.')

        # Get the time variable
        time_var = input_data['time']
        time_values = time_var.values
        time_start = time_values[0]
        time_end = time_values[-1]

        # Get original time units and calendar from the attributes
        orig_units = getattr(time_var, 'units', None)
        orig_calendar = getattr(time_var, 'calendar', 'standard')

        # Convert time values to seconds since reference_date
        seconds_since_ref = np.array([float((t - reference_date) / np.timedelta64(1, 's')) for t in time_values])

        return {
            'time_values': time_values,
            'time_start': time_start,
            'time_end': time_end,
            'original_units': orig_units,
            'original_calendar': orig_calendar,
            'seconds_since_reference': seconds_since_ref,
            'reference_date': reference_date,
            'num_times': len(time_values),
        }

    def _map_dfm_variables(self, time_info) -> Dict:
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


if __name__ == '__main__':
    # Example usage
    input_file = '/sedtrails/sample-data/inlet_sedtrails.nc'
    plugin = FormatPlugin(input_file)

    plugin.load()

    print(plugin.variables)  # List available variables

    data = plugin.convert()  # Convert to SedtrailsData format
    print(data)
