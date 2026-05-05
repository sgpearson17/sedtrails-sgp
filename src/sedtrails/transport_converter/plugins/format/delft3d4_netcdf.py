"""A plugin for converting Delft3D4 NetCDF format to SedTRAILS format.

By default, the map output from Delft3D-4 is written to binary `trim-*.dat` files.
At present it is not possible to directly read these files in with Python, so it is
better to instead write the map output as `*.nc` files. To enable `*.nc` output in
Delft3D-4, add the following lines to the `*.mdf` file:
```
FlNcdf= #maphis#
ncFormat=4
```
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from sedtrails.transport_converter.plugins import BaseFormatPlugin
from sedtrails.transport_converter.sedtrails_data import SedtrailsData
from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata


class FormatPlugin(BaseFormatPlugin):
    """Plugin for converting Delft3D4 NetCDF format to SedTRAILS format."""

    def __init__(self, input_file: str, morfac: float = 1.0):
        """
        Initialize the plugin with the input file.

        Parameters:
        -----------
        input_file : str
            Path to the Delft3D4 NetCDF file.
        morfac : float, optional
            Morphological acceleration factor for time decompression (default: 1.0)
        """
        super().__init__()
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f'Input file not found: {self.input_file}')
        self.morfac = morfac
        self.input_data: Optional[xr.Dataset] = None
        self._input_variables: List[str] = []

    @property
    def variables(self) -> List[str]:
        """
        Get the variables in the input dataset.

        Returns:
        --------
        List
            List of variable names in the input dataset.
        """

        if self.input_data is None:
            self.load()

        if not self._input_variables:
            try:
                self._input_variables = list(self.input_data.data_vars)
            except AttributeError as e:
                raise ValueError('Input data could not retrieve variables') from e
            else:
                print(f'Variables in {self.input_file}:')

        return self._input_variables

    def convert(
        self, current_time=None, reading_interval=None, reference_date: Optional[np.datetime64] = None
    ) -> SedtrailsData:
        """
        Delft3D4 NetCDF to SedtrailsData.

        Parameters:
        -----------
        current_time : float, optional
            Current simulation time in seconds
        reading_interval : float, optional
            Reading interval in seconds
        reference_date : np.datetime64, optional
            Reference date for converting time values

        Returns:
        --------
        SedtrailsData
            The converted SedtrailsData object.
        """

        if reference_date is None:
            reference_date = np.datetime64('1970-01-01T00:00:00')

        self.load()
        time_info = self._get_time_info(self.input_data, reference_date=reference_date)

        time_info = self._decompress_time(time_info)

        time_start_idx, time_end_idx = self._calculate_time_slice(current_time, reading_interval, time_info)

        if time_start_idx is not None or time_end_idx is not None:
            time_slice = slice(time_start_idx, time_end_idx)
            time_info = self._slice_time_info(time_info, time_slice)

        mapped_data = self._map_delft3d4_variables(time_info, time_start_idx, time_end_idx)
        seconds_since_ref = time_info['seconds_since_reference']
        self.reference_date = time_info['reference_date']

        depth_avg_velocity_magnitude = np.sqrt(
            mapped_data['flow_velocity_x'] ** 2 + mapped_data['flow_velocity_y'] ** 2
        )

        bed_load_magnitude = np.sqrt(
            mapped_data['bed_load_transport_x'] ** 2 + mapped_data['bed_load_transport_y'] ** 2
        )

        suspended_transport_magnitude = np.sqrt(
            mapped_data['suspended_transport_x'] ** 2 + mapped_data['suspended_transport_y'] ** 2
        )

        mean_bed_shear_stress = np.sqrt(
            mapped_data['bed_shear_stress_x'] ** 2 + mapped_data['bed_shear_stress_y'] ** 2
        )

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

        nonlinear_wave_velocity = {
            'x': np.zeros_like(mapped_data['flow_velocity_x']),
            'y': np.zeros_like(mapped_data['flow_velocity_y']),
            'magnitude': np.zeros_like(depth_avg_velocity_magnitude),
        }

        metadata = SedtrailsMetadata(
            flowfield_domain={
                'x_min': np.min(mapped_data['x']),
                'x_max': np.max(mapped_data['x']),
                'y_min': np.min(mapped_data['y']),
                'y_max': np.max(mapped_data['y']),
            }
        )

        return SedtrailsData(
            times=seconds_since_ref,
            reference_date=self.reference_date,
            x=mapped_data['x'],
            y=mapped_data['y'],
            bed_level=mapped_data['bed_level'],
            depth_avg_flow_velocity=depth_avg_flow_velocity,
            fractions=1,
            bed_load_transport=bed_load_transport,
            suspended_transport=suspended_transport,
            water_depth=mapped_data['water_depth'],
            mean_bed_shear_stress=mean_bed_shear_stress,
            max_bed_shear_stress=mapped_data['max_bed_shear_stress'],
            sediment_concentration=mapped_data['sediment_concentration'],
            nonlinear_wave_velocity=nonlinear_wave_velocity,
            metadata=metadata,
        )

    def get_seeding_coordinates(self):
        """Return spatial coordinates required for particle seeding."""
        self.load()

        if 'XZ' in self.input_data and 'YZ' in self.input_data:
            return self.input_data['XZ'].values, self.input_data['YZ'].values
        if 'XCOR' in self.input_data and 'YCOR' in self.input_data:
            return self.input_data['XCOR'].values, self.input_data['YCOR'].values

        raise KeyError("Required variables 'XZ'/'YZ' or 'XCOR'/'YCOR' not found in dataset")

    def load(self) -> Any:
        """Reads and loads a Delft3D4 NetCDF file using xarray."""

        if self.input_data is None:
            try:
                time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
                self.input_data = xr.open_dataset(
                    self.input_file,
                    decode_times=time_coder,
                    decode_timedelta=True,
                )
            except TypeError:
                self.input_data = xr.open_dataset(
                    self.input_file,
                    decode_times=True,
                    decode_timedelta=True,
                )
            except Exception as e:
                raise IOError(f'Failed to open NetCDF file: {e}') from e
            else:
                print(f'Successfully loaded (Xarray): {self.input_file}')

    def _get_variable(self, name: str) -> xr.DataArray:
        """Return an xarray DataArray for variable `name`."""
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')

        if name in self.input_data:
            return self.input_data[name]
        raise KeyError(f"Variable '{name}' not found in dataset")

    def _decompress_time(self, time_info: Dict) -> Dict:
        """Apply morfac decompression to time values."""
        if self.morfac == 1.0:
            return time_info

        decompressed_info = time_info.copy()
        time_start = time_info['time_start']
        decompressed_time_values = time_start + (time_info['time_values'] - time_start) * self.morfac

        decompressed_info['time_values'] = decompressed_time_values
        decompressed_info['time_start'] = decompressed_time_values[0]
        decompressed_info['time_end'] = decompressed_time_values[-1]
        decompressed_info['seconds_since_reference'] = np.array(
            [float((t - time_info['reference_date']) / np.timedelta64(1, 's')) for t in decompressed_time_values]
        )

        return decompressed_info

    def _get_time_info(self, input_data: xr.Dataset, reference_date: np.datetime64) -> Dict:
        """
        Get and transform time information of a dataset.
        """

        if not isinstance(reference_date, np.datetime64):
            raise TypeError('reference_date must be a numpy datetime64 object')

        if input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')

        time_var = input_data['time']
        time_values = time_var.values
        time_start = time_values[0]
        time_end = time_values[-1]

        orig_units = getattr(time_var, 'units', None)
        orig_calendar = getattr(time_var, 'calendar', 'standard')

        if not np.issubdtype(np.asarray(time_values).dtype, np.datetime64):
            decoded = None
            if orig_units is not None:
                try:
                    decoded = xr.coding.times.decode_cf_datetime(time_values, orig_units, orig_calendar)
                except Exception:
                    decoded = None
            if decoded is None:
                try:
                    decoded = xr.coding.times.cftime_to_nptime(time_values)
                except Exception:
                    decoded = None
            if decoded is not None:
                time_values = decoded
            else:
                time_values = reference_date + np.asarray(time_values, dtype=float) * np.timedelta64(1, 's')

        try:
            seconds_since_ref = np.array([float((t - reference_date) / np.timedelta64(1, 's')) for t in time_values])
        except Exception:
            try:
                time_values = xr.coding.times.cftime_to_nptime(time_values)
                seconds_since_ref = np.array(
                    [float((t - reference_date) / np.timedelta64(1, 's')) for t in time_values]
                )
            except Exception as inner_e:
                raise TypeError('Time values could not be converted to seconds since reference_date') from inner_e

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

    def _slice_time_info(self, time_info: Dict, time_slice: slice) -> Dict:
        """Slice time info to specified range."""
        sliced_info = time_info.copy()
        sliced_info['time_values'] = time_info['time_values'][time_slice]
        sliced_info['seconds_since_reference'] = time_info['seconds_since_reference'][time_slice]
        sliced_info['num_times'] = len(sliced_info['time_values'])
        if len(sliced_info['time_values']) > 0:
            sliced_info['time_start'] = sliced_info['time_values'][0]
            sliced_info['time_end'] = sliced_info['time_values'][-1]
        return sliced_info

    def _calculate_time_slice(self, current_time, reading_interval, time_info):
        """Calculate time slice indices based on current time and reading interval."""

        if current_time is None or reading_interval is None:
            return None, None

        if reading_interval <= 0 or reading_interval >= time_info['seconds_since_reference'][-1]:
            return None, None

        times_array = time_info['seconds_since_reference']
        current_idx = np.searchsorted(times_array, current_time)

        netcdf_timestep = times_array[1] - times_array[0] if len(times_array) > 1 else 1.0
        chunk_steps = max(10, int(reading_interval / netcdf_timestep))

        start_idx = max(0, current_idx - chunk_steps // 4)
        end_idx = min(len(times_array), current_idx + chunk_steps)

        return start_idx, end_idx

    def _select_first_dims(self, var: xr.DataArray) -> xr.DataArray:
        """Select the first index for known extra dimensions (layers, fractions)."""
        selection = {}
        for dim in [
            'KMAXOUT_RESTR',
            'KMAXOUT',
            'LSED',
            'LSEDTOT',
            'LSTSCI',
            'SIG_LYR',
            'SIG_INTF',
            'nlyr',
            'nlyrp1',
            'layer',
        ]:
            if dim in var.dims:
                selection[dim] = 0
        if selection:
            var = var.isel(**selection)
        return var

    def _ensure_grid_shape(self, values: np.ndarray, grid_shape: tuple) -> np.ndarray:
        """Trim arrays to match the grid shape when possible."""
        if values.shape[-2:] == grid_shape:
            return values
        if len(values.shape) >= 2:
            return values[..., : grid_shape[0], : grid_shape[1]]
        return values

    def _interpolate_to_centers(self, values: np.ndarray, axis: int) -> np.ndarray:
        """Interpolate staggered-grid values to cell centers along a given axis."""
        if values.ndim < 2:
            return values

        rolled = np.roll(values, -1, axis=axis)
        centered = 0.5 * (values + rolled)
        indexer = [slice(None)] * centered.ndim
        indexer[axis] = -1
        centered[tuple(indexer)] = np.nan
        return centered

    def _map_delft3d4_variables(
        self, time_info: Dict, time_start_idx: Optional[int] = None, time_end_idx: Optional[int] = None
    ) -> Dict:
        """Map Delft3D4 variables to SedtrailsData structure."""
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')

        num_times = time_info['num_times']
        time_slice = (
            slice(time_start_idx, time_end_idx)
            if time_start_idx is not None or time_end_idx is not None
            else slice(None)
        )

        data: Dict[str, np.ndarray] = {}

        if 'XZ' in self.input_data and 'YZ' in self.input_data:
            data['x'] = self.input_data['XZ'].values
            data['y'] = self.input_data['YZ'].values
        elif 'XCOR' in self.input_data and 'YCOR' in self.input_data:
            data['x'] = self.input_data['XCOR'].values
            data['y'] = self.input_data['YCOR'].values
        else:
            raise KeyError("Required variables 'XZ'/'YZ' or 'XCOR'/'YCOR' not found in dataset")

        grid_shape = data['x'].shape

        bed_level_var = None
        for candidate in ['DPS0', 'DP0']:
            if candidate in self.input_data:
                bed_level_var = self.input_data[candidate]
                break

        if bed_level_var is not None:
            bed_level_vals = self._select_first_dims(bed_level_var).values
            data['bed_level'] = self._ensure_grid_shape(bed_level_vals, grid_shape)
        else:
            data['bed_level'] = np.zeros(grid_shape)
            print("Warning: Variables 'DPS0' and 'DP0' not found, using zeros for bed level")

        variable_map = {
            'water_depth': 'DPS',
            'flow_velocity_x': 'U1',
            'flow_velocity_y': 'V1',
            'bed_shear_stress_x': 'TAUKSI',
            'bed_shear_stress_y': 'TAUETA',
            'max_bed_shear_stress': 'TAUMAX',
            'bed_load_transport_x': 'SBUU',
            'bed_load_transport_y': 'SBVV',
            'suspended_transport_x': 'SSUU',
            'suspended_transport_y': 'SSVV',
            'sediment_concentration': 'R1',
        }

        for key, var_name in variable_map.items():
            if var_name in self.input_data:
                var = self._select_first_dims(self.input_data[var_name])
                if 'time' in var.dims:
                    values = var.isel(time=time_slice).values
                else:
                    values = np.broadcast_to(var.values, (num_times, *var.shape))

                if key in {
                    'flow_velocity_x',
                    'bed_shear_stress_x',
                    'bed_load_transport_x',
                    'suspended_transport_x',
                }:
                    values = self._interpolate_to_centers(values, axis=-2)
                elif key in {
                    'flow_velocity_y',
                    'bed_shear_stress_y',
                    'bed_load_transport_y',
                    'suspended_transport_y',
                }:
                    values = self._interpolate_to_centers(values, axis=-1)

                data[key] = self._ensure_grid_shape(values, grid_shape)
            else:
                data[key] = np.zeros((num_times, *grid_shape))
                print(f"Warning: Variable '{var_name}' not found, using zeros")

        if 'water_depth' in data and np.all(data['water_depth'] == 0) and 'S1' in self.input_data:
            s1 = self._select_first_dims(self.input_data['S1'])
            if 'time' in s1.dims:
                s1_vals = s1.isel(time=time_slice).values
            else:
                s1_vals = np.broadcast_to(s1.values, (num_times, *s1.shape))
            data['water_depth'] = self._ensure_grid_shape(s1_vals + data['bed_level'], grid_shape)

        return data

