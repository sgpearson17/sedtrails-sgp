"""A plugin for converting XBeach NetCDF mean output to SedTRAILS format."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from sedtrails.transport_converter.plugins import BaseFormatPlugin
from sedtrails.transport_converter.sedtrails_data import SedtrailsData
from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata
from sedtrails.transport_converter.time_utils import decompress_time_info


class FormatPlugin(BaseFormatPlugin):
    """
    Plugin for converting XBeach NetCDF mean variables to SedTRAILS format.

    XBeach writes instantaneous variables on ``globaltime`` and averaged
    variables on ``meantime``. This plugin deliberately maps only ``*_mean``
    variables and uses ``meantime`` as the SedTRAILS time coordinate.

    Parameters
    ----------
    input_file : str
        Path to the XBeach NetCDF file.
    morfac : float, default 1.0
        Morphological acceleration factor for time decompression.
    """

    TIME_DIM = 'meantime'
    FRACTION_DIM = 'sediment_classes'

    def __init__(self, input_file: str, morfac: float = 1.0):
        """
        Initialize an XBeach format plugin.

        Parameters
        ----------
        input_file : str
            Path to the XBeach NetCDF file.
        morfac : float, default 1.0
            Morphological acceleration factor for time decompression. The
            default is 1.0, which leaves input times unchanged.

        Raises
        ------
        FileNotFoundError
            If ``input_file`` does not exist.
        """
        super().__init__()
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f'Input file not found: {self.input_file}')
        self.morfac = morfac
        self.input_data: xr.Dataset | None = None
        self._input_variables: List[str] = []

    @property
    def variables(self) -> List[str]:
        """
        Get the variables in the input dataset.

        Returns
        -------
        list[str]
            List of variable names in the input dataset.

        Raises
        ------
        OSError
            If the input NetCDF file cannot be opened.
        ValueError
            If the loaded dataset does not expose data variables.
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
        self,
        current_time=None,
        reading_interval=None,
        reference_date: Optional[np.datetime64] = None,
    ) -> SedtrailsData:
        """
        Convert XBeach NetCDF mean output to SedtrailsData.

        Parameters
        ----------
        current_time : float, optional
            Current simulation time in seconds.
        reading_interval : float, optional
            Reading interval in seconds.
        reference_date : np.datetime64, optional
            Reference date for converting time values.

        Returns
        -------
        SedtrailsData
            Converted SedTRAILS data. Spatial ``ny,nx`` fields are flattened
            to a single spatial axis. XBeach sediment transport components are
            summed over source ``sediment_classes`` before being stored.

        Raises
        ------
        KeyError
            If a required XBeach coordinate, time coordinate, or ``*_mean``
            variable is missing.
        OSError
            If the input NetCDF file cannot be opened.
        TypeError
            If ``reference_date`` is not a ``numpy.datetime64``.
        ValueError
            If required variables have incompatible dimensions.
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

        mapped_data = self._map_xbeach_variables(time_info, time_start_idx, time_end_idx)
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
            'x': mapped_data['nonlinear_wave_velocity_x'],
            'y': mapped_data['nonlinear_wave_velocity_y'],
            'magnitude': np.sqrt(
                mapped_data['nonlinear_wave_velocity_x'] ** 2 + mapped_data['nonlinear_wave_velocity_y'] ** 2
            ),
        }

        metadata = SedtrailsMetadata(
            flowfield_domain={
                'x_min': np.nanmin(mapped_data['x']),
                'x_max': np.nanmax(mapped_data['x']),
                'y_min': np.nanmin(mapped_data['y']),
                'y_max': np.nanmax(mapped_data['y']),
            }
        )
        metadata.update(
            {
                'model': 'xbeach',
                'time_coordinate': self.TIME_DIM,
                'uses_mean_variables': True,
                'source_sediment_classes': mapped_data['source_sediment_classes'],
                'sediment_transport_fraction_handling': 'sum_over_source_sediment_classes',
                'max_bed_shear_stress_source': 'taubx_mean/tauby_mean magnitude',
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
            mean_bed_shear_stress=mapped_data['mean_bed_shear_stress'],
            # XBeach sample output has taub*_mean but no *_mean maximum. Use the
            # mean magnitude so downstream physics can run without reading *_max.
            max_bed_shear_stress=mapped_data['mean_bed_shear_stress'],
            sediment_concentration=mapped_data['sediment_concentration'],
            nonlinear_wave_velocity=nonlinear_wave_velocity,
            metadata=metadata,
        )

    def get_seeding_coordinates(self):
        """
        Return the spatial coordinates required for particle seeding.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Flattened X and Y cell-center coordinates, each with shape
            ``(n_points,)``.

        Raises
        ------
        KeyError
            If ``globalx`` or ``globaly`` is missing from the input dataset.
        OSError
            If the input NetCDF file cannot be opened.
        ValueError
            If ``globalx`` and ``globaly`` have incompatible shapes.
        """
        self.load()
        return self._get_grid_coordinates()

    def get_time_bounds(self, reference_date: Optional[np.datetime64] = None) -> tuple[float, float]:
        """
        Return input mean-time bounds in seconds since the reference date.

        Parameters
        ----------
        reference_date : np.datetime64, optional
            Reference date used to convert the input time coordinate to seconds.
            If omitted, the Unix epoch is used.

        Returns
        -------
        tuple[float, float]
            First and last input timestamps in seconds since `reference_date`.

        Raises
        ------
        KeyError
            If the XBeach ``meantime`` coordinate is missing.
        OSError
            If the input NetCDF file cannot be opened.
        TypeError
            If ``reference_date`` is not a ``numpy.datetime64``.
        ValueError
            If the input data contains no mean-time values.
        """
        if reference_date is None:
            reference_date = np.datetime64('1970-01-01T00:00:00')

        self.load()
        time_info = self._get_time_info(self.input_data, reference_date=reference_date)
        time_info = self._decompress_time(time_info)
        times = np.asarray(time_info['seconds_since_reference'], dtype=float)
        if times.size == 0:
            raise ValueError('Input data contains no mean time values')
        return float(times[0]), float(times[-1])

    def load(self) -> Any:
        """
        Read and cache the XBeach NetCDF dataset.

        Returns
        -------
        xarray.Dataset
            The opened XBeach dataset. Repeated calls return the cached
            dataset.

        Raises
        ------
        OSError
            If the NetCDF file cannot be opened by xarray.
        """
        if self.input_data is None:
            try:
                self.input_data = xr.open_dataset(self.input_file, decode_times=False, decode_timedelta=False)
            except Exception as e:
                raise IOError(f'Failed to open XBeach NetCDF file: {e}') from e
            else:
                print(f'Successfully loaded (Xarray): {self.input_file}')
        return self.input_data

    def _decompress_time(self, time_info: Dict) -> Dict:
        """Apply morfac decompression to time values."""
        return decompress_time_info(time_info, self.morfac)

    def _get_time_info(self, input_data: xr.Dataset, reference_date: np.datetime64) -> Dict:
        """Get XBeach mean-time information in seconds since `reference_date`."""
        if not isinstance(reference_date, np.datetime64):
            raise TypeError('reference_date must be a numpy datetime64 object')
        if input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')
        if self.TIME_DIM not in input_data:
            raise KeyError(
                f"Required mean time coordinate '{self.TIME_DIM}' not found. "
                'XBeach conversion requires *_mean output on the meantime axis.'
            )

        time_var = input_data[self.TIME_DIM]
        raw_time_values = np.asarray(time_var.values)
        orig_units = time_var.attrs.get('units')
        orig_calendar = time_var.attrs.get('calendar', 'standard')

        if np.issubdtype(raw_time_values.dtype, np.datetime64):
            time_values = raw_time_values
            seconds_since_ref = np.array(
                [float((t - reference_date) / np.timedelta64(1, 's')) for t in time_values],
                dtype=float,
            )
        elif orig_units and 'since' in orig_units:
            decoded = xr.coding.times.decode_cf_datetime(raw_time_values, orig_units, orig_calendar)
            try:
                decoded = xr.coding.times.cftime_to_nptime(decoded)
            except Exception:
                pass
            time_values = np.asarray(decoded)
            seconds_since_ref = np.array(
                [float((t - reference_date) / np.timedelta64(1, 's')) for t in time_values],
                dtype=float,
            )
        else:
            seconds_since_ref = np.asarray(raw_time_values, dtype=float)
            time_values = np.array(
                [
                    reference_date + np.timedelta64(int(round(seconds * 1_000_000)), 'us')
                    for seconds in seconds_since_ref
                ]
            )

        return {
            'time_values': time_values,
            'time_start': time_values[0],
            'time_end': time_values[-1],
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

    def _map_xbeach_variables(
        self, time_info, time_start_idx: Optional[int] = None, time_end_idx: Optional[int] = None
    ) -> Dict:
        """
        Map XBeach ``*_mean`` variables to SedtrailsData fields.

        Spatial ``ny,nx`` dimensions are flattened to a single SedTRAILS
        spatial axis. XBeach sediment transport variables are summed over the
        source ``sediment_classes`` dimension before entering SedTRAILS.
        """
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')

        num_times = time_info['num_times']
        time_slice = (
            slice(time_start_idx, time_end_idx)
            if time_start_idx is not None or time_end_idx is not None
            else slice(None)
        )

        x, y = self._get_grid_coordinates()
        grid_size = x.size

        bed_load_transport_x, bed_load_classes_x = self._mean_transport_component(
            'Subg_mean', time_slice, num_times, grid_size
        )
        bed_load_transport_y, bed_load_classes_y = self._mean_transport_component(
            'Svbg_mean', time_slice, num_times, grid_size
        )
        suspended_transport_x, suspended_classes_x = self._mean_transport_component(
            'Susg_mean', time_slice, num_times, grid_size
        )
        suspended_transport_y, suspended_classes_y = self._mean_transport_component(
            'Svsg_mean', time_slice, num_times, grid_size
        )
        source_sediment_classes = max(
            bed_load_classes_x,
            bed_load_classes_y,
            suspended_classes_x,
            suspended_classes_y,
        )

        data = {
            'x': x,
            'y': y,
            'bed_level': self._mean_scalar('zb_mean', time_slice, num_times, grid_size),
            'water_depth': self._mean_scalar('hh_mean', time_slice, num_times, grid_size),
            'flow_velocity_x': self._mean_scalar('ue_mean', time_slice, num_times, grid_size),
            'flow_velocity_y': self._mean_scalar('ve_mean', time_slice, num_times, grid_size),
            'sediment_concentration': self._mean_scalar('cctot_mean', time_slice, num_times, grid_size),
            'bed_load_transport_x': bed_load_transport_x,
            'bed_load_transport_y': bed_load_transport_y,
            'suspended_transport_x': suspended_transport_x,
            'suspended_transport_y': suspended_transport_y,
            'source_sediment_classes': source_sediment_classes,
        }

        taubx = self._mean_scalar('taubx_mean', time_slice, num_times, grid_size)
        tauby = self._mean_scalar('tauby_mean', time_slice, num_times, grid_size)
        data['mean_bed_shear_stress'] = np.sqrt(taubx**2 + tauby**2)

        if 'ua_mean' in self.input_data and 'thetamean_mean' in self.input_data:
            theta = self._mean_scalar('thetamean_mean', time_slice, num_times, grid_size)
            ua = self._mean_scalar('ua_mean', time_slice, num_times, grid_size)
            data['nonlinear_wave_velocity_x'] = ua * np.cos(theta)
            data['nonlinear_wave_velocity_y'] = ua * np.sin(theta)
        else:
            data['nonlinear_wave_velocity_x'] = np.zeros((num_times, grid_size), dtype=float)
            data['nonlinear_wave_velocity_y'] = np.zeros((num_times, grid_size), dtype=float)
            print("Warning: Variable 'ua_mean' not found, using zeros for nonlinear wave velocity")

        return data

    def _get_grid_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return flattened XBeach cell-center coordinates."""
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')
        if 'globalx' not in self.input_data or 'globaly' not in self.input_data:
            raise KeyError("Required XBeach coordinates 'globalx' and/or 'globaly' not found")

        x = np.asarray(self.input_data['globalx'].values, dtype=float).ravel()
        y = np.asarray(self.input_data['globaly'].values, dtype=float).ravel()
        if x.shape != y.shape:
            raise ValueError(f'globalx and globaly must have the same flattened shape, got {x.shape} and {y.shape}')
        return x, y

    def _mean_scalar(self, var_name: str, time_slice: slice, num_times: int, grid_size: int) -> np.ndarray:
        """Read a scalar ``*_mean`` variable as ``(time, spatial)``."""
        var = self._require_mean_variable(var_name)
        if self.FRACTION_DIM in var.dims:
            raise ValueError(f"Expected scalar mean variable '{var_name}', found fraction dimension {var.dims}")
        var = self._select_time(var, time_slice)
        if self.TIME_DIM in var.dims:
            spatial_dims = [dim for dim in var.dims if dim != self.TIME_DIM]
            var = var.transpose(self.TIME_DIM, *spatial_dims)
            values = np.asarray(var.values, dtype=float)
        else:
            values = np.broadcast_to(np.asarray(var.values, dtype=float), (num_times, *var.shape))
        return values.reshape(num_times, grid_size)

    def _mean_transport_component(
        self, var_name: str, time_slice: slice, num_times: int, grid_size: int
    ) -> tuple[np.ndarray, int]:
        """Read a transport ``*_mean`` variable as total ``(time, spatial)`` data."""
        var = self._require_mean_variable(var_name)
        var = self._select_time(var, time_slice)
        source_sediment_classes = int(var.sizes.get(self.FRACTION_DIM, 1))
        if self.FRACTION_DIM in var.dims:
            var = var.sum(dim=self.FRACTION_DIM)
        return self._reshape_mean_spatial(var, num_times, grid_size), source_sediment_classes

    def _reshape_mean_spatial(self, var: xr.DataArray, num_times: int, grid_size: int) -> np.ndarray:
        """Return a mean variable with flattened spatial dimensions."""
        if self.TIME_DIM in var.dims:
            spatial_dims = [dim for dim in var.dims if dim != self.TIME_DIM]
            var = var.transpose(self.TIME_DIM, *spatial_dims)
            return np.asarray(var.values, dtype=float).reshape(num_times, grid_size)
        values = np.asarray(var.values, dtype=float).reshape(grid_size)
        return np.broadcast_to(values, (num_times, grid_size))

    def _require_mean_variable(self, var_name: str) -> xr.DataArray:
        """Return a required XBeach ``*_mean`` variable or raise a clear error."""
        if not var_name.endswith('_mean'):
            raise ValueError(f"XBeach plugin only reads *_mean variables, got '{var_name}'")
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')
        if var_name not in self.input_data:
            raise KeyError(f"Required XBeach mean variable '{var_name}' not found in dataset")
        return self.input_data[var_name]

    def _select_time(self, var: xr.DataArray, time_slice: slice) -> xr.DataArray:
        """Apply the XBeach mean-time slice when present."""
        if self.TIME_DIM in var.dims:
            return var.isel({self.TIME_DIM: time_slice})
        return var

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
