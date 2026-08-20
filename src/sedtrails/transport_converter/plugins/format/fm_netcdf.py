"""A plugin for converting Delft3D Flexible Mesh NetCDF to SedTRAILS format."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr
import xugrid as xu

from sedtrails.transport_converter.plugins import BaseFormatPlugin
from sedtrails.transport_converter.plugins.format._xugrid_compat import create_ugrid2d
from sedtrails.particle_tracer.coordinate_transform import infer_coordinate_system_from_attrs
from sedtrails.transport_converter.domain_mask import (
    ConnectivityMaskResult,
    classify_boundary_edges_from_config,
    delaunay_connectivity,
    filter_connectivity_by_inner_polygons,
    inner_boundary_files_from_config,
    load_inner_boundary_polygons,
    triangulate_face_connectivity,
)
from sedtrails.transport_converter.sedtrails_data import SedtrailsData
from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata
from sedtrails.transport_converter.time_utils import decompress_time_info


class FormatPlugin(BaseFormatPlugin):
    """Convert Delft3D Flexible Mesh NetCDF data to SedTRAILS format.

    Parameters
    ----------
    input_file : str
        Path to the Delft3D Flexible Mesh NetCDF file.
    morfac : float, optional
        Morphological acceleration factor used to decompress model time.
    """

    SELECTABLE_FIELDS = frozenset(
        {
            'bed_level',
            'depth_avg_flow_velocity',
            'bed_load_transport',
            'suspended_transport',
            'water_depth',
            'mean_bed_shear_stress',
            'max_bed_shear_stress',
            'sediment_concentration',
            'nonlinear_wave_velocity',
        }
    )

    _VARIABLE_MAP = {
        'bed_level': 'bedlevel',
        'water_depth': 'waterdepth',
        'flow_velocity_x': 'sea_water_x_velocity',
        'flow_velocity_y': 'sea_water_y_velocity',
        'mean_bed_shear_stress': 'mean_bss_magnitude',
        'max_bed_shear_stress': 'max_bss_magnitude',
        'bed_load_transport_x': 'bedload_x_comp',
        'bed_load_transport_y': 'bedload_y_comp',
        'suspended_transport_x': 'susload_x_comp',
        'suspended_transport_y': 'susload_y_comp',
        'sediment_concentration': 'suspended_sed_conc',
    }

    _FIELD_COMPONENT_KEYS = {
        'bed_level': ('bed_level',),
        'depth_avg_flow_velocity': ('flow_velocity_x', 'flow_velocity_y'),
        'bed_load_transport': ('bed_load_transport_x', 'bed_load_transport_y'),
        'suspended_transport': ('suspended_transport_x', 'suspended_transport_y'),
        'water_depth': ('water_depth',),
        'mean_bed_shear_stress': ('mean_bed_shear_stress',),
        'max_bed_shear_stress': ('max_bed_shear_stress',),
        'sediment_concentration': ('sediment_concentration',),
    }

    def __init__(self, input_file: str, morfac: float = 1.0):
        """
        Initialize the plugin with the input file.

        Parameters:
        -----------
        input_file : str
            Path to the Delft3D Flexible Mesh NetCDF file.
        morfac : float, optional
            Morphological acceleration factor for time decompression (default: 1.0)
        """
        super().__init__()
        self.input_file = Path(input_file)
        self.morfac = morfac
        self.input_data = None  # holds Dataset after reading
        self._input_variables: List[str] = []
        self.domain_config: Dict[str, Any] = {}
        self.coordinate_system: str | None = None
        self.source_crs: str | None = None
        self.metric_crs: str | None = None
        self._inner_boundary_polygons: list[np.ndarray] | None = None
        self._inner_boundary_polygons_signature: str | None = None
        self._last_inner_boundary_mask: ConnectivityMaskResult | None = None
        self._active_triangular_connectivity_cache: dict[str, Any] | None = None
        self._boundary_edge_classification_cache: dict[str, Any] | None = None

    def __post_init__(self):
        # Check if the input file exists
        if not self.input_file.exists():
            raise FileNotFoundError(f'Input file not found: {self.input_file}')

    @property
    def variables(self) -> List[str]:
        """
        Get the variables in the input dataset.

        Returns
        -------
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

    def _decompress_time(self, time_info: Dict) -> Dict:
        """
        Apply morfac decompression to time values.

        Parameters:
        -----------
        time_info : Dict
            Original time information

        Returns:
        --------
        Dict
            Time information with decompressed time values
        """
        return decompress_time_info(time_info, self.morfac)

    def convert(
        self,
        current_time=None,
        reading_interval=None,
        reference_date: Optional[np.datetime64] = None,
        required_fields=None,
        max_memory_bytes: int | None = None,
    ) -> SedtrailsData:
        """
        Delft3D from Flexible Mesh NetCDF.

        Parameters
        ----------
        current_time : float, optional
            Current simulation time in seconds
        reading_interval : float, optional
            Reading interval in seconds
        reference_date : np.datetime64, optional
            Reference date for converting time values.
        required_fields : sequence of str, optional
            SedTRAILS source fields to materialize. Omission preserves the
            historical all-fields conversion.
        max_memory_bytes : int, optional
            Estimated byte limit for time-varying fields in this window.

        Returns
        -------
        SedtrailsData
            The converted SedtrailsData object.
        """

        if reference_date is None:
            reference_date = np.datetime64('1970-01-01T00:00:00')

        # Read the NetCDF file
        self.load()
        time_info = self._get_time_info(self.input_data, reference_date=reference_date)

        # Apply morfac decompression to time before time slicing
        time_info = self._decompress_time(time_info)

        selected_fields = self._selectable_required_fields(required_fields)
        # Determine if we need to slice based on current_time and reading_interval
        time_start_idx, time_end_idx = self._calculate_time_slice(current_time, reading_interval, time_info)
        time_start_idx, time_end_idx = self._limit_time_slice_by_memory(
            time_info,
            time_start_idx,
            time_end_idx,
            current_time=current_time,
            selected_fields=selected_fields,
            max_memory_bytes=max_memory_bytes,
        )

        # Apply time slicing if needed
        if time_start_idx is not None or time_end_idx is not None:
            time_slice = slice(time_start_idx, time_end_idx)
            time_info = self._slice_time_info(time_info, time_slice)

        # Map the variables to SedtrailsData structure
        mapped_data = self._map_dfm_variables(
            time_info,
            time_start_idx,
            time_end_idx,
            required_fields=selected_fields,
        )
        seconds_since_ref = time_info['seconds_since_reference']
        self.reference_date = time_info['reference_date']

        # TODO: DFM slicing can introduce an extra leading singleton dimension; remove only that axis.
        for key, value in mapped_data.items():
            if isinstance(value, np.ndarray) and value.ndim > 2 and value.shape[0] == 1:
                mapped_data[key] = np.squeeze(value, axis=0)

        depth_avg_flow_velocity = self._mapped_vector_field(
            mapped_data,
            'flow_velocity',
            selected='depth_avg_flow_velocity' in selected_fields,
        )
        bed_load_transport = self._mapped_vector_field(
            mapped_data,
            'bed_load_transport',
            selected='bed_load_transport' in selected_fields,
        )
        suspended_transport = self._mapped_vector_field(
            mapped_data,
            'suspended_transport',
            selected='suspended_transport' in selected_fields,
        )
        if 'nonlinear_wave_velocity' in selected_fields:
            if depth_avg_flow_velocity is not None:
                nonlinear_wave_velocity = {
                    'x': np.zeros_like(depth_avg_flow_velocity['x']),
                    'y': np.zeros_like(depth_avg_flow_velocity['y']),
                    'magnitude': np.zeros_like(depth_avg_flow_velocity['magnitude']),
                }
            else:
                zero_shape = (len(seconds_since_ref), mapped_data['x'].size)
                nonlinear_wave_velocity = {
                    'x': np.zeros(zero_shape, dtype=float),
                    'y': np.zeros(zero_shape, dtype=float),
                    'magnitude': np.zeros(zero_shape, dtype=float),
                }
        else:
            nonlinear_wave_velocity = None

        # Create SedtrailsMetadata object
        metadata = SedtrailsMetadata(
            flowfield_domain={
                'x_min': np.min(mapped_data['x']),
                'x_max': np.max(mapped_data['x']),
                'y_min': np.min(mapped_data['y']),
                'y_max': np.max(mapped_data['y']),
            }
        )
        metadata.add('coordinate_system', self._coordinate_system())
        self._add_crs_metadata(metadata)

        # Create SedtrailsData object
        sedtrails_data = SedtrailsData(
            times=seconds_since_ref,
            reference_date=self.reference_date,
            x=mapped_data['x'],
            y=mapped_data['y'],
            bed_level=mapped_data.get('bed_level'),
            depth_avg_flow_velocity=depth_avg_flow_velocity,
            fractions=1,  # Default to 1 fraction
            bed_load_transport=bed_load_transport,
            suspended_transport=suspended_transport,
            water_depth=mapped_data.get('water_depth'),
            mean_bed_shear_stress=mapped_data.get('mean_bed_shear_stress'),
            max_bed_shear_stress=mapped_data.get('max_bed_shear_stress'),
            sediment_concentration=mapped_data.get('sediment_concentration'),
            nonlinear_wave_velocity=nonlinear_wave_velocity,
            node_x=mapped_data['x'],
            node_y=mapped_data['y'],
            face_node_connectivity=mapped_data.get('face_node_connectivity'),
            face_node_fill_value=-1,
            metadata=metadata,
        )

        self._add_inner_boundary_metadata(sedtrails_data.metadata)
        self._add_boundary_edge_metadata(
            sedtrails_data.metadata,
            sedtrails_data.x,
            sedtrails_data.y,
            sedtrails_data.face_node_connectivity,
        )

        return sedtrails_data

    def get_seeding_field_data(self):
        """Return active geometry required for particle seeding.

        Returns
        -------
        types.SimpleNamespace
            Object with ``x`` and ``y`` node coordinates, active triangular
            ``face_node_connectivity``, optional
            ``boundary_edge_classification`` metadata, and
            ``face_node_fill_value``.

        Raises
        ------
        KeyError
            If the required ``net_xcc`` or ``net_ycc`` variables are missing.
        FileNotFoundError
            If configured inner-boundary or boundary-class polygon files do
            not exist.
        ValueError
            If configured Tekal polygon blocks are malformed.
        """
        self.load()
        if 'net_xcc' not in self.input_data or 'net_ycc' not in self.input_data:
            raise KeyError("Required variables 'net_xcc' and/or 'net_ycc' not found in dataset")

        x = self.input_data['net_xcc'].values
        y = self.input_data['net_ycc'].values
        connectivity = self._active_triangular_connectivity(x, y)
        boundary_edge_classification = self._boundary_edge_classification(x, y, connectivity)
        return SimpleNamespace(
            x=x,
            y=y,
            face_node_connectivity=connectivity,
            boundary_edge_classification=boundary_edge_classification,
            face_node_fill_value=-1,
            coordinate_system=self._coordinate_system(),
            source_crs=self.source_crs,
            metric_crs=self.metric_crs,
        )

    def get_seeding_coordinates(self):
        """
        Return only the spatial coordinates required for particle seeding.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            X and Y coordinates used for particle seeding.
        """
        self.load()

        if 'net_xcc' not in self.input_data or 'net_ycc' not in self.input_data:
            raise KeyError("Required variables 'net_xcc' and/or 'net_ycc' not found in dataset")

        return self.input_data['net_xcc'].values, self.input_data['net_ycc'].values

    def get_max_exposure_depth_fields(self):
        """
        Compute per-node maximum erosion depth and maximum bed shear stress over the
        full dataset (all time steps), without loading every time step into memory.

        Returns
        -------
        max_erosion : np.ndarray, shape (n_nodes,)
            Maximum erosion depth per node [m]: max(bed_level_t0 - bed_level_t) over all t,
            clipped to >= 0.  Zero for nodes with a static bed level.
        max_bss : np.ndarray, shape (n_nodes,)
            Maximum bed shear stress per node over all time steps [N/m²].
        """
        self.load()

        bed_var = 'bedlevel'
        bss_var = 'max_bss_magnitude'

        if bed_var not in self.input_data:
            raise KeyError(f"Required variable '{bed_var}' not found in dataset")
        if bss_var not in self.input_data:
            raise KeyError(f"Required variable '{bss_var}' not found in dataset")

        bed = self.input_data[bed_var]
        bss = self.input_data[bss_var]

        if 'time' in bed.dims:
            bed_initial = bed.isel(time=0).values.astype(float)
            bed_min = bed.min(dim='time').values.astype(float)
            max_erosion = np.maximum(bed_initial - bed_min, 0.0)
        else:
            max_erosion = np.zeros(np.asarray(bed.values).shape, dtype=float)

        if 'time' in bss.dims:
            max_bss = bss.max(dim='time').values.astype(float)
        else:
            max_bss = np.asarray(bss.values, dtype=float)

        return max_erosion, max_bss

    def get_time_bounds(self, reference_date: Optional[np.datetime64] = None) -> tuple[float, float]:
        """
        Return input time bounds in seconds since the configured reference date.

        Parameters
        ----------
        reference_date : np.datetime64, optional
            Reference date used to convert the input time coordinate to seconds.
            If omitted, the Unix epoch is used.

        Returns
        -------
        tuple of float
            First and last input timestamps, in seconds since `reference_date`.

        Raises
        ------
        ValueError
            If the input data contains no time values.
        """
        if reference_date is None:
            reference_date = np.datetime64('1970-01-01T00:00:00')

        self.load()
        time_info = self._get_time_info(self.input_data, reference_date=reference_date)
        time_info = self._decompress_time(time_info)
        times = np.asarray(time_info['seconds_since_reference'], dtype=float)
        if times.size == 0:
            raise ValueError('Input data contains no time values')
        return float(times[0]), float(times[-1])

    def _calculate_time_slice(self, current_time, reading_interval, time_info):
        """Calculate time slice indices based on current time and reading interval."""

        if current_time is None or reading_interval is None:
            return None, None
        if reading_interval <= 0:
            return None, None

        times_array = np.asarray(time_info['seconds_since_reference'], dtype=float)
        if times_array.size <= 2:
            return None, None
        forcing_span = times_array[-1] - times_array[0]
        if forcing_span <= 0 or reading_interval >= forcing_span:
            return None, None

        start_idx, bracket_end_idx = self._interpolation_bracket(times_array, current_time)
        requested_end_time = float(current_time) + float(reading_interval)
        requested_upper_idx = int(np.searchsorted(times_array, requested_end_time, side='left'))
        requested_upper_idx = min(
            times_array.size - 1,
            max(bracket_end_idx - 1, requested_upper_idx),
        )
        return start_idx, requested_upper_idx + 1

    @classmethod
    def _selectable_required_fields(cls, required_fields) -> set[str]:
        """Return supported source fields selected for materialization."""
        if required_fields is None:
            return set(cls.SELECTABLE_FIELDS)
        return set(required_fields).intersection(cls.SELECTABLE_FIELDS)

    @staticmethod
    def _mapped_vector_field(data: Dict, prefix: str, *, selected: bool) -> Dict | None:
        """Build one selected vector field and its magnitude."""
        if not selected:
            return None
        x_values = data[f'{prefix}_x']
        y_values = data[f'{prefix}_y']
        return {
            'x': x_values,
            'y': y_values,
            'magnitude': np.hypot(x_values, y_values),
        }

    @staticmethod
    def _interpolation_bracket(
        times: np.ndarray,
        current_time: float | None,
    ) -> tuple[int, int]:
        """Return a two-plane slice bracketing the requested time."""
        num_times = int(times.size)
        if num_times <= 1:
            return 0, num_times
        if current_time is None:
            return 0, 2

        lower = int(np.searchsorted(times, current_time, side='right')) - 1
        lower = min(max(lower, 0), num_times - 2)
        return lower, lower + 2

    def _limit_time_slice_by_memory(
        self,
        time_info: Dict,
        start_idx: int | None,
        end_idx: int | None,
        *,
        current_time: float | None,
        selected_fields: set[str],
        max_memory_bytes: int | None,
    ) -> tuple[int | None, int | None]:
        """Cap a time window by estimated retained field bytes."""
        if max_memory_bytes is None:
            return start_idx, end_idx

        try:
            memory_limit = int(max_memory_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError('max_memory_bytes must be an integer byte count') from exc

        times = np.asarray(time_info['seconds_since_reference'], dtype=float)
        num_times = int(times.size)
        bytes_per_plane = self._estimate_bytes_per_time_plane(selected_fields)
        if num_times == 0 or bytes_per_plane == 0:
            return start_idx, end_idx

        required_planes = min(2, num_times)
        required_bytes = required_planes * bytes_per_plane
        if memory_limit < required_bytes:
            fields_text = ', '.join(sorted(selected_fields)) or '<none>'
            raise MemoryError(
                f'Delft3D-FM forcing requires at least {required_bytes} bytes for '
                f'{required_planes} interpolation planes ({bytes_per_plane} bytes/plane) '
                f'for fields [{fields_text}], but max_memory_bytes={memory_limit}. '
                'Increase inputs.max_eulerian_memory_mb or request fewer fields.'
            )

        max_planes = min(
            num_times,
            max(required_planes, memory_limit // bytes_per_plane),
        )
        desired_start = 0 if start_idx is None else int(start_idx)
        desired_end = num_times if end_idx is None else int(end_idx)
        if desired_end - desired_start <= max_planes:
            return start_idx, end_idx

        bracket_start, bracket_end = self._interpolation_bracket(times, current_time)
        latest_start = desired_end - max_planes
        bounded_start = min(max(bracket_start, desired_start), latest_start)
        bounded_start = max(desired_start, bounded_start)
        if bracket_end > bounded_start + max_planes:
            bounded_start = bracket_end - max_planes
        return bounded_start, bounded_start + max_planes

    def _estimate_bytes_per_time_plane(self, selected_fields: set[str]) -> int:
        """Estimate retained array bytes for one selected forcing plane."""
        if self.input_data is None or not selected_fields:
            return 0
        if 'net_xcc' not in self.input_data:
            return 0

        spatial_size = int(np.prod(self.input_data['net_xcc'].shape, dtype=np.int64))
        fallback_bytes = spatial_size * np.dtype(np.float64).itemsize
        total_bytes = 0
        for field_name in selected_fields:
            if field_name == 'nonlinear_wave_velocity':
                total_bytes += 3 * fallback_bytes
                continue

            component_keys = self._FIELD_COMPONENT_KEYS.get(field_name, ())
            variable_names = [self._VARIABLE_MAP[key] for key in component_keys]
            component_bytes = [
                self._source_component_plane_bytes(
                    variable_name,
                    fallback_bytes,
                )
                for variable_name in variable_names
            ]
            total_bytes += sum(component_bytes)
            if len(component_bytes) == 2:
                total_bytes += self._source_hypot_plane_bytes(
                    variable_names[0],
                    variable_names[1],
                    spatial_size,
                )
        return total_bytes

    def estimate_source_bytes_per_time_plane(self, required_fields=None) -> int:
        """Return exact source-backed bytes retained for one forcing plane.

        Parameters
        ----------
        required_fields : sequence of str, optional
            SedTRAILS fields that the runtime plans to read. Omission selects
            every field supported by this plugin.

        Returns
        -------
        int
            Bytes for one source time plane, including derived vector
            magnitudes. Only dataset shape and dtype metadata are inspected.
        """
        if self.input_data is None:
            self.load()
        selected_fields = self._selectable_required_fields(required_fields)
        return self._estimate_bytes_per_time_plane(selected_fields)

    def _source_component_plane_bytes(
        self,
        variable_name: str,
        fallback_bytes: int,
    ) -> int:
        """Return retained bytes for one source component and time plane."""
        fallback_count = fallback_bytes // np.dtype(np.float64).itemsize
        element_count, dtype = self._source_component_plane_spec(
            variable_name,
            fallback_count,
        )
        return element_count * dtype.itemsize

    def _source_component_plane_spec(
        self,
        variable_name: str,
        fallback_count: int,
    ) -> tuple[int, np.dtype]:
        """Return source element count and dtype without reading values."""
        if self.input_data is None or variable_name not in self.input_data:
            return fallback_count, np.dtype(np.float64)

        variable = self.input_data[variable_name]
        element_count = 1
        for dimension, size in variable.sizes.items():
            if dimension in {'time', 'layer'}:
                continue
            element_count *= int(size)
        try:
            dtype = np.dtype(variable.dtype)
        except TypeError:
            dtype = np.dtype(np.float64)
        return element_count, dtype

    def _source_hypot_plane_bytes(
        self,
        x_variable_name: str,
        y_variable_name: str,
        fallback_count: int,
    ) -> int:
        """Return exact bytes of a NumPy hypot-derived magnitude plane."""
        x_count, x_dtype = self._source_component_plane_spec(
            x_variable_name,
            fallback_count,
        )
        y_count, y_dtype = self._source_component_plane_spec(
            y_variable_name,
            fallback_count,
        )
        magnitude_dtype = np.hypot(
            np.zeros((), dtype=x_dtype),
            np.zeros((), dtype=y_dtype),
        ).dtype
        return max(x_count, y_count) * magnitude_dtype.itemsize

    def load(self) -> Any:
        """
        Reads and loads a Delft3D Flexible Mesh NetCDF file using xugrid.

        Returns
        -------
        Any
            Requested value.
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
                    print(f'Successfully loaded (Xarray): {self.input_file}')
            else:
                print('Successfully loaded (Xugrid)', self.input_file)

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

    def _map_dfm_variables(
        self,
        time_info,
        time_start_idx: Optional[int] = None,
        time_end_idx: Optional[int] = None,
        required_fields=None,
    ) -> Dict:
        """
        Map Delft3D Flexible Mesh variables to SedtrailsData structure.

        Parameters:
        -----------
        time_info : Dict
            Time information
        time_start_idx : int, optional
            Start time index for slicing
        time_end_idx : int, optional
            End time index for slicing

        Returns:
        --------
        Dict
            Dictionary with mapped variables
        """
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call read_data() first.')

        # Get time information
        num_times = time_info['num_times']
        time_slice = (
            slice(time_start_idx, time_end_idx)
            if time_start_idx is not None or time_end_idx is not None
            else slice(None)
        )

        selected_fields = self._selectable_required_fields(required_fields)

        # Extract data from dataset
        data = {}

        # First, get spatial coordinates (typically not time-dependent)
        for key in ['x', 'y']:
            var_name = f'net_{key}cc'
            if var_name in self.input_data:
                data[key] = self.input_data[var_name].values
            else:
                raise KeyError(f"Required variable '{var_name}' not found in dataset")

        data['face_node_connectivity'] = self._active_triangular_connectivity(data['x'], data['y'])

        # Determine the spatial grid dimensions
        grid_shape = data['x'].shape

        selected_component_keys = {
            component_key
            for field_name in selected_fields
            for component_key in self._FIELD_COMPONENT_KEYS.get(field_name, ())
        }
        for key, var_name in self._VARIABLE_MAP.items():
            if key not in selected_component_keys:
                continue
            if var_name in self.input_data:
                var = self.input_data[var_name]

                # Check if variable has time dimension
                if 'time' in var.dims:
                    # Check if variable has layer dimension
                    if 'layer' in var.dims:
                        # For variables with time and layer, select layer 0 and apply time slice
                        data[key] = var.isel(layer=0, time=time_slice).values
                    else:
                        # For variables with time but no layer, apply time slice
                        data[key] = var.isel(time=time_slice).values
                else:
                    # For variables without time dimension, broadcast to all time steps
                    data[key] = np.broadcast_to(var.values, (num_times, *var.shape))
            else:
                # Default to zeros if not found
                data[key] = np.zeros((num_times, *grid_shape))
                print(f"Warning: Variable '{var_name}' not found, using zeros")

        return data

    def _active_triangular_connectivity(self, node_x: np.ndarray, node_y: np.ndarray) -> np.ndarray:
        coordinate_system = self._coordinate_system()
        cache = self._active_triangular_connectivity_cache
        if self._geometry_cache_matches(cache, node_x, node_y):
            self._last_inner_boundary_mask = cache['mask_result']
            return cache['triangles']

        source_connectivity = self._source_face_node_connectivity(node_x, node_y)
        if source_connectivity is None:
            candidate_connectivity = delaunay_connectivity(
                node_x,
                node_y,
                coordinate_system=coordinate_system,
                source_crs=self.source_crs,
                metric_crs=self.metric_crs,
                runtime_geometry=getattr(self, 'runtime_geometry', 'planar'),
            )
        else:
            candidate_connectivity = source_connectivity

        mask_result = filter_connectivity_by_inner_polygons(
            node_x,
            node_y,
            candidate_connectivity,
            self._get_inner_boundary_polygons(),
            coordinate_system=coordinate_system,
            source_crs=self.source_crs,
            metric_crs=self.metric_crs,
            runtime_geometry=getattr(self, 'runtime_geometry', 'planar'),
        )
        self._last_inner_boundary_mask = mask_result
        triangles = triangulate_face_connectivity(mask_result.connectivity)
        self._active_triangular_connectivity_cache = {
            'node_x': np.asarray(node_x),
            'node_y': np.asarray(node_y),
            'domain_signature': self._domain_config_signature(),
            'coordinate_system': coordinate_system,
            'source_crs': self.source_crs,
            'metric_crs': self.metric_crs,
            'mask_result': mask_result,
            'triangles': triangles,
        }
        return triangles

    def _get_inner_boundary_polygons(self) -> list[np.ndarray]:
        domain_signature = self._domain_config_signature()
        if self._inner_boundary_polygons is None or self._inner_boundary_polygons_signature != domain_signature:
            self._inner_boundary_polygons = load_inner_boundary_polygons(getattr(self, 'domain_config', {}))
            self._inner_boundary_polygons_signature = domain_signature
        return self._inner_boundary_polygons

    def _add_inner_boundary_metadata(self, metadata: SedtrailsMetadata) -> None:
        inner_files = inner_boundary_files_from_config(getattr(self, 'domain_config', {}))
        if not inner_files and not self._get_inner_boundary_polygons():
            return

        metadata.add('inner_boundary_pol_files', inner_files)
        metadata.add('inner_boundary_polygon_count', len(self._get_inner_boundary_polygons()))
        if self._last_inner_boundary_mask is not None:
            metadata.add('inner_boundary_masked_face_count', self._last_inner_boundary_mask.removed_count)
            metadata.add('inner_boundary_active_face_count', int(self._last_inner_boundary_mask.connectivity.shape[0]))

    def _add_boundary_edge_metadata(
        self,
        metadata: SedtrailsMetadata,
        node_x: np.ndarray,
        node_y: np.ndarray,
        connectivity: np.ndarray | None,
    ) -> None:
        if connectivity is None:
            return

        classification_metadata = self._boundary_edge_classification(node_x, node_y, connectivity)
        if classification_metadata is not None:
            metadata.add('boundary_edge_classification', classification_metadata)

    def _boundary_edge_classification(
        self,
        node_x: np.ndarray,
        node_y: np.ndarray,
        connectivity: np.ndarray | None,
    ) -> dict | None:
        if connectivity is None:
            return None

        coordinate_system = self._coordinate_system()
        cache = self._boundary_edge_classification_cache
        if self._geometry_cache_matches(cache, node_x, node_y, connectivity):
            return cache['metadata']

        classification = classify_boundary_edges_from_config(
            node_x,
            node_y,
            connectivity,
            getattr(self, 'domain_config', {}),
            coordinate_system=coordinate_system,
            source_crs=self.source_crs,
            metric_crs=self.metric_crs,
            runtime_geometry=getattr(self, 'runtime_geometry', 'planar'),
        )
        classification_metadata = None if classification is None else classification.to_metadata()
        self._boundary_edge_classification_cache = {
            'node_x': np.asarray(node_x),
            'node_y': np.asarray(node_y),
            'connectivity': np.asarray(connectivity),
            'domain_signature': self._domain_config_signature(),
            'coordinate_system': coordinate_system,
            'source_crs': self.source_crs,
            'metric_crs': self.metric_crs,
            'metadata': classification_metadata,
        }
        return classification_metadata

    def _domain_config_signature(self) -> str:
        return repr(getattr(self, 'domain_config', {}) or {})

    def _coordinate_system(self) -> str:
        """Return the coordinate-system label inferred from D-Flow FM coordinates."""
        if self.coordinate_system is not None and str(self.coordinate_system).lower() != 'auto':
            return str(self.coordinate_system)
        if self.input_data is None:
            return 'projected'
        return infer_coordinate_system_from_attrs(
            self.input_data.get('net_xcc'),
            self.input_data.get('net_ycc'),
        )

    def _add_crs_metadata(self, metadata: SedtrailsMetadata) -> None:
        """Add configured CRS labels to SedTRAILS metadata."""
        self._add_runtime_coordinate_metadata(metadata)
        if self._coordinate_system() != 'geographic':
            return
        if self.source_crs is not None:
            metadata.add('source_crs', self.source_crs)
        if self.metric_crs is not None:
            metadata.add('metric_crs', self.metric_crs)

    def _geometry_cache_matches(
        self,
        cache: dict[str, Any] | None,
        node_x: np.ndarray,
        node_y: np.ndarray,
        connectivity: np.ndarray | None = None,
    ) -> bool:
        if cache is None or cache.get('domain_signature') != self._domain_config_signature():
            return False
        if cache.get('coordinate_system') != self._coordinate_system():
            return False
        if cache.get('source_crs') != self.source_crs or cache.get('metric_crs') != self.metric_crs:
            return False
        if not self._arrays_equal(cache.get('node_x'), node_x) or not self._arrays_equal(cache.get('node_y'), node_y):
            return False
        if connectivity is None:
            return True
        return self._arrays_equal(cache.get('connectivity'), connectivity)

    @staticmethod
    def _arrays_equal(left: np.ndarray | None, right: np.ndarray) -> bool:
        if left is None:
            return False
        right_array = np.asarray(right)
        return left.shape == right_array.shape and np.array_equal(left, right_array)

    def _source_face_node_connectivity(
        self,
        point_x: np.ndarray,
        point_y: np.ndarray,
    ) -> np.ndarray | None:
        """Return source connectivity expressed in the point-coordinate index space."""
        point_count = int(np.asarray(point_x).size)
        if self._point_coordinates_are_mesh_nodes():
            return self._primal_connectivity_for_nodes(point_count)
        return self._dual_face_center_connectivity(point_x, point_y)

    def _primal_connectivity_for_nodes(self, node_count: int) -> np.ndarray | None:
        """Return primal topology only when points are verified mesh nodes."""
        for variable_name in _FACE_NODE_CONNECTIVITY_CANDIDATES:
            if variable_name in self.input_data:
                connectivity = self._normalize_face_node_connectivity(
                    self.input_data[variable_name],
                    node_count=node_count,
                    variable_name=variable_name,
                )
                if self._connectivity_compatible_with_points(connectivity, node_count):
                    return connectivity
        return None

    def _point_coordinates_are_mesh_nodes(self) -> bool:
        """Return whether net_xcc/net_ycc explicitly share a node dimension."""
        if self.input_data is None or 'net_xcc' not in self.input_data or 'net_ycc' not in self.input_data:
            return False

        x_variable = self.input_data['net_xcc']
        y_variable = self.input_data['net_ycc']
        if tuple(x_variable.dims) != tuple(y_variable.dims):
            return False

        locations = {
            str(variable.attrs.get('location', '')).strip().lower()
            for variable in (x_variable, y_variable)
        }
        if 'face' in locations:
            return False
        if 'node' in locations:
            return True

        dimension_text = ' '.join(str(dimension).lower() for dimension in x_variable.dims)
        if 'face' in dimension_text or 'elem' in dimension_text:
            return False
        if 'node' in dimension_text:
            return True

        for x_name, y_name in zip(_NODE_X_CANDIDATES, _NODE_Y_CANDIDATES, strict=True):
            if x_name not in self.input_data or y_name not in self.input_data:
                continue
            if (
                tuple(self.input_data[x_name].dims) == tuple(x_variable.dims)
                and tuple(self.input_data[y_name].dims) == tuple(y_variable.dims)
            ):
                return True
        return False

    def _dual_face_center_connectivity(
        self,
        face_x: np.ndarray,
        face_y: np.ndarray,
    ) -> np.ndarray | None:
        """Derive face-centre triangles from authoritative primal UGRID topology."""
        point_count = int(np.asarray(face_x).size)
        try:
            if isinstance(self.input_data, xu.UgridDataset):
                grid = self.input_data.grid
                if not isinstance(grid, xu.Ugrid2d) or int(grid.n_face) != point_count:
                    return None
            else:
                node_variables = self._source_node_coordinate_variables()
                if node_variables is None:
                    return None
                node_x_variable, node_y_variable = node_variables
                source_faces = None
                for variable_name in _FACE_NODE_CONNECTIVITY_CANDIDATES:
                    if variable_name not in self.input_data:
                        continue
                    source_faces = self._normalize_face_node_connectivity(
                        self.input_data[variable_name],
                        node_count=int(np.asarray(node_x_variable).size),
                        variable_name=variable_name,
                    )
                    break
                if source_faces is None or source_faces.shape[0] != point_count:
                    return None
                grid = create_ugrid2d(
                    np.asarray(node_x_variable),
                    np.asarray(node_y_variable),
                    source_faces,
                    is_projected=self._coordinate_system() != 'geographic',
                )
            triangulation, face_index = grid.centroid_triangulation
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

        centroid_triangles = np.asarray(triangulation[2], dtype=np.int64)
        source_face_index = np.asarray(face_index, dtype=np.int64)
        if centroid_triangles.size == 0 or source_face_index.size == 0:
            return None

        triangles = source_face_index[centroid_triangles]
        valid = np.all((triangles >= 0) & (triangles < point_count), axis=1)
        valid &= triangles[:, 0] != triangles[:, 1]
        valid &= triangles[:, 0] != triangles[:, 2]
        valid &= triangles[:, 1] != triangles[:, 2]
        if not np.all(valid):
            triangles = triangles[valid]
        return None if triangles.size == 0 else triangles

    def _source_node_coordinate_variables(self):
        """Return a paired source node-coordinate variable set when available."""
        if self.input_data is None:
            return None
        for x_name, y_name in zip(_NODE_X_CANDIDATES, _NODE_Y_CANDIDATES, strict=True):
            if x_name in self.input_data and y_name in self.input_data:
                return self.input_data[x_name], self.input_data[y_name]
        return None

    @staticmethod
    def _connectivity_compatible_with_points(connectivity: np.ndarray, node_count: int) -> bool:
        if connectivity.size == 0:
            return False

        valid = connectivity >= 0
        if not np.any(valid):
            return False
        if int(np.max(connectivity[valid])) >= node_count:
            return False
        return bool(np.all(np.count_nonzero(valid, axis=1) >= 3))

    @staticmethod
    def _normalize_face_node_connectivity(face_nodes_var, node_count: int, variable_name: str | None = None) -> np.ndarray:
        faces = np.asarray(face_nodes_var.values if hasattr(face_nodes_var, 'values') else face_nodes_var)
        if faces.ndim != 2:
            raise ValueError(f'Face-node connectivity must be 2-D, got shape {faces.shape}')

        if faces.shape[0] <= 8 and faces.shape[1] > 8:
            faces = faces.T

        fill_value = _variable_fill_value(face_nodes_var)
        start_index = _variable_start_index(face_nodes_var)
        if start_index is None:
            if variable_name in {'NetElemNode', 'net_elem_node', 'net_element_node'}:
                start_index = 1
            else:
                valid_mask = (
                    np.isfinite(faces)
                    if np.issubdtype(faces.dtype, np.floating)
                    else np.ones(faces.shape, dtype=bool)
                )
                if fill_value is not None:
                    valid_mask &= faces != fill_value
                valid_values = faces[valid_mask]
                if valid_values.size and np.nanmin(valid_values) >= 1 and np.nanmax(valid_values) <= node_count:
                    start_index = 1
                else:
                    start_index = 0

        if (
            np.issubdtype(faces.dtype, np.signedinteger)
            and int(start_index) == 0
            and (fill_value is None or int(fill_value) < 0)
        ):
            return faces

        output_dtype = faces.dtype if np.issubdtype(faces.dtype, np.signedinteger) else np.int64
        normalized = faces.astype(output_dtype, copy=True)
        normalized -= int(start_index)
        invalid = normalized < 0
        if fill_value is not None:
            invalid |= faces == fill_value
        normalized[invalid] = -1
        return normalized


_FACE_NODE_CONNECTIVITY_CANDIDATES = (
    'mesh2d_face_nodes',
    'Mesh2_face_nodes',
    'NetElemNode',
    'net_elem_node',
    'net_element_node',
)

_NODE_X_CANDIDATES = (
    'mesh2d_node_x',
    'Mesh2_node_x',
    'Mesh2d_node_x',
    'NetNode_x',
    'net_node_x',
)

_NODE_Y_CANDIDATES = (
    'mesh2d_node_y',
    'Mesh2_node_y',
    'Mesh2d_node_y',
    'NetNode_y',
    'net_node_y',
)


def _variable_start_index(variable) -> int | None:
    attrs = getattr(variable, 'attrs', {}) or {}
    if 'start_index' in attrs:
        return int(attrs['start_index'])
    return None


def _variable_fill_value(variable):
    attrs = getattr(variable, 'attrs', {}) or {}
    encoding = getattr(variable, 'encoding', {}) or {}
    return encoding.get('_FillValue', attrs.get('_FillValue'))


if __name__ == '__main__':
    # Example usage
    input_file = '/sedtrails/sample-data/inlet_sedtrails.nc'
    plugin = FormatPlugin(input_file, morfac=3.0)

    plugin.load()

    print(plugin.variables)  # List available variables

    data = plugin.convert()  # Convert to SedtrailsData format
    print(data)
