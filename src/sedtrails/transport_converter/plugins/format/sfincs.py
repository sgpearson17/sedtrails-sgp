from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr
import xugrid as xu

from sedtrails.transport_converter.plugins.format._xugrid_compat import create_ugrid2d
from sedtrails.particle_tracer.coordinate_transform import (
    build_coordinate_transform,
    infer_coordinate_system_from_attrs,
)
from sedtrails.transport_converter.domain_mask import (
    classify_boundary_edges_from_config,
    delaunay_connectivity,
    filter_connectivity_by_inner_polygons,
    inner_boundary_files_from_config,
    load_inner_boundary_polygons,
    points_inside_any_polygon,
)
from sedtrails.transport_converter.plugins import BaseFormatPlugin
from sedtrails.transport_converter.sedtrails_data import SedtrailsData, SedtrailsMetadata
from sedtrails.transport_converter.time_utils import decompress_time_info


class FormatPlugin(BaseFormatPlugin):
    """
    Plugin for converting SFINCS NetCDF to SedTRAILS format.
    """

    SELECTABLE_FIELDS = frozenset(
        {
            'bed_level',
            'depth_avg_flow_velocity',
            'water_depth',
        }
    )

    _VARIABLE_MAP = {
        'bed_level': 'zb',
        'water_depth': 'h',
        'flow_velocity_x': 'u',
        'flow_velocity_y': 'v',
    }

    _FIELD_COMPONENT_KEYS = {
        'bed_level': ('bed_level',),
        'depth_avg_flow_velocity': ('flow_velocity_x', 'flow_velocity_y'),
        'water_depth': ('water_depth',),
    }

    def __init__(self, input_file: str, morfac: float = 1.0):
        """
        Initialize the plugin with the input file.

        Parameters:
        -----------
        input_file : str
            Path to the SFINCS NetCDF file.
        morfac : float, optional
            Morphological acceleration factor for time decompression. The
            default is 1.0, which leaves input times unchanged.
        """
        super().__init__()
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f'Input file not found: {self.input_file}')
        self.morfac = morfac
        self.input_data = None  # holds Dataset after reading
        self._input_variables: List[str] = []
        self.domain_config: Dict[str, Any] = {}
        self.coordinate_system: str | None = None
        self.source_crs: str | None = None
        self.metric_crs: str | None = None
        self._inner_boundary_polygons: list[np.ndarray] | None = None
        self._inner_boundary_polygons_signature: str | None = None
        self._last_inner_boundary_mask: SimpleNamespace | None = None
        self._active_face_mask_cache: dict[str, Any] | None = None
        self._active_face_center_triangles_cache: dict[str, Any] | None = None
        self._boundary_edge_classification_cache: dict[str, Any] | None = None
        self._mapped_static_geometry_cache: dict[str, Any] | None = None

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
                ds = self.input_data
                if isinstance(self.input_data, xu.UgridDataset):
                    ds = (
                        getattr(self.input_data, 'dataset', None)
                        or getattr(self.input_data, '_dataset', None)
                        or getattr(self.input_data, 'ds', None)
                    )
                    if ds is None:
                        raise ValueError('Underlying xarray Dataset not found on UgridDataset')
                self._input_variables = list(ds.data_vars)
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
        required_fields=None,
        max_memory_bytes: int | None = None,
    ) -> SedtrailsData:
        """
        SedtrailsData from SFINCS Netcdf.

        Parameters
        ----------
        current_time : float, optional
            Current simulation time in seconds
        reading_interval : float, optional
            Reading interval in seconds
        reference_date : np.datetime64, optional
            Reference date for converting time values
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
        mapped_data = self._map_sfincs_variables(
            time_info,
            time_start_idx,
            time_end_idx,
            required_fields=selected_fields,
        )
        seconds_since_ref = time_info['seconds_since_reference']
        self.reference_date = time_info['reference_date']

        depth_avg_flow_velocity = self._mapped_vector_field(
            mapped_data,
            'flow_velocity',
            selected='depth_avg_flow_velocity' in selected_fields,
        )

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
            water_depth=mapped_data.get('water_depth'),
            metadata=metadata,
            fractions=0,  # Default to 0 fractions
            # this has no meaning for SFINCS runs
            bed_load_transport=None,
            suspended_transport=None,
            mean_bed_shear_stress=None,
            max_bed_shear_stress=None,
            sediment_concentration=None,
            nonlinear_wave_velocity=None,
            node_x=mapped_data.get('node_x'),
            node_y=mapped_data.get('node_y'),
            face_node_connectivity=mapped_data.get('face_node_connectivity'),
            particle_face_connectivity=mapped_data.get('particle_face_connectivity'),
            face_node_fill_value=-1,
        )

        self._add_inner_boundary_metadata(sedtrails_data.metadata)
        self._add_boundary_edge_metadata(
            sedtrails_data.metadata,
            sedtrails_data.x,
            sedtrails_data.y,
            sedtrails_data.particle_face_connectivity,
        )

        return sedtrails_data

    def get_seeding_field_data(self):
        """Return active SFINCS geometry required for particle seeding.

        Returns
        -------
        types.SimpleNamespace
            Object with active face-centre ``x`` and ``y`` coordinates,
            triangular particle connectivity, optional
            ``boundary_edge_classification`` metadata, and
            ``face_node_fill_value``.

        Raises
        ------
        TypeError
            If a loaded UGRID dataset does not contain a ``Ugrid2d`` grid.
        KeyError
            If required mesh variables are missing in the xarray fallback path.
        FileNotFoundError
            If configured inner-boundary or boundary-class polygon files do
            not exist.
        ValueError
            If configured Tekal polygon blocks are malformed, or all SFINCS
            faces are masked by inner-boundary polygons.
        """
        self.load()
        x, y = self._active_face_coordinates()
        particle_triangles = self._active_face_center_triangles(x, y)
        boundary_edge_classification = self._boundary_edge_classification(x, y, particle_triangles)
        return SimpleNamespace(
            x=x,
            y=y,
            face_node_connectivity=particle_triangles,
            particle_face_connectivity=particle_triangles,
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
        return self._active_face_coordinates()

    def _face_coordinates(self):
        """Return SFINCS face centroid coordinates."""

        # Fast path: xugrid already provides face centroids.
        if isinstance(self.input_data, xu.UgridDataset):
            grid = self.input_data.grid
            if not isinstance(grid, xu.Ugrid2d):
                raise TypeError(f'Expected Ugrid2d, got {type(grid).__name__}')
            return np.asarray(grid.face_x), np.asarray(grid.face_y)

        node_x_var, node_y_var, face_nodes_var, start_index, fill_value = self._get_face_node_mesh_variables()

        return compute_face_centroids(
            node_x_var,
            node_y_var,
            face_nodes_var,
            start_index=start_index,
            fill_value=fill_value,
        )

    def _active_face_coordinates(self):
        """Return SFINCS face centroids after applying configured island masks."""
        face_x, face_y = self._face_coordinates()
        active_mask = self._active_face_mask(face_x, face_y)
        return np.asarray(face_x)[active_mask], np.asarray(face_y)[active_mask]

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

    def _decompress_time(self, time_info: Dict) -> Dict:
        """
        Apply morfac decompression to time values.

        Parameters
        ----------
        time_info : dict
            Time information returned by `_get_time_info`.

        Returns
        -------
        dict
            Time information with decompressed time values and seconds since
            reference.
        """
        return decompress_time_info(time_info, self.morfac)

    def load(self) -> Any:
        """
        Reads and loads a SFINCS NetCDF file using xugrid.

        Returns
        -------
        Any
            Requested value.
        """

        if self.input_data is None:
            try:
                # Keep arrays lazy; forcing windows materialize selected slices.
                self.input_data = xu.open_dataset(
                    self.input_file,
                    decode_times=True,
                    decode_timedelta=True,
                )
            except Exception as e:
                print(f'Could not open file with xugrid: {e} \n Trying with Xarray...')
                # Fallback to regular xarray
                try:
                    try:
                        self.input_data = xr.open_dataset(
                            self.input_file, decode_times=True, use_cftime=True, decode_timedelta=True
                        )
                    except TypeError:
                        # Older xarray versions may not support use_cftime
                        self.input_data = xr.open_dataset(self.input_file, decode_times=True, decode_timedelta=True)
                except Exception as e:
                    # If time decoding fails, retry without decoding
                    msg = str(e)
                    if 'decode time' in msg or 'decode_times' in msg or 'cftime' in msg:
                        try:
                            self.input_data = xr.open_dataset(
                                self.input_file, decode_times=False, decode_timedelta=False
                            )
                        except Exception as e:
                            raise IOError(f'Failed to open NetCDF file: {e}') from e
                        else:
                            print(f'Successfully loaded (Xarray, decode_times=False): {self.input_file}')
                    else:
                        raise IOError(f'Failed to open NetCDF file: {e}') from e
                else:
                    print(f'Successfully loaded (Xarray): {self.input_file}')
            else:
                print(f'Successfully loaded (Xugrid): {self.input_file}')

    def _get_variable(self, name: str):
        """
        Return an xarray DataArray for variable `name` for either xr.Dataset or xu.UgridDataset.
        """
        if self.input_data is None:
            raise ValueError('Dataset not loaded. Call load() first.')

        ds = self.input_data
        #   if isinstance(self.input_data, xu.UgridDataset):
        #       ds = (
        #           getattr(self.input_data, 'dataset', None)
        #           or getattr(self.input_data, '_dataset', None)
        #           or getattr(self.input_data, 'ds', None)
        #       )
        #       if ds is None:
        #           raise KeyError(f"Underlying xarray Dataset not found on UgridDataset for variable '{name}'")

        if name in ds:
            return ds[name]
        raise KeyError(f"Variable '{name}' not found in dataset")

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

        # Normalize time values if they are not already datetime-like
        if not np.issubdtype(np.asarray(time_values).dtype, np.datetime64):
            decoded = None
            if orig_units is not None:
                try:
                    decoded = xr.coding.times.decode_cf_datetime(time_values, orig_units, orig_calendar)
                except Exception:
                    decoded = None
            if decoded is not None:
                try:
                    decoded = xr.coding.times.cftime_to_nptime(decoded)
                except Exception:
                    pass
                time_values = decoded
            else:
                # Fallback: treat numeric values as seconds since reference_date
                time_values = reference_date + np.asarray(time_values, dtype=float) * np.timedelta64(1, 's')

        # Convert time values to seconds since reference_date
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

    def _map_sfincs_variables(
        self,
        time_info,
        time_start_idx: Optional[int] = None,
        time_end_idx: Optional[int] = None,
        required_fields=None,
    ) -> Dict:
        """
        Map SFINCS variables to SedtrailsData structure.

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

        selected_fields = self._selectable_required_fields(required_fields)
        # Get time information
        num_times = time_info['num_times']
        time_slice = (
            slice(time_start_idx, time_end_idx)
            if time_start_idx is not None or time_end_idx is not None
            else slice(None)
        )

        static_geometry = self._mapped_static_geometry()
        active_face_mask = static_geometry['active_face_mask']
        data = {
            key: static_geometry[key]
            for key in (
                'x',
                'y',
                'node_x',
                'node_y',
                'face_node_connectivity',
                'particle_face_connectivity',
            )
        }

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
            try:
                var = self._get_variable(var_name)
            except KeyError:
                # Default to zeros if not found
                data[key] = np.zeros((num_times, *grid_shape))
                print(f"Warning: Variable '{var_name}' not found, using zeros")
                continue

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

            data[key] = self._filter_face_field(data[key], active_face_mask)

        return data

    def _mapped_static_geometry(self) -> dict[str, np.ndarray]:
        """Return cached static SFINCS geometry used by forcing windows."""
        node_x_var, node_y_var, face_nodes_var, start_index, fill_value = (
            self._get_face_node_mesh_variables()
        )
        signature = self._mapped_static_geometry_signature(
            node_x_var,
            node_y_var,
            face_nodes_var,
            start_index,
            fill_value,
        )
        cache = self._mapped_static_geometry_cache
        if cache is not None and cache.get('signature') == signature:
            self._last_inner_boundary_mask = cache['mask_result']
            return cache

        face_node_connectivity = normalize_face_node_connectivity(
            face_nodes_var,
            start_index=start_index,
            fill_value=fill_value,
        )
        face_x, face_y = compute_face_centroids(
            node_x_var,
            node_y_var,
            face_nodes_var,
            start_index=start_index,
            fill_value=fill_value,
            normalized_connectivity=face_node_connectivity,
        )
        active_face_mask = self._active_face_mask(face_x, face_y)
        if np.all(active_face_mask):
            active_x = np.asarray(face_x)
            active_y = np.asarray(face_y)
            active_connectivity = face_node_connectivity
        else:
            active_x = np.asarray(face_x)[active_face_mask]
            active_y = np.asarray(face_y)[active_face_mask]
            active_connectivity = face_node_connectivity[active_face_mask]

        cache = {
            'signature': signature,
            'x': active_x,
            'y': active_y,
            'node_x': np.asarray(node_x_var),
            'node_y': np.asarray(node_y_var),
            'face_node_connectivity': active_connectivity,
            'particle_face_connectivity': self._active_face_center_triangles(
                active_x,
                active_y,
                source_grid=(
                    np.asarray(node_x_var),
                    np.asarray(node_y_var),
                    face_node_connectivity,
                ),
            ),
            'active_face_mask': active_face_mask,
            'mask_result': self._last_inner_boundary_mask,
        }
        self._mapped_static_geometry_cache = cache
        return cache

    def _mapped_static_geometry_signature(
        self,
        node_x_var,
        node_y_var,
        face_nodes_var,
        start_index,
        fill_value,
    ) -> tuple:
        """Return an identity signature for static source geometry and masking."""
        return (
            id(self.input_data),
            self._geometry_source_token(node_x_var),
            self._geometry_source_token(node_y_var),
            self._geometry_source_token(face_nodes_var),
            int(start_index),
            None if fill_value is None else int(fill_value),
            self._domain_config_signature(),
            self._coordinate_system(),
            self.source_crs,
            self.metric_crs,
            getattr(self, 'runtime_geometry', 'planar'),
        )

    @staticmethod
    def _geometry_source_token(values) -> tuple:
        """Return a stable token for one lazy or in-memory geometry source."""
        variable = getattr(values, 'variable', None)
        storage = getattr(variable, '_data', None) if variable is not None else None
        if storage is None:
            storage = values
        return (
            id(storage),
            tuple(getattr(values, 'shape', ())),
            str(getattr(values, 'dtype', '')),
        )

    def _active_face_mask(self, face_x: np.ndarray, face_y: np.ndarray) -> np.ndarray:
        """Return faces whose centroids are outside configured inner-boundary polygons."""
        x = np.asarray(face_x, dtype=float).ravel()
        y = np.asarray(face_y, dtype=float).ravel()
        if x.shape != y.shape:
            raise ValueError(f'face_x and face_y must have the same shape, got {x.shape} and {y.shape}')

        coordinate_system = self._coordinate_system()
        cache = self._active_face_mask_cache
        if self._geometry_cache_matches(cache, x, y):
            self._last_inner_boundary_mask = cache['mask_result']
            return cache['active_mask']

        polygons = self._get_inner_boundary_polygons()
        if not polygons:
            active_mask = np.ones(x.shape[0], dtype=bool)
            removed_count = 0
        else:
            transform = build_coordinate_transform(
                x,
                y,
                coordinate_system,
                source_crs=self.source_crs,
                metric_crs=self.metric_crs,
            )
            metric_x, metric_y = transform.source_to_metric(x, y)
            inside = points_inside_any_polygon(
                np.column_stack((metric_x, metric_y)),
                transform.polygons_to_metric(polygons),
            )
            active_mask = ~inside
            removed_count = int(np.count_nonzero(inside))

        active_count = int(np.count_nonzero(active_mask))
        self._last_inner_boundary_mask = SimpleNamespace(
            active_mask=active_mask,
            removed_count=removed_count,
            active_count=active_count,
        )
        if x.size and active_count == 0:
            raise ValueError('All SFINCS faces were masked by domain.inner_boundary_pol_files')
        self._active_face_mask_cache = {
            'node_x': x,
            'node_y': y,
            'domain_signature': self._domain_config_signature(),
            'coordinate_system': coordinate_system,
            'source_crs': self.source_crs,
            'metric_crs': self.metric_crs,
            'active_mask': active_mask,
            'mask_result': self._last_inner_boundary_mask,
        }
        return active_mask

    def _active_face_center_triangles(
        self,
        face_x: np.ndarray,
        face_y: np.ndarray,
        source_grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Build active particle-location triangles over SFINCS face-centre coordinates."""
        coordinate_system = self._coordinate_system()
        cache = self._active_face_center_triangles_cache
        if self._geometry_cache_matches(cache, face_x, face_y):
            return cache['triangles']

        candidate_connectivity = self._source_face_center_triangles(
            face_x,
            face_y,
            source_grid=source_grid,
        )
        if candidate_connectivity is None:
            candidate_connectivity = delaunay_connectivity(
                face_x,
                face_y,
                coordinate_system=coordinate_system,
                source_crs=self.source_crs,
                metric_crs=self.metric_crs,
                runtime_geometry=getattr(self, 'runtime_geometry', 'planar'),
            )
        triangles = filter_connectivity_by_inner_polygons(
            face_x,
            face_y,
            candidate_connectivity,
            self._get_inner_boundary_polygons(),
            coordinate_system=coordinate_system,
            source_crs=self.source_crs,
            metric_crs=self.metric_crs,
            runtime_geometry=getattr(self, 'runtime_geometry', 'planar'),
        ).connectivity
        self._active_face_center_triangles_cache = {
            'node_x': np.asarray(face_x),
            'node_y': np.asarray(face_y),
            'domain_signature': self._domain_config_signature(),
            'coordinate_system': coordinate_system,
            'source_crs': self.source_crs,
            'metric_crs': self.metric_crs,
            'triangles': triangles,
        }
        return triangles

    def _source_face_center_triangles(
        self,
        active_face_x: np.ndarray,
        active_face_y: np.ndarray,
        source_grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray | None:
        """Derive face-centre triangles from authoritative UGRID topology.

        Parameters
        ----------
        active_face_x, active_face_y : np.ndarray
            Face-centre coordinates after applying the active face mask.

        Returns
        -------
        np.ndarray or None
            Triangles indexing the active face-centre arrays. ``None`` means
            that the source topology does not contain enough shared-node
            information and the bounded geometric fallback is required.
        """
        mask_cache = self._active_face_mask_cache
        if (
            mask_cache is not None
            and mask_cache.get('domain_signature') == self._domain_config_signature()
        ):
            full_face_x = mask_cache['node_x']
            full_face_y = mask_cache['node_y']
            active_mask = mask_cache['active_mask']
        else:
            try:
                full_face_x, full_face_y = self._face_coordinates()
                active_mask = self._active_face_mask(full_face_x, full_face_y)
            except (KeyError, TypeError, ValueError):
                return None

        active_x = np.asarray(active_face_x)
        active_y = np.asarray(active_face_y)
        if not self._masked_coordinates_equal(
            full_face_x,
            full_face_y,
            active_mask,
            active_x,
            active_y,
        ):
            return None

        try:
            if isinstance(self.input_data, xu.UgridDataset):
                grid = self.input_data.grid
                if not isinstance(grid, xu.Ugrid2d):
                    return None
            elif source_grid is not None:
                source_node_x, source_node_y, source_faces = source_grid
                grid = create_ugrid2d(
                    source_node_x,
                    source_node_y,
                    source_faces,
                    is_projected=self._coordinate_system() != 'geographic',
                )
            else:
                node_x_var, node_y_var, face_nodes_var, start_index, fill_value = (
                    self._get_face_node_mesh_variables()
                )
                source_faces = normalize_face_node_connectivity(
                    face_nodes_var,
                    start_index=start_index,
                    fill_value=fill_value,
                )
                grid = create_ugrid2d(
                    np.asarray(node_x_var),
                    np.asarray(node_y_var),
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

        source_triangles = source_face_index[centroid_triangles]
        valid = np.all(
            (source_triangles >= 0) & (source_triangles < active_mask.size),
            axis=1,
        )
        valid &= source_triangles[:, 0] != source_triangles[:, 1]
        valid &= source_triangles[:, 0] != source_triangles[:, 2]
        valid &= source_triangles[:, 1] != source_triangles[:, 2]
        if not np.all(valid):
            source_triangles = source_triangles[valid]
        if source_triangles.size == 0:
            return None

        if np.all(active_mask):
            return source_triangles

        active_triangles = np.empty(source_triangles.shape[0], dtype=bool)
        chunk_size = 262_144
        for start in range(0, source_triangles.shape[0], chunk_size):
            stop = min(start + chunk_size, source_triangles.shape[0])
            active_triangles[start:stop] = np.all(
                active_mask[source_triangles[start:stop]],
                axis=1,
            )
        source_triangles = source_triangles[active_triangles]
        if source_triangles.size == 0:
            return np.empty((0, 3), dtype=np.int64)

        active_source_faces = np.flatnonzero(active_mask)
        return np.searchsorted(active_source_faces, source_triangles).astype(np.int64, copy=False)

    @staticmethod
    def _masked_coordinates_equal(
        full_x: np.ndarray,
        full_y: np.ndarray,
        active_mask: np.ndarray,
        active_x: np.ndarray,
        active_y: np.ndarray,
    ) -> bool:
        """Compare masked coordinates without a full-size temporary copy."""
        full_x_array = np.asarray(full_x)
        full_y_array = np.asarray(full_y)
        if (
            full_x_array.shape != active_mask.shape
            or full_y_array.shape != active_mask.shape
            or active_x.shape != active_y.shape
            or int(np.count_nonzero(active_mask)) != active_x.size
        ):
            return False

        active_offset = 0
        chunk_size = 262_144
        for start in range(0, active_mask.size, chunk_size):
            stop = min(start + chunk_size, active_mask.size)
            block_mask = active_mask[start:stop]
            block_count = int(np.count_nonzero(block_mask))
            active_stop = active_offset + block_count
            if not np.array_equal(
                full_x_array[start:stop][block_mask],
                active_x[active_offset:active_stop],
            ):
                return False
            if not np.array_equal(
                full_y_array[start:stop][block_mask],
                active_y[active_offset:active_stop],
            ):
                return False
            active_offset = active_stop
        return True

    def _filter_face_field(self, field_value: np.ndarray, active_face_mask: np.ndarray) -> np.ndarray:
        """Filter arrays with a trailing face dimension by the active-face mask."""
        values = np.asarray(field_value)
        if values.shape and values.shape[-1] == active_face_mask.size:
            return values[..., active_face_mask]
        return values

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
            metadata.add('inner_boundary_active_face_count', self._last_inner_boundary_mask.active_count)

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
        """Return the coordinate-system label inferred from SFINCS mesh coordinates."""
        if self.coordinate_system is not None and str(self.coordinate_system).lower() != 'auto':
            return str(self.coordinate_system)
        if self.input_data is None:
            return 'projected'
        try:
            x_var = self._get_variable('mesh2d_node_x')
            y_var = self._get_variable('mesh2d_node_y')
        except Exception:
            return 'projected'
        return infer_coordinate_system_from_attrs(x_var, y_var)

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

    def _get_face_node_mesh_variables(self):
        """Return UGRID node coordinates and face-node connectivity with indexing metadata."""
        node_x_var = self._get_variable('mesh2d_node_x')
        node_y_var = self._get_variable('mesh2d_node_y')

        try:
            face_nodes_var = self._get_variable('mesh2d_face_nodes')
        except KeyError as err:
            if not isinstance(self.input_data, xu.UgridDataset):
                raise
            grid = self.input_data.grid
            if not isinstance(grid, xu.Ugrid2d):
                raise TypeError(f'Expected Ugrid2d, got {type(grid).__name__}') from err
            face_nodes_var = grid.face_node_connectivity
            return node_x_var, node_y_var, face_nodes_var, 0, -1

        start_index = face_nodes_var.attrs.get('start_index', 0)
        fill_value = face_nodes_var.encoding.get('_FillValue', face_nodes_var.attrs.get('_FillValue', -1))
        return node_x_var, node_y_var, face_nodes_var, start_index, fill_value

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
                f'SFINCS forcing requires at least {required_bytes} bytes for '
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

        spatial_size = self._source_face_count()
        fallback_bytes = spatial_size * np.dtype(np.float64).itemsize
        total_bytes = 0
        for field_name in selected_fields:
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
        try:
            variable = self._get_variable(variable_name)
        except KeyError:
            return fallback_count, np.dtype(np.float64)
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

    def _source_face_count(self) -> int:
        """Return the source face count without materializing connectivity."""
        try:
            face_nodes = self._get_variable('mesh2d_face_nodes')
        except KeyError:
            if isinstance(self.input_data, xu.UgridDataset):
                return int(self.input_data.grid.n_face)
            return 0

        if face_nodes.ndim != 2:
            return 0
        first, second = face_nodes.shape
        if first <= 8 and second > 8:
            return int(second)
        return int(first)


def normalize_face_node_connectivity(mesh2d_face_nodes, start_index=1, fill_value=-999):
    """
    Return zero-based face-node connectivity with invalid entries set to -1.

    Parameters
    ----------
    mesh2d_face_nodes : object
        Face-node connectivity array.
    start_index : object
        Index base used by the face-node connectivity.
    fill_value : object
        Padding value used for missing face nodes.

    Returns
    -------
    np.ndarray
        Zero-based face-node connectivity with invalid entries set to -1.
    """
    faces = np.asarray(mesh2d_face_nodes)
    start_index = int(start_index)
    if (
        np.issubdtype(faces.dtype, np.signedinteger)
        and start_index == 0
        and (fill_value is None or int(fill_value) < 0)
    ):
        return faces

    output_dtype = faces.dtype if np.issubdtype(faces.dtype, np.signedinteger) else np.int64
    normalized = faces.astype(output_dtype, copy=True)
    normalized -= start_index
    invalid = normalized < 0
    if fill_value is not None:
        invalid |= faces == fill_value
    normalized[invalid] = -1
    return normalized


def compute_face_centroids(
    mesh2d_node_x,
    mesh2d_node_y,
    mesh2d_face_nodes,
    start_index=1,
    fill_value=-999,
    normalized_connectivity=None,
):
    """
    Compute face centroids from UGRID mesh node coordinates and face->node connectivity.

    Parameters
    ----------
    mesh2d_node_x : array-like, shape (nnode,)
    mesh2d_node_y : array-like, shape (nnode,)
    mesh2d_face_nodes : array-like, shape (nface, max_nodes_per_face)
        Node indices per face; may be padded with fill_value.
    start_index : int
        0 or 1 depending on file convention.
    fill_value : int
        Padding value in mesh2d_face_nodes.
    normalized_connectivity : array-like, optional
        Pre-normalized zero-based connectivity. Supplying it avoids repeating
        normalization when static topology is already cached.

    Returns
    -------
    face_x : np.ndarray, shape (nface,)
    face_y : np.ndarray, shape (nface,)
    """
    node_x = np.asarray(mesh2d_node_x)
    node_y = np.asarray(mesh2d_node_y)
    if normalized_connectivity is None:
        faces = normalize_face_node_connectivity(
            mesh2d_face_nodes,
            start_index=start_index,
            fill_value=fill_value,
        )
    else:
        faces = np.asarray(normalized_connectivity)

    face_x = np.full(faces.shape[0], np.nan, dtype=np.float64)
    face_y = np.full(faces.shape[0], np.nan, dtype=np.float64)
    usable_node_count = min(node_x.size, node_y.size)
    if usable_node_count == 0:
        return face_x, face_y

    chunk_size = 262_144
    for start in range(0, faces.shape[0], chunk_size):
        stop = min(start + chunk_size, faces.shape[0])
        block = faces[start:stop]
        valid = (block >= 0) & (block < usable_node_count)
        counts = np.count_nonzero(valid, axis=1)
        populated = counts > 0
        if not np.any(populated):
            continue
        clipped = np.clip(block, 0, usable_node_count - 1)
        block_x = np.where(valid, node_x[clipped], 0.0)
        block_y = np.where(valid, node_y[clipped], 0.0)
        face_x[start:stop][populated] = (
            np.sum(block_x[populated], axis=1) / counts[populated]
        )
        face_y[start:stop][populated] = (
            np.sum(block_y[populated], axis=1) / counts[populated]
        )

    return face_x, face_y
