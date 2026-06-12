"""
NetCDF Writer
=============

Writes NetCDF files produced by the SedTrails Particle Tracer System.

"""

import logging
import netCDF4 as nc4
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from .xarray_dataset import create_sedtrails_dataset, populate_population_metadata, populate_flowfield_metadata

logger = logging.getLogger(__name__)

# Proactively close+reopen the streaming file every N successful writes to
# refresh the OS file descriptor — critical on network drives (SMB/NFS) where
# long-lived HDF5 handles become stale after network reconnects or idle timeouts.
_REOPEN_INTERVAL = 10


class NetCDFWriter:
    """
    A class for writing NetCDF files for the SedTrails Particle Tracer System using xarray.

    This class provides methods to:
    - Create xarray datasets with SedTrails structure
    - Add metadata to datasets (populations, flow fields, simulation info)
    - Write xarray datasets to NetCDF files with optional timestep trimming

    Examples
    --------

    # Method 1: Step-by-step approach
    output_dir = "some/path"
    writer = NetCDFWriter(output_dir)

    # Create the xarray dataset
    dataset = writer.create_dataset(
    N_particles=total_particles,
    N_populations=n_populations,
    N_timesteps=n_timesteps,
    N_flowfields=n_flowfields
    )

    # Populate metadata to the dataset
    writer.add_metadata(dataset, populations, flow_field_names)

    # Write the dataset to a netcdf file
    writer.write(dataset, filename)

    # Optional: Write with timestep trimming
    writer.write(dataset, filename, trim_to_actual_timesteps=True, actual_timesteps=100)

    # Method 2: All-in-one approach
    writer.create_and_write_simulation_results(
    populations, flow_field_names, n_timesteps, filename
    )

    Attributes
    ----------
    output_dir : pathlib.Path
        The directory where output files (NetCDF, images, etc.) are stored.

    """

    def __init__(self, output_dir):
        output_dir = Path(output_dir)
        # If the output directory already exists, we add a timestamp to avoid overwriting
        # if output_dir.exists():
        #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        #     output_dir = output_dir.parent / f'{output_dir.name}_{timestamp}'
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_filename(self, filename):
        """
        Validates the file name to ensure it is a NetCDF file.
        """
        if not filename.endswith('.nc'):
            raise ValueError('Output file must have a .nc extension.')

    def write(self, xr_dataset, filename, trim_to_actual_timesteps=False, actual_timesteps=None):
        """
        Write an xr.Dataset to a NetCDF file in the output directory.

        Parameters
        ----------
        xr_dataset : xr.Dataset
            The xarray dataset to write.
        filename : str
            The name of the NetCDF file to write (should end with .nc).
        trim_to_actual_timesteps : bool, optional
            Whether to trim the dataset to actual timesteps used (default: False)
        actual_timesteps : int, optional
            Number of actual timesteps to keep (if None, tries to determine automatically)

        Returns
        -------
        pathlib.Path
            Path to the written file

        """
        self._validate_filename(filename)
        output_path = self.output_dir / filename
        if not isinstance(xr_dataset, xr.Dataset):
            raise TypeError('Input must be an xr.Dataset.')

        output_dataset = xr_dataset

        # Handle trimming if requested
        if trim_to_actual_timesteps:
            if actual_timesteps is None:
                # Try to determine actual timesteps by finding the last non-NaN time value
                if 'time' in xr_dataset and 'n_timesteps' in xr_dataset.dims:
                    time_data = xr_dataset['time'].values
                    if not np.all(np.isnan(time_data)):
                        # Find the last timestep with any non-NaN values
                        non_nan_mask = ~np.isnan(time_data)
                        if np.any(non_nan_mask):
                            last_timestep = np.max(np.where(non_nan_mask)[1]) + 1
                            actual_timesteps = last_timestep

            if actual_timesteps is not None and 'n_timesteps' in xr_dataset.dims:
                output_dataset = xr_dataset.isel(n_timesteps=slice(0, actual_timesteps))

        # Add standard metadata if not present
        if 'title' not in output_dataset.attrs:
            output_dataset.attrs['title'] = 'SedTrails Particle Simulation Results'
        if 'institution' not in output_dataset.attrs:
            output_dataset.attrs['institution'] = 'SedTrails Particle Tracer System'
        if 'created_on' not in output_dataset.attrs:
            output_dataset.attrs['created_on'] = datetime.now().isoformat()

        output_dataset.to_netcdf(output_path)
        return output_path

    def create_dataset(self, N_particles, N_populations, N_timesteps, N_flowfields, name_strlen=24):
        """
        Create an xarray dataset with the SedTrails structure.

        Parameters
        ----------
        N_particles : int
            Number of particles
        N_populations : int
            Number of populations
        N_timesteps : int
            Number of timesteps
        N_flowfields : int
            Number of flow fields
        name_strlen : int, optional
            Maximum length for string variables (default: 24)

        Returns
        -------
        xr.Dataset
            The created xarray dataset
        """
        return create_sedtrails_dataset(
            N_particles=N_particles,
            N_populations=N_populations,
            N_timesteps=N_timesteps,
            N_flowfields=N_flowfields,
            name_strlen=name_strlen,
        )

    def add_metadata(self, dataset, populations, flow_field_names, simulation_metadata=None):
        """
        Add metadata to the xarray dataset.

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to add metadata to
        populations : list
            List of population objects from the simulation
        flow_field_names : list
            List of flow field names used in the simulation
        simulation_metadata : dict, optional
            Additional simulation metadata to include as global attributes

        Returns
        -------
        xr.Dataset
            The dataset with metadata added (modifies in place and returns)
        """
        # Populate population and flow field metadata
        populate_population_metadata(dataset, populations)
        if flow_field_names:
            populate_flowfield_metadata(dataset, flow_field_names)

        # Add simulation metadata as global attributes if provided
        if simulation_metadata:
            dataset.attrs.update(simulation_metadata)

        # Add standard metadata
        dataset.attrs['title'] = 'SedTRAILS Particle Simulation Results'
        dataset.attrs['institution'] = 'SedTRAILS Particle Tracer System'
        dataset.attrs['created_on'] = datetime.now().isoformat()

        return dataset

    def create_and_write_simulation_results(
        self,
        populations,
        flow_field_names,
        N_timesteps,
        filename='simulation_results.nc',
        simulation_metadata=None,
        name_strlen=24,
    ):
        """
        Convenience method that creates dataset, adds metadata, and writes to file in one call.

        Parameters
        ----------
        populations : list
            List of population objects from the simulation
        flow_field_names : list
            List of flow field names used in the simulation
        N_timesteps : int
            Number of timesteps in the simulation
        filename : str, optional
            The name of the NetCDF file to write (default: 'simulation_results.nc')
        simulation_metadata : dict, optional
            Additional simulation metadata to include as global attributes
        name_strlen : int, optional
            Maximum length for string variables (default: 24)

        Returns
        -------
        pathlib.Path
            Path to the written file

        """
        # Calculate dimensions
        total_particles = sum([len(pop.particles['x']) for pop in populations])
        n_populations = len(populations)
        n_flowfields = len(flow_field_names) if flow_field_names else 1

        # Create dataset
        dataset = self.create_dataset(
            N_particles=total_particles,
            N_populations=n_populations,
            N_timesteps=N_timesteps,
            N_flowfields=n_flowfields,
            name_strlen=name_strlen,
        )

        # Add metadata
        self.add_metadata(dataset, populations, flow_field_names, simulation_metadata)

        # Write to file
        return self.write(dataset, filename)


    def open_output(
        self,
        filename: str,
        n_slots: int,
        N_particles: int,
        N_populations: int,
        N_flowfields: int,
        populations: list,
        flow_field_names: list,
        name_strlen: int = 24,
    ):
        """
        Open a streaming output file with pre-allocated dimensions.

        Creates the file, defines all dimensions and variables, writes static
        population and flow-field metadata, and returns the open handle.
        The file is kept open throughout the simulation; call ``close_output()``
        when done (ideally inside a ``try/finally`` block).

        Parameters
        ----------
        filename : str
            Name of the NetCDF file to create (must end with ``.nc``).
        n_slots : int
            Number of time slots to pre-allocate (one per ``save_interval``
            boundary plus one for the initial state).
        N_particles : int
            Total number of particles across all populations.
        N_populations : int
            Number of particle populations.
        N_flowfields : int
            Number of flow-field tracers.
        populations : list
            Population objects; used to write static metadata (name, type, count).
        flow_field_names : list
            Names of the flow fields; written as static metadata.
        name_strlen : int, optional
            Maximum character length for string variables (default 24).

        Returns
        -------
        netCDF4.Dataset
            Open file handle for use with ``record_output()`` and ``close_output()``.
        """
        self._validate_filename(filename)
        output_path = self.output_dir / filename

        ds = nc4.Dataset(str(output_path), 'w', format='NETCDF4')

        # Dimensions — n_timesteps is fixed (pre-allocated, no copy work on writes)
        ds.createDimension('n_particles', N_particles)
        ds.createDimension('n_populations', N_populations)
        ds.createDimension('n_timesteps', n_slots)
        ds.createDimension('n_flowfields', N_flowfields)
        ds.createDimension('name_strlen', name_strlen)

        # Global attributes
        ds.title = 'SedTRAILS Particle Simulation Results'
        ds.institution = 'SedTRAILS Particle Tracer System'
        ds.created_on = datetime.now().isoformat()

        # Static metadata variables
        ds.createVariable('population_name', 'S1', ('n_populations', 'name_strlen'))
        ds.createVariable('population_particle_type', 'i4', ('n_populations',))
        ds.createVariable('population_start_idx', 'i4', ('n_populations',))
        ds.createVariable('population_count', 'i4', ('n_populations',))
        ds.createVariable('population_repr_volume', 'f8', ('n_populations',))
        ds.createVariable('trajectory_id', 'S1', ('n_particles', 'name_strlen'))
        ds.createVariable('population_id', 'i4', ('n_particles',))
        ds.createVariable('flowfield_name', 'S1', ('n_flowfields', 'name_strlen'))

        # Time-varying trajectory variables — unwritten slots stay at fill value
        ds.createVariable('time', 'f8', ('n_particles', 'n_timesteps'), fill_value=np.nan)
        ds.createVariable('x', 'f8', ('n_particles', 'n_timesteps'), fill_value=np.nan)
        ds.createVariable('y', 'f8', ('n_particles', 'n_timesteps'), fill_value=np.nan)
        ds.createVariable('z', 'f8', ('n_particles', 'n_timesteps'), fill_value=np.nan)
        ds.createVariable('burial_depth', 'f8', ('n_particles', 'n_timesteps'), fill_value=np.nan)
        ds.createVariable('mixing_depth', 'f8', ('n_particles', 'n_timesteps'), fill_value=np.nan)
        ds.createVariable('status_alive', 'i4', ('n_particles', 'n_timesteps'), fill_value=-1)
        ds.createVariable('status_buried', 'i4', ('n_particles', 'n_timesteps'), fill_value=-1)
        ds.createVariable('status_domain', 'i4', ('n_particles', 'n_timesteps'), fill_value=-1)
        ds.createVariable('status_transported', 'i4', ('n_particles', 'n_timesteps'), fill_value=-1)
        ds.createVariable('status_released', 'i4', ('n_particles', 'n_timesteps'), fill_value=-1)
        ds.createVariable('status_mobile', 'i4', ('n_particles', 'n_timesteps'), fill_value=-1)
        ds.createVariable('covered_distance', 'f8', ('n_flowfields', 'n_particles', 'n_timesteps'), fill_value=np.nan)

        # Write static population metadata
        particle_offset = 0
        for pop_idx, population in enumerate(populations):
            pop_name = getattr(population, 'name', f'population_{pop_idx}')
            ds['population_name'][pop_idx, :] = np.array(
                list(pop_name[:name_strlen].ljust(name_strlen)), dtype='S1'
            )
            ds['population_particle_type'][pop_idx] = int(getattr(population, 'particle_type', 0))
            ds['population_start_idx'][pop_idx] = particle_offset
            n_part = len(population.particles['x'])
            ds['population_count'][pop_idx] = n_part
            repr_vol = getattr(population, 'repr_volume', np.nan)
            ds['population_repr_volume'][pop_idx] = float(repr_vol) if repr_vol is not None else np.nan
            ds['population_id'][particle_offset:particle_offset + n_part] = pop_idx

            for i in range(n_part):
                traj_id = f'traj_{particle_offset + i}'
                ds['trajectory_id'][particle_offset + i, :] = np.array(
                    list(traj_id[:name_strlen].ljust(name_strlen)), dtype='S1'
                )
            particle_offset += n_part

        # Write flow-field metadata
        if flow_field_names:
            for ff_idx, ff_name in enumerate(flow_field_names[:N_flowfields]):
                ds['flowfield_name'][ff_idx, :] = np.array(
                    list(ff_name[:name_strlen].ljust(name_strlen)), dtype='S1'
                )

        # Store path so record_output can reopen on network/HDF errors
        self._streaming_path = str(output_path)
        self._write_count = 0

        ds.sync()
        return ds

    @staticmethod
    def _write_slot(h, populations: list, slot_idx: int, current_time: float) -> None:
        """Write one save-interval slot to an open netCDF4 handle."""
        particle_offset = 0
        for population in populations:
            num_particles = len(population.particles['x'])
            sl = slice(particle_offset, particle_offset + num_particles)

            h['time'][sl, slot_idx] = current_time
            h['x'][sl, slot_idx] = np.asarray(population.particles['x'])
            h['y'][sl, slot_idx] = np.asarray(population.particles['y'])
            h['z'][sl, slot_idx] = np.asarray(
                population.particles.get('z', np.zeros(num_particles))
            )
            h['burial_depth'][sl, slot_idx] = np.asarray(population.particles['burial_depth'])
            h['mixing_depth'][sl, slot_idx] = np.asarray(
                population.particles.get('mixing_depth', np.full(num_particles, np.nan))
            )
            h['status_alive'][sl, slot_idx] = np.asarray(
                population.particles.get('status_alive', np.ones(num_particles, dtype=np.int32))
            )
            h['status_buried'][sl, slot_idx] = np.asarray(
                population.particles.get('status_buried', np.zeros(num_particles, dtype=np.int32))
            )
            h['status_domain'][sl, slot_idx] = np.asarray(
                population.particles.get('status_domain', np.ones(num_particles, dtype=np.int32))
            )
            h['status_transported'][sl, slot_idx] = np.asarray(
                population.particles.get('status_transported', np.zeros(num_particles, dtype=np.int32))
            )
            h['status_released'][sl, slot_idx] = np.asarray(
                population.particles.get('status_released', np.ones(num_particles, dtype=np.int32))
            )
            h['status_mobile'][sl, slot_idx] = np.asarray(population.particles['status_mobile'])

            particle_offset += num_particles

        h.sync()

    def _reopen_handle(self, nc_handle) -> 'nc4.Dataset':
        """Close a broken/stale handle and reopen the file in read-write mode."""
        path = getattr(self, '_streaming_path', None) or nc_handle.filepath()
        try:
            nc_handle.close()
        except Exception:
            pass
        return nc4.Dataset(str(path), 'r+', format='NETCDF4')

    def record_output(
        self, nc_handle, populations: list, slot_idx: int, current_time: float
    ) -> 'nc4.Dataset':
        """
        Write current particle state to one time slot in the streaming output file.

        Syncs to disk after every write so data is safe even if the process is
        interrupted mid-simulation.

        On network (SMB/NFS) drives the HDF5 file descriptor can go stale after
        a reconnect or server-side idle timeout.  Two defences are applied:

        1. **Proactive reopen** every ``_REOPEN_INTERVAL`` writes refreshes the OS
           file descriptor before it can go stale.
        2. **Reactive reopen**: on any HDF/IO ``RuntimeError`` or ``OSError`` the
           handle is closed, the file is reopened in ``'r+'`` mode, and the write
           is retried once.

        Parameters
        ----------
        nc_handle : netCDF4.Dataset
            Open handle returned by ``open_output()`` or a previous call to this
            method.  May be silently replaced by a fresh handle on reopen.
        populations : list
            Current population objects whose particle state will be written.
        slot_idx : int
            Zero-based index of the time slot to write into.
        current_time : float
            Simulation time (seconds) to store in the ``time`` variable.

        Returns
        -------
        netCDF4.Dataset
            The (possibly refreshed) file handle.  **Callers must reassign**:
            ``nc_handle = writer.record_output(nc_handle, ...)``
        """
        # Proactive reopen — prevents stale FD on long-running network-drive writes
        self._write_count = getattr(self, '_write_count', 0) + 1
        if self._write_count % _REOPEN_INTERVAL == 0:
            logger.debug('Proactive netCDF handle reopen at slot %d (write #%d)',
                         slot_idx, self._write_count)
            nc_handle = self._reopen_handle(nc_handle)

        # Attempt write; on HDF/IO error reopen and retry once
        try:
            self._write_slot(nc_handle, populations, slot_idx, current_time)
        except (RuntimeError, OSError) as exc:
            err_str = str(exc)
            if not any(kw in err_str for kw in ('HDF', 'NetCDF', 'errno', 'I/O')):
                raise
            logger.warning(
                'HDF/network error writing slot %d: %r — reopening output file and retrying',
                slot_idx, exc,
            )
            nc_handle = self._reopen_handle(nc_handle)
            self._write_slot(nc_handle, populations, slot_idx, current_time)
            logger.info('Retry succeeded for slot %d', slot_idx)

        return nc_handle

    def close_output(self, nc_handle) -> Path:
        """
        Close the streaming output file and return its path.

        Parameters
        ----------
        nc_handle : netCDF4.Dataset
            Open handle returned by ``open_output()``.

        Returns
        -------
        pathlib.Path
            Absolute path to the closed NetCDF file.
        """
        path = Path(nc_handle.filepath())
        nc_handle.close()
        return path


if __name__ == '__main__':
    pass
