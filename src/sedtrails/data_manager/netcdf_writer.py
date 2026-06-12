"""
NetCDF Writer
=============

Writes NetCDF files produced by the SedTrails Particle Tracer System.

"""

import logging
import netCDF4 as nc4
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Proactively close+reopen the streaming file every N successful writes to
# refresh the OS file descriptor. This matters on network drives (SMB/NFS) where
# long-lived HDF5 handles become stale after network reconnects or idle timeouts.
_REOPEN_INTERVAL = 10


class NetCDFWriter:
    """
    Streaming NetCDF writer for SedTRAILS particle trajectory output.

    The writer keeps one NetCDF4 handle open and writes one saved trajectory
    slot at a time. It deliberately avoids building an in-memory xarray Dataset
    for the full particle/time cube.

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

        # Dimensions: n_timesteps is fixed (pre-allocated, no copy work on writes)
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

        # Time-varying trajectory variables: unwritten slots stay at fill value
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
        # Proactive reopen prevents stale FD on long-running network-drive writes.
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
                'HDF/network error writing slot %d: %r; reopening output file and retrying',
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
