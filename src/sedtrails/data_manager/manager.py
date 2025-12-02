from sedtrails.data_manager.simulation_buffer import SimulationDataBuffer
from sedtrails.data_manager.memory_manager import MemoryManager
from sedtrails.data_manager.netcdf_writer import NetCDFWriter
from sedtrails.data_manager.xarray_dataset import collect_timestep_data
import logging
import numpy as np
from pathlib import Path
from typing import Union


# Mesh setup. # TODO: Can this be a default? I don't see the need to
NODE_X = np.array([0, 1, 1, 0])
NODE_Y = np.array([0, 0, 1, 1])
FACE_NODE_CONNECTIVITY = np.array([[0, 1, 2, 3]])
FILL_VALUE = -1


class DataManager:
    """
    A class to manage data and files produced by SedTrails, including simulation data buffering,
    memory checks, and SedTrails-specific NetCDF output.

    This class provides a unified interface for:
    - Traditional buffered data management for particle trajectories
    - SedTrails xarray dataset creation and management
    - NetCDF file writing with proper metadata and structure

    Example Usage for SedTrails Output
    ----------------------------------
    
    # Initialize DataManager
    data_manager = DataManager("output/")

    # Create SedTrails dataset (using composition)
    dataset = data_manager.writer.create_dataset(
        N_particles=100, N_populations=2, N_timesteps=50, N_flowfields=1
    )

    # Add metadata (using composition)
    data_manager.writer.add_metadata(dataset, populations, flow_field_names)

    # During simulation loop (DataManager provides this convenience method)
    data_manager.collect_timestep_data(dataset, populations, timestep, current_time)

    # Write final results (using composition)
    output_path = data_manager.writer.write(dataset, filename, trim_to_actual_timesteps=True)
    """

    def __init__(self, output_dir: Union[str, Path], max_bytes=512 * 1024 * 1024):
        """
        Initialize the DataManager with a output data directory.
        All other resources are initialized lazily.

        This class acts as the main interface between the simulation and the data management
        system. It handles adding new simulation data, checking memory usage, and writing
        data to disk when necessary.

        Parameters
        ----------
        output_dir : str or Path
            Path to the output directory where data will be stored.
        max_bytes : int, default=512*1024*1024
            Maximum memory usage in bytes before writing to disk.

        Attributes
        ----------
        
        output_dir : Path
            Path to the output directory where data will be stored.
        data_buffer : SimulationDataBuffer
            Buffer for temporarily storing simulation data.
        memory_manager : MemoryManager
            Utility for checking if the buffer exceeds the memory limit.
        writer : NetCDFWriter
            Writer instance for outputting NetCDF files.
        _mesh_info : tuple or None
            Mesh information (node_x, node_y, face_node_connectivity, fill_value).
        file_counter : int
            Counter for naming output files uniquely.
        """

        self.data_buffer = SimulationDataBuffer()
        self.memory_manager = MemoryManager(max_bytes=max_bytes)
        self.writer = NetCDFWriter(output_dir)
        self.output_dir = self.writer.output_dir  # Make sure we are using the same results directory
        self._mesh_info = None
        self.file_counter = 0

    def _cleanup_chunk_files(self):
        """
        Remove all intermediate chunk files (.sim_buffer_*.nc) from the output directory.

        This is called automatically by finalize() when cleanup_chunks=True.
        """
        
        import os
        from pathlib import Path

        output_dir = Path(self.writer.output_dir)

        # Find all chunk files
        chunk_files = [
            output_dir / f for f in os.listdir(output_dir) if f.startswith('.sim_buffer_') and f.endswith('.nc')
        ]

        # Remove each chunk file
        for chunk_file in chunk_files:
            try:
                chunk_file.unlink()  # Delete the file
            except FileNotFoundError:
                # File already deleted, ignore
                pass
            except Exception as e:
                # Log warning but don't fail the operation
                logging.warning(f'Could not delete chunk file {chunk_file}: {e}')

    def set_mesh(
        self, node_x=NODE_X, node_y=NODE_Y, face_node_connectivity=FACE_NODE_CONNECTIVITY, fill_value=FILL_VALUE
    ):
        """
        Set or update the mesh information for output.
        """
        
        self._mesh_info = (node_x, node_y, face_node_connectivity, fill_value)

    def add_data(self, particle_id, time, x, y):
        """
        Add a new data point to the simulation data buffer.
        If the buffer exceeds the memory limit, write to disk and clear the buffer.

        Parameters
        ----------
        
        particle_id : int
            Unique identifier for the particle.
        time : float or int
            Simulation time or time step.
        x : float
            X-coordinate of the particle.
        y : float
            Y-coordinate of the particle.
        """
        
        self.data_buffer.add(particle_id, time, x, y)
        if self._mesh_info is not None:
            node_x, node_y, face_node_connectivity, fill_value = self._mesh_info
            if self.memory_manager.is_limit_exceeded(self.data_buffer.get_data()):
                filename = f'.sim_buffer_{self.file_counter}.nc'
                self.data_buffer.write_to_disk(
                    node_x, node_y, face_node_connectivity, fill_value, self.writer, filename
                )
                self.file_counter += 1

    def write(self, filename=None):
        """
        Write out the current buffer to a NetCDF file, even if memory is not exceeded.

        Parameters
        ----------
        
        filename : str or None
            Name of the output NetCDF file. If None, uses a default naming scheme.
        """
        
        if self._mesh_info is None:
            raise ValueError('Mesh information must be set before writing data.')
        node_x, node_y, face_node_connectivity, fill_value = self._mesh_info
        if filename is None:
            filename = f'.sim_buffer_{self.file_counter}.nc'
        self.data_buffer.write_to_disk(node_x, node_y, face_node_connectivity, fill_value, self.writer, filename)
        self.file_counter += 1

    def merge(self, merged_filename='merged_output.nc'):
        """
        Merge all chunked NetCDF files into a single file.

        Parameters
        ----------
        
        merged_filename : str
            Name of the merged output file.
        """
        
        SimulationDataBuffer.merge_output_files(self.writer.output_dir, merged_filename)

    def dump(self, merge=True, merged_filename='final_output.nc', cleanup_chunks=True):
        """
        Finalize all data operations: write current buffer if it contains data and optionally merge all files.

        This is the main function to call at the end of a simulation to handle all
        file I/O operations. It abstracts away the details of writing buffers and
        merging multiple chunk files.

        Parameters
        ----------
        
        merge : bool, default=True
            Whether to merge all chunk files into a single output file.
        merged_filename : str, default="final_output.nc"
            Name of the final merged output file.
        cleanup_chunks : bool, default=True
            Whether to delete intermediate chunk files after merging.
            Only applies when merge=True.

        Returns
        -------
        
        str
            Path to the final output file (merged file if merge=True, otherwise the last chunk file).

        Examples
        --------
        
        >>> dm = DataManager("output/")
        >>> dm.set_mesh(node_x, node_y, face_connectivity)
        >>> # ... add simulation data ...
        >>> final_file = dm.dump()  # Automatically writes buffer if needed and merges
        >>>
        >>> # Keep chunk files after merging
        >>> final_file = dm.dump(cleanup_chunks=False)
        >>>
        >>> # Or just write without merging
        >>> final_file = dm.dump(merge=False)
        """
        
        if self._mesh_info is None:
            raise ValueError('Mesh information must be set before finalizing data.')

        final_output_path = None

        # Check if buffer contains data and write it if so
        buffer_data = self.data_buffer.get_data()
        if buffer_data['particle_id'].size > 0:
            self.write()

        # Merge all chunk files into a single file
        if merge:
            SimulationDataBuffer.merge_output_files(self.writer.output_dir, merged_filename)
            final_output_path = self.writer.output_dir / merged_filename

            # Clean up intermediate chunk files after successful merge
            if cleanup_chunks:
                self._cleanup_chunk_files()
        else:
            # Return the path to the last written file
            if self.file_counter > 0:
                last_file = f'.sim_buffer_{self.file_counter - 1}.nc'
                final_output_path = self.writer.output_dir / last_file

        return str(final_output_path) if final_output_path else None

    def collect_timestep_data(self, dataset, populations, timestep, current_time):
        """
        Collect data from all populations for a specific timestep into the dataset.

        Parameters
        ----------
        
        dataset : xr.Dataset
            The xarray dataset to populate
        populations : list
            List of population objects from the simulation
        timestep : int
            Current timestep index
        current_time : float
            Current simulation time
        """
        
        collect_timestep_data(dataset, populations, timestep, current_time)
