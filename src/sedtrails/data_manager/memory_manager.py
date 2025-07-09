"""
Memory Manager
==============
Manages memory allocation and deallocation for the simulation data buffer.
"""

import sys
import xugrid as xu
from sedtrails.data_manager.netcdf_writer import NetCDFWriter

class MemoryManager:
    """
    Manages memory usage for the simulation data buffer. If the buffer exceeds the memory limit,
    writes its contents to a NetCDF file and clears the buffer.

    The current memory limit is set to 512 MB

    Attributes
    ----------
    max_bytes : int
        Maximum allowed memory usage in bytes.
    writer : NetCDFWriter
        Writer instance to write buffer to NetCDF when limit is exceeded.
    file_counter : int
        Counter for naming output files uniquely.
    """

    def __init__(self, output_dir, max_bytes=512 * 1024 * 1024):
        """
        Initialize the memory manager.

        Parameters
        ----------
        output_dir : str or Path
            Directory to write NetCDF files.
        max_bytes : int
            Maximum allowed memory usage in bytes (default: 512 MB).
        """
        self.max_bytes = max_bytes
        self.writer = NetCDFWriter(output_dir)
        self.file_counter = 0

    def buffer_size_bytes(self, buffer):
        """
        Estimate the memory usage of the buffer in bytes.

        Parameters
        ----------
        buffer : dict
            The simulation data buffer (dict of lists or numpy arrays).

        Returns
        -------
        int
            Estimated memory usage in bytes.
        """
        size = 0
        for v in buffer.values():
            if hasattr(v, 'nbytes'):
                size += v.nbytes
            else:
                size += sys.getsizeof(v)
                if hasattr(v, '__iter__') and not isinstance(v, (str, bytes)):
                    size += sum(sys.getsizeof(item) for item in v)
        return size

    def enforce_limit(self, sim_buffer, node_x, node_y, face_node_connectivity, fill_value=-1):
        """
        Enforce the memory limit: if over, write buffer to NetCDF and clear it.

        Currently the output files are split into chunks of 512 MB: sim_buffer_0.nc,
        sim_buffer_1.nc, etc. This will prevent any data loss due to memory overflow. 
        In the future we need to check if this creates issues in managing many files.

        Parameters
        ----------
        sim_buffer : SimulationDataBuffer
            The simulation data buffer instance from simulation_buffer.py.
        node_x, node_y, face_node_connectivity, fill_value : arrays/values
            Mesh info for NetCDF writing.
        """
        if self.buffer_size_bytes(sim_buffer.buffer) > self.max_bytes:
            filename = f"sim_buffer_{self.file_counter}.nc"
            ugrid_ds = sim_buffer.to_ugrid_dataset(node_x, node_y, face_node_connectivity, fill_value)
            self.writer.write(ugrid_ds, filename)
            sim_buffer.clear()
            self.file_counter += 1

    def merge_output_files(self, merged_filename="merged_output.nc"):
        """
        Merge all sim_buffer_*.nc files in the output directory into a single NetCDF file.

        Merging is based on the assumption that all files have the same structure and can be 
        combined by coordinates, e.g., time, observation index, etc. This way, the merged 
        file "merged_output.nc" will contain the entire simulation duration and all particles.
        Each file chunk should have non-overlapping coordinate values.

        Parameters
        ----------
        merged_filename : str
            Name of the merged output file.
        """
        files = sorted(self.writer.output_dir.glob("sim_buffer_*.nc"))
        if not files:
            raise FileNotFoundError("No sim_buffer_*.nc files found to merge.")
        ds = xu.open_mfdataset(files, combine="by_coords")
        ds.ugrid.to_netcdf(self.writer.output_dir / merged_filename)            