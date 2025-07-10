from sedtrails.data_manager.simulation_buffer import SimulationDataBuffer
from sedtrails.data_manager.memory_manager import MemoryManager
from sedtrails.data_manager.netcdf_writer import NetCDFWriter

class DataManager:
    """
    A class to manage data and files produced by SedTrails, also simulation data buffering
    and memory checks.
    """

    def __init__(self, output_dir: str, max_bytes=512 * 1024 * 1024):
        """
        Initialize the DataManager with a output data directory.
        All other resources are initialized lazily.

        This class acts as the main interface between the simulation and the data management
        system. It handles adding new simulation data, checking memory usage, and writing
        data to disk when necessary.

        Attributes
        ----------
        output_dir : str
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

        self.output_dir = output_dir
        self.data_buffer = SimulationDataBuffer()
        self.memory_manager = MemoryManager(output_dir=output_dir, max_bytes=max_bytes)
        self.writer = NetCDFWriter(output_dir)
        self._mesh_info = None
        self.file_counter = 0

    def set_mesh(self, node_x, node_y, face_node_connectivity, fill_value=-1):
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
                filename = f".sim_buffer_{self.file_counter}.nc"
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
            raise ValueError("Mesh information must be set before writing data.")
        node_x, node_y, face_node_connectivity, fill_value = self._mesh_info
        if filename is None:
            filename = f".sim_buffer_{self.file_counter}.nc"
        self.data_buffer.write_to_disk(
            node_x, node_y, face_node_connectivity, fill_value, self.writer, filename
        )
        self.file_counter += 1

    def merge(self, merged_filename="merged_output.nc"):
        """
        Merge all chunked NetCDF files into a single file.

        Parameters
        ----------
        merged_filename : str
            Name of the merged output file.
        """
        SimulationDataBuffer.merge_output_files(self.writer.output_dir, merged_filename)