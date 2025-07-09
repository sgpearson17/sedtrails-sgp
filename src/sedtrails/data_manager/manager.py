from sedtrails.data_manager import SimulationDataBuffer, MemoryManager


class DataManager:
    """
    A class to maange data and files produced by SedTrails.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the DataManager with a output data directory.

        Parameters
        ----------
        output_dir : str
            Path to the output directory where data will be stored.
        """

        self.output_dir = output_dir
        self.data_buffer = SimulationDataBuffer()
        self.memory_manager = MemoryManager(output_dir)

    def add_data(self, particle_id, time, x, y):
        """
        Add a new data point to the simulation data buffer.

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
        if self.memory_manager.buffer_size_bytes(self.data_buffer.get_data()) > self.memory_manager.max_bytes:
            self.memory_manager.enforce_limit(self.data_buffer, [], [], [])

        # TODO: complete this class
        pass
