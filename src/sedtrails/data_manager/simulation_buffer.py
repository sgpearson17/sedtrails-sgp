import xugrid as xu
import numpy as np


class SimulationDataBuffer:
    """
    Temporarily stores chunks of simulation data in memory, while they wait to be written to a simulation file.

    Attributes
    ----------
    buffer : dict
        Dictionary holding simulation data arrays (particle ID, positions, and times at the moment).
    """

    def __init__(self):
        """
        Initialize an empty simulation data buffer.
        """
        self.buffer = {
            'particle_id': [],
            'time': [],
            'x': [],
            'y': [],
            # Add other fields as needed (e.g., velocity, status, etc.)
        }

    def add(self, particle_id, time, x, y):
        """
        Add a new data point to the buffer.

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
        self.buffer['particle_id'].append(particle_id)
        self.buffer['time'].append(time)
        self.buffer['x'].append(x)
        self.buffer['y'].append(y)

    def clear(self):
        """
        Clear the buffer.
        """
        for key in self.buffer:
            self.buffer[key] = []

    def get_data(self):
        """
        Return the buffer as a dictionary of numpy arrays.

        Returns
        -------
        dict
            Dictionary with keys as field names and values as numpy arrays.
        """
        return {k: np.array(v) for k, v in self.buffer.items()}

    def to_ugrid_dataset(self, node_x, node_y, face_node_connectivity, fill_value=-1):
        """
        Convert the buffer contents to a xu.UgridDataset for writing.

        Parameters
        ----------
        node_x : np.ndarray
            Mesh node x-coordinates.
        node_y : np.ndarray
            Mesh node y-coordinates.
        face_node_connectivity : np.ndarray
            Mesh face connectivity.
        fill_value : int, optional
            Fill value for unused nodes (default: -1).

        Returns
        -------
        xu.UgridDataset
            UGRID-compliant dataset containing the buffered simulation data.
        """
        mesh = xu.Ugrid2d(
            node_x=node_x, node_y=node_y, face_node_connectivity=face_node_connectivity, fill_value=fill_value
        )
        ds = mesh.to_dataset()
        data = self.get_data()
        # Use 'time' as the coordinate and dimension
        ds = ds.assign_coords(time=('time', data['time']))
        ds['particle_id'] = ('time', data['particle_id'])
        ds['x'] = ('time', data['x'])
        ds['y'] = ('time', data['y'])
        return xu.UgridDataset(ds)

    def write_to_disk(self, node_x, node_y, face_node_connectivity, fill_value, writer, filename):
        """
        Write the current buffer to disk as a UGRID NetCDF file and clear the buffer.

        Parameters
        ----------
        node_x : np.ndarray
            Mesh node x-coordinates.
        node_y : np.ndarray
            Mesh node y-coordinates.
        face_node_connectivity : np.ndarray
            Mesh face connectivity.
        fill_value : int
            Fill value for unused nodes.
        writer : NetCDFWriter
            Writer instance to handle NetCDF output.
        filename : str
            Name of the output NetCDF file.
        """
        ugrid_ds = self.to_ugrid_dataset(node_x, node_y, face_node_connectivity, fill_value)
        writer.write(ugrid_ds, filename)
        self.clear()