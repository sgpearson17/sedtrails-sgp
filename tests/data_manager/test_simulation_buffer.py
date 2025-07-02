import numpy as np
import xugrid as xu
import pytest
from sedtrails.data_manager.simulation_buffer import SimulationDataBuffer

def test_add_and_get_data():
    """
    Test adding data to the buffer and retrieving it as numpy arrays.
    """
    buffer = SimulationDataBuffer()
    buffer.add(1, 0.0, 10.0, 20.0)
    buffer.add(2, 1.0, 11.0, 21.0)
    data = buffer.get_data()
    assert np.array_equal(data["particle_id"], np.array([1, 2]))
    assert np.array_equal(data["time"], np.array([0.0, 1.0]))
    assert np.array_equal(data["x"], np.array([10.0, 11.0]))
    assert np.array_equal(data["y"], np.array([20.0, 21.0]))

def test_clear():
    """
    Test clearing the buffer.
    """
    buffer = SimulationDataBuffer()
    buffer.add(1, 0.0, 10.0, 20.0)
    buffer.clear()
    data = buffer.get_data()
    for arr in data.values():
        assert arr.size == 0

def test_to_ugrid_dataset():
    """
    Test converting the buffer to a xu.UgridDataset.
    """
    buffer = SimulationDataBuffer()
    buffer.add(1, 0.0, 10.0, 20.0)
    buffer.add(2, 1.0, 11.0, 21.0)
    node_x = np.array([0, 1])
    node_y = np.array([0, 1])
    face_node_connectivity = np.array([[0, 1]])
    fill_value = -1
    ugrid_ds = buffer.to_ugrid_dataset(node_x, node_y, face_node_connectivity, fill_value)
    assert isinstance(ugrid_ds, xu.UgridDataset)
    assert "particle_id" in ugrid_ds
    assert "x" in ugrid_ds
    assert "y" in ugrid_ds
    assert "time" in ugrid_ds