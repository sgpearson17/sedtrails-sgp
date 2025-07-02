import numpy as np
import xugrid as xu
from sedtrails.data_manager.memory_manager import MemoryManager
from sedtrails.data_manager.simulation_buffer import SimulationDataBuffer

def create_test_mesh():
    """
    Create a minimal mesh for testing purposes.

    Returns
    -------
    tuple
        node_x, node_y, face_node_connectivity, fill_value
    """    
    node_x = np.array([0, 1, 1, 0])
    node_y = np.array([0, 0, 1, 1])
    face_node_connectivity = np.array([[0, 1, 2, 3]])
    fill_value = -1
    return node_x, node_y, face_node_connectivity, fill_value

def test_buffer_size_bytes_counts_numpy_and_lists(tmp_path):
    """
    Test that buffer_size_bytes correctly counts memory for numpy arrays and lists.
    """    
    mm = MemoryManager(tmp_path)
    buffer = {
        "a": np.zeros(100, dtype=np.float64),
        "b": [1] * 100
    }
    size = mm.buffer_size_bytes(buffer)
    assert size > 0

def test_enforce_limit_writes_file_and_clears(tmp_path):
    """
    Test that enforce_limit writes a NetCDF file and clears the buffer when over the memory limit.
    """
    node_x, node_y, face_node_connectivity, fill_value = create_test_mesh()
    mm = MemoryManager(tmp_path, max_bytes=1_000)  # Set a very low limit for test
    sim_buffer = SimulationDataBuffer()
    # Add enough data to exceed the limit
    for i in range(5000):  # Increase to ensure buffer exceeds limit
        sim_buffer.add(i, float(i), float(i), float(i))
    print("Buffer size (bytes):", mm.buffer_size_bytes(sim_buffer.buffer))
    mm.enforce_limit(sim_buffer, node_x, node_y, face_node_connectivity, fill_value)
    # The NetCDFWriter may create a timestamped subdirectory, so check mm.writer.output_dir
    # This makes the test more robust to changes in output directory structure and
    # keeps our code modular
    expected_file = mm.writer.output_dir / "sim_buffer_0.nc"
    assert expected_file.exists()
    # Buffer should be cleared
    data = sim_buffer.get_data()
    for arr in data.values():
        assert arr.size == 0

def test_enforce_limit_does_not_write_if_under_limit(tmp_path):
    """
    Test that enforce_limit does not write a file or clear the buffer if under the memory limit.
    """    
    node_x, node_y, face_node_connectivity, fill_value = create_test_mesh()
    mm = MemoryManager(tmp_path, max_bytes=10_000_000)  # High limit
    sim_buffer = SimulationDataBuffer()
    for i in range(10):
        sim_buffer.add(i, float(i), float(i), float(i))
    mm.enforce_limit(sim_buffer, node_x, node_y, face_node_connectivity, fill_value)
    # No file should be written
    assert not any(f.name.startswith("sim_buffer_") and f.suffix == ".nc" for f in tmp_path.iterdir())
    # Buffer should not be cleared
    data = sim_buffer.get_data()
    assert all(arr.size == 10 for arr in data.values())