import pytest
import xarray as xr
import numpy as np
from sedtrails.data_manager.netcdf_writer import NetCDFWriter


@pytest.fixture
def tmp_output_dir(tmp_path):
    """
    Pytest fixture to provide a temporary output directory for NetCDF files.
    """
    return tmp_path / 'output'


class MockPopulation:
    """Mock population class for testing."""

    def __init__(self, name, particle_type=0):
        self.name = name
        self.particle_type = particle_type
        # Mock particle data
        self.particles = {
            'x': np.array([1.0, 2.0, 3.0]),
            'y': np.array([1.5, 2.5, 3.5]),
            'burial_depth': np.array([0.1, 0.2, 0.0]),
            'mixing_depth': np.array([0.5, 0.6, 0.4]),
        }


def test_netcdf_writer_creates_dataset(tmp_output_dir):
    """
    Test that NetCDFWriter successfully creates a SedTrails dataset.
    """
    writer = NetCDFWriter(tmp_output_dir)

    # Create a SedTrails dataset
    dataset = writer.create_dataset(N_particles=10, N_populations=2, N_timesteps=5, N_flowfields=1, name_strlen=24)

    # Check that the dataset has the expected structure
    assert 'x' in dataset.variables
    assert 'y' in dataset.variables
    assert 'time' in dataset.variables
    assert 'population_name' in dataset.variables
    assert 'flowfield_name' in dataset.variables
    assert dataset.dims['n_particles'] == 10
    assert dataset.dims['n_populations'] == 2
    assert dataset.dims['n_timesteps'] == 5
    assert dataset.dims['n_flowfields'] == 1


def test_netcdf_writer_adds_metadata(tmp_output_dir):
    """
    Test that NetCDFWriter successfully adds metadata to a dataset.
    """
    writer = NetCDFWriter(tmp_output_dir)

    # Create dataset
    dataset = writer.create_dataset(N_particles=3, N_populations=1, N_timesteps=2, N_flowfields=1)

    # Create mock data
    populations = [MockPopulation('test_pop')]
    flow_field_names = ['water_velocity']
    metadata = {'test_attr': 'test_value'}

    # Add metadata
    writer.add_metadata(dataset, populations, flow_field_names, metadata)

    # Check that metadata was added
    assert dataset.attrs['title'] == 'SedTrails Particle Simulation Results'
    assert dataset.attrs['institution'] == 'SedTrails Particle Tracer System'
    assert dataset.attrs['test_attr'] == 'test_value'
    assert 'created_on' in dataset.attrs


def test_netcdf_writer_writes_file(tmp_output_dir):
    """
    Test that NetCDFWriter successfully writes an xarray Dataset to a NetCDF file.
    """
    writer = NetCDFWriter(tmp_output_dir)

    # Create and populate a simple dataset
    dataset = writer.create_dataset(N_particles=3, N_populations=1, N_timesteps=2, N_flowfields=1)

    # Add some test data
    dataset['x'][0, 0] = 1.0
    dataset['y'][0, 0] = 2.0
    dataset['time'][0, 0] = 0.0

    filename = 'test_output.nc'
    output_path = writer.write(dataset, filename)

    # Check that file was created
    assert output_path.exists()
    assert output_path.name == filename

    # Verify we can read it back
    loaded_dataset = xr.open_dataset(output_path)
    assert 'x' in loaded_dataset.variables
    assert 'y' in loaded_dataset.variables
    assert loaded_dataset['x'][0, 0] == 1.0
    loaded_dataset.close()


def test_netcdf_writer_writes_with_trimming(tmp_output_dir):
    """
    Test that NetCDFWriter successfully trims timesteps when writing.
    """
    writer = NetCDFWriter(tmp_output_dir)

    # Create dataset with 5 timesteps
    dataset = writer.create_dataset(N_particles=2, N_populations=1, N_timesteps=5, N_flowfields=1)

    # Fill only first 2 timesteps with data
    dataset['x'][:, 0] = [1.0, 2.0]
    dataset['x'][:, 1] = [1.1, 2.1]
    dataset['time'][:, 0] = 0.0
    dataset['time'][:, 1] = 1.0

    filename = 'test_trimmed.nc'
    output_path = writer.write(dataset, filename, trim_to_actual_timesteps=True, actual_timesteps=2)

    # Check that file was created
    assert output_path.exists()

    # Verify trimming worked
    loaded_dataset = xr.open_dataset(output_path)
    assert loaded_dataset.dims['n_timesteps'] == 2  # Should be trimmed from 5 to 2
    loaded_dataset.close()


def test_netcdf_writer_create_and_write_simulation_results(tmp_output_dir):
    """
    Test the all-in-one convenience method for creating and writing simulation results.
    """
    writer = NetCDFWriter(tmp_output_dir)

    # Create mock data
    populations = [MockPopulation('population_1'), MockPopulation('population_2')]
    flow_field_names = ['water_velocity']

    filename = 'simulation_results.nc'
    output_path = writer.create_and_write_simulation_results(
        populations=populations, flow_field_names=flow_field_names, N_timesteps=3, filename=filename
    )

    # Check that file was created
    assert output_path.exists()
    assert output_path.name == filename

    # Verify the content
    loaded_dataset = xr.open_dataset(output_path)
    assert loaded_dataset.dims['n_particles'] == 6  # 3 particles per population * 2 populations
    assert loaded_dataset.dims['n_populations'] == 2
    assert loaded_dataset.dims['n_flowfields'] == 1
    loaded_dataset.close()
