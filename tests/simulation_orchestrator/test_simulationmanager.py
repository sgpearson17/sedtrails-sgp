"""
Unit tests for the Simulation class.
"""

import numpy as np
import pytest
import xarray as xr

from sedtrails.simulation_orchestrator.simulation_manager import Simulation


class TestSimulationManagerExpandTimeDimension:
    """Tests for the _expand_time_dimension method."""

    @pytest.fixture
    def simulation_manager(self):
        """Create a SimulationManager instance for testing."""
        # Create instance without initializing (we only need the method)
        manager = object.__new__(Simulation)
        return manager

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample xarray dataset with time dimension."""
        n_particles = 10
        n_timesteps = 100
        n_populations = 2

        dataset = xr.Dataset(
            {
                'x': (['time', 'particle'], np.random.rand(n_timesteps, n_particles)),
                'y': (['time', 'particle'], np.random.rand(n_timesteps, n_particles)),
                'burial_depth': (['time', 'particle'], np.random.rand(n_timesteps, n_particles)),
                'population_id': (['particle'], np.random.randint(0, n_populations, n_particles)),
                'static_field': (['particle'], np.random.rand(n_particles)),  # No time dimension
            },
            coords={
                'time': np.arange(n_timesteps),
                'particle': np.arange(n_particles),
            },
        )
        return dataset

    def test_expand_increases_time_dimension(self, simulation_manager, sample_dataset):
        """Test that expansion increases the time dimension size."""
        original_size = len(sample_dataset.time)
        new_size = 150

        expanded = simulation_manager._expand_time_dimension(sample_dataset, new_size)

        assert len(expanded.time) == new_size
        assert len(expanded.time) > original_size

    def test_expand_preserves_original_data(self, simulation_manager, sample_dataset):
        """Test that original data is preserved after expansion."""
        original_size = len(sample_dataset.time)
        new_size = 150

        # Store original values
        original_x = sample_dataset['x'].values.copy()
        original_y = sample_dataset['y'].values.copy()

        expanded = simulation_manager._expand_time_dimension(sample_dataset, new_size)

        # Check that original timesteps are unchanged
        np.testing.assert_array_equal(expanded['x'].isel(time=slice(0, original_size)).values, original_x)
        np.testing.assert_array_equal(expanded['y'].isel(time=slice(0, original_size)).values, original_y)

    def test_expand_fills_new_timesteps_with_nan(self, simulation_manager, sample_dataset):
        """Test that new timesteps are filled with NaN."""
        original_size = len(sample_dataset.time)
        new_size = 150

        expanded = simulation_manager._expand_time_dimension(sample_dataset, new_size)

        # Check that new timesteps contain NaN
        new_x_data = expanded['x'].isel(time=slice(original_size, new_size)).values
        assert np.all(np.isnan(new_x_data))

        new_y_data = expanded['y'].isel(time=slice(original_size, new_size)).values
        assert np.all(np.isnan(new_y_data))

    def test_expand_only_affects_time_dependent_variables(self, simulation_manager, sample_dataset):
        """Test that variables without time dimension are not affected."""
        original_static = sample_dataset['static_field'].values.copy()
        new_size = 150

        expanded = simulation_manager._expand_time_dimension(sample_dataset, new_size)

        # Static field should be unchanged
        np.testing.assert_array_equal(expanded['static_field'].values, original_static)
        # Static field should not have time dimension
        assert 'time' not in expanded['static_field'].dims

    def test_expand_updates_time_coordinate(self, simulation_manager, sample_dataset):
        """Test that time coordinate is updated correctly."""
        new_size = 150

        expanded = simulation_manager._expand_time_dimension(sample_dataset, new_size)

        # Time coordinate should be sequential from 0 to new_size-1
        expected_time = np.arange(new_size)
        np.testing.assert_array_equal(expanded.time.values, expected_time)

    def test_expand_preserves_data_types(self, simulation_manager, sample_dataset):
        """Test that data types are preserved after expansion."""
        original_dtypes = {var: sample_dataset[var].dtype for var in sample_dataset.data_vars}
        new_size = 150

        expanded = simulation_manager._expand_time_dimension(sample_dataset, new_size)

        for var in expanded.data_vars:
            assert expanded[var].dtype == original_dtypes[var], f'Data type changed for {var}'

    def test_expand_preserves_dimensions(self, simulation_manager, sample_dataset):
        """Test that dimension names are preserved."""
        original_dims = {var: sample_dataset[var].dims for var in sample_dataset.data_vars}
        new_size = 150

        expanded = simulation_manager._expand_time_dimension(sample_dataset, new_size)

        for var in expanded.data_vars:
            assert expanded[var].dims == original_dims[var], f'Dimensions changed for {var}'

    def test_expand_handles_multiple_expansions(self, simulation_manager, sample_dataset):
        """Test that multiple consecutive expansions work correctly."""
        # First expansion
        expanded1 = simulation_manager._expand_time_dimension(sample_dataset, 150)
        assert len(expanded1.time) == 150

        # Second expansion
        expanded2 = simulation_manager._expand_time_dimension(expanded1, 200)
        assert len(expanded2.time) == 200

        # Original data should still be intact
        original_size = len(sample_dataset.time)
        np.testing.assert_array_equal(
            expanded2['x'].isel(time=slice(0, original_size)).values, sample_dataset['x'].values
        )

    def test_expand_with_different_dimension_orders(self, simulation_manager):
        """Test expansion with different dimension orders."""
        # Create dataset with time not as first dimension
        dataset = xr.Dataset(
            {
                'variable1': (['particle', 'time'], np.random.rand(10, 50)),
                'variable2': (['time', 'particle', 'depth'], np.random.rand(50, 10, 3)),
            },
            coords={
                'time': np.arange(50),
                'particle': np.arange(10),
                'depth': np.arange(3),
            },
        )

        new_size = 100
        expanded = simulation_manager._expand_time_dimension(dataset, new_size)

        assert len(expanded.time) == new_size
        assert expanded['variable1'].shape == (10, new_size)
        assert expanded['variable2'].shape == (new_size, 10, 3)

    def test_expand_with_empty_dataset(self, simulation_manager):
        """Test expansion with a dataset that has no data variables."""
        dataset = xr.Dataset(coords={'time': np.arange(10)})

        new_size = 20
        expanded = simulation_manager._expand_time_dimension(dataset, new_size)

        assert len(expanded.time) == new_size

    @pytest.mark.parametrize(
        'original_size,new_size',
        [
            (100, 150),
            (50, 100),
            (200, 300),
            (10, 1000),
        ],
    )
    def test_expand_various_sizes(self, simulation_manager, original_size, new_size):
        """Test expansion with various size combinations."""
        # Create appropriately sized dataset
        dataset = xr.Dataset(
            {
                'x': (['time', 'particle'], np.random.rand(original_size, 10)),
            },
            coords={
                'time': np.arange(original_size),
                'particle': np.arange(10),
            },
        )

        # Store original data before expansion
        original_x = dataset['x'].values.copy()

        expanded = simulation_manager._expand_time_dimension(dataset, new_size)

        assert len(expanded.time) == new_size
        # Check that original data is preserved
        np.testing.assert_array_equal(
            expanded['x'].isel(time=slice(0, original_size)).values,
            original_x,  # Compare against the stored original, not dataset['x']
        )
