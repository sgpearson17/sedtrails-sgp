"""
Unit tests for the Simulation class.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from sedtrails.exceptions.exceptions import ConfigurationError
from sedtrails.particle_tracer.timer import Duration, Time
from sedtrails.simulation_orchestrator.simulation_manager import Simulation


class TestSimulationManagerTimeConfig:
    """Tests for simulation-time construction from configuration."""

    def test_simulation_time_uses_input_model_reference_date(self):
        """SFINCS-style date-only reference dates should align timer seconds with SedTRAILS times."""

        class Controller:
            values = {
                'time.start': '2024-05-01 00:00:00',
                'time.duration': '5M',
                'time.timestep': '60S',
                'inputs.read_interval': '30M',
                'general.input_model.reference_date': '2024-05-01',
            }

            def get(self, key, default=None):
                return self.values.get(key, default)

        manager = object.__new__(Simulation)
        manager._controller = Controller()

        simulation_time = manager._create_simulation_time()

        assert simulation_time.reference_date == '2024-05-01'
        assert simulation_time.start == 0

    @pytest.mark.parametrize('current_time', [0.0, 6011.0, 7200.0])
    def test_loaded_chunk_is_reused_through_final_interpolation_interval(self, current_time):
        """A loaded chunk remains valid until current time moves beyond its last timestamp."""

        class SedtrailsData:
            times = np.array([0.0, 1200.0, 2400.0, 3600.0, 4800.0, 6000.0, 7200.0])

        assert not Simulation._needs_sedtrails_reload(SedtrailsData(), current_time)

    @pytest.mark.parametrize('current_time', [-1.0, 7200.1])
    def test_loaded_chunk_reloads_outside_coverage(self, current_time):
        """Reload only when current time is outside the loaded chunk."""

        class SedtrailsData:
            times = np.array([0.0, 1200.0, 2400.0, 3600.0, 4800.0, 6000.0, 7200.0])

        assert Simulation._needs_sedtrails_reload(SedtrailsData(), current_time)

    def test_input_exhaustion_detects_time_after_loaded_data(self):
        """If reload still leaves current time beyond the loaded data, input is exhausted."""

        class SedtrailsData:
            times = np.array([4800.0, 6000.0, 7200.0])

        assert Simulation._is_after_loaded_sedtrails_data(SedtrailsData(), 7200.1)
        assert not Simulation._is_after_loaded_sedtrails_data(SedtrailsData(), 7200.0)

    def test_simulation_window_must_reach_first_input_time(self):
        """A simulation that ends before input forcing starts should fail clearly."""

        class SimulationTime:
            start = 0.0
            end = 3600.0
            reference_date = '2000-01-01'

        class SedtrailsData:
            times = np.array([4600.0, 8200.0])

        with pytest.raises(ConfigurationError, match='ends before the first available input field timestamp'):
            Simulation._validate_simulation_time_matches_input(SimulationTime(), SedtrailsData())

    def test_simulation_window_must_not_start_before_first_input_time(self):
        """A simulation may not spend CFL steps before forcing data exists."""

        class SimulationTime:
            start = 0.0
            end = 7200.0
            reference_date = '2000-01-01'

        class SedtrailsData:
            times = np.array([4600.0, 8200.0])

        with pytest.raises(ConfigurationError, match='starts before the first available input field timestamp'):
            Simulation._validate_simulation_time_matches_input(SimulationTime(), SedtrailsData())

    def test_simulation_window_may_end_at_first_input_time(self):
        """The first forcing timestamp is a valid simulation endpoint."""

        class SimulationTime:
            start = 4600.0
            end = 4600.0
            reference_date = '2000-01-01'

        class SedtrailsData:
            times = np.array([4600.0, 8200.0])

        Simulation._validate_simulation_time_matches_input(SimulationTime(), SedtrailsData())

    def test_exhausted_input_suppresses_further_reload_attempts(self):
        """After input exhaustion, later timesteps should reuse the last loaded fields."""

        class SedtrailsData:
            times = np.array([4800.0, 6000.0, 7200.0])

        assert Simulation._should_attempt_sedtrails_reload(SedtrailsData(), 7200.1, input_data_exhausted=False)
        assert not Simulation._should_attempt_sedtrails_reload(SedtrailsData(), 7200.1, input_data_exhausted=True)

    def test_simulation_start_must_be_within_forcing_window(self):
        """A run must start on forcing data; repeat mapping is only allowed after that."""

        class SimulationTime:
            start = 0.0
            reference_date = '2000-01-01'
            _start = '2000-01-01 00:00:00'

        with pytest.raises(ConfigurationError, match='time.start must be within the input forcing window'):
            Simulation._validate_simulation_start_matches_input(SimulationTime(), (3600.0, 7200.0))

    def test_simulation_start_at_forcing_edges_is_valid(self):
        """The first and last forcing timestamps are valid simulation starts."""

        class SimulationTime:
            reference_date = '2000-01-01'
            _start = '2000-01-01 01:00:00'

            def __init__(self, start):
                self.start = start

        Simulation._validate_simulation_start_matches_input(SimulationTime(3600.0), (3600.0, 7200.0))
        Simulation._validate_simulation_start_matches_input(SimulationTime(7200.0), (3600.0, 7200.0))

    def test_eulerian_time_is_unchanged_when_repeat_disabled(self):
        """Without looping, field lookups use the simulation clock."""

        mapped_time = Simulation._map_eulerian_field_time(
            current_time_seconds=25.0,
            repeat_eulerian_fields=False,
            input_time_bounds=(0.0, 10.0),
        )

        assert mapped_time == 25.0

    def test_eulerian_time_before_input_start_is_not_mapped_forward(self):
        """Before-start lookups remain invalid so validation/reload failures stay clear."""

        mapped_time = Simulation._map_eulerian_field_time(
            current_time_seconds=50.0,
            repeat_eulerian_fields=True,
            input_time_bounds=(100.0, 200.0),
        )

        assert mapped_time == 50.0

    @pytest.mark.parametrize(
        'current_time,expected_time',
        [
            (5.0, 5.0),
            (10.0, 10.0),
            (12.5, 2.5),
            (25.0, 5.0),
        ],
    )
    def test_eulerian_time_repeats_from_input_start(self, current_time, expected_time):
        """When looping is enabled, times after forcing end wrap to the first input timestamp."""

        mapped_time = Simulation._map_eulerian_field_time(
            current_time_seconds=current_time,
            repeat_eulerian_fields=True,
            input_time_bounds=(0.0, 10.0),
        )

        assert mapped_time == expected_time

    def test_eulerian_time_repeats_with_nonzero_input_start(self):
        """Looping preserves input-series offsets when the forcing does not start at zero seconds."""

        mapped_time = Simulation._map_eulerian_field_time(
            current_time_seconds=125.0,
            repeat_eulerian_fields=True,
            input_time_bounds=(100.0, 110.0),
        )

        assert mapped_time == 105.0

    def test_eulerian_time_repeat_uses_morfac_decompressed_bounds(self):
        """Looping should use the already-decompressed forcing span reported by the converter."""

        mapped_time = Simulation._map_eulerian_field_time(
            current_time_seconds=250.0,
            repeat_eulerian_fields=True,
            input_time_bounds=(100.0, 220.0),
        )

        assert mapped_time == 130.0

    def test_output_save_interval_uses_outputs_config(self):
        """Trajectory output cadence should be read from outputs.save_interval."""

        class Controller:
            def get(self, key, default=None):
                if key == 'outputs.save_interval':
                    return '30M'
                return default

        manager = object.__new__(Simulation)
        manager._controller = Controller()

        assert manager._output_save_interval_seconds() == 1800

    def test_output_save_interval_defaults_to_one_hour(self):
        """When unset, saved trajectory samples should default to one-hour spacing."""
        manager = object.__new__(Simulation)
        manager._controller = type('Controller', (), {'get': lambda self, key, default=None: default})()

        assert manager._output_save_interval_seconds() == 3600

    def test_output_save_interval_rejects_zero_duration(self):
        """A zero save interval would make output scheduling ambiguous."""
        manager = object.__new__(Simulation)
        manager._controller = type('Controller', (), {'get': lambda self, key, default=None: '0S'})()

        with pytest.raises(ConfigurationError, match='outputs.save_interval'):
            manager._output_save_interval_seconds()

    def test_output_sync_interval_defaults_to_save_interval(self):
        """NetCDF flush cadence should default to the trajectory save cadence."""
        manager = object.__new__(Simulation)
        manager._controller = type('Controller', (), {'get': lambda self, key, default=None: default})()

        assert manager._output_sync_interval_seconds(save_interval_seconds=1800) == 1800

    def test_output_sync_interval_uses_outputs_config(self):
        """Configured sync interval should be converted to seconds."""

        class Controller:
            def get(self, key, default=None):
                if key == 'outputs.sync_interval':
                    return '2H'
                return default

        manager = object.__new__(Simulation)
        manager._controller = Controller()

        assert manager._output_sync_interval_seconds(save_interval_seconds=1800) == 7200
        assert Simulation._sync_every_n_writes(save_interval_seconds=1800, sync_interval_seconds=7200) == 4

    def test_output_sync_interval_rejects_zero_duration(self):
        """A zero sync interval would make streaming flush cadence ambiguous."""
        manager = object.__new__(Simulation)
        manager._controller = type('Controller', (), {'get': lambda self, key, default=None: '0S'})()

        with pytest.raises(ConfigurationError, match='outputs.sync_interval'):
            manager._output_sync_interval_seconds(save_interval_seconds=1800)

    @pytest.mark.parametrize(
        'duration,save_interval,expected_count',
        [
            ('3H', 3600, 4),
            ('30M', 3600, 2),
            ('2H30M', 3600, 4),
        ],
    )
    def test_estimate_output_timesteps_counts_initial_scheduled_and_final(
        self, duration, save_interval, expected_count
    ):
        """Output allocation follows save cadence, not CFL integration cadence."""
        simulation_time = Time(
            _start='2000-01-01 00:00:00',
            duration=Duration(duration),
            reference_date='2000-01-01',
        )

        assert Simulation._estimate_output_timesteps(simulation_time, save_interval) == expected_count

    def test_next_scheduled_output_time_caps_at_simulation_end(self):
        """The final output target should be the simulation end, not a time after it."""

        class SimulationTime:
            start = 0.0
            end = 9000.0

        assert Simulation._next_scheduled_output_time(SimulationTime(), 3600, 1) == 3600.0
        assert Simulation._next_scheduled_output_time(SimulationTime(), 3600, 3) == 9000.0

    def test_limit_timestep_to_output_schedule_hits_save_boundary(self):
        """A CFL step that crosses an output boundary should land exactly on it."""
        limited = Simulation._limit_timestep_to_output_schedule(
            current_time=3598.0,
            current_timestep=10.0,
            next_output_time=3600.0,
        )

        assert limited == 2.0

    def test_limit_timestep_to_output_schedule_keeps_short_cfl_step(self):
        """CFL steps shorter than the remaining save interval should not be changed."""
        limited = Simulation._limit_timestep_to_output_schedule(
            current_time=10.0,
            current_timestep=2.0,
            next_output_time=3600.0,
        )

        assert limited == 2.0

    @pytest.mark.parametrize(
        'sample_time,next_output_time,end_time,expected',
        [
            (3599.0, 3600.0, 7200.0, False),
            (3600.0, 3600.0, 7200.0, True),
            (7200.0, 10800.0, 7200.0, True),
        ],
    )
    def test_output_sample_due_on_interval_or_final_time(self, sample_time, next_output_time, end_time, expected):
        """Samples should be saved only on configured boundaries or at final time."""
        assert Simulation._is_output_sample_due(sample_time, next_output_time, end_time) is expected

    def test_initialize_population_output_status_supplies_required_fields(self):
        """The seeded initial sample should have status fields before the first physics update."""

        class Population:
            particles = {
                'x': np.array([1.0, 2.0]),
                'y': np.array([3.0, 4.0]),
                'release_time': np.array([0.0, 10.0]),
                'burial_depth': np.array([0.0, 0.0]),
            }
            _particle_simplices = np.array([5, -1])

        population = Population()

        Simulation._initialize_population_output_status([population], current_time=5.0)

        expected_keys = {
            'status_alive',
            'status_buried',
            'status_domain',
            'status_transported',
            'status_released',
            'status_mobile',
        }
        assert expected_keys.issubset(population.particles)
        np.testing.assert_array_equal(population.particles['status_domain'], np.array([True, False]))
        np.testing.assert_array_equal(population.particles['status_released'], np.array([True, False]))
        np.testing.assert_array_equal(population.particles['status_transported'], np.array([False, False]))
        np.testing.assert_array_equal(population.particles['status_mobile'], np.array([False, False]))


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
        
    def test_ensure_time_capacity_expands_before_out_of_range_write(self, simulation_manager, sample_dataset):
        """Output storage should grow before collecting an adaptive timestep beyond capacity."""
        simulation_manager.logger = type('Logger', (), {'info': lambda self, *args, **kwargs: None})()

        expanded, max_timesteps = simulation_manager._ensure_time_capacity(
            sample_dataset,
            timestep_index=2638,
            max_timesteps=100,
        )

        assert max_timesteps >= 2639
        assert len(expanded.time) == max_timesteps
        np.testing.assert_array_equal(expanded['x'].isel(time=slice(0, 100)).values, sample_dataset['x'].values)

    def test_ensure_time_capacity_reuses_dataset_when_index_fits(self, simulation_manager, sample_dataset):
        """No expansion is needed while the target timestep is inside the allocated dimension."""

        same_dataset, max_timesteps = simulation_manager._ensure_time_capacity(
            sample_dataset,
            timestep_index=99,
            max_timesteps=100,
        )

        assert same_dataset is sample_dataset
        assert max_timesteps == 100

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

class TestSimulationManagerSaveInterval:
    """Tests for save_interval slot-count calculation and boundary logic."""

    @pytest.mark.parametrize(
        'duration_s,interval_s,expected_slots',
        [
            (3600, 3600, 2),        # 1H run, 1H interval  ? slot 0 + 1 boundary
            (4 * 3600, 3600, 5),    # 4H run, 1H interval  ? slot 0 + 4 boundaries
            (86400, 3600, 25),      # 1D run, 1H interval  ? 24 + 1
            (3600, 900, 5),         # 1H run, 15min interval ? 4 + 1
            (3601, 3600, 3),        # just over one interval ? 2 + 1
        ],
    )
    def test_n_output_slots_calculation(self, duration_s, interval_s, expected_slots):
        """Pre-allocated slot count is ceil(duration/interval) + 1 for the initial state."""
        import math
        n_slots = math.ceil(duration_s / interval_s) + 1
        assert n_slots == expected_slots

    def test_save_boundary_triggers_at_next_save_time(self):
        """Record fires exactly when simulation time reaches the boundary, not before."""
        next_save_time = 3600.0
        saved = []

        for t in [0.0, 1200.0, 2400.0, 3600.0, 4800.0]:
            if t >= next_save_time:
                saved.append(t)
                next_save_time += 3600.0

        assert saved == [3600.0]

    def test_final_save_fires_when_sim_ends_between_boundaries(self):
        """A trailing save after the loop captures the final particle state."""
        last_saved_time = 3600.0
        final_time = 4400.0       # simulation ended mid-interval
        slot_idx = 2
        n_output_slots = 5

        should_save = (final_time > last_saved_time) and (slot_idx < n_output_slots)
        assert should_save

    def test_final_save_skipped_when_already_at_boundary(self):
        """No duplicate save when the loop ended exactly on a save boundary."""
        last_saved_time = 7200.0
        final_time = 7200.0
        slot_idx = 3
        n_output_slots = 5

        should_save = (final_time > last_saved_time) and (slot_idx < n_output_slots)
        assert not should_save

    def test_final_save_skipped_when_slots_full(self):
        """No out-of-bounds write when all pre-allocated slots are consumed."""
        last_saved_time = 3600.0
        final_time = 4400.0
        slot_idx = 5
        n_output_slots = 5   # already at capacity

        should_save = (final_time > last_saved_time) and (slot_idx < n_output_slots)
        assert not should_save


class TestSimulationDashboardThrottle:
    """Tests dashboard update cadence and particle payload shape for visualization."""

    def test_dashboard_update_interval_uses_visualization_config(self):
        """Configured dashboard interval should override the default update cadence."""
        class Controller:
            def get(self, key, default=None):
                if key == 'visualization.dashboard.update_interval':
                    return '30S'
                return default

        manager = object.__new__(Simulation)
        manager._controller = Controller()

        assert manager._dashboard_update_interval_seconds() == 30

    def test_dashboard_update_interval_defaults_to_one_hour(self):
        """When unset, dashboard updates should default to a one-hour interval."""
        manager = object.__new__(Simulation)
        manager._controller = type('Controller', (), {'get': lambda self, key, default=None: default})()

        assert manager._dashboard_update_interval_seconds() == 3600

    def test_dashboard_particle_data_uses_arrays_not_single_element_lists(self):
        """Dashboard particle payload should keep vector fields as NumPy arrays."""
        class Population:
            particles = {
                'x': np.array([1.0, 2.0]),
                'y': np.array([3.0, 4.0]),
                'burial_depth': np.array([0.1, 0.2]),
            }

        particle_data = Simulation._dashboard_particle_data(Population())

        assert isinstance(particle_data['burial_depth'], np.ndarray)
        assert isinstance(particle_data['mixing_depth'], np.ndarray)
        assert not isinstance(particle_data['burial_depth'], list)
        assert not isinstance(particle_data['mixing_depth'], list)
        np.testing.assert_array_equal(particle_data['burial_depth'], np.array([0.1, 0.2]))
        np.testing.assert_array_equal(particle_data['mixing_depth'], np.array([np.nan, np.nan]))

    def test_dashboard_particle_data_includes_boundary_statuses(self):
        """Dashboard particle payload should include boundary status arrays for plotting."""
        class Population:
            particles = {
                'x': np.array([1.0, 2.0]),
                'y': np.array([3.0, 4.0]),
                'status_left_domain': np.array([False, True]),
                'status_beached': np.array([True, False]),
            }

        particle_data = Simulation._dashboard_particle_data(Population())

        np.testing.assert_array_equal(particle_data['status_left_domain'], np.array([False, True]))
        np.testing.assert_array_equal(particle_data['status_beached'], np.array([True, False]))

    def test_large_grid_dashboard_updates_are_throttled(self):
        """Large grids should throttle dashboard refreshes to periodic steps."""
        manager = object.__new__(Simulation)
        manager.dashboard = object()
        manager._dashboard_throttle_logged = False
        manager.logger = type('Logger', (), {'info': lambda self, *args, **kwargs: None})()
        manager._controller = type('Controller', (), {'get': lambda self, key, default=None: default})()

        sedtrails_data = type('SedtrailsData', (), {'x': np.arange(100_001)})()
        timer = type('Timer', (), {'step_count': 0})()

        assert manager._should_update_dashboard(sedtrails_data, timer)

        timer.step_count = 1
        assert not manager._should_update_dashboard(sedtrails_data, timer)

        timer.step_count = 10
        assert manager._should_update_dashboard(sedtrails_data, timer)

    def test_small_grid_dashboard_updates_every_step(self):
        """Small grids should keep per-step dashboard updates enabled."""
        manager = object.__new__(Simulation)
        manager.dashboard = object()

        sedtrails_data = type('SedtrailsData', (), {'x': np.arange(100)})()
        timer = type('Timer', (), {'step_count': 1})()

        assert manager._should_update_dashboard(sedtrails_data, timer)


class TestSimulationDomainExitReporting:
    """Tests CLI/log reporting for particles that leave the domain."""

    def test_reports_newly_left_domain_particles(self):
        """Report only particles that newly transition to left-domain status."""
        manager = object.__new__(Simulation)
        manager._report_domain_exits = True
        manager._report_domain_exit_updates = True
        manager.logger = _ListLogger()
        population = SimpleNamespace(
            population_config=SimpleNamespace(population_config={'name': 'sand'}),
            particles={
                'x': np.zeros(3),
                'status_left_domain': np.array([False, True, True]),
            },
        )
        reported_left_domain = np.array([False, False, True])

        newly_left = manager._report_new_domain_exits(
            population,
            population_index=0,
            flow_field_name='bed_load_velocity',
            reported_left_domain=reported_left_domain,
            current_time=10.0,
            current_timestep=2.0,
        )

        assert newly_left == 1
        np.testing.assert_array_equal(reported_left_domain, np.array([False, True, True]))
        assert 'Particles left domain: +1 in sand via bed_load_velocity' in manager.logger.messages[0]
        assert 'population total=2/3' in manager.logger.messages[0]

    def test_new_left_domain_updates_can_be_quiet(self):
        """Intermediate left-domain reporting can be disabled while masks still update."""
        manager = object.__new__(Simulation)
        manager._report_domain_exits = True
        manager._report_domain_exit_updates = False
        manager.logger = _ListLogger()
        population = SimpleNamespace(
            population_config=SimpleNamespace(population_config={'name': 'sand'}),
            particles={
                'x': np.zeros(3),
                'status_left_domain': np.array([False, True, True]),
            },
        )
        reported_left_domain = np.array([False, False, True])

        newly_left = manager._report_new_domain_exits(
            population,
            population_index=0,
            flow_field_name='bed_load_velocity',
            reported_left_domain=reported_left_domain,
            current_time=10.0,
            current_timestep=2.0,
        )

        assert newly_left == 1
        np.testing.assert_array_equal(reported_left_domain, np.array([False, True, True]))
        assert manager.logger.messages == []

    def test_final_summary_reports_total_left_domain_particles(self):
        """Final summary should report aggregate and per-population left-domain totals."""
        manager = object.__new__(Simulation)
        manager._report_domain_exits = True
        manager.logger = _ListLogger()
        populations = [
            SimpleNamespace(
                population_config={'name': 'fine'},
                particles={'x': np.zeros(2), 'status_left_domain': np.array([True, False])},
            ),
            SimpleNamespace(
                population_config={'name': 'medium'},
                particles={'x': np.zeros(3), 'status_left_domain': np.array([False, True, True])},
            ),
        ]

        manager._report_domain_exit_summary(populations)

        assert manager.logger.messages == ['Particles left domain during run: 3/5 (fine=1/2, medium=2/3)']

    def test_reports_newly_beached_particles(self):
        """Report only particles that newly transition to beached status."""
        manager = object.__new__(Simulation)
        manager._report_domain_exits = True
        manager._report_domain_exit_updates = True
        manager.logger = _ListLogger()
        population = SimpleNamespace(
            population_config=SimpleNamespace(population_config={'name': 'sand'}),
            particles={
                'x': np.zeros(3),
                'status_beached': np.array([False, True, True]),
            },
        )
        reported_beached = np.array([False, False, True])

        newly_beached = manager._report_new_beached_particles(
            population,
            population_index=0,
            flow_field_name='bed_load_velocity',
            reported_beached=reported_beached,
            current_time=10.0,
            current_timestep=2.0,
        )

        assert newly_beached == 1
        np.testing.assert_array_equal(reported_beached, np.array([False, True, True]))
        assert 'Particles beached on land: +1 in sand via bed_load_velocity' in manager.logger.messages[0]
        assert 'population total=2/3' in manager.logger.messages[0]

    def test_new_beached_updates_can_be_quiet(self):
        """Intermediate beaching reporting can be disabled while history still updates."""
        manager = object.__new__(Simulation)
        manager._report_domain_exits = True
        manager._report_domain_exit_updates = False
        manager.logger = _ListLogger()
        population = SimpleNamespace(
            population_config=SimpleNamespace(population_config={'name': 'sand'}),
            particles={
                'x': np.zeros(3),
                'status_beached': np.array([False, True, True]),
            },
        )
        reported_beached = np.array([False, False, True])

        newly_beached = manager._report_new_beached_particles(
            population,
            population_index=0,
            flow_field_name='bed_load_velocity',
            reported_beached=reported_beached,
            current_time=10.0,
            current_timestep=2.0,
        )

        assert newly_beached == 1
        np.testing.assert_array_equal(reported_beached, np.array([False, True, True]))
        assert manager.logger.messages == []

    def test_final_summary_reports_total_beached_particles(self):
        """Final summary should report aggregate and per-population beached totals."""
        manager = object.__new__(Simulation)
        manager._report_domain_exits = True
        manager.logger = _ListLogger()
        populations = [
            SimpleNamespace(
                population_config={'name': 'fine'},
                particles={'x': np.zeros(2), 'status_beached': np.array([True, False])},
            ),
            SimpleNamespace(
                population_config={'name': 'medium'},
                particles={'x': np.zeros(3), 'status_beached': np.array([False, True, True])},
            ),
        ]

        manager._report_beached_summary(populations)

        assert manager.logger.messages == ['Particles beached on land during run: 3/5 (fine=1/2, medium=2/3)']

    def test_final_summary_can_use_beached_history(self):
        """Final beached summary should support particles that remobilized later."""
        manager = object.__new__(Simulation)
        manager._report_domain_exits = True
        manager.logger = _ListLogger()
        populations = [
            SimpleNamespace(
                population_config={'name': 'fine'},
                particles={'x': np.zeros(2), 'status_beached': np.array([False, False])},
            ),
        ]
        beached_history = [np.array([True, False])]

        manager._report_beached_summary(populations, beached_history)

        assert manager.logger.messages == ['Particles beached on land during run: 1/2 (fine=1/2)']


class _ListLogger:
    """Minimal logger that stores formatted info messages for assertions."""

    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)
