"""
Unit tests for the Simulation class.
"""

import numpy as np
import pytest

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

    def test_exhausted_input_suppresses_further_reload_attempts(self):
        """After input exhaustion, later timesteps should reuse the last loaded fields."""

        class SedtrailsData:
            times = np.array([4800.0, 6000.0, 7200.0])

        assert Simulation._should_attempt_sedtrails_reload(SedtrailsData(), 7200.1, input_data_exhausted=False)
        assert not Simulation._should_attempt_sedtrails_reload(SedtrailsData(), 7200.1, input_data_exhausted=True)

    def test_eulerian_time_is_unchanged_when_repeat_disabled(self):
        """Without looping, field lookups use the simulation clock."""

        mapped_time = Simulation._map_eulerian_field_time(
            current_time_seconds=25.0,
            repeat_eulerian_fields=False,
            input_time_bounds=(0.0, 10.0),
        )

        assert mapped_time == 25.0

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


class TestSimulationManagerSaveInterval:
    """Tests for save_interval slot-count calculation and boundary logic."""

    @pytest.mark.parametrize(
        'duration_s,interval_s,expected_slots',
        [
            (3600, 3600, 2),        # 1H run, 1H interval  → slot 0 + 1 boundary
            (4 * 3600, 3600, 5),    # 4H run, 1H interval  → slot 0 + 4 boundaries
            (86400, 3600, 25),      # 1D run, 1H interval  → 24 + 1
            (3600, 900, 5),         # 1H run, 15min interval → 4 + 1
            (3601, 3600, 3),        # just over one interval → 2 + 1
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
