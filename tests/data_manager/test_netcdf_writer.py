import numpy as np
import pytest
import xarray as xr

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
        self.particles = {
            'x': np.array([1.0, 2.0, 3.0]),
            'y': np.array([1.5, 2.5, 3.5]),
            'burial_depth': np.array([0.1, 0.2, 0.0]),
            'mixing_depth': np.array([0.5, 0.6, 0.4]),
            'status_mobile': np.array([1, 0, 1], dtype=np.int32),
            'status_alive': np.array([1, 1, 1], dtype=np.int32),
            'status_buried': np.array([0, 0, 0], dtype=np.int32),
            'status_domain': np.array([1, 1, 1], dtype=np.int32),
            'status_transported': np.array([0, 1, 0], dtype=np.int32),
            'status_released': np.array([1, 1, 1], dtype=np.int32),
        }


# ---------------------------------------------------------------------------
# Streaming output tests
# ---------------------------------------------------------------------------

class TestNetCDFWriterStreaming:
    """Tests for open_output / record_output / close_output."""

    N_PARTICLES = 3
    N_SLOTS = 4
    N_POPULATIONS = 1
    N_FLOWFIELDS = 1

    @pytest.fixture
    def writer(self, tmp_path):
        return NetCDFWriter(tmp_path / 'output')

    @pytest.fixture
    def population(self):
        return MockPopulation('test_pop')

    @pytest.fixture
    def open_handle(self, writer, population, tmp_path):
        """Open a streaming file and yield the handle; close in teardown."""
        handle = writer.open_output(
            'stream.nc',
            self.N_SLOTS,
            self.N_PARTICLES,
            self.N_POPULATIONS,
            self.N_FLOWFIELDS,
            [population],
            ['water_velocity'],
        )
        yield handle
        if handle.isopen():
            handle.close()

    def test_open_creates_file_with_correct_dimensions(self, open_handle):
        ds = open_handle
        assert ds.dimensions['n_particles'].size == self.N_PARTICLES
        assert ds.dimensions['n_timesteps'].size == self.N_SLOTS
        assert ds.dimensions['n_populations'].size == self.N_POPULATIONS
        assert ds.dimensions['n_flowfields'].size == self.N_FLOWFIELDS

    def test_open_creates_all_trajectory_variables(self, open_handle):
        expected = {'x', 'y', 'z', 'time', 'burial_depth', 'mixing_depth',
                    'status_alive', 'status_buried', 'status_domain',
                    'status_transported', 'status_released', 'status_mobile',
                    'covered_distance'}
        assert expected.issubset(set(open_handle.variables))

    def test_open_writes_population_metadata(self, open_handle):
        assert open_handle['population_count'][0] == self.N_PARTICLES
        assert open_handle['population_start_idx'][0] == 0

    def test_open_writes_flowfield_metadata(self, open_handle):
        name_chars = open_handle['flowfield_name'][0, :].data
        name = b''.join(name_chars).decode('ascii').strip()
        assert name == 'water_velocity'

    def test_record_writes_coordinates_to_correct_slot(self, writer, population):
        handle = writer.open_output(
            'stream.nc', self.N_SLOTS, self.N_PARTICLES,
            self.N_POPULATIONS, self.N_FLOWFIELDS, [population], ['vel'],
        )
        writer.record_output(handle, [population], slot_idx=0, current_time=100.0)
        writer.record_output(handle, [population], slot_idx=1, current_time=200.0)

        np.testing.assert_array_almost_equal(handle['x'][:, 0], population.particles['x'])
        np.testing.assert_array_almost_equal(handle['x'][:, 1], population.particles['x'])
        assert handle['time'][0, 0] == pytest.approx(100.0)
        assert handle['time'][0, 1] == pytest.approx(200.0)
        handle.close()

    def test_record_writes_status_fields(self, writer, population):
        handle = writer.open_output(
            'stream.nc', self.N_SLOTS, self.N_PARTICLES,
            self.N_POPULATIONS, self.N_FLOWFIELDS, [population], ['vel'],
        )
        writer.record_output(handle, [population], slot_idx=0, current_time=0.0)

        np.testing.assert_array_equal(
            handle['status_mobile'][:, 0], population.particles['status_mobile']
        )
        handle.close()

    def test_unwritten_slots_are_fill_values(self, writer, population):
        """Slots not yet written should contain the declared fill value, not zeros."""
        handle = writer.open_output(
            'stream.nc', self.N_SLOTS, self.N_PARTICLES,
            self.N_POPULATIONS, self.N_FLOWFIELDS, [population], ['vel'],
        )
        writer.record_output(handle, [population], slot_idx=0, current_time=0.0)
        # slot 1 is unwritten; returned as a masked array (fill_value=NaN)
        slot1 = handle['x'][:, 1]
        assert np.all(np.ma.getmaskarray(slot1))
        handle.close()

    def test_close_returns_correct_path(self, writer, population):
        handle = writer.open_output(
            'stream.nc', self.N_SLOTS, self.N_PARTICLES,
            self.N_POPULATIONS, self.N_FLOWFIELDS, [population], ['vel'],
        )
        path = writer.close_output(handle)
        assert path.name == 'stream.nc'
        assert path.exists()

    def test_streaming_round_trip(self, writer, population, tmp_path):
        """Full open-record-close cycle produces a valid, readable NetCDF file."""
        times = [0.0, 3600.0, 7200.0]
        handle = writer.open_output(
            'round_trip.nc', len(times), self.N_PARTICLES,
            self.N_POPULATIONS, self.N_FLOWFIELDS, [population], ['water_velocity'],
        )
        for slot, t in enumerate(times):
            writer.record_output(handle, [population], slot_idx=slot, current_time=t)
        path = writer.close_output(handle)

        # Read back with xarray and verify
        ds = xr.open_dataset(path)
        assert ds.sizes['n_timesteps'] == len(times)
        assert ds.sizes['n_particles'] == self.N_PARTICLES
        # time shape: (n_particles, n_timesteps); all particles share the same time per slot
        time_vals = ds['time'].values[0, :]   # particle 0 across all slots
        np.testing.assert_array_almost_equal(time_vals, times)
        np.testing.assert_array_almost_equal(
            ds['x'].values[:, 0], population.particles['x']  # all particles at slot 0
        )
        ds.close()

    def test_two_populations_particle_offsets(self, writer, tmp_path):
        """Particles from separate populations must land in consecutive index ranges."""
        pop_a = MockPopulation('pop_a')
        pop_b = MockPopulation('pop_b')
        pop_b.particles['x'] = np.array([10.0, 20.0, 30.0])
        n_total = len(pop_a.particles['x']) + len(pop_b.particles['x'])

        handle = writer.open_output(
            'two_pops.nc', 2, n_total, 2, 1, [pop_a, pop_b], ['vel'],
        )
        writer.record_output(handle, [pop_a, pop_b], slot_idx=0, current_time=0.0)

        np.testing.assert_array_almost_equal(handle['x'][:3, 0], pop_a.particles['x'])
        np.testing.assert_array_almost_equal(handle['x'][3:, 0], pop_b.particles['x'])
        handle.close()
