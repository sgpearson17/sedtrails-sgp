import numpy as np
import pytest

from sedtrails.data_manager.xarray_dataset import (
    collect_timestep_data,
    create_sedtrails_dataset,
    populate_population_metadata,
)


class MockPopulation:
    """Minimal population stub exposing particle arrays used by dataset export."""

    def __init__(self, name='population', repr_volume=np.nan):
        """Provides deterministic particle state for a single-timestep test."""
        self.name = name
        self.repr_volume = repr_volume
        # Keys mirror the particle fields consumed by collect_timestep_data.
        self.particles = {
            'x': np.array([1.0, 2.0]),
            'y': np.array([3.0, 4.0]),
            'burial_depth': np.array([0.0, 0.1]),
            'status_mobile': np.array([True, False]),
        }


def test_collect_timestep_data_exports_status_mobile():
    """Checks boolean mobility flags are written as integer status values."""
    ds = create_sedtrails_dataset(N_particles=2, N_populations=1, N_timesteps=1, N_flowfields=1)

    collect_timestep_data(ds, [MockPopulation()], timestep=0, current_time=12.0)

    np.testing.assert_array_equal(ds['status_mobile'].values[:, 0], np.array([1, 0]))


def test_collect_timestep_data_requires_status_mobile():
    """Missing status_mobile should fail instead of exporting silent all-zero mobility."""
    population = MockPopulation()
    del population.particles['status_mobile']
    ds = create_sedtrails_dataset(N_particles=2, N_populations=1, N_timesteps=1, N_flowfields=1)

    with pytest.raises(KeyError, match="status_mobile"):
        collect_timestep_data(ds, [population], timestep=0, current_time=12.0)


def test_create_sedtrails_dataset_includes_population_repr_volume():
    """Population representative-volume metadata should be present by default."""
    ds = create_sedtrails_dataset(N_particles=2, N_populations=1, N_timesteps=1, N_flowfields=1)

    assert 'population_repr_volume' in ds
    assert np.isnan(ds['population_repr_volume'].values[0])


def test_populate_population_metadata_writes_repr_volume():
    """Population metadata export should include representative volume values."""
    ds = create_sedtrails_dataset(N_particles=2, N_populations=1, N_timesteps=1, N_flowfields=1)
    population = MockPopulation(name='sand', repr_volume=12.5)

    populate_population_metadata(ds, [population])

    assert ds['population_repr_volume'].values[0] == pytest.approx(12.5)
