import matplotlib

matplotlib.use('Agg')

import numpy as np
import xarray as xr

from sedtrails.pathway_visualizer.trajectories import plot_trajectories

def test_plot_trajectories_accepts_fixed_width_population_names(monkeypatch):
    n_particles = 2
    n_timesteps = 3

    ds = xr.Dataset(
        data_vars={
            'x': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0, 2.0], [0.0, 1.5, 3.0]])),
            'y': (('n_particles', 'n_timesteps'), np.array([[0.0, 0.5, 1.0], [0.0, 0.25, 0.5]])),
            'time': (('n_particles', 'n_timesteps'), np.array([[0.0, 60.0, 120.0], [0.0, 60.0, 120.0]])),
            'population_id': (('n_particles',), np.array([0, 1], dtype=int)),
            # 1D fixed-width byte strings as produced by current writer path
            'population_name': (('n_populations',), np.array([b'pop_A', b'pop_B'], dtype='S24')),
        },
        coords={
            'n_particles': np.arange(n_particles),
            'n_timesteps': np.arange(n_timesteps),
            'n_populations': np.arange(2),
        },
    )

    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)

    # This previously raised: "too many indices"
    plot_trajectories(ds)
