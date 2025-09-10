import numpy as np
import xarray as xr


def create_sedtrails_dataset(N_particles, N_populations, N_timesteps, N_flowfields, name_strlen=16):
    ds = xr.Dataset(
        {
            # Population metadata
            'population_name': (('n_populations', 'name_strlen'), np.chararray((N_populations, name_strlen))),
            'population_particle_type': ('n_populations', np.random.randint(0, 3, N_populations)),
            'population_start_idx': ('n_populations', np.arange(N_populations)),
            'population_count': ('n_populations', np.random.randint(1, N_particles, N_populations)),
            # Trajectory metadata
            'trajectory_id': (('n_particles', 'name_strlen'), np.chararray((N_particles, name_strlen))),
            'population_id': ('n_particles', np.random.randint(0, N_populations, N_particles)),
            # Core trajectory variables
            'time': (('n_particles', 'n_timesteps'), np.random.rand(N_particles, N_timesteps)),
            'x': (('n_particles', 'n_timesteps'), np.random.rand(N_particles, N_timesteps)),
            'y': (('n_particles', 'n_timesteps'), np.random.rand(N_particles, N_timesteps)),
            'z': (('n_particles', 'n_timesteps'), np.random.rand(N_particles, N_timesteps)),
            'burial_depth': (('n_particles', 'n_timesteps'), np.random.rand(N_particles, N_timesteps)),
            'mixing_depth': (('n_particles', 'n_timesteps'), np.random.rand(N_particles, N_timesteps)),
            # Status variables
            'status_alive': (('n_particles', 'n_timesteps'), np.random.randint(0, 2, (N_particles, N_timesteps))),
            'status_buried': (('n_particles', 'n_timesteps'), np.random.randint(0, 2, (N_particles, N_timesteps))),
            'status_domain': (('n_particles', 'n_timesteps'), np.random.randint(0, 2, (N_particles, N_timesteps))),
            'status_transported': (('n_particles', 'n_timesteps'), np.random.randint(0, 2, (N_particles, N_timesteps))),
            'status_released': (('n_particles', 'n_timesteps'), np.random.randint(0, 2, (N_particles, N_timesteps))),
            'status_mobile': (('n_particles', 'n_timesteps'), np.random.randint(0, 2, (N_particles, N_timesteps))),
            # Transport distances per flow field
            'covered_distance': (
                ('n_flowfields', 'n_particles', 'n_timesteps'),
                np.random.rand(N_flowfields, N_particles, N_timesteps),
            ),
            'flowfield_name': (('n_flowfields', 'name_strlen'), np.chararray((N_flowfields, name_strlen))),
        },
        coords={
            'n_particles': np.arange(N_particles),
            'n_populations': np.arange(N_populations),
            'n_timesteps': np.arange(N_timesteps),
            'n_flowfields': np.arange(N_flowfields),
            'name_strlen': np.arange(name_strlen),
        },
    )

    return ds
