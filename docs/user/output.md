# Simulation Output

:::warning
Explain the output file and its content
:::

At the end of the simulation, an output file ``sedtrails_results.nc`` is written in [netcdf](https://www.unidata.ucar.edu/software/netcdf) (``*.nc``) format. The files have generally been written in the spirit of [CF conventions for multidimensional arrays of trajectories](https://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#_multidimensional_array_representation_of_trajectories).

## Output File Inspection

By entering the command ``sedtrails inspect -f C:\your-filepath-here\sedtrails_results.nc``, you can check the data written to your file. For the example file, the following is displayed in the terminal:

```
  status_buried: ('n_particles', 'n_timesteps') int64 (15, 880)
  status_domain: ('n_particles', 'n_timesteps') int64 (15, 880)
  status_transported: ('n_particles', 'n_timesteps') int64 (15, 880)
  status_released: ('n_particles', 'n_timesteps') int64 (15, 880)
  status_mobile: ('n_particles', 'n_timesteps') int64 (15, 880)
  covered_distance: ('n_flowfields', 'n_particles', 'n_timesteps') float64 (2, 15, 880)
  flowfield_name: ('n_flowfields', 'name_strlen') |S1 (2, 24)
```

This indicates the variables written (e.g., ``status_buried``) as well as the dimensions and data types of each variable. For instance, it indicates that the ``covered_distance`` variable is stored as a ``float64`` number for 15 ``'n_particles'`` transported by 2 ``'n_flowfields'`` at 880 ``'n_timesteps'``.

## Output File Contents


