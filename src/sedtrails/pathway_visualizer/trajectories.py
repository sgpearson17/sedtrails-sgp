"""
Module for visualizing particle trajectories from SedTRAILS NetCDF output files.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def read_netcdf(results_file_path: Path) -> xr.Dataset:
    """Read a SedTRAILS NetCDF file and return the xarray Dataset.

    Parameters
    ----------
    results_file_path : pathlib.Path
        Path to the SedTRAILS NetCDF file.

    Returns
    -------
    xr.Dataset
        The xarray Dataset containing the SedTRAILS results.

    """
    results_file_path = Path(results_file_path)
    if not results_file_path.exists():
        raise FileNotFoundError(f"NetCDF file '{results_file_path}' not found.")
    ds = xr.open_dataset(results_file_path)
    return ds


def plot_trajectories(ds, save_plot=False, output_dir=None):
    """Plot particle trajectories from the NetCDF dataset."""

    # Extract trajectory data
    x_data = ds['x'].values  # shape: (n_particles, n_timesteps)
    y_data = ds['y'].values  # shape: (n_particles, n_timesteps)
    time_data = ds['time'].values  # shape: (n_particles, n_timesteps)

    n_particles, n_timesteps = x_data.shape

    print(f'\nPlotting trajectories for {n_particles} particles over {n_timesteps} timesteps...')

    # Create the plot with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # Extract population information for color coding
    population_ids = ds['population_id'].values if 'population_id' in ds else np.zeros(n_particles, dtype=int)
    n_populations = int(ds.sizes['n_populations']) if 'n_populations' in ds.sizes else 1

    # Get population names
    population_names = []
    if 'population_name' in ds:
        for i in range(n_populations):
            name_bytes = ds['population_name'][i, :].values
            name = ''.join([char.decode('utf-8') for char in name_bytes if char != b'\x00']).strip()
            population_names.append(name)
    else:
        population_names = [f'Population {i}' for i in range(n_populations)]

    # Define colors for populations
    pop_colors = plt.cm.Set1(np.linspace(0, 1, n_populations))

    # Plot 1: All trajectories on spatial map (individual particle colors)
    ax1.set_title(f'(a) Particle Trajectories - Individual Colors (n={n_particles})')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')

    # Plot each particle trajectory
    try:
        cmap = plt.cm.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, n_particles))
    except (AttributeError, ValueError):
        # Fallback to a basic color cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [color_cycle[i % len(color_cycle)] for i in range(n_particles)]

    # Define colors for populations
    try:
        pop_cmap = plt.cm.get_cmap('Set1')
        pop_colors = pop_cmap(np.linspace(0, 1, n_populations))
    except (AttributeError, ValueError):
        # Fallback to basic colors
        base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        pop_colors = [base_colors[i % len(base_colors)] for i in range(n_populations)]

    # Plot 1: All trajectories on spatial map (individual particle colors)
    ax1.set_title(f'(a) Particle Trajectories - Individual Colors (n={n_particles})')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')

    # Plot each particle trajectory
    try:
        cmap = plt.cm.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, n_particles))
    except (AttributeError, ValueError):
        # Fallback to a basic color cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [color_cycle[i % len(color_cycle)] for i in range(n_particles)]

    for i in range(n_particles):
        # Filter out NaN values (particles may not exist for all timesteps)
        mask = ~(np.isnan(x_data[i, :]) | np.isnan(y_data[i, :]))
        if np.any(mask):
            x_traj = x_data[i, mask]
            y_traj = y_data[i, mask]

            # Plot trajectory line
            ax1.plot(
                x_traj,
                y_traj,
                color=colors[i],
                alpha=0.7,
                linewidth=1,
                label=f'Particle {i}' if n_particles <= 10 else None,
            )

            # Mark start and end points
            ax1.scatter(
                x_traj[0], y_traj[0], color=colors[i], marker='o', s=50, edgecolor='black', linewidth=1, zorder=5
            )
            ax1.scatter(
                x_traj[-1], y_traj[-1], color=colors[i], marker='s', s=50, edgecolor='black', linewidth=1, zorder=5
            )

    # Add legend if few particles
    if n_particles <= 10:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Plot 2: Time series of distances from initial position
    ax2.set_title('(b) Distance from Initial Position vs Time')
    ax2.set_xlabel('Time [hours]')
    ax2.set_ylabel('Distance from Initial Position [m]')

    # Find the minimum time across all particles to use as reference
    min_time = np.nanmin(time_data)

    for i in range(n_particles):
        # Calculate distance from initial position for each timestep
        mask = ~(np.isnan(x_data[i, :]) | np.isnan(y_data[i, :]))
        if np.any(mask):
            x_traj = x_data[i, mask]
            y_traj = y_data[i, mask]
            time_traj = time_data[i, mask]

            # Get initial position (first valid point)
            x0 = x_traj[0]
            y0 = y_traj[0]

            # Convert time to hours relative to simulation start
            time_hours = (time_traj - min_time) / 3600.0

            # Calculate distance from initial position
            distances = np.sqrt((x_traj - x0) ** 2 + (y_traj - y0) ** 2)
            ax2.plot(time_hours, distances, color=colors[i], alpha=0.7, linewidth=1)

    ax2.grid(True, alpha=0.3)

    # Plot 3: Trajectories colored by population
    ax3.set_title('(c) Particle Trajectories - Colored by Population')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')

    # Plot trajectories grouped by population
    for pop_idx in range(n_populations):
        particles_in_pop = []
        # Find particles belonging to this population
        for i in range(n_particles):
            if population_ids[i] == pop_idx:
                particles_in_pop.append(i)

        # Plot trajectories for this population
        for i in particles_in_pop:
            mask = ~(np.isnan(x_data[i, :]) | np.isnan(y_data[i, :]))
            if np.any(mask):
                x_traj = x_data[i, mask]
                y_traj = y_data[i, mask]

                # Plot trajectory line with population color
                ax3.plot(
                    x_traj,
                    y_traj,
                    color=pop_colors[pop_idx],
                    alpha=0.7,
                    linewidth=1,
                    label=population_names[pop_idx] if i == particles_in_pop[0] else '',
                )

                # Mark start and end points
                ax3.scatter(
                    x_traj[0],
                    y_traj[0],
                    color=pop_colors[pop_idx],
                    marker='o',
                    s=50,
                    edgecolor='black',
                    linewidth=1,
                    zorder=5,
                )
                ax3.scatter(
                    x_traj[-1],
                    y_traj[-1],
                    color=pop_colors[pop_idx],
                    marker='s',
                    s=50,
                    edgecolor='black',
                    linewidth=1,
                    zorder=5,
                )

    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')

    # Plot 4: Distance from initial position by population with statistics
    ax4.set_title('(d) Distance from Initial Position by Population')
    ax4.set_xlabel('Time [hours]')
    ax4.set_ylabel('Distance from Initial Position [m]')

    # Create containers for population statistics
    population_stats = {}

    # First pass: collect all data for each population
    for pop_idx in range(n_populations):
        population_stats[pop_idx] = {
            'times': [],
            'distances': [],
            'name': population_names[pop_idx],
            'color': pop_colors[pop_idx],
        }

        # Find particles belonging to this population
        particles_in_pop = [i for i in range(n_particles) if population_ids[i] == pop_idx]

        # Plot individual particle distances for this population
        for i in particles_in_pop:
            mask = ~(np.isnan(x_data[i, :]) | np.isnan(y_data[i, :]))
            if np.any(mask):
                x_traj = x_data[i, mask]
                y_traj = y_data[i, mask]
                time_traj = time_data[i, mask]

                # Get initial position
                x0 = x_traj[0]
                y0 = y_traj[0]

                # Convert time to hours relative to simulation start
                time_hours = (time_traj - min_time) / 3600.0

                # Calculate distance from initial position
                distances = np.sqrt((x_traj - x0) ** 2 + (y_traj - y0) ** 2)

                # Plot individual particle line (thin, transparent)
                ax4.plot(time_hours, distances, color=pop_colors[pop_idx], alpha=0.3, linewidth=0.8)

                # Store data for statistics
                population_stats[pop_idx]['times'].append(time_hours)
                population_stats[pop_idx]['distances'].append(distances)

    # Second pass: compute and plot population statistics
    for pop_idx in range(n_populations):
        if population_stats[pop_idx]['times']:
            # Create a common time grid for interpolation
            all_times = np.concatenate(population_stats[pop_idx]['times'])
            min_t, max_t = np.min(all_times), np.max(all_times)
            common_time = np.linspace(min_t, max_t, 100)

            # Interpolate all particle distances onto common time grid
            interpolated_distances = []
            for time_arr, dist_arr in zip(
                population_stats[pop_idx]['times'], population_stats[pop_idx]['distances'], strict=True
            ):
                if len(time_arr) > 1:  # Need at least 2 points for interpolation
                    interp_dist = np.interp(common_time, time_arr, dist_arr)
                    interpolated_distances.append(interp_dist)

            if interpolated_distances:
                # Convert to array for easy statistics
                distances_array = np.array(interpolated_distances)

                # Compute mean and standard deviation
                mean_distances = np.mean(distances_array, axis=0)
                std_distances = np.std(distances_array, axis=0)

                # Plot mean line (thick)
                ax4.plot(
                    common_time,
                    mean_distances,
                    color=pop_colors[pop_idx],
                    linewidth=3,
                    label=f'{population_names[pop_idx]} (mean)',
                )

                # Plot standard deviation bands
                ax4.fill_between(
                    common_time,
                    mean_distances - std_distances,
                    mean_distances + std_distances,
                    color=pop_colors[pop_idx],
                    alpha=0.2,
                    label=f'{population_names[pop_idx]} (±1σ)',
                )

    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot if requested
    if save_plot:
        if output_dir is None or output_dir == '.':
            # Extract the source file path from the dataset encoding
            source_file = ds.encoding.get('source', '.')
            output_dir = Path(source_file).parent
        else:
            output_dir = Path(output_dir)

        output_file = output_dir / 'particle_trajectories.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'Plot saved to: {output_file}')

    plt.show()
