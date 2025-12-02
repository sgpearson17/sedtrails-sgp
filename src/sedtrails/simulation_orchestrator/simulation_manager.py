import logging
import os
import sys
from typing import Any, Optional

import numpy as np
from pathlib import Path
import xarray as xr
from tqdm import tqdm

from sedtrails.application_interfaces.configuration_controller import ConfigurationController
from sedtrails.data_manager import DataManager
from sedtrails.exceptions.exceptions import ConfigurationError
from sedtrails.particle_tracer import ParticleSeeder
from sedtrails.particle_tracer.data_retriever import FieldDataRetriever  # Updated import
from sedtrails.particle_tracer.particle import Particle
from sedtrails.particle_tracer.timer import Duration, Time, Timer
from sedtrails.pathway_visualizer import SimulationDashboard
from sedtrails.simulation_orchestrator.global_logger import log_simulation_state, setup_logging
from sedtrails.transport_converter.format_converter import FormatConverter, SedtrailsData
from sedtrails.transport_converter.physics_converter import PhysicsConverter


class Simulation:
    """Class to encapsulate the particle simulation process."""

    def __init__(self, config_file: str, enable_dashboard: Optional[bool] = None):
        """
        Initialize the simulation with the given configuration.

        Parameters
        ----------
        config_file : str
            Path to the configuration file.
        enable_dashboard : bool, optional
            Override the dashboard setting from configuration. If None, uses config value.
        """
        self._config_file = config_file
        self._enable_dashboard_override = enable_dashboard

        self._start_time = None
        self._config_is_read = False
        self._populations_config = None

        # Validate config file exists early
        if not os.path.exists(config_file):
            raise ConfigurationError(f'Configuration file not found: {config_file}')

        # Try to read config and update logger directory
        try:
            # self._controller = ConfigurationController(self._config_file)
            self._controller = ConfigurationController(self._config_file)
            self._controller.load_config(self._config_file)

            # TODO: logger has a circular dependency with controller. The logger needs refactoring.
            self.logger = logging.getLogger(__name__)

            self._config_is_read = True

            # self.logger_manager.setup_logger()
            # self._controller.log_after_load_config()

        except Exception:
            # Global exception handler will catch and log this
            raise

        for population in self.populations_config:
            tracer_config = population['tracer_methods']
        # Initialize other components
        self.format_converter = FormatConverter(self._get_format_config())
        # Added population_config for the soulsby method
        self.physics_converter = PhysicsConverter(self._get_physics_config(), tracer_config)
        self.data_manager = DataManager(self._get_output_dir())
        self.data_manager.set_mesh()  # TODO: was this ever answered? is it needed?
        self.particles: list[Particle] = []  # List to hold particles
        self.dashboard = self._create_dashboard()  #
        self.writer = self.data_manager.writer

        setup_logging(output_dir=str(self.writer.output_dir))  # Initialize logging in the results directory
        self.logger = logging.getLogger(__name__)
        self.logger.info('Configuration loaded')

    def _create_dashboard(self):
        """Create and return a dashboard instance."""
        # Use override if provided, otherwise fall back to config setting
        dashboard_enabled = (
            self._enable_dashboard_override
            if self._enable_dashboard_override is not None
            else self._controller.get('visualization.dashboard.enable', False)
        )

        if dashboard_enabled:
            reference_date = self._controller.get('general.input_model.reference_date', '1970-01-01')
            figsize = (12, 8)
            dashboard = SimulationDashboard(reference_date=reference_date)
            dashboard.initialize_dashboard(figsize)
            # Force initial display and bring window to front
            dashboard.fig.show()
            dashboard.fig.canvas.draw()
            dashboard.fig.canvas.flush_events()

            # Try to bring window to front (cross-platform)
            try:
                dashboard.fig.canvas.manager.window.raise_()
                dashboard.fig.canvas.manager.window.activateWindow()
            except AttributeError:
                pass  # Some backends don't support this

            return dashboard
        else:
            return None

    def _get_format_config(self):
        """
        Returns configuration parameters required for the format converter.
        """

        format_config = {
            'input_file': self._controller.get('inputs.data'),
            'input_format': self._controller.get('general.input_model.format'),  # Specify the input format
            'reference_date': self._controller.get('general.input_model.reference_date'),
            'morfac': self._controller.get('general.input_model.morfac', 1.0),
        }

        return format_config

    def _get_output_dir(self):
        """
        Returns the output directory for the simulation.
        
        If the output directory is not explicitly specified or is a default value,
        uses the directory containing the config file as the base directory.
        
        Returns
        -------
        Path
            Path object representing the output directory
        """
        output_dir = self._controller.get('outputs.directory')
        
        # Check if output_dir is a default or relative path that should be relative to config file
        if output_dir is None or output_dir in ['.', './output', 'output', './results', 'results']:
            # Use the config file's directory as the base
            config_file_dir = Path(self._config_file).parent
            if output_dir in ['./output', 'output']:
                # Preserve the 'output' subdirectory but make it relative to config file
                output_dir = config_file_dir / 'output'
            elif output_dir in ['./results', 'results']:
                # Preserve the 'results' subdirectory but make it relative to config file
                output_dir = config_file_dir / 'results'
            else:
                # Use config file directory directly
                output_dir = config_file_dir
        else:
            # Convert to Path object for consistency
            output_dir = Path(output_dir)
        
        return output_dir

    def _get_physics_config(self):
        """
        Returns configuration parameters required for the physics converter.
        """
        from sedtrails.transport_converter.physics_converter import PhysicsConfig

        # TODO implement configuration controller for multiple populations in a robust way
        # TODO make sure the tracer_methods in config file match exactly with the plugin file name
        tracer_method = self._controller.get('particles.populations')[0].get('tracer_methods', 'vanwesten')
        if isinstance(tracer_method, dict):
            tracer_method = list(tracer_method.keys())[0]
        config = PhysicsConfig(
            tracer_method=tracer_method,
            gravity=self._controller.get('physics.constants.g', 9.81),
            von_karman_constant=self._controller.get('physics.constants.von_karman', 0.40),
            kinematic_viscosity=self._controller.get('physics.constants.kinematic_viscosity', 1.36e-6),
            water_density=self._controller.get('physics.constants.rho_w', 1027.0),
            particle_density=self._controller.get('physics.constants.rho_s', 2650.0),
            porosity=self._controller.get('physics.constants.porosity', 0.4),
            grain_diameter=self._controller.get('physics.constants.grain_diameter', 2.5e-4),
            morfac=self._controller.get('physics.constants.morphology_factor', 1.0),
            # trapped_exposed_method=self._controller.get('physics.trapped_exposed_method', 'reduced_velocity'), # other option; 'probabilistic_exposure'
        )

        return config

    @property
    def config(self):
        """
        Returns the full configuration settings for the simulation.
        """
        if not self._config_is_read:
            self._controller.load_config(self._config_file)
        return self._controller.get_config()  # delagates to the controller

    @property
    def populations_config(self):
        """
        Returns the configuration paramters for 'populations'.
        """
        if self._populations_config is None:
            self._populations_config = self.config.get('particles', {}).get('populations', {})
        return self._populations_config

    @property
    def start_time(self):
        """
        Get the start time parameter for the simulation.
        """
        if not self._start_time:
            self._start_time = self._controller.get('time.start_time')  # defaults to Unix epoch
        return self._start_time

    @property
    def flow_field(self) -> SedtrailsData:
        """
        Returns input flow field data in SedtrailsData format.
        """

        return self.format_converter.convert_to_sedtrails()

    def validate_config(self) -> bool:  # TODO: this is not used anywhere
        """
        Validates the configuration file.

        """
        if not self._config_is_read:  # assure config is read only once
            try:
                self._controller.load_config(self._config_file)
                self._config_is_read = True
                return True
            except Exception as e:
                raise ConfigurationError(f'Error validating configuration file: {e}')  # noqa: B904
                return False  # validation fails
            else:
                return True  # validation succeeds
        else:
            # if config is already read, the file is already validated
            return True

    def get_parameter(self, key: str) -> Any:  # TODO: this is not used anywhere. Is it needed?
        """
        Returns the value of a specific parameter in the configuration file.

        Parameters
        ----------
        key : str
            The dot-separated key to retrieve.

        Returns
        -------
        Any
            The value associated with the key in the configuration file.

        Raises
        ------
        Warning
            If the key is not found in the configuration file.
        """

        import warnings

        if not self._config_is_read:  # assure config is read only once
            self._controller.load_config(self._config_file)

        value = self._controller.get(key, None)
        if value is None:
            warnings.warn(f'Key "{key}" not found in configuration file', UserWarning, stacklevel=2)
        return value

    def run(self):
        """
        Executes the particle simulation workflow.
        """

        # Loading configuration
        if not self._config_is_read:  # assure config is read only once
            self._controller.load_config(self._config_file)
            self._config_is_read = True

        # Time configuration
        simulation_time = Time(
            _start=self._controller.get('time.start'),
            duration=Duration(self._controller.get('time.duration')),
            time_step=Duration(self._controller.get('time.timestep')),
            read_input_interval=Duration(self._controller.get('inputs.read_interval')),
        )

        timer = Timer(simulation_time=simulation_time, cfl_condition=self._controller.get('time.cfl_condition'))

        # Load SedTRAILS data for 'x' and 'y' (needed for population seerder)
        sedtrails_data = self.format_converter.convert_to_sedtrails(
            current_time=simulation_time._start, reading_interval=1
        )

        populations_config = self._controller.get('particles.populations', [])
        seeder = ParticleSeeder(populations_config)  # intialize seeder with population config
        populations = seeder.seed(sedtrails_data)  # seed particles for all populations using current sedtrails data

        # Set initial values
        sedtrails_data = None

        # Initialize progress bar
        pbar = tqdm(
            total=100,
            desc='Computing positions',
            unit='%',
            bar_format='{l_bar}{bar}| {n:.1f}% [{elapsed}<{remaining}, {postfix}]',
        )

        log_simulation_state(
            self.logger,
            {
                'status': 'simulation_started',
                'command': ' '.join(sys.argv),
                'python_version': sys.version.split()[0],
                'config_file': self._config_file,
                'working_directory': os.getcwd(),
            },
        )

        # Determine flow field names from configuration
        flow_field_names = []
        for population in populations_config:
            if 'tracer_methods' in population and (
                'vanwesten' in population['tracer_methods'] or 'soulsby' in population['tracer_methods']
            ):
                if 'vanwesten' in population['tracer_methods']:
                    flow_field_names = population['tracer_methods']['vanwesten']['flow_field_name']
                elif 'soulsby' in population['tracer_methods']:
                    flow_field_names = population['tracer_methods']['soulsby']['flow_field_name']
                break  # Use the first population's flow fields for now

        # Create SedTrails dataset using DataManager's writer (composition)
        total_particles = sum([len(pop.particles['x']) for pop in populations])
        estimated_timesteps = (simulation_time.duration.seconds // simulation_time.time_step.seconds) + 1
        max_timesteps = estimated_timesteps * 2  # Initial buffer

        xr_data = self.data_manager.writer.create_dataset(
            N_particles=total_particles,
            N_populations=len(populations),
            N_timesteps=max_timesteps,
            N_flowfields=len(flow_field_names) if flow_field_names else 1,
        )

        # Initialize metadata using DataManager's writer (composition)
        self.data_manager.writer.add_metadata(xr_data, populations, flow_field_names)

        # Main simulation loop with variable timestep
        while not timer.stop:
            # Check if current time is within loaded SedTRAILS data
            current_time_seconds = timer.current
            if sedtrails_data is None or current_time_seconds > sedtrails_data.times[-2]:
                # Avoid recreating SedTRAILS data if current time is before the first time step
                if sedtrails_data is not None and current_time_seconds < sedtrails_data.times[0]:
                    timer.advance()
                    continue
                # Convert to SedTRAILS format
                sedtrails_data = self.format_converter.convert_to_sedtrails(
                    current_time=current_time_seconds, reading_interval=simulation_time.read_input_interval.seconds
                )

                # Convert physics fields with transport probability configuration
                for pop in self.populations_config:
                    self.physics_converter.convert_physics(
                        sedtrails_data=sedtrails_data,
                        transport_probability_method=pop.get('transport_probability'),
                    )

                # Create new FieldDataRetriever with updated data
                retriever = FieldDataRetriever(sedtrails_data)  # TODO: should the retriever only be created once?

            # TODO: integrate loop over flow fields into CFL Condition
            # Collect flow fields for CFL computation
            tracer_methods = {}
            # TODO: this loops over populations_config, but only the last one is used. This must be fixed
            # to handle multiple populations with different tracer methods and flow fields
            for population in populations_config:
                tracer_methods = population['tracer_methods']

            flow_data_list = []
            for flow_field_name in flow_field_names:
                flow_data_list.append(retriever.get_flow_field(timer.current, flow_field_name))

            # Compute CFL-based timestep across all flow fields
            timer.compute_cfl_timestep(flow_data_list, sedtrails_data)

            # Main loop
            for population in populations:
                for _method in tracer_methods:
                    for flow_field_name in flow_field_names:
                        # Obtain scalar field information
                        # TODO: Consider moving van westen specific fields to the plugin itself
                        mixing_depth = retriever.get_scalar_field(timer.current, 'mixing_layer_thickness')['magnitude']
                        bed_level = retriever.get_scalar_field(timer.current, 'bed_level')['magnitude']
                        if self.physics_converter.config.tracer_method == 'vanwesten':
                            transport_prob = retriever.get_scalar_field(
                                timer.current, flow_field_name.replace('velocity', 'probability')
                            )['magnitude']
                        else:  # soulsby
                            transport_prob = np.ones_like(bed_level)

                        # Update information at particle positions
                        population.update_information(
                            current_time=timer.current,
                            mixing_depth=mixing_depth,
                            bed_level=bed_level,
                            transport_probability=transport_prob,
                        )

                        # Update particle burial depth
                        if self.physics_converter.config.tracer_method == 'vanwesten':
                            population.update_burial_depth()

                        # Determining status
                        population.update_status()

                        # Get flow field information
                        flow_field = retriever.get_flow_field(timer.current, flow_field_name)

                        # Update particle position
                        population.update_position(flow_field=flow_field, current_timestep=timer.current_timestep)

            # Collect data from all populations for this timestep using DataManager
            self.data_manager.collect_timestep_data(xr_data, populations, timer.step_count, timer.current)

            # Check if we need to expand the time dimension
            if timer.step_count >= max_timesteps - 10:  # 10-step safety margin
                old_max = max_timesteps
                max_timesteps = int(max_timesteps * 1.5)  # Expand by 50%
                self.logger.info(f'Expanding time dimension from {old_max} to {max_timesteps}')
                xr_data = self._expand_time_dimension(xr_data, max_timesteps)

            # Update dashboard if enabled
            if self.dashboard is not None:
                # For dashboard, use first population data
                first_population = populations[0]
                particle_data = {
                    'x': first_population.particles['x'],
                    'y': first_population.particles['y'],
                    'burial_depth': first_population.particles['burial_depth'],
                    'mixing_depth': first_population.particles['mixing_depth'],
                }
                # Get bathymetry data
                bathymetry = retriever.get_scalar_field(timer.current, 'bed_level')['magnitude']

                # Particle data including burial_depth and mixing_depth

                # Get simulation timing
                # plot_interval_str = self._controller.get('output.plot_interval', '1H')
                plot_interval_seconds = 3600  # self._parse_duration(plot_interval_str)

                # Update dashboard with timing info

                self.dashboard.update(
                    flow_field,
                    bathymetry,
                    particle_data,
                    timer.current,
                    timer.current_timestep,
                    plot_interval_seconds,
                    simulation_start_time=simulation_time.start,  # Add this
                    simulation_end_time=simulation_time.end,  # Add this
                )

            timer.advance()

            # Saving and plotting
            # TODO: enable saving and plotting again: addapt writer with structure issue 297
            # TODO: remove default insertion on configuration retrieval
            # interval_output = self._controller.get('output.interval_output', '1H')
            # interval_plot = self._controller.get('output.interval_plot', '1D')

            # # Data manager
            # if timer.step_count == 1 or (timer.current - simulation_time.start) // interval_output > (
            #     (timer.current - simulation_time.start - timer.current_timestep) // interval_output
            # ):
            #     # self.data_manager.add_data()

            # # Plotting
            # TODO: enable plotting from saved data file and from memory.
            # if timer.step_count == 1 or (timer.current - simulation_time.start) // interval_plot > (
            #     (timer.current - simulation_time.start - timer.current_timestep) // interval_plot
            # ):
            #     plot_particle_trajectory(
            #         flow_data=flow_field,
            #         trajectory_x=self.data_manager.get_trajectory_x(),
            #         trajectory_y=self.data_manager.get_trajectory_y(),
            #         title=f'Particle Trajectory - {simulation_time.duration.seconds} seconds, {timer.step_count} steps',
            #         save_path=f'{self.data_manager.output_dir}/trajectory_plot_{timer.step_count:05d}.png',
            #     )

            # Calculate progress percentage based on simulation time
            # elapsed_time = timer.current - simulation_time.start
            # progress_percent = (elapsed_time / simulation_time.duration.seconds) * 100

            # # Update progress bar
            # pbar.n = progress_percent
            # pbar.set_postfix(
            #     {
            #         'Step': timer.step_count,
            #         'Time': f'{timer.current:.0f}s',
            #         'dt': f'{timer.current_timestep:.2f}s',
            #     }
            # )
            # Update progress bar
            if simulation_time.duration.seconds > 0:  # Avoid undefined progress when duration is zero
                elapsed_time = timer.current - simulation_time.start
                progress_percent = (elapsed_time / simulation_time.duration.seconds) * 100
                pbar.update(progress_percent - pbar.n)  # increment by delta
            else:
                pbar.update(0)  # avoid ZeroDivisionError if duration is 0
            pbar.set_postfix(
                {
                    'Step': timer.step_count,
                    'Time': f'{timer.current:.0f}s',
                    'dt': f'{timer.current_timestep:.2f}s',
                }
            )

        # End of Simulation
        pbar.close()

        # Keep dashboard open after simulation ends

        if self.dashboard is not None:
            print('\nSimulation completed successfully!')
            self.dashboard.keep_window_open()

        # Finalize results
        # self.data_manager.dump()  # Write remaining data to disk. # TODO: not working

        # Write final results to NetCDF using DataManager's writer (composition)
        actual_timesteps = timer.step_count + 1
        output_file = self.data_manager.writer.write(
            xr_data, filename='sedtrails_results.nc', trim_to_actual_timesteps=True, actual_timesteps=actual_timesteps
        )
        print(f'Simulation results saved to: {output_file}')



    def _expand_time_dimension(self, xr_data: xr.Dataset, new_max_timesteps: int) -> xr.Dataset:
        """
        Expand the time dimension of the xarray dataset to accommodate more timesteps.

        This method is called when the number of timesteps approaches the allocated buffer size
        due to adaptive CFL-based timestepping. It pads all time-dependent data variables with
        NaN values and updates the time coordinate accordingly.

        Parameters
        ----------
        xr_data : xr.Dataset
            The xarray dataset containing simulation results with time dimension.
        new_max_timesteps : int
            The new maximum number of timesteps to allocate. Must be larger than the current
            time dimension size.

        Returns
        -------
        xr.Dataset
            The expanded dataset with additional timesteps allocated along the time dimension.
            New timestep values are filled with NaN.

        Notes
        -----
        - Only data variables that have a 'time' dimension are expanded.
        - The time coordinate is updated to range from 0 to new_max_timesteps - 1.
        - All newly allocated timesteps are filled with NaN values.
        - This operation creates a copy of the dataset data in memory.

        Examples
        --------
        >>> # Expand dataset from 1000 to 1500 timesteps
        >>> expanded_data = self._expand_time_dimension(xr_data, 1500)
        """


        current_size = len(xr_data.time)
        additional_steps = new_max_timesteps - current_size

        # Create a dictionary to hold expanded data variables
        expanded_vars = {}

        # Pad all data variables along time dimension
        for var_name in xr_data.data_vars:
            var = xr_data[var_name]

            if 'time' in var.dims:
                # Get the shape and create padding
                pad_shape = list(var.shape)
                time_dim_idx = var.dims.index('time')
                pad_shape[time_dim_idx] = additional_steps

                # Create NaN-filled array for padding
                pad_data = np.full(pad_shape, np.nan, dtype=var.dtype)

                # Create DataArray for padding with the same dimensions (except time)
                # Build coordinates for the padded array
                pad_coords = {}
                for dim in var.dims:
                    if dim == 'time':
                        pad_coords[dim] = np.arange(current_size, new_max_timesteps)
                    else:
                        pad_coords[dim] = var.coords[dim]

                pad_array = xr.DataArray(pad_data, dims=var.dims, coords=pad_coords)

                # Concatenate along time dimension
                expanded_vars[var_name] = xr.concat([var, pad_array], dim='time')
            else:
                # Keep non-time variables as is
                expanded_vars[var_name] = var

        # Create new dataset with expanded variables and all original coordinates (replace 'time')
        expanded_dataset = xr.Dataset(
            expanded_vars,
            coords={coord: (np.arange(new_max_timesteps) if coord == 'time' else xr_data.coords[coord]) for coord in xr_data.coords}
        )
        # Copy attributes
        expanded_dataset.attrs = xr_data.attrs.copy()
        for var_name in xr_data.data_vars:
            if var_name in expanded_dataset.data_vars:
                expanded_dataset[var_name].attrs = xr_data[var_name].attrs.copy()

        return expanded_dataset


# if __name__ == '__main__':
#     sim = Simulation(config_file='examples/config.example_natascia.yaml')
#     sim.run()

#     # NOTE: This will failed on the output saving. But that's success
