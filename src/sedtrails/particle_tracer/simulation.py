import time
import numpy as np

from sedtrails.transport_converter.format_converter import FormatConverter, SedtrailsData
from sedtrails.transport_converter.physics_converter import PhysicsConverter
from sedtrails.particle_tracer.data_retriever import FlowFieldDataRetriever
from sedtrails.particle_tracer.particle import Sand
from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator
from sedtrails.configuration_interface.configuration_controller import ConfigurationController
from typing import Any


class Simulation:
    """Class to encapsulate the particle simulation process."""

    def __init__(self, config_file: str):
        """
        Initialize the simulation with the given configuration.

        Parameters
        ----------
        config_file : str
            Path to the configuration file.
        """
        self._config_file = config_file

        self._start_time = None
        self._config_is_read = False

        # Lazy initialization of controllers and converters
        self._controller = ConfigurationController(self._config_file)
        self.format_converter = FormatConverter(self._get_format_config())
        self.physics_converter = PhysicsConverter(self._get_physics_config())
        self.data_manager = ''

    def _get_format_config(self):
        """
        Returns configuration parameters required for the format converter.
        """

        format_config = {
            'input_file': self._controller.get('folder_settings.input_data'),
            'input_format': self._controller.get('general.input_model.format'),  # Specify the input format
            'reference_date': self._controller.get(
                'general.input_model.reference_date'
            ),  # TODO: shall this be added to the json schema??
        }

        return format_config

    def _get_physics_config(self):
        """
        Returns configuration parameters required for the physics converter.
        """
        from sedtrails.transport_converter.physics_converter import PhysicsConfig

        config = PhysicsConfig(
            tracer_method=self._controller.get('physics.tracer_method', 'van_westen'),
            gravity=self._controller.get('physics.constants.g', 9.81),
            von_karman_constant=self._controller.get('physics.constants.von_karman', 0.40),
            kinematic_viscosity=self._controller.get('physics.constants.kinematic_viscosity', 1.36e-6),
            water_density=self._controller.get('physics.constants.rho_w', 1027.0),
            particle_density=self._controller.get('physics.constants.rho_s', 2650.0),
            # TODO: These parameters are missing from json schema.
            porosity=self._controller.get('physics.porosity', 0.4),
            grain_diameter=self._controller.get('physics.grain_diameter', 2.5e-4),
            morfac=self._controller.get('physics.morfac', 1.0),
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
    def start_time(self):
        """
        Get the start time for the simulation.
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

    def get_parameter(self, key: str) -> Any:
        """
        Returns the value of a specific parameter in the configuration file.

        Parameters
        ----------
        key : str
            The dot-separted key to retrieve.

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

        if not self._config_is_read:
            self._controller.load_config(self._config_file)

        value = self._controller.get(key, None)
        if value is None:
            warnings.warn(f'Key "{key}" not found in configuration file', UserWarning, stacklevel=2)
        return value

    def run(self):
        """
        Run the particle simulation using both original and Numba-optimized implementations.
        """

        if not self._config_is_read:  # assure config is read only once
            print('Reading configuration file...')
            self._controller.load_config(self._config_file)
            self._config_is_read = True

        sedtrails_data = self.format_converter.convert_to_sedtrails()
        # Add physics calculations to the SedtrailsData
        self.physics_converter.convert_physics(sedtrails_data)

        print('Data conversion completed')

        # Initialize flow field data retriever
        retriever = FlowFieldDataRetriever(sedtrails_data)
        retriever.flow_field_name = 'suspended_velocity'  # TODO: shouldn't this be read from config?

        initial_time = sedtrails_data.times[2]  #  returns seconds since reference date as floats
        # initial_flow = retriever.get_flow_field(initial_time)  # expects secondes since reference date

        # # retrieve start time from configuration or use default
        # START_TIME = self._controller.get('time.start_time')

        from sedtrails.particle_tracer.timer import Duration

        TIMESTEP = Duration(self._controller.get('time.timestep'))
        # TODO: should be managed by data manager

        OUTPUT_DIR = self._controller.get('folder_settings.output_dir')

        # Start at the 3rd timestep (index 2)
        TIMESTEP_INDEX = 2

        # Get the initial flow field at specified timestep
        # TODO: this should be handled by time class
        initial_time = sedtrails_data.times[TIMESTEP_INDEX]
        # Duration: 6.333 hours = 6 hours and 20 minutes = 380 minutes = 22,800 seconds
        print(self._controller.get('time.duration'))

        DURATION = Duration(self._controller.get('time.duration'))

        DURATION_SECONDS = DURATION.to_seconds()  # Total duration in seconds

        # Calculate number of steps
        NUM_STEPS = int(DURATION_SECONDS / TIMESTEP.to_seconds())

        # Particle seeding parameters
        # TODO: this should be handle by the seeding tool.
        particle_positions = {}
        strategy = self._controller.get('particle_seeding.strategy')
        if 'point' in strategy:
            id = 1
            for point in strategy['point']['points']:
                _point = point.split(',')
                particle_positions[str(id)] = (_point[0], _point[1])
                id += 1

        # Create a particle at the specified position
        # TODO: this should be handled by the seeding tool.
        # ===== STEP 4: Simulation with Numba Implementation =====
        print('\n=== STEP 4: Running Simulation with Numba Implementation ===')
        particles = []
        for id, (x, y) in particle_positions.items():
            START_X = float(x)
            START_Y = float(y)
            particle = Sand(id=id, _x=START_X, _y=START_Y, name='Test Particle')
            print(f'Created particle at position ({particle.x:.2f}, {particle.y:.2f})')
            particles.append(particle)

        print(f'Starting time at index {TIMESTEP_INDEX}: {initial_time} seconds')
        # print(f'Will run for {DURATION_HOURS:.3f} hours ({NUM_STEPS} steps) with timestep {TIMESTEP} seconds')

        # Store trajectory
        trajectory_numba_x = [particles[0].x]  # TODO: just handle multiple particles
        trajectory_numba_y = [particles[0].y]

        # First get the initial flow data to create calculator
        flow_data = retriever.get_flow_field(initial_time)

        # Create and compile the numba calculator - this will include compilation time
        print('Creating and compiling Numba calculator...')
        compile_start = time.time()
        numba_calc = create_numba_particle_calculator(grid_x=flow_data['x'], grid_y=flow_data['y'])
        compile_time = time.time() - compile_start
        print(f'Numba calculator compiled in {compile_time:.4f} seconds')

        # Warm up with one calculation to trigger JIT compilation
        print('Warming up JIT compilation...')
        warmup_start = time.time()
        _ = numba_calc['update_particles'](
            np.array([particles[0].x]),
            np.array([particles[0].y]),
            flow_data['u'],
            flow_data['v'],
            TIMESTEP.to_seconds(),
        )
        warmup_time = time.time() - warmup_start
        print(f'JIT warm-up completed in {warmup_time:.4f} seconds')

        # Start timer after compilation
        simulation_start = time.time()
        current_time = initial_time

        for step in range(1, NUM_STEPS + 1):
            # Update current time
            current_time = initial_time + step * TIMESTEP.to_seconds()

            # Get flow field at current time
            flow_data = retriever.get_flow_field(current_time)

            # Update particle position using Numba calculator
            new_x, new_y = numba_calc['update_particles'](
                np.array([particles[0].x]),
                np.array([particles[0].y]),
                flow_data['u'],
                flow_data['v'],
                TIMESTEP.to_seconds(),
            )

            # Update particle with new position
            particles[0].x = new_x[0]
            particles[0].y = new_y[0]

            # Store trajectory
            trajectory_numba_x.append(particles[0].x)
            trajectory_numba_y.append(particles[0].y)

            # Print progress every 20%
            if step % max(1, NUM_STEPS // 5) == 0:
                percent_complete = (step / NUM_STEPS) * 100
                elapsed_hours = step * TIMESTEP.to_seconds() / 3600
                print(
                    f'Step {step}/{NUM_STEPS} ({percent_complete:.1f}%) - '
                    f'Time: {elapsed_hours:.2f} hours - '
                    f'Position: ({particles[0].x:.2f}, {particles[0].y:.2f})'
                )

        numba_time = time.time() - simulation_start
        print(f'Numba implementation completed in {numba_time:.4f} seconds')
        print(f'Average speed: {NUM_STEPS / numba_time:.1f} steps/second')
        print(f'Final position: ({particles[0].x:.4f}, {particles[0].y:.4f})')

        # ===== STEP 4: Visualize Results =====

        print('\n=== STEP: Visualizing Results ===')

        # Convert trajectory to numpy arrays
        trajectory_numba_x = np.array(trajectory_numba_x)
        trajectory_numba_y = np.array(trajectory_numba_y)

        # Plot flow field with particle trajectory using the function
        final_flow = retriever.get_flow_field(current_time)

        from sedtrails.pathway_visualizer.visualization_utils import plot_particle_trajectory

        plot_particle_trajectory(
            flow_data=final_flow,
            trajectory_x=trajectory_numba_x,
            trajectory_y=trajectory_numba_y,
            title=f'Particle Trajectory - {DURATION_SECONDS} seconds, {NUM_STEPS} steps',
            save_path=OUTPUT_DIR,
        )
        print(f'Trajectory plot saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    sim = Simulation(config_file='examples/config.example.yaml')
    sim.run()
