import time
import numpy as np

from sedtrails.transport_converter.format_converter import FormatConverter, SedtrailsData
from sedtrails.transport_converter.physics_converter import PhysicsConverter
from sedtrails.particle_tracer.data_retriever import FlowFieldDataRetriever
from sedtrails.particle_tracer.particle import Sand
from sedtrails.particle_tracer.position_calculator import ParticlePositionCalculator
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
        self._controller = ConfigurationController()
        self._start_time = None
        self._config_is_read = False

        # Lazy initialization of converters
        self.format_converter = FormatConverter(self._get_format_config())
        self.physics_converter = PhysicsConverter(self._get_physics_config())
        self.data_manager = ''

    def _get_format_config(self):
        """
        Returns configuration parameters required for the format converter.
        """

        format_config = {
            'input_file': self._controller.get('folder_settings.input_data'),
            'input_format': self._controller.get('input_model'),  # Specify the input format
            'reference_date': self._controller.get(
                'reference_date', '1970-01-01'
            ),  # TODO: shall this be added to the json schema??
        }

        return format_config

    def _get_physics_config(self):
        """
        Returns configuration parameters required for the physics converter.
        """
        from sedtrails.transport_converter.physics_converter import PhysicsConfig

        config = PhysicsConfig(
            tracer_method=self._controller.get('physics.tracer_method'),
            gravity=self._controller.get('physics.constants.g'),
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
            self._controller.read_config(self._config_file)
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
            self._controller.read_config(self._config_file)

        value = self._controller.get(key, None)
        if value is None:
            warnings.warn(f'Key "{key}" not found in configuration file', UserWarning, stacklevel=2)
        return value

    def run(self):
        """
        Run the particle simulation using both original and Numba-optimized implementations.
        """

        sedtrails_data = self.format_converter.convert_to_sedtrails()
        # Add physics calculations to the SedtrailsData
        self.physics_converter.convert_physics(sedtrails_data)

        print('Data conversion completed')

        # TODO: continue here: keep integrating the workflow in this method
        # Initialize flow field data retriever
        retriever = FlowFieldDataRetriever(sedtrails_data)
        retriever.flow_field_name = 'suspended_velocity'  # TODO: shouldn't this be read from config?

        initial_time = sedtrails_data.times[2]  #  returns seconds since reference date as floats
        initial_flow = retriever.get_flow_field(initial_time)  # expects secondes since reference date

        # # retrieve start time from configuration or use default
        # START_TIME = self._controller.get('time.start_time')

        TIMESTEP = self._controller.get('time.timestep')
        # TODO: should be managed by data manager

        OUTPUT_DIR = self._controller.get('folder_settings.output_dir')

        # Start at the 3rd timestep (index 2)
        TIMESTEP_INDEX = 2

        # Get the initial flow field at specified timestep
        # TODO: this should be handled by time class
        initial_time = sedtrails_data.times[TIMESTEP_INDEX]
        # Duration: 6.333 hours = 6 hours and 20 minutes = 380 minutes = 22,800 seconds
        DURATION_HOURS = 6.333
        DURATION_SECONDS = DURATION_HOURS * 3600

        # Calculate number of steps
        NUM_STEPS = int(DURATION_SECONDS / TIMESTEP)

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
        print(f'Will run for {DURATION_HOURS:.3f} hours ({NUM_STEPS} steps) with timestep {TIMESTEP} seconds')

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
            np.array([particle_numba.x]), np.array([particle_numba.y]), flow_data['u'], flow_data['v'], TIMESTEP
        )
        warmup_time = time.time() - warmup_start
        print(f'JIT warm-up completed in {warmup_time:.4f} seconds')

        # Start timer after compilation
        simulation_start = time.time()
        current_time = initial_time

        for step in range(1, NUM_STEPS + 1):
            # Update current time
            current_time = initial_time + step * TIMESTEP

            # Get flow field at current time
            flow_data = retriever.get_flow_field(current_time)

            # Update particle position using Numba calculator
            new_x, new_y = numba_calc['update_particles'](
                np.array([particle_numba.x]), np.array([particle_numba.y]), flow_data['u'], flow_data['v'], TIMESTEP
            )

            # Update particle with new position
            particle_numba.x = new_x[0]
            particle_numba.y = new_y[0]

            # Store trajectory
            trajectory_numba_x.append(particle_numba.x)
            trajectory_numba_y.append(particle_numba.y)

            # Print progress every 20%
            if step % max(1, NUM_STEPS // 5) == 0:
                percent_complete = (step / NUM_STEPS) * 100
                elapsed_hours = step * TIMESTEP / 3600
                print(
                    f'Step {step}/{NUM_STEPS} ({percent_complete:.1f}%) - '
                    f'Time: {elapsed_hours:.2f} hours - '
                    f'Position: ({particle_numba.x:.2f}, {particle_numba.y:.2f})'
                )

        numba_time = time.time() - simulation_start
        print(f'Numba implementation completed in {numba_time:.4f} seconds')
        print(f'Average speed: {NUM_STEPS / numba_time:.1f} steps/second')
        print(f'Final position: ({particle_numba.x:.4f}, {particle_numba.y:.4f})')
