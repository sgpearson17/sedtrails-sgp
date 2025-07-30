import time
import os
import sys
import numpy as np

from sedtrails.transport_converter.format_converter import FormatConverter, SedtrailsData
from sedtrails.transport_converter.physics_converter import PhysicsConverter
from sedtrails.particle_tracer.data_retriever import FlowFieldDataRetriever
from sedtrails.particle_tracer.particle import Sand
from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator
from sedtrails.configuration_interface.configuration_controller import ConfigurationController
from sedtrails.data_manager import DataManager
from sedtrails.particle_tracer.timer import Time, Duration, Timer
from sedtrails.logger.logger import log_simulation_state, log_exception, LoggerManager
from sedtrails.exceptions.exceptions import ConfigurationError
from typing import Any

def setup_global_exception_logging(logger):
    """Setup global exception logging for unhandled exceptions."""
    original_excepthook = sys.excepthook
    
    def exception_handler(exc_type, exc_value, exc_traceback):
        # Don't log KeyboardInterrupt (Ctrl+C)
        if issubclass(exc_type, KeyboardInterrupt):
            original_excepthook(exc_type, exc_value, exc_traceback)
            return
        
        # Log all other exceptions
        log_exception(exc_value, logger, "Global Exception Handler")
        log_simulation_state({
            "status": "simulation_failed",
            "error_type": exc_type.__name__,
            "error_message": str(exc_value)
        }, logger)
        
        # Call original handler
        original_excepthook(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = exception_handler

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

        # Validate config file exists early
        if not os.path.exists(config_file):
            raise ConfigurationError(f"Configuration file not found: {config_file}")

        # Try to read config and update logger directory
        try:
            self._controller = ConfigurationController(self._config_file)
            self._controller.load_config(self._config_file)
            self._config_is_read = True            
            output_dir = self._controller.get('folder_settings.output_dir', 'results')
            
        except Exception:
            # Global exception handler will catch and log this
            raise

        # Initialize other components
        self.format_converter = FormatConverter(self._get_format_config())
        self.physics_converter = PhysicsConverter(self._get_physics_config())
        self.data_manager = DataManager(self._get_output_dir())
        self.data_manager.set_mesh()

        logger_manager = LoggerManager(output_dir)
        self.logger = logger_manager.setup_logger()

        # Setup global exception handling
        setup_global_exception_logging(self.logger)

    def _get_format_config(self):
        """
        Returns configuration parameters required for the format converter.
        """

        format_config = {
            'input_file': self._controller.get('folder_settings.input_data'),
            'input_format': self._controller.get('general.input_model.format'),  # Specify the input format
            'reference_date': self._controller.get('general.input_model.reference_date'),
        }

        return format_config

    def _get_output_dir(self):
        """
        Returns the output directory for the simulation.
        """
        return self._controller.get('folder_settings.output_dir')

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

    def validate_config(self) -> bool:
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

    def get_parameter(self, key: str) -> Any:
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

        from tqdm import tqdm

        if not self._config_is_read:  # assure config is read only once
            self._controller.load_config(self._config_file)
            self._config_is_read = True

            log_simulation_state({
                "state": "config_loading",
                "config_file_path": self._config_file,
                "timestamp": time.time()
            }, self.logger)

        # Log the command that started the simulation
        log_simulation_state({
            "status": "simulation_started",
            "command": " ".join(sys.argv),
            "config_file": self._config_file,
            "working_directory": os.getcwd(),
            "python_version": sys.version.split()[0]
        }, self.logger)            

        sedtrails_data = self.format_converter.convert_to_sedtrails()
        # Add physics calculations to the SedtrailsData
        self.physics_converter.convert_physics(sedtrails_data)
        # Initialize flow field data retriever
        retriever = FlowFieldDataRetriever(sedtrails_data)
        # TODO: shouldn't this be read from config? https://github.com/sedtrails/sedtrails/issues/222
        retriever.flow_field_name = 'suspended_velocity'

        start_time = self._controller.get('time.start', None)
        if start_time is None:
            start_time = '1970-01-01 00:00:00'  # TODO: This should be the start of flow field time series

        simulation_time = Time(
            start_time,
            duration=Duration(self._controller.get('time.duration')),
            time_step=Duration(self._controller.get('time.timestep')),
        )

        log_simulation_state({
            "state": "data_conversion_completed",
            "num_timesteps": len(sedtrails_data.times),
            "flow_field_name": "suspended_velocity",
            "start_time": start_time,
            "simulation_duration_seconds": simulation_time.duration.seconds,
            "simulation_timestep_seconds": simulation_time.time_step.seconds
        }, self.logger)

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

        # TODO: PARTICLE SEEDING
        particles = []
        for id, (x, y) in particle_positions.items():
            START_X = float(x)
            START_Y = float(y)
            particle = Sand(id=id, _x=START_X, _y=START_Y, name='Test Particle')
            particles.append(particle)

        log_simulation_state({
            "state": "particles_initialized",
            "num_particles": len(particles),
            "particle_positions": {p.id: {"x": round(p.x, 2), "y": round(p.y, 2)} for p in particles},
            "seeding_strategy": strategy
        }, self.logger)

        # Store trajectory1
        trajectory_numba_x = [particles[0].x]  # TODO: must handle multiple particles
        trajectory_numba_y = [particles[0].y]

        # First get the initial flow data to create calculator
        flow_data = retriever.get_flow_field(simulation_time.start)

        # Create and compile the numba calculator - this will include compilation time
        log_simulation_state({
            "state": "numba_compilation_started",
            "grid_size_x": len(flow_data['x']),
            "grid_size_y": len(flow_data['y'])
        }, self.logger)
        compile_start = time.time()
        numba_calc = create_numba_particle_calculator(grid_x=flow_data['x'], grid_y=flow_data['y'])
        compile_time = time.time() - compile_start
        log_simulation_state({
            "status": "compilation_complete", 
            "time_sec": round(compile_time, 2)
        }, self.logger)

        TIME_STEP_SECONDS = simulation_time.time_step.seconds
        # Warm up with one calculation to trigger JIT compilation
        log_simulation_state({
            "status": "warming_up_jit"
        }, self.logger)
        warmup_start = time.time()
        _ = numba_calc['update_particles'](
            np.array([particles[0].x]),
            np.array([particles[0].y]),
            flow_data['u'],
            flow_data['v'],
            TIME_STEP_SECONDS,
        )
        warmup_time = time.time() - warmup_start
        log_simulation_state({
            "status": "warmup_complete",
            "time_sec": round(warmup_time, 2)
        }, self.logger)
        # Start timer after compilation
        timer = Timer(simulation_time=simulation_time)
        for _step in tqdm(range(1, timer.steps + 1), desc='Computing positions', unit='Steps'):
            # Get flow field at current time
            flow_data = retriever.get_flow_field(timer.current)
            # Update particle position using Numba calculator
            new_x, new_y = numba_calc['update_particles'](
                np.array([particles[0].x]),
                np.array([particles[0].y]),
                flow_data['u'],
                flow_data['v'],
                TIME_STEP_SECONDS,
            )

            # Update particle with new position
            particles[0].x = new_x[0]
            particles[0].y = new_y[0]

            # Store trajectory
            trajectory_numba_x.append(particles[0].x)
            trajectory_numba_y.append(particles[0].y)

            ## TEST data manager
            self.data_manager.add_data(
                particle_id=particles[0].id,
                time=timer.current,
                x=particles[0].x,
                y=particles[0].y,
            )

            # Advance timer
            timer.advance()

        simulation_end_time = time.time()
        total_time = simulation_end_time - compile_start  # Total time including compilation
        log_simulation_state({
            "status": "simulation_complete",
            "total_steps": timer.steps,
            "total_time_sec": round(total_time, 2),
            "final_position": f"({particles[0].x:.2f}, {particles[0].y:.2f})"
        }, self.logger)

        # Finalize results
        self.data_manager.dump()  # Write remaining data to disk

        # Convert trajectory to numpy arrays
        trajectory_numba_x = np.array(trajectory_numba_x)
        trajectory_numba_y = np.array(trajectory_numba_y)

        log_simulation_state({
            "status": "creating_visualization",
            "trajectory_points": len(trajectory_numba_x)
        }, self.logger)

        # Plot flow field with particle trajectory using the function
        final_flow = retriever.get_flow_field(timer.current)

        from sedtrails.pathway_visualizer.visualization_utils import plot_particle_trajectory

        plot_particle_trajectory(
            flow_data=final_flow,
            trajectory_x=trajectory_numba_x,
            trajectory_y=trajectory_numba_y,
            title=f'Particle Trajectory - {simulation_time.duration.seconds} seconds, {timer.steps} steps',
            save_path=self.data_manager.output_dir + '/trajectory_plot.png',
        )
        log_simulation_state({
            "status": "visualization_complete",
            "output_plot_path": self.data_manager.output_dir + '/trajectory_plot.png'
        }, self.logger)

if __name__ == '__main__':
    sim = Simulation(config_file='examples/config.example.yaml')
    sim.run()
