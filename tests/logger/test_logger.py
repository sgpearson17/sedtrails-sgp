"""
Test suite for the SedTRAILS logging system.

This module contains comprehensive tests for:
- Global exception logging functionality
- Logger configuration and initialization
- Exception handling across different modules
- Log file creation and content validation
- Integration with simulation components
"""

import pytest
import os
import tempfile
import shutil
import yaml
import sys
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path

from sedtrails.simulation import Simulation, setup_global_exception_logging
from sedtrails.logger.logger import _logger_manager, log_exception, log_simulation_state
from sedtrails.exceptions.exceptions import (
    ConfigurationError,
    SedtrailsException,
    DataConversionError,
    ParticleInitializationError,
    VisualizationError
)


# Test Configuration Data
def get_valid_config(output_dir):
    """Get a valid configuration for testing."""
    return {
        'general': {
            'input_model': {
                'format': 'fm_netcdf',
                'reference_date': '1970-01-01'
            }
        },
        'folder_settings': {
            'input_data': './sample-data/inlet_sedtrails.nc',
            'output_dir': output_dir
        },
        'time': {
            'start': '2016-09-21 19:30:00',
            'timestep': '30S',
            'duration': '6H18M18S'
        },
        'particle_seeding': {
            'strategy': {
                'point': {
                    'points': ['40000,17000']
                }
            }
        },
        'physics': {
            'tracer_method': 'van_westen',
            'constants': {
                'g': 9.81,
                'von_karman': 0.40,
                'kinematic_viscosity': 1.36e-6,
                'rho_w': 1027.0,
                'rho_s': 2650.0
            }
        }
    }


def get_invalid_format_config(output_dir):
    """Get configuration with invalid format for testing."""
    return {
        'general': {
            'input_model': {
                'format': 'nonexistent_format',
                'reference_date': '1970-01-01'
            }
        },
        'folder_settings': {
            'input_data': './sample-data/inlet_sedtrails.nc',
            'output_dir': output_dir
        }
    }


def get_missing_file_config(output_dir):
    """Get configuration with missing input file for testing."""
    return {
        'general': {
            'input_model': {
                'format': 'fm_netcdf',
                'reference_date': '1970-01-01'
            }
        },
        'folder_settings': {
            'input_data': './nonexistent_file.nc',
            'output_dir': output_dir
        }
    }


def get_invalid_time_config(output_dir):
    """Get configuration with invalid time format for testing."""
    config = get_valid_config(output_dir)
    config['time']['start'] = '9999-99-99 99:99:99'
    return config


def get_invalid_physics_config(output_dir):
    """Get configuration with invalid physics parameters for testing."""
    config = get_valid_config(output_dir)
    config['physics']['tracer_method'] = 'nonexistent_method'
    config['physics']['constants']['g'] = 'not_a_number'
    return config


class LoggerTestBase:
    """Base class for logger tests with proper isolation."""
    
    def reset_logging_completely(self):
        """Completely reset all logging state for clean test isolation."""
        # Clear all loggers and their handlers
        for name in list(logging.Logger.manager.loggerDict.keys()):
            logger = logging.getLogger(name)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                if hasattr(handler, 'close'):
                    handler.close()
        
        # Clear root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            if hasattr(handler, 'close'):
                handler.close()
        
        # Reset logger manager
        _logger_manager.logger = None
        _logger_manager.log_dir = self.test_results_dir


class TestLoggerBasicFunctionality(LoggerTestBase):
    """Test basic logger functionality."""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test with complete isolation."""
        # Store original state
        self.original_excepthook = sys.excepthook
        
        # Create fresh temp directory
        self.temp_dir = tempfile.mkdtemp()
        self.test_results_dir = os.path.join(self.temp_dir, 'results')
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        # Complete logging reset
        self.reset_logging_completely()
        
        yield
        
        # Complete cleanup
        self.reset_logging_completely()
        sys.excepthook = self.original_excepthook
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_exception_basic(self):
        """Test that log_exception creates a log file with correct content."""
        test_exception = ValueError("Test error message")
        
        log_exception(test_exception, "Test Context")
        
        log_file = os.path.join(self.test_results_dir, 'log.txt')
        assert os.path.exists(log_file), f"Log file not created at {log_file}"
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert "=== ERROR: Test Context ===" in log_content
        assert "Exception type: ValueError" in log_content
        assert "Exception message: Test error message" in log_content
        assert "Stack trace:" in log_content

    def test_log_simulation_state(self):
        """Test that log_simulation_state works correctly."""
        state_data = {
            "status": "simulation_started",
            "config_file": "test.yaml",
            "python_version": "3.9.7"
        }
        
        log_simulation_state(state_data)
        
        log_file = os.path.join(self.test_results_dir, 'log.txt')
        assert os.path.exists(log_file)
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert "=== SIMULATION START ===" in log_content
        assert "Config: test.yaml" in log_content
        assert "Python: 3.9.7" in log_content

    def test_multiple_exceptions_logged(self):
        """Test that multiple exceptions are properly logged."""
        log_exception(ValueError("First error"), "Context 1")
        log_exception(TypeError("Second error"), "Context 2")
        
        log_file = os.path.join(self.test_results_dir, 'log.txt')
        assert os.path.exists(log_file)
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert "First error" in log_content
        assert "Second error" in log_content
        assert "Context 1" in log_content
        assert "Context 2" in log_content

    def test_logger_directory_creation(self):
        """Test that logger creates directory if it doesn't exist."""
        non_existent_dir = os.path.join(self.temp_dir, 'new_logs')
        _logger_manager.log_dir = non_existent_dir
        _logger_manager.logger = None
        
        log_exception(RuntimeError("Test"), "Directory Creation Test")
        
        assert os.path.exists(non_existent_dir)
        log_file = os.path.join(non_existent_dir, 'log.txt')
        assert os.path.exists(log_file)


class TestGlobalExceptionHandling(LoggerTestBase):
    """Test global exception handling functionality."""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test with complete isolation."""
        self.original_excepthook = sys.excepthook
        
        self.temp_dir = tempfile.mkdtemp()
        self.test_results_dir = os.path.join(self.temp_dir, 'results')
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        self.reset_logging_completely()
        
        yield
        
        self.reset_logging_completely()
        sys.excepthook = self.original_excepthook
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_global_exception_hook_setup(self):
        """Test that global exception hook is properly installed."""
        setup_global_exception_logging()
        assert sys.excepthook != self.original_excepthook
        assert callable(sys.excepthook)

    def test_global_exception_handler_logs_exceptions(self):
        """Test that global exception handler logs exceptions when directory is set."""
        setup_global_exception_logging()
        
        test_exception = RuntimeError("Global handler test error")
        sys.excepthook(RuntimeError, test_exception, None)
        
        log_file = os.path.join(self.test_results_dir, 'log.txt')
        assert os.path.exists(log_file), "Global exception handler didn't create log file"
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert "=== ERROR: Global Exception Handler ===" in log_content
        assert "Exception type: RuntimeError" in log_content
        assert "Global handler test error" in log_content

    def test_keyboard_interrupt_not_logged(self):
        """Test that KeyboardInterrupt is not logged."""
        setup_global_exception_logging()
        
        # Test that KeyboardInterrupt doesn't create a log file
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt("User cancelled"), None)
        
        log_file = os.path.join(self.test_results_dir, 'log.txt')
        assert not os.path.exists(log_file), "KeyboardInterrupt should not create log file"
        
        # Verify that other exceptions DO get logged (as a control test)
        sys.excepthook(RuntimeError, RuntimeError("Test error"), None)
        assert os.path.exists(log_file), "Other exceptions should create log file"
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Verify KeyboardInterrupt is not in the log, but RuntimeError is
        assert "KeyboardInterrupt" not in log_content
        assert "RuntimeError" in log_content

    def test_simulation_failure_state_logged(self):
        """Test that simulation failure state is logged with global exception."""
        setup_global_exception_logging()
        
        test_exception = RuntimeError("Simulation failed")
        sys.excepthook(RuntimeError, test_exception, None)
        
        log_file = os.path.join(self.test_results_dir, 'log.txt')
        assert os.path.exists(log_file)
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert "=== SIMULATION FAILED ===" in log_content
        assert "Error type: RuntimeError" in log_content
        assert "Error message: Simulation failed" in log_content


class TestExceptionTypes(LoggerTestBase):
    """Test logging of different exception types."""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test with complete isolation."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_results_dir = os.path.join(self.temp_dir, 'results')
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        # Critical: Reset logging state for EACH test
        self.reset_logging_completely()
        
        yield
        
        self.reset_logging_completely()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.parametrize("exception_type,error_message", [
        (ImportError, "Failed to import required module"),
        (FileNotFoundError, "Required file not found"), 
        (ValueError, "Invalid configuration value"),
        (TypeError, "Invalid parameter type"),
        (ConfigurationError, "Configuration validation failed"),
        (PermissionError, "Permission denied"),
    ])
    def test_exception_logging(self, exception_type, error_message):
        """Test that various exception types are properly logged."""
        # Force complete reset for each parameterized test
        self.reset_logging_completely()
        
        test_exception = exception_type(error_message)
        log_exception(test_exception, f"Test - {exception_type.__name__}")
        
        log_file = os.path.join(self.test_results_dir, 'log.txt')
        assert os.path.exists(log_file), f"Log file not created for {exception_type.__name__}"
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert f"Exception type: {exception_type.__name__}" in log_content
        assert f"Exception message: {error_message}" in log_content
        assert "Stack trace:" in log_content
        assert f"Test - {exception_type.__name__}" in log_content

    def test_exception_with_traceback(self):
        """Test that exceptions with tracebacks are properly logged."""
        def inner_function():
            raise ValueError("Inner error")
        
        def outer_function():
            inner_function()
        
        try:
            outer_function()
        except ValueError as e:
            log_exception(e, "Traceback Test")
        
        log_file = os.path.join(self.test_results_dir, 'log.txt')
        assert os.path.exists(log_file)
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert "Stack trace:" in log_content
        assert "inner_function" in log_content
        assert "outer_function" in log_content


class TestSimulationIntegration(LoggerTestBase):
    """Test logging integration with Simulation class."""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test with complete isolation."""
        self.original_excepthook = sys.excepthook
        
        self.temp_dir = tempfile.mkdtemp()
        self.test_results_dir = os.path.join(self.temp_dir, 'results')
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        self.reset_logging_completely()
        
        yield
        
        self.reset_logging_completely()
        sys.excepthook = self.original_excepthook
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_configuration_error_logging(self):
        """Test that configuration errors can be logged."""
        nonexistent_config = os.path.join(self.temp_dir, 'nonexistent.yaml')
        
        try:
            Simulation(config_file=nonexistent_config)
            pytest.fail("Expected ConfigurationError to be raised")
        except ConfigurationError as e:
            log_exception(e, "Configuration Test")
        
        log_file = os.path.join(self.test_results_dir, 'log.txt')
        assert os.path.exists(log_file), f"Log file not found at {log_file}"
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert "Configuration file not found" in log_content
        assert "Exception type: ConfigurationError" in log_content

    def test_yaml_parsing_error_logging(self):
        """Test that YAML parsing errors can be logged."""
        malformed_yaml = """
        general:
          input_model:
            format: "fm_netcdf"
            reference_date: "1970-01-01
        folder_settings:
          output_dir: "{}"
        """.format(self.test_results_dir)
        
        config_path = os.path.join(self.temp_dir, 'malformed.yaml')
        with open(config_path, 'w') as f:
            f.write(malformed_yaml)
        
        try:
            Simulation(config_file=config_path)
            # If no exception, log a test one
            log_exception(RuntimeError("No YAML error occurred"), "YAML Test")
        except Exception as e:
            log_exception(e, "YAML Parsing Error")
        
        log_file = os.path.join(self.test_results_dir, 'log.txt')
        assert os.path.exists(log_file), "Log file not created"
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert "Exception type:" in log_content
        assert "Exception message:" in log_content

    def test_permission_error_simulation(self):
        """Test logging of permission-related errors."""
        # Test permission error logging directly
        permission_error = PermissionError("Permission denied: /root/restricted_access")
        log_exception(permission_error, "Permission Error Test")
        
        log_file = os.path.join(self.test_results_dir, 'log.txt')
        assert os.path.exists(log_file)
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert "Exception type: PermissionError" in log_content
        assert "Permission denied" in log_content
        assert "Permission Error Test" in log_content


class TestLoggerConfiguration(LoggerTestBase):
    """Test logger configuration and setup."""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test with complete isolation."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_results_dir = os.path.join(self.temp_dir, 'results')
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        self.reset_logging_completely()
        
        yield
        
        self.reset_logging_completely()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_logger_initialization_message(self):
        """Test that logger initialization is properly logged."""
        log_exception(RuntimeError("Test"), "Init Test")
        
        log_file = os.path.join(self.test_results_dir, 'log.txt')
        assert os.path.exists(log_file)
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert "=== LOGGING INITIALIZED ===" in log_content
        assert "Log file: log.txt" in log_content
        assert "Location:" in log_content

    def test_custom_log_directory(self):
        """Test that custom log directory works correctly."""
        custom_output_dir = os.path.join(self.temp_dir, 'custom_output')
        os.makedirs(custom_output_dir, exist_ok=True)
        
        _logger_manager.logger = None
        _logger_manager.log_dir = custom_output_dir
        
        log_exception(ValueError("Custom directory test"), "Directory Test")
        
        log_file = os.path.join(custom_output_dir, 'log.txt')
        assert os.path.exists(log_file), f"Log file not found at {log_file}"
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert "Custom directory test" in log_content
        assert "Directory Test" in log_content

    def test_log_file_permissions(self):
        """Test that log file is created with proper permissions."""
        log_exception(RuntimeError("Test"), "Permission Test")
        
        log_file = os.path.join(self.test_results_dir, 'log.txt')
        assert os.path.exists(log_file)
        assert os.access(log_file, os.R_OK)  # Readable
        assert os.access(log_file, os.W_OK)  # Writable


if __name__ == '__main__':
    pytest.main([__file__, '-v'])