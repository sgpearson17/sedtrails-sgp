"""
Test suite for the SedTRAILS logging system.

This module contains comprehensive tests for:
- Global exception logging and integration with the simulation class
- Logger configuration, initialization, and directory management
- Exception handling and stack trace logging
- Log file creation and content validation
- Simulation state logging and error reporting
- Test isolation and cleanup for reliable results

Covers both direct logger usage and simulation-driven logging.
"""

import pytest
import os
import tempfile
import shutil
import sys
import logging

from sedtrails.simulation import Simulation, setup_global_exception_logging
from sedtrails.logger.logger import logger_manager, log_exception, log_simulation_state
from sedtrails.exceptions.exceptions import ConfigurationError

class LoggerTestBase:
    """Base class for logger tests with proper isolation."""
    
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
        sys.excepthook = self.original_excepthook
        shutil.rmtree(self.temp_dir, ignore_errors=True)

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
        logger_manager.logger = None
        logger_manager.log_dir = self.test_results_dir

class TestLoggerBasicFunctionality(LoggerTestBase):
    """Test basic logger functionality."""

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

    def test_logger_directory_creation(self):
        """Test that logger creates directory if it doesn't exist."""
        non_existent_dir = os.path.join(self.temp_dir, 'new_logs')
        logger_manager.log_dir = non_existent_dir
        logger_manager.logger = None
        
        log_exception(RuntimeError("Test"), "Directory Creation Test")
        
        assert os.path.exists(non_existent_dir)
        log_file = os.path.join(non_existent_dir, 'log.txt')
        assert os.path.exists(log_file)

class TestGlobalExceptionHandling(LoggerTestBase):
    """Test global exception handling functionality."""

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

class TestLoggerConfiguration(LoggerTestBase):
    """Test logger configuration and setup."""

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
        
        logger_manager.logger = None
        logger_manager.log_dir = custom_output_dir
        
        log_exception(ValueError("Custom directory test"), "Directory Test")
        
        log_file = os.path.join(custom_output_dir, 'log.txt')
        assert os.path.exists(log_file), f"Log file not found at {log_file}"
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        assert "Custom directory test" in log_content
        assert "Directory Test" in log_content

if __name__ == '__main__':
    pytest.main([__file__, '-v'])