"""
Test suite for the SedTRAILS logging system (new structure).

Covers:
- Global exception logging via installed excepthook
- Logger configuration and idempotent initialization
- Log file creation and content validation
- Simulation state logging helper
- Test isolation and cleanup
"""

import pytest
import os
import tempfile
import shutil
import sys
import logging

from sedtrails.logger.logger import setup_logging, log_simulation_state

LOG_FILENAME = "log.txt"


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


class TestLoggerBasicFunctionality(LoggerTestBase):
    """Test basic logger functionality with the new setup."""

    def test_log_exception_basic(self):
        """Test that a logged exception creates a log file with correct content."""
        setup_logging(self.test_results_dir)
        logger = logging.getLogger(f"sedtrails.{__name__}")

        try:
            raise ValueError("Test error message")
        except ValueError:
            logger.exception("Test Context")

        log_file = os.path.join(self.test_results_dir, LOG_FILENAME)
        assert os.path.exists(log_file), f"Log file not created at {log_file}"
        
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        assert "Test Context" in log_content
        assert "ValueError" in log_content
        assert "Test error message" in log_content

    def test_log_simulation_state(self):
        """Test that log_simulation_state works correctly."""
        setup_logging(self.test_results_dir)
        logger = logging.getLogger(f"sedtrails.{__name__}")

        state_data = {
            "status": "simulation_started",
            "config_file": "test.yaml",
            "python_version": "3.9.7"
        }
        log_simulation_state(logger, state_data)
        
        log_file = os.path.join(self.test_results_dir, LOG_FILENAME)
        assert os.path.exists(log_file)
        
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        assert "=== SIMULATION START ===" in log_content
        assert "Config: test.yaml" in log_content
        assert "Python: 3.9.7" in log_content

    def test_multiple_exceptions_logged(self):
        """Test that multiple exceptions are properly logged."""
        setup_logging(self.test_results_dir)
        logger = logging.getLogger(f"sedtrails.{__name__}")

        try:
            raise ValueError("First error")
        except ValueError:
            logger.exception("Context 1")

        try:
            raise TypeError("Second error")
        except TypeError:
            logger.exception("Context 2")
        
        log_file = os.path.join(self.test_results_dir, LOG_FILENAME)
        assert os.path.exists(log_file)
        
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        assert "First error" in log_content
        assert "Second error" in log_content
        assert "Context 1" in log_content
        assert "Context 2" in log_content

    def test_exception_with_traceback(self):
        """Test that exceptions with tracebacks are properly logged."""
        setup_logging(self.test_results_dir)
        logger = logging.getLogger(f"sedtrails.{__name__}")

        def inner_function():
            raise ValueError("Inner error")
        
        def outer_function():
            inner_function()

        try:
            outer_function()
        except ValueError:
            logger.exception("Traceback Test")
        
        log_file = os.path.join(self.test_results_dir, LOG_FILENAME)
        assert os.path.exists(log_file)
        
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        assert "Traceback Test" in log_content
        assert "inner_function" in log_content
        assert "outer_function" in log_content


class TestGlobalExceptionHandling(LoggerTestBase):
    """Test global exception handling functionality."""

    def test_global_exception_hook_setup(self):
        """Test that global exception hook is properly installed by setup_logging."""
        setup_logging(self.test_results_dir)
        assert sys.excepthook != self.original_excepthook
        assert callable(sys.excepthook)

    def test_global_exception_logging(self):
        setup_logging(self.test_results_dir)
        # Simulate an unhandled exception
        sys.excepthook(RuntimeError, RuntimeError("Global test error"), None)

        log_file = os.path.join(self.test_results_dir, LOG_FILENAME)
        assert os.path.exists(log_file)
        with open(log_file, encoding='utf-8') as f:
            content = f.read()

        assert "Unhandled exception" in content
        assert "RuntimeError" in content
        assert "Global test error" in content

    def test_simulation_failure_state_logged(self):
        """Test that a simulated unhandled exception is logged as failure."""
        setup_logging(self.test_results_dir)
        test_exception = RuntimeError("Simulation failed")
        sys.excepthook(RuntimeError, test_exception, None)
        
        log_file = os.path.join(self.test_results_dir, LOG_FILENAME)
        assert os.path.exists(log_file)
        
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        assert "Unhandled exception" in log_content
        assert "RuntimeError" in log_content
        assert "Simulation failed" in log_content


class TestLoggerConfiguration(LoggerTestBase):
    """Test logger configuration and setup."""

    def test_logger_initialization_message(self):
        """Test that logger initialization writes its startup messages."""
        setup_logging(self.test_results_dir)
        
        log_file = os.path.join(self.test_results_dir, LOG_FILENAME)
        assert os.path.exists(log_file)
        
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        assert "=== LOGGING INITIALIZED ===" in log_content
        assert "Log file:" in log_content  # setup_logging logs the file path

    def test_logger_directory_created(self):
        """Test that setup_logging creates the directory if it does not exist."""
        non_existent_dir = os.path.join(self.temp_dir, 'does_not_exist')
        # Ensure the directory does not exist
        if os.path.exists(non_existent_dir):
            shutil.rmtree(non_existent_dir)

        # Should not raise; it should create the directory and log file
        setup_logging(non_existent_dir)
        log_file = os.path.join(non_existent_dir, LOG_FILENAME)
        assert os.path.exists(non_existent_dir)
        assert os.path.exists(log_file)