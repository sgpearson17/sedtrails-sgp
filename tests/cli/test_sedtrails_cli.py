"""
Unit tests for the SedTRAILS CLI commands using Typer's CliRunner.
"""

import pytest
import yaml
import os
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import Mock, patch
from sedtrails.application_interfaces.cli import app


class TestSedtrailsCLI:
    """
    Test suite for the Sedtrails CLI commands.
    """

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_config_data(self):
        """Sample configuration data for testing."""
        return {
            'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '1970-01-01'}},
            'folder_settings': {'input_data': 'input_data.nc', 'output_dir': 'output'},
            'time': {'timestep': '30S', 'duration': '1D'},
            'particles': {'count': 100, 'release_locations': [[0.0, 0.0]]},
        }

    @pytest.fixture
    def mock_run_simulation(self):
        """Mock run_simulation API function."""
        with patch('sedtrails.application_interfaces.api.run_simulation') as mock:
            mock.return_value = 'output'
            yield mock

    def test_run_simulation_default_success(self, runner, sample_config_data, mock_run_simulation):
        """Test successful run with default config file name."""
        with runner.isolated_filesystem():
            # Create default config file
            with open('sedtrails.yml', 'w') as f:
                yaml.dump(sample_config_data, f)

            result = runner.invoke(app, ['run'])

            assert result.exit_code == 0
            assert "Starting simulation from 'sedtrails.yml'..." in result.stdout
            assert "Simulation complete. Results saved to 'output'." in result.stdout

            # Verify run_simulation was called correctly
            mock_run_simulation.assert_called_once_with(
                config_file='sedtrails.yml',
                verbose=True
            )

    def test_run_simulation_custom_config(self, runner, sample_config_data, mock_run_simulation):
        """Test run with custom config file."""
        with runner.isolated_filesystem():
            custom_config = 'my_config.yml'

            # Create custom config file
            with open(custom_config, 'w') as f:
                yaml.dump(sample_config_data, f)

            result = runner.invoke(app, ['run', '--config', custom_config])

            assert result.exit_code == 0
            assert f"Starting simulation from '{custom_config}'..." in result.stdout
            assert "Simulation complete. Results saved to 'output'." in result.stdout

            # Verify run_simulation was called with custom config
            mock_run_simulation.assert_called_once_with(
                config_file=custom_config,
                verbose=True
            )

    def test_run_simulation_short_option(self, runner, sample_config_data, mock_run_simulation):
        """Test run with short option flag."""
        with runner.isolated_filesystem():
            custom_config = 'config.yml'

            # Create config file
            with open(custom_config, 'w') as f:
                yaml.dump(sample_config_data, f)

            result = runner.invoke(app, ['run', '-c', custom_config])

            assert result.exit_code == 0
            assert f"Starting simulation from '{custom_config}'..." in result.stdout
            assert "Simulation complete. Results saved to 'output'." in result.stdout

            # Verify run_simulation was called with short option
            mock_run_simulation.assert_called_once_with(
                config_file=custom_config,
                verbose=True
            )

    def test_run_simulation_error(self, runner, sample_config_data, mock_run_simulation):
        """Test run when simulation fails."""
        with runner.isolated_filesystem():
            # Create config file
            with open('sedtrails.yml', 'w') as f:
                yaml.dump(sample_config_data, f)

            # Mock run_simulation to raise an exception
            mock_run_simulation.side_effect = Exception('Simulation failed')

            result = runner.invoke(app, ['run'])

            assert result.exit_code == 1
            assert 'Error running simulation: Simulation failed' in result.stdout
            mock_run_simulation.assert_called_once()
