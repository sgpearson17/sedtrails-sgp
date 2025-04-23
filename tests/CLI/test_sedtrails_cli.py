"""
Unit tests for the SedTRAILS CLI commands using Typer's CliRunner.
"""

from typer.testing import CliRunner
from sedtrails.sedtrails import app

runner = CliRunner()


class TestSedtrailsCLI:
    """
    Test suite for the Sedtrails CLI commands.
    """

    def test_load_config_default(self):
        """
        Use default config file name ("sedtrails.yml")
        """
        result = runner.invoke(app, ['load-config'])
        assert result.exit_code == 0
        assert "Loading and validating configuration from 'sedtrails.yml'" in result.stdout
        assert 'Configuration validated successfully' in result.stdout
        # Check dummy configuration is printed
        assert 'setting1' in result.stdout

    def test_load_config_custom(self):
        """
        Pass a custom configuration file name
        """
        custom_config = 'dummy_config.yml'
        result = runner.invoke(app, ['load-config', '--config', custom_config])
        assert result.exit_code == 0
        assert f"Loading and validating configuration from '{custom_config}'" in result.stdout
        assert 'Configuration validated successfully' in result.stdout

    def test_run_simulation_default(self):
        """
        Test run-simulation with default config and output file names.
        """
        result = runner.invoke(app, ['run-simulation'])
        assert result.exit_code == 0
        # Ensure it validates configuration and then simulates.
        assert "Validating configuration from 'sedtrails.yml'" in result.stdout
        assert 'Configuration validated successfully' in result.stdout
        assert 'Running simulation...' in result.stdout
        assert "Simulation complete. Output saved to 'sedtrails.nc'" in result.stdout

    def test_run_simulation_custom(self):
        """
        Test run-simulation with custom config and output file.
        """
        custom_config = 'custom_config.yml'
        custom_output = 'custom_output.nc'
        result = runner.invoke(
            app,
            ['run-simulation', '--config', custom_config, '--output', custom_output],
        )
        assert result.exit_code == 0
        assert f"Validating configuration from '{custom_config}'" in result.stdout
        assert 'Configuration validated successfully' in result.stdout
        assert 'Running simulation...' in result.stdout
        assert f"Simulation complete. Output saved to '{custom_output}'" in result.stdout

    def test_analyze_default(self):
        """
        Test the analyze subcommand with default file names.
        """
        result = runner.invoke(app, ['analyze'])
        assert result.exit_code == 0
        assert "Performing statistical analysis on 'sedtrails.nc'" in result.stdout
        assert "Analysis complete. Results saved to 'analysis.nc'" in result.stdout

    def test_analyze_custom(self):
        """
        Test the analyze subcommand with custom input and output file names.
        """
        custom_input = 'custom_simulation.nc'
        custom_output = 'custom_analysis.nc'
        result = runner.invoke(app, ['analyze', '--input', custom_input, '--output', custom_output])
        assert result.exit_code == 0
        assert f"Performing statistical analysis on '{custom_input}'" in result.stdout
        assert f"Analysis complete. Results saved to '{custom_output}'" in result.stdout

    def test_network_analysis_default(self):
        """
        Test the network-analysis subcommand with default file names.
        """
        result = runner.invoke(app, ['network-analysis'])
        assert result.exit_code == 0
        assert "Performing network analysis on 'sedtrails.nc'" in result.stdout
        assert "Network analysis complete. Results saved to 'analysis.nc'" in result.stdout

    def test_network_analysis_custom(self):
        """
        Test the network-analysis subcommand with custom file names.
        """
        custom_input = 'custom_simulation.nc'
        custom_output = 'custom_network.nc'
        result = runner.invoke(
            app,
            [
                'network-analysis',
                '--input',
                custom_input,
                '--output',
                custom_output,
            ],
        )
        assert result.exit_code == 0
        assert f"Performing network analysis on '{custom_input}'" in result.stdout
        assert f"Network analysis complete. Results saved to '{custom_output}'" in result.stdout
