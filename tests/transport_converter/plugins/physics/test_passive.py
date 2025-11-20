"""
Unit tests for passive tracer physics plugin.

This module comprehensively tests the passive tracer physics plugin implemented in 
sedtrails.transport_converter.plugins.physics.passive. The tests validate:

1. **Plugin Interface Compliance**: Ensures proper inheritance and method implementation
2. **Data Handling**: Validates input processing and output generation
3. **Physics Implementation**: Tests the passive tracer physics logic
4. **Integration**: Tests interaction with SedtrailsData structures
5. **Edge Cases**: Boundary conditions and error handling

Testing Strategy:
- **Mock Data**: Use realistic synthetic data for controlled testing
- **Interface Validation**: Verify plugin conforms to BasePhysicsPlugin interface
- **Physics Validation**: Ensure passive tracers correctly follow flow fields
- **Data Integrity**: Verify data structures and field additions
- **Error Handling**: Test robustness with invalid inputs

Physics Background:
Passive tracers are particles that move with the flow without affecting it or being
affected by sediment transport processes. They represent neutrally buoyant particles
that perfectly follow the fluid motion, making them ideal for studying flow patterns
and mixing processes.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from numpy.testing import assert_allclose, assert_array_equal

from sedtrails.transport_converter.plugins.physics.passive import PhysicsPlugin
from sedtrails.transport_converter.plugins.physics.plugin import BasePhysicsPlugin
from sedtrails.transport_converter import SedtrailsData


class TestPassiveTracerPluginInterface:
    """Test the passive tracer plugin interface compliance.
    
    This class validates that the passive tracer plugin properly implements
    the required BasePhysicsPlugin interface and follows the expected patterns
    for physics plugins in the SedTRAILS framework.
    
    Key aspects tested:
    - Proper inheritance from BasePhysicsPlugin
    - Required method implementations
    - Method signatures and parameters
    - Initialization behavior
    """
    
    def test_inherits_from_base_plugin(self):
        """Test that PhysicsPlugin inherits from BasePhysicsPlugin.
        
        This ensures the plugin follows the established plugin architecture
        and can be used interchangeably with other physics plugins.
        """
        assert issubclass(PhysicsPlugin, BasePhysicsPlugin), (
            "PhysicsPlugin must inherit from BasePhysicsPlugin"
        )
    
    def test_has_required_methods(self):
        """Test that PhysicsPlugin implements all required methods.
        
        The BasePhysicsPlugin interface requires specific methods to be implemented.  
        This test ensures the passive tracer plugin provides all necessary methods.
        """
        # Required methods from BasePhysicsPlugin
        required_methods = ['add_physics']
        
        for method_name in required_methods:
            assert hasattr(PhysicsPlugin, method_name), (
                f"PhysicsPlugin must implement {method_name} method"
            )
            assert callable(getattr(PhysicsPlugin, method_name)), (
                f"{method_name} must be callable"
            )
    
    def test_add_physics_signature(self):
        """Test that add_physics method has the correct signature.
        
        The add_physics method must accept specific parameters to integrate
        properly with the SedTRAILS transport converter framework.
        """
        import inspect
        
        sig = inspect.signature(PhysicsPlugin.add_physics)
        params = list(sig.parameters.keys())
        
        # Expected parameters: self, sedtrails_data, grain_properties, transport_probability_method
        assert 'self' in params, "add_physics must accept 'self' parameter"
        assert 'sedtrails_data' in params, "add_physics must accept 'sedtrails_data' parameter"
        assert 'grain_properties' in params, "add_physics must accept 'grain_properties' parameter"
        assert 'transport_probability_method' in params, "add_physics must accept 'transport_probability_method' parameter"
    
    def test_initialization(self):
        """Test plugin initialization with config parameters.
        
        The plugin should properly initialize with configuration objects
        and store necessary parameters for later use.
        """
        # Mock configuration objects
        mock_config = Mock()
        mock_tracer_config = Mock()
        
        # Create plugin instance
        plugin = PhysicsPlugin(mock_config, mock_tracer_config)
        
        # Verify initialization
        assert plugin.config is mock_config, "Plugin should store config reference"
        assert isinstance(plugin, BasePhysicsPlugin), "Plugin should be instance of BasePhysicsPlugin"


class TestPassiveTracerPhysics:
    """Test the passive tracer physics implementation.
    
    This class validates the core physics logic of the passive tracer plugin,
    ensuring that particles correctly follow flow fields without additional
    transport processes.
    
    Key physics principles tested:
    - Passive tracers move at flow velocity
    - No settling or resuspension
    - No bed load transport
    - Perfect fluid following behavior
    """
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration object with typical passive tracer settings."""
        config = Mock()
        # Add any specific config attributes needed for passive tracers
        return config
    
    @pytest.fixture
    def mock_tracer_config(self):
        """Mock tracer configuration object."""
        return Mock()
    
    @pytest.fixture
    def sample_sedtrails_data(self):
        """Create sample SedtrailsData for testing.
        
        This fixture provides realistic synthetic data representing typical
        flow conditions in a coastal environment. The data includes:
        - Spatial grid (10x8 cells)
        - Time series (3 time steps)
        - Flow velocity fields with realistic magnitudes and directions
        """
        # Create realistic synthetic flow data
        nx, ny = 10, 8
        nt = 3
        
        # Spatial coordinates (example coastal grid, 100m resolution)
        x = np.linspace(0, 900, nx)  # 900m in x-direction
        y = np.linspace(0, 700, ny)  # 700m in y-direction
        
        # Create mesh grid for 2D fields
        X, Y = np.meshgrid(x, y)
        
        # Time-varying flow velocity (realistic coastal flow: 0.1-0.5 m/s)
        np.random.seed(42)  # For reproducible tests
        
        # Base flow pattern: alongshore flow with some cross-shore component
        flow_u_base = 0.3 * np.ones((ny, nx))  # Alongshore flow ~0.3 m/s
        flow_v_base = 0.1 * np.sin(2 * np.pi * X / 900) * np.cos(2 * np.pi * Y / 700)  # Cross-shore circulation
        
        # Add temporal variation
        flow_velocity_x = np.zeros((nt, ny, nx))
        flow_velocity_y = np.zeros((nt, ny, nx))
        
        for t in range(nt):
            # Add tidal variation (±20% modulation)
            tidal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * t / 12.4)  # 12.4 hour tidal cycle
            flow_velocity_x[t] = flow_u_base * tidal_factor
            flow_velocity_y[t] = flow_v_base * tidal_factor
        
        # Compute magnitude
        flow_velocity_magnitude = np.sqrt(flow_velocity_x**2 + flow_velocity_y**2)
        
        # Create mock SedtrailsData object
        sedtrails_data = Mock(spec=SedtrailsData)
        
        # Set up flow velocity data
        sedtrails_data.depth_avg_flow_velocity = {
            'x': flow_velocity_x,
            'y': flow_velocity_y,
            'magnitude': flow_velocity_magnitude
        }
        
        # Mock the add_physics_field method
        sedtrails_data.add_physics_field = Mock()
        
        return sedtrails_data
    
    @pytest.fixture
    def sample_grain_properties(self):
        """Sample grain properties (not used by passive tracers but required by interface)."""
        return {
            'grain_diameter': 250e-6,  # 250 microns
            'settling_velocity': 0.02,  # m/s
            'critical_shields': 0.047,
            'dimensionless_grain_size': 5.2
        }
    
    def test_particle_velocity_equals_flow_velocity(self, mock_config, mock_tracer_config, 
                                                   sample_sedtrails_data, sample_grain_properties):
        """Test that particle velocities equal flow velocities for passive tracers.
        
        This is the fundamental physics of passive tracers: they move exactly
        at the flow velocity without any lag, settling, or other modifications.
        The test verifies that:
        - Particle velocity components match flow velocity components exactly
        - No additional physics processes are applied
        - Data structures are preserved
        """
        # Create plugin instance
        plugin = PhysicsPlugin(mock_config, mock_tracer_config)
        
        # Call add_physics method
        plugin.add_physics(sample_sedtrails_data, sample_grain_properties, 'no_probability')
        
        # Verify that add_physics_field was called once with correct arguments
        sample_sedtrails_data.add_physics_field.assert_called_once()
        
        # Get the call arguments
        call_args = sample_sedtrails_data.add_physics_field.call_args
        field_name = call_args[0][0]
        field_data = call_args[0][1]
        
        # Verify field name
        assert field_name == 'particle_velocity', (
            "Passive tracer should add 'particle_velocity' field"
        )
        
        # Verify field structure
        expected_keys = {'x', 'y', 'magnitude'}
        assert set(field_data.keys()) == expected_keys, (
            f"Particle velocity field should have keys {expected_keys}"
        )
        
        # Verify that particle velocities exactly match flow velocities
        flow_data = sample_sedtrails_data.depth_avg_flow_velocity
        
        # Use numpy array comparison for exact equality (passive tracers should be identical to flow)
        assert_array_equal(
            field_data['x'], flow_data['x'],
            err_msg="Particle velocity x-component should exactly equal flow velocity x-component"
        )
        assert_array_equal(
            field_data['y'], flow_data['y'],
            err_msg="Particle velocity y-component should exactly equal flow velocity y-component"
        )
        assert_array_equal(
            field_data['magnitude'], flow_data['magnitude'],
            err_msg="Particle velocity magnitude should exactly equal flow velocity magnitude"
        )
    
    def test_no_dependence_on_grain_properties(self, mock_config, mock_tracer_config, 
                                              sample_sedtrails_data):
        """Test that passive tracers are independent of grain properties.
        
        Unlike active sediment transport, passive tracers should not depend on
        particle size, density, settling velocity, or other grain characteristics.
        This test verifies that different grain properties produce identical results.
        """
        plugin = PhysicsPlugin(mock_config, mock_tracer_config)
        
        # Test with different grain property sets
        grain_props_1 = {
            'grain_diameter': 100e-6,  # Fine sand
            'settling_velocity': 0.01,
            'critical_shields': 0.035,
        }
        
        grain_props_2 = {
            'grain_diameter': 500e-6,  # Coarse sand
            'settling_velocity': 0.05,
            'critical_shields': 0.055,
        }
        
        # Create separate mock data objects to capture different calls
        data_1 = Mock(spec=SedtrailsData)
        data_1.depth_avg_flow_velocity = sample_sedtrails_data.depth_avg_flow_velocity
        data_1.add_physics_field = Mock()
        
        data_2 = Mock(spec=SedtrailsData)
        data_2.depth_avg_flow_velocity = sample_sedtrails_data.depth_avg_flow_velocity
        data_2.add_physics_field = Mock()
        
        # Run physics with different grain properties
        plugin.add_physics(data_1, grain_props_1, 'no_probability')
        plugin.add_physics(data_2, grain_props_2, 'no_probability')
        
        # Extract results
        result_1 = data_1.add_physics_field.call_args[0][1]
        result_2 = data_2.add_physics_field.call_args[0][1]
        
        # Results should be identical regardless of grain properties
        for component in ['x', 'y', 'magnitude']:
            assert_array_equal(
                result_1[component], result_2[component],
                err_msg=f"Passive tracer {component} velocity should be independent of grain properties"
            )
    
    def test_no_dependence_on_transport_probability_method(self, mock_config, mock_tracer_config,
                                                          sample_sedtrails_data, sample_grain_properties):
        """Test that passive tracers are independent of transport probability method.
        
        Transport probability methods apply to active sediment transport but should
        not affect passive tracers, which always move with the flow.
        """
        plugin = PhysicsPlugin(mock_config, mock_tracer_config)
        
        # Test different transport probability methods
        methods = ['no_probability', 'soulsby_probability', 'other_method']
        
        results = []
        for method in methods:
            # Create separate mock for each test
            mock_data = Mock(spec=SedtrailsData)
            mock_data.depth_avg_flow_velocity = sample_sedtrails_data.depth_avg_flow_velocity
            mock_data.add_physics_field = Mock()
            
            plugin.add_physics(mock_data, sample_grain_properties, method)
            
            # Extract result
            result = mock_data.add_physics_field.call_args[0][1]
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            for component in ['x', 'y', 'magnitude']:
                assert_array_equal(
                    results[0][component], results[i][component],
                    err_msg=f"Passive tracer velocities should be independent of transport probability method"
                )


class TestPassiveTracerDataIntegration:
    """Test passive tracer integration with SedtrailsData structures.
    
    This class validates that the passive tracer plugin correctly interacts
    with SedtrailsData objects, including proper data access, field addition,
    and error handling for missing or invalid data.
    """
    
    @pytest.fixture
    def plugin(self):
        """Create plugin instance for testing."""
        return PhysicsPlugin(Mock(), Mock())
    
    def test_handles_missing_flow_velocity_gracefully(self, plugin):
        """Test behavior when flow velocity data is missing.
        
        The plugin should handle cases where the required flow velocity data
        is not available in the SedtrailsData object.
        """
        # Create SedtrailsData with missing flow velocity
        sedtrails_data = Mock(spec=SedtrailsData)
        sedtrails_data.depth_avg_flow_velocity = None
        
        # This should raise an appropriate error
        with pytest.raises(TypeError):  # 'NoneType' object is not subscriptable
            plugin.add_physics(sedtrails_data, {}, 'no_probability')
    
    def test_handles_incomplete_flow_velocity_data(self, plugin):
        """Test behavior with incomplete flow velocity data.
        
        Flow velocity should have 'x', 'y', and 'magnitude' components.
        Test handling of cases where some components are missing.
        """
        # Create SedtrailsData with incomplete flow velocity data
        sedtrails_data = Mock(spec=SedtrailsData)
        sedtrails_data.depth_avg_flow_velocity = {
            'x': np.random.rand(5, 5),
            # Missing 'y' and 'magnitude' components
        }
        sedtrails_data.add_physics_field = Mock()
        
        # This should raise a KeyError when trying to access missing components
        with pytest.raises(KeyError):
            plugin.add_physics(sedtrails_data, {}, 'no_probability')
    
    def test_preserves_data_structure(self, plugin):
        """Test that the plugin preserves the original data structure.
        
        The passive tracer plugin should not modify the original flow velocity
        data and should maintain proper array shapes and data types.
        """
        # Create test data with specific shapes and types
        nt, ny, nx = 2, 4, 6
        flow_x = np.random.rand(nt, ny, nx).astype(np.float32)
        flow_y = np.random.rand(nt, ny, nx).astype(np.float64)
        flow_mag = np.sqrt(flow_x**2 + flow_y**2)
        
        sedtrails_data = Mock(spec=SedtrailsData)
        sedtrails_data.depth_avg_flow_velocity = {
            'x': flow_x,
            'y': flow_y, 
            'magnitude': flow_mag
        }
        sedtrails_data.add_physics_field = Mock()
        
        # Run physics
        plugin.add_physics(sedtrails_data, {}, 'no_probability')
        
        # Extract added field data
        added_field = sedtrails_data.add_physics_field.call_args[0][1]
        
        # Verify shapes are preserved
        assert added_field['x'].shape == (nt, ny, nx), "X-velocity shape should be preserved"
        assert added_field['y'].shape == (nt, ny, nx), "Y-velocity shape should be preserved"
        assert added_field['magnitude'].shape == (nt, ny, nx), "Magnitude shape should be preserved"
        
        # Verify data types are preserved (or at least compatible)
        assert np.issubdtype(added_field['x'].dtype, np.floating), "X-velocity should be floating point"
        assert np.issubdtype(added_field['y'].dtype, np.floating), "Y-velocity should be floating point"
        assert np.issubdtype(added_field['magnitude'].dtype, np.floating), "Magnitude should be floating point"


class TestPassiveTracerEdgeCases:
    """Test edge cases and boundary conditions for passive tracers.
    
    Edge case testing ensures the passive tracer plugin is robust across
    the full range of possible input conditions, including extreme values,
    special array configurations, and numerical edge cases.
    
    Testing philosophy:
    - Handle zero velocities gracefully
    - Manage very large/small velocity magnitudes
    - Process different array shapes correctly
    - Maintain numerical stability
    """
    
    @pytest.fixture
    def plugin(self):
        """Create plugin instance for edge case testing."""
        return PhysicsPlugin(Mock(), Mock())
    
    @pytest.mark.parametrize("velocity_scale", [0.0, 1e-10, 1e-6, 1.0, 10.0, 1e6])
    def test_velocity_magnitude_range(self, plugin, velocity_scale):
        """Test passive tracer behavior across velocity magnitude ranges.
        
        This parametrized test validates that the passive tracer plugin
        handles the full range of possible velocity magnitudes:
        
        - **Zero velocity (0.0)**: Stagnant conditions, tracers should be stationary
        - **Very small (1e-10, 1e-6)**: Near-zero flows, numerical precision tests
        - **Typical (1.0)**: Normal coastal flow conditions
        - **Large (10.0)**: High-energy flows (storms, rivers)
        - **Very large (1e6)**: Extreme conditions, overflow/stability tests
        """
        # Create test data with specified velocity scale
        ny, nx = 3, 4
        base_u = np.ones((1, ny, nx)) * velocity_scale
        base_v = np.ones((1, ny, nx)) * velocity_scale * 0.5
        base_mag = np.sqrt(base_u**2 + base_v**2)
        
        sedtrails_data = Mock(spec=SedtrailsData)
        sedtrails_data.depth_avg_flow_velocity = {
            'x': base_u,
            'y': base_v,
            'magnitude': base_mag
        }
        sedtrails_data.add_physics_field = Mock()
        
        # Run physics - should not raise any errors
        plugin.add_physics(sedtrails_data, {}, 'no_probability')
        
        # Verify physics field was added
        sedtrails_data.add_physics_field.assert_called_once()
        
        # Extract results
        result = sedtrails_data.add_physics_field.call_args[0][1]
        
        # Results should be finite and preserve the input values
        for component in ['x', 'y', 'magnitude']:
            assert np.all(np.isfinite(result[component])), (
                f"All {component} velocities should be finite for scale {velocity_scale}"
            )
        
        # Verify values are preserved exactly
        assert_array_equal(result['x'], base_u)
        assert_array_equal(result['y'], base_v) 
        assert_array_equal(result['magnitude'], base_mag)
    
    @pytest.mark.parametrize("array_shape", [
        (1, 1, 1),      # Minimal grid
        (1, 10, 1),     # 1D line (y-direction)
        (1, 1, 10),     # 1D line (x-direction)
        (3, 5, 7),      # Typical 2D grid
        (10, 100, 200), # Large grid
    ])
    def test_different_array_shapes(self, plugin, array_shape):
        """Test passive tracer behavior with different array shapes.
        
        This test ensures the plugin correctly handles various spatial grid
        configurations that might be encountered in different modeling scenarios:
        
        - **Point models**: Single grid cell (1,1,1)
        - **1D models**: Line geometries for channel/beach profiles
        - **2D models**: Typical coastal/estuarine grids
        - **Large grids**: High-resolution regional models
        
        The plugin should preserve array shapes and handle all configurations.
        """
        nt, ny, nx = array_shape
        
        # Create realistic velocity data for the given shape
        np.random.seed(42)  # Reproducible random data
        flow_x = np.random.uniform(-1, 1, (nt, ny, nx))
        flow_y = np.random.uniform(-1, 1, (nt, ny, nx))
        flow_mag = np.sqrt(flow_x**2 + flow_y**2)
        
        sedtrails_data = Mock(spec=SedtrailsData)
        sedtrails_data.depth_avg_flow_velocity = {
            'x': flow_x,
            'y': flow_y,
            'magnitude': flow_mag
        }
        sedtrails_data.add_physics_field = Mock()
        
        # Run physics
        plugin.add_physics(sedtrails_data, {}, 'no_probability')
        
        # Verify correct field addition
        sedtrails_data.add_physics_field.assert_called_once()
        result = sedtrails_data.add_physics_field.call_args[0][1]
        
        # Verify shapes are preserved
        assert result['x'].shape == array_shape, f"X-velocity shape mismatch for {array_shape}"
        assert result['y'].shape == array_shape, f"Y-velocity shape mismatch for {array_shape}"
        assert result['magnitude'].shape == array_shape, f"Magnitude shape mismatch for {array_shape}"
        
        # Verify values are exactly preserved
        assert_array_equal(result['x'], flow_x, err_msg=f"X-velocity values not preserved for shape {array_shape}")
        assert_array_equal(result['y'], flow_y, err_msg=f"Y-velocity values not preserved for shape {array_shape}")
        assert_array_equal(result['magnitude'], flow_mag, err_msg=f"Magnitude values not preserved for shape {array_shape}")
    
    def test_zero_velocity_fields(self, plugin):
        """Test handling of zero velocity fields.
        
        Zero velocity represents stagnant conditions where passive tracers
        should remain stationary. This is an important edge case for:
        - Initialization conditions
        - Areas of flow recirculation
        - Numerical solver convergence
        """
        # Create zero velocity field
        shape = (2, 5, 8)
        zero_field = np.zeros(shape)
        
        sedtrails_data = Mock(spec=SedtrailsData)
        sedtrails_data.depth_avg_flow_velocity = {
            'x': zero_field.copy(),
            'y': zero_field.copy(),
            'magnitude': zero_field.copy()
        }
        sedtrails_data.add_physics_field = Mock()
        
        # Run physics
        plugin.add_physics(sedtrails_data, {}, 'no_probability')
        
        # Extract results
        result = sedtrails_data.add_physics_field.call_args[0][1]
        
        # All particle velocities should be zero
        assert_array_equal(result['x'], zero_field, err_msg="Particle x-velocity should be zero when flow is zero")
        assert_array_equal(result['y'], zero_field, err_msg="Particle y-velocity should be zero when flow is zero")
        assert_array_equal(result['magnitude'], zero_field, err_msg="Particle velocity magnitude should be zero when flow is zero")
    
    def test_nan_and_inf_handling(self, plugin):
        """Test behavior with NaN and infinite velocity values.
        
        While not physically realistic, NaN and infinite values can occur
        in numerical models due to:
        - Numerical instabilities
        - Division by zero
        - Uninitialized data regions
        
        The plugin should preserve these values to maintain data integrity
        and allow downstream error detection.
        """
        # Create velocity field with special values
        shape = (1, 3, 3)
        flow_x = np.array([[[1.0, np.nan, 2.0],
                           [np.inf, 0.0, -np.inf],
                           [3.0, 4.0, 5.0]]])
        flow_y = np.array([[[0.5, 1.5, np.nan],
                           [2.5, np.inf, 3.5],
                           [-np.inf, 4.5, 5.5]]])
        flow_mag = np.sqrt(flow_x**2 + flow_y**2)  # Will contain NaN and inf
        
        sedtrails_data = Mock(spec=SedtrailsData)
        sedtrails_data.depth_avg_flow_velocity = {
            'x': flow_x,
            'y': flow_y,
            'magnitude': flow_mag
        }
        sedtrails_data.add_physics_field = Mock()
        
        # Run physics - should not crash
        plugin.add_physics(sedtrails_data, {}, 'no_probability')
        
        # Extract results
        result = sedtrails_data.add_physics_field.call_args[0][1]
        
        # Special values should be preserved exactly
        assert_array_equal(result['x'], flow_x, err_msg="Special values in x-velocity should be preserved")
        assert_array_equal(result['y'], flow_y, err_msg="Special values in y-velocity should be preserved")
        assert_array_equal(result['magnitude'], flow_mag, err_msg="Special values in magnitude should be preserved")


class TestPassiveTracerIntegration:
    """Integration tests for passive tracer plugin.
    
    These tests validate the plugin behavior in realistic usage scenarios,
    including interaction with actual SedtrailsData objects and integration
    with the broader SedTRAILS framework.
    """
    
    def test_realistic_coastal_scenario(self):
        """Test passive tracer plugin with realistic coastal flow data.
        
        This integration test simulates a typical coastal modeling scenario
        with realistic flow patterns, spatial scales, and temporal evolution.
        It validates that the plugin works correctly in conditions similar
        to actual field applications.
        """
        # Create realistic coastal flow scenario
        # Spatial domain: 2km x 1km coastal area, 50m resolution
        nx, ny = 40, 20
        nt = 24  # 24 hour simulation
        
        x = np.linspace(0, 2000, nx)  # 2km alongshore
        y = np.linspace(0, 1000, ny)  # 1km cross-shore
        X, Y = np.meshgrid(x, y)
        
        # Realistic tidal flow with alongshore and cross-shore components
        flow_velocity_x = np.zeros((nt, ny, nx))
        flow_velocity_y = np.zeros((nt, ny, nx))
        
        for t in range(nt):
            # Tidal elevation (semidiurnal, 2m amplitude)
            tidal_phase = 2 * np.pi * t / 12.42  # 12.42 hour tidal period
            tidal_elevation = 2.0 * np.sin(tidal_phase)
            
            # Alongshore tidal current (0.5 m/s amplitude)
            flow_velocity_x[t] = 0.5 * np.sin(tidal_phase) * (1 + 0.3 * Y / 1000)
            
            # Cross-shore circulation (weaker, depth-dependent)
            flow_velocity_y[t] = 0.1 * np.cos(tidal_phase) * np.sin(np.pi * Y / 1000)
        
        flow_velocity_magnitude = np.sqrt(flow_velocity_x**2 + flow_velocity_y**2)
        
        # Create plugin and run physics
        plugin = PhysicsPlugin(Mock(), Mock())
        
        # Create mock SedtrailsData
        sedtrails_data = Mock(spec=SedtrailsData)
        sedtrails_data.depth_avg_flow_velocity = {
            'x': flow_velocity_x,
            'y': flow_velocity_y,
            'magnitude': flow_velocity_magnitude
        }
        sedtrails_data.add_physics_field = Mock()
        
        # Run with realistic parameters
        grain_properties = {
            'grain_diameter': 200e-6,
            'settling_velocity': 0.025,
            'critical_shields': 0.047
        }
        
        plugin.add_physics(sedtrails_data, grain_properties, 'soulsby_probability')
        
        # Verify realistic results
        sedtrails_data.add_physics_field.assert_called_once()
        result = sedtrails_data.add_physics_field.call_args[0][1]
        
        # Check that particle velocities match flow velocities exactly
        assert_array_equal(result['x'], flow_velocity_x)
        assert_array_equal(result['y'], flow_velocity_y)
        assert_array_equal(result['magnitude'], flow_velocity_magnitude)
        
        # Verify realistic velocity ranges
        max_speed = np.max(result['magnitude'])
        assert 0 <= max_speed <= 1.0, f"Maximum particle speed ({max_speed:.3f} m/s) should be realistic for coastal flows"


if __name__ == "__main__":
    """
    Direct test execution capability.
    
    This allows the test file to be run directly with:
    python test_passive.py
    
    Enables quick validation during development and provides verbose output
    for understanding test coverage and plugin behavior.
    """
    # Run tests when script is executed directly with comprehensive output
    pytest.main([__file__, "-v", "--tb=short"])