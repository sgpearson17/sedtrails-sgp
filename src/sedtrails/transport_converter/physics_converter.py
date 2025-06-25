"""
Physics Converter Module for Sediment Transport

This module adds physics-based calculations to existing SedtrailsData objects
using the physics library functions and allowing method selection.

References:
-----------
van Westen, B., de Schipper, M. A., Pearson, S. G., & Luijendijk, A. P. (2025). 
Lagrangian modelling reveals sediment pathways at evolving coasts. 
Scientific Reports, 15(1), 8793.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

# Import physics library
from sedtrails.transport_converter import physics_lib

class PhysicsMethod:
    VAN_WESTEN_2025 = "van_westen_2025"
    SOULSBY_2011 = "soulsby_2011"


@dataclass
class PhysicsConfig:
    """Configuration parameters for physics calculations."""
    
    # Physical constants
    gravity: float = 9.81  # m/s^2
    von_karman_constant: float = 0.40  # [-]
    kinematic_viscosity: float = 1.36e-6  # m^2/s (10°C, 35 ppt)
    water_density: float = 1027.0  # kg/m^3 (10°C, 35 ppt)
    particle_density: float = 2650.0  # kg/m^3 (quartz)
    porosity: float = 0.4  # [-]
    
    # Sediment properties
    grain_diameter: float = 2.5e-4  # m (250 μm)
    
    # Morphological acceleration factor
    morfac: float = 1.0

    # Physics methods
    tracer_method: str = PhysicsMethod.VAN_WESTEN_2025

    # Methods in physics library
    suspended_velocity_method: str = physics_lib.SuspendedVelocityMethod.VAN_WESTEN_2025
    mixing_layer_method: str = physics_lib.MixingLayerMethod.BERTIN_2008


class PhysicsConverter:
    """
    Add physics calculations to existing SedtrailsData using different methods.
    
    Each method (van Westen, Soulsby, etc.) has its own complete workflow.
    """
    
    def __init__(self, config: Optional[PhysicsConfig] = None):
        """
        Initialize the physics converter.
        
        Parameters:
        -----------
        config : PhysicsConfig, optional
            Configuration for physics calculations. If None, uses defaults.
        """
        self.config = config or PhysicsConfig()
        
        # Calculate time-independent grain properties using library
        self._calculate_grain_properties()
    
    def _calculate_grain_properties(self):
        """Calculate time-independent grain properties using physics library."""
        (self.dimensionless_grain_size, 
         self.critical_shields, 
         self.settling_velocity, 
         self.critical_shear_stress) = physics_lib.compute_grain_properties(
            self.config.grain_diameter,
            self.config.gravity,
            self.config.particle_density,
            self.config.water_density,
            self.config.kinematic_viscosity
        )
    
    def add_physics_to_sedtrails_data(self, sedtrails_data, method: Optional[str] = None):
        """
        Add physics calculations to existing SedtrailsData object using specified method.

        Parameters:
        -----------
        sedtrails_data : SedtrailsData
            Existing SedtrailsData object to enhance with physics
        method : str, optional
            Physics method to use:
            - "van_westen_2025": van Westen et al. (2025) approach (default)
            - "soulsby_2011": Soulsby et al. (2011) approach 
            If None, uses the method from self.config.tracer_method.
        """
        chosen_method = method or self.config.tracer_method
        print(f"Adding physics to SedtrailsData using {chosen_method} method...")

        if chosen_method == PhysicsMethod.VAN_WESTEN_2025:
            self._add_physics_van_westen(sedtrails_data)

        elif chosen_method == PhysicsMethod.SOULSBY_2011:
            self._add_physics_soulsby(sedtrails_data)

        else:
            raise ValueError(f"Unknown physics method: {chosen_method}. "
                             f"Available methods: '{PhysicsMethod.VAN_WESTEN_2025}', '{PhysicsMethod.SOULSBY_2011}'")

        print(f"Physics calculations complete using {chosen_method} method!")

    def _add_physics_van_westen(self, sedtrails_data):
        """
        Add physics using van Westen et al. (2025) approach.
        
        This method follows the workflow from van Westen et al. (2025):
        1. Compute shear velocities and Shields number
        2. Compute bed load velocity (Soulsby equation)
        3. Compute suspended velocity using complex ratio formula
        4. Compute layer thicknesses and mixing layer
        """
        print("Using van Westen et al. (2025) to compute transport velocities and add to SedTRAILS data...")
        
        # === LOAD: Extract data ===
        flow_velocity_magnitude = sedtrails_data.depth_avg_flow_velocity['magnitude']
        mean_bed_shear_stress = sedtrails_data.mean_bed_shear_stress
        max_bed_shear_stress = sedtrails_data.max_bed_shear_stress
        
        bed_load_transport_x = sedtrails_data.bed_load_transport['x']
        bed_load_transport_y = sedtrails_data.bed_load_transport['y']
        bed_load_transport_magnitude = sedtrails_data.bed_load_transport['magnitude']

        suspended_transport_x = sedtrails_data.suspended_transport['x']
        suspended_transport_y = sedtrails_data.suspended_transport['y']
        suspended_transport_magnitude = sedtrails_data.suspended_transport['magnitude']

        # FIX: Automatically detect and handle fraction dimension
        # Detect number of fractions from data shape (assuming shape is [time, fractions, spatial])
        detected_fractions = bed_load_transport_magnitude.shape[1] if len(bed_load_transport_magnitude.shape) > 2 else 1
        
        print(f"Detected {detected_fractions} sediment fraction(s)")
        
        # Error if more than 1 fraction
        if detected_fractions > 1:
            raise NotImplementedError(
                f"Multiple sediment fractions ({detected_fractions}) are not yet supported. "
                f"The physics converter currently only handles single-fraction sediment transport. "
                f"Please aggregate fractions or implement multi-fraction physics calculations."
            )
        
        # For physics calculations, we need to work with squeezed data (no fraction dimension)
        # but we'll add the dimension back to velocities at the end
        if len(bed_load_transport_magnitude.shape) > 2:
            print("Working with squeezed transport data for physics calculations")
            bed_load_transport_x_calc = bed_load_transport_x.squeeze(axis=1)
            bed_load_transport_y_calc = bed_load_transport_y.squeeze(axis=1)
            bed_load_transport_magnitude_calc = bed_load_transport_magnitude.squeeze(axis=1)
            
            suspended_transport_x_calc = suspended_transport_x.squeeze(axis=1)
            suspended_transport_y_calc = suspended_transport_y.squeeze(axis=1)
            suspended_transport_magnitude_calc = suspended_transport_magnitude.squeeze(axis=1)
            has_fraction_dim = True
        else:
            # Data already doesn't have fraction dimension
            bed_load_transport_x_calc = bed_load_transport_x
            bed_load_transport_y_calc = bed_load_transport_y
            bed_load_transport_magnitude_calc = bed_load_transport_magnitude
            
            suspended_transport_x_calc = suspended_transport_x
            suspended_transport_y_calc = suspended_transport_y
            suspended_transport_magnitude_calc = suspended_transport_magnitude
            has_fraction_dim = False
        
        # === COMPUTE ===

        # Compute shear velocities
        mean_shear_velocity = physics_lib.compute_shear_velocity(
            mean_bed_shear_stress, self.config.water_density)
        max_shear_velocity = physics_lib.compute_shear_velocity(
            max_bed_shear_stress, self.config.water_density)
        
        # Compute Shields number
        shields_number = physics_lib.compute_shields(
            max_bed_shear_stress, self.config.gravity,
            self.config.particle_density, self.config.water_density, self.config.grain_diameter)
        
        # Compute transport velocities (these will have shape [time, spatial])
        bed_load_velocity = physics_lib.compute_bed_load_velocity(
            shields_number, self.critical_shields, mean_shear_velocity)
        
        suspended_velocity = physics_lib.compute_suspended_velocity(
            flow_velocity_magnitude, bed_load_velocity, self.settling_velocity,
            self.config.von_karman_constant, max_shear_velocity,
            shields_number, self.critical_shields,
            method=physics_lib.SuspendedVelocityMethod.SOULSBY_2011)
        
        # Compute layer thicknesses using squeezed transport data
        bed_load_layer_thickness = physics_lib.compute_transport_layer_thickness(
            bed_load_transport_magnitude_calc, bed_load_velocity,
            self.config.particle_density, self.config.porosity)
        
        suspended_layer_thickness = physics_lib.compute_transport_layer_thickness(
            suspended_transport_magnitude_calc, suspended_velocity,
            self.config.particle_density, self.config.porosity)
        
        # Compute directions from magnitudes using squeezed transport data
        suspended_velocity_x, suspended_velocity_y = physics_lib.compute_directions_from_magnitude(
            suspended_velocity, suspended_transport_x_calc, suspended_transport_y_calc, suspended_transport_magnitude_calc)
        bed_load_velocity_x, bed_load_velocity_y = physics_lib.compute_directions_from_magnitude(
            bed_load_velocity, bed_load_transport_x_calc, bed_load_transport_y_calc, bed_load_transport_magnitude_calc)
        
        # Compute mixing layer thickness using Bertin et al. (2008) method
        mixing_layer_thickness = physics_lib.compute_mixing_layer_thickness(
            max_bed_shear_stress, self.critical_shear_stress,
            method=physics_lib.MixingLayerMethod.BERTIN_2008)
        
        # === EXPAND DIMENSIONS TO MATCH ORIGINAL STRUCTURE ===
        if has_fraction_dim:
            print("Expanding velocity dimensions to match transport data structure")
            # Add fraction dimension (axis=1) to all computed physics quantities
            shields_number = shields_number[:, np.newaxis, :]
            bed_load_layer_thickness = bed_load_layer_thickness[:, np.newaxis, :]
            suspended_layer_thickness = suspended_layer_thickness[:, np.newaxis, :]
            mixing_layer_thickness = mixing_layer_thickness[:, np.newaxis, :]
            
            # Expand velocity components
            bed_load_velocity = bed_load_velocity[:, np.newaxis, :]
            bed_load_velocity_x = bed_load_velocity_x[:, np.newaxis, :]
            bed_load_velocity_y = bed_load_velocity_y[:, np.newaxis, :]
            
            suspended_velocity = suspended_velocity[:, np.newaxis, :]
            suspended_velocity_x = suspended_velocity_x[:, np.newaxis, :]
            suspended_velocity_y = suspended_velocity_y[:, np.newaxis, :]
        
        # === STORE ===
        
        print("Adding physics fields to SedtrailsData...")
        
        # Physics parameters (scalar fields)
        sedtrails_data.add_physics_field('shields_number', shields_number)
        sedtrails_data.add_physics_field('bed_load_layer_thickness', bed_load_layer_thickness)
        sedtrails_data.add_physics_field('suspended_layer_thickness', suspended_layer_thickness)
        sedtrails_data.add_physics_field('mixing_layer_thickness', mixing_layer_thickness)
        
        # Sediment velocities (vector fields)
        sedtrails_data.add_physics_field('bed_load_velocity', {
            'x': bed_load_velocity_x, 'y': bed_load_velocity_y, 'magnitude': bed_load_velocity
        })
        sedtrails_data.add_physics_field('suspended_velocity', {
            'x': suspended_velocity_x, 'y': suspended_velocity_y, 'magnitude': suspended_velocity
        })

    
    def _add_physics_soulsby(self, sedtrails_data):
        """
        Add physics using Soulsby et al. (2011) approach.
        """
        
        # Placeholder for Soulsby calculations
        raise NotImplementedError(
            "Soulsby et al. (2011) method not yet implemented. "
            "This would have a completely different workflow:\n"
            "1. Focus on individual particle tracking velocities\n"
            "2. Different approach to settling and resuspension\n"
            "3. Particle-specific rather than layer-based calculations\n"
            "See: Soulsby, R. L., et al. (2011). Lagrangian model for simulating "
            "the dispersal of sand-sized particles in coastal waters."
        )