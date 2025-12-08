"""A plugin for Soulsby et al. (2011) sediment transport physics calculations."""

from cmath import tau
import numpy as np
from sedtrails.transport_converter import physics_lib
from sedtrails.transport_converter import SedtrailsData
from sedtrails.transport_converter.plugins import BasePhysicsPlugin


class PhysicsPlugin(BasePhysicsPlugin):  # all clases should be called the PhysicsPlugin
    """
    Plugin for Soulsby et al. (2011) sediment transport physics calculations.
    This plugin implements the physics calculations as described in Soulsby et al. (2011).
    """

    def __init__(self, config, tracer_config):
        super().__init__()
        self.config = config

    def add_physics(self, sedtrails_data: SedtrailsData, grain_properties: dict[str, float], transport_probability_method: str) -> None:
        """
        Add physics using Soulsby et al. (2011) approach.
        '1. Focus on individual particle tracking velocities\n'
        '2. Different approach to settling and resuspension\n'
        '3. Particle-specific rather than layer-based calculations\n'
        'See: Soulsby, R. L., et al. (2011). Lagrangian model for simulating '
        'the dispersal of sand-sized particles in coastal waters.'
        """
        print('Using Soulsby et al. (2011) to compute transport velocities and add to SedTRAILS data...')

        # === LOAD: Extract data ===

        # Extract constants
        g = self.config.gravity
        rho_s = self.config.particle_density
        rho_w = self.config.water_density
        kinematic_viscosity = self.config.kinematic_viscosity
        von_karman = self.config.von_karman_constant

        # Extract shear stresses
        mean_bed_shear_stress = sedtrails_data.mean_bed_shear_stress
        if mean_bed_shear_stress is None:
            raise ValueError('Missing required shear stress values in SedtrailsData.')
        max_bed_shear_stress = sedtrails_data.max_bed_shear_stress
        if mean_bed_shear_stress is None or max_bed_shear_stress is None:
            raise ValueError('Missing required shear stress values in SedtrailsData.')

        # Extract particle properties
        grain_size = self.config.tracer_grain_size
        background_grain_size = self.config.background_grain_size
        dimensionless_grain_size = grain_properties.get('dimensionless_grain_size')
        critical_shields = grain_properties.get('critical_shields')
        settling_velocity = grain_properties.get('settling_velocity')
        
        # Validate required grain properties
        if critical_shields is None:
            raise ValueError('critical_shields is required for Soulsby physics calculations but was not found in grain_properties')
        if dimensionless_grain_size is None:
            raise ValueError('dimensionless_grain_size is required for Soulsby physics calculations but was not found in grain_properties')
        if settling_velocity is None:
            raise ValueError('settling_velocity is required for Soulsby physics calculations but was not found in grain_properties')

        # Extract flow velocities (we should be able to change these based on configuration)
        flow_velocity_x = sedtrails_data.depth_avg_flow_velocity['x']
        flow_velocity_y = sedtrails_data.depth_avg_flow_velocity['y']
        flow_velocity_magnitude = sedtrails_data.depth_avg_flow_velocity['magnitude']
        

        # Extract Soulsby et al. (2011) empirical parameters
        soulsby_b_e = self.config.soulsby_b_e
        soulsby_theta_s = self.config.soulsby_theta_s
        soulsby_gamma_e = self.config.soulsby_gamma_e
        soulsby_mu_d = self.config.soulsby_mu_d
        suspended_lag = self.config.suspended_lag
        deposition_bed_shear_stress = self.config.deposition_bed_shear_stress
        
        if suspended_lag:
            print('  - Suspended load settling lag enabled.')
            # Extract water depth
            water_depth = sedtrails_data.water_depth

        # === COMPUTE ===

        # Compute shear velocities
        mean_shear_velocity = physics_lib.compute_shear_velocity(
            mean_bed_shear_stress,
            rho_w,
        )
        max_shear_velocity = physics_lib.compute_shear_velocity(
            max_bed_shear_stress,
            rho_w,
        )

        # Compute Shields number
        shields_number = physics_lib.compute_shields(
            max_bed_shear_stress,
            g,
            rho_s,
            rho_w,
            grain_size,
        )

        # Compute bed load velocity
        bed_load_velocity = physics_lib.compute_bed_load_velocity(
            shields_number,
            critical_shields,
            mean_shear_velocity,
        )

        # Compute additional particle properties
        nd_background_grain_diameter = (g * (rho_s / rho_w - 1) / (kinematic_viscosity**2)) ** (
            1 / 3
        ) * background_grain_size  # nd = non-dimensional
        grain_size_ratio = grain_size / background_grain_size
        rouse_number = settling_velocity / (von_karman * max_shear_velocity)

        # Compute Soulsby et al. (2011) physical parameters
        background_theta_max = max_bed_shear_stress / (g * (rho_s - rho_w) * background_grain_size)
        background_theta_cr = 0.3 / (1 + 1.2 * nd_background_grain_diameter) + 0.055 * (
            1 - np.exp(-0.020 * nd_background_grain_diameter)
        )
        theta_max = max_bed_shear_stress / (g * (rho_s - rho_w) * grain_size)
        theta_cr = 0.3 / (1 + 1.2 * dimensionless_grain_size) + 0.055 * (1 - np.exp(-0.020 * dimensionless_grain_size))
        theta_cr_exp = (
            theta_cr
            * np.sqrt(8 / (3 * (grain_size_ratio**2) + 6 * grain_size_ratio - 1))
            * (
                (3.2260 * grain_size_ratio)
                / (
                    4 * grain_size_ratio
                    - 2 * (grain_size_ratio + 1 - np.sqrt(grain_size_ratio**2 + 2 * grain_size_ratio - 1 / 3))
                )
            )
        )

        # Compute the transition probability b [-] (Equations 3 and 4)
        soulsby_b = np.zeros(theta_max.shape)
        if transport_probability_method != 'no_probability':
            mask = background_theta_max > background_theta_cr
            soulsby_b[mask] = soulsby_b_e * (
                1 - np.exp(-(background_theta_max[mask] - background_theta_cr) / soulsby_theta_s)
            )
 

 
        # Compute the transition probability a [-] (Equation 5)
        soulsby_a = soulsby_gamma_e * soulsby_b / (1 - soulsby_gamma_e)

                       
        # # Add quasi steady option to soulsby (default), otherwise do settling lag
        # # I think this belongs at the particle level
        # if transport_response_method == 'quasi_steady':
            
        #     if is_bedload
        #         if tau > tau_cr
        #             # normal behaviour
        #         else 
        #             # deposited on bed
        #     elif is_suspended
        #         if tau > tau_cr
        #             # normal behaviour, P=1
        #         elif tau > tau_d
        #             # settling with lag
        #             # don't stop moving though
        #         else 
        #             # deposited on bed
        #     else
        #         # immobile
            

        # # First vectorize loops in Soulsby 

        # # Then go through and modify P
        # # Add bedload and susp load states
        # # If susp and below tau_c but above tau_d, p > nonzero until tau_d, then stop.

        # # More advanced: incorporate zc stuff

        # Compute probability/proportion of time a particle is moving P [-] (Equation 6)
        # P represents the fraction of time a grain is in motion at any given location
        # Only compute for cells where shear stress exceeds the critical threshold
        mask = theta_max > theta_cr_exp
        soulsby_P = np.zeros(theta_max.shape)
        soulsby_P[mask] = (1 + ((np.pi / (6 * soulsby_mu_d)) / (theta_max[mask] - theta_cr_exp)) ** 4) ** (-1 / 4)

        # Compute velocity reduction factors R [-]
        # These factors reduce the flow velocity to obtain grain velocities based on transport mode
        # Different reduction factors apply for bed load (Rb) vs suspended load (Rs)
        
        Rb = np.zeros(bed_load_velocity.shape)  # Bed load velocity reduction factor
        Rs = np.zeros(bed_load_velocity.shape)  # Suspended load velocity reduction factor
        soulsby_R = np.zeros(bed_load_velocity.shape)  # Final velocity reduction factor (either Rb or Rs)

        # Compute Rb (Equation 8): bed load velocity reduction factor
        # Only apply where shear stress exceeds critical threshold
        # Limit to maximum of 1 (grain velocity cannot exceed flow velocity)
        mask = theta_max > theta_cr_exp
        Rb[mask] = bed_load_velocity[mask] / flow_velocity_magnitude[mask]
        Rb = np.clip(Rb, 0, 1)

        # Compute Rs (Equation 11): suspended load velocity reduction factor
        # Accounts for vertical distribution of suspended sediment using Rouse number
        # Only compute where Rb is non-zero (i.e., where sediment is mobile)
        # if suspended_lag:
        #     z_s = self.calculate_macdonald_susp_load_height(rouse_number)  # Height of centroid of suspended load
        #     Rs = np.clip(Rs, 0, 1)  # Limit to maximum of 1
        #     Rs = np.nan_to_num(Rs, nan=0.0)  # Handle division by zero or invalid operations
        # else:
        mask = Rb != 0
        Rs[mask] = (
            Rb[mask] * (1 - rouse_number[mask])
            / (8 / 7 - rouse_number[mask])
            * ((8 / 7 * Rb[mask]) ** (8 - 7 * rouse_number[mask]) - 1)
            / ((8 / 7 * Rb[mask]) ** (7 - 7 * rouse_number[mask]) - 1)
        )
        Rs = np.clip(Rs, 0, 1)  # Limit to maximum of 1
        Rs = np.nan_to_num(Rs, nan=0.0)  # Handle division by zero or invalid operations
    
        # Select appropriate reduction factor based on transport mode
        # Rouse number < 2.5 indicates suspended load dominates, otherwise bed load dominates
        suspended_mask = rouse_number < 2.5
        # if suspended_lag:
        # make Rs=array of ones, then ...
        # STUART FIX THIS!!!
        #otherwise use original soulsby logic
        soulsby_R = np.where(suspended_mask, Rs, Rb)

        # Compute grain velocities
        grain_velocity_magnitude = np.multiply(soulsby_P, soulsby_R, flow_velocity_magnitude)  # (Equation 1)
        grain_velocity_x = np.multiply((flow_velocity_x / flow_velocity_magnitude), grain_velocity_magnitude)
        grain_velocity_y = np.multiply((flow_velocity_y / flow_velocity_magnitude), grain_velocity_magnitude)

        # Replace NaN values with zeros (occurs when flow velocity magnitude is zero) and inf to a huge
        grain_velocity_magnitude = np.nan_to_num(grain_velocity_magnitude)
        grain_velocity_x = np.nan_to_num(grain_velocity_x)
        grain_velocity_y = np.nan_to_num(grain_velocity_y)

        # Empty fields for Soulsby (only in van westen)
        mixing_layer_thickness = np.zeros_like(grain_velocity_magnitude)

        # === STORE ===

        print('Adding physics fields to SedtrailsData...')

        # Sediment velocities (vector fields)
        sedtrails_data.add_physics_field(
            'grain_velocity',
            {'x': grain_velocity_x, 'y': grain_velocity_y, 'magnitude': grain_velocity_magnitude},
        )
        # Soulsby et al. (2011) parameters (scalar fields)
        sedtrails_data.add_physics_field('soulsby_a', soulsby_a)
        sedtrails_data.add_physics_field('soulsby_b', soulsby_b)
        sedtrails_data.add_physics_field('mixing_layer_thickness', mixing_layer_thickness)

        # Suspended load settling lag parameters
        if suspended_lag:
            # calculate Shields number for deposition condition
            theta_deposition = physics_lib.compute_shields(
                deposition_bed_shear_stress,
                self.config.gravity,
                self.config.particle_density,
                self.config.water_density,
                self.config.grain_diameter,
                )
            
            z_s = self.calculate_macdonald_susp_load_height(rouse_number, water_depth)
            k_s = self.calculate_macdonald_bed_roughness(theta_max, theta_cr, grain_size, water_depth)
            suspended_load_velocity = self.calculate_macdonald_suspended_load_velocity(max_shear_velocity, z_s, k_s)
            
            # add suspended load velocity field
            sedtrails_data.add_physics_field('suspended_load_velocity', suspended_load_velocity)
            # add mask indicating whether particles can currently be suspended
            sedtrails_data.add_physics_field('suspended_mask', suspended_mask)
            # add mask indicating whether deposition is allowed
            sedtrails_data.add_physics_field('deposition_allowed', theta_max<theta_deposition)

    # JUST ENCODE WHOLE MACDONALD CALCULATION?
    def calculate_macdonald_susp_load_height(self, rouse_number, water_depth):
        """Calculate height of centroid of suspended load using MacDonald et al. (2006) Eq. 27"""
        log_arg = np.log(rouse_number) - 0.4
        tanh_arg = 1.2 * log_arg
        
        # Calculate MacDonald height
        z_s = water_depth * 0.0398 * (10 ** (-1.08 * np.tanh(tanh_arg)))
        return z_s
    
    def calculate_macdonald_bed_roughness(self, theta_max, theta_cr, grain_size, water_depth):
        """Calculate bed roughness using MacDonald et al. (2006) Eq. 12"""
        
        # calculate equilibrium bedform height (eq 12)
        # Initialize eta_b as zeros with same shape as theta_max
        eta_b = np.zeros_like(theta_max)
        
        # Compute theta ratio
        theta_ratio = theta_max / theta_cr
        
        # Apply conditions using np.where for vectorized operations
        # Condition 1: 1 < theta_ratio < 24
        mask_middle = (theta_ratio > 1) & (theta_ratio < 24)
        eta_b[mask_middle] = (
            0.11 * water_depth[mask_middle] 
            * (grain_size / water_depth[mask_middle])**0.3 
            * (1 - np.exp(-0.5 * (theta_ratio[mask_middle] - 1))) 
            * (24 - theta_ratio[mask_middle])
        )
        # Note: eta_b remains 0 for theta_ratio <= 1 or theta_ratio >= 24
        
        # skin friction roughness (eq 13) - NB should be d90 but here we assume uniform grain size
        k_s = 3 * grain_size
        
        # take the greater of bedform height and skin friction roughness
        k_s = np.where(eta_b > k_s, eta_b, k_s)
        
        return k_s
    
    def calculate_macdonald_suspended_load_velocity(self, shear_velocity, z_s, k_s):
        """Calculate suspended load velocity using MacDonald et al. (2006) Eq. 29"""
        suspended_load_velocity = 2.5 * shear_velocity * np.log(30 * z_s / k_s)
        
        return suspended_load_velocity
    