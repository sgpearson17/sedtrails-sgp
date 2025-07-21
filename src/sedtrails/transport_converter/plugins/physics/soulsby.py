"""A plugin for Soulsby et al. (2011) sediment transport physics calculations."""

import numpy as np
from sedtrails.transport_converter import physics_lib
from sedtrails.transport_converter import SedtrailsData
from sedtrails.transport_converter.plugins import BasePhysicsPlugin


class PhysicsPlugin(BasePhysicsPlugin):  # all clases should be called the PhysicsPlugin
    """
    Plugin for Soulsby et al. (2011) sediment transport physics calculations.
    This plugin implements the physics calculations as described in Soulsby et al. (2011).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def add_physics(self, sedtrails_data: SedtrailsData, grain_properties: dict[str, float]) -> None:
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

        # Extract shear stresses

        mean_bed_shear_stress = sedtrails_data.mean_bed_shear_stress

        max_bed_shear_stress = sedtrails_data.max_bed_shear_stress

        critical_shear_stress = grain_properties.get('critical_shear_stress')
        if critical_shear_stress is None:
            raise ValueError("Missing required 'critical_shear_stress' value in grain_prperties.")

        # Extract flow velocities (we should be able to change these based on configuration)
        flow_velocity_x = sedtrails_data.depth_avg_flow_velocity['x']
        flow_velocity_y = sedtrails_data.depth_avg_flow_velocity['y']
        flow_velocity_magnitude = sedtrails_data.depth_avg_flow_velocity['magnitude']


        # === COMPUTE ===

        # Compute shear velocities

        mean_shear_velocity = physics_lib.compute_shear_velocity(
            mean_bed_shear_stress,
            self.config.water_density,
        )

        max_shear_velocity = physics_lib.compute_shear_velocity(
            max_bed_shear_stress,
            self.config.water_density,
        )

        # Compute Shields number
        shields_number = physics_lib.compute_shields(
            max_bed_shear_stress,
            self.config.gravity,
            self.config.particle_density,
            self.config.water_density,
            self.config.grain_diameter,
        )

        # Compute transport velocities (these will have shape [time, spatial])

        critical_shields = grain_properties.get('critical_shields')
        if critical_shields is None:
            raise ValueError("Missing required 'critical_shields' value in grain_prperties.")
       
        bed_load_velocity = physics_lib.compute_bed_load_velocity(shields_number, critical_shields, mean_shear_velocity)
       
        settling_velocity = grain_properties.get('settling_velocity')
        if settling_velocity is None:
            raise ValueError("Missing required 'settling_velocity' value in grain_prperties.")
        
        # Compute Soulsby et al. (2011) physical parameters

        gravity = self.config.gravity
        particle_density = self.config.particle_density
        water_density = self.config.water_density
        kinematic_viscosity = self.config.kinematic_viscosity
        grain_diameter = self.config.grain_diameter #do we already differentiate between background and tracer?
        von_karman_constant = self.config.von_karman_constant

        Dstar_Tracer = (gravity * (particle_density / water_density - 1) / (kinematic_viscosity**2)) ** (1 / 3)\
              * grain_diameter
        
        Dstar_Background = (gravity * (particle_density / water_density - 1) / (kinematic_viscosity**2)) ** (1 / 3)\
              * grain_diameter
        
        ratio = grain_diameter / grain_diameter

        theta_max_Background = max_bed_shear_stress / (gravity * (particle_density - water_density) * grain_diameter)

        theta_cr_Background = 0.3 / (1 + 1.2 * Dstar_Background) + 0.055 * (1 - np.exp(-0.020 * Dstar_Background))

        theta_max_Tracer = max_bed_shear_stress / (gravity * (particle_density - water_density) * grain_diameter)

        theta_cr_Tracer = 0.3 / (1 + 1.2 * Dstar_Tracer) + 0.055 * (1 - np.exp(-0.020 * Dstar_Tracer))

        theta_cr_Tracer_exp = (theta_cr_Tracer * np.sqrt(8 / (3 * (ratio**2) + 6 * ratio - 1)) * ((3.2260 * ratio)
                / (4 * ratio - 2 * (ratio + 1 - np.sqrt(ratio**2 + 2 * ratio - 1 / 3)))))
        
        rouse = settling_velocity / (von_karman_constant * max_shear_velocity)


        # Define Soulsby et al. (2011) empirical parameters (these should be defined in the configuration file)
        soulsby2011_b_e = 1.7e-7 # maximum free-to-trapped transition probability per second [1/s] default = 1.7e-7
        soulsby2011_theta_s = 0.1 # transition scale value [-] default = 0.1
        soulsby2011_gamma_e = 0.1 # long-term equilibrium proportion of free particles [-] default = 0.1
        soulsby2011_mu_d    = 0.5 # dynamic friction coefficient (0.5-1.0) [-] default = 0.5


        # Compute the transition probability b [-]
        b = np.zeros(theta_max_Tracer.shape)
        for i in range(0,theta_max_Tracer.shape[0]):
            for j in range(0,theta_max_Tracer.shape[1]):
                if theta_max_Background[i][j] > theta_cr_Background:
                    b[i][j] = soulsby2011_b_e * (1 - np.exp(\
                        -(theta_max_Background[i][j] - theta_cr_Background)\
                            / soulsby2011_theta_s))
                    else:
                    b[i][j] = 0
        

        # Compute the transition probability a [-]
        a = soulsby2011_gamma_e * b / (1 - soulsby2011_gamma_e)
    

        # Compute probability/proportion of time a particle is moving P [-]
        P = np.zeros(theta_max_Tracer.shape)
        for i in range(0,theta_max_Tracer.shape[0]):
            for j in range(0,theta_max_Tracer.shape[1]):
                if theta_max_Tracer[i][j] > theta_cr_Tracer_exp:
                    P[i][j] = (1 + ((np.pi / (6 * soulsby2011_mu_d))\
                                     / (theta_max_Tracer[i][j] - theta_cr_Tracer_exp))\
                                          ** 4) ** (-1/4)
                    else:
                    P[i][j] = 0

        # Compute a reduction factor which is applied on the flow velocities to obtain grain velocities

        Rb = np.zeros(bed_load_velocity.shape) # Bed load velocity reduction factor
        Rs = np.zeros(bed_load_velocity.shape) # Suspended load velocity reduction factor
        R = np.zeros(bed_load_velocity.shape) # Final velocity reduction factor
        
        for i in range(0, theta_max_Tracer.shape[0]):
            for j in range(0, theta_max_Tracer.shape[1]):
                if theta_max_Tracer[i][j] > theta_cr_Tracer_exp:
                    print(bed_load_velocity[i][j] / flow_velocity_magnitude[i][j])
                    Rb[i][j] = bed_load_velocity[i][j] / flow_velocity_magnitude[i][j]
                    if Rb[i][j] > 1:
                        Rb[i][j] = 1  # apply velocity limiter (grain velocity cannot exceed flow velocity)
                        else:
                        Rb[i][j] = 0
                        
        for i in range(0, theta_max_Tracer.shape[0]):
            for j in range(0, theta_max_Tracer.shape[1]):
                if Rb[i][j] == 0:
                    Rs[i][j] = 0
                    else:
                    # didn't find a good way of presenting this huge equation readable in python!!!!
                    Rs[i][j] = np.multiply(
                        np.divide(np.multiply(Rb[i][j], (1 - rouse[i][j])), (8 / 7 - rouse[i][j])),
                        np.divide(
                            (np.power((8 / 7 * Rb[i][j]), (8 - 7 * rouse[i][j])) - 1),
                            (np.power((8 / 7 * Rb[i][j]), (7 - 7 * rouse[i][j])) - 1),
                            ),
                     )
                    if Rs[i][j] > 1:
                        Rs[i][j] = 1 # Apply velocity limiter (grain velocity cannot exceed flow velocity)
                        elif np.isnan(Rs[i][j]):  # any idea why this is not working?
                        Rs[i][j] = 0
                        
        for i in range(0, theta_max_Tracer.shape[0]):
            for j in range(0, theta_max_Tracer.shape[1]):
                if (rouse[i][j] < 2.5):  # if all material is in suspension use Rs (for suspended load)
                    R[i][j] = Rs[i][j]
                    else:  # otherwise use Rb (for bed load)
                    R[i][j] = Rb[i][j]

        # Compute grain velocities
        grain_velocity_magnitude = np.multiply(P,R,flow_velocity_magnitude)
        grain_velocity_x = np.multiply((flow_velocity_x/flow_velocity_magnitude),grain_velocity_magnitude)
        grain_velocity_y = np.multiply((flow_velocity_y/flow_velocity_magnitude),grain_velocity_magnitude)


        # === STORE ===

        print('Adding physics fields to SedtrailsData...')

        # Sediment velocities (vector fields)
        sedtrails_data.add_physics_field(
            'bed_load_velocity', {'x': grain_velocity_x, 'y': grain_velocity_y, 'magnitude': grain_velocity_magnitude}
        )
