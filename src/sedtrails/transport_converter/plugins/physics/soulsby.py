"""A plugin for Soulsby et al. (2011) sediment transport physics calculations."""

import numpy as np

from sedtrails.transport_converter import SedtrailsData, physics_lib
from sedtrails.transport_converter.plugins import BasePhysicsPlugin


class PhysicsPlugin(BasePhysicsPlugin):  # all clases should be called the PhysicsPlugin
    """
    Plugin for Soulsby et al. (2011) sediment transport physics calculations.
    This plugin implements the physics calculations as described in Soulsby et al. (2011).
    """

    def __init__(self, config, tracer_config):
        super().__init__()
        self.config = config

    def add_physics(
        self, sedtrails_data: SedtrailsData, grain_properties: dict[str, float], transport_probability_method: str
    ) -> None:
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
            raise ValueError(
                'critical_shields is required for Soulsby physics calculations but was not found in grain_properties'
            )
        if dimensionless_grain_size is None:
            raise ValueError(
                'dimensionless_grain_size is required for Soulsby physics calculations but was not found in grain_properties'
            )
        if settling_velocity is None:
            raise ValueError(
                'settling_velocity is required for Soulsby physics calculations but was not found in grain_properties'
            )

        # Extract flow velocities (we should be able to change these based on configuration)
        flow_velocity_x = sedtrails_data.depth_avg_flow_velocity['x']
        flow_velocity_y = sedtrails_data.depth_avg_flow_velocity['y']
        flow_velocity_magnitude = sedtrails_data.depth_avg_flow_velocity['magnitude']

        # Extract Soulsby et al. (2011) empirical parameters
        soulsby_b_e = self.config.soulsby_b_e
        soulsby_theta_s = self.config.soulsby_theta_s
        soulsby_gamma_e = self.config.soulsby_gamma_e
        soulsby_mu_d = self.config.soulsby_mu_d

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

        soulsby_b = np.zeros_like(theta_max, dtype=float)

        if transport_probability_method != 'no_probability':
            mask_b = background_theta_max > background_theta_cr
            delta_b = background_theta_max - background_theta_cr  # safe even if negative; mask controls use
            # b = b_e * (1 - exp(-(theta - theta_cr)/theta_s)) for theta > theta_cr else 0
            # (Assumes soulsby_theta_s > 0)
            soulsby_b = np.where(
                mask_b,
                soulsby_b_e * (1.0 - np.exp(-delta_b / soulsby_theta_s)),
                0.0,
            )

        # a (Eq. 5)  -- scalar denominator guard
        den_gamma = 1.0 - soulsby_gamma_e
        soulsby_a = np.divide(
            soulsby_gamma_e * soulsby_b,
            den_gamma,
            out=np.zeros_like(soulsby_b),
            where=(den_gamma != 0.0),
        )

        delta_theta = theta_max - theta_cr_exp
        mask_P = delta_theta > 0.0

        const_P = np.pi / (6.0 * soulsby_mu_d)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            term = (const_P / delta_theta) ** 4
            soulsby_P = np.where(mask_P, (1.0 + term) ** (-0.25), 0.0)

        # Clean any numerical junk
        soulsby_P = np.nan_to_num(soulsby_P, nan=0.0, posinf=0.0, neginf=0.0)

        Rb = np.zeros_like(bed_load_velocity, dtype=float)

        # Only compute where theta > theta_cr_exp AND flow_velocity_magnitude > 0 (improvement #1)
        mask_Rb = (theta_max > theta_cr_exp) & (flow_velocity_magnitude > 0.0)

        np.divide(
            bed_load_velocity,
            flow_velocity_magnitude,
            out=Rb,
            where=mask_Rb,
        )

        # Apply limiter and NaN handling (improvement #2)
        Rb = np.nan_to_num(Rb, nan=0.0, posinf=0.0, neginf=0.0)
        Rb = np.clip(Rb, 0.0, 1.0)

        Rs = np.zeros_like(Rb, dtype=float)

        rouse = rouse_number.astype(float)
        base = (8.0 / 7.0) * Rb
        denA = (8.0 / 7.0) - rouse

        exp1 = 8.0 - 7.0 * rouse
        exp2 = 7.0 - 7.0 * rouse

        eps = 1e-12  # numerical safety threshold

        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            pow1 = np.power(base, exp1)
            pow2 = np.power(base, exp2)
            denPow = pow2 - 1.0

            Rs_calc = Rb * (1.0 - rouse) / denA * (pow1 - 1.0) / denPow

        valid_Rs = (
            (Rb > 0.0)
            & np.isfinite(rouse)
            & np.isfinite(Rb)
            & (np.abs(denA) > eps)  # avoids 8/7 - rouse == 0 singularity
            & (np.abs(denPow) > eps)  # avoids pow2 - 1 == 0 singularity (notably around rouse==1)
        )

        Rs = np.where(valid_Rs, Rs_calc, 0.0)
        Rs = np.nan_to_num(Rs, nan=0.0, posinf=0.0, neginf=0.0)
        Rs = np.clip(Rs, 0.0, 1.0)

        soulsby_R = np.where(rouse < 2.5, Rs, Rb)

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
