"""A plugin for Soulsby et al. (2011) sediment transport physics calculations."""

import logging
from math import isfinite
from numbers import Real

import numpy as np
from sedtrails.transport_converter import physics_lib
from sedtrails.transport_converter import SedtrailsData
from sedtrails.transport_converter.plugins import BasePhysicsPlugin

logger = logging.getLogger(__name__)


def _safe_velocity_direction(
    flow_velocity_x: np.ndarray,
    flow_velocity_y: np.ndarray,
    flow_velocity_magnitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flow unit vectors, using zero direction where speed is invalid."""
    valid_speed = np.isfinite(flow_velocity_magnitude) & (flow_velocity_magnitude > 0.0)
    direction_x = np.zeros_like(flow_velocity_magnitude, dtype=float)
    direction_y = np.zeros_like(flow_velocity_magnitude, dtype=float)

    np.divide(flow_velocity_x, flow_velocity_magnitude, out=direction_x, where=valid_speed)
    np.divide(flow_velocity_y, flow_velocity_magnitude, out=direction_y, where=valid_speed)

    return direction_x, direction_y


class PhysicsPlugin(BasePhysicsPlugin):  # all clases should be called the PhysicsPlugin
    """
    Plugin for Soulsby et al. (2011) sediment transport physics calculations.
    This plugin implements the physics calculations as described in Soulsby et al. (2011).
    """

    def __init__(self, config, tracer_config):
        """Initialize the Soulsby physics plugin.

        Parameters
        ----------
        config : PhysicsConfig
            Physics configuration supplied by the converter.
        tracer_config : dict
            Method-specific tracer configuration retained for interface
            compatibility.
        """
        super().__init__()
        self.config = config
        if type(config.include_nonlinear_wave_velocity) is not bool:
            raise ValueError('include_nonlinear_wave_velocity must be true or false.')
        if (
            not isinstance(config.facua, Real)
            or isinstance(config.facua, bool)
            or not isfinite(config.facua)
            or config.facua < 0
        ):
            raise ValueError('facua must be a finite, non-negative number.')

    def add_physics(self, sedtrails_data: SedtrailsData, grain_properties: dict[str, float], transport_probability_method: str) -> None:
        """
        Add physics using Soulsby et al. (2011) approach.
                '1. Focus on individual particle tracking velocities
        '
                '2. Different approach to settling and resuspension
        '
                '3. Particle-specific rather than layer-based calculations
        '
                'See: Soulsby, R. L., et al. (2011). Lagrangian model for simulating '
                'the dispersal of sand-sized particles in coastal waters.'

        Parameters
        ----------
        sedtrails_data : SedtrailsData
            SedTRAILS data object to process.
        grain_properties : dict[str, float]
            Precomputed grain-property values.
        transport_probability_method : str
            Transport probability method name.
        """
        logger.info('Using Soulsby et al. (2011) to compute transport velocities and add to SedTRAILS data')

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
        if self.config.include_nonlinear_wave_velocity:
            wave_velocity = sedtrails_data.nonlinear_wave_velocity
            if wave_velocity is None or any(key not in wave_velocity for key in ('x', 'y')):
                raise ValueError('Soulsby wave coupling requires nonlinear_wave_velocity x/y components.')
            wave_x = np.asarray(wave_velocity['x'], dtype=float)
            wave_y = np.asarray(wave_velocity['y'], dtype=float)
            if wave_x.shape != flow_velocity_x.shape or wave_y.shape != flow_velocity_y.shape:
                raise ValueError('Nonlinear wave velocity components must match the flow velocity shape.')
            flow_velocity_x = flow_velocity_x + self.config.facua * wave_x
            flow_velocity_y = flow_velocity_y + self.config.facua * wave_y
            if not np.isfinite(flow_velocity_x).all() or not np.isfinite(flow_velocity_y).all():
                raise ValueError('Soulsby effective flow velocity must contain finite x/y components.')
            flow_velocity_magnitude = np.hypot(flow_velocity_x, flow_velocity_y)

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
        rouse_denominator = von_karman * max_shear_velocity
        rouse_number = np.full_like(max_shear_velocity, np.inf, dtype=float)
        np.divide(
            settling_velocity,
            rouse_denominator,
            out=rouse_number,
            where=np.isfinite(rouse_denominator) & (rouse_denominator > 0.0),
        )

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
        soulsby_b = np.zeros_like(theta_max, dtype=float)
        if transport_probability_method != 'no_probability':
            above_background_threshold = background_theta_max > background_theta_cr
            excess_background_shields = background_theta_max[above_background_threshold] - background_theta_cr
            soulsby_b[above_background_threshold] = soulsby_b_e * (
                1 - np.exp(-excess_background_shields / soulsby_theta_s)
            )

        # Compute the transition probability a [-] (Equation 5)
        soulsby_a = soulsby_gamma_e * soulsby_b / (1 - soulsby_gamma_e)
        # Compute probability/proportion of time a particle is moving P [-] (Equation 6)
        soulsby_P = np.zeros_like(theta_max, dtype=float)
        above_motion_threshold = theta_max > theta_cr_exp
        excess_tracer_shields = theta_max[above_motion_threshold] - theta_cr_exp
        soulsby_P[above_motion_threshold] = (
            1 + ((np.pi / (6 * soulsby_mu_d)) / excess_tracer_shields) ** 4
        ) ** (-1 / 4)

        # Compute a reduction factor which is applied on the flow velocities to obtain grain velocities

        Rb = np.zeros(bed_load_velocity.shape)  # Bed load velocity reduction factor
        Rs = np.zeros(bed_load_velocity.shape)  # Suspended load velocity reduction factor
        soulsby_R = np.zeros(bed_load_velocity.shape)  # Final velocity reduction factor

        # Compute Rb (Equation 8)
        valid_bed_flow = (
            above_motion_threshold
            & np.isfinite(flow_velocity_magnitude)
            & (flow_velocity_magnitude > 0.0)
        )
        np.divide(bed_load_velocity, flow_velocity_magnitude, out=Rb, where=valid_bed_flow)
        Rb[Rb > 1] = 1  # apply velocity limiter (grain velocity cannot exceed flow velocity)
        Rb[~np.isfinite(Rb)] = 0

        # Compute Rs (Equation 11)
        positive_bed_reduction = Rb != 0
        rb = Rb[positive_bed_reduction]
        local_rouse_number = rouse_number[positive_bed_reduction]
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            relative_velocity = (8 / 7 * rb) ** (8 - 7 * local_rouse_number)
            relative_velocity_lower = (8 / 7 * rb) ** (7 - 7 * local_rouse_number)
            suspended_reduction = (
                rb
                * (1 - local_rouse_number)
                / (8 / 7 - local_rouse_number)
                * (relative_velocity - 1)
                / (relative_velocity_lower - 1)
            )
        suspended_reduction[suspended_reduction > 1] = 1  # grain velocity cannot exceed flow velocity
        suspended_reduction[np.isnan(suspended_reduction)] = 0
        Rs[positive_bed_reduction] = suspended_reduction

        # Compute R
        use_suspended_reduction = rouse_number < 2.5
        soulsby_R[use_suspended_reduction] = Rs[use_suspended_reduction]
        soulsby_R[~use_suspended_reduction] = Rb[~use_suspended_reduction]

        # Compute grain velocities
        grain_velocity_magnitude = soulsby_P * soulsby_R * flow_velocity_magnitude  # (Equation 1)
        flow_direction_x, flow_direction_y = _safe_velocity_direction(
            flow_velocity_x,
            flow_velocity_y,
            flow_velocity_magnitude,
        )
        grain_velocity_x = flow_direction_x * grain_velocity_magnitude
        grain_velocity_y = flow_direction_y * grain_velocity_magnitude

        # Replace non-finite values with zero transport.
        grain_velocity_magnitude = np.nan_to_num(grain_velocity_magnitude, nan=0.0, posinf=0.0, neginf=0.0)
        grain_velocity_x = np.nan_to_num(grain_velocity_x, nan=0.0, posinf=0.0, neginf=0.0)
        grain_velocity_y = np.nan_to_num(grain_velocity_y, nan=0.0, posinf=0.0, neginf=0.0)

        # Empty fields for Soulsby (only in van westen)
        mixing_layer_thickness = np.zeros_like(grain_velocity_magnitude)

        # === STORE ===

        logger.info('Adding physics fields to SedtrailsData')

        # Sediment velocities (vector fields)
        sedtrails_data.add_physics_field(
            'grain_velocity',
            {'x': grain_velocity_x, 'y': grain_velocity_y, 'magnitude': grain_velocity_magnitude},
        )
        # Soulsby et al. (2011) parameters (scalar fields)
        sedtrails_data.add_physics_field('soulsby_a', soulsby_a)
        sedtrails_data.add_physics_field('soulsby_b', soulsby_b)
        sedtrails_data.add_physics_field('mixing_layer_thickness', mixing_layer_thickness)
