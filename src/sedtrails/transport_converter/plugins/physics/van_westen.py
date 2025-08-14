"""A plugin for van Westen et al. (2025) sediment transport physics calculations."""

import numpy as np
from sedtrails.transport_converter import physics_lib
from sedtrails.transport_converter.plugins import BasePhysicsPlugin
from sedtrails.transport_converter import SedtrailsData


class PhysicsPlugin(BasePhysicsPlugin):  # all clases should be called the PhysicsPlugin
    """
    Plugin for van Westen et al. (2025) sediment transport physics calculations.
    This plugin implements the physics calculations as described in van Westen et al. (2025).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def add_physics(self, sedtrails_data: SedtrailsData, grain_properties: dict[str, float]) -> None:
        """
        Add physics to SedtrailsData object using van Westen et al. (2025) approach.

        This method follows the workflow from van Westen et al. (2025):
        1. Compute shear velocities and Shields number
        2. Compute bed load velocity (Soulsby equation)
        3. Compute suspended velocity using complex ratio formula
        4. Compute layer thicknesses and mixing layer

        Parameters:
        -----------
        sedtrails_data : SedTRAILSData
            The SedTRAILS data object containing transport data.
        grain_properties : dict[str, float]
            Dictionary containing grain properties such as 'critical_shields' and 'settling_velocity'.

        """
        # print('Using van Westen et al. (2025) to compute transport velocities and add to SedTRAILS data...')

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

        # print(f'Detected {detected_fractions} sediment fraction(s)')

        # Error if more than 1 fraction
        if detected_fractions > 1:
            raise NotImplementedError(
                f'Multiple sediment fractions ({detected_fractions}) are not yet supported. '
                f'The physics converter currently only handles single-fraction sediment transport. '
                f'Please aggregate fractions or implement multi-fraction physics calculations.'
            )

        # For physics calculations, we need to work with squeezed data (no fraction dimension)
        # but we'll add the dimension back to velocities at the end
        if len(bed_load_transport_magnitude.shape) > 2:
            # print('Working with squeezed transport data for physics calculations')
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
        mean_shear_velocity = physics_lib.compute_shear_velocity(mean_bed_shear_stress, self.config.water_density)
        max_shear_velocity = physics_lib.compute_shear_velocity(max_bed_shear_stress, self.config.water_density)

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

        suspended_velocity = physics_lib.compute_suspended_velocity(
            flow_velocity_magnitude,
            bed_load_velocity,
            settling_velocity,
            self.config.von_karman_constant,
            max_shear_velocity,
            shields_number,
            critical_shields,
            method=physics_lib.SuspendedVelocityMethod.SOULSBY_2011,
        )

        # Compute layer thicknesses using squeezed transport data
        bed_load_layer_thickness = physics_lib.compute_transport_layer_thickness(
            bed_load_transport_magnitude_calc, bed_load_velocity, self.config.particle_density, self.config.porosity
        )

        suspended_layer_thickness = physics_lib.compute_transport_layer_thickness(
            suspended_transport_magnitude_calc, suspended_velocity, self.config.particle_density, self.config.porosity
        )

        # Compute directions from magnitudes using squeezed transport data
        suspended_velocity_x, suspended_velocity_y = physics_lib.compute_directions_from_magnitude(
            suspended_velocity,
            suspended_transport_x_calc,
            suspended_transport_y_calc,
            suspended_transport_magnitude_calc,
        )
        bed_load_velocity_x, bed_load_velocity_y = physics_lib.compute_directions_from_magnitude(
            bed_load_velocity, bed_load_transport_x_calc, bed_load_transport_y_calc, bed_load_transport_magnitude_calc
        )

        # Compute mixing layer thickness using Bertin et al. (2008) method

        critical_shear_stress = grain_properties.get('critical_shear_stress')
        if critical_shear_stress is None:
            raise ValueError("Missing required 'critical_shear_stress' value in grain_prperties.")
        mixing_layer_thickness = physics_lib.compute_mixing_layer_thickness(
            max_bed_shear_stress,
            critical_shear_stress,
            method=physics_lib.MixingLayerMethod.BERTIN_2008,
        )

        # === EXPAND DIMENSIONS TO MATCH ORIGINAL STRUCTURE ===
        if has_fraction_dim:
            # print('Expanding velocity dimensions to match transport data structure')
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

        # # If the "reduced velocity" method is used, the velocities are reduced by the trapped/exposed ratios
        # if self.config.trapped_exposed_method == 'reduced_velocity':
        #     bed_load_velocity *= bed_load_exposed_ratio
        #     suspended_velocity *= suspended_exposed_ratio

        #     # Recompute the directions
        #     suspended_velocity_x, suspended_velocity_y = physics_lib.compute_directions_from_magnitude(
        #         suspended_velocity, suspended_transport_x_calc, suspended_transport_y_calc, suspended_transport_magnitude_calc,
        #     )
        #     bed_load_velocity_x, bed_load_velocity_y = physics_lib.compute_directions_from_magnitude(
        #         bed_load_velocity, bed_load_transport_x_calc, bed_load_transport_y_calc, bed_load_transport_magnitude_calc
        #     )

        # print('Adding physics fields to SedtrailsData...')

        # Physics parameters (scalar fields)
        sedtrails_data.add_physics_field('shields_number', shields_number)
        sedtrails_data.add_physics_field('bed_load_layer_thickness', bed_load_layer_thickness)
        sedtrails_data.add_physics_field('suspended_layer_thickness', suspended_layer_thickness)
        sedtrails_data.add_physics_field('mixing_layer_thickness', mixing_layer_thickness)

        # Sediment velocities (vector fields)
        sedtrails_data.add_physics_field(
            'bed_load_velocity', {'x': bed_load_velocity_x, 'y': bed_load_velocity_y, 'magnitude': bed_load_velocity}
        )
        sedtrails_data.add_physics_field(
            'suspended_velocity',
            {'x': suspended_velocity_x, 'y': suspended_velocity_y, 'magnitude': suspended_velocity},
        )
