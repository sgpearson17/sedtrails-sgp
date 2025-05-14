"""
Class representing a Fluid physical properties used by the transport converter.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Fluid:
    """
    For example, water and wind.

    Attributes
    ----------
    abs_shear_mean : float
        absolute mean shear stress of the fluid [N/m^2].
    abs_shear_max : float
        absolute maximum shear stress of the fluid [N/m^2].
    velocity_magnitud : float
        total velocity magnitude of the fluid [m/s].
    friction_mean : float
        mean friction velocity of the fluid [m/s].
    friction_max : float
        maximum friction velocity of the fluid [m/s].
    """

    abs_shear_mean: float = None
    abs_shear_max: float = 0.0
    velocity_magnitude: float = 0.0
    friction_mean: float = 0.0
    friction_max: float = 0.0


class FluidPhysics:
    """
    Class for storing physics converted values of a Fluid.
    """

    fluid: Fluid = None

    # concrete method
    def abstau_mean(self, D) -> float:
        """
        Calculates the absolute mean shear stress of the fluid [N/m^2].
        """
        return np.abs(D.tau_mean)

    def abstau_max(self, D) -> float:
        """
        Calculates the absolute maximum shear stress of the fluid [N/m^2].
        """
        return np.abs(D.tau_max)

    def U_mag(self, Ux, Uy) -> float:
        """
        Calculates the total velocity magnitude of the fluid [m/s].
        """
        return np.sqrt(Ux**2 + Uy**2)

    def ustar_mean(self, abstau_mean, S) -> float:
        """
        Calculates the mean friction velocity of the fluid [m/s].
        """
        return np.sqrt(abstau_mean / S.rhoFluid)

    def ustar_max(self, abstau_max, S) -> float:
        """
        Calculates the maximum friction velocity of the fluid [m/s].
        """
        return np.sqrt(abstau_max / S.rhoFluid)
