from dataclasses import dataclass
import numpy as np


@dataclass
class Fluid:
    """
    For example, water and wind.
    """
    
    # concrete method
    def abstau_mean(self, D) -> float:
        """
        Calculate the absolute mean shear stress [N/m^2] of the fluid.
        """
        return np.abs(D.tau_mean)
    
    def abstau_max(self, D) -> float:
        """
        Calculate the absolute maximum shear stress [N/m^2] of the fluid.
        """
        return np.abs(D.tau_max)
    
    def U_mag(self, Ux, Uy) -> float:
        """
        Calculate the total velocity magnitude [m/s] of the fluid.
        """
        return np.sqrt(Ux ** 2 + Uy ** 2)
    
    def ustar_mean(self, abstau_mean, S) -> float:
        """
        Calculate the mean friction velocity [m/s] of the fluid.
        """
        return np.sqrt(abstau_mean / S.rhoFluid)

    def ustar_max(self, abstau_max, S) -> float:
        """
        Calculate the maximum friction velocity [m/s] of the fluid.
        """
        return np.sqrt(abstau_max / S.rhoFluid)
