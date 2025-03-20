from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


# Abstract class
class Medium(ABC):
    """
    For example, water and wind.
    """

    @abstractmethod
    def ustar_mean(self) -> float:
        """
        Calculate the mean friction velocity [m/s] of the medium.
        """
        pass

    def ustar_max(self) -> float:
        """
        Calculate the maximum friction velocity [m/s] of the medium.
        """
        pass

    def ustar_cr(self) -> float:
        """
        Calculate the critical friction velocity [m/s] in the medium.
        """
        pass

    def theta_max_t(self) -> float:
        """
        Calculate the maximum Shields number for the tracer particle [-] in the medium.
        """
        pass

    def theta_max_b(self) -> float:
        """
        Calculate the maximum Shields number for the background particle [-] in the medium.
        """
        pass

    def rouse(self) -> float:
        """
        Calculate the Rouse number [-] in the medium.
        """
        pass

    def U_bed(self) -> float:
        """
        Calculate the bedload velocity [m/s] in the medium.
        """
        pass

    # concrete method
    def abstau_mean(self, D) -> float:
        """
        Calculate the absolute mean shear stress [N/m^2] of the medium.
        """
        return np.abs(D.tau_mean)
    
    def abstau_max(self, D) -> float:
        """
        Calculate the absolute maximum shear stress [N/m^2] of the medium.
        """

        return np.abs(D.tau_max)
    
    def U_mag(self, Ux, Uy) -> float:
        """
        Calculate the total velocity magnitude [m/s] of the medium.
        """
        return np.sqrt(Ux ** 2 + Uy ** 2)


class Water(Medium):
    """
    Class representing water properties.
    """
    def ustar_mean(self, abstau_mean, S) -> float:
        """
        Calculate the mean friction velocity [m/s] of water.
        """
        return np.sqrt(abstau_mean / S.rhoWater)

    def ustar_max(self, abstau_max, S) -> float:
        """
        Calculate the maximum friction velocity [m/s] of water.
        """
        return np.sqrt(abstau_max / S.rhoWater)

    def ustar_cr(self, S) -> float:
        """
        Calculate the critical friction velocity [m/s] in water.
        """
        return np.sqrt(S.tau_cr / S.rhoW);

    def theta_max_t(self, abstau_max, S) -> float:
        """
        Calculate the maximum Shields number for the tracer particle [-] in water.
        """
        return abstau_max / (S.g * (S.rhoS - S.rhoWater) * S.dTracer)

    def theta_max_b(self, abstau_max, S) -> float:
        """
        Calculate the maximum Shields number for the background particle [-] in water.
        """
        return abstau_max / (S.g * (S.rhoS - S.rhoWater) * S.dBackground)

    def rouse(self, w_s_t, ustar_max) -> float:
        """
        Calculate the Rouse number [-] in water.
        """
        return w_s_t / (0.4 * ustar_max)

    def U_bed(self, ustar_mean, theta_cr_t_exp, theta_max_t) -> float:
        """
        Calculate the bedload velocity [m/s] in water.
        """
        return 10 * ustar_mean * (1 - 0.7 * np.sqrt(theta_cr_t_exp / theta_max_t))
    

class Wind(Medium):
    """
    Class representing wind properties.
    """
    def ustar_mean(self, abstau_mean, S) -> float:
        """
        Calculate the mean friction velocity [m/s] of wind.
        """
        return np.sqrt(abstau_mean / S.rhoWind)

    def ustar_max(self, abstau_max, S) -> float:
        """
        Calculate the maximum friction velocity [m/s] of wind.
        """
        return np.sqrt(abstau_max / S.rhoWind)

    def ustar_cr(self, abstau_max, S) -> float:
        """
        Calculate the critical friction velocity [m/s] in wind.
        """
        return abstau_max / (S.g * (S.rhoS - S.rhoWind) * S.dTracer)

    def theta_max_t(self, abstau_max, S) -> float:
        """
        Calculate the maximum Shields number for the tracer particle [-] in wind.
        """
        return abstau_max / (S.g * (S.rhoS - S.rhoWind) * S.dTracer)

    def theta_max_b(self, abstau_max, S) -> float:
        """
        Calculate the maximum Shields number for the background particle [-] in wind.
        """
        return abstau_max / (S.g * (S.rhoS - S.rhoWind) * S.dBackground)

    def rouse(self, w_s_t, ustar_max) -> float:
        """
        Calculate the Rouse number [-] in wind.
        """
        return w_s_t / (0.4 * ustar_max)

    def U_bed(self, ustar_mean, theta_cr_t_exp, theta_max_t) -> float:
        """
        Calculate the bedload velocity [m/s] in wind.
        """
        return 10 * ustar_mean * (1 - 0.7 * np.sqrt(theta_cr_t_exp / theta_max_t))
