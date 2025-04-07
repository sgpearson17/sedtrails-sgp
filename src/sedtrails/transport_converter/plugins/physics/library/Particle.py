"""
Classes representing particle physical properties used by the transport converter.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


# Abstract class
class Particle(ABC):
    """
    Class representing particle physical properties used by the transport converter
    for all particles (for example, passive, sand and mud).
    
    Attributes
    ----------
    Dstar_Tracer : float
        dimentionless diameter of the tracer sediment particle [-].
    Dstar_Background : float
        dimentionless diameter of the background sediment particle [-].
    ratio : float
        ratio of tracer sediment particle diameter to background sediment particle diameter [-].
    """

# I am not quite sure if we should include a class for "Passive(Particle)" here
# because it does not have any physical properties.

    # concrete method
    def Dstar_Tracer(self, S) -> float:
        """
        Calculates the dimentionless diameter of the tracer sediment particle [-].
        """
        return (S.g * (S.rhoParticle / S.rhoFluid - 1) / (S.visc_kin ** 2)) ** (1/3) * S.dTracer

    def Dstar_Background(self, S) -> float:
        """
        Calculates the dimentionless diameter of the background sediment particle [-].
        """
        return (S.g * (S.rhoParticle / S.rhoFluid - 1) / (S.visc_kin ** 2)) ** (1/3) * S.dBackground

    def ratio(self, S) -> float:
        """
        Calculates the ratio of tracer sediment particle diameter to background sediment particle diameter [-].
        """
        return S.dTracer / S.dBackground


@dataclass
class Sand(Particle):
    """
    Class representing particle physical properties used by the transport converter for sand.
    
    Attributes
    ----------
    w_s_t : float
        settling velocity of the tracer sand particle [m/s].
    theta_cr_Tracer : float
        critical Shields number for the tracer sand particle [-].
    theta_cr_Background : float
        critical Shields number for the background sand particle [-].
    theta_cr_Tracer_exp : float
        adjusted critical Shields number for the tracer sand particle to
        take into account hiding/exposure when having a diameter different
        than the background sand particle [-].
    ustar_cr : float
        critical friction velocity for sand particle erosion [m/s].
    theta_max_Tracer : float
        maximum Shields number for the tracer sand particle [-].
    theta_max_Background
        maximum Shields number for the background sand particle [-].
    rouse : float
        Rouse number for the tracer sand particle [-].
    U_bed : float
        bedload velocity of the tracer sand particle [m/s].
    """

    def w_s_t(self, Dstar_Tracer, S) -> float:
        """
        Calculates the settling velocity of the tracer sand particle [m/s].
        """
        return (S.visc_kin / S.dTracer) * (np.sqrt(10.36 ** 2 + 1.049 * (Dstar_Tracer ** 3)) - 10.36)

    def theta_cr_Tracer(self, Dstar_Tracer) -> float:
        """
        Calculates the critical Shields number for the tracer sand particle [-].
        """
        return 0.3 / (1 + 1.2 * Dstar_Tracer) + 0.055 * (1 - np.exp(-0.020 * Dstar_Tracer))

    def theta_cr_Background(self, Dstar_Background) -> float:
        """
        Calculates the critical Shields number for the background sand particle [-].

        
        """
        return 0.3 / (1 + 1.2 * Dstar_Background) + 0.055 * (1 - np.exp(-0.020 * Dstar_Background))

    def theta_cr_Tracer_exp(self, theta_cr_Tracer, ratio) -> float:
        """
        Calculates the adsjusted critical Shields number for the tracer sand particle to
        take into account hiding/exposure when having a diameter different than the background
        sand particles [-].
        """
        return theta_cr_Tracer * np.sqrt(8 / (3 * (ratio ** 2) + 6 * ratio - 1)) * ((3.2260 * ratio) /
                    (4 * ratio - 2 * (ratio + 1 - np.sqrt(ratio ** 2 + 2 * ratio - 1/3))))

    def ustar_cr(self, S) -> float:
        """
        Calculates the critical friction velocity for sand particle erosion [m/s].
        """
        return np.sqrt(S.tau_cr / S.rhoFluid)

    def theta_max_Tracer(self, abstau_max, S) -> float:
        """
        Calculates the maximum Shields number for the tracer sand particle [-].
        """
        return abstau_max / (S.g * (S.rhoParticle - S.rhoFluid) * S.dTracer)

    def theta_max_Background(self, abstau_max, S) -> float:
        """
        Calculates the maximum Shields number for the background sand particle [-].
        """
        return abstau_max / (S.g * (S.rhoParticle - S.rhoFluid) * S.dBackground)

    def rouse(self, w_s_t, ustar_max) -> float:
        """
        Calculates the Rouse number for the tracer sand particle [-].
        """
        return w_s_t / (0.4 * ustar_max)

    def U_bed(self, ustar_mean, theta_cr_Tracer_exp, theta_max_Tracer) -> float:
        """
        Calculates the bedload velocity of the tracer sand particle [m/s].
        """
        return 10 * ustar_mean * (1 - 0.7 * np.sqrt(theta_cr_Tracer_exp / theta_max_Tracer))


@dataclass
class Mud(Particle):
    """
    Placeholder for class representing particle physical properties used by the transport converter for mud.
    """
