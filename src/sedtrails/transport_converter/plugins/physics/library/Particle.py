from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


# Abstract class
class Particle(ABC):
    """
    For example, passive, sediment and biological.
    """

    @abstractmethod
    def Dstar_Tracer(self) -> float:
        """
        Calculate the diamentionless diameter [m] of the tracer particle.
        """
        pass

    def Dstar_Background(self) -> float:
        """
        Calculate the diamentionless diameter [m] of the background particle.
        """
        pass

    def w_s_t(self) -> float:
        """
        Calculate the settling velocity [m/s] of the tracer particle.
        """
        pass

    def theta_cr_Tracer(self) -> float:
        """
        Calculate the critical Shields number [-] for the tracer particle.
        """
        pass

    def theta_cr_Background(self) -> float:
        """
        Calculate the critical Shields number [-] for the background particle.
        """
        pass

    def ratio(self) -> float:
        """
        Calculate the ratio of tracer particle diameter [-] to background particle diameter.
        """
        pass

    def theta_cr_Tracer_exp(self) -> float:
        """
        Calculate the adsjusted critical Shields number [-] for the tracer particle to
        take into account hiding/exposure when having a diameter different than the background particles.
        """
        pass

    def ustar_cr(self) -> float:
        """
        Calculate the critical friction velocity [m/s] for particle erosion in the fluid.
        """
        pass

    def theta_max_Tracer(self) -> float:
        """
        Calculate the maximum Shields number [-] for the tracer particle in the fluid.
        """
        pass

    def theta_max_Background(self) -> float:
        """
        Calculate the maximum Shields number [-] for the background particle in the fluid.
        """
        pass

    def rouse(self) -> float:
        """
        Calculate the Rouse number [-] for the tracer particle in the fluid.
        """
        pass

    def U_bed(self) -> float:
        """
        Calculate the bedload velocity [m/s] of the tracer particle in the fluid.
        """
        pass


class Sediment(Particle):
    """
    Class representing sediment particles properties.
    """

    # concrete method
    def Dstar_Tracer(self, S) -> float:
        """
        Calculate the diamentionless diameter [m] of the tracer sediment particle.
        """
        return (S.g * (S.rhoParticle / S.rhoFluid - 1) / (S.visc_kin ** 2)) ** (1/3) * S.dTracer

    def Dstar_Background(self, S) -> float:
        """
        Calculate the diamentionless diameter [m] of the background sediment particle.
        """
        return (S.g * (S.rhoParticle / S.rhoFluid - 1) / (S.visc_kin ** 2)) ** (1/3) * S.dBackground

    def ratio(self, S) -> float:
        """
        Calculate the ratio of tracer sediment particle diameter [-] to background sediment particle diameter.
        """
        return S.dTracer / S.dBackground

class Sand(Sediment(Particle)):
    """
    Class representing sand particles properties.
    """

    def w_s_t(self, Dstar_Tracer, S) -> float:
        """
        Calculate the settling velocity [m/s] of the tracer sand particle.
        """
        return (S.visc_kin / S.dTracer) * (np.sqrt(10.36 ** 2 + 1.049 * (Dstar_Tracer ** 3)) - 10.36)

    def theta_cr_Tracer(self, Dstar_Tracer) -> float:
        """
        Calculate the critical Shields number [-] for the tracer sand particle.
        """
        return 0.3 / (1 + 1.2 * Dstar_Tracer) + 0.055 * (1 - np.exp(-0.020 * Dstar_Tracer))

    def theta_cr_Background(self, Dstar_Background) -> float:
        """
        Calculate the critical Shields number [-] for the background sand particle.
        """
        return 0.3 / (1 + 1.2 * Dstar_Background) + 0.055 * (1 - np.exp(-0.020 * Dstar_Background))

    def theta_cr_Tracer_exp(self, theta_cr_Tracer, ratio) -> float:
        """
        Calculate the adsjusted critical Shields number [-] for the tracer sand particle to
        take into account hiding/exposure when having a diameter different than the background sand particles.
        """
        return theta_cr_Tracer * np.sqrt(8 / (3 * (ratio ** 2) + 6 * ratio - 1)) * ((3.2260 * ratio) /\
                    (4 * ratio - 2 * (ratio + 1 - np.sqrt(ratio ** 2 + 2 * ratio - 1/3))))

    def ustar_cr(self, S) -> float:
        """
        Calculate the critical friction velocity [m/s] for sand particle erosion in the fluid.
        """
        return np.sqrt(S.tau_cr / S.rhoFluid)

    def theta_max_Tracer(self, abstau_max, S) -> float:
        """
        Calculate the maximum Shields number [-] for the tracer sand particle in the fluid.
        """
        return abstau_max / (S.g * (S.rhoParticle - S.rhoFluid) * S.dTracer)

    def theta_max_Background(self, abstau_max, S) -> float:
        """
        Calculate the maximum Shields number [-] for the background sand particle in the fluid.
        """
        return abstau_max / (S.g * (S.rhoParticle - S.rhoFluid) * S.dBackground)

    def rouse(self, w_s_t, ustar_max) -> float:
        """
        Calculate the Rouse number [-] for the tracer sand particle in the fluid.
        """
        return w_s_t / (0.4 * ustar_max)

    def U_bed(self, ustar_mean, theta_cr_Tracer_exp, theta_max_Tracer) -> float:
        """
        Calculate the bedload velocity [m/s] of the tracer sand particle in the fluid.
        """
        return 10 * ustar_mean * (1 - 0.7 * np.sqrt(theta_cr_Tracer_exp / theta_max_Tracer))

class Mud(Sediment(Particle)):
    """
    Class representing mud particles properties.
    """
    def w_s_t(self) -> float:
        """
        Calculate the settling velocity [m/s] of the tracer mud particle.
        """
        return

    def theta_cr_Tracer(self) -> float:
        """
        Calculate the critical Shields number [-] for the tracer mud particle.
        """
        return

    def theta_cr_Background(self) -> float:
        """
        Calculate the critical Shields number [-] for the background mud particle.
        """
        return

    def theta_cr_Tracer_exp(self) -> float:
        """
        Calculate the adsjusted critical Shields number [-] for the tracer mud particle to
        take into account hiding/exposure when having a diameter different than the background mud particles.
        """
        return

    def ustar_cr(self) -> float:
        """
        Calculate the critical friction velocity [m/s] for mud particle erosion in the fluid.
        """
        return

    def theta_max_Tracer(self) -> float:
        """
        Calculate the maximum Shields number [-] for the tracer mud particle in the fluid.
        """
        return

    def theta_max_Background(self) -> float:
        """
        Calculate the maximum Shields number [-] for the background mud particle in the fluid.
        """
        return

    def rouse(self) -> float:
        """
        Calculate the Rouse number [-] for the tracer mud particle in the fluid.
        """
        return

    def U_bed(self) -> float:
        """
        Calculate the bedload velocity [m/s] of the tracer mud particle in the fluid.
        """
        return

class Biological(Particle):
    """
    Class representing biologcal particles properties.
    """

class Propagule(Biological(Particle)):
    """
    Class representing mangrove propagules properties.
    """

class Larva(Biological(Particle)):
    """
    Class representing coral larvae properties.
    """

class eDNA(Biological(Particle)):
    """
    Class representing eDNA properties.
    """
