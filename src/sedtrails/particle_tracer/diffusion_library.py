import math
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

__all__ = [
    'DiffusionStrategy',
    'BrownianDiffusionStrategy',
    'DiffusionCalculator',
]


class DiffusionStrategy(ABC):
    """Abstract base class for diffusion strategies."""

    @abstractmethod
    def calculate(
        self,
        dt: float,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        kh: float,
        rng: np.random.Generator | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply diffusion to the given positions and velocities.

           Parameters
           ----------
            dt: Time step for diffusion calculation
            x: Array of x-coordinates.
            y: Array of y-coordinates.
            u: Array of x-velocity components.
            v: Array of y-velocity components.
            kh: Diffusion coefficient.
            rng: Optional random-number generator for reproducible diffusion.

        Returns
        -------
            Tuple of updated x and y positions after diffusion (xdif, ydif).
        """
        pass


class BrownianDiffusionStrategy(DiffusionStrategy):
    """Isotropic Brownian diffusion using a constant kh coefficient."""

    def calculate(
        self,
        dt: float,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        kh: float,
        rng: np.random.Generator | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return calculate.

        Parameters
        ----------
        dt : float
            Integration timestep in seconds.
        x : np.ndarray
            X coordinate value or array.
        y : np.ndarray
            Y coordinate value or array.
        u : np.ndarray
            X velocity component.
        v : np.ndarray
            Y velocity component.
        kh : float
            Horizontal diffusivity value or array.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Array containing the computed values.
        """
        if kh == 0.0 or dt == 0.0:
            return x.copy(), y.copy()

        sigma = math.sqrt(2.0 * kh * dt)
        normal = np.random.standard_normal if rng is None else rng.standard_normal
        dx = sigma * normal(x.shape)
        dy = sigma * normal(y.shape)
        return x + dx, y + dy


class DiffusionCalculator:
    """
    Main class for calculating diffusion effects.
    """

    def __init__(self, strategy: DiffusionStrategy, rng: np.random.Generator | None = None):
        """Initialize with a diffusion strategy.

        Parameters
        ----------
            strategy: DiffusionStrategy object to use.
            rng: Optional population-specific random-number generator.
        """
        self._strategy = strategy
        self._rng = rng

    @property
    def strategy(self) -> DiffusionStrategy:
        """
        Get the current diffusion strategy.

        Returns
        -------
        DiffusionStrategy
            The strategy value.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: DiffusionStrategy):
        """Change the diffusion strategy.

        Parameters
        ----------
            strategy: New DiffusionStrategy object to use
        """
        self._strategy = strategy

    def calc_diffusion(
        self,
        x: float,
        y: float,
        u: np.ndarray,
        v: np.ndarray,
        kh: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate diffusion effect on position.

        Parameters
        ----------
            x: Current x position
            y: Current y position
            u: Velocity field x-component (2D array)
            v: Velocity field y-component (2D array)
            kh: Diffusion coefficient
            dt: Current time step

        Returns
        -------
            Tuple of (x_diffusion, y_diffusion) representing position changes
        """
        return self._strategy.calculate(dt, x, y, u, v, kh, rng=self._rng)
