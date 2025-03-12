"""
ParticlePositionCalculator Class

This class performs particle advection over an unstructured grid using a
4‑stage Runge–Kutta integration scheme. It supports barycentric interpolation
of grid fields (e.g., velocities) via a matplotlib Triangulation, with an option
to use multiprocessing for very large particle sets.

Usage:
    1. Initialize with grid nodes, velocities, and face‐node connectivity.
    2. Call update_particles() with particle positions, time step, and diffusion.

Dependencies: numpy, matplotlib, multiprocessing, typing, math
"""

import multiprocessing as mp
from math import pi
from typing import Callable, Tuple

import matplotlib.tri as mtri
import numpy as np
from numpy.typing import NDArray


class ParticlePositionCalculator:
    def __init__(
        self,
        grid_x: NDArray[np.float64],
        grid_y: NDArray[np.float64],
        grid_u: NDArray[np.float64],
        grid_v: NDArray[np.float64],
        igeo: int = 0,
    ):
        """
        Initialize the ParticleInterpolator.

        Parameters
        ----------
        grid_x, grid_y : array_like, shape (N,)
            Coordinates of grid nodes.
        grid_u, grid_v : array_like, shape (N,)
            Velocity components at the grid nodes.
        igeo : int, optional
            Option flag. If igeo==1, grid velocities are adjusted for geographic
            coordinates (i.e. scaled by cosine(latitude) and a geofactor).
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_u = grid_u
        self.grid_v = grid_v
        self.igeo = igeo
        self.geofac = np.float64(
            6378137
        )  # conversion factor for geographic coordinates

        # Build the triangulation used for interpolation uisng Delaunay
        self.tri, self.trifinder = self.build_triangulation()

    def build_triangulation(
        self,
    ) -> Tuple[mtri.Triangulation, Callable[[float, float], int]]:
        """
        Build a matplotlib Triangulation and its trifinder using the grid nodes and triangles.

        Returns
        -------
        tri : matplotlib.tri.Triangulation
            The triangulation object.
        trifinder : callable
            A function that maps (x,y) points to a triangle index (or -1 if outside).
        """
        tri = mtri.Triangulation(self.grid_x, self.grid_y)  # Delaunay
        trifinder = tri.get_trifinder()
        return tri, trifinder

    def _interpolate_field(
        self, field: NDArray, part_x: NDArray, part_y: NDArray
    ) -> NDArray:
        """
        Interpolate a scalar field at particle positions using barycentric coordinates.

        Parameters
        ----------
        field : array_like, shape (N,)
            Field defined at grid nodes (e.g., grid_u or grid_v).
        part_x, part_y : array_like, shape (P,)
            Particle positions.

        Returns
        -------
        interp_vals : ndarray, shape (P,)
            Interpolated field values at the particle positions.
        """
        # Use the trifinder to determine which triangle contains each particle.
        tri_idx = self.trifinder(part_x, part_y)
        interp_vals = np.zeros_like(part_x)
        valid = tri_idx >= 0
        if not np.any(valid):
            return interp_vals

        # Get vertex indices for the containing triangle for valid particles.
        v1 = self.tri.triangles[tri_idx[valid], 1]
        v2 = self.tri.triangles[tri_idx[valid], 2]
        v0 = self.tri.triangles[tri_idx[valid], 0]

        # Coordinates for triangle vertices.
        x0 = self.grid_x[v0]
        y0 = self.grid_y[v0]
        x1 = self.grid_x[v1]
        y1 = self.grid_y[v1]
        x2 = self.grid_x[v2]
        y2 = self.grid_y[v2]

        # Compute barycentric coordinates.
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        with np.errstate(divide="ignore", invalid="ignore"):
            w1 = (
                (y1 - y2) * (part_x[valid] - x2) + (x2 - x1) * (part_y[valid] - y2)
            ) / den
            w2 = (
                (y2 - y0) * (part_x[valid] - x2) + (x0 - x2) * (part_y[valid] - y2)
            ) / den
        w3 = 1.0 - w1 - w2

        # Weighted sum of field values.
        interp_vals[valid] = w1 * field[v0] + w2 * field[v1] + w3 * field[v2]
        return interp_vals

    @staticmethod
    def _interpolate_field_worker(
        args: Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray],
    ) -> NDArray:
        """
        Static helper function for parallel field interpolation.

        Parameters
        ----------
        args : tuple
            Contains (grid_x, grid_y, field, triangles, part_x_chunk, part_y_chunk).

        Returns
        -------
        interp_vals : ndarray
            Interpolated field values for the chunk of particles.
        """
        (grid_x, grid_y, field, triangles, part_x_chunk, part_y_chunk) = args
        # Build a temporary triangulation and trifinder.
        tri = mtri.Triangulation(grid_x, grid_y)
        trifinder = tri.get_trifinder()

        part_x_chunk = np.asarray(part_x_chunk, dtype=np.float64)
        part_y_chunk = np.asarray(part_y_chunk, dtype=np.float64)

        tri_idx = trifinder(part_x_chunk, part_y_chunk)
        interp_vals = np.zeros_like(part_x_chunk)
        valid = tri_idx >= 0
        if not np.any(valid):
            return interp_vals

        v0 = triangles[tri_idx[valid], 0]
        v1 = triangles[tri_idx[valid], 1]
        v2 = triangles[tri_idx[valid], 2]
        x0 = grid_x[v0]
        y0 = grid_y[v0]
        x1 = grid_x[v1]
        y1 = grid_y[v1]
        x2 = grid_x[v2]
        y2 = grid_y[v2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        with np.errstate(divide="ignore", invalid="ignore"):
            w1 = (
                (y1 - y2) * (part_x_chunk[valid] - x2)
                + (x2 - x1) * (part_y_chunk[valid] - y2)
            ) / den
            w2 = (
                (y2 - y0) * (part_x_chunk[valid] - x2)
                + (x0 - x2) * (part_y_chunk[valid] - y2)
            ) / den
        w3 = 1.0 - w1 - w2
        interp_vals[valid] = w1 * field[v0] + w2 * field[v1] + w3 * field[v2]
        return interp_vals

    def parallel_interpolate_field(
        self, field: NDArray, part_x: NDArray, part_y: NDArray, num_workers: int
    ) -> NDArray:
        """
        Parallel version of _interpolate_field using multiprocessing.

        Parameters
        ----------
        field : ndarray, shape (N,)
            Field defined on the grid.
        part_x, part_y : ndarray, shape (P,)
            Particle positions.
        num_workers : int
            Number of processes to use.

        Returns
        -------
        interp_vals : ndarray, shape (P,)
            Interpolated field values at particle positions.
        """
        P = len(part_x)
        indices = np.array_split(np.arange(P), num_workers)
        args_list = []
        for idx in indices:
            args = (
                self.grid_x,
                self.grid_y,
                field,
                self.tri.triangles,
                part_x[idx],
                part_y[idx],
            )
            args_list.append(args)
        with mp.Pool(num_workers) as pool:
            results = pool.map(
                ParticlePositionCalculator._interpolate_field_worker, args_list
            )
        return np.concatenate(results)

    def interpolate_field(
        self,
        field: NDArray,
        part_x: NDArray,
        part_y: NDArray,
        parallel: bool = False,
        num_workers: int = None,
    ) -> NDArray:
        """
        Public method to interpolate a grid field at given particle positions.

        Parameters
        ----------
        field : array_like, shape (N,)
            Grid field (e.g., grid_u or grid_v).
        part_x, part_y : array_like, shape (P,)
            Particle positions.
        parallel : bool, optional
            If True, use multiprocessing.
        num_workers : int, optional
            Number of processes for parallel interpolation (defaults to all CPUs).

        Returns
        -------
        interp_vals : ndarray, shape (P,)
            Interpolated field values.
        """
        if parallel:
            if num_workers is None:
                num_workers = mp.cpu_count()
            return self.parallel_interpolate_field(field, part_x, part_y, num_workers)
        else:
            return self._interpolate_field(field, part_x, part_y)

    def update_particles(
        self,
        x0: NDArray,
        y0: NDArray,
        dt: np.float64,
        rndfac: np.float64,
        parallel: bool = False,
        num_workers: int = None,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Update particle positions using a 4‑stage Runge–Kutta integration on the grid.

        Parameters
        ----------
        x0, y0 : array_like, shape (P,)
            Particle positions at the current time.
        dt : float
            Time step.
        rndfac : float
            Diffusion coefficient. If > 0, random diffusion is added.
        parallel : bool, optional
            If True, use parallel interpolation.
        num_workers : int, optional
            Number of processes to use for parallel interpolation.

        Returns
        -------
        x_new, y_new : ndarray, shape (P,)
            Updated particle positions.
        xdiff, ydiff : ndarray, shape (P,)
            Diffusion increments (zero if rndfac <= 0).
        """
        geofac = self.geofac

        # Adjust grid velocities if using geographic coordinates.
        if self.igeo == 1:
            grid_u_adj = self.grid_u / (self.geofac * np.cos(np.deg2rad(self.grid_y)))
            grid_v_adj = self.grid_v / geofac
        else:
            grid_u_adj = self.grid_u
            grid_v_adj = self.grid_v

        # Helper to choose the interpolation method.
        def get_interp(field, part_x, part_y):
            return self.interpolate_field(
                field, part_x, part_y, parallel=parallel, num_workers=num_workers
            )

        # 4‑stage Runge–Kutta integration.
        # Stage 1.
        up1 = get_interp(grid_u_adj, x0, y0)
        vp1 = get_interp(grid_v_adj, x0, y0)
        x1a = x0 + 0.5 * up1 * dt
        y1a = y0 + 0.5 * vp1 * dt

        # Stage 2.
        up2 = get_interp(grid_u_adj, x1a, y1a)
        vp2 = get_interp(grid_v_adj, x1a, y1a)
        x1b = x0 + 0.5 * up2 * dt
        y1b = y0 + 0.5 * vp2 * dt

        # Stage 3.
        up3 = get_interp(grid_u_adj, x1b, y1b)
        vp3 = get_interp(grid_v_adj, x1b, y1b)
        x1c = x0 + up3 * dt
        y1c = y0 + vp3 * dt

        # Stage 4.
        up4 = get_interp(grid_u_adj, x1c, y1c)
        vp4 = get_interp(grid_v_adj, x1c, y1c)

        # Combine stages (RK4 integration).
        x_new = x0 + dt / 6.0 * (up1 + 2.0 * up2 + 2.0 * up3 + up4)
        y_new = y0 + dt / 6.0 * (vp1 + 2.0 * vp2 + 2.0 * vp3 + vp4)

        # Add random diffusion if requested.
        if rndfac > 0.0:
            vel_mag = np.sqrt(((x_new - x0) / dt) ** 2 + ((y_new - y0) / dt) ** 2)
            rndnr_mag = np.random.randn(*vel_mag.shape)
            mag_diff = np.abs(rndnr_mag * rndfac) * vel_mag
            rndnr_angle = np.random.rand(*vel_mag.shape)
            angle_diff = rndnr_angle * 2 * pi
            xdiff = mag_diff * np.cos(angle_diff) * dt
            ydiff = mag_diff * np.sin(angle_diff) * dt
            x_new += xdiff
            y_new += ydiff
        else:
            xdiff = np.zeros_like(x_new)
            ydiff = np.zeros_like(y_new)

        return x_new, y_new, xdiff, ydiff
