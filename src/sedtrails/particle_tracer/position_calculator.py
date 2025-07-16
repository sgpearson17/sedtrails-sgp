"""
ParticlePositionCalculator Class

This class performs particle advection over an unstructured grid using a
4-stage Runge-Kutta integration scheme. It supports barycentric interpolation
of grid fields (e.g., velocities) via a matplotlib Triangulation, with an option
to use multiprocessing for very large particle sets.

Usage:
    1. Initialize with grid nodes, velocities, and face-node connectivity.
    2. Call update_particles() with particle positions, and time step.

Dependencies: numpy, numba, scipy
"""

from typing import Optional, Tuple

import numba
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from scipy.spatial import Delaunay

# ------------------------------------------------------------------------------
# Numba‐compiled core routines
# ------------------------------------------------------------------------------


@njit
def _interpolate_field(
    field: NDArray, x_points: NDArray, y_points: NDArray, grid_x: NDArray, grid_y: NDArray, triangles: NDArray
) -> NDArray:
    """
    Interpolates values of a scalar field at given (x, y) points using barycentric
    interpolation over a triangulated grid.

    Parameters
    ----------
    field : ndarray of shape (M,)
        Values of the scalar field at the grid points.
    x_points : ndarray of shape (N,)
        X-coordinates of the target interpolation points.
    y_points : ndarray of shape (N,)
        Y-coordinates of the target interpolation points.
    grid_x : ndarray of shape (M,)
        X-coordinates of the grid points (same length as `field`).
    grid_y : ndarray of shape (M,)
        Y-coordinates of the grid points (same length as `field`).
    triangles : ndarray of shape (K, 3)
        Indices of grid points forming triangles in the triangulation. Each row
        represents a triangle using indices into `grid_x`, `grid_y`, and `field`.

    Returns
    -------
    out : ndarray of shape (N,)
        Interpolated field values at the given `(x_points, y_points)`.

    """
    n = x_points.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x, y = x_points[i], y_points[i]
        val = 0.0
        for j in range(triangles.shape[0]):
            v0, v1, v2 = triangles[j]
            x0, y0 = grid_x[v0], grid_y[v0]
            x1, y1 = grid_x[v1], grid_y[v1]
            x2, y2 = grid_x[v2], grid_y[v2]

            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-10:
                continue

            w1 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
            w2 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
            w3 = 1.0 - w1 - w2
            if w1 >= -1e-10 and w2 >= -1e-10 and w3 >= -1e-10:
                val = w1 * field[v0] + w2 * field[v1] + w3 * field[v2]
                break
        out[i] = val
    return out


@njit
def _update_particles_rk4(
    x0: NDArray,
    y0: NDArray,
    grid_u: NDArray,
    grid_v: NDArray,
    grid_x: NDArray,
    grid_y: NDArray,
    triangles: NDArray,
    dt: np.float32,
    igeo: int,
    geofac: np.float32,
) -> tuple[NDArray, NDArray]:
    """
    Update particle positions using 4th-order Runge-Kutta (RK4) integration.

    This function integrates particle positions over a single time step using
    barycentric interpolation of velocity fields defined on an unstructured grid.

    Parameters
    ----------
    x0 : array_like, shape (n_particles,)
        Initial x-coordinates of particles.
    y0 : array_like, shape (n_particles,)
        Initial y-coordinates of particles.
    grid_u : array_like, shape (n_nodes,)
        X-component (u) of velocity field defined at grid nodes.
    grid_v : array_like, shape (n_nodes,)
        Y-component (v) of velocity field defined at grid nodes.
    grid_x : array_like, shape (n_nodes,)
        X-coordinates of the grid nodes.
    grid_y : array_like, shape (n_nodes,)
        Y-coordinates of the grid nodes.
    triangles : array_like, shape (n_triangles, 3)
        Triangle connectivity (indices into grid_x/grid_y).
    dt : np.float32
        Time step for RK4 integration.
    igeo : int
        If 1, apply geographic coordinate correction to velocity components.
    geofac : np.float32
        Geographic scaling factor (e.g., Earth's radius in meters).

    Returns
    -------
    x_new : ndarray, shape (n_particles,)
        Updated x-coordinates of particles after one RK4 time step.
    y_new : ndarray, shape (n_particles,)
        Updated y-coordinates of particles after one RK4 time step.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    grid_u = np.asarray(grid_u, dtype=np.float64)
    grid_v = np.asarray(grid_v, dtype=np.float64)

    n = x0.shape[0]
    x_new = np.empty(n, dtype=np.float64)
    y_new = np.empty(n, dtype=np.float64)

    # adjust for geographic coords?
    u_adj = grid_u.copy()
    v_adj = grid_v.copy()
    if igeo == 1:
        for k in range(grid_v.shape[0]):
            coslat = np.cos(np.deg2rad(grid_y[k]))
            u_adj[k] = grid_u[k] / (geofac * coslat)
            v_adj[k] = grid_v[k] / geofac

    for i in range(n):
        xi, yi = x0[i], y0[i]
        # four RK stages
        ups = np.zeros(4, dtype=np.float64)
        vps = np.zeros(4, dtype=np.float64)
        xs = np.zeros(4, dtype=np.float64)
        ys = np.zeros(4, dtype=np.float64)

        xs[0], ys[0] = xi, yi

        for stage in range(4):
            # pick input position
            xa = xs[stage]
            ya = ys[stage]

            # interpolate velocity at (xa,ya)
            up = 0.0
            vp = 0.0
            for j in range(triangles.shape[0]):
                v0, v1, v2 = triangles[j]
                x0t, y0t = grid_x[v0], grid_y[v0]
                x1t, y1t = grid_x[v1], grid_y[v1]
                x2t, y2t = grid_x[v2], grid_y[v2]

                denom = (y1t - y2t) * (x0t - x2t) + (x2t - x1t) * (y0t - y2t)
                if abs(denom) < 1e-10:
                    continue

                w1 = ((y1t - y2t) * (xa - x2t) + (x2t - x1t) * (ya - y2t)) / denom
                w2 = ((y2t - y0t) * (xa - x2t) + (x0t - x2t) * (ya - y2t)) / denom
                w3 = 1.0 - w1 - w2
                if w1 >= -1e-10 and w2 >= -1e-10 and w3 >= -1e-10:
                    up = w1 * u_adj[v0] + w2 * u_adj[v1] + w3 * u_adj[v2]
                    vp = w1 * v_adj[v0] + w2 * v_adj[v1] + w3 * v_adj[v2]
                    break

            ups[stage] = up
            vps[stage] = vp

            if stage == 0:
                xs[1] = xi + 0.5 * up * dt
                ys[1] = yi + 0.5 * vp * dt
                xs[2] = xi + 0.5 * up * dt
                ys[2] = yi + 0.5 * vp * dt
                xs[3] = xi + up * dt
                ys[3] = yi + vp * dt
            elif stage == 1:
                xs[2] = xi + 0.5 * ups[1] * dt
                ys[2] = yi + 0.5 * vps[1] * dt
            # stage 2 and 3 already set by above logic

        # combine
        x_new[i] = xi + dt / 6.0 * (ups[0] + 2 * ups[1] + 2 * ups[2] + ups[3])
        y_new[i] = yi + dt / 6.0 * (vps[0] + 2 * vps[1] + 2 * vps[2] + vps[3])

    return x_new, y_new


@njit(parallel=True)
def _update_particles_rk4_parallel(
    x0: NDArray,
    y0: NDArray,
    grid_u: NDArray,
    grid_v: NDArray,
    grid_x: NDArray,
    grid_y: NDArray,
    triangles: NDArray,
    dt: np.float32,
    igeo: int,
    geofac: np.float32,
) -> tuple[NDArray, NDArray]:
    """
    Update particle positions using 4th-order Runge-Kutta (RK4) integration in parallel.

    This function integrates particle positions over a single time step using
    barycentric interpolation of velocity fields defined on an unstructured grid.

    Parameters
    ----------
    x0 : array_like, shape (n_particles,)
        Initial x-coordinates of particles.
    y0 : array_like, shape (n_particles,)
        Initial y-coordinates of particles.
    grid_u : array_like, shape (n_nodes,)
        X-component (u) of velocity field defined at grid nodes.
    grid_v : array_like, shape (n_nodes,)
        Y-component (v) of velocity field defined at grid nodes.
    grid_x : array_like, shape (n_nodes,)
        X-coordinates of the grid nodes.
    grid_y : array_like, shape (n_nodes,)
        Y-coordinates of the grid nodes.
    triangles : array_like, shape (n_triangles, 3)
        Triangle connectivity (indices into grid_x/grid_y).
    dt : np.float32
        Time step for RK4 integration.
    igeo : int
        If 1, apply geographic coordinate correction to velocity components.
    geofac : np.float32
        Geographic scaling factor (e.g., Earth's radius in meters).

    Returns
    -------
    x_new : ndarray, shape (n_particles,)
        Updated x-coordinates of particles after one RK4 time step.
    y_new : ndarray, shape (n_particles,)
        Updated y-coordinates of particles after one RK4 time step.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    grid_u = np.asarray(grid_u, dtype=np.float64)
    grid_v = np.asarray(grid_v, dtype=np.float64)

    n = x0.shape[0]
    x_new = np.empty(n, dtype=np.float64)
    y_new = np.empty(n, dtype=np.float64)

    u_adj = grid_u.copy()
    v_adj = grid_v.copy()
    if igeo == 1:
        for k in range(grid_v.shape[0]):
            coslat = np.cos(np.deg2rad(grid_y[k]))
            u_adj[k] = grid_u[k] / (geofac * coslat)
            v_adj[k] = grid_v[k] / geofac

    for i in prange(n):
        xi, yi = x0[i], y0[i]
        # do the same four‐stage RK4 as above
        ups = np.zeros(4, dtype=np.float64)
        vps = np.zeros(4, dtype=np.float64)
        xs = np.zeros(4, dtype=np.float64)
        ys = np.zeros(4, dtype=np.float64)

        xs[0], ys[0] = xi, yi
        for stage in range(4):
            xa = xs[stage]
            ya = ys[stage]
            up = 0.0
            vp = 0.0
            for j in range(triangles.shape[0]):
                v0, v1, v2 = triangles[j]
                x0t, y0t = grid_x[v0], grid_y[v0]
                x1t, y1t = grid_x[v1], grid_y[v1]
                x2t, y2t = grid_x[v2], grid_y[v2]

                denom = (y1t - y2t) * (x0t - x2t) + (x2t - x1t) * (y0t - y2t)
                if abs(denom) < 1e-10:
                    continue

                w1 = ((y1t - y2t) * (xa - x2t) + (x2t - x1t) * (ya - y2t)) / denom
                w2 = ((y2t - y0t) * (xa - x2t) + (x0t - x2t) * (ya - y2t)) / denom
                w3 = 1.0 - w1 - w2
                if w1 >= -1e-10 and w2 >= -1e-10 and w3 >= -1e-10:
                    up = w1 * u_adj[v0] + w2 * u_adj[v1] + w3 * u_adj[v2]
                    vp = w1 * v_adj[v0] + w2 * v_adj[v1] + w3 * v_adj[v2]
                    break

            ups[stage] = up
            vps[stage] = vp

            if stage == 0:
                xs[1] = xi + 0.5 * up * dt
                ys[1] = yi + 0.5 * vp * dt
                xs[2] = xi + 0.5 * up * dt
                ys[2] = yi + 0.5 * vp * dt
                xs[3] = xi + up * dt
                ys[3] = yi + vp * dt
            elif stage == 1:
                xs[2] = xi + 0.5 * ups[1] * dt
                ys[2] = yi + 0.5 * vps[1] * dt

        x_new[i] = xi + dt / 6.0 * (ups[0] + 2 * ups[1] + 2 * ups[2] + ups[3])
        y_new[i] = yi + dt / 6.0 * (vps[0] + 2 * vps[1] + 2 * vps[2] + vps[3])

    return x_new, y_new


# ------------------------------------------------------------------------------
# Public API class ParticlePositionCalculator
# ------------------------------------------------------------------------------


class ParticlePositionCalculator:
    """
    Numba‐optimized particle tracer with RK4 integration on an unstructured grid.
    """

    def __init__(
        self,
        grid_x: NDArray,
        grid_y: NDArray,
        grid_u: NDArray,
        grid_v: NDArray,
        triangles: Optional[NDArray] = None,
        igeo: int = 0,
    ) -> None:
        """
        Initialize the particle position calculator.

        Parameters
        ----------
        grid_x : ndarray of shape (M,)
            X-coordinates of the velocity field grid.
        grid_y : ndarray of shape (M,)
            Y-coordinates of the velocity field grid.
        grid_u : ndarray of shape (M,)
            X-component (u) of the velocity field.
        grid_v : ndarray of shape (M,)
            Y-component (v) of the velocity field.
        triangles : ndarray of shape (K, 3), optional
            Triangle connectivity array for the grid. If None, it is computed via Delaunay triangulation.
        igeo : int, default=0
            If 1, assumes geographic coordinates (degrees), scaled using Earth's radius.
        """
        self.grid_x = np.asarray(grid_x, dtype=np.float64)
        self.grid_y = np.asarray(grid_y, dtype=np.float64)
        self.grid_u = np.asarray(grid_u, dtype=np.float64)
        self.grid_v = np.asarray(grid_v, dtype=np.float64)
        self.igeo = int(igeo)
        self.geofac = 6378137.0

        if triangles is None:
            pts = np.column_stack((self.grid_x, self.grid_y))
            tri = Delaunay(pts)
            tris = tri.simplices
        else:
            tris = triangles

        self.triangles = np.asarray(tris, dtype=np.int64)

    def interpolate_field(self, field: NDArray, x_pts: NDArray, y_pts: NDArray) -> NDArray:
        """
        Barycentric interpolation of a scalar field at given (x, y) points.

        Parameters
        ----------
        field : ndarray of shape (M,)
            Field values defined at grid points.
        x_pts : ndarray of shape (N,)
            X-coordinates of the interpolation points.
        y_pts : ndarray of shape (N,)
            Y-coordinates of the interpolation points.

        Returns
        -------
        ndarray of shape (N,)
            Interpolated field values at the specified coordinates.
        """
        fld = np.asarray(field, dtype=np.float64)
        xs = np.asarray(x_pts, dtype=np.float64)
        ys = np.asarray(y_pts, dtype=np.float64)

        return _interpolate_field(fld, xs, ys, self.grid_x, self.grid_y, self.triangles)

    def update_particles(
        self, x0: NDArray, y0: NDArray, dt: np.float32, parallel: bool = False, num_workers: Optional[int] = None
    ) -> Tuple[NDArray, NDArray]:
        """
        Perform one Runge-Kutta 4th order (RK4) time step to update particle positions.

        Parameters
        ----------
        x0 : ndarray of shape (N,)
            Initial x-positions of the particles.
        y0 : ndarray of shape (N,)
            Initial y-positions of the particles.
        dt : np.float32
            Time step size.
        parallel : bool, default=False
            If True, use parallelized Numba version with `prange`.
        num_workers : int, optional
            Number of threads for parallel execution. Ignored if `parallel=False`.

        Returns
        -------
        Tuple of (ndarray, ndarray)
            Updated x and y particle positions after one RK4 step.
        """
        xs = np.asarray(x0, dtype=np.float64)
        ys = np.asarray(y0, dtype=np.float64)
        dt = np.float32(dt)

        if parallel:
            if num_workers is not None:
                numba.set_num_threads(num_workers)
            return _update_particles_rk4_parallel(
                xs, ys, self.grid_u, self.grid_v, self.grid_x, self.grid_y, self.triangles, dt, self.igeo, self.geofac
            )
        else:
            return _update_particles_rk4(
                xs, ys, self.grid_u, self.grid_v, self.grid_x, self.grid_y, self.triangles, dt, self.igeo, self.geofac
            )
