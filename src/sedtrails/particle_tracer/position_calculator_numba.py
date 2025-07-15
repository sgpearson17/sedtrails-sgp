"""
Numba-optimized ParticlePositionCalculator

This module provides a Numba-accelerated implementation of particle
position calculation using a 4-stage Runge-Kutta integration scheme
on unstructured grids. It replaces the matplotlib triangulation and
multiprocessing approaches with Numba-compatible alternatives.

Dependencies: numpy, numba, scipy (only for initial triangulation)
"""

import numpy as np
from numba import njit, prange


def create_numba_particle_calculator(grid_x, grid_y, triangles=None):
    """
    Factory function to create a Numba-optimized particle calculator.

    This function preprocesses the grid data and returns optimized Numba functions
    for particle position calculation.

    Parameters
    ----------
    grid_x, grid_y : array_like
        Coordinates of grid nodes
    triangles : array_like, optional
        Triangle connectivity (node indices). If None, Delaunay triangulation
        will be computed once (not using Numba).

    Returns
    -------
    dict
        Dictionary containing optimized Numba functions
    """
    # If triangles not provided, compute them once using scipy
    if triangles is None:
        from scipy.spatial import Delaunay

        points = np.column_stack((grid_x, grid_y))
        triangulation = Delaunay(points)
        triangles = triangulation.simplices

    # Pre-compute triangle information for faster lookup
    triangles = np.asarray(triangles, dtype=np.int64)

    # Return the calculator functions WITHOUT trying to decorate them again
    return {
        'triangles': triangles,
        'find_triangle': lambda x, y: find_triangle(x, y, grid_x, grid_y, triangles),
        'interpolate_field': lambda field, x, y: interpolate_field(field, x, y, grid_x, grid_y, triangles),
        'update_particles': lambda x0, y0, grid_u, grid_v, dt, igeo=0: update_particles_rk4(
            x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo
        ),
        'update_particles_parallel': lambda x0, y0, grid_u, grid_v, dt, igeo=0: update_particles_rk4_parallel(
            x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo
        ),
    }


@njit
def find_triangle(x, y, grid_x, grid_y, triangles):
    """
    Find which triangle contains the point (x, y).

    Parameters
    ----------
    x, y : float
        Coordinates of the point
    grid_x, grid_y : array_like
        Coordinates of grid nodes
    triangles : array_like
        Triangle connectivity (node indices)

    Returns
    -------
    int
        Triangle index or -1 if outside all triangles
    """
    for i in range(len(triangles)):
        # Get triangle vertices
        v0, v1, v2 = triangles[i]
        x0, y0 = grid_x[v0], grid_y[v0]
        x1, y1 = grid_x[v1], grid_y[v1]
        x2, y2 = grid_x[v2], grid_y[v2]

        # Barycentric coordinates calculation
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-10:
            continue

        w1 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
        w2 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
        w3 = 1.0 - w1 - w2

        # If all weights are between 0 and 1, point is inside triangle
        if (w1 >= -1e-10) and (w2 >= -1e-10) and (w3 >= -1e-10):
            return i

    return -1


@njit
def interpolate_field(field, x_points, y_points, grid_x, grid_y, triangles):
    """
    Interpolate field values at given points using barycentric coordinates.

    Parameters
    ----------
    field : array_like
        Field values at grid nodes
    x_points, y_points : array_like
        Coordinates of points where to interpolate
    grid_x, grid_y : array_like
        Coordinates of grid nodes
    triangles : array_like
        Triangle connectivity (node indices)

    Returns
    -------
    array_like
        Interpolated field values
    """
    n_points = len(x_points)
    result = np.zeros(n_points, dtype=np.float64)

    for i in range(n_points):
        x, y = x_points[i], y_points[i]
        # Find containing triangle
        tri_idx = -1

        for j in range(len(triangles)):
            # Get triangle vertices
            v0, v1, v2 = triangles[j]
            x0, y0 = grid_x[v0], grid_y[v0]
            x1, y1 = grid_x[v1], grid_y[v1]
            x2, y2 = grid_x[v2], grid_y[v2]

            # Barycentric coordinates calculation
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-10:
                continue

            w1 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
            w2 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
            w3 = 1.0 - w1 - w2

            # If all weights are between 0 and 1, point is inside triangle
            if (w1 >= -1e-10) and (w2 >= -1e-10) and (w3 >= -1e-10):
                tri_idx = j

                # Compute interpolated value using barycentric coordinates
                v0, v1, v2 = triangles[tri_idx]
                result[i] = w1 * field[v0] + w2 * field[v1] + w3 * field[v2]
                break

    return result


@njit
def update_particles_rk4(x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo=0):
    """
    Update particle positions using a 4-stage Runge-Kutta integration scheme.

    Parameters
    ----------
    x0, y0 : array_like
        Initial particle positions
    grid_u, grid_v : array_like
        Velocity components at grid nodes
    grid_x, grid_y : array_like
        Coordinates of grid nodes
    triangles : array_like
        Triangle connectivity (node indices)
    dt : float
        Time step
    igeo : int, optional
        Flag for geographic coordinates adjustment (default: 0)

    Returns
    -------
    tuple
        Updated particle positions (x_new, y_new)
    """
    # Convert inputs to arrays if they aren't already
    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    grid_u = np.asarray(grid_u, dtype=np.float64)
    grid_v = np.asarray(grid_v, dtype=np.float64)

    n_particles = len(x0)
    x_new = np.zeros(n_particles, dtype=np.float64)
    y_new = np.zeros(n_particles, dtype=np.float64)

    # Geographic adjustment factor
    geofac = 6378137.0

    # Adjust grid velocities if using geographic coordinates
    grid_u_adj = np.copy(grid_u)
    grid_v_adj = np.copy(grid_v)

    if igeo == 1:
        for i in range(len(grid_y)):
            cos_lat = np.cos(np.deg2rad(grid_y[i]))
            grid_u_adj[i] = grid_u[i] / (geofac * cos_lat)
            grid_v_adj[i] = grid_v[i] / geofac

    # RK4 integration for each particle
    for i in range(n_particles):
        xi, yi = x0[i], y0[i]

        # Stage 1
        up1 = 0.0
        vp1 = 0.0
        # tri_idx = -1

        # Find containing triangle
        for j in range(len(triangles)):
            # Get triangle vertices
            v0, v1, v2 = triangles[j]
            x0_tri, y0_tri = grid_x[v0], grid_y[v0]
            x1_tri, y1_tri = grid_x[v1], grid_y[v1]
            x2_tri, y2_tri = grid_x[v2], grid_y[v2]

            # Barycentric coordinates calculation
            denom = (y1_tri - y2_tri) * (x0_tri - x2_tri) + (x2_tri - x1_tri) * (y0_tri - y2_tri)
            if abs(denom) < 1e-10:
                continue

            w1 = ((y1_tri - y2_tri) * (xi - x2_tri) + (x2_tri - x1_tri) * (yi - y2_tri)) / denom
            w2 = ((y2_tri - y0_tri) * (xi - x2_tri) + (x0_tri - x2_tri) * (yi - y2_tri)) / denom
            w3 = 1.0 - w1 - w2

            # If all weights are between 0 and 1, point is inside triangle
            if (w1 >= -1e-10) and (w2 >= -1e-10) and (w3 >= -1e-10):
                # tri_idx = j

                # Interpolate velocity
                up1 = w1 * grid_u_adj[v0] + w2 * grid_u_adj[v1] + w3 * grid_u_adj[v2]
                vp1 = w1 * grid_v_adj[v0] + w2 * grid_v_adj[v1] + w3 * grid_v_adj[v2]
                break

        x1a = xi + 0.5 * up1 * dt
        y1a = yi + 0.5 * vp1 * dt

        # Stage 2
        up2 = 0.0
        vp2 = 0.0
        # tri_idx = -1

        # Find containing triangle
        for j in range(len(triangles)):
            v0, v1, v2 = triangles[j]
            x0_tri, y0_tri = grid_x[v0], grid_y[v0]
            x1_tri, y1_tri = grid_x[v1], grid_y[v1]
            x2_tri, y2_tri = grid_x[v2], grid_y[v2]

            denom = (y1_tri - y2_tri) * (x0_tri - x2_tri) + (x2_tri - x1_tri) * (y0_tri - y2_tri)
            if abs(denom) < 1e-10:
                continue

            w1 = ((y1_tri - y2_tri) * (x1a - x2_tri) + (x2_tri - x1_tri) * (y1a - y2_tri)) / denom
            w2 = ((y2_tri - y0_tri) * (x1a - x2_tri) + (x0_tri - x2_tri) * (y1a - y2_tri)) / denom
            w3 = 1.0 - w1 - w2

            if (w1 >= -1e-10) and (w2 >= -1e-10) and (w3 >= -1e-10):
                # tri_idx = j
                up2 = w1 * grid_u_adj[v0] + w2 * grid_u_adj[v1] + w3 * grid_u_adj[v2]
                vp2 = w1 * grid_v_adj[v0] + w2 * grid_v_adj[v1] + w3 * grid_v_adj[v2]
                break

        x1b = xi + 0.5 * up2 * dt
        y1b = yi + 0.5 * vp2 * dt

        # Stage 3
        up3 = 0.0
        vp3 = 0.0
        # tri_idx = -1

        # Find containing triangle
        for j in range(len(triangles)):
            v0, v1, v2 = triangles[j]
            x0_tri, y0_tri = grid_x[v0], grid_y[v0]
            x1_tri, y1_tri = grid_x[v1], grid_y[v1]
            x2_tri, y2_tri = grid_x[v2], grid_y[v2]

            denom = (y1_tri - y2_tri) * (x0_tri - x2_tri) + (x2_tri - x1_tri) * (y0_tri - y2_tri)
            if abs(denom) < 1e-10:
                continue

            w1 = ((y1_tri - y2_tri) * (x1b - x2_tri) + (x2_tri - x1_tri) * (y1b - y2_tri)) / denom
            w2 = ((y2_tri - y0_tri) * (x1b - x2_tri) + (x0_tri - x2_tri) * (y1b - y2_tri)) / denom
            w3 = 1.0 - w1 - w2

            if (w1 >= -1e-10) and (w2 >= -1e-10) and (w3 >= -1e-10):
                # tri_idx = j
                up3 = w1 * grid_u_adj[v0] + w2 * grid_u_adj[v1] + w3 * grid_u_adj[v2]
                vp3 = w1 * grid_v_adj[v0] + w2 * grid_v_adj[v1] + w3 * grid_v_adj[v2]
                break

        x1c = xi + up3 * dt
        y1c = yi + vp3 * dt

        # Stage 4
        up4 = 0.0
        vp4 = 0.0
        # tri_idx = -1

        # Find containing triangle
        for j in range(len(triangles)):
            v0, v1, v2 = triangles[j]
            x0_tri, y0_tri = grid_x[v0], grid_y[v0]
            x1_tri, y1_tri = grid_x[v1], grid_y[v1]
            x2_tri, y2_tri = grid_x[v2], grid_y[v2]

            denom = (y1_tri - y2_tri) * (x0_tri - x2_tri) + (x2_tri - x1_tri) * (y0_tri - y2_tri)
            if abs(denom) < 1e-10:
                continue

            w1 = ((y1_tri - y2_tri) * (x1c - x2_tri) + (x2_tri - x1_tri) * (y1c - y2_tri)) / denom
            w2 = ((y2_tri - y0_tri) * (x1c - x2_tri) + (x0_tri - x2_tri) * (y1c - y2_tri)) / denom
            w3 = 1.0 - w1 - w2

            if (w1 >= -1e-10) and (w2 >= -1e-10) and (w3 >= -1e-10):
                # tri_idx = j
                up4 = w1 * grid_u_adj[v0] + w2 * grid_u_adj[v1] + w3 * grid_u_adj[v2]
                vp4 = w1 * grid_v_adj[v0] + w2 * grid_v_adj[v1] + w3 * grid_v_adj[v2]
                break

        # Combine stages (RK4 integration)
        x_new[i] = xi + dt / 6.0 * (up1 + 2.0 * up2 + 2.0 * up3 + up4)
        y_new[i] = yi + dt / 6.0 * (vp1 + 2.0 * vp2 + 2.0 * vp3 + vp4)

    return x_new, y_new


@njit(parallel=True)
def update_particles_rk4_parallel(x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo=0):
    """
    Parallel version of update_particles_rk4 using Numba's prange.

    This function is optimized for large particle sets.

    Parameters are the same as update_particles_rk4.
    """
    # Convert inputs to arrays if they aren't already
    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    grid_u = np.asarray(grid_u, dtype=np.float64)
    grid_v = np.asarray(grid_v, dtype=np.float64)

    n_particles = len(x0)
    x_new = np.zeros(n_particles, dtype=np.float64)
    y_new = np.zeros(n_particles, dtype=np.float64)

    # Geographic adjustment factor
    geofac = 6378137.0

    # Adjust grid velocities if using geographic coordinates
    grid_u_adj = np.copy(grid_u)
    grid_v_adj = np.copy(grid_v)

    if igeo == 1:
        for i in range(len(grid_y)):
            cos_lat = np.cos(np.deg2rad(grid_y[i]))
            grid_u_adj[i] = grid_u[i] / (geofac * cos_lat)
            grid_v_adj[i] = grid_v[i] / geofac

    # RK4 integration for each particle in parallel
    for i in prange(n_particles):
        xi, yi = x0[i], y0[i]

        # Stage 1
        up1 = 0.0
        vp1 = 0.0
        # tri_idx = -1

        # Find containing triangle
        for j in range(len(triangles)):
            # Get triangle vertices
            v0, v1, v2 = triangles[j]
            x0_tri, y0_tri = grid_x[v0], grid_y[v0]
            x1_tri, y1_tri = grid_x[v1], grid_y[v1]
            x2_tri, y2_tri = grid_x[v2], grid_y[v2]

            # Barycentric coordinates calculation
            denom = (y1_tri - y2_tri) * (x0_tri - x2_tri) + (x2_tri - x1_tri) * (y0_tri - y2_tri)
            if abs(denom) < 1e-10:
                continue

            w1 = ((y1_tri - y2_tri) * (xi - x2_tri) + (x2_tri - x1_tri) * (yi - y2_tri)) / denom
            w2 = ((y2_tri - y0_tri) * (xi - x2_tri) + (x0_tri - x2_tri) * (yi - y2_tri)) / denom
            w3 = 1.0 - w1 - w2

            # If all weights are between 0 and 1, point is inside triangle
            if (w1 >= -1e-10) and (w2 >= -1e-10) and (w3 >= -1e-10):
                # tri_idx = j

                # Interpolate velocity
                up1 = w1 * grid_u_adj[v0] + w2 * grid_u_adj[v1] + w3 * grid_u_adj[v2]
                vp1 = w1 * grid_v_adj[v0] + w2 * grid_v_adj[v1] + w3 * grid_v_adj[v2]
                break

        x1a = xi + 0.5 * up1 * dt
        y1a = yi + 0.5 * vp1 * dt

        # Stage 2
        up2 = 0.0
        vp2 = 0.0
        # tri_idx = -1

        # Find containing triangle
        for j in range(len(triangles)):
            v0, v1, v2 = triangles[j]
            x0_tri, y0_tri = grid_x[v0], grid_y[v0]
            x1_tri, y1_tri = grid_x[v1], grid_y[v1]
            x2_tri, y2_tri = grid_x[v2], grid_y[v2]

            denom = (y1_tri - y2_tri) * (x0_tri - x2_tri) + (x2_tri - x1_tri) * (y0_tri - y2_tri)
            if abs(denom) < 1e-10:
                continue

            w1 = ((y1_tri - y2_tri) * (x1a - x2_tri) + (x2_tri - x1_tri) * (y1a - y2_tri)) / denom
            w2 = ((y2_tri - y0_tri) * (x1a - x2_tri) + (x0_tri - x2_tri) * (y1a - y2_tri)) / denom
            w3 = 1.0 - w1 - w2

            if (w1 >= -1e-10) and (w2 >= -1e-10) and (w3 >= -1e-10):
                # tri_idx = j
                up2 = w1 * grid_u_adj[v0] + w2 * grid_u_adj[v1] + w3 * grid_u_adj[v2]
                vp2 = w1 * grid_v_adj[v0] + w2 * grid_v_adj[v1] + w3 * grid_v_adj[v2]
                break

        x1b = xi + 0.5 * up2 * dt
        y1b = yi + 0.5 * vp2 * dt

        # Stage 3
        up3 = 0.0
        vp3 = 0.0
        # tri_idx = -1

        # Find containing triangle
        for j in range(len(triangles)):
            v0, v1, v2 = triangles[j]
            x0_tri, y0_tri = grid_x[v0], grid_y[v0]
            x1_tri, y1_tri = grid_x[v1], grid_y[v1]
            x2_tri, y2_tri = grid_x[v2], grid_y[v2]

            denom = (y1_tri - y2_tri) * (x0_tri - x2_tri) + (x2_tri - x1_tri) * (y0_tri - y2_tri)
            if abs(denom) < 1e-10:
                continue

            w1 = ((y1_tri - y2_tri) * (x1b - x2_tri) + (x2_tri - x1_tri) * (y1b - y2_tri)) / denom
            w2 = ((y2_tri - y0_tri) * (x1b - x2_tri) + (x0_tri - x2_tri) * (y1b - y2_tri)) / denom
            w3 = 1.0 - w1 - w2

            if (w1 >= -1e-10) and (w2 >= -1e-10) and (w3 >= -1e-10):
                # tri_idx = j
                up3 = w1 * grid_u_adj[v0] + w2 * grid_u_adj[v1] + w3 * grid_u_adj[v2]
                vp3 = w1 * grid_v_adj[v0] + w2 * grid_v_adj[v1] + w3 * grid_v_adj[v2]
                break

        x1c = xi + up3 * dt
        y1c = yi + vp3 * dt

        # Stage 4
        up4 = 0.0
        vp4 = 0.0
        # tri_idx = -1

        # Find containing triangle
        for j in range(len(triangles)):
            v0, v1, v2 = triangles[j]
            x0_tri, y0_tri = grid_x[v0], grid_y[v0]
            x1_tri, y1_tri = grid_x[v1], grid_y[v1]
            x2_tri, y2_tri = grid_x[v2], grid_y[v2]

            denom = (y1_tri - y2_tri) * (x0_tri - x2_tri) + (x2_tri - x1_tri) * (y0_tri - y2_tri)
            if abs(denom) < 1e-10:
                continue

            w1 = ((y1_tri - y2_tri) * (x1c - x2_tri) + (x2_tri - x1_tri) * (y1c - y2_tri)) / denom
            w2 = ((y2_tri - y0_tri) * (x1c - x2_tri) + (x0_tri - x2_tri) * (y1c - y2_tri)) / denom
            w3 = 1.0 - w1 - w2

            if (w1 >= -1e-10) and (w2 >= -1e-10) and (w3 >= -1e-10):
                # tri_idx = j
                up4 = w1 * grid_u_adj[v0] + w2 * grid_u_adj[v1] + w3 * grid_u_adj[v2]
                vp4 = w1 * grid_v_adj[v0] + w2 * grid_v_adj[v1] + w3 * grid_v_adj[v2]
                break

        # Combine stages (RK4 integration)
        x_new[i] = xi + dt / 6.0 * (up1 + 2.0 * up2 + 2.0 * up3 + up4)
        y_new[i] = yi + dt / 6.0 * (vp1 + 2.0 * vp2 + 2.0 * vp3 + vp4)

    return x_new, y_new
