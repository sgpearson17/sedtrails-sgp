"""
Particle position calculation on an unstructured triangular grid.

The public factory name is kept for compatibility with existing callers, but
the implementation now caches SciPy Delaunay geometry and uses its simplex
search instead of scanning every triangle for every interpolation point when
connectivity is not supplied explicitly.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


TRIANGLE_TOLERANCE = 1e-10


@dataclass
class GridGeometry:
    """Cached spatial geometry shared by particle populations on the same grid."""

    grid_x: np.ndarray
    grid_y: np.ndarray
    triangulation: Any
    triangles: np.ndarray
    outer_envelope: np.ndarray
    triangle_finder: Any = None

    @classmethod
    def from_points(cls, grid_x, grid_y, triangles=None):
        x = np.asarray(grid_x, dtype=np.float64).ravel()
        y = np.asarray(grid_y, dtype=np.float64).ravel()

        if x.shape != y.shape:
            raise ValueError(f'grid_x and grid_y must have the same shape, got {x.shape} and {y.shape}')

        points = np.column_stack((x, y))
        finite_points = points[np.isfinite(points).all(axis=1)]
        if finite_points.shape[0] < 3:
            raise ValueError('At least three finite grid points are required for particle interpolation')
        if finite_points.shape[0] != points.shape[0]:
            raise ValueError('grid_x and grid_y must contain only finite coordinates')

        triangle_finder = None
        if triangles is None:
            # Delaunay owns the simplex search structure and affine transforms.
            triangulation = Delaunay(points)
            triangle_array = np.asarray(triangulation.simplices, dtype=np.int64)
        else:
            import matplotlib.tri as mtri

            triangle_array = np.asarray(triangles, dtype=np.int64)
            if triangle_array.ndim != 2 or triangle_array.shape[1] != 3:
                raise ValueError('triangles must be an array with shape (n_triangles, 3)')
            triangulation = None
            triangle_finder = mtri.Triangulation(x, y, triangle_array).get_trifinder()

        unique_points = np.unique(finite_points, axis=0)
        if unique_points.shape[0] >= 3:
            try:
                outer_envelope = unique_points[ConvexHull(unique_points).vertices]
            except Exception:
                outer_envelope = _bounding_box(unique_points)
        else:
            outer_envelope = _bounding_box(unique_points)

        return cls(
            grid_x=x,
            grid_y=y,
            triangulation=triangulation,
            triangles=triangle_array,
            outer_envelope=outer_envelope,
            triangle_finder=triangle_finder,
        )

    def find_triangle(self, x, y) -> int:
        """Return the containing simplex index, or -1 outside the triangulation."""
        if self.triangulation is not None:
            simplex = self.triangulation.find_simplex(np.array([[x, y]], dtype=np.float64), tol=TRIANGLE_TOLERANCE)
            return int(simplex[0])
        return int(self.triangle_finder([x], [y])[0])

    def barycentric_weights(self, x_points, y_points):
        """Return simplex indices and barycentric weights for points."""
        points = _points_array(x_points, y_points)
        weights = np.zeros((points.shape[0], 3), dtype=np.float64)

        if self.triangulation is not None:
            simplices = self.triangulation.find_simplex(points, tol=TRIANGLE_TOLERANCE)
            valid = simplices >= 0
            if not np.any(valid):
                return simplices, weights

            transform = self.triangulation.transform[simplices[valid]]
            delta = points[valid] - transform[:, 2, :]
            first_two = np.einsum('ijk,ik->ij', transform[:, :2, :], delta)
            weights[valid, :2] = first_two
            weights[valid, 2] = 1.0 - first_two.sum(axis=1)
            return simplices, weights

        simplices = np.asarray(self.triangle_finder(points[:, 0], points[:, 1]), dtype=np.int64)
        valid_indices = np.flatnonzero(simplices >= 0)
        if valid_indices.size:
            vertices = self.triangles[simplices[valid_indices]]
            local_weights, nondegenerate = _triangle_barycentric_weights(
                self.grid_x,
                self.grid_y,
                vertices,
                points[valid_indices],
            )
            weights[valid_indices[nondegenerate]] = local_weights[nondegenerate]
            simplices[valid_indices[~nondegenerate]] = -1

        return simplices, weights

    def interpolate_field(self, field, x_points, y_points):
        """Barycentrically interpolate a nodal field at point coordinates."""
        values = np.asarray(field, dtype=np.float64).ravel()
        simplices, weights = self.barycentric_weights(x_points, y_points)
        out = np.zeros(len(simplices), dtype=np.float64)

        valid = simplices >= 0
        if np.any(valid):
            vertices = self.triangles[simplices[valid]]
            out[valid] = np.sum(values[vertices] * weights[valid], axis=1)

        return out

    def update_particles(self, x0, y0, grid_u, grid_v, dt, igeo=0):
        """Advance particle positions one RK4 step using barycentric velocity sampling."""
        x0 = np.asarray(x0, dtype=np.float64)
        y0 = np.asarray(y0, dtype=np.float64)
        if x0.size == 0:
            return x0.copy(), y0.copy()

        grid_u_adj, grid_v_adj = self._velocity_arrays(grid_u, grid_v, igeo)

        u1, v1 = self.interpolate_vector(grid_u_adj, grid_v_adj, x0, y0)
        x1 = x0 + 0.5 * u1 * dt
        y1 = y0 + 0.5 * v1 * dt

        u2, v2 = self.interpolate_vector(grid_u_adj, grid_v_adj, x1, y1)
        x2 = x0 + 0.5 * u2 * dt
        y2 = y0 + 0.5 * v2 * dt

        u3, v3 = self.interpolate_vector(grid_u_adj, grid_v_adj, x2, y2)
        x3 = x0 + u3 * dt
        y3 = y0 + v3 * dt

        u4, v4 = self.interpolate_vector(grid_u_adj, grid_v_adj, x3, y3)

        x_new = x0 + dt / 6.0 * (u1 + 2.0 * u2 + 2.0 * u3 + u4)
        y_new = y0 + dt / 6.0 * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
        return x_new, y_new

    def interpolate_vector(self, grid_u, grid_v, x_points, y_points):
        """Interpolate vector components at point coordinates."""
        return (
            self.interpolate_field(grid_u, x_points, y_points),
            self.interpolate_field(grid_v, x_points, y_points),
        )

    def _velocity_arrays(self, grid_u, grid_v, igeo):
        grid_u = np.asarray(grid_u, dtype=np.float64).ravel()
        grid_v = np.asarray(grid_v, dtype=np.float64).ravel()

        if igeo != 1:
            return grid_u, grid_v

        geofac = 6378137.0
        cos_lat = np.cos(np.deg2rad(self.grid_y))
        return grid_u / (geofac * cos_lat), grid_v / geofac


def _bounding_box(points):
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    min_x = float(np.min(points[:, 0]))
    max_x = float(np.max(points[:, 0]))
    min_y = float(np.min(points[:, 1]))
    max_y = float(np.max(points[:, 1]))
    return np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]], dtype=np.float64)


def _points_array(x_points, y_points):
    x = np.asarray(x_points, dtype=np.float64)
    y = np.asarray(y_points, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f'x_points and y_points must have the same shape, got {x.shape} and {y.shape}')
    return np.column_stack((x.ravel(), y.ravel()))


def _triangle_barycentric_weights(grid_x, grid_y, vertices, points):
    x0 = grid_x[vertices[:, 0]]
    y0 = grid_y[vertices[:, 0]]
    x1 = grid_x[vertices[:, 1]]
    y1 = grid_y[vertices[:, 1]]
    x2 = grid_x[vertices[:, 2]]
    y2 = grid_y[vertices[:, 2]]

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    nondegenerate = np.abs(denom) >= TRIANGLE_TOLERANCE

    weights = np.zeros((points.shape[0], 3), dtype=np.float64)
    if np.any(nondegenerate):
        x = points[nondegenerate, 0]
        y = points[nondegenerate, 1]
        denom_valid = denom[nondegenerate]
        weights[nondegenerate, 0] = (
            (y1[nondegenerate] - y2[nondegenerate]) * (x - x2[nondegenerate])
            + (x2[nondegenerate] - x1[nondegenerate]) * (y - y2[nondegenerate])
        ) / denom_valid
        weights[nondegenerate, 1] = (
            (y2[nondegenerate] - y0[nondegenerate]) * (x - x2[nondegenerate])
            + (x0[nondegenerate] - x2[nondegenerate]) * (y - y2[nondegenerate])
        ) / denom_valid
        weights[nondegenerate, 2] = 1.0 - weights[nondegenerate, 0] - weights[nondegenerate, 1]

    return weights, nondegenerate


def create_grid_geometry(grid_x, grid_y, triangles=None) -> GridGeometry:
    """Create cached grid geometry for repeated particle interpolation."""
    return GridGeometry.from_points(grid_x, grid_y, triangles=triangles)


def create_numba_particle_calculator(grid_x, grid_y, triangles=None, grid_geometry=None):
    """
    Create particle interpolation/update callables.

    The returned dictionary keeps the historical keys used by ParticlePopulation.
    """
    geometry = grid_geometry if grid_geometry is not None else create_grid_geometry(grid_x, grid_y, triangles)

    return {
        'geometry': geometry,
        'triangles': geometry.triangles,
        'find_triangle': geometry.find_triangle,
        'interpolate_field': geometry.interpolate_field,
        'update_particles': geometry.update_particles,
        'update_particles_parallel': geometry.update_particles,
    }


def find_triangle(x, y, grid_x, grid_y, triangles=None):
    """Compatibility wrapper for one-off triangle lookup."""
    return create_grid_geometry(grid_x, grid_y, triangles=triangles).find_triangle(x, y)


def interpolate_field(field, x_points, y_points, grid_x, grid_y, triangles=None):
    """Compatibility wrapper for one-off scalar interpolation."""
    return create_grid_geometry(grid_x, grid_y, triangles=triangles).interpolate_field(field, x_points, y_points)


def update_particles_rk4(x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo=0):
    """Compatibility wrapper for one-off RK4 particle updates."""
    return create_grid_geometry(grid_x, grid_y, triangles=triangles).update_particles(x0, y0, grid_u, grid_v, dt, igeo)


def update_particles_rk4_parallel(x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo=0):
    """Compatibility wrapper for the historical parallel update function."""
    return update_particles_rk4(x0, y0, grid_u, grid_v, grid_x, grid_y, triangles, dt, igeo)
