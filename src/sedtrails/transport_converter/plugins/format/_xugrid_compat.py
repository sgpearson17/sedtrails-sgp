"""Compatibility helpers for supported xugrid releases."""

import inspect

import numpy as np
import xugrid as xu


def create_ugrid2d(
    node_x: np.ndarray,
    node_y: np.ndarray,
    face_node_connectivity: np.ndarray,
    *,
    is_projected: bool,
) -> xu.Ugrid2d:
    """Create a Ugrid2d instance across xugrid projection-argument renames.

    Parameters
    ----------
    node_x, node_y : np.ndarray
        Coordinates of the source mesh nodes.
    face_node_connectivity : np.ndarray
        Zero-based face-to-node connectivity, with ``-1`` as the fill value.
    is_projected : bool
        Whether the coordinates use a projected coordinate reference system.

    Returns
    -------
    xugrid.Ugrid2d
        Grid constructed with the projection keyword accepted by the installed
        xugrid release.
    """
    parameters = inspect.signature(xu.Ugrid2d).parameters
    projection_keyword = 'is_projected' if 'is_projected' in parameters else 'projected'
    return xu.Ugrid2d(
        node_x,
        node_y,
        -1,
        face_node_connectivity,
        **{projection_keyword: is_projected},
    )
