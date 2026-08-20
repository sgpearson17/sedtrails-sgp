"""Bounded-memory helpers for particle-sized operations."""

from __future__ import annotations

PARTICLE_OPERATION_CHUNK_SIZE = 65_536


def particle_chunk_slices(size: int):
    """Yield contiguous slices bounded by the particle operation chunk size.

    Parameters
    ----------
    size : int
        Number of particles or points to process.

    Yields
    ------
    slice
        A contiguous slice with at most the configured chunk size.
    """
    if size < 0:
        raise ValueError('size must be non-negative.')
    for start in range(0, size, PARTICLE_OPERATION_CHUNK_SIZE):
        yield slice(start, min(start + PARTICLE_OPERATION_CHUNK_SIZE, size))
