"""
Memory Manager
==============
Manages memory allocation and deallocation for the simulation data buffer.
"""

import sys


class MemoryManager:
    """
    Manages memory usage for the simulation data buffer. If the buffer exceeds the memory limit,
    writes its contents to a NetCDF file and clears the buffer.

    The current memory limit is set to 512 MB

    Attributes
    ----------
    max_bytes : int
        Maximum allowed memory usage in bytes.
    writer : NetCDFWriter
        Writer instance to write buffer to NetCDF when limit is exceeded.
    file_counter : int
        Counter for naming output files uniquely.
    """

    def __init__(self, max_bytes=512 * 1024 * 1024):
        """
        Initialize the memory manager.

        Parameters
        ----------
        max_bytes : int
            Maximum allowed memory usage in bytes (default: 512 MB).
        """
        self.max_bytes = max_bytes
        self.file_counter = 0

    def buffer_size_bytes(self, buffer):
        """
        Estimate the memory usage of the buffer in bytes.

        Parameters
        ----------
        buffer : dict
            The simulation data buffer (dict of lists or numpy arrays).

        Returns
        -------
        int
            Estimated memory usage in bytes.
        """
        size = 0
        for v in buffer.values():
            if hasattr(v, 'nbytes'):
                size += v.nbytes
            else:
                size += sys.getsizeof(v)
                if hasattr(v, '__iter__') and not isinstance(v, (str, bytes)):
                    size += sum(sys.getsizeof(item) for item in v)
        return size

    def is_limit_exceeded(self, buffer):
        """
        Check if the buffer exceeds the memory limit.

        Parameters
        ----------
        buffer : dict
            The simulation data buffer (dict of lists or numpy arrays).

        Returns
        -------
        bool
            True if buffer size exceeds max_bytes, False otherwise.
        """
        return self.buffer_size_bytes(buffer) > self.max_bytes
