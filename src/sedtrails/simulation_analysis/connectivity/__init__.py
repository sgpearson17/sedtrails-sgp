"""
SedTrails Connectivity Analysis Module

This module provides tools for analyzing particle connectivity using network analysis.
Translates functionality from MATLAB Brain Connectivity Toolbox to Python NetworkX.

Key components:
- Adjacency matrix compilation from trajectory data
- Connectivity polygon generation
- Network analysis and visualization
- Time-varying connectivity analysis
"""

from .adjacency_matrix import AdjacencyMatrixCompiler
from .connectivity_polygons import ConnectivityPolygonGenerator
from .network_analysis import NetworkAnalyzer
from .connectivity_io import ConnectivityIO

__all__ = [
    'AdjacencyMatrixCompiler',
    'ConnectivityPolygonGenerator', 
    'NetworkAnalyzer',
    'ConnectivityIO'
]
