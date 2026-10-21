"""
A common interface for format plugins.
Plugins must inherit from this class and implement the convert method.
"""

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any

import numpy as np

from sedtrails.particle_tracer.coordinate_transform import build_coordinate_transform
from sedtrails.transport_converter.domain_mask import (
    ConnectivityMaskResult,
    classify_boundary_edges_from_config,
    domain_point_mask,
    filter_connectivity_by_domain,
    inner_boundary_files_from_config,
    load_inner_boundary_polygons,
)
from sedtrails.transport_converter.tekal import read_tekal_polygons
from sedtrails.transport_converter.sedtrails_data import SedtrailsData
from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata


DomainPolygons = tuple[
    list[np.ndarray], list[np.ndarray], tuple[float, float, float, float] | None,
]


class BaseFormatPlugin(ABC):
    """
    Abstract base class for format plugins.
    """

    def __init__(self) -> None:
        self.domain_config: dict[str, Any] = {}
        self._domain_polygons_cache: tuple[str, DomainPolygons] | None = None
        self._boundary_edge_classification_cache: dict[str, Any] | None = None
        self._last_inner_boundary_mask: ConnectivityMaskResult | SimpleNamespace | None = None

    def _domain_config_signature(self) -> str:
        return repr((
            self.domain_config,
            getattr(self, 'runtime_geometry', None) or 'planar',
            getattr(self, 'source_crs', None),
            getattr(self, 'metric_crs', None),
            self._coordinate_system(),
        ))

    def _coordinate_system(self) -> str:
        return getattr(self, 'coordinate_system', None) or 'projected'

    def _domain_polygons(self) -> DomainPolygons:
        """Load model-independent extent and cutout definitions once per config."""
        signature = self._domain_config_signature()
        cache = self._domain_polygons_cache
        if cache is not None and cache[0] == signature:
            return cache[1]
        config = self.domain_config
        has_subset = 'subset_x' in config or 'subset_y' in config
        if config.get('pol_file') and has_subset:
            raise ValueError('Use domain.pol_file or domain.subset_x/subset_y, not both')
        bounds = None
        if has_subset:
            limits = []
            for name in ('subset_x', 'subset_y'):
                try:
                    lower, upper = (float(value) for value in config[name].split(':'))
                except (KeyError, AttributeError, TypeError, ValueError) as error:
                    raise ValueError(f'domain.{name} must be a finite minimum:maximum range') from error
                if not np.isfinite([lower, upper]).all() or lower >= upper:
                    raise ValueError(f'domain.{name} must have finite minimum < maximum')
                limits.extend((lower, upper))
            bounds = (limits[0], limits[1], limits[2], limits[3])
        inner = load_inner_boundary_polygons(config)
        outer = read_tekal_polygons(config.get('pol_file'))
        if config.get('pol_file') and not outer:
            raise ValueError('domain.pol_file must contain at least one polygon')
        self._domain_polygons_cache = (signature, (inner, outer, bounds))
        return inner, outer, bounds

    def _get_inner_boundary_polygons(self) -> list[np.ndarray]:
        return self._domain_polygons()[0]

    def _domain_coordinate_options(self) -> dict[str, Any]:
        return {
            'coordinate_system': self._coordinate_system(),
            'source_crs': getattr(self, 'source_crs', None),
            'metric_crs': getattr(self, 'metric_crs', None),
            'runtime_geometry': getattr(self, 'runtime_geometry', None) or 'planar',
        }

    def _filter_domain_connectivity(
        self, x: np.ndarray, y: np.ndarray, connectivity: np.ndarray,
    ) -> ConnectivityMaskResult:
        """Apply the same domain mask to source topology in seeding and conversion."""
        inner, outer, bounds = self._domain_polygons()
        result = filter_connectivity_by_domain(
            x, y, connectivity, inner, outer, bounds, **self._domain_coordinate_options(),
        )
        if len(connectivity) and not result.active_count:
            raise ValueError('All particle-tracking faces were masked by domain configuration')
        self._last_inner_boundary_mask = result
        return result

    def _domain_point_mask(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mask source face centers before model-specific field compaction."""
        inner, outer, bounds = self._domain_polygons()
        if not inner and not outer and bounds is None:
            return np.ones(np.size(x), dtype=bool), np.zeros(np.size(x), dtype=bool)
        transform = build_coordinate_transform(x, y, **self._domain_coordinate_options())
        return domain_point_mask(
            np.column_stack((x, y)), inner, outer, bounds, transform=transform,
        )

    def _add_inner_boundary_metadata(self, metadata: SedtrailsMetadata) -> None:
        inner = self._get_inner_boundary_polygons()
        if not inner:
            return
        metadata.add('inner_boundary_pol_files', inner_boundary_files_from_config(self.domain_config))
        metadata.add('inner_boundary_polygon_count', len(inner))
        result = self._last_inner_boundary_mask
        if result is not None:
            metadata.add('inner_boundary_masked_face_count', result.inner_removed_count)
            metadata.add('inner_boundary_active_face_count', result.active_count)

    def _add_boundary_edge_metadata(
        self, metadata: SedtrailsMetadata, x: np.ndarray, y: np.ndarray,
        connectivity: np.ndarray | None,
    ) -> None:
        classification = self._boundary_edge_classification(x, y, connectivity)
        if classification is not None:
            metadata.add('boundary_edge_classification', classification)

    def _boundary_edge_classification(
        self, x: np.ndarray, y: np.ndarray, connectivity: np.ndarray | None,
    ) -> dict | None:
        if connectivity is None:
            return None
        cache = self._boundary_edge_classification_cache
        if cache is not None and self._geometry_cache_matches(cache, x, y, connectivity):
            return cache['metadata']
        result = classify_boundary_edges_from_config(
            x, y, connectivity, self.domain_config, **self._domain_coordinate_options(),
        )
        metadata = None if result is None else result.to_metadata()
        self._boundary_edge_classification_cache = {
            'node_x': np.asarray(x), 'node_y': np.asarray(y),
            'connectivity': np.asarray(connectivity),
            'domain_signature': self._domain_config_signature(),
            'coordinate_system': self._coordinate_system(),
            'source_crs': getattr(self, 'source_crs', None),
            'metric_crs': getattr(self, 'metric_crs', None),
            'metadata': metadata,
        }
        return metadata

    def _geometry_cache_matches(
        self, cache: dict[str, Any] | None, node_x: np.ndarray, node_y: np.ndarray,
        connectivity: np.ndarray | None = None,
    ) -> bool:
        if cache is None or cache.get('domain_signature') != self._domain_config_signature():
            return False
        if cache.get('coordinate_system') != self._coordinate_system():
            return False
        if (
            cache.get('source_crs') != getattr(self, 'source_crs', None)
            or cache.get('metric_crs') != getattr(self, 'metric_crs', None)
        ):
            return False
        if not self._arrays_equal(cache.get('node_x'), node_x) or not self._arrays_equal(cache.get('node_y'), node_y):
            return False
        return connectivity is None or self._arrays_equal(cache.get('connectivity'), connectivity)

    @staticmethod
    def _arrays_equal(left: np.ndarray | None, right: np.ndarray) -> bool:
        return left is not None and left.shape == np.shape(right) and np.array_equal(left, right)

    @abstractmethod
    def convert(self, *args, **kwargs) -> SedtrailsData:
        """
        Converts flow-field data from varios formats to the sedtrails format.

        Returns
        -------
        SedtrailsData
            The converted SedtrailsData object.

        Parameters
        ----------
        *args : object
            Additional positional arguments passed through to the implementation.
        **kwargs : object
            Additional keyword arguments passed through to the implementation.
        """
        pass

    def _add_runtime_coordinate_metadata(self, metadata) -> None:
        """Add explicitly configured runtime geometry metadata.

        Parameters
        ----------
        metadata : object
            SedTRAILS metadata container exposing ``add(name, value)``.
        """
        for name in (
            'runtime_geometry',
            'surface_model',
            'earth_radius_m',
            'longitude_wrap',
            'velocity_basis',
        ):
            value = getattr(self, name, None)
            if value is not None:
                metadata.add(name, value)
