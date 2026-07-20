"""Coordinate transforms for metric particle geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from pyproj import CRS, Geod, Transformer

DEFAULT_SOURCE_CRS = 'EPSG:4326'
DEFAULT_METRIC_CRS = 'auto_utm'
DEFAULT_EARTH_RADIUS_M = 6_371_008.8
_GEOGRAPHIC_SYSTEMS = {'geographic', 'spherical', 'lonlat', 'latlon'}
_PROJECTED_SYSTEMS = {'projected', 'cartesian'}
_GEODETIC_RUNTIME_GEOMETRIES = {'geodetic', 'geodesic', 'spherical'}
_PLANAR_RUNTIME_GEOMETRIES = {'planar', 'local-projected', 'projected'}
_UTM_MIN_LAT = -80.0
_UTM_MAX_LAT = 84.0
_MAX_AUTO_UTM_LONGITUDE_OFFSET_DEGREES = 3.0


@dataclass
class CoordinateTransform:
    """Map model source coordinates to projected metric runtime coordinates.

    Parameters
    ----------
    coordinate_system : str
        Coordinate-system label. ``"projected"`` leaves coordinates
        unchanged. Geographic labels use ``pyproj`` to transform source
        longitude/latitude coordinates to a projected metric CRS.
    source_crs : str, default='EPSG:4326'
        CRS for geographic source coordinates.
    metric_crs : str, optional
        Projected CRS used for metric runtime coordinates. Required for
        geographic transforms and usually resolved to an EPSG UTM code by
        :func:`build_coordinate_transform`.
    runtime_geometry : {"planar", "geodetic"}, default="planar"
        Runtime horizontal geometry. Planar geographic runs use ``metric_crs``.
        Geodetic runs retain longitude/latitude at public boundaries and use
        intrinsic surface geometry in the particle backend.
    surface_model : {"from_crs", "sphere", "ellipsoid"}, default="sphere"
        Surface model used by geodetic particle geometry.
    earth_radius_m : float, default=6371008.8
        Sphere radius in metres when ``surface_model`` is ``"sphere"``.
    longitude_wrap : {"auto", "-180_180", "0_360"}, default="auto"
        Longitude convention used for serialized output.
    velocity_basis : {"auto", "east_north", "source_xy", "grid_aligned"}
        Basis of horizontal vector components supplied by the input model.
    """

    coordinate_system: str = 'projected'
    source_crs: str | CRS | None = DEFAULT_SOURCE_CRS
    metric_crs: str | CRS | None = None
    runtime_geometry: str = 'planar'
    surface_model: str = 'sphere'
    earth_radius_m: float = DEFAULT_EARTH_RADIUS_M
    longitude_wrap: str = 'auto'
    velocity_basis: str = 'auto'
    _source_crs_obj: CRS | None = field(init=False, default=None, repr=False)
    _metric_crs_obj: CRS | None = field(init=False, default=None, repr=False)
    _forward_transformer: Transformer | None = field(init=False, default=None, repr=False)
    _inverse_transformer: Transformer | None = field(init=False, default=None, repr=False)
    _source_geod: Geod | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.coordinate_system = _normalize_coordinate_system(self.coordinate_system)
        self.runtime_geometry = _normalize_runtime_geometry(self.runtime_geometry)
        self.surface_model = _normalize_surface_model(self.surface_model)
        self.longitude_wrap = _normalize_longitude_wrap(self.longitude_wrap)
        self.velocity_basis = _normalize_velocity_basis(self.velocity_basis)
        self.earth_radius_m = _validate_earth_radius(self.earth_radius_m)
        if not self.is_geographic:
            if self.runtime_geometry == 'geodetic':
                raise ValueError('runtime_geometry="geodetic" requires a geographic coordinate system.')
            self.source_crs = None
            self.metric_crs = None
            return

        self._source_crs_obj = CRS.from_user_input(self.source_crs or DEFAULT_SOURCE_CRS)
        if not self._source_crs_obj.is_geographic:
            raise ValueError(
                f'source_crs must be geographic for geographic input, got {self._source_crs_obj.to_string()!r}.'
            )
        self.source_crs = _crs_label(self._source_crs_obj)
        self._source_geod = self._source_crs_obj.get_geod()

        if self.is_geodetic:
            if self.metric_crs not in (None, ''):
                raise ValueError('metric_crs is not used with runtime_geometry="geodetic".')
            self.metric_crs = None
            return

        if self.metric_crs is None:
            raise ValueError('metric_crs is required for planar geographic coordinate transforms.')
        self._metric_crs_obj = CRS.from_user_input(self.metric_crs)
        _validate_metric_crs(self._metric_crs_obj)
        self.metric_crs = _crs_label(self._metric_crs_obj)
        self._forward_transformer = Transformer.from_crs(
            self._source_crs_obj,
            self._metric_crs_obj,
            always_xy=True,
        )
        self._inverse_transformer = Transformer.from_crs(
            self._metric_crs_obj,
            self._source_crs_obj,
            always_xy=True,
        )

    @property
    def is_geographic(self) -> bool:
        """Return whether this transform maps lon/lat source coordinates to metres."""
        return _normalize_coordinate_system(self.coordinate_system) in _GEOGRAPHIC_SYSTEMS

    @property
    def is_geodetic(self) -> bool:
        """Return whether intrinsic surface geometry is used at runtime."""
        return self.is_geographic and self.runtime_geometry == 'geodetic'

    def source_to_metric(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Convert source coordinates to metric runtime coordinates.

        Parameters
        ----------
        x, y : array-like
            Source coordinates. Geographic inputs are longitude/latitude in
            ``source_crs`` coordinates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Metric x/y arrays with the same shapes as the inputs.
        """
        x_array = np.asarray(x, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)
        if not self.is_geographic or self.is_geodetic:
            return x_array, y_array

        if x_array.size == 1:
            metric_x, metric_y = self._forward_transformer.transform(
                float(x_array.ravel()[0]),
                float(y_array.ravel()[0]),
            )
            return np.asarray(metric_x, dtype=np.float64).reshape(x_array.shape), np.asarray(
                metric_y,
                dtype=np.float64,
            ).reshape(y_array.shape)

        metric_x, metric_y = self._forward_transformer.transform(x_array, y_array)
        return np.asarray(metric_x, dtype=np.float64).reshape(x_array.shape), np.asarray(
            metric_y,
            dtype=np.float64,
        ).reshape(y_array.shape)

    def metric_to_source(self, metric_x, metric_y) -> tuple[np.ndarray, np.ndarray]:
        """Convert metric runtime coordinates back to source coordinates.

        Parameters
        ----------
        metric_x, metric_y : array-like
            Runtime coordinates in ``metric_crs`` metres for geographic grids,
            or unchanged source-coordinate values for projected grids.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Source-coordinate arrays with the same shapes as the inputs.
        """
        x_array = np.asarray(metric_x, dtype=np.float64)
        y_array = np.asarray(metric_y, dtype=np.float64)
        if not self.is_geographic or self.is_geodetic:
            return x_array, y_array

        if x_array.size == 1:
            source_x, source_y = self._inverse_transformer.transform(
                float(x_array.ravel()[0]),
                float(y_array.ravel()[0]),
            )
            return np.asarray(source_x, dtype=np.float64).reshape(x_array.shape), np.asarray(
                source_y,
                dtype=np.float64,
            ).reshape(y_array.shape)

        source_x, source_y = self._inverse_transformer.transform(x_array, y_array)
        return np.asarray(source_x, dtype=np.float64).reshape(x_array.shape), np.asarray(
            source_y,
            dtype=np.float64,
        ).reshape(y_array.shape)

    def polygons_to_metric(self, polygons: Iterable[np.ndarray]) -> list[np.ndarray]:
        """Return polygon coordinate arrays transformed to metric coordinates.

        Parameters
        ----------
        polygons : iterable of np.ndarray
            Polygon vertex arrays with at least two columns.

        Returns
        -------
        list[np.ndarray]
            Polygon arrays in metric coordinates. Extra columns are preserved.
        """
        metric_polygons: list[np.ndarray] = []
        for polygon in polygons:
            polygon_array = np.asarray(polygon, dtype=float)
            if polygon_array.ndim != 2 or polygon_array.shape[1] < 2:
                metric_polygons.append(polygon_array)
                continue
            metric_x, metric_y = self.source_to_metric(polygon_array[:, 0], polygon_array[:, 1])
            transformed = polygon_array.copy()
            transformed[:, 0] = metric_x
            transformed[:, 1] = metric_y
            metric_polygons.append(transformed)
        return metric_polygons

    def metric_velocity_basis(
        self,
        x,
        y,
        *,
        metric_x=None,
        metric_y=None,
        displacement: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return projected basis vectors for east/north velocity components.

        Parameters
        ----------
        x, y : array-like
            Source coordinates for the basis calculation. Geographic inputs
            are longitude/latitude in ``source_crs`` coordinates.
        metric_x, metric_y : array-like, optional
            Precomputed projected coordinates for ``x`` and ``y``. Supplying
            these avoids one repeated projection.
        displacement : float, default=1.0
            Geodesic displacement in metres used for finite-difference basis
            vectors.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``east_x, east_y, north_x, north_y`` arrays. Multiplying eastward
            and northward velocities by these arrays maps them to the projected
            metric x/y runtime basis.
        """
        x_array = np.asarray(x, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)
        if y_array.shape != x_array.shape:
            raise ValueError(f'x and y must have the same shape, got {x_array.shape} and {y_array.shape}')
        if not self.is_geographic:
            ones = np.ones_like(x_array, dtype=np.float64)
            zeros = np.zeros_like(x_array, dtype=np.float64)
            return ones, zeros, zeros, ones
        if self.is_geodetic:
            raise ValueError('Projected velocity basis vectors are not defined for geodetic runtime geometry.')

        if displacement <= 0.0:
            raise ValueError('displacement must be positive.')

        if metric_x is None or metric_y is None:
            base_x, base_y = self.source_to_metric(x_array, y_array)
        else:
            base_x = np.asarray(metric_x, dtype=np.float64).reshape(x_array.shape)
            base_y = np.asarray(metric_y, dtype=np.float64).reshape(y_array.shape)

        distance = np.full(x_array.shape, float(displacement), dtype=np.float64)
        east_lon, east_lat, _ = self._source_geod.fwd(
            x_array,
            y_array,
            np.full(x_array.shape, 90.0, dtype=np.float64),
            distance,
        )
        north_lon, north_lat, _ = self._source_geod.fwd(
            x_array,
            y_array,
            np.zeros(x_array.shape, dtype=np.float64),
            distance,
        )
        east_x, east_y = self.source_to_metric(east_lon, east_lat)
        north_x, north_y = self.source_to_metric(north_lon, north_lat)
        return (
            (east_x - base_x) / displacement,
            (east_y - base_y) / displacement,
            (north_x - base_x) / displacement,
            (north_y - base_y) / displacement,
        )

    def metadata(self) -> dict[str, float | int | str]:
        """Return serializable metadata for this transform."""
        if not self.is_geographic:
            return {
                'coordinate_metadata_version': 2,
                'coordinate_system': 'projected',
                'source_coordinate_system': 'projected',
                'runtime_geometry': 'planar',
                'runtime_coordinate_system': 'source',
                'metric_coordinate_system': 'source',
                'horizontal_distance_units': 'm',
            }

        metadata: dict[str, float | int | str] = {
            'coordinate_metadata_version': 2,
            'coordinate_system': 'geographic',
            'source_coordinate_system': 'geographic',
            'runtime_geometry': self.runtime_geometry,
            'runtime_coordinate_system': 'geodetic_surface' if self.is_geodetic else 'metric_projected',
            'source_crs': str(self.source_crs),
            'surface_model': self.surface_model,
            'earth_radius_m': self.earth_radius_m,
            'longitude_wrap': self.longitude_wrap,
            'velocity_basis': self.velocity_basis,
            'horizontal_distance_units': 'm',
        }
        if self.is_geodetic:
            metadata['metric_coordinate_system'] = 'geodetic_surface'
            return metadata

        metadata['metric_crs'] = str(self.metric_crs)
        metadata['metric_coordinate_system'] = _metric_coordinate_system_label(self._metric_crs_obj)
        utm_zone = _utm_zone_from_crs(self._metric_crs_obj)
        if utm_zone is not None:
            metadata['utm_zone'] = utm_zone
            metadata['utm_hemisphere'] = 'north' if _is_northern_utm(self._metric_crs_obj) else 'south'
        return metadata


def build_coordinate_transform(
    x,
    y,
    coordinate_system: str | None = None,
    *,
    source_crs: str | CRS | None = DEFAULT_SOURCE_CRS,
    metric_crs: str | CRS | None = DEFAULT_METRIC_CRS,
    runtime_geometry: str | None = 'planar',
    surface_model: str = 'sphere',
    earth_radius_m: float = DEFAULT_EARTH_RADIUS_M,
    longitude_wrap: str = 'auto',
    velocity_basis: str = 'auto',
) -> CoordinateTransform:
    """Build a coordinate transform for a grid.

    Parameters
    ----------
    x, y : array-like
        Source-coordinate arrays used to choose an automatic UTM CRS when
        ``metric_crs`` is ``"auto_utm"``.
    coordinate_system : str, optional
        Coordinate-system label. Missing values default to ``"projected"``.
    source_crs : str or pyproj.CRS, optional
        CRS for geographic source coordinates. Defaults to WGS84 lon/lat.
    metric_crs : str or pyproj.CRS, optional
        Projected metric CRS. Use ``"auto_utm"`` to infer from grid centre.
    runtime_geometry : {"auto", "planar", "geodetic"}, optional
        Runtime geometry. ``"auto"`` attempts a valid regional projection and
        falls back to intrinsic geodetic geometry for ocean-scale domains.

    Returns
    -------
    CoordinateTransform
        Transform object suitable for repeated vectorized coordinate mapping.
    """
    normalized = _normalize_coordinate_system(coordinate_system)
    if normalized not in _GEOGRAPHIC_SYSTEMS:
        return CoordinateTransform('projected')

    geometry = _normalize_runtime_geometry(runtime_geometry)
    if geometry == 'auto':
        try:
            resolved_metric_crs = _resolve_metric_crs(x, y, metric_crs)
        except ValueError:
            geometry = 'geodetic'
            resolved_metric_crs = None
        else:
            geometry = 'planar'
    elif geometry == 'geodetic':
        resolved_metric_crs = None
    else:
        resolved_metric_crs = _resolve_metric_crs(x, y, metric_crs)

    return CoordinateTransform(
        'geographic',
        source_crs or DEFAULT_SOURCE_CRS,
        resolved_metric_crs,
        runtime_geometry=geometry,
        surface_model=surface_model,
        earth_radius_m=earth_radius_m,
        longitude_wrap=longitude_wrap,
        velocity_basis=velocity_basis,
    )


def coordinate_transform_from_metadata(metadata) -> CoordinateTransform:
    """Build a coordinate transform from stored coordinate metadata.

    Parameters
    ----------
    metadata : mapping or object
        Coordinate metadata containing ``coordinate_system`` and, for
        geographic runs, ``source_crs`` and ``metric_crs``.

    Returns
    -------
    CoordinateTransform
        Transform object for output or restart coordinate conversion.
    """
    coordinate_system = coordinate_system_from_metadata(metadata)
    if coordinate_system != 'geographic':
        return CoordinateTransform('projected')
    source_crs = _metadata_value(metadata, 'source_crs', DEFAULT_SOURCE_CRS)
    runtime_geometry = _metadata_value(metadata, 'runtime_geometry', 'planar')
    metric_crs = _metadata_value(metadata, 'metric_crs', None)
    if _normalize_runtime_geometry(runtime_geometry) == 'planar' and metric_crs is None:
        raise ValueError('Geographic coordinate metadata must include metric_crs.')
    return CoordinateTransform(
        'geographic',
        source_crs,
        metric_crs,
        runtime_geometry=runtime_geometry,
        surface_model=_metadata_value(metadata, 'surface_model', 'sphere'),
        earth_radius_m=_metadata_value(metadata, 'earth_radius_m', DEFAULT_EARTH_RADIUS_M),
        longitude_wrap=_metadata_value(metadata, 'longitude_wrap', 'auto'),
        velocity_basis=_metadata_value(metadata, 'velocity_basis', 'auto'),
    )


def coordinate_system_from_metadata(metadata, default: str = 'projected') -> str:
    """Return a normalized coordinate-system label from metadata.

    Parameters
    ----------
    metadata : object
        Metadata object with optional ``coordinate_system`` attribute or
        ``get`` method.
    default : str, default='projected'
        Fallback when no metadata label exists.

    Returns
    -------
    str
        Normalized coordinate-system label.
    """
    return _normalize_coordinate_system(_metadata_value(metadata, 'coordinate_system', default))


def source_crs_from_metadata(metadata, default: str = DEFAULT_SOURCE_CRS) -> str:
    """Return the configured source CRS label from metadata."""
    return str(_metadata_value(metadata, 'source_crs', default))


def metric_crs_from_metadata(metadata, default: str = DEFAULT_METRIC_CRS) -> str:
    """Return the configured metric CRS label from metadata."""
    return str(_metadata_value(metadata, 'metric_crs', default))


def infer_coordinate_system_from_attrs(x_var=None, y_var=None) -> str:
    """Infer a coordinate system from NetCDF coordinate metadata.

    Parameters
    ----------
    x_var, y_var : object, optional
        xarray/netCDF-like variables with ``attrs`` dictionaries.

    Returns
    -------
    str
        ``"geographic"`` when longitude/latitude units or standard names are
        detected, otherwise ``"projected"``.
    """
    x_text = _metadata_text(x_var)
    y_text = _metadata_text(y_var)
    if _looks_longitude(x_text) and _looks_latitude(y_text):
        return 'geographic'
    return 'projected'


def _resolve_metric_crs(x, y, metric_crs: str | CRS | None) -> str | CRS:
    if metric_crs is None or str(metric_crs).strip().lower() in {'', 'auto', 'auto_utm', 'auto-utm'}:
        return _auto_utm_crs(x, y)
    return metric_crs


def _auto_utm_crs(x, y) -> str:
    lon, lat = _finite_lon_lat(x, y)
    if lon.size == 0:
        raise ValueError('Cannot infer auto_utm CRS without finite longitude/latitude coordinates.')

    if np.nanmin(lat) < _UTM_MIN_LAT or np.nanmax(lat) > _UTM_MAX_LAT:
        raise ValueError(
            f'Cannot infer UTM CRS for latitudes outside [{_UTM_MIN_LAT}, {_UTM_MAX_LAT}] degrees. '
            'Configure general.input_model.metric_crs explicitly.'
        )

    lon_span = float(np.nanmax(lon) - np.nanmin(lon))
    if lon_span > 180.0:
        raise ValueError('Cannot infer auto_utm CRS for antimeridian-crossing grids. Configure metric_crs explicitly.')

    mean_lon = float(np.nanmean(lon))
    mean_lat = float(np.nanmean(lat))
    zone = _utm_zone(mean_lon)
    central_meridian = -183.0 + 6.0 * zone
    max_offset = float(np.nanmax(np.abs(lon - central_meridian)))
    if max_offset > _MAX_AUTO_UTM_LONGITUDE_OFFSET_DEGREES:
        raise ValueError(
            f'Grid spans beyond UTM zone {zone} bounds for auto_utm '
            f'(max longitude offset {max_offset:.3f} deg). Configure metric_crs explicitly.'
        )

    epsg = (32600 if mean_lat >= 0.0 else 32700) + zone
    return f'EPSG:{epsg}'


def _finite_lon_lat(x, y) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(x, dtype=np.float64).ravel()
    lat = np.asarray(y, dtype=np.float64).ravel()
    if lon.shape != lat.shape:
        raise ValueError(f'x and y must have the same shape, got {lon.shape} and {lat.shape}')
    finite = np.isfinite(lon) & np.isfinite(lat)
    return lon[finite], lat[finite]


def _utm_zone(longitude: float) -> int:
    zone = int(np.floor((longitude + 180.0) / 6.0)) + 1
    return max(1, min(60, zone))


def _normalize_coordinate_system(value: str | None) -> str:
    if value is None:
        return 'projected'
    normalized = str(value).strip().lower().replace('_', '-')
    if normalized in {'geo', 'geographic', 'spherical', 'lon-lat', 'lonlat', 'latlon', 'latitude-longitude'}:
        return 'geographic'
    if normalized in _PROJECTED_SYSTEMS:
        return 'projected'
    raise ValueError(
        f'Unknown coordinate_system {value!r}; expected "projected", "geographic", or a supported alias.'
    )


def _normalize_runtime_geometry(value: str | None) -> str:
    if value is None:
        return 'planar'
    normalized = str(value).strip().lower().replace('_', '-')
    if normalized == 'auto':
        return 'auto'
    if normalized in _GEODETIC_RUNTIME_GEOMETRIES:
        return 'geodetic'
    if normalized in _PLANAR_RUNTIME_GEOMETRIES:
        return 'planar'
    raise ValueError(
        f'Unknown runtime_geometry {value!r}; expected "auto", "planar", or "geodetic".'
    )


def _normalize_surface_model(value: str | None) -> str:
    normalized = 'from_crs' if value is None else str(value).strip().lower().replace('-', '_')
    if normalized not in {'from_crs', 'sphere', 'ellipsoid'}:
        raise ValueError(
            f'Unknown surface_model {value!r}; expected "from_crs", "sphere", or "ellipsoid".'
        )
    return normalized


def _normalize_longitude_wrap(value: str | None) -> str:
    normalized = 'auto' if value is None else str(value).strip().lower().replace('-', '_')
    aliases = {
        '180_180': '-180_180',
        '_180_180': '-180_180',
        'minus180_180': '-180_180',
        '0_360': '0_360',
        'auto': 'auto',
    }
    if normalized in aliases:
        return aliases[normalized]
    if str(value).strip() == '-180_180':
        return '-180_180'
    raise ValueError(
        f'Unknown longitude_wrap {value!r}; expected "auto", "-180_180", or "0_360".'
    )


def _normalize_velocity_basis(value: str | None) -> str:
    normalized = 'auto' if value is None else str(value).strip().lower().replace('-', '_')
    if normalized not in {'auto', 'east_north', 'source_xy', 'grid_aligned'}:
        raise ValueError(
            f'Unknown velocity_basis {value!r}; expected "auto", "east_north", "source_xy", or "grid_aligned".'
        )
    return normalized


def _validate_earth_radius(value: float) -> float:
    radius = float(value)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError('earth_radius_m must be a finite positive number.')
    return radius


def _validate_metric_crs(crs: CRS) -> None:
    if not crs.is_projected:
        raise ValueError(f'metric_crs must be projected, got {crs.to_string()!r}.')
    axes = tuple(crs.axis_info[:2])
    if len(axes) < 2:
        raise ValueError(f'metric_crs must define two horizontal axes, got {crs.to_string()!r}.')
    invalid_units = [
        axis.unit_name
        for axis in axes
        if axis.unit_conversion_factor is None or not np.isclose(float(axis.unit_conversion_factor), 1.0)
    ]
    if invalid_units:
        raise ValueError(
            f'metric_crs horizontal axes must use metres, got units {invalid_units!r} for {crs.to_string()!r}.'
        )


def _metadata_value(metadata, key: str, default: Any = None) -> Any:
    if metadata is None:
        return default
    if hasattr(metadata, 'get'):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def _metadata_text(var) -> str:
    attrs = getattr(var, 'attrs', {}) or {}
    pieces = [str(attrs.get(name, '')) for name in ('units', 'standard_name', 'long_name', 'axis') if name in attrs]
    return ' '.join(pieces).lower()


def _looks_longitude(text: str) -> bool:
    return 'degree_east' in text or 'degrees_east' in text or 'longitude' in text


def _looks_latitude(text: str) -> bool:
    return 'degree_north' in text or 'degrees_north' in text or 'latitude' in text


def _crs_label(crs: CRS) -> str:
    authority = crs.to_authority()
    if authority is not None:
        return f'{authority[0]}:{authority[1]}'
    return crs.to_string()


def _metric_coordinate_system_label(crs: CRS | None) -> str:
    if _utm_zone_from_crs(crs) is not None:
        return 'utm'
    return 'projected_crs'


def _utm_zone_from_crs(crs: CRS | None) -> int | None:
    if crs is None:
        return None
    authority = crs.to_authority()
    if authority is None or authority[0].upper() != 'EPSG':
        return None
    epsg = int(authority[1])
    if 32601 <= epsg <= 32660:
        return epsg - 32600
    if 32701 <= epsg <= 32760:
        return epsg - 32700
    return None


def _is_northern_utm(crs: CRS | None) -> bool:
    authority = crs.to_authority() if crs is not None else None
    if authority is None or authority[0].upper() != 'EPSG':
        return True
    return 32601 <= int(authority[1]) <= 32660
