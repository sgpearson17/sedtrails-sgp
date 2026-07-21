import numpy as np
import pytest

import sedtrails.transport_converter.sedtrails_data as sedtrails_data_module
from sedtrails.transport_converter.sedtrails_data import SedtrailsData
from sedtrails.transport_converter.sedtrails_metadata import SedtrailsMetadata


def _build_sedtrails_data(
    x,
    y,
    coordinate_system=None,
    particle_face_connectivity=None,
    face_node_connectivity=None,
    **coordinate_metadata,
):
    """Builds a minimal SedtrailsData instance for metadata resolution tests."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    npoints = x.size
    ntimes = 2

    zeros_time_space = np.zeros((ntimes, npoints), dtype=float)
    zero_vector = {
        'x': zeros_time_space.copy(),
        'y': zeros_time_space.copy(),
        'magnitude': zeros_time_space.copy(),
    }

    metadata = SedtrailsMetadata(
        flowfield_domain={
            'x_min': float(np.min(x)),
            'x_max': float(np.max(x)),
            'y_min': float(np.min(y)),
            'y_max': float(np.max(y)),
        }
    )
    if coordinate_system is not None:
        metadata.add('coordinate_system', coordinate_system)
    for key, value in coordinate_metadata.items():
        metadata.add(key, value)

    # Keep all dynamic fields simple/zeroed so tests isolate spatial metadata behavior.
    return SedtrailsData(
        times=np.array([0.0, 1.0], dtype=float),
        reference_date=np.datetime64('1970-01-01T00:00:00'),
        x=x,
        y=y,
        bed_level=np.zeros(npoints, dtype=float),
        depth_avg_flow_velocity=zero_vector,
        fractions=1,
        bed_load_transport=zero_vector,
        suspended_transport=zero_vector,
        water_depth=zeros_time_space.copy(),
        mean_bed_shear_stress=zeros_time_space.copy(),
        max_bed_shear_stress=zeros_time_space.copy(),
        sediment_concentration=zeros_time_space.copy(),
        nonlinear_wave_velocity=zero_vector,
        metadata=metadata,
        particle_face_connectivity=particle_face_connectivity,
        face_node_connectivity=face_node_connectivity,
    )


@pytest.fixture
def isolated_grid_metadata_cache():
    """Provide an empty reusable grid-metadata cache for one test."""
    SedtrailsData._clear_grid_metadata_cache()
    yield
    SedtrailsData._clear_grid_metadata_cache()


def test_min_resolution_ignores_duplicate_coordinates():
    """Checks min resolution uses distinct points and ignores duplicates."""
    data = _build_sedtrails_data(
        x=[0.0, 1.0, 1.0, 2.0],
        y=[0.0, 0.0, 0.0, 0.0],
    )

    assert data.metadata.min_resolution == pytest.approx(1.0)


def test_min_resolution_none_when_no_distinct_points():
    """Checks degenerate coordinates produce no resolution/envelope metadata."""
    data = _build_sedtrails_data(
        x=[5.0, 5.0, 5.0],
        y=[7.0, 7.0, 7.0],
    )

    assert data.metadata.min_resolution is None
    assert data.metadata.outer_envelope == []


def test_geographic_min_resolution_is_stored_in_meters():
    """Checks lon/lat grids compute CFL resolution in metres, not degrees."""
    data = _build_sedtrails_data(
        x=[4.0, 4.001, 4.0],
        y=[52.0, 52.0, 52.001],
        coordinate_system='geographic',
    )

    assert data.metadata.coordinate_system == 'geographic'
    assert data.metadata.runtime_coordinate_system == 'metric_projected'
    assert data.metadata.metric_coordinate_system == 'utm'
    assert data.metadata.source_crs == 'EPSG:4326'
    assert data.metadata.metric_crs == 'EPSG:32631'
    assert data.metadata.min_resolution_m == pytest.approx(68.45, rel=0.02)
    assert data.metadata.min_resolution == pytest.approx(data.metadata.min_resolution_m)


def test_geographic_topology_metadata_resolves_implicit_auto_utm():
    """Topology metadata resolves planar geographic distances without a full gather."""
    data = _build_sedtrails_data(
        x=[4.0, 4.001, 4.0],
        y=[52.0, 52.0, 52.001],
        coordinate_system='geographic',
        particle_face_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
    )

    assert data.metadata.metric_crs == 'EPSG:32631'
    assert data.metadata.min_resolution_m == pytest.approx(68.45, rel=0.02)


def test_ocean_scale_topology_metadata_avoids_global_geometry_and_bounds_batches(
    monkeypatch,
    isolated_grid_metadata_cache,
):
    """Authoritative topology avoids KDTree/hull work and bounds edge batches."""
    point_count = 2_048
    x = np.arange(point_count, dtype=np.float64)
    y = np.zeros(point_count, dtype=np.float64)
    triangles = np.column_stack(
        (
            np.arange(point_count - 2, dtype=np.int32),
            np.arange(1, point_count - 1, dtype=np.int32),
            np.arange(2, point_count, dtype=np.int32),
        )
    )
    monkeypatch.setattr(SedtrailsData, '_GRID_METADATA_CHUNK_SIZE', 64)
    monkeypatch.setattr(SedtrailsData, '_GRID_METADATA_FALLBACK_MAX_POINTS', 1_000)
    monkeypatch.setattr(
        sedtrails_data_module,
        'cKDTree',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('topology metadata must not build a KDTree')
        ),
    )
    monkeypatch.setattr(
        sedtrails_data_module,
        'ConvexHull',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('topology metadata must not build a convex hull')
        ),
    )
    batch_sizes = []
    original_distances = SedtrailsData._topology_edge_distances

    def recording_distances(flat_x, flat_y, start_nodes, end_nodes, transform):
        batch_sizes.append(start_nodes.size)
        return original_distances(flat_x, flat_y, start_nodes, end_nodes, transform)

    monkeypatch.setattr(
        SedtrailsData,
        '_topology_edge_distances',
        staticmethod(recording_distances),
    )

    data = _build_sedtrails_data(
        x,
        y,
        particle_face_connectivity=triangles,
    )

    assert data.metadata.min_resolution == pytest.approx(1.0)
    assert max(batch_sizes) <= 64
    assert np.asarray(data.metadata.outer_envelope).shape == (4, 2)


def test_large_topology_free_metadata_requires_connectivity(
    monkeypatch,
    isolated_grid_metadata_cache,
):
    """Large topology-free grids fail before global fallback allocation."""
    monkeypatch.setattr(SedtrailsData, '_GRID_METADATA_FALLBACK_MAX_POINTS', 4)

    with pytest.raises(ValueError, match='requires particle_face_connectivity'):
        _build_sedtrails_data(
            x=np.arange(5, dtype=float),
            y=np.zeros(5, dtype=float),
        )


def test_geodetic_resolution_and_envelope_cross_antimeridian():
    """Ocean-scale metadata should use surface metres and a compact seam-free envelope."""
    data = _build_sedtrails_data(
        x=[179.9, -179.9, 180.0],
        y=[0.0, 0.0, 0.2],
        coordinate_system='geographic',
        runtime_geometry='geodetic',
        surface_model='sphere',
    )

    assert data.metadata.runtime_coordinate_system == 'geodetic_surface'
    assert data.metadata.min_resolution_m == pytest.approx(22_239.0, rel=0.02)
    envelope = np.asarray(data.metadata.outer_envelope)
    assert np.ptp(envelope[:, 0]) < 1.0


def test_grid_metadata_reuses_equivalent_geometry(
    monkeypatch,
    isolated_grid_metadata_cache,
):
    """Equivalent forcing windows should calculate static grid metadata once."""
    original = SedtrailsData._calculate_grid_metadata
    calls = []

    def counted_calculation(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(SedtrailsData, '_calculate_grid_metadata', counted_calculation)
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    first = _build_sedtrails_data(x, y)
    first.metadata.outer_envelope.append([999.0, 999.0])
    second = _build_sedtrails_data(x.copy(), y.copy())

    assert len(calls) == 1
    assert second.metadata.min_resolution == pytest.approx(1.0)
    assert [999.0, 999.0] not in second.metadata.outer_envelope


def test_grid_metadata_cache_invalidates_changed_geometry_and_settings(
    monkeypatch,
    isolated_grid_metadata_cache,
):
    """Coordinate or transform changes must not reuse stale metadata."""
    original = SedtrailsData._calculate_grid_metadata
    call_count = 0

    def counted_calculation(self):
        nonlocal call_count
        call_count += 1
        return original(self)

    monkeypatch.setattr(SedtrailsData, '_calculate_grid_metadata', counted_calculation)
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])

    projected = _build_sedtrails_data(x, y)
    changed_geometry = _build_sedtrails_data(2.0 * x, 2.0 * y)
    geographic = _build_sedtrails_data(
        x,
        y,
        coordinate_system='geographic',
        runtime_geometry='geodetic',
    )

    assert call_count == 3
    assert projected.metadata.min_resolution == pytest.approx(1.0)
    assert changed_geometry.metadata.min_resolution == pytest.approx(2.0)
    assert geographic.metadata.runtime_coordinate_system == 'geodetic_surface'


def test_grid_metadata_cache_enforces_count_and_byte_bounds(
    monkeypatch,
    isolated_grid_metadata_cache,
):
    """Reusable metadata retention should obey both configured bounds."""
    monkeypatch.setattr(SedtrailsData, '_GRID_METADATA_CACHE_MAX_ENTRIES', 2)
    for offset in range(3):
        _build_sedtrails_data(
            x=[float(offset), float(offset + 1), float(offset)],
            y=[0.0, 0.0, 1.0],
        )

    assert len(SedtrailsData._grid_metadata_cache) == 2

    SedtrailsData._clear_grid_metadata_cache()
    monkeypatch.setattr(SedtrailsData, '_GRID_METADATA_CACHE_MAX_BYTES', 1)
    _build_sedtrails_data(x=[0.0, 1.0, 0.0], y=[0.0, 0.0, 1.0])

    assert not SedtrailsData._grid_metadata_cache
    assert SedtrailsData._grid_metadata_cache_bytes == 0
