"""Tests for the format converter."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from sedtrails.particle_tracer.position_calculator_numba import create_grid_geometry
from sedtrails.transport_converter.format_converter import FormatConverter
from sedtrails.transport_converter.plugins.format import _xugrid_compat, fm_netcdf, sfincs


def _existing_input_path():
    """Returns a guaranteed-existing file path for plugin constructor inputs."""
    return __file__


def test_create_ugrid2d_supports_projection_keyword_rename(monkeypatch):
    """Uses the projection keyword accepted by the installed xugrid release."""
    node_x = np.array([0.0, 1.0, 0.0])
    node_y = np.array([0.0, 0.0, 1.0])
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    class LegacyUgrid2d:
        """Mimics the xugrid 0.14 projection argument."""

        def __init__(self, *_args, projected):
            self.value = projected

    monkeypatch.setattr(_xugrid_compat.xu, 'Ugrid2d', LegacyUgrid2d)
    assert _xugrid_compat.create_ugrid2d(node_x, node_y, faces, is_projected=False).value is False

    class ModernUgrid2d:
        """Mimics the xugrid 0.15 projection argument."""

        def __init__(self, *_args, is_projected):
            self.value = is_projected

    monkeypatch.setattr(_xugrid_compat.xu, 'Ugrid2d', ModernUgrid2d)
    assert _xugrid_compat.create_ugrid2d(node_x, node_y, faces, is_projected=True).value is True


def _write_square_pol(path, xmin, ymin, xmax, ymax):
    """Write a single-block Tekal polygon around a square."""
    path.write_text(
        f"""
island
4 2
{xmin} {ymin}
{xmax} {ymin}
{xmax} {ymax}
{xmin} {ymax}
""".strip()
    )


def _sfincs_dataset_from_face_centers(face_centers):
    """Build a small SFINCS-like UGRID dataset with quad faces at requested centroids."""
    centers = np.asarray(face_centers, dtype=float)
    node_x = []
    node_y = []
    faces = []
    half_size = 0.05
    for center_x, center_y in centers:
        start = len(node_x)
        node_x.extend([center_x - half_size, center_x + half_size, center_x + half_size, center_x - half_size])
        node_y.extend([center_y - half_size, center_y - half_size, center_y + half_size, center_y + half_size])
        faces.append([start, start + 1, start + 2, start + 3])

    n_faces = centers.shape[0]
    ds = xr.Dataset(
        data_vars={
            'mesh2d_node_x': (('node',), np.asarray(node_x)),
            'mesh2d_node_y': (('node',), np.asarray(node_y)),
            'mesh2d_face_nodes': (('face', 'nmax'), np.asarray(faces, dtype=np.int64)),
            'zb': (('face',), -np.arange(1, n_faces + 1, dtype=float)),
            'h': (('time', 'face'), np.vstack((np.arange(n_faces), np.arange(n_faces) + 10.0))),
            'u': (('time', 'face'), np.full((2, n_faces), 0.1)),
            'v': (('time', 'face'), np.zeros((2, n_faces))),
        },
        coords={'time': np.array(['2024-01-01T00:00:00', '2024-01-01T00:01:00'], dtype='datetime64[ns]')},
    )
    ds['mesh2d_face_nodes'].attrs['start_index'] = 0
    ds['mesh2d_face_nodes'].encoding['_FillValue'] = -1
    return ds


class _PluginWithCoordinateReader:
    """Test double exposing get_seeding_coordinates to bypass conversion."""

    def __init__(self):
        """Tracks whether convert was called unexpectedly."""
        self.convert_called = False

    def get_seeding_coordinates(self):
        """Provides deterministic coordinate arrays for seeding tests."""
        return [1.0, 2.0], [3.0, 4.0]

    def convert(self, *_args, **_kwargs):
        """Fails if fallback conversion is invoked when direct coordinates exist."""
        self.convert_called = True
        raise RuntimeError('convert should not be called when get_seeding_coordinates is available')


class _PluginWithSeedingFieldData:
    """Test double exposing the rich seeding field-data fast path."""

    def __init__(self):
        """Tracks whether less specific fallbacks were called unexpectedly."""
        self.coordinate_reader_called = False
        self.convert_called = False

    def get_seeding_field_data(self):
        """Provides deterministic field geometry for seeding tests."""
        return SimpleNamespace(
            x=[1.0, 2.0, 3.0],
            y=[4.0, 5.0, 6.0],
            face_node_connectivity=[[0, 1, 2]],
            particle_face_connectivity=[[0, 2, 1]],
            face_face_connectivity=[[-1, -1, -1]],
            boundary_edge_classification={'edge_nodes': [[0, 1]], 'edge_classes': ['open']},
            face_node_fill_value=-99,
            coordinate_system='geographic',
        )

    def get_seeding_coordinates(self):
        """Fails if the richer field-data path is skipped."""
        self.coordinate_reader_called = True
        raise RuntimeError('get_seeding_coordinates should not be called when get_seeding_field_data is available')

    def convert(self, *_args, **_kwargs):
        """Fails if fallback conversion is invoked when direct field data exists."""
        self.convert_called = True
        raise RuntimeError('convert should not be called when get_seeding_field_data is available')


class _PluginWithConvertOnly:
    """Test double that only supports convert-based seeding field retrieval."""

    class _SedtrailsLike:
        """Minimal Sedtrails-like object exposing x/y coordinates."""

        def __init__(self, x, y):
            """Stores coordinate arrays returned by the fake conversion step."""
            self.x = x
            self.y = y

    def convert(self, *_args, **_kwargs):
        """Returns a minimal converted object for fallback seeding logic."""
        return self._SedtrailsLike(x=(10.0, 20.0), y=(30.0, 40.0))


class _RecordingSelectivePlugin:
    """Record optional conversion constraints passed by FormatConverter."""

    def __init__(self):
        self.arguments = None

    def convert(
        self,
        current_time,
        reading_interval,
        reference_date,
        required_fields=None,
        max_memory_bytes=None,
    ):
        """Return minimal converted data while recording read constraints."""
        self.arguments = (
            current_time,
            reading_interval,
            reference_date,
            required_fields,
            max_memory_bytes,
        )
        return SimpleNamespace(metadata=SimpleNamespace(add=lambda *_args: None))


class _RecordingLegacyPlugin:
    """Legacy plugin accepting only the original conversion arguments."""

    def __init__(self):
        self.arguments = None

    def convert(self, current_time, reading_interval, reference_date):
        """Return minimal data while recording positional arguments."""
        self.arguments = (current_time, reading_interval, reference_date)
        return SimpleNamespace(metadata=SimpleNamespace(add=lambda *_args: None))


class _EstimatingSelectivePlugin:
    """Record source-plane estimation without converting source fields."""

    def __init__(self):
        self.load_calls = 0
        self.selected_fields = None

    def load(self):
        """Record metadata loading."""
        self.load_calls += 1

    def _selectable_required_fields(self, required_fields):
        """Filter derived-only names from the source selection."""
        return set(required_fields) & {'bed_level', 'water_depth'}

    def _estimate_bytes_per_time_plane(self, selected_fields):
        """Return a deterministic estimate for the selected source fields."""
        self.selected_fields = selected_fields
        return 128 * len(selected_fields)


class _PublicEstimatingPlugin:
    """Expose only the supported source-plane estimator API."""

    def __init__(self):
        self.required_fields = None

    def estimate_source_bytes_per_time_plane(self, required_fields):
        """Record the field selection and return an exact byte count."""
        self.required_fields = required_fields
        return 384


def test_convert_passes_selective_read_constraints_to_capable_plugin():
    """Forward required fields and byte budget to selective format plugins."""
    converter = FormatConverter(
        {
            'input_file': 'dummy.nc',
            'input_format': 'dummy',
            'reference_date': '2001-02-03',
        }
    )
    plugin = _RecordingSelectivePlugin()
    converter._format_plugin = plugin

    converter.convert_to_sedtrails(
        current_time=10.0,
        reading_interval=60.0,
        required_fields=('bed_level', 'depth_avg_flow_velocity'),
        max_memory_bytes=4096,
    )

    assert plugin.arguments == (
        10.0,
        60.0,
        np.datetime64('2001-02-03'),
        ('bed_level', 'depth_avg_flow_velocity'),
        4096,
    )


def test_convert_keeps_legacy_plugin_signature_compatible():
    """Do not pass new optional keywords to legacy plugins."""
    converter = FormatConverter(
        {
            'input_file': 'dummy.nc',
            'input_format': 'dummy',
            'reference_date': '2001-02-03',
        }
    )
    plugin = _RecordingLegacyPlugin()
    converter._format_plugin = plugin

    converter.convert_to_sedtrails(
        current_time=10.0,
        reading_interval=60.0,
        required_fields=('bed_level',),
        max_memory_bytes=4096,
    )

    assert plugin.arguments == (
        10.0,
        60.0,
        np.datetime64('2001-02-03'),
    )


def test_source_plane_estimate_uses_selective_plugin_metadata():
    """Estimate exact selected source bytes without running conversion."""
    converter = FormatConverter(
        {
            'input_file': 'dummy.nc',
            'input_format': 'dummy',
            'reference_date': '2001-02-03',
        }
    )
    plugin = _EstimatingSelectivePlugin()
    converter._format_plugin = plugin

    estimate = converter.estimate_source_bytes_per_time_plane(
        ('bed_level', 'water_depth', 'grain_velocity')
    )

    assert estimate == 256
    assert plugin.load_calls == 1
    assert plugin.selected_fields == {'bed_level', 'water_depth'}


def test_source_plane_estimate_keeps_legacy_plugins_compatible():
    """Return no exact estimate when a legacy plugin lacks the metadata hook."""
    converter = FormatConverter(
        {
            'input_file': 'dummy.nc',
            'input_format': 'dummy',
            'reference_date': '2001-02-03',
        }
    )
    converter._format_plugin = _RecordingLegacyPlugin()

    assert converter.estimate_source_bytes_per_time_plane(('bed_level',)) is None


def test_source_plane_estimate_prefers_public_plugin_api():
    """Use a plugin's public estimator without depending on private hooks."""
    converter = FormatConverter(
        {
            'input_file': 'dummy.nc',
            'input_format': 'dummy',
            'reference_date': '2001-02-03',
        }
    )
    plugin = _PublicEstimatingPlugin()
    converter._format_plugin = plugin
    required_fields = ('bed_level', 'water_depth')

    estimate = converter.estimate_source_bytes_per_time_plane(required_fields)

    assert estimate == 384
    assert plugin.required_fields == required_fields


def test_optional_connectivity_preserves_compact_indices():
    """Seeder field conversion downcasts safe int64 topology once."""
    connectivity = np.array([[0, 1, 2]], dtype=np.int64)

    compact = FormatConverter._optional_connectivity_array(connectivity)

    assert compact.dtype == np.int32
    np.testing.assert_array_equal(compact, connectivity)


def test_get_seeding_field_data_prefers_field_data_reader():
    """Ensures rich seeding field-data plugins preserve mesh metadata."""
    converter = FormatConverter({'input_file': 'dummy.nc', 'input_format': 'dummy', 'reference_date': '1999-12-31'})
    plugin = _PluginWithSeedingFieldData()
    converter._format_plugin = plugin

    field_data = converter.get_seeding_field_data()

    np.testing.assert_array_equal(field_data.x, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(field_data.y, np.array([4.0, 5.0, 6.0]))
    np.testing.assert_array_equal(field_data.face_node_connectivity, np.array([[0, 1, 2]]))
    np.testing.assert_array_equal(field_data.particle_face_connectivity, np.array([[0, 2, 1]]))
    np.testing.assert_array_equal(field_data.face_face_connectivity, np.array([[-1, -1, -1]]))
    assert field_data.boundary_edge_classification == {'edge_nodes': [[0, 1]], 'edge_classes': ['open']}
    assert field_data.face_node_fill_value == -99
    assert field_data.metadata.coordinate_system == 'geographic'
    assert field_data.reference_date == np.datetime64('1999-12-31')
    assert not plugin.coordinate_reader_called
    assert not plugin.convert_called


def test_auto_coordinate_override_preserves_resolved_plugin_metadata():
    """The schema default auto value must not replace inferred geographic data."""
    converter = FormatConverter(
        {
            'input_file': 'dummy.nc',
            'input_format': 'dummy',
            'reference_date': '1999-12-31',
            'coordinate_system': 'auto',
            'runtime_geometry': 'geodetic',
            'surface_model': 'sphere',
        }
    )
    converter._format_plugin = _PluginWithSeedingFieldData()

    field_data = converter.get_seeding_field_data()

    assert field_data.metadata.coordinate_system == 'geographic'
    assert field_data.metadata.runtime_geometry == 'geodetic'


def test_get_seeding_field_data_prefers_coordinate_reader():
    """Ensures coordinate-reader plugins are used without calling convert."""
    converter = FormatConverter({'input_file': 'dummy.nc', 'input_format': 'dummy', 'reference_date': '2000-01-01'})
    plugin = _PluginWithCoordinateReader()
    converter._format_plugin = plugin

    field_data = converter.get_seeding_field_data()

    np.testing.assert_array_equal(field_data.x, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(field_data.y, np.array([3.0, 4.0]))
    assert field_data.reference_date == np.datetime64('2000-01-01')
    assert not plugin.convert_called


def test_get_seeding_field_data_falls_back_to_convert():
    """Ensures convert fallback is used when no coordinate reader is available."""
    converter = FormatConverter({'input_file': 'dummy.nc', 'input_format': 'dummy', 'reference_date': '2001-02-03'})
    converter._format_plugin = _PluginWithConvertOnly()

    field_data = converter.get_seeding_field_data()

    np.testing.assert_array_equal(field_data.x, np.array([10.0, 20.0]))
    np.testing.assert_array_equal(field_data.y, np.array([30.0, 40.0]))
    assert field_data.reference_date == np.datetime64('2001-02-03')


def test_configure_format_plugin_sets_coordinate_system_override():
    """Converter-level coordinate-system overrides should reach format plugins."""
    converter = FormatConverter(
        {
            'input_file': 'dummy.nc',
            'input_format': 'dummy',
            'coordinate_system': 'geographic',
            'source_crs': 'EPSG:4326',
            'metric_crs': 'EPSG:32631',
            'domain_config': {'x_range': '0:1'},
        }
    )
    plugin = SimpleNamespace()

    converter._configure_format_plugin(plugin)

    assert plugin.coordinate_system == 'geographic'
    assert plugin.source_crs == 'EPSG:4326'
    assert plugin.metric_crs == 'EPSG:32631'
    assert plugin.domain_config == {'x_range': '0:1'}


def test_fm_convert_masks_inner_boundary_faces_from_domain_config(monkeypatch, tmp_path):
    """FM conversion excludes source faces whose centroids fall inside island polygons."""

    island_file = tmp_path / 'island.pol'
    island_file.write_text(
        """
island
4 2
9 9
12 9
12 12
9 12
""".strip()
    )
    ds = xr.Dataset(
        data_vars={
            'net_xcc': (('node',), np.array([0.0, 1.0, 0.0, 10.0, 11.0, 10.0])),
            'net_ycc': (('node',), np.array([0.0, 0.0, 1.0, 10.0, 10.0, 11.0])),
            'NetElemNode': (('face', 'nmax'), np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)),
        },
        coords={'time': np.array(['2024-01-01T00:00:00', '2024-01-01T00:01:00'], dtype='datetime64[ns]')},
    )
    ds['NetElemNode'].attrs['start_index'] = 0

    def fake_load(self):
        self.input_data = ds

    monkeypatch.setattr(fm_netcdf.FormatPlugin, 'load', fake_load)
    plugin = fm_netcdf.FormatPlugin(_existing_input_path())
    plugin.domain_config = {'inner_boundary_pol_files': [str(island_file)]}

    sedtrails_data = plugin.convert(reference_date=np.datetime64('2024-01-01T00:00:00'))
    seeding_data = plugin.get_seeding_field_data()

    np.testing.assert_array_equal(sedtrails_data.face_node_connectivity, np.array([[0, 1, 2]], dtype=np.int64))
    np.testing.assert_array_equal(seeding_data.face_node_connectivity, np.array([[0, 1, 2]], dtype=np.int64))
    assert sedtrails_data.metadata.inner_boundary_polygon_count == 1
    assert sedtrails_data.metadata.inner_boundary_masked_face_count == 1


def test_fm_convert_stores_boundary_edge_class_overrides(monkeypatch, tmp_path):
    """FM conversion stores user override classes for active boundary edges."""

    open_file = tmp_path / 'open.pol'
    land_file = tmp_path / 'land.pol'
    _write_square_pol(open_file, 0.25, -0.10, 0.75, 0.10)
    _write_square_pol(land_file, -0.10, 0.25, 0.10, 0.75)
    ds = xr.Dataset(
        data_vars={
            'net_xcc': (('node',), np.array([0.0, 1.0, 0.0])),
            'net_ycc': (('node',), np.array([0.0, 0.0, 1.0])),
            'NetElemNode': (('face', 'nmax'), np.array([[0, 1, 2]], dtype=np.int64)),
        },
        coords={'time': np.array(['2024-01-01T00:00:00', '2024-01-01T00:01:00'], dtype='datetime64[ns]')},
    )
    ds['NetElemNode'].attrs['start_index'] = 0

    def fake_load(self):
        self.input_data = ds

    monkeypatch.setattr(fm_netcdf.FormatPlugin, 'load', fake_load)
    plugin = fm_netcdf.FormatPlugin(_existing_input_path())
    plugin.domain_config = {'boundary_class_pol_files': {'open': [str(open_file)], 'land': [str(land_file)]}}

    sedtrails_data = plugin.convert(reference_date=np.datetime64('2024-01-01T00:00:00'))

    edge_classes = sedtrails_data.metadata.boundary_edge_classification
    assert edge_classes['class_counts'] == {'open': 1, 'land': 1, 'unclassified': 1, 'ambiguous': 0}
    assert edge_classes['polygon_counts'] == {'open': 1, 'land': 1}
    assert edge_classes['class_pol_files']['open'] == [str(open_file)]


def test_format_converter_applies_domain_config_to_fm_plugin(monkeypatch, tmp_path):
    """FormatConverter must pass domain polygons to plugin seeding geometry."""

    open_file = tmp_path / 'open.pol'
    _write_square_pol(open_file, 0.25, -0.10, 0.75, 0.10)
    ds = xr.Dataset(
        data_vars={
            'net_xcc': (('node',), np.array([0.0, 1.0, 0.0])),
            'net_ycc': (('node',), np.array([0.0, 0.0, 1.0])),
            'NetElemNode': (('face', 'nmax'), np.array([[0, 1, 2]], dtype=np.int64)),
        },
        coords={'time': np.array(['2024-01-01T00:00:00', '2024-01-01T00:01:00'], dtype='datetime64[ns]')},
    )
    ds['NetElemNode'].attrs['start_index'] = 0

    def fake_load(self):
        self.input_data = ds

    monkeypatch.setattr(fm_netcdf.FormatPlugin, 'load', fake_load)
    converter = FormatConverter(
        {
            'input_file': _existing_input_path(),
            'input_format': 'fm_netcdf',
            'reference_date': '2024-01-01T00:00:00',
            'domain_config': {'boundary_class_pol_files': {'open': [str(open_file)]}},
        }
    )

    field_data = converter.get_seeding_field_data()

    edge_classes = field_data.boundary_edge_classification
    assert edge_classes['class_counts']['open'] == 1
    assert edge_classes['polygon_counts']['open'] == 1
    assert edge_classes['class_pol_files']['open'] == [str(open_file)]


def test_fm_uses_delaunay_when_source_connectivity_does_not_index_cell_centers(monkeypatch, tmp_path):
    """FM native node connectivity should not be used with cell-center coordinates."""

    open_file = tmp_path / 'open.pol'
    _write_square_pol(open_file, 0.25, -0.10, 0.75, 0.10)
    ds = xr.Dataset(
        data_vars={
            'net_xcc': (('face',), np.array([0.0, 1.0, 1.0, 0.0])),
            'net_ycc': (('face',), np.array([0.0, 0.0, 1.0, 1.0])),
            'NetElemNode': (('elem', 'nmax'), np.array([[10, 11, 12]], dtype=np.int64)),
        },
        coords={'time': np.array(['2024-01-01T00:00:00', '2024-01-01T00:01:00'], dtype='datetime64[ns]')},
    )
    ds['NetElemNode'].attrs['start_index'] = 0

    def fake_load(self):
        self.input_data = ds

    monkeypatch.setattr(fm_netcdf.FormatPlugin, 'load', fake_load)
    plugin = fm_netcdf.FormatPlugin(_existing_input_path())
    plugin.domain_config = {'boundary_class_pol_files': {'open': [str(open_file)]}}

    field_data = plugin.get_seeding_field_data()

    assert np.max(field_data.face_node_connectivity) < field_data.x.size
    assert field_data.boundary_edge_classification['class_counts']['open'] == 1


def test_fm_rejects_primal_topology_when_node_ids_fit_face_count(monkeypatch):
    """Face-centre coordinates must not reuse numerically compatible primal topology."""
    ds = xr.Dataset(
        data_vars={
            'net_xcc': (('face',), np.arange(6, dtype=float)),
            'net_ycc': (('face',), np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])),
            'NetElemNode': (
                ('element', 'nmax'),
                np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            ),
        }
    )
    ds['NetElemNode'].attrs['start_index'] = 0
    fallback = np.array([[0, 4, 5]], dtype=np.int64)
    calls = []

    def fake_delaunay(*_args, **_kwargs):
        calls.append(True)
        return fallback

    monkeypatch.setattr(fm_netcdf, 'delaunay_connectivity', fake_delaunay)
    plugin = fm_netcdf.FormatPlugin(_existing_input_path())
    plugin.input_data = ds

    triangles = plugin._active_triangular_connectivity(
        ds['net_xcc'].values,
        ds['net_ycc'].values,
    )

    assert calls == [True]
    np.testing.assert_array_equal(triangles, fallback)


def test_fm_derives_face_center_topology_from_shared_primal_mesh(monkeypatch):
    """FM face centres use centroid-dual triangles from authoritative UGRID nodes."""
    ds = xr.Dataset(
        data_vars={
            'net_xcc': (('face',), np.array([0.5, 1.5, 0.5, 1.5])),
            'net_ycc': (('face',), np.array([0.5, 0.5, 1.5, 1.5])),
            'mesh2d_node_x': (
                ('node',),
                np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            ),
            'mesh2d_node_y': (
                ('node',),
                np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
            ),
            'mesh2d_face_nodes': (
                ('face', 'nmax'),
                np.array(
                    [
                        [0, 1, 4, 3],
                        [1, 2, 5, 4],
                        [3, 4, 7, 6],
                        [4, 5, 8, 7],
                    ],
                    dtype=np.int32,
                ),
            ),
        }
    )
    ds['mesh2d_face_nodes'].attrs['start_index'] = 0
    ds['mesh2d_face_nodes'].encoding['_FillValue'] = -1

    def unexpected_delaunay(*_args, **_kwargs):
        raise AssertionError('authoritative primal topology should avoid Delaunay')

    monkeypatch.setattr(fm_netcdf, 'delaunay_connectivity', unexpected_delaunay)
    plugin = fm_netcdf.FormatPlugin(_existing_input_path())
    plugin.input_data = ds

    triangles = plugin._active_triangular_connectivity(
        ds['net_xcc'].values,
        ds['net_ycc'].values,
    )

    assert {tuple(sorted(face)) for face in triangles} == {
        (0, 1, 3),
        (0, 2, 3),
    }


def test_fm_active_geometry_and_boundary_classification_are_cached(monkeypatch):
    """Repeated FM geometry requests should reuse static connectivity and edge classes."""
    plugin = fm_netcdf.FormatPlugin(_existing_input_path())
    plugin.input_data = xr.Dataset()
    node_x = np.array([0.0, 1.0, 0.0])
    node_y = np.array([0.0, 0.0, 1.0])
    connectivity = np.array([[0, 1, 2]], dtype=np.int64)
    call_counts = {'delaunay': 0, 'classify': 0}

    def fake_delaunay(_node_x, _node_y, **_kwargs):
        call_counts['delaunay'] += 1
        return connectivity

    class FakeClassification:
        def to_metadata(self):
            return {'edge_nodes': [[0, 1]], 'edge_classes': ['open']}

    def fake_classify(*_args, **_kwargs):
        call_counts['classify'] += 1
        return FakeClassification()

    monkeypatch.setattr(fm_netcdf, 'delaunay_connectivity', fake_delaunay)
    monkeypatch.setattr(fm_netcdf, 'classify_boundary_edges_from_config', fake_classify)

    first_connectivity = plugin._active_triangular_connectivity(node_x, node_y)
    second_connectivity = plugin._active_triangular_connectivity(node_x, node_y)
    first_classes = plugin._boundary_edge_classification(node_x, node_y, first_connectivity)
    second_classes = plugin._boundary_edge_classification(node_x, node_y, second_connectivity)

    assert call_counts == {'delaunay': 1, 'classify': 1}
    np.testing.assert_array_equal(first_connectivity, connectivity)
    np.testing.assert_array_equal(second_connectivity, connectivity)
    assert first_classes == second_classes == {'edge_nodes': [[0, 1]], 'edge_classes': ['open']}


def test_sfincs_get_seeding_coordinates_uses_ugrid_face_coordinates(monkeypatch):
    """Checks SFINCS seeding coordinates come directly from UGRID face centers."""
    class FakeUgrid2d:
        """Minimal UGRID-like grid with predefined face coordinates."""

        def __init__(self):
            self.face_x = np.array([1.0, 2.0, 3.0])
            self.face_y = np.array([4.0, 5.0, 6.0])

    class FakeUgridDataset:
        """Dataset wrapper exposing a single UGRID-like grid object."""

        def __init__(self):
            self.grid = FakeUgrid2d()

    def fake_load(self):
        """Injects synthetic UGRID input data into the plugin."""
        self.input_data = FakeUgridDataset()

    monkeypatch.setattr(sfincs.xu, 'UgridDataset', FakeUgridDataset)
    monkeypatch.setattr(sfincs.xu, 'Ugrid2d', FakeUgrid2d)
    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin(_existing_input_path())
    x, y = plugin.get_seeding_coordinates()

    np.testing.assert_array_equal(x, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(y, np.array([4.0, 5.0, 6.0]))


def test_sfincs_get_seeding_coordinates_fallback_computes_centroids(monkeypatch):
    """Verifies centroid fallback from face-node connectivity when UGRID is unavailable."""
    ds = xr.Dataset(
        data_vars={
            'mesh2d_node_x': (('node',), np.array([0.0, 2.0, 2.0, 0.0])),
            'mesh2d_node_y': (('node',), np.array([0.0, 0.0, 2.0, 2.0])),
            'mesh2d_face_nodes': (('face', 'nmax'), np.array([[1, 2, 3, 4]], dtype=np.int64)),
        }
    )
    ds['mesh2d_face_nodes'].attrs['start_index'] = 1
    ds['mesh2d_face_nodes'].encoding['_FillValue'] = -999

    def fake_load(self):
        """Injects synthetic mesh dataset into the plugin."""
        self.input_data = ds

    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin(_existing_input_path())
    x, y = plugin.get_seeding_coordinates()

    np.testing.assert_allclose(x, np.array([1.0]))
    np.testing.assert_allclose(y, np.array([1.0]))


def test_sfincs_uses_source_topology_for_face_center_triangles(monkeypatch):
    """Shared UGRID nodes provide dual triangles without global Delaunay."""

    ds = xr.Dataset(
        data_vars={
            'mesh2d_node_x': (('node',), np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])),
            'mesh2d_node_y': (('node',), np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])),
            'mesh2d_face_nodes': (
                ('face', 'nmax'),
                np.array(
                    [
                        [0, 1, 4, 3],
                        [1, 2, 5, 4],
                        [3, 4, 7, 6],
                        [4, 5, 8, 7],
                    ],
                    dtype=np.int64,
                ),
            ),
        }
    )
    ds['mesh2d_face_nodes'].attrs['start_index'] = 0
    ds['mesh2d_face_nodes'].encoding['_FillValue'] = -1

    def unexpected_delaunay(*_args, **_kwargs):
        raise AssertionError('SFINCS source topology should avoid Delaunay')

    monkeypatch.setattr(sfincs, 'delaunay_connectivity', unexpected_delaunay)
    plugin = sfincs.FormatPlugin(_existing_input_path())
    plugin.input_data = ds

    field_data = plugin.get_seeding_field_data()

    assert field_data.particle_face_connectivity.shape == (2, 3)
    assert {tuple(sorted(face)) for face in field_data.particle_face_connectivity} == {
        (0, 1, 3),
        (0, 2, 3),
    }


def test_sfincs_convert_stores_face_node_mesh_geometry(monkeypatch):
    """Checks convert stores node coordinates and normalized face connectivity."""
    ds = xr.Dataset(
        data_vars={
            'mesh2d_node_x': (('node',), np.array([0.0, 2.0, 2.0, 0.0])),
            'mesh2d_node_y': (('node',), np.array([0.0, 0.0, 2.0, 2.0])),
            'mesh2d_face_nodes': (('face', 'nmax'), np.array([[1, 2, 3, 4]], dtype=np.int64)),
            'zb': (('face',), np.array([-1.0])),
            'h': (('time', 'face'), np.array([[1.0], [2.0]])),
            'u': (('time', 'face'), np.array([[0.1], [0.2]])),
            'v': (('time', 'face'), np.array([[0.0], [0.0]])),
        },
        coords={'time': np.array(['2024-01-01T00:00:00', '2024-01-01T00:01:00'], dtype='datetime64[ns]')},
    )
    ds['mesh2d_face_nodes'].attrs['start_index'] = 1
    ds['mesh2d_face_nodes'].encoding['_FillValue'] = -999

    def fake_load(self):
        """Injects synthetic SFINCS dataset for conversion testing."""
        self.input_data = ds

    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin(_existing_input_path())
    sedtrails_data = plugin.convert(reference_date=np.datetime64('2024-01-01T00:00:00'))

    np.testing.assert_array_equal(sedtrails_data.node_x, np.array([0.0, 2.0, 2.0, 0.0]))
    np.testing.assert_array_equal(sedtrails_data.node_y, np.array([0.0, 0.0, 2.0, 2.0]))
    np.testing.assert_array_equal(sedtrails_data.face_node_connectivity, np.array([[0, 1, 2, 3]]))
    assert sedtrails_data.face_node_fill_value == -1
    assert sedtrails_data.mesh_geometry()['face_node_connectivity'].shape == (1, 4)


def test_sfincs_convert_masks_inner_boundary_faces_from_domain_config(monkeypatch, tmp_path):
    """SFINCS conversion excludes faces whose centroids fall inside island polygons."""
    island_file = tmp_path / 'island.pol'
    _write_square_pol(island_file, 0.8, 0.8, 1.2, 1.2)
    ds = _sfincs_dataset_from_face_centers([(0.0, 0.0), (1.0, 1.0)])

    def fake_load(self):
        """Injects synthetic SFINCS dataset for conversion testing."""
        self.input_data = ds

    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin(_existing_input_path())
    plugin.domain_config = {'inner_boundary_pol_files': [str(island_file)]}
    sedtrails_data = plugin.convert(reference_date=np.datetime64('2024-01-01T00:00:00'))

    np.testing.assert_allclose(sedtrails_data.x, np.array([0.0]))
    np.testing.assert_allclose(sedtrails_data.y, np.array([0.0]))
    np.testing.assert_array_equal(sedtrails_data.face_node_connectivity, np.array([[0, 1, 2, 3]]))
    np.testing.assert_array_equal(sedtrails_data.water_depth, np.array([[0.0], [10.0]]))
    np.testing.assert_allclose(sedtrails_data.depth_avg_flow_velocity['x'], np.full((2, 1), 0.1))
    assert sedtrails_data.node_x.shape == (8,)
    assert sedtrails_data.metadata.inner_boundary_polygon_count == 1
    assert sedtrails_data.metadata.inner_boundary_masked_face_count == 1
    assert sedtrails_data.metadata.inner_boundary_active_face_count == 1


def test_sfincs_seeding_geometry_respects_inner_boundary_holes(monkeypatch, tmp_path):
    """SFINCS seeding triangles should leave configured island interiors outside the mesh."""
    island_file = tmp_path / 'island.pol'
    _write_square_pol(island_file, 0.85, 0.85, 1.15, 1.15)
    face_centers = [
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, 2.0),
        (0.0, 2.0),
        (0.8, 0.8),
        (1.2, 0.8),
        (1.2, 1.2),
        (0.8, 1.2),
        (1.0, 1.0),
    ]
    ds = _sfincs_dataset_from_face_centers(face_centers)

    def fake_load(self):
        """Injects synthetic SFINCS dataset for seeding-geometry testing."""
        self.input_data = ds

    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin(_existing_input_path())
    plugin.domain_config = {'inner_boundary_pol_files': [str(island_file)]}
    field_data = plugin.get_seeding_field_data()
    grid_geometry = create_grid_geometry(
        field_data.x,
        field_data.y,
        triangles=field_data.particle_face_connectivity,
    )

    assert field_data.x.shape == (8,)
    assert field_data.particle_face_connectivity.shape[1] == 3
    simplices = grid_geometry.locate_points(
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 0.4, 1.6]),
    )
    assert simplices[0] == -1
    assert simplices[1] >= 0
    assert simplices[2] >= 0


def test_sfincs_convert_stores_boundary_edge_class_overrides(monkeypatch, tmp_path):
    """SFINCS conversion stores user override classes for active particle-domain edges."""
    open_file = tmp_path / 'open.pol'
    land_file = tmp_path / 'land.pol'
    _write_square_pol(open_file, 0.75, -0.10, 1.25, 0.10)
    _write_square_pol(land_file, -0.10, 0.75, 0.10, 1.25)
    ds = _sfincs_dataset_from_face_centers([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])

    def fake_load(self):
        """Injects synthetic SFINCS dataset for boundary-class testing."""
        self.input_data = ds

    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin(_existing_input_path())
    plugin.domain_config = {'boundary_class_pol_files': {'open': [str(open_file)], 'land': [str(land_file)]}}
    sedtrails_data = plugin.convert(reference_date=np.datetime64('2024-01-01T00:00:00'))

    edge_classes = sedtrails_data.metadata.boundary_edge_classification
    assert edge_classes['class_counts']['open'] == 1
    assert edge_classes['class_counts']['land'] == 1
    assert edge_classes['class_counts']['ambiguous'] == 0
    assert edge_classes['polygon_counts'] == {'open': 1, 'land': 1}


def test_sfincs_active_face_mask_is_cached(monkeypatch, tmp_path):
    """Repeated SFINCS active-mask requests should reuse polygon containment results."""
    island_file = tmp_path / 'island.pol'
    _write_square_pol(island_file, 0.8, 0.8, 1.2, 1.2)
    plugin = sfincs.FormatPlugin(_existing_input_path())
    plugin.domain_config = {'inner_boundary_pol_files': [str(island_file)]}
    call_count = 0

    def fake_points_inside_any_polygon(_points, _polygons):
        nonlocal call_count
        call_count += 1
        return np.array([False, True])

    monkeypatch.setattr(sfincs, 'points_inside_any_polygon', fake_points_inside_any_polygon)

    first_mask = plugin._active_face_mask(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    second_mask = plugin._active_face_mask(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    assert call_count == 1
    np.testing.assert_array_equal(first_mask, np.array([True, False]))
    np.testing.assert_array_equal(second_mask, np.array([True, False]))


def test_sfincs_static_mapped_geometry_is_cached_and_invalidated(monkeypatch):
    """Forcing reloads reuse static arrays until source geometry or domain changes."""
    dataset = _sfincs_dataset_from_face_centers(
        [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    )
    plugin = sfincs.FormatPlugin(_existing_input_path())
    plugin.input_data = dataset
    calls = {'centroids': 0, 'normalize': 0}
    original_centroids = sfincs.compute_face_centroids
    original_normalize = sfincs.normalize_face_node_connectivity

    def recording_centroids(*args, **kwargs):
        calls['centroids'] += 1
        return original_centroids(*args, **kwargs)

    def recording_normalize(*args, **kwargs):
        calls['normalize'] += 1
        return original_normalize(*args, **kwargs)

    monkeypatch.setattr(sfincs, 'compute_face_centroids', recording_centroids)
    monkeypatch.setattr(sfincs, 'normalize_face_node_connectivity', recording_normalize)
    time_info = {'num_times': 2}

    first = plugin._map_sfincs_variables(time_info, required_fields=('water_depth',))
    second = plugin._map_sfincs_variables(time_info, required_fields=('water_depth',))

    assert calls == {'centroids': 1, 'normalize': 1}
    assert second['x'] is first['x']
    assert second['face_node_connectivity'] is first['face_node_connectivity']
    assert second['particle_face_connectivity'] is first['particle_face_connectivity']

    plugin.domain_config = {'inner_boundary_pol_files': []}
    third = plugin._map_sfincs_variables(time_info, required_fields=('water_depth',))
    assert calls == {'centroids': 2, 'normalize': 2}
    assert third['x'] is not first['x']

    changed_dataset = dataset.copy(deep=True)
    changed_dataset['mesh2d_node_x'] = changed_dataset['mesh2d_node_x'] + 0.25
    plugin.input_data = changed_dataset
    fourth = plugin._map_sfincs_variables(time_info, required_fields=('water_depth',))
    assert calls == {'centroids': 3, 'normalize': 3}
    assert fourth['x'] is not third['x']


def test_sfincs_get_seeding_coordinates_raises_on_non_ugrid2d(monkeypatch):
    """Ensures non-UGRID grids raise a clear TypeError in coordinate retrieval."""
    class FakeNonUgrid2d:
        """Placeholder grid type that intentionally does not match Ugrid2d."""

        pass

    class FakeUgridDataset:
        """Dataset wrapper exposing an invalid grid type."""

        def __init__(self):
            self.grid = FakeNonUgrid2d()

    def fake_load(self):
        """Injects non-UGRID dataset into the plugin to trigger validation."""
        self.input_data = FakeUgridDataset()

    monkeypatch.setattr(sfincs.xu, 'UgridDataset', FakeUgridDataset)
    monkeypatch.setattr(sfincs.xu, 'Ugrid2d', type('OtherGrid', (), {}))
    monkeypatch.setattr(sfincs.FormatPlugin, 'load', fake_load)

    plugin = sfincs.FormatPlugin(_existing_input_path())

    with pytest.raises(TypeError, match='Expected Ugrid2d'):
        plugin.get_seeding_coordinates()
