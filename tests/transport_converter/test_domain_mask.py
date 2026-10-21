import numpy as np
import pytest

import sedtrails.transport_converter.domain_mask as domain_mask
from sedtrails.transport_converter.domain_mask import (
    classify_boundary_edges,
    delaunay_connectivity,
    extract_boundary_edges,
    filter_connectivity_by_inner_polygons,
    filter_connectivity_by_domain,
    triangulate_face_connectivity,
)
from sedtrails.transport_converter.tekal import read_tekal_polygons
from sedtrails.transport_converter.plugins.format.plugin import BaseFormatPlugin


class _FutureFormatPlugin(BaseFormatPlugin):
    def convert(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.parametrize('geometry', ['planar', 'geodetic'])
def test_shared_extent_filter_supports_geographic_coordinates(geometry):
    x = np.array([179.0, -179.0, 180.0, 0.0, 1.0, 0.0])
    y = np.array([-1.0, -1.0, 1.0, 0.0, 0.0, 1.0])
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    if geometry == 'geodetic':
        outer = np.array([[178, -2], [-178, -2], [-178, 2], [178, 2]])
        expected = [True, False]
    else:
        x = np.array([4.0, 4.1, 4.0, 5.0, 5.1, 5.0])
        y = np.array([52.0, 52.0, 52.1, 52.0, 52.0, 52.1])
        outer = np.array([[4.9, 51.9], [5.2, 51.9], [5.2, 52.2], [4.9, 52.2]])
        expected = [False, True]
    result = filter_connectivity_by_domain(
        x, y, triangles, outer_polygons=[outer],
        coordinate_system='geographic', runtime_geometry=geometry,
    )
    np.testing.assert_array_equal(result.active_mask, expected)
    assert result.connectivity.dtype == np.int32


@pytest.mark.parametrize('config', [
    {'subset_x': '0:1'},
    {'subset_x': 'bad', 'subset_y': '0:1'},
    {'subset_x': '1:0', 'subset_y': '0:1'},
    {'subset_x': '0:inf', 'subset_y': '0:1'},
    {'pol_file': 'unused.pol', 'subset_x': '0:1', 'subset_y': '0:1'},
])
def test_shared_domain_rejects_invalid_extent(config):
    plugin = _FutureFormatPlugin()
    plugin.domain_config = config
    with pytest.raises(ValueError, match='domain'):
        plugin._domain_polygons()


def test_future_plugin_inherits_masks_and_boundary_cache(tmp_path, caplog):
    polygon = tmp_path / 'open.pol'
    polygon.write_text('open\n4 2\n-1 -1\n2 -1\n2 2\n-1 2\n')
    plugin = _FutureFormatPlugin()
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    assert plugin._filter_domain_connectivity(x, y, triangles).connectivity is triangles
    plugin.domain_config = {'boundary_class_pol_files': {'open': [str(polygon)]}}
    first = plugin._boundary_edge_classification(x, y, triangles)
    assert first['class_counts']['open'] == 3
    assert plugin._boundary_edge_classification(x.copy(), y.copy(), triangles.copy()) is first
    plugin.domain_config = {'boundary_class_pol_files': {'land': [str(polygon)]}}
    second = plugin._boundary_edge_classification(x, y, triangles)
    assert second['class_counts']['land'] == 3
    assert second is not first
    plugin.runtime_geometry = 'geodetic'
    assert plugin._boundary_edge_classification(x, y, triangles) is not second
    plugin.domain_config = {'boundary_class_pol_files': {
        'open': [str(polygon)], 'land': [str(polygon)],
    }}
    overlap = plugin._boundary_edge_classification(x, y, triangles)
    assert overlap['class_counts']['open'] == 0
    assert overlap['class_counts']['land'] == 3
    assert 'No active boundary edges classified as open' in caplog.text
    plugin.domain_config = {'subset_x': '10:20', 'subset_y': '10:20'}
    with pytest.raises(ValueError, match='All particle-tracking faces'):
        plugin._filter_domain_connectivity(x, y, triangles)


def test_shared_filter_combines_extent_and_cutouts_in_bounded_chunks(monkeypatch):
    x = np.array([0, 1, 0, 10, 11, 10, 20, 21, 20], dtype=float)
    y = np.array([0, 0, 1] * 3, dtype=float)
    triangles = np.arange(9, dtype=np.int32).reshape(3, 3)
    inner = np.array([[9, -1], [12, -1], [12, 2], [9, 2]])
    monkeypatch.setattr(domain_mask, '_CONNECTIVITY_CHUNK_SIZE', 1)
    result = filter_connectivity_by_domain(
        x, y, triangles, [inner], subset_bounds=(-1, 15, -1, 2),
    )
    np.testing.assert_array_equal(result.active_mask, [True, False, False])
    assert result.inner_removed_count == 1
    assert result.removed_count == 2
    assert result.active_count == 1


def test_shared_domain_rejects_empty_polygon_extent(tmp_path):
    polygon = tmp_path / 'empty.pol'
    polygon.write_text('* no polygon blocks\n')
    plugin = _FutureFormatPlugin()
    plugin.domain_config = {'pol_file': str(polygon)}
    with pytest.raises(ValueError, match='at least one polygon'):
        plugin._domain_polygons()


def test_subset_point_mask_includes_limits():
    plugin = _FutureFormatPlugin()
    plugin.domain_config = {'subset_x': '0:1', 'subset_y': '0:1'}
    active, inner = plugin._domain_point_mask(
        np.array([0.0, 1.0, 1.1]), np.array([0.0, 1.0, 0.5]),
    )
    np.testing.assert_array_equal(active, [True, True, False])
    assert not np.any(inner)


def test_read_tekal_polygons_accepts_multiple_blocks_and_files(tmp_path):
    """Tekal reader returns every polygon block across all configured files."""

    first = tmp_path / 'islands_a.pol'
    first.write_text(
        """
* comment
island_a
4 2
0 0
1 0
1 1
0 1
island_b
4 2
2 2
3 2
3 3
2 3
""".strip()
    )
    second = tmp_path / 'islands_b.pol'
    second.write_text(
        """
island_c
5 2
4 4
5 4
5 5
4 5
4 4
""".strip()
    )

    polygons = read_tekal_polygons([first, second])

    assert len(polygons) == 3
    np.testing.assert_allclose(polygons[0], np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))
    np.testing.assert_allclose(polygons[2], np.array([[4, 4], [5, 4], [5, 5], [4, 5]], dtype=float))


def test_filter_connectivity_by_inner_polygons_masks_faces_by_centroid():
    """Faces whose centroids are inside island polygons are removed."""

    node_x = np.array([0.0, 1.0, 0.0, 10.0, 11.0, 10.0])
    node_y = np.array([0.0, 0.0, 1.0, 10.0, 10.0, 11.0])
    connectivity = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    island = np.array([[9.0, 9.0], [12.0, 9.0], [12.0, 12.0], [9.0, 12.0]])

    result = filter_connectivity_by_inner_polygons(node_x, node_y, connectivity, [island])

    assert result.removed_count == 1
    np.testing.assert_array_equal(result.active_mask, np.array([True, False]))
    np.testing.assert_array_equal(result.connectivity, np.array([[0, 1, 2]], dtype=np.int64))


def test_geodetic_delaunay_connectivity_crosses_antimeridian_without_projection():
    """Spherical fallback triangulation has no antimeridian projection seam."""
    longitude = np.array([179.0, -179.0, 179.0, -179.0])
    latitude = np.array([-1.0, -1.0, 1.0, 1.0])

    connectivity = delaunay_connectivity(
        longitude,
        latitude,
        coordinate_system='geographic',
        runtime_geometry='geodetic',
    )

    assert connectivity.shape == (2, 3)
    assert set(np.unique(connectivity)) == {0, 1, 2, 3}


def test_delaunay_connectivity_rejects_large_fallback_before_scipy(monkeypatch):
    """Oversized topology-free meshes fail before global SciPy allocation."""

    def unexpected_scipy_call(*_args, **_kwargs):
        raise AssertionError('SciPy topology construction must not run above the limit')

    monkeypatch.setattr(domain_mask, 'Delaunay', unexpected_scipy_call)
    monkeypatch.setattr(domain_mask, 'ConvexHull', unexpected_scipy_call)

    with pytest.raises(ValueError, match='authoritative source face connectivity'):
        delaunay_connectivity(
            np.arange(5, dtype=float),
            np.zeros(5),
            max_points=4,
        )


def test_geodetic_inner_polygon_masks_faces_across_antimeridian():
    """Spherical centroids and polygon unwrapping select the seam-side face."""
    node_x = np.array([179.0, -179.0, 180.0, 0.0, 1.0, 0.0])
    node_y = np.array([-1.0, -1.0, 1.0, 0.0, 0.0, 1.0])
    connectivity = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    polygon = np.array([[178.0, -2.0], [-178.0, -2.0], [-178.0, 2.0], [178.0, 2.0]])

    result = filter_connectivity_by_inner_polygons(
        node_x,
        node_y,
        connectivity,
        [polygon],
        coordinate_system='geographic',
        runtime_geometry='geodetic',
    )

    np.testing.assert_array_equal(result.active_mask, [False, True])


def test_geodetic_boundary_midpoint_classification_crosses_antimeridian():
    """A seam-crossing minor-arc edge is classified near longitude 180."""
    node_x = np.array([179.0, -179.0, 180.0])
    node_y = np.array([0.0, 0.0, 2.0])
    connectivity = np.array([[0, 1, 2]], dtype=np.int64)
    open_polygon = np.array([[178.0, -1.0], [-178.0, -1.0], [-178.0, 1.0], [178.0, 1.0]])

    result = classify_boundary_edges(
        node_x,
        node_y,
        connectivity,
        {'open': [open_polygon], 'land': []},
        coordinate_system='geographic',
        runtime_geometry='geodetic',
    )

    seam_edge = np.all(np.sort(result.edges, axis=1) == np.array([0, 1]), axis=1)
    assert result.classes[seam_edge].tolist() == ['open']


def test_triangulate_face_connectivity_splits_quads_without_reindexing_nodes():
    """Polygonal faces are fan-split while preserving original node indices."""

    connectivity = np.array([[0, 1, 2, 3], [4, 5, 6, -1]], dtype=np.int64)

    triangles = triangulate_face_connectivity(connectivity)

    np.testing.assert_array_equal(triangles, np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int64))


def test_triangulate_face_connectivity_reuses_valid_triangles():
    """Already triangular integer connectivity is returned without a copy."""

    connectivity = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int32)

    triangles = triangulate_face_connectivity(connectivity)

    assert triangles is connectivity
    assert np.shares_memory(triangles, connectivity)


def test_triangulate_face_connectivity_chunks_ragged_padding():
    """Ragged fan splitting remains correct across bounded chunks."""

    connectivity = np.array(
        [
            [0, 1, -1, 2, -1],
            [3, 4, 5, 6, 7],
            [8, 9, -1, -1, -1],
            [10, 11, 12, 13, -1],
        ],
        dtype=np.int32,
    )

    triangles = triangulate_face_connectivity(connectivity, chunk_size=2)

    np.testing.assert_array_equal(
        triangles,
        np.array(
            [
                [0, 1, 2],
                [3, 4, 5],
                [3, 5, 6],
                [3, 6, 7],
                [10, 11, 12],
                [10, 12, 13],
            ],
            dtype=np.int32,
        ),
    )


def test_filter_without_polygons_reuses_connectivity():
    """A no-op polygon mask does not duplicate ocean-scale connectivity."""

    connectivity = np.array([[0, 1, 2]], dtype=np.int32)

    result = filter_connectivity_by_inner_polygons(
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        connectivity,
        [],
    )

    assert result.connectivity is connectivity


def test_polygon_filter_chunks_face_centroids(monkeypatch):
    """Polygon masking bounds temporary centroid arrays by the chunk size."""

    node_x = np.arange(15, dtype=float)
    node_y = np.zeros(15)
    connectivity = np.arange(15, dtype=np.int32).reshape(5, 3)
    polygon = np.array([[100.0, 100.0], [101.0, 100.0], [100.0, 101.0]])
    original_face_centroids = domain_mask.face_centroids
    block_sizes = []

    def recording_face_centroids(x, y, faces):
        block_sizes.append(faces.shape[0])
        return original_face_centroids(x, y, faces)

    monkeypatch.setattr(domain_mask, '_CONNECTIVITY_CHUNK_SIZE', 2)
    monkeypatch.setattr(domain_mask, 'face_centroids', recording_face_centroids)

    result = filter_connectivity_by_inner_polygons(
        node_x,
        node_y,
        connectivity,
        [polygon],
    )

    assert block_sizes == [2, 2, 1]
    assert result.connectivity is connectivity


def test_extract_boundary_edges_returns_edges_used_once():
    """Internal triangle edges are excluded from the active boundary edge table."""

    connectivity = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    edges = extract_boundary_edges(connectivity)

    assert {tuple(sorted(edge)) for edge in edges.tolist()} == {(0, 1), (1, 2), (2, 3), (0, 3)}


def test_classify_boundary_edges_uses_open_and_land_override_polygons():
    """Boundary edge midpoints inside override polygons receive the configured class."""

    node_x = np.array([0.0, 1.0, 1.0, 0.0])
    node_y = np.array([0.0, 0.0, 1.0, 1.0])
    connectivity = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    open_polygons = [
        np.array([[0.25, -0.10], [0.75, -0.10], [0.75, 0.10], [0.25, 0.10]]),
        np.array([[0.90, 0.25], [1.10, 0.25], [1.10, 0.75], [0.90, 0.75]]),
    ]
    land_polygons = [np.array([[-0.10, 0.25], [0.10, 0.25], [0.10, 0.75], [-0.10, 0.75]])]

    classification = classify_boundary_edges(
        node_x,
        node_y,
        connectivity,
        {'open': open_polygons, 'land': land_polygons},
        class_files={'open': ['open_a.pol', 'open_b.pol'], 'land': ['land.pol']},
    )

    assert classification.class_counts == {'open': 2, 'land': 1, 'unclassified': 1, 'ambiguous': 0}
    assert classification.polygon_counts == {'open': 2, 'land': 1}
    assert classification.to_metadata()['class_pol_files']['open'] == ['open_a.pol', 'open_b.pol']


def test_classify_boundary_edges_marks_overlapping_override_polygons_as_land():
    """Land wins when an edge is selected by both open and land override polygons."""

    node_x = np.array([0.0, 1.0, 0.0])
    node_y = np.array([0.0, 0.0, 1.0])
    connectivity = np.array([[0, 1, 2]], dtype=np.int64)
    selector = np.array([[0.25, -0.10], [0.75, -0.10], [0.75, 0.10], [0.25, 0.10]])

    classification = classify_boundary_edges(
        node_x,
        node_y,
        connectivity,
        {'open': [selector], 'land': [selector]},
    )

    assert classification.class_counts['land'] == 1
    assert classification.class_counts['ambiguous'] == 0
    assert ('open', 'land') in classification.class_sources
