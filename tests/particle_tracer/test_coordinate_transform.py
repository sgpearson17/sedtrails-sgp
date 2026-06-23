import numpy as np
import pytest

from sedtrails.particle_tracer.coordinate_transform import build_coordinate_transform


def test_auto_utm_selects_northern_epsg_zone():
    """Auto UTM should select the local northern EPSG zone from lon/lat grids."""
    transform = build_coordinate_transform(
        np.array([4.0, 4.001]),
        np.array([52.0, 52.001]),
        coordinate_system='geographic',
    )

    assert transform.metric_crs == 'EPSG:32631'
    assert transform.metadata()['metric_coordinate_system'] == 'utm'
    metric_x, metric_y = transform.source_to_metric(np.array([4.0]), np.array([52.0]))
    source_x, source_y = transform.metric_to_source(metric_x, metric_y)
    np.testing.assert_allclose(source_x, [4.0], atol=1.0e-10)
    np.testing.assert_allclose(source_y, [52.0], atol=1.0e-10)


def test_auto_utm_selects_southern_epsg_zone():
    """Auto UTM should use EPSG:327xx for southern hemisphere grids."""
    transform = build_coordinate_transform(
        np.array([18.0, 18.001]),
        np.array([-34.0, -34.001]),
        coordinate_system='geographic',
    )

    assert transform.metric_crs == 'EPSG:32734'
    assert transform.metadata()['utm_hemisphere'] == 'south'


def test_explicit_metric_crs_override_is_preserved():
    """Explicit projected CRS values should bypass auto UTM inference."""
    transform = build_coordinate_transform(
        np.array([4.0, 4.001]),
        np.array([52.0, 52.001]),
        coordinate_system='geographic',
        metric_crs='EPSG:32632',
    )

    assert transform.metric_crs == 'EPSG:32632'


def test_auto_utm_rejects_wide_multizone_grids():
    """Auto UTM should fail when one UTM zone is not a defensible metric CRS."""
    with pytest.raises(ValueError, match='metric_crs explicitly'):
        build_coordinate_transform(
            np.array([0.0, 12.0]),
            np.array([52.0, 52.0]),
            coordinate_system='geographic',
        )
