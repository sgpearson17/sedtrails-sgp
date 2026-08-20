import numpy as np
import pytest

from sedtrails.particle_tracer.coordinate_transform import (
    build_coordinate_transform,
    coordinate_transform_from_metadata,
)


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


def test_auto_runtime_geometry_selects_geodetic_for_multizone_grid():
    """Ocean-scale geographic grids should not be forced into one UTM zone."""
    transform = build_coordinate_transform(
        np.array([0.0, 12.0]),
        np.array([52.0, 52.0]),
        coordinate_system='geographic',
        runtime_geometry='auto',
    )

    assert transform.is_geodetic
    assert transform.metric_crs is None
    assert transform.metadata()['runtime_coordinate_system'] == 'geodetic_surface'


@pytest.mark.parametrize('coordinate_system', ['geograpic', 'unknown', 'degrees'])
def test_unknown_coordinate_system_is_rejected(coordinate_system):
    """Coordinate-system typos must not silently switch to Cartesian physics."""
    with pytest.raises(ValueError, match='Unknown coordinate_system'):
        build_coordinate_transform(
            np.array([4.0]),
            np.array([52.0]),
            coordinate_system=coordinate_system,
        )


def test_projected_source_crs_is_rejected_for_geographic_input():
    """Geographic coordinates require a geographic source CRS."""
    with pytest.raises(ValueError, match='source_crs must be geographic'):
        build_coordinate_transform(
            np.array([4.0]),
            np.array([52.0]),
            coordinate_system='geographic',
            source_crs='EPSG:3857',
            metric_crs='EPSG:32631',
        )


@pytest.mark.parametrize('metric_crs', ['EPSG:4326', 'EPSG:2263'])
def test_non_metric_runtime_crs_is_rejected(metric_crs):
    """Angular and feet-based projected CRSs must not be treated as metres."""
    with pytest.raises(ValueError, match='metric_crs'):
        build_coordinate_transform(
            np.array([-74.0]),
            np.array([40.7]),
            coordinate_system='geographic',
            metric_crs=metric_crs,
        )


def test_geodetic_metadata_roundtrip_preserves_runtime_contract():
    """Serialized geodetic metadata should rebuild an equivalent transform."""
    transform = build_coordinate_transform(
        np.array([179.9, -179.9]),
        np.array([10.0, 10.1]),
        coordinate_system='geographic',
        runtime_geometry='geodetic',
        surface_model='sphere',
        earth_radius_m=6_371_000.0,
        longitude_wrap='0_360',
        velocity_basis='east_north',
    )

    rebuilt = coordinate_transform_from_metadata(transform.metadata())

    assert rebuilt.is_geodetic
    assert rebuilt.surface_model == 'sphere'
    assert rebuilt.earth_radius_m == 6_371_000.0
    assert rebuilt.longitude_wrap == '0_360'
    assert rebuilt.velocity_basis == 'east_north'
