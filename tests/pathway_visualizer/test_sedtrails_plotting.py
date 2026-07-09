import numpy as np
import pytest
import xarray as xr

from sedtrails.pathway_visualizer.sedtrails_plotting import load_from_xarray
from sedtrails.pathway_visualizer.sedtrails_plotting import (
    TrajectoryArrays,
    particles_include_exclude,
    source_distance_from_baseline,
)


def test_load_from_xarray_accepts_time_major_layout():
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0, 20.0], [1.0, 21.0], [2.0, 22.0]])),
            'time': (('n_timesteps',), np.array([0.0, 60.0, 120.0])),
            'status_alive': (('n_timesteps', 'n_particles'), np.array([[1, 1], [1, 0], [1, 0]])),
            'population_id': (('n_particles',), np.array([0, 1], dtype=int)),
            'trajectory_id': (('n_particles',), np.array([0, 1], dtype=np.int64)),
        }
    )

    tr = load_from_xarray(ds)

    assert tr.x.shape == (2, 3)
    np.testing.assert_array_equal(tr.x[0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(tr.y[1], np.array([20.0, 21.0, 22.0]))
    np.testing.assert_array_equal(tr.time[1], np.array([0.0, 60.0, 120.0]))
    np.testing.assert_array_equal(tr.status_alive[1], np.array([1, 0, 0]))
    assert tr.trajectory_id == ['0', '1']


def test_load_from_xarray_converts_decoded_datetime_time_to_seconds():
    ds = xr.Dataset(
        data_vars={
            'x': (('n_timesteps', 'n_particles'), np.array([[0.0], [1.0], [2.0]])),
            'y': (('n_timesteps', 'n_particles'), np.array([[0.0], [0.5], [1.0]])),
            'time': (
                ('n_timesteps',),
                np.array(
                    [
                        '2020-01-01T00:00:00',
                        '2020-01-01T00:01:00',
                        '2020-01-01T00:02:00',
                    ],
                    dtype='datetime64[ns]',
                ),
            ),
        },
        attrs={'reference_date': '2020-01-01 00:00:00'},
    )

    tr = load_from_xarray(ds)

    np.testing.assert_array_equal(tr.time[0], np.array([0.0, 60.0, 120.0]))


def test_load_from_xarray_accepts_checkpoint_layout():
    ds = xr.Dataset(
        data_vars={
            'x': (('n_particles',), np.array([2.0, 12.0])),
            'y': (('n_particles',), np.array([1.0, 8.0])),
            'time': ((), 180.0),
            'status_alive': (('n_particles',), np.array([1, 1], dtype=np.uint8)),
            'population_id': (('n_particles',), np.array([0, 1], dtype=int)),
        },
        attrs={'sedtrails_file_kind': 'checkpoint'},
    )

    tr = load_from_xarray(ds)

    assert tr.x.shape == (2, 1)
    np.testing.assert_array_equal(tr.x[:, 0], np.array([2.0, 12.0]))
    np.testing.assert_array_equal(tr.time[:, 0], np.array([180.0, 180.0]))
    np.testing.assert_array_equal(tr.status_alive[:, 0], np.array([1, 1], dtype=np.uint8))


def test_load_from_xarray_rejects_particle_major_layout():
    ds = xr.Dataset(
        data_vars={
            'x': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0]])),
            'y': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0]])),
            'time': (('n_particles', 'n_timesteps'), np.array([[0.0, 60.0]])),
        }
    )

    with pytest.raises(ValueError, match='time-major trajectory arrays'):
        load_from_xarray(ds)


def test_source_distance_from_baseline_defaults_to_minimum_source_distance():
    tr = TrajectoryArrays(
        time=np.array([[0.0], [0.0], [0.0]]),
        x=np.array([[10.0], [20.0], [15.0]]),
        y=np.array([[0.0], [0.0], [0.0]]),
    )

    distance = source_distance_from_baseline(tr)

    np.testing.assert_allclose(distance, np.array([0.0, 10.0, 5.0]))


def test_particles_include_exclude_can_require_all_include_polygons():
    tr = TrajectoryArrays(
        time=np.tile(np.array([0.0, 1.0, 2.0]), (3, 1)),
        x=np.array(
            [
                [0.0, 5.0, 10.0],
                [0.0, 5.0, 6.0],
                [0.0, 10.0, 20.0],
            ]
        ),
        y=np.zeros((3, 3)),
    )
    poly_a = np.array([[4.0, -1.0], [6.0, -1.0], [6.0, 1.0], [4.0, 1.0]])
    poly_b = np.array([[9.0, -1.0], [11.0, -1.0], [11.0, 1.0], [9.0, 1.0]])

    any_mask = particles_include_exclude(tr, [poly_a, poly_b])
    all_mask = particles_include_exclude(tr, [poly_a, poly_b], require_all_include=True)

    np.testing.assert_array_equal(any_mask, np.array([True, True, True]))
    np.testing.assert_array_equal(all_mask, np.array([True, False, False]))
