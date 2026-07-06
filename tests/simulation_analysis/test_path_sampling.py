import numpy as np

from sedtrails.simulation_analysis.path_sampling import interpolation_indices, sample_field_at_trajectories


def test_interpolation_indices_support_datetime64():
    field_time = np.array(['2020-01-01T00:00', '2020-01-01T01:00'], dtype='datetime64[m]')
    target_time = np.array(['2020-01-01T00:30'], dtype='datetime64[m]')

    lower, upper, weight = interpolation_indices(field_time, target_time)

    np.testing.assert_array_equal(lower, [0])
    np.testing.assert_array_equal(upper, [1])
    np.testing.assert_allclose(weight, [0.5])


def test_sample_field_at_trajectories_interpolates_space_and_time():
    field_time = np.array([0.0, 10.0])
    grid_x = np.array([0.0, 1.0, 0.0, 1.0])
    grid_y = np.array([0.0, 0.0, 1.0, 1.0])
    base = grid_x + grid_y
    field_values = np.stack([base, base + 10.0], axis=0)

    sampled = sample_field_at_trajectories(
        field_time,
        grid_x,
        grid_y,
        field_values,
        trajectory_time=np.array([5.0]),
        trajectory_x=np.array([[0.5]]),
        trajectory_y=np.array([[0.5]]),
    )

    np.testing.assert_allclose(sampled.values, [[6.0]])
    np.testing.assert_array_equal(sampled.lower_time_index, [0])
    np.testing.assert_array_equal(sampled.upper_time_index, [1])
    np.testing.assert_allclose(sampled.time_weight, [0.5])


def test_sample_field_at_trajectories_selects_sediment_fraction():
    field_time = np.array([0.0])
    grid_x = np.array([0.0, 1.0, 0.0, 1.0])
    grid_y = np.array([0.0, 0.0, 1.0, 1.0])
    field_values = np.zeros((1, 2, 4), dtype=float)
    field_values[0, 1, :] = 4.0

    sampled = sample_field_at_trajectories(
        field_time,
        grid_x,
        grid_y,
        field_values,
        trajectory_time=np.array([0.0]),
        trajectory_x=np.array([[0.25, 0.75]]),
        trajectory_y=np.array([[0.25, 0.75]]),
        sediment_fraction=1,
    )

    np.testing.assert_allclose(sampled.values, [[4.0, 4.0]])
