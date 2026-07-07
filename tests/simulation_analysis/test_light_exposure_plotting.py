import matplotlib
import numpy as np

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from sedtrails.simulation_analysis.light_exposure import compute_light_exposure
from sedtrails.simulation_analysis.light_exposure_plotting import plot_particle_bleaching_potential


def test_plot_particle_bleaching_potential_builds_five_panel_figure():
    time = np.array(['2020-01-01T00:00', '2020-01-01T01:00', '2020-01-01T02:00'], dtype='datetime64[m]')
    water_depth = np.array([2.0, 2.5, 2.0])
    particle_z = np.array([0.1, 0.2, 0.1])
    surface_light = np.array([0.0, 100.0, 50.0])
    kd = np.array([1.0, 1.2, 1.1])
    exposure = compute_light_exposure(
        time,
        water_depth[:, None],
        kd[:, None],
        z=particle_z[:, None],
        z_convention='height_above_bed',
        surface_light=surface_light,
        reference_surface_light=100.0,
    )

    fig, axes = plot_particle_bleaching_potential(
        time,
        water_depth,
        particle_z,
        {'Total': np.array([0.1, 0.2, 0.3])},
        surface_light,
        kd,
        exposure,
    )

    assert len(axes) == 5
    assert len(fig.axes) == 7  # five panels, colorbar, and exposure twin axis
    plt.close(fig)
