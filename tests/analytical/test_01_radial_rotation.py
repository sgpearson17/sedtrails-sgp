import numpy as np

from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator

from ._helpers import (
    integrate_time_dependent,
    liu_weisberg_skill,
    maybe_save_artifact,
    maybe_save_plot,
    rect_grid,
    write_metrics,
)


def test_radial_rotation_closed_orbits(tmp_path):
    domain = 20_000.0
    gx, gy = rect_grid(-domain / 2, domain / 2, 201, -domain / 2, domain / 2, 201)
    calculator = create_numba_particle_calculator(gx, gy)

    omega = 2.0 * np.pi / 86_400.0

    def velocity_fn(_t):
        return -omega * gy, omega * gx

    x0 = np.zeros(4, dtype=np.float64)
    y0 = np.array([1_000.0, 2_000.0, 3_000.0, 4_000.0])

    x_end, y_end, times, xs, ys = integrate_time_dependent(
        calculator,
        x0,
        y0,
        velocity_fn,
        total_time=86_400.0,
        dt=300.0,
        return_history=True,
        history_stride=12,
    )

    closing_error = np.sqrt((x_end - x0) ** 2 + (y_end - y0) ** 2)
    maybe_save_artifact(tmp_path, '01_radial_rotation_closing_error', closing_error)

    analytic_x = np.stack([x0 * np.cos(omega * t) - y0 * np.sin(omega * t) for t in times])
    analytic_y = np.stack([x0 * np.sin(omega * t) + y0 * np.cos(omega * t) for t in times])
    skill_mean, skill_std, _ = liu_weisberg_skill(analytic_x, analytic_y, xs, ys)

    write_metrics(
        tmp_path,
        '01_radial_rotation_metrics',
        {
            'max_closing_error_m': float(np.max(closing_error)),
            'mean_closing_error_m': float(np.mean(closing_error)),
            'liu_weisberg_skill_mean': skill_mean,
            'liu_weisberg_skill_std': skill_std,
        },
    )

    def _plot():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for idx in range(xs.shape[1]):
            ax.plot(
                xs[:, idx],
                ys[:, idx],
                color='tab:blue',
                linewidth=1.0,
                marker='.',
                markersize=3,
                label='sedtrails' if idx == 0 else None,
            )
            r = np.hypot(x0[idx], y0[idx])
            theta = np.arctan2(y0[idx], x0[idx]) + omega * times
            ax.plot(
                r * np.cos(theta),
                r * np.sin(theta),
                color='tab:orange',
                linestyle='--',
                linewidth=1.0,
                label='analytic' if idx == 0 else None,
            )
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Radial rotation: SedTRAILS vs analytic')
        ax.legend()
        ax.axis('equal')
        return fig

    maybe_save_plot(tmp_path, '01_radial_rotation_comparison', _plot)

    assert np.max(closing_error) < 25.0
