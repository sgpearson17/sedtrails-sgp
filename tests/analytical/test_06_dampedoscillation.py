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


def test_damped_inertial_oscillation(tmp_path):
    u_g = 0.04
    u_0 = 0.3
    gamma = 1.0 / (2.89 * 86_400.0)
    gamma_g = 1.0 / (28.9 * 86_400.0)
    coriolis = 1.0e-4

    gx, gy = rect_grid(-20_000.0, 20_000.0, 5, -20_000.0, 20_000.0, 5)
    calculator = create_numba_particle_calculator(gx, gy)

    def velocity_fn(t):
        u = u_g * np.exp(-gamma_g * t) + (u_0 - u_g) * np.exp(-gamma * t) * np.cos(coriolis * t)
        v = -(u_0 - u_g) * np.exp(-gamma * t) * np.sin(coriolis * t)
        return u * np.ones_like(gx), v * np.ones_like(gy)

    def true_values(t, x_start, y_start):
        x_true = x_start + (u_g / gamma_g) * (1.0 - np.exp(-gamma_g * t))
        x_true += coriolis * ((u_0 - u_g) / (coriolis**2 + gamma**2)) * (
            (gamma / coriolis)
            + np.exp(-gamma * t) * (np.sin(coriolis * t) - (gamma / coriolis) * np.cos(coriolis * t))
        )
        y_true = y_start - ((u_0 - u_g) / (coriolis**2 + gamma**2)) * coriolis * (
            1.0 - np.exp(-gamma * t) * (np.cos(coriolis * t) + (gamma / coriolis) * np.sin(coriolis * t))
        )
        return x_true, y_true

    x0 = np.array([0.0])
    y0 = np.array([0.0])

    total_time = 4.0 * 86_400.0
    x_end, y_end, times, xs, ys = integrate_time_dependent(
        calculator,
        x0,
        y0,
        velocity_fn,
        total_time=total_time,
        dt=300.0,
        use_temporal=True,
        return_history=True,
        history_stride=int(3_600.0 / 300.0),
    )

    x_true, y_true = true_values(total_time, x0[0], y0[0])

    maybe_save_artifact(tmp_path, '06_dampedoscillation_xy_end', np.array([x_end[0], y_end[0], x_true, y_true]))

    analytic_x = []
    analytic_y = []
    for t in times:
        x_val, y_val = true_values(t, x0[0], y0[0])
        analytic_x.append(x_val)
        analytic_y.append(y_val)
    analytic_x = np.asarray(analytic_x).reshape(-1, 1)
    analytic_y = np.asarray(analytic_y).reshape(-1, 1)
    skill_mean, skill_std, _ = liu_weisberg_skill(analytic_x, analytic_y, xs, ys)

    write_metrics(
        tmp_path,
        '06_dampedoscillation_metrics',
        {
            'abs_x_error_m': float(abs(x_end[0] - x_true)),
            'abs_y_error_m': float(abs(y_end[0] - y_true)),
            'liu_weisberg_skill_mean': skill_mean,
            'liu_weisberg_skill_std': skill_std,
        },
    )

    def _plot():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(xs[:, 0], ys[:, 0], color='tab:blue', label='sedtrails', marker='.', markersize=3)
        ax.plot(analytic_x, analytic_y, color='tab:orange', linestyle='--', label='analytic')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Damped oscillation: SedTRAILS vs analytic')
        ax.legend()
        return fig

    maybe_save_plot(tmp_path, '06_dampedoscillation_comparison', _plot)

    np.testing.assert_allclose(x_end[0], x_true, rtol=1e-2, atol=20.0)
    np.testing.assert_allclose(y_end[0], y_true, rtol=1e-2, atol=20.0)
