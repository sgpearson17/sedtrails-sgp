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


def test_longitudinal_shear_adapted_linear_shear(tmp_path):
    gx, gy = rect_grid(-5_000_000.0, 5_000_000.0, 81, -8_000_000.0, 8_000_000.0, 81)
    calculator = create_numba_particle_calculator(gx, gy)

    def velocity_fn(_t):
        return np.ones_like(gy), np.zeros_like(gy)

    x0 = np.zeros(31, dtype=np.float64)
    y0 = np.linspace(-45.0, 45.0, x0.size) * 111_000.0

    total_time = 57.0 * 86_400.0
    x_end, y_end, times, xs, ys = integrate_time_dependent(
        calculator,
        x0,
        y0,
        velocity_fn,
        total_time=total_time,
        dt=300.0,
        return_history=True,
        history_stride=int(86_400.0 / 300.0),
    )

    x_true = x0 + total_time
    y_true = y0

    maybe_save_artifact(tmp_path, '02_longitudinal_shear_x_error', x_end - x_true)
    maybe_save_artifact(tmp_path, '02_longitudinal_shear_y_error', y_end - y_true)

    analytic_x = np.stack([x0 + t for t in times])
    analytic_y = np.stack([y0 for _ in times])
    skill_mean, skill_std, _ = liu_weisberg_skill(analytic_x, analytic_y, xs, ys)

    write_metrics(
        tmp_path,
        '02_longitudinal_shear_metrics',
        {
            'max_abs_x_error_m': float(np.max(np.abs(x_end - x_true))),
            'mean_abs_x_error_m': float(np.mean(np.abs(x_end - x_true))),
            'max_abs_y_error_m': float(np.max(np.abs(y_end - y_true))),
            'liu_weisberg_skill_mean': skill_mean,
            'liu_weisberg_skill_std': skill_std,
        },
    )

    def _plot():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for idx in range(xs.shape[1]):
            analytic_x = x0[idx] + times
            analytic_y = np.full_like(times, y0[idx])
            ax.plot(
                xs[:, idx] / 1000.0,
                ys[:, idx] / 1000.0,
                color='tab:blue',
                linewidth=1.0,
                marker='.',
                markersize=3,
                label='sedtrails' if idx == 0 else None,
            )
            ax.plot(
                analytic_x / 1000.0,
                analytic_y / 1000.0,
                color='tab:orange',
                linestyle='--',
                linewidth=1.0,
                label='analytic' if idx == 0 else None,
            )
        ax.set_xlabel('zonal distance [km]')
        ax.set_ylabel('meridional distance [km]')
        ax.set_title('Longitudinal shear: SedTRAILS vs analytic')
        ax.legend()
        return fig

    maybe_save_plot(tmp_path, '02_longitudinal_shear_comparison', _plot)

    np.testing.assert_allclose(x_end, x_true, rtol=1e-2, atol=2.0)
    np.testing.assert_allclose(y_end, y_true, rtol=1e-6, atol=1e-6)
