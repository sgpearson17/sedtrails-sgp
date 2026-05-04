import numpy as np

from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator

from ._helpers import (
    liu_weisberg_skill,
    make_retriever,
    maybe_save_artifact,
    maybe_save_plot,
    rect_grid,
    write_metrics,
)


def test_time_oscillation_with_field_retriever_integration(tmp_path):
    omega = 2.0 * np.pi / 86_400.0
    amplitude = 0.1

    times = np.arange(0.0, 4.0 * 86_400.0 + 300.0, 300.0)
    gx, gy = rect_grid(-20_000.0, 20_000.0, 2, 0.0, 40_000.0, 2)

    u_time = np.array([amplitude * np.cos(omega * t) * np.ones_like(gx) for t in times])
    v_time = np.array([amplitude * np.ones_like(gx) for _ in times])

    retriever = make_retriever(times, gx, gy, u_time, v_time)
    calculator = create_numba_particle_calculator(gx, gy)

    x0 = np.linspace(-10_000.0, 10_000.0, 20)
    y0 = np.zeros_like(x0)

    dt = 300.0
    nsteps = int(4.0 * 86_400.0 / dt)
    x = x0.copy()
    y = y0.copy()
    simplex_ids = None

    xs = [x.copy()]
    ys = [y.copy()]
    times_hist = [0.0]

    for step in range(nsteps):
        t = step * dt
        flow = retriever.get_flow_field(t, 'depth_avg_flow_velocity')
        x, y, simplex_ids = calculator['update_particles_with_simplex'](
            x,
            y,
            flow['u'],
            flow['v'],
            dt,
            simplex_ids,
            0,
        )
        if (step + 1) % int(10_800.0 / dt) == 0:
            xs.append(x.copy())
            ys.append(y.copy())
            times_hist.append(t + dt)

    total_time = nsteps * dt
    x_true = x0 + amplitude / omega * np.sin(omega * total_time)
    y_true = y0 + amplitude * total_time

    maybe_save_artifact(tmp_path, '03_timeoscillation_x_error', x - x_true)
    maybe_save_artifact(tmp_path, '03_timeoscillation_y_error', y - y_true)

    times_arr = np.asarray(times_hist)
    analytic_x = np.stack([x0 + amplitude / omega * np.sin(omega * t) for t in times_arr])
    analytic_y = np.stack([y0 + amplitude * t for t in times_arr])
    skill_mean, skill_std, _ = liu_weisberg_skill(analytic_x, analytic_y, np.stack(xs), np.stack(ys))

    write_metrics(
        tmp_path,
        '03_timeoscillation_metrics',
        {
            'max_abs_x_error_m': float(np.max(np.abs(x - x_true))),
            'mean_abs_x_error_m': float(np.mean(np.abs(x - x_true))),
            'max_abs_y_error_m': float(np.max(np.abs(y - y_true))),
            'liu_weisberg_skill_mean': skill_mean,
            'liu_weisberg_skill_std': skill_std,
        },
    )

    def _plot():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        xs_arr = np.stack(xs)
        ys_arr = np.stack(ys)
        for idx in range(xs_arr.shape[1]):
            analytic_x = x0[idx] + amplitude / omega * np.sin(omega * times_arr)
            analytic_y = y0[idx] + amplitude * times_arr
            ax.plot(
                xs_arr[:, idx],
                ys_arr[:, idx],
                color='tab:blue',
                linewidth=1.0,
                marker='.',
                markersize=3,
                label='sedtrails' if idx == 0 else None,
            )
            ax.plot(
                analytic_x,
                analytic_y,
                color='tab:orange',
                linestyle='--',
                linewidth=1.0,
                label='analytic' if idx == 0 else None,
            )
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Time oscillation: SedTRAILS vs analytic')
        ax.legend()
        return fig

    maybe_save_plot(tmp_path, '03_timeoscillation_comparison', _plot)

    np.testing.assert_allclose(x, x_true, rtol=1e-2, atol=4.0)
    np.testing.assert_allclose(y, y_true, rtol=1e-2, atol=4.0)
