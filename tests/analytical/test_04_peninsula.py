import numpy as np

from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator

from ._helpers import (
    integrate_time_dependent,
    maybe_save_artifact,
    maybe_save_plot,
    rect_grid,
    write_metrics,
)


def test_peninsula_streamfunction_conservation(tmp_path):
    gx, gy = rect_grid(0.0, 100.0, 81, 0.0, 50.0, 41)

    u0 = 0.001
    x_center = 50.0
    radius = 0.32 * 50.0

    def psi_fn(x, y):
        return u0 * radius**2 * y / ((x - x_center) ** 2 + y**2) - u0 * y

    def velocity_fn(_t):
        denom = np.maximum(((gx - x_center) ** 2 + gy**2) ** 2, 1e-12)
        u = u0 - u0 * radius**2 * ((gx - x_center) ** 2 - gy**2) / denom
        v = -2.0 * u0 * radius**2 * ((gx - x_center) * gy) / denom
        return u, v

    calculator = create_numba_particle_calculator(gx, gy)

    base_y = np.linspace(8.0, 30.0, 10)
    extra_y = np.array([2.0, 4.0, 6.0])
    y0 = np.sort(np.concatenate([extra_y, base_y]))
    x0 = np.full(y0.size, 3.0)
    psi0 = psi_fn(x0, y0)

    x_end, y_end, times, xs, ys = integrate_time_dependent(
        calculator,
        x0,
        y0,
        velocity_fn,
        total_time=86_400.0,
        dt=300.0,
        return_history=True,
        stop_x=100.0,
        history_stride=int(3_600.0 / 300.0),
    )
    psi_end = psi_fn(x_end, y_end)

    psi_rel = np.abs(psi_end - psi0) / np.maximum(np.abs(psi0), 1.0)
    maybe_save_artifact(tmp_path, '04_peninsula_streamfunction_relative_error', psi_rel)

    write_metrics(
        tmp_path,
        '04_peninsula_metrics',
        {
            'max_rel_psi_error': float(np.max(psi_rel)),
            'mean_rel_psi_error': float(np.mean(psi_rel)),
        },
    )

    def _plot():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x_vals = np.linspace(np.min(gx), np.max(gx), 400)
        y_vals = np.linspace(np.min(gy), np.max(gy), 200)
        xx, yy = np.meshgrid(x_vals, y_vals, indexing='xy')
        psi_grid = psi_fn(xx, yy)
        ax.contourf(xx, yy, psi_grid >= 0.0, levels=[0.5, 1.5], colors=['#d9c2a3'])
        ax.contour(xx, yy, psi_grid, levels=np.sort(psi0), colors='k', linewidths=0.7, alpha=0.8)
        for idx in range(xs.shape[1]):
            ax.plot(
                xs[:, idx],
                ys[:, idx],
                color='tab:blue',
                linewidth=1.0,
                marker='.',
                markersize=2,
                markevery=3,
                label='sedtrails' if idx == 0 else None,
            )
        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')
        ax.set_ylim(0.0, 35.0)
        ax.set_title('Peninsula flow: SedTRAILS trajectories')
        ax.legend()
        return fig

    def _plot_psi():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for idx in range(xs.shape[1]):
            psi_hist = psi_fn(xs[:, idx], ys[:, idx])
            ax.plot(times, psi_hist - psi0[idx], label=f'particle {idx + 1}')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('psi error')
        ax.set_title('Peninsula flow: streamfunction error')
        return fig

    maybe_save_plot(tmp_path, '04_peninsula_trajectories', _plot)
    maybe_save_plot(tmp_path, '04_peninsula_psi_error', _plot_psi)

    assert np.max(psi_rel) < 1e-2
