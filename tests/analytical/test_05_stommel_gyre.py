import numpy as np

from sedtrails.particle_tracer.position_calculator_numba import create_numba_particle_calculator

from ._helpers import (
    integrate_time_dependent,
    maybe_save_artifact,
    maybe_save_plot,
    rect_grid,
    write_metrics,
)


def test_stommel_gyre_streamfunction_conservation(tmp_path):
    a = b = 10_000.0
    eps = 0.05
    amplitude = 100.0

    gx, gy = rect_grid(0.0, a, 201, 0.0, b, 201)

    l1 = (-1.0 + np.sqrt(1.0 + 4.0 * np.pi**2 * eps**2)) / (2.0 * eps)
    l2 = (-1.0 - np.sqrt(1.0 + 4.0 * np.pi**2 * eps**2)) / (2.0 * eps)
    c1 = (1.0 - np.exp(l2)) / (np.exp(l2) - np.exp(l1))
    c2 = -(1.0 + c1)

    def psi_fn(x, y):
        xi = x / a
        yi = y / b
        return amplitude * (c1 * np.exp(l1 * xi) + c2 * np.exp(l2 * xi) + 1.0) * np.sin(np.pi * yi)

    def velocity_fn(_t):
        xi = gx / a
        yi = gy / b
        common = c1 * np.exp(l1 * xi) + c2 * np.exp(l2 * xi) + 1.0
        dcommon_dxi = c1 * l1 * np.exp(l1 * xi) + c2 * l2 * np.exp(l2 * xi)

        u = -amplitude * common * (np.pi / b) * np.cos(np.pi * yi)
        v = amplitude * dcommon_dxi * (1.0 / a) * np.sin(np.pi * yi)
        return u, v

    calculator = create_numba_particle_calculator(gx, gy)

    x0 = np.arange(100.0, 9_901.0, 1000.0)
    y0 = np.full_like(x0, 5_000.0)
    psi0 = psi_fn(x0, y0)

    target_idx = int(np.argmax(x0))
    start_point = np.array([x0[target_idx], y0[target_idx]])
    state = {'moved': False}

    def stop_fn(_step, t, x, y):
        dist = np.hypot(x[target_idx] - start_point[0], y[target_idx] - start_point[1])
        if not state['moved'] and dist > 500.0:
            state['moved'] = True
        if state['moved'] and dist < 50.0 and t > 86_400.0:
            return True
        return False

    x_end, y_end, times, xs, ys = integrate_time_dependent(
        calculator,
        x0,
        y0,
        velocity_fn,
        total_time=200.0 * 86_400.0,
        dt=300.0,
        return_history=True,
        history_stride=int(86_400.0 / 300.0),
        stop_fn=stop_fn,
    )
    psi_end = psi_fn(x_end, y_end)

    psi_rel = np.abs(psi_end - psi0) / np.maximum(np.abs(psi0), 1.0)
    maybe_save_artifact(tmp_path, '05_stommel_streamfunction_relative_error', psi_rel)

    write_metrics(
        tmp_path,
        '05_stommel_metrics',
        {
            'max_rel_psi_error': float(np.max(psi_rel)),
            'mean_rel_psi_error': float(np.mean(psi_rel)),
        },
    )

    def _plot():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x_vals = np.unique(gx)
        y_vals = np.unique(gy)
        xx, yy = np.meshgrid(x_vals, y_vals, indexing='xy')
        psi_grid = psi_fn(xx, yy)
        ax.contour(xx, yy, psi_grid, levels=12, colors='k', linewidths=0.6, alpha=0.5)
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
        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')
        ax.set_title('Stommel gyre: SedTRAILS trajectories')
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
        ax.set_title('Stommel gyre: streamfunction error')
        return fig

    maybe_save_plot(tmp_path, '05_stommel_trajectories', _plot)
    maybe_save_plot(tmp_path, '05_stommel_psi_error', _plot_psi)

    assert np.max(psi_rel) < 1e-2
