import math
import numpy as np

from sedtrails.particle_tracer.diffusion_library import BrownianDiffusion

from ._helpers import maybe_save_artifact, maybe_save_plot, write_metrics


def test_brownian_random_walk_moments(tmp_path):
    np.random.seed(123456)

    n_particles = 100_000

    x = np.zeros(n_particles, dtype=np.float64)
    y = np.zeros(n_particles, dtype=np.float64)
    dt = 300.0
    kh = 100.0
    total_time = 86_400.0
    steps = int(total_time / dt)

    diffusion = BrownianDiffusion()
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    for _ in range(steps):
        x, y = diffusion.calculate(dt=dt, x=x, y=y, u=u, v=v, kh=kh)

    dx = x
    dy = y

    expected_var = 2.0 * kh * total_time
    mean_tol = 3.0 * math.sqrt(expected_var / n_particles)

    maybe_save_artifact(
        tmp_path,
        '07_brownian_stats',
        np.array([dx.mean(), dy.mean(), dx.var(ddof=1), dy.var(ddof=1)]),
    )

    write_metrics(
        tmp_path,
        '07_brownian_metrics',
        {
            'mean_dx': float(dx.mean()),
            'mean_dy': float(dy.mean()),
            'var_dx': float(dx.var(ddof=1)),
            'var_dy': float(dy.var(ddof=1)),
            'expected_var': float(expected_var),
        },
    )

    def _plot():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.hist(dx / 1000.0, bins=60, alpha=0.6, density=True, label='dx')
        ax.hist(dy / 1000.0, bins=60, alpha=0.6, density=True, label='dy')
        ax.set_xlabel('displacement [km]')
        ax.set_ylabel('density')
        ax.set_title('Brownian motion: displacement histograms')
        ax.legend()
        return fig

    maybe_save_plot(tmp_path, '07_brownian_histograms', _plot)

    assert abs(dx.mean()) < mean_tol
    assert abs(dy.mean()) < mean_tol
    np.testing.assert_allclose(dx.var(ddof=1), expected_var, rtol=0.1)
    np.testing.assert_allclose(dy.var(ddof=1), expected_var, rtol=0.1)
