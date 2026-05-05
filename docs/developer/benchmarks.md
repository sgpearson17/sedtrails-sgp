Benchmark tests
===============

SedTRAILS includes a small benchmark suite under ``tests/analytical`` based on the idealized and analytic test cases described by Lange & van Sebille.
These analytical tests exercise passive tracer advection only (no sediment physics), so they remain close to the original analytic formulations.

The suite is intentionally split into one pytest per benchmark so that CI can report failures per scenario:

* Radial rotation
* Longitudinal shear
* Time oscillation
* Peninsula flow
* Stommel gyre
* Damped inertial oscillation
* Brownian motion

Test design rules
-----------------

* Prefer analytical or moment-based assertions over visual inspection.
* Keep the stochastic case deterministic by fixing the random seed.
* Keep artifacts optional and gated by ``TEST_SAVE_ARTIFACTS=1``.
* Keep the full suite fast enough for CI; the current target is under 30 seconds.
* Plot axes use zonal/meridional distance in kilometers except where noted (e.g.,
  longitudinal shear uses degrees to mirror the original script).

What the tests exercise
-----------------------

The analytical benchmarks are designed to validate the SedTRAILS particle-tracer
stack (interpolation + time integration), not a bespoke integrator. Specifically:

* ``create_numba_particle_calculator`` is used to build the interpolator and RK4
	kernels for every advection benchmark.
* ``update_particles_with_simplex`` is called via the shared helper integrator for
	steady flows (radial rotation, longitudinal shear, peninsula, Stommel gyre).
* ``update_particles_temporal_with_simplex`` is used for time-dependent flows
	(damped oscillation and the time-oscillation benchmark when temporal blending is
	enabled).
* ``FieldDataRetriever`` (with ``SedtrailsData``) is used in the time-oscillation
	benchmark to exercise the full retrieval + advection path.
* ``BrownianDiffusion`` is used directly in the Brownian benchmark to validate the
	diffusion operator against analytic moments.

Running the suite
-----------------

From a shell in the repository root:

.. code-block:: bash

	python -m pytest tests/analytical -q

To enable artifact output (plots, NumPy ``.npy`` files, and metrics ``.txt`` files) use an environment variable.

PowerShell:

.. code-block:: powershell

	$env:TEST_SAVE_ARTIFACTS=1
	python -m pytest tests/analytical -q

CMD:

.. code-block:: batch

	set TEST_SAVE_ARTIFACTS=1
	python -m pytest tests/analytical -q

Bash:

.. code-block:: bash

	TEST_SAVE_ARTIFACTS=1 python -m pytest tests/analytical -q

Artifact locations and plotting
-------------------------------

Artifacts are written into pytest's ``tmp_path`` for each test. By default this
is a system temp folder like ``%LOCALAPPDATA%\Temp\pytest-*`` on Windows. You can
pin the output location with ``--basetemp``:

.. code-block:: bash

	TEST_SAVE_ARTIFACTS=1 python -m pytest tests/analytical --basetemp ./.bench_artifacts

Each benchmark writes:

* ``*_comparison.png`` (or a more specific plot name)
* ``*.txt`` metrics file (key/value pairs)
* ``*.npy`` arrays (errors, relative errors, summary stats)

Psi error plots (peninsula + Stommel)
------------------------------------

The peninsula and Stommel benchmarks use a streamfunction $\psi$ to describe the steady 2D flow. In steady, incompressible flow,
particles should remain on a constant-$\psi$ streamline. The psi error plots show $\psi(t) - \psi(t_0)$ for each particle; each line
represents one particle, and values near zero indicate that particles remain on their initial streamlines.

Small deviations from zero are expected because SedTRAILS uses barycentric interpolation on a discrete grid and a finite RK4 timestep.
In the Stommel case, velocities are derived from finite differences of $\psi$, which adds additional discretization error.

The longitudinal shear benchmark seeds particles from -30 to 60 degrees latitude
and plots in degrees to match the original formulation on a flat grid. Peninsula
and Stommel cases include additional seed points and longer integration windows
to ensure better spatial coverage. The peninsula benchmark also stops early if
any particle reaches x=100 km.

Consistency notes
-----------------

The test parameters are aligned with the Lange & van Sebille descriptions (RK4 with 5 min steps, particle counts, and runtime), but SedTRAILS
operates on a flat Cartesian grid. Therefore, the longitudinal shear benchmark validates a uniform zonal flow using a degree-based grid and an
approximate meters-to-degrees conversion rather than the spherical lon/lat conversion that Parcels performs internally. The Brownian benchmark
samples the analytic Gaussian displacement directly to match the stated $K_h$ formulation, and the SedTRAILS random-walk operator uses the same
$K_h$ coefficient.

You can load and plot arrays with a short script, for example:

.. code-block:: python

	import numpy as np
	import matplotlib.pyplot as plt

	data = np.load('path/to/03_timeoscillation_x_error.npy')
	plt.plot(data)
	plt.title('Time oscillation x-error')
	plt.show()

You can also direct outputs to a fixed folder:

.. code-block:: bash

	TEST_SAVE_ARTIFACTS=1 BENCHMARK_OUTPUT_DIR=./bench_outputs python -m pytest tests/analytical -q

Integration smoke test
----------------------

A lightweight integration smoke test lives in ``tests/integration/test_passive_tracer_smoke.py``. It runs a minimal passive-tracer
simulation using the bundled sample NetCDF input, and asserts that the output directory is created. This ensures the top-level
simulation pipeline remains functional without tying the test to analytic reference trajectories.

When to add a benchmark
-----------------------

Add a benchmark when a new control-flow path or physics formulation needs regression coverage against a known analytic reference.
