Singh et al. (2023) benchmark

Overview
- Port of the analytical tracer dispersal model from Singh et al. (2023).
- Generates the 3-panel figure for data vs analytical solution.
- Adds hooks for later SedTRAILS comparisons and Delft3D v4 setup files.

Data
- By default, data is downloaded from Zenodo into: .cache/singh2023/
- You can override with --data-dir or disable downloads with --no-download.
- Expected filenames:
  - DEM_600lps.mat
  - DEM_800lps.mat
  - DEM_950lps.mat
  - DEM_1600lps.mat
  - Tracers_Step_length_600lps.mat
  - Tracers_Step_length_800lps.mat
  - Tracers_Step_length_950lps.mat
  - Tracers_Step_length_1600lps.mat

Run
- From repo root:
  - python tests/benchmarks/Singh2023/run_singh2023_analysis.py --flow-lps 600
  - python tests/benchmarks/Singh2023/run_singh2023_analysis.py --flow-lps 600 --sedtrails-nc examples/output/sedtrails_results.nc

Output
- tests/benchmarks/Singh2023/outputs/singh2023_analytical_tracer_validation.png
- tests/benchmarks/Singh2023/outputs/singh2023_sedtrails_cdf.png

SedTRAILS comparison
- Optional overlay of SedTRAILS travel distance PDF (panel a) and arrival-time PDF curves (panel c).
- Optional cumulative arrival fraction is saved as a separate figure.
- Use --distance-mode to switch between covered_distance, streamwise (x), or euclidean distances (default: streamwise).
- Use --x-positions to change breakthrough curve positions (meters).

Delft3D v4 BCT
- Example usage to write upstream discharge and downstream water level:

```python
from pathlib import Path
import numpy as np

from tests.benchmarks.Singh2023.delft3d_v4 import write_flume_bct

times_minutes = np.array([0.0, 60.0, 120.0])
discharge_upstream = np.array([0.6, 0.6, 0.6])
water_level_downstream = np.array([1.2, 1.2, 1.2])

write_flume_bct(
  output_path=Path("tests/benchmarks/Singh2023/outputs/flume.bct"),
  times_minutes=times_minutes,
  discharge_upstream=discharge_upstream,
  water_level_downstream=water_level_downstream,
)
```

Next steps needed
- Add a Delft3D v4 .bnd template so the boundary locations match the BCT tables.
