from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import requests
from scipy.io import loadmat


ZENODO_RECORD_ID = "6703451"
ZENODO_BASE_URL = f"https://zenodo.org/record/{ZENODO_RECORD_ID}/files"
ZENODO_FILES = {
    "DEM_600lps.mat": "3afc51242a6849cab47ef79ae14817de",
    "DEM_800lps.mat": "0505e0b9676468ab6f733b0adaa32f1a",
    "DEM_950lps.mat": "ed03fbbaa3725f6eaf8837d1734d1b1c",
    "DEM_1600lps.mat": "ce8ce57e9044ab74ac0a2e1c734616a5",
    "Tracers_Step_length_600lps.mat": "e7b7a0a55676120b6bde7f09b6b90b7c",
    "Tracers_Step_length_800lps.mat": "c9f1c6ef6f849449b058f6e57152fd32",
    "Tracers_Step_length_950lps.mat": "fcdab50602d4fd22b4033b08e3834da8",
    "Tracers_Step_length_1600lps.mat": "4da39d43f6658f9f74bbf7ea77902ca2",
}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "bold",
            "font.style": "italic",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )


def load_tracer_steps(mat_path: Path) -> np.ndarray:
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "small" in data:
        return np.asarray(data["small"]).ravel()
    if "Lt" in data:
        lt_obj = data["Lt"]
        if hasattr(lt_obj, "small"):
            return np.asarray(lt_obj.small).ravel()
    for key, value in data.items():
        if key.startswith("__"):
            continue
        return np.asarray(value).ravel()
    raise KeyError(f"Could not find tracer steps in {mat_path}")


def md5sum(file_path: Path) -> str:
    hasher = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_zenodo_file(file_name: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    url = f"{ZENODO_BASE_URL}/{file_name}?download=1"
    dest_path = target_dir / file_name
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with temp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    temp_path.replace(dest_path)
    return dest_path


def ensure_data_files(
    data_dir: Path,
    flow_lps: int,
    allow_download: bool,
) -> None:
    required = {
        f"DEM_{flow_lps}lps.mat",
        f"Tracers_Step_length_{flow_lps}lps.mat",
    }

    missing: list[str] = []
    for file_name in required:
        path = data_dir / file_name
        if not path.exists():
            missing.append(file_name)
            continue
        expected_md5 = ZENODO_FILES.get(file_name)
        if expected_md5 is not None:
            if md5sum(path) != expected_md5:
                missing.append(file_name)

    if not missing:
        return

    if not allow_download:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            "Missing or invalid data files: "
            f"{missing_str}. Provide --data-dir with valid files or allow downloads."
        )

    for file_name in missing:
        dest_path = download_zenodo_file(file_name, data_dir)
        expected_md5 = ZENODO_FILES.get(file_name)
        if expected_md5 and md5sum(dest_path) != expected_md5:
            raise ValueError(f"MD5 mismatch for {dest_path}")


def compute_analytical_concentration(
    x: np.ndarray, t: np.ndarray, ub_virtual: float, dx: float
) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    t_safe = np.where(t <= 0, np.nan, t)
    coef = 1.0 / np.sqrt(4.0 * np.pi * dx * t_safe)
    term1 = np.exp(-(-x - ub_virtual * t_safe) ** 2 / (4.0 * dx * t_safe))
    term2 = np.exp(-(x - ub_virtual * t_safe) ** 2 / (4.0 * dx * t_safe))
    return coef * (term1 + term2)


def load_sedtrails_dataset(nc_path: Path) -> xr.Dataset:
    if not nc_path.exists():
        raise FileNotFoundError(f"SedTRAILS netCDF not found: {nc_path}")
    return xr.open_dataset(nc_path)


def get_time_axis(ds: xr.Dataset) -> np.ndarray:
    time = ds["time"].values
    if time.ndim == 2:
        return time[0, :]
    if time.ndim == 1:
        return time
    raise ValueError("Unsupported time array shape in SedTRAILS dataset")


def compute_sedtrails_distance_at_time(
    ds: xr.Dataset,
    t_target: float,
    mode: str,
    flowfield_index: int = 0,
) -> np.ndarray:
    time_axis = get_time_axis(ds)
    idx = int(np.nanargmin(np.abs(time_axis - t_target)))

    released = None
    if "status_released" in ds:
        released = ds["status_released"].values[:, idx] == 1

    if mode == "covered_distance" and "covered_distance" in ds:
        distances = ds["covered_distance"].values[flowfield_index, :, idx]
    elif mode == "streamwise":
        x = ds["x"].values
        distances = x[:, idx] - x[:, 0]
    elif mode == "euclidean":
        x = ds["x"].values
        y = ds["y"].values
        z = ds["z"].values
        distances = np.sqrt(
            (x[:, idx] - x[:, 0]) ** 2
            + (y[:, idx] - y[:, 0]) ** 2
            + (z[:, idx] - z[:, 0]) ** 2
        )
    else:
        raise ValueError(
            "Unsupported distance mode. Use covered_distance, streamwise, or euclidean."
        )

    distances = np.asarray(distances, dtype=float)
    if released is not None:
        distances = distances[released]
    return distances


def compute_first_passage_times(
    ds: xr.Dataset,
    x_positions: list[float],
) -> dict[float, np.ndarray]:
    time = ds["time"].values
    x = ds["x"].values

    if time.ndim == 2:
        time_axis = time[0, :]
    else:
        time_axis = time

    released = None
    if "status_released" in ds:
        released = ds["status_released"].values == 1

    results: dict[float, np.ndarray] = {}
    for x_pos in x_positions:
        arrival_times = np.full(x.shape[0], np.nan, dtype=float)
        for particle_idx in range(x.shape[0]):
            meets = x[particle_idx, :] >= x_pos
            if released is not None:
                meets = meets & released[particle_idx, :]
            if np.any(meets):
                first_idx = int(np.argmax(meets))
                arrival_times[particle_idx] = time_axis[first_idx]
        results[x_pos] = arrival_times
    return results


def plot_analytical_vs_data(
    tracer_steps: np.ndarray,
    t_exp: float,
    flume_length: float,
    output_path: Path,
    sedtrails_distances: np.ndarray | None = None,
    sedtrails_arrival_times: dict[float, np.ndarray] | None = None,
) -> None:
    lt_mean = float(np.mean(tracer_steps))
    ub_virtual = lt_mean / t_exp
    dx = float(np.var(tracer_steps, ddof=1) / (2.0 * t_exp))

    configure_plot_style()
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(3, 1, figsize=(8.3 / 2.54, 20.0 / 2.54))
    x = np.arange(0.0, flume_length + 0.1, 0.1)

    t = t_exp
    c_analytical = compute_analytical_concentration(x, t, ub_virtual, dx)

    ax = axes[0]
    ax.grid(True)
    ax.hist(tracer_steps, bins="auto", density=True, color=cmap(0.35))
    if sedtrails_distances is not None and sedtrails_distances.size > 0:
        ax.hist(
            sedtrails_distances,
            bins="auto",
            density=True,
            color=cmap(0.7),
            alpha=0.6,
        )
    ax.plot(x, c_analytical, "-", linewidth=1.5, color=cmap(0.1))
    ax.set_ylabel("PDF [-]")
    ax.set_xlabel("L_T [m]")
    ax.set_title("(a)")

    ax = axes[1]
    ax.grid(True)
    t_vals = np.arange(0.0, t_exp * 20.0 + 1.0, 2.0 * t_exp)
    line_colors = cmap(np.linspace(0.1, 0.9, len(t_vals)))
    for color, t_val in zip(line_colors, t_vals):
        c_vals = compute_analytical_concentration(x, t_val, ub_virtual, dx)
        mask = np.isfinite(c_vals)
        if np.any(mask):
            ax.plot(x[mask], c_vals[mask], linewidth=1.5, color=color)
    ax.set_ylabel("PDF [-]")
    ax.set_xlabel("L_T [m]")
    ax.set_title("(b)")

    norm = plt.Normalize(vmin=0.0, vmax=t_vals.max() / 60.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="vertical")
    cb.set_label("Time [min]")
    cb.set_ticks([0.0, t_vals.max() / 120.0, t_vals.max() / 60.0])

    ax = axes[2]
    ax.grid(True)
    t_vals = np.arange(0.0, t_exp * 20.0 + 1.0, 0.5 * t_exp)
    t_plot = t_vals / 60.0
    x_positions = [0.0, 20.0, 55.0]
    for x_pos, color in zip(x_positions, [cmap(0.1), cmap(0.45), cmap(0.8)]):
        c_vals = compute_analytical_concentration(x_pos, t_vals, ub_virtual, dx)
        mask = np.isfinite(c_vals)
        ax.plot(t_plot[mask], c_vals[mask], linewidth=1.5, color=color)
        if sedtrails_arrival_times is not None and x_pos in sedtrails_arrival_times:
            arrivals = sedtrails_arrival_times[x_pos]
            arrivals = arrivals[np.isfinite(arrivals)]
            if arrivals.size > 0:
                counts, edges = np.histogram(arrivals / 60.0, bins="auto", density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.plot(
                    centers,
                    counts,
                    "--",
                    linewidth=1.2,
                    color=color,
                )
    ax.set_ylabel("PDF [-]")
    ax.set_xlabel("Time [min]")
    ax.set_title("(c)")
    ax.legend(["x = 0", "x = 20", "x = 55"])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def plot_sedtrails_cdf(
    sedtrails_arrival_times: dict[float, np.ndarray],
    output_path: Path,
) -> None:
    configure_plot_style()
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(1, 1, figsize=(8.3 / 2.54, 10.0 / 2.54))
    ax.grid(True)

    x_positions = sorted(sedtrails_arrival_times.keys())
    colors = cmap(np.linspace(0.1, 0.9, len(x_positions)))
    for x_pos, color in zip(x_positions, colors):
        arrivals = sedtrails_arrival_times[x_pos]
        arrivals = arrivals[np.isfinite(arrivals)]
        if arrivals.size == 0:
            continue
        sorted_arrivals = np.sort(arrivals / 60.0)
        cdf = np.arange(1, sorted_arrivals.size + 1) / sorted_arrivals.size
        ax.plot(sorted_arrivals, cdf, ":", linewidth=1.5, color=color)

    ax.set_ylabel("Cumulative fraction [-]")
    ax.set_xlabel("Time [min]")
    ax.legend([f"x = {x_pos:g}" for x_pos in x_positions])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def build_data_paths(data_dir: Path, flow_lps: int) -> tuple[Path, Path]:
    dem_path = data_dir / f"DEM_{flow_lps}lps.mat"
    tracer_path = data_dir / f"Tracers_Step_length_{flow_lps}lps.mat"
    return dem_path, tracer_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Singh et al. (2023) analytical tracer dispersal benchmark"
    )
    parser.add_argument("--flow-lps", type=int, default=600)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / ".cache" / "singh2023",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Disable automatic downloads from Zenodo",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[0] / "outputs",
    )
    parser.add_argument("--t-exp", type=float, default=300.0)
    parser.add_argument("--flume-length", type=float, default=55.0)
    parser.add_argument(
        "--sedtrails-nc",
        type=Path,
        default=None,
        help="Optional SedTRAILS output netCDF for comparison",
    )
    parser.add_argument(
        "--distance-mode",
        choices=["covered_distance", "streamwise", "euclidean"],
        default="streamwise",
        help="SedTRAILS distance mode for travel distance PDF",
    )
    parser.add_argument(
        "--x-positions",
        type=str,
        default="0,20,55",
        help="Comma-separated x positions for breakthrough curves (meters)",
    )
    args = parser.parse_args()

    ensure_data_files(args.data_dir, args.flow_lps, allow_download=not args.no_download)
    _, tracer_path = build_data_paths(args.data_dir, args.flow_lps)
    if not tracer_path.exists():
        raise FileNotFoundError(
            f"Missing tracer file: {tracer_path}. Provide --data-dir or allow downloads."
        )

    tracer_steps = load_tracer_steps(tracer_path)
    output_path = args.output_dir / "singh2023_analytical_tracer_validation.png"

    sedtrails_distances = None
    sedtrails_arrival_times = None
    if args.sedtrails_nc is not None:
        ds = load_sedtrails_dataset(args.sedtrails_nc)
        sedtrails_distances = compute_sedtrails_distance_at_time(
            ds,
            args.t_exp,
            mode=args.distance_mode,
        )
        x_positions = [float(item) for item in args.x_positions.split(",") if item]
        sedtrails_arrival_times = compute_first_passage_times(ds, x_positions)
        ds.close()

    plot_analytical_vs_data(
        tracer_steps,
        args.t_exp,
        args.flume_length,
        output_path,
        sedtrails_distances=sedtrails_distances,
        sedtrails_arrival_times=sedtrails_arrival_times,
    )

    if sedtrails_arrival_times is not None:
        cdf_output = args.output_dir / "singh2023_sedtrails_cdf.png"
        plot_sedtrails_cdf(sedtrails_arrival_times, cdf_output)


if __name__ == "__main__":
    main()
