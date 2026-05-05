from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_dem_mat(mat_path: Path, key: str | None = None) -> np.ndarray:
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if key is not None:
        return np.asarray(data[key])
    for name, value in data.items():
        if name.startswith("__"):
            continue
        return np.asarray(value)
    raise KeyError(f"Could not find DEM data in {mat_path}")


def write_dep(bathymetry: np.ndarray, output_path: Path, float_format: str = "{:.4f}") -> None:
    if bathymetry.ndim != 2:
        raise ValueError("Bathymetry must be a 2D array for Delft3D DEP output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in bathymetry:
            row_str = " ".join(float_format.format(float(value)) for value in row)
            handle.write(f"{row_str}\n")


@dataclass
class BctParameter:
    name: str
    unit: str


@dataclass
class BctTimeSeriesBlock:
    table_name: str
    contents: str
    location: str
    reference_time: str
    parameters: list[BctParameter]
    times: np.ndarray
    values: np.ndarray
    time_function: str = "non-equidistant"
    time_unit: str = "minutes"
    interpolation: str = "linear"
    time_parameter_unit: str = "[min]"


def _write_keyword_line(handle, keyword: str, value: str) -> None:
    handle.write(f"{keyword:<20} {value}\n")


def _format_quoted(value: str) -> str:
    return f"'{value}'"


def _format_float(value: float, float_format: str) -> str:
    return float_format.format(float(value))


def write_bct(
    boundaries: list[BctTimeSeriesBlock],
    output_path: Path,
    float_format: str = "{:.4f}",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for block in boundaries:
            times = np.asarray(block.times, dtype=float)
            values = np.asarray(block.values, dtype=float)
            if values.ndim != 2:
                raise ValueError("BCT values must be a 2D array (n_times, n_params)")
            if values.shape[0] != times.size:
                raise ValueError("BCT times and values length mismatch")
            if values.shape[1] != len(block.parameters):
                raise ValueError("BCT values must match number of parameters")

            _write_keyword_line(handle, "table-name", _format_quoted(block.table_name))
            _write_keyword_line(handle, "contents", _format_quoted(block.contents))
            _write_keyword_line(handle, "location", _format_quoted(block.location))
            _write_keyword_line(handle, "time-function", _format_quoted(block.time_function))
            _write_keyword_line(handle, "reference-time", block.reference_time)
            _write_keyword_line(handle, "time-unit", _format_quoted(block.time_unit))
            _write_keyword_line(handle, "interpolation", _format_quoted(block.interpolation))
            _write_keyword_line(
                handle,
                "parameter",
                f"{_format_quoted('time ')} unit {_format_quoted(block.time_parameter_unit)}",
            )
            for param in block.parameters:
                _write_keyword_line(
                    handle,
                    "parameter",
                    f"{_format_quoted(param.name)} unit {_format_quoted(param.unit)}",
                )
            _write_keyword_line(handle, "records-in-table", str(times.size))

            for time_value, row in zip(times, values):
                row_values = " ".join(_format_float(value, float_format) for value in row)
                handle.write(f"{_format_float(time_value, float_format)} {row_values}\n")
            handle.write("\n")


def make_discharge_boundary_block(
    table_name: str,
    location: str,
    times_minutes: np.ndarray,
    discharge_end_a: np.ndarray,
    discharge_end_b: np.ndarray | None = None,
    reference_time: str = "20250101",
    contents: str = "logarithmic",
    interpolation: str = "linear",
) -> BctTimeSeriesBlock:
    times = np.asarray(times_minutes, dtype=float)
    q_a = np.asarray(discharge_end_a, dtype=float)
    if discharge_end_b is None:
        q_b = q_a
    else:
        q_b = np.asarray(discharge_end_b, dtype=float)

    if q_a.shape != times.shape or q_b.shape != times.shape:
        raise ValueError("Discharge arrays must match the times array shape")

    values = np.column_stack([q_a, q_b])
    parameters = [
        BctParameter("flux/discharge (q) end A", "[m3/s]"),
        BctParameter("flux/discharge (q) end B", "[m3/s]"),
    ]
    return BctTimeSeriesBlock(
        table_name=table_name,
        contents=contents,
        location=location,
        reference_time=reference_time,
        parameters=parameters,
        times=times,
        values=values,
        time_unit="minutes",
        interpolation=interpolation,
        time_parameter_unit="[min]",
    )


def make_water_level_boundary_block(
    table_name: str,
    location: str,
    times_minutes: np.ndarray,
    water_level_end_a: np.ndarray,
    water_level_end_b: np.ndarray | None = None,
    reference_time: str = "20250101",
    contents: str = "uniform",
    interpolation: str = "linear",
) -> BctTimeSeriesBlock:
    times = np.asarray(times_minutes, dtype=float)
    z_a = np.asarray(water_level_end_a, dtype=float)
    if water_level_end_b is None:
        z_b = z_a
    else:
        z_b = np.asarray(water_level_end_b, dtype=float)

    if z_a.shape != times.shape or z_b.shape != times.shape:
        raise ValueError("Water level arrays must match the times array shape")

    values = np.column_stack([z_a, z_b])
    parameters = [
        BctParameter("water elevation (z) end A", "[m]"),
        BctParameter("water elevation (z) end B", "[m]"),
    ]
    return BctTimeSeriesBlock(
        table_name=table_name,
        contents=contents,
        location=location,
        reference_time=reference_time,
        parameters=parameters,
        times=times,
        values=values,
        time_unit="minutes",
        interpolation=interpolation,
        time_parameter_unit="[min]",
    )


def write_flume_bct(
    output_path: Path,
    times_minutes: np.ndarray,
    discharge_upstream: np.ndarray,
    water_level_downstream: np.ndarray,
    reference_time: str = "20250101",
    upstream_table_name: str = "bct1",
    upstream_location: str = "upstream",
    downstream_table_name: str = "bct2",
    downstream_location: str = "downstream",
) -> None:
    upstream_block = make_discharge_boundary_block(
        table_name=upstream_table_name,
        location=upstream_location,
        times_minutes=times_minutes,
        discharge_end_a=discharge_upstream,
        discharge_end_b=None,
        reference_time=reference_time,
        contents="logarithmic",
        interpolation="linear",
    )
    downstream_block = make_water_level_boundary_block(
        table_name=downstream_table_name,
        location=downstream_location,
        times_minutes=times_minutes,
        water_level_end_a=water_level_downstream,
        water_level_end_b=None,
        reference_time=reference_time,
        contents="uniform",
        interpolation="linear",
    )
    write_bct([upstream_block, downstream_block], output_path)


def write_bct(boundaries: list[BctBoundarySeries], output_path: Path) -> None:
    raise NotImplementedError(
        "BCT format varies by Delft3D configuration. "
        "Provide a sample .bct file or template so the writer can match the expected layout."
    )
