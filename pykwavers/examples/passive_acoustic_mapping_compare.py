#!/usr/bin/env python3
"""Compare pykwavers DAS-PAM against external k-Wave-style references.

The deterministic case is a single impulsive passive source observed by a
linear receive aperture. The acceptance criterion is source localization on the
same Cartesian grid, plus normalized-map agreement with an independent Python
implementation of the delay law used by KWave.jl's `beamform_delay_and_sum`.
When `--plot` is supplied, the script also reconstructs a true 3-D cavitation
volume and writes side-by-side maximum-intensity projection images.

Run from the repository root after building pykwavers:

    python pykwavers/examples/passive_acoustic_mapping_compare.py --plot
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def synthetic_case() -> dict[str, np.ndarray | float]:
    c0 = 1500.0
    fs = 1.0e6
    dt = 1.0 / fs
    source = np.array([1.0e-3, 0.0, 18.0e-3], dtype=np.float64)
    sensors_x = np.linspace(-6.0e-3, 6.0e-3, 16, dtype=np.float64)
    sensors = np.column_stack([sensors_x, np.zeros_like(sensors_x), np.zeros_like(sensors_x)])
    samples = 96
    passive = np.zeros((sensors.shape[0], samples), dtype=np.float64)
    for idx, sensor in enumerate(sensors):
        delay = np.linalg.norm(source - sensor) / c0
        sample = int(round(delay / dt))
        if 0 <= sample < samples:
            passive[idx, sample] = 1.0

    grid_x = np.linspace(-5.0e-3, 5.0e-3, 41, dtype=np.float64)
    grid_z = np.linspace(10.0e-3, 24.0e-3, 57, dtype=np.float64)
    xx, zz = np.meshgrid(grid_x, grid_z, indexing="ij")
    grid = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()]).astype(np.float64)
    return {
        "c0": c0,
        "fs": fs,
        "dt": dt,
        "source": source,
        "sensors": sensors,
        "passive": passive,
        "grid": grid,
        "grid_x": grid_x,
        "grid_z": grid_z,
        "shape": np.array(xx.shape),
    }


def synthetic_volume_case() -> dict[str, np.ndarray | float]:
    c0 = 1500.0
    fs = 1.0e6
    dt = 1.0 / fs
    source = np.array([1.5e-3, -1.0e-3, 18.0e-3], dtype=np.float64)
    aperture_x = np.linspace(-6.0e-3, 6.0e-3, 4, dtype=np.float64)
    aperture_y = np.linspace(-6.0e-3, 6.0e-3, 4, dtype=np.float64)
    sx, sy = np.meshgrid(aperture_x, aperture_y, indexing="ij")
    sensors = np.column_stack([sx.ravel(), sy.ravel(), np.zeros(sx.size)])
    samples = 96
    passive = np.zeros((sensors.shape[0], samples), dtype=np.float64)
    for idx, sensor in enumerate(sensors):
        delay = np.linalg.norm(source - sensor) / c0
        sample = int(round(delay / dt))
        if 0 <= sample < samples:
            passive[idx, sample] = 1.0

    grid_x = np.linspace(-5.0e-3, 5.0e-3, 25, dtype=np.float64)
    grid_y = np.linspace(-5.0e-3, 5.0e-3, 25, dtype=np.float64)
    grid_z = np.linspace(10.0e-3, 24.0e-3, 35, dtype=np.float64)
    xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)
    return {
        "c0": c0,
        "fs": fs,
        "dt": dt,
        "source": source,
        "sensors": sensors,
        "passive": passive,
        "grid": grid,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_z": grid_z,
        "shape": np.array(xx.shape),
    }


def kwave_julia_style_reference(
    passive: np.ndarray,
    sensors: np.ndarray,
    grid: np.ndarray,
    c0: float,
    dt: float,
    window_size: int,
) -> np.ndarray:
    out = np.zeros(grid.shape[0], dtype=np.float64)
    n_samples = passive.shape[1]
    window = max(1, min(window_size, n_samples))
    for point_idx, point in enumerate(grid):
        summed = np.zeros(window, dtype=np.float64)
        for t in range(window):
            for sensor_idx, sensor in enumerate(sensors):
                sample_idx = t + np.linalg.norm(point - sensor) / (c0 * dt)
                if 0.0 <= sample_idx <= n_samples - 1:
                    lo = int(math.floor(sample_idx))
                    hi = min(lo + 1, n_samples - 1)
                    frac = sample_idx - lo
                    summed[t] += (1.0 - frac) * passive[sensor_idx, lo] + frac * passive[sensor_idx, hi]
        out[point_idx] = float(np.mean(summed * summed))
    return out


def normalize(values: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(values)))
    if peak == 0.0:
        raise ValueError("reference map has zero peak")
    return values / peak


def argmax_position(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return grid[int(np.argmax(values))]


def grid_diagonal(case: dict[str, np.ndarray | float]) -> float:
    axes = [case["grid_x"]]
    if "grid_y" in case:
        axes.append(case["grid_y"])
    axes.append(case["grid_z"])
    spacings = [float(np.min(np.diff(axis))) for axis in axes]
    return math.sqrt(sum(spacing * spacing for spacing in spacings))


def localization_tolerance(case: dict[str, np.ndarray | float]) -> float:
    return max(grid_diagonal(case), float(case["c0"]) * float(case["dt"]))


def run_pykwavers(case: dict[str, np.ndarray | float], window_size: int) -> tuple[np.ndarray, float]:
    import pykwavers as kw

    start = time.perf_counter()
    values = kw.passive_acoustic_map_das(
        case["passive"],
        case["sensors"],
        case["grid"],
        float(case["c0"]),
        float(case["fs"]),
        window_size=window_size,
        apodization="none",
        coherence_weighting=False,
    )
    return np.asarray(values, dtype=np.float64), time.perf_counter() - start


def run_julia_reference(case: dict[str, np.ndarray | float], root: Path) -> dict[str, float] | None:
    if shutil.which("julia") is None:
        return None
    output_dir = root / "pykwavers" / "examples" / "output" / "pam"
    output_dir.mkdir(parents=True, exist_ok=True)
    script = output_dir / "_pam_julia_compare_tmp.jl"
    beamform_path = (root / "external" / "k-wave-julia" / "KWave.jl" / "src" / "reconstruction" / "beamform.jl").as_posix()
    script.write_text(
        f"""
include("{beamform_path}")
using DelimitedFiles
sensor_data = readdlm(ARGS[1], ',')
sensor_positions = readdlm(ARGS[2], ',')[:, 1]
grid = readdlm(ARGS[3], ',')
grid_x = sort(unique(grid[:, 1]))
grid_z = sort(unique(grid[:, 3]))
image = beamform_delay_and_sum(sensor_data, sensor_positions, 1500.0, 1.0e-6, grid_x, grid_z; apodization=:rectangular)
idx = argmax(image)
println(maximum(image))
println(Tuple(idx)[1])
println(Tuple(idx)[2])
""",
        encoding="utf-8",
    )
    try:
        sensor_data_path = output_dir / "_pam_sensor_data.csv"
        sensor_positions_path = output_dir / "_pam_sensor_positions.csv"
        grid_path = output_dir / "_pam_grid.csv"
        np.savetxt(sensor_data_path, case["passive"], delimiter=",")
        np.savetxt(sensor_positions_path, case["sensors"], delimiter=",")
        np.savetxt(grid_path, case["grid"], delimiter=",")
        completed = subprocess.run(
            [
                "julia",
                str(script),
                str(sensor_data_path),
                str(sensor_positions_path),
                str(grid_path),
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
        )
        lines = completed.stdout.strip().splitlines()
        return {"peak": float(lines[0]), "ix": float(lines[1]), "iz": float(lines[2])}
    finally:
        for path in output_dir.glob("_pam_*"):
            path.unlink(missing_ok=True)


def maybe_plot(case: dict[str, np.ndarray | float], rust_map: np.ndarray, ref_map: np.ndarray, root: Path) -> None:
    import matplotlib.pyplot as plt

    shape = tuple(int(v) for v in case["shape"])
    extent = [
        float(case["grid_z"][0]) * 1e3,
        float(case["grid_z"][-1]) * 1e3,
        float(case["grid_x"][0]) * 1e3,
        float(case["grid_x"][-1]) * 1e3,
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [
        ("pykwavers DAS-PAM", normalize(rust_map).reshape(shape)),
        ("KWave.jl delay-law reference", normalize(ref_map).reshape(shape)),
        ("absolute difference", np.abs(normalize(rust_map) - normalize(ref_map)).reshape(shape)),
    ]
    for ax, (title, image) in zip(axes, panels):
        im = ax.imshow(image.T, origin="lower", extent=extent, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("x [mm]")
        fig.colorbar(im, ax=ax)
    output_dir = root / "pykwavers" / "examples" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "passive_acoustic_mapping_compare.png"
    fig.savefig(output, dpi=160)
    print(f"wrote {output}")


def plot_volume_detection(
    case: dict[str, np.ndarray | float],
    rust_map: np.ndarray,
    ref_map: np.ndarray,
    root: Path,
) -> None:
    import matplotlib.pyplot as plt

    shape = tuple(int(v) for v in case["shape"])
    rust_volume = normalize(rust_map).reshape(shape)
    ref_volume = normalize(ref_map).reshape(shape)
    diff_volume = np.abs(rust_volume - ref_volume)
    source = np.asarray(case["source"], dtype=np.float64)
    rust_peak = argmax_position(rust_map, case["grid"])
    ref_peak = argmax_position(ref_map, case["grid"])

    projections = [
        (
            "xy",
            "x [mm]",
            "y [mm]",
            [float(case["grid_y"][0]) * 1e3, float(case["grid_y"][-1]) * 1e3,
             float(case["grid_x"][0]) * 1e3, float(case["grid_x"][-1]) * 1e3],
            lambda volume: volume.max(axis=2),
            lambda point: (point[1] * 1e3, point[0] * 1e3),
        ),
        (
            "xz",
            "z [mm]",
            "x [mm]",
            [float(case["grid_z"][0]) * 1e3, float(case["grid_z"][-1]) * 1e3,
             float(case["grid_x"][0]) * 1e3, float(case["grid_x"][-1]) * 1e3],
            lambda volume: volume.max(axis=1),
            lambda point: (point[2] * 1e3, point[0] * 1e3),
        ),
        (
            "yz",
            "z [mm]",
            "y [mm]",
            [float(case["grid_z"][0]) * 1e3, float(case["grid_z"][-1]) * 1e3,
             float(case["grid_y"][0]) * 1e3, float(case["grid_y"][-1]) * 1e3],
            lambda volume: volume.max(axis=0),
            lambda point: (point[2] * 1e3, point[1] * 1e3),
        ),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    volumes = [
        ("pykwavers detection", rust_volume),
        ("delay-law reference", ref_volume),
        ("absolute difference", diff_volume),
    ]
    for row, (plane, xlabel, ylabel, extent, project, point_xy) in enumerate(projections):
        for col, (title, volume) in enumerate(volumes):
            image = project(volume)
            im = axes[row, col].imshow(image.T, origin="lower", extent=extent, aspect="auto")
            axes[row, col].set_title(f"{title} {plane}")
            axes[row, col].set_xlabel(xlabel)
            axes[row, col].set_ylabel(ylabel)
            true_xy = point_xy(source)
            axes[row, col].scatter([true_xy[0]], [true_xy[1]], marker="+", c="white", s=90, linewidths=1.8)
            if col == 0:
                peak_xy = point_xy(rust_peak)
                axes[row, col].scatter([peak_xy[0]], [peak_xy[1]], marker="o", facecolors="none", edgecolors="cyan", s=80)
            elif col == 1:
                peak_xy = point_xy(ref_peak)
                axes[row, col].scatter([peak_xy[0]], [peak_xy[1]], marker="o", facecolors="none", edgecolors="cyan", s=80)
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.02)

    output_dir = root / "pykwavers" / "examples" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "passive_acoustic_mapping_volume_compare.png"
    fig.savefig(output, dpi=160)
    print(f"wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "external" / "k-wave-python"))
    case = synthetic_case()
    rust_map, rust_seconds = run_pykwavers(case, window_size=1)
    ref_start = time.perf_counter()
    ref_map = kwave_julia_style_reference(
        case["passive"], case["sensors"], case["grid"], float(case["c0"]), float(case["dt"]), 1
    )
    ref_seconds = time.perf_counter() - ref_start

    source = case["source"]
    rust_peak = argmax_position(rust_map, case["grid"])
    ref_peak = argmax_position(ref_map, case["grid"])
    rust_error_m = float(np.linalg.norm(rust_peak - source))
    ref_error_m = float(np.linalg.norm(ref_peak - source))
    normalized_l2 = float(np.linalg.norm(normalize(rust_map) - normalize(ref_map)) / math.sqrt(rust_map.size))

    grid_tolerance = localization_tolerance(case)
    if rust_error_m > grid_tolerance:
        raise AssertionError(f"pykwavers localization error {rust_error_m:g} exceeds one grid diagonal {grid_tolerance:g}")
    if ref_error_m > grid_tolerance:
        raise AssertionError(f"reference localization error {ref_error_m:g} exceeds one grid diagonal {grid_tolerance:g}")

    julia = run_julia_reference(case, root)
    report = {
        "pykwavers_seconds": rust_seconds,
        "python_reference_seconds": ref_seconds,
        "pykwavers_peak_m": rust_peak.tolist(),
        "reference_peak_m": ref_peak.tolist(),
        "source_m": source.tolist(),
        "pykwavers_grid_error_m": rust_error_m,
        "reference_grid_error_m": ref_error_m,
        "normalized_l2": normalized_l2,
        "julia_reference": julia,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.plot:
        maybe_plot(case, rust_map, ref_map, root)
        volume_case = synthetic_volume_case()
        volume_rust_map, volume_rust_seconds = run_pykwavers(volume_case, window_size=1)
        volume_ref_start = time.perf_counter()
        volume_ref_map = kwave_julia_style_reference(
            volume_case["passive"],
            volume_case["sensors"],
            volume_case["grid"],
            float(volume_case["c0"]),
            float(volume_case["dt"]),
            1,
        )
        volume_ref_seconds = time.perf_counter() - volume_ref_start
        volume_rust_peak = argmax_position(volume_rust_map, volume_case["grid"])
        volume_ref_peak = argmax_position(volume_ref_map, volume_case["grid"])
        volume_source = volume_case["source"]
        volume_grid_tolerance = localization_tolerance(volume_case)
        volume_rust_error_m = float(np.linalg.norm(volume_rust_peak - volume_source))
        volume_ref_error_m = float(np.linalg.norm(volume_ref_peak - volume_source))
        if volume_rust_error_m > volume_grid_tolerance:
            raise AssertionError(
                f"pykwavers volume localization error {volume_rust_error_m:g} "
                f"exceeds one grid diagonal {volume_grid_tolerance:g}"
            )
        if volume_ref_error_m > volume_grid_tolerance:
            raise AssertionError(
                f"reference volume localization error {volume_ref_error_m:g} "
                f"exceeds one grid diagonal {volume_grid_tolerance:g}"
            )
        volume_report = {
            "volume_pykwavers_seconds": volume_rust_seconds,
            "volume_python_reference_seconds": volume_ref_seconds,
            "volume_pykwavers_peak_m": volume_rust_peak.tolist(),
            "volume_reference_peak_m": volume_ref_peak.tolist(),
            "volume_source_m": volume_source.tolist(),
            "volume_pykwavers_grid_error_m": volume_rust_error_m,
            "volume_reference_grid_error_m": volume_ref_error_m,
            "volume_normalized_l2": float(
                np.linalg.norm(normalize(volume_rust_map) - normalize(volume_ref_map))
                / math.sqrt(volume_rust_map.size)
            ),
        }
        print(json.dumps(volume_report, indent=2, sort_keys=True))
        plot_volume_detection(volume_case, volume_rust_map, volume_ref_map, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
