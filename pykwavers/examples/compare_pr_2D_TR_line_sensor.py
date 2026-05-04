"""
Comparison: k-wave-python pr_2D_TR_line_sensor vs pykwavers.

This script reuses the vendored 2D line-sensor initial-pressure example
configuration, then compares:
  1. the forward sensor matrices produced by k-wave-python and pykwavers
  2. the k-wave-python time-reversal reconstruction against the pykwavers
     native time-reversal binding on the same reference sensor data
  3. the FFT line reconstruction against the vendored k-wave-python reference
     on the same aligned sensor matrix

The reconstruction comparisons use the reference k-wave sensor matrices so the
metrics isolate reconstruction parity rather than forward-solver drift.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    compute_trace_metrics,
    save_side_by_side_parity_figure,
)

bootstrap_example_paths()

TR_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "pr_2D_TR_line_sensor_time_reversal_compare.png"
FFT_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "pr_2D_TR_line_sensor_fft_compare.png"
PRESSURE_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "pr_2D_TR_line_sensor_pressure_compare.png"

import pykwavers as kw
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.reconstruction import TimeReversal
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

from compare_pr_2D_FFT_line_sensor import (
    _build_example_inputs,
    _build_kgrid,
    _reconstruct_line_sensor_kwave,
    _reconstruct_line_sensor_pykwavers,
    _resample_reconstruction_to_source_grid,
    run_kwave_reference,
    run_pykwavers_reference,
)


def _build_line_sensor_positions(grid_points: tuple[int, int], spacing: tuple[float, float]) -> np.ndarray:
    """Return the inner-grid line sensor positions in `(n_sensors, 3)` order."""
    ny = int(grid_points[1])
    y = np.arange(ny, dtype=np.float64) * float(spacing[1])
    x = np.zeros(ny, dtype=np.float64)
    z = np.zeros(ny, dtype=np.float64)
    return np.stack((x, y, z), axis=1)


def _align_sensor_matrix(pressure: np.ndarray) -> np.ndarray:
    """Match the sensor-time alignment used by the FFT parity example."""
    aligned = np.asarray(pressure, dtype=np.float64)
    if aligned.ndim != 2:
        raise AssertionError(f"expected a 2-D sensor matrix, got {aligned.shape}")
    if aligned.shape[1] > 1:
        aligned = aligned[:, 1:]
    return aligned


def _reconstruct_time_reversal_kwave(
    *,
    kgrid: kWaveGrid,
    medium: kWaveMedium,
    sensor,
    pressure: np.ndarray,
    pml_size: tuple[int, int],
) -> np.ndarray:
    """Reconstruct the source image using vendored k-wave-python time reversal."""
    sensor_recon = deepcopy(sensor)
    sensor_recon.recorded_pressure = np.asarray(pressure, dtype=np.float64)

    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=Vector(list(pml_size)),
        smooth_p0=False,
        save_to_disk=True,
    )
    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=False,
        verbose_level=0,
        show_sim_log=False,
    )

    tr = TimeReversal(kgrid, medium, sensor_recon)
    p0_recon = tr(kspaceFirstOrder2D, simulation_options, execution_options)
    return np.asarray(p0_recon, dtype=np.float64)


def _reconstruct_time_reversal_pykwavers(
    *,
    pressure: np.ndarray,
    sensor_positions: np.ndarray,
    grid_points: tuple[int, int],
    spacing: tuple[float, float],
    sound_speed: float,
    dt: float,
    pml_size: tuple[int, int],
) -> np.ndarray:
    """Reconstruct the source image using the native pykwavers binding."""
    grid = kw.Grid(
        nx=int(grid_points[0]),
        ny=int(grid_points[1]),
        nz=1,
        dx=float(spacing[0]),
        dy=float(spacing[1]),
        dz=float(spacing[0]),
    )
    reconstruction = kw.time_reversal_reconstruction(
        np.asarray(pressure, dtype=np.float64),
        np.asarray(sensor_positions, dtype=np.float64),
        grid,
        float(sound_speed),
        float(1.0 / dt),
        pml_size=int(pml_size[0]),
    )
    return np.asarray(reconstruction, dtype=np.float64).squeeze()


def run_comparison() -> dict[str, object]:
    """Run the comparison and return forward and reconstruction metrics."""
    inputs = _build_example_inputs()
    grid_points = tuple(int(v) for v in inputs["grid_points"])
    spacing = tuple(float(v) for v in inputs["spacing"])
    sound_speed = float(inputs["sound_speed"])
    pml_size = tuple(int(v) for v in inputs["pml_size"])
    source_pressure = np.asarray(inputs["source"].p0, dtype=np.float64)

    kw_results = run_kwave_reference()
    py_results = run_pykwavers_reference()

    kw_pressure_raw = np.asarray(kw_results["pressure"], dtype=np.float64)
    py_pressure_raw = np.asarray(py_results["pressure"], dtype=np.float64)
    kw_pressure = _align_sensor_matrix(kw_pressure_raw)
    py_pressure = _align_sensor_matrix(py_pressure_raw)
    if kw_pressure.shape != py_pressure.shape:
        raise AssertionError(f"sensor matrix shape mismatch: {kw_pressure.shape} != {py_pressure.shape}")

    kgrid = _build_kgrid(grid_points, spacing, sound_speed)
    sensor_positions = _build_line_sensor_positions(grid_points, spacing)

    reference_pressure = kw_pressure_raw

    kw_tr = _reconstruct_time_reversal_kwave(
        kgrid=kgrid,
        medium=inputs["medium"],
        sensor=inputs["sensor"],
        pressure=reference_pressure,
        pml_size=pml_size,
    )
    py_tr = _reconstruct_time_reversal_pykwavers(
        pressure=reference_pressure,
        sensor_positions=sensor_positions,
        grid_points=grid_points,
        spacing=spacing,
        sound_speed=sound_speed,
        dt=float(kw_results["dt"]),
        pml_size=pml_size,
    )

    kw_fft = _resample_reconstruction_to_source_grid(
        _reconstruct_line_sensor_kwave(
            kw_pressure,
            spacing=spacing,
            dt=float(kw_results["dt"]),
            sound_speed=sound_speed,
        ),
        kgrid=kgrid,
        dt=float(kw_results["dt"]),
        sound_speed=sound_speed,
    )
    py_fft = _resample_reconstruction_to_source_grid(
        _reconstruct_line_sensor_pykwavers(
            kw_pressure,
            spacing=spacing,
            dt=float(py_results["dt"]),
            sound_speed=sound_speed,
        ),
        kgrid=kgrid,
        dt=float(py_results["dt"]),
        sound_speed=sound_speed,
    )

    if kw_tr.shape != py_tr.shape:
        raise AssertionError(f"time-reversal shape mismatch: {kw_tr.shape} != {py_tr.shape}")
    if kw_fft.shape != py_fft.shape:
        raise AssertionError(f"FFT reconstruction shape mismatch: {kw_fft.shape} != {py_fft.shape}")

    time_reversal_summary = compute_image_metrics(kw_tr, py_tr)
    fft_summary = compute_image_metrics(kw_fft, py_fft)

    reference_metrics = {
        "kwave_time_reversal": compute_image_metrics(source_pressure, kw_tr),
        "pykwavers_time_reversal": compute_image_metrics(source_pressure, py_tr),
        "kwave_fft": compute_image_metrics(source_pressure, kw_fft),
        "pykwavers_fft": compute_image_metrics(source_pressure, py_fft),
    }

    representative_rows = [0, kw_pressure.shape[0] // 2, kw_pressure.shape[0] - 1]
    trace_metrics = {
        row: compute_trace_metrics(kw_pressure[row], py_pressure[row])
        for row in dict.fromkeys(representative_rows)
    }

    return {
        "kwave": {
            "pressure": kw_pressure,
            "time_reversal": kw_tr,
            "fft_reconstruction": kw_fft,
        },
        "pykwavers": {
            "pressure": py_pressure,
            "time_reversal": py_tr,
            "fft_reconstruction": py_fft,
        },
        "summary": time_reversal_summary,
        "fft_summary": fft_summary,
        "reference_metrics": reference_metrics,
        "trace_metrics": trace_metrics,
    }


_R_TARGET = 0.93
_RMS_MIN = 0.80
_RMS_MAX = 1.20
_PSNR_TARGET = 20.0


def main() -> int:
    """Execute the comparison, print metric summaries, and write a metrics file."""
    result = run_comparison()

    tr_r = float(result["summary"]["pearson_r"])
    tr_rms = float(result["summary"]["rms_ratio"])
    tr_psnr = float(result["summary"]["psnr_db"])
    fft_r = float(result["fft_summary"]["pearson_r"])
    fft_rms = float(result["fft_summary"]["rms_ratio"])
    fft_psnr = float(result["fft_summary"]["psnr_db"])

    # PASS if both TR and FFT reconstructions meet targets
    tr_checks = {
        "pearson_r": tr_r >= _R_TARGET,
        "rms_ratio": _RMS_MIN <= tr_rms <= _RMS_MAX,
        "psnr_db": tr_psnr >= _PSNR_TARGET,
    }
    fft_checks = {
        "pearson_r": fft_r >= _R_TARGET,
        "rms_ratio": _RMS_MIN <= fft_rms <= _RMS_MAX,
        "psnr_db": fft_psnr >= _PSNR_TARGET,
    }
    overall_status = "PASS" if all(tr_checks.values()) and all(fft_checks.values()) else "FAIL"

    kw_ref = result["reference_metrics"]["kwave_time_reversal"]
    py_ref = result["reference_metrics"]["pykwavers_time_reversal"]

    print("=" * 80)
    print("k-wave-python pr_2D_TR_line_sensor vs pykwavers")
    print("=" * 80)
    print(f"Time reversal Pearson r: {tr_r:.6f}  (target >= {_R_TARGET})")
    print(f"Time reversal RMS ratio: {tr_rms:.6f}  (target [{_RMS_MIN}, {_RMS_MAX}])")
    print(f"Time reversal PSNR [dB]: {tr_psnr:.6f}  (target >= {_PSNR_TARGET})")
    print(f"FFT Pearson r:           {fft_r:.6f}  (target >= {_R_TARGET})")
    print(f"FFT RMS ratio:           {fft_rms:.6f}  (target [{_RMS_MIN}, {_RMS_MAX}])")
    print(f"FFT PSNR [dB]:           {fft_psnr:.6f}  (target >= {_PSNR_TARGET})")
    print(f"k-Wave TR vs p0 r:       {kw_ref['pearson_r']:.6f}")
    print(f"pykwavers TR vs p0 r:    {py_ref['pearson_r']:.6f}")
    print(f"Status:                  {overall_status}")

    for row, metrics in result["trace_metrics"].items():
        print(
            f"Sensor row {row}: corr={metrics['pearson_r']:.6f}, "
            f"rmse={metrics['rmse']:.6e}, peak_ratio={metrics['peak_ratio']:.6f}"
        )

    # --- Structured metrics file ---
    output_path = DEFAULT_OUTPUT_DIR / "pr_2D_TR_line_sensor_metrics.txt"
    tr_figure_path = save_side_by_side_parity_figure(
        result["kwave"]["time_reversal"],
        result["pykwavers"]["time_reversal"],
        TR_FIGURE_PATH,
        title="pr_2D_TR_line_sensor time-reversal parity",
        reference_label="k-wave-python TR",
        candidate_label="pykwavers TR",
        cmap="seismic",
    )
    fft_figure_path = save_side_by_side_parity_figure(
        result["kwave"]["fft_reconstruction"],
        result["pykwavers"]["fft_reconstruction"],
        FFT_FIGURE_PATH,
        title="pr_2D_TR_line_sensor FFT reconstruction parity",
        reference_label="k-wave-python FFT",
        candidate_label="pykwavers FFT",
        cmap="seismic",
    )
    pressure_figure_path = save_side_by_side_parity_figure(
        result["kwave"]["pressure"],
        result["pykwavers"]["pressure"],
        PRESSURE_FIGURE_PATH,
        title="pr_2D_TR_line_sensor forward sensor parity",
        reference_label="k-wave-python pressure",
        candidate_label="pykwavers pressure",
        cmap="seismic",
    )
    with output_path.open("w") as fh:
        fh.write("pr_2D_TR_line_sensor parity metrics\n")
        fh.write(f"parity_status: {overall_status}\n\n")
        fh.write("Time-reversal reconstruction (kwave vs pykwavers):\n")
        fh.write(f"  pearson_r = {tr_r:.6f}  (target >= {_R_TARGET})\n")
        fh.write(f"  rms_ratio = {tr_rms:.6f}  (target [{_RMS_MIN}, {_RMS_MAX}])\n")
        fh.write(f"  psnr_db   = {tr_psnr:.6f}  (target >= {_PSNR_TARGET} dB)\n\n")
        fh.write("FFT line reconstruction (kwave vs pykwavers):\n")
        fh.write(f"  pearson_r = {fft_r:.6f}  (target >= {_R_TARGET})\n")
        fh.write(f"  rms_ratio = {fft_rms:.6f}  (target [{_RMS_MIN}, {_RMS_MAX}])\n")
        fh.write(f"  psnr_db   = {fft_psnr:.6f}  (target >= {_PSNR_TARGET} dB)\n\n")
        fh.write("Reconstruction vs ground-truth p0:\n")
        fh.write(f"  kwave     TR pearson_r = {kw_ref['pearson_r']:.6f}\n")
        fh.write(f"  pykwavers TR pearson_r = {py_ref['pearson_r']:.6f}\n\n")
        fh.write("Forward sensor traces (representative rows):\n")
        for row, m in result["trace_metrics"].items():
            fh.write(
                f"  row={row}: pearson_r={m['pearson_r']:.6f}  "
                f"rms_ratio={m['rms_ratio']:.6f}  rmse={m['rmse']:.6e}\n"
            )
        fh.write(f"\nfigure_time_reversal: {tr_figure_path.name}\n")
        fh.write(f"figure_fft: {fft_figure_path.name}\n")
        fh.write(f"figure_pressure: {pressure_figure_path.name}\n")
    print(f"Metrics written to: {output_path}")
    print(f"Figures written to: {tr_figure_path}, {fft_figure_path}, {pressure_figure_path}")

    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
