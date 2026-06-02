#!/usr/bin/env python3
"""Histotripsy cavitation-volume comparison.

This example compares two clinically relevant histotripsy mechanisms in the
same focused-aperture volume:

1. Classical intrinsic-threshold histotripsy: cavitation probability follows
   the Maxwell et al. single-pulse error-function threshold model.
2. Millisecond-pulse histotripsy: sub-intrinsic-threshold rarefaction drives
   many-cycle bubble growth and collapse. Bubble internal temperature is
   estimated from adiabatic gas compression during collapse; bulk heating is
   not used as the cavitation gate.

The acoustic field is not prescribed as a cosmetic Gaussian. It is evaluated
from the Rayleigh-Sommerfeld focused circular-aperture integral in the existing
HIFU example, then rotated into an axisymmetric volume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

from hifu_procedure_simulation import AcousticConfig, BubbleConfig, _rk4_bubble_step, focused_aperture_intensity

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CLASSICAL_FIGURE = OUTPUT_DIR / "histotripsy_intrinsic_threshold_volume.png"
MS_CAVITATION_FIGURE = OUTPUT_DIR / "histotripsy_ms_pulse_cavitation_volume.png"
BUBBLE_TEMPERATURE_FIGURE = OUTPUT_DIR / "histotripsy_bubble_internal_temperature_volume.png"
COMPARISON_FIGURE = OUTPUT_DIR / "histotripsy_mechanism_compare.png"
PRESSURE_RESPONSE_FIGURE = OUTPUT_DIR / "histotripsy_ms_pressure_response.png"
REPORT_PATH = OUTPUT_DIR / "histotripsy_cavitation_metrics.json"


@dataclass(frozen=True)
class HistotripsyConfig:
    density_kg_m3: float = 1060.0
    sound_speed_m_s: float = 1540.0
    specific_heat_j_kg_k: float = 3600.0
    intrinsic_threshold_pa: float = 28.2e6
    threshold_sigma_pa: float = 0.96e6
    classical_peak_negative_pressure_pa: float = 29.0e6
    ms_peak_negative_pressure_pa: float = 5.0e6
    ms_nucleus_radius_m: float = 0.1e-6
    bubble_temperature_baseline_k: float = 310.15
    inertial_radius_ratio_threshold: float = 1.8
    bubble_response_cycles: int = 16
    bubble_pressure_samples: int = 72


@dataclass(frozen=True)
class VolumeGrid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    xx: np.ndarray
    yy: np.ndarray
    zz: np.ndarray


def build_volume_grid() -> VolumeGrid:
    x = np.linspace(-8.0e-3, 8.0e-3, 81, dtype=np.float64)
    y = np.linspace(-8.0e-3, 8.0e-3, 81, dtype=np.float64)
    z = np.linspace(20.0e-3, 50.0e-3, 101, dtype=np.float64)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return VolumeGrid(x=x, y=y, z=z, xx=xx, yy=yy, zz=zz)


def axisymmetric_intensity_volume(grid: VolumeGrid) -> np.ndarray:
    radial = np.linspace(0.0, np.sqrt(2.0) * 8.0e-3, 161, dtype=np.float64)
    rr, rz = np.meshgrid(radial, grid.z, indexing="ij")
    acoustic = AcousticConfig(
        frequency_hz=1.0e6,
        sound_speed_m_s=1540.0,
        density_kg_m3=1060.0,
        aperture_radius_m=30.0e-3,
        focal_length_m=35.0e-3,
        target_peak_intensity_w_m2=1.0,
        absorption_np_m=8.0,
        aperture_radial_samples=40,
        aperture_angular_samples=96,
    )
    intensity_rz = focused_aperture_intensity(rr, rz, acoustic)
    radius_volume = np.sqrt(grid.xx * grid.xx + grid.yy * grid.yy)
    flat_radius = radius_volume.ravel()
    flat_z_indices = np.tile(np.arange(grid.z.size), grid.x.size * grid.y.size)
    volume = np.empty_like(flat_radius)
    for z_index in range(grid.z.size):
        mask = flat_z_indices == z_index
        volume[mask] = np.interp(flat_radius[mask], radial, intensity_rz[:, z_index])
    volume = volume.reshape(grid.xx.shape)
    peak = float(np.max(volume))
    if peak <= 0.0 or not np.isfinite(peak):
        raise ValueError("Rayleigh-Sommerfeld aperture field produced invalid volume")
    return volume / peak


def pressure_from_normalized_intensity(
    normalized_intensity: np.ndarray,
    peak_negative_pressure_pa: float,
) -> np.ndarray:
    return peak_negative_pressure_pa * np.sqrt(np.clip(normalized_intensity, 0.0, None))


def intrinsic_cavitation_probability(
    pnp_pa: np.ndarray,
    config: HistotripsyConfig,
) -> np.ndarray:
    argument = (pnp_pa - config.intrinsic_threshold_pa) / (config.threshold_sigma_pa * np.sqrt(2.0))
    return 0.5 * (1.0 + erf(argument))


def simulate_ms_bubble_response(
    pressure_levels_pa: np.ndarray,
    config: HistotripsyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Rmax/R0 and gas-collapse temperature for ms-pulse nuclei.

    The branch integrates the Keller-Miksis equation until either the requested
    cycle window completes or an inertial collapse event occurs after prior
    expansion. Collapse is terminated at the van der Waals hard-core radius
    used in standard sonoluminescence bubble models, R_h = R0 / 8.86, rather
    than integrating through the singular post-collapse state. The gas
    temperature estimate follows the adiabatic invariant
    T V^(gamma-1) = constant, hence
    T_c = T0 (Rmax / Rh)^(3(gamma-1)) at hard-core collapse.
    """
    acoustic = AcousticConfig(
        frequency_hz=1.0e6,
        sound_speed_m_s=config.sound_speed_m_s,
        density_kg_m3=config.density_kg_m3,
    )
    bubble = BubbleConfig(
        equilibrium_radius_m=config.ms_nucleus_radius_m,
        dt_s=0.5e-9,
        cycles_per_window=config.bubble_response_cycles,
        control_windows=1,
        target_inertial_radius_ratio=config.inertial_radius_ratio_threshold,
    )
    hard_core_radius = bubble.equilibrium_radius_m / 8.86
    steps = int(round(config.bubble_response_cycles / (acoustic.frequency_hz * bubble.dt_s)))
    radius_ratio = np.empty_like(pressure_levels_pa)
    gas_temperature_k = np.empty_like(pressure_levels_pa)

    for level_index, pressure_pa in enumerate(pressure_levels_pa):
        radius = bubble.equilibrium_radius_m
        velocity = 0.0
        r_max = radius
        r_min_after_expansion = radius
        expanded = False
        collapsed = False
        scale = 1.0
        for step in range(steps):
            time_s = step * bubble.dt_s
            radius, velocity, acceleration = _rk4_bubble_step(
                radius, velocity, time_s, scale, float(pressure_pa), acoustic, bubble
            )
            if not (np.isfinite(radius) and np.isfinite(velocity) and np.isfinite(acceleration)):
                collapsed = expanded
                break
            r_max = max(r_max, radius)
            if radius >= config.inertial_radius_ratio_threshold * bubble.equilibrium_radius_m:
                expanded = True
            if expanded:
                r_min_after_expansion = min(r_min_after_expansion, radius)
                if radius <= hard_core_radius:
                    collapsed = True
                    break
        radius_ratio[level_index] = r_max / bubble.equilibrium_radius_m
        collapse_radius = hard_core_radius if collapsed else max(r_min_after_expansion, hard_core_radius)
        compression_ratio = max(r_max / collapse_radius, 1.0)
        gas_temperature_k[level_index] = config.bubble_temperature_baseline_k * compression_ratio ** (
            3.0 * (bubble.polytropic_exponent - 1.0)
        )

    return radius_ratio, gas_temperature_k


def ms_pressure_response_curve(config: HistotripsyConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pressure_levels = np.linspace(0.0, config.ms_peak_negative_pressure_pa, config.bubble_pressure_samples)
    radius_ratio, gas_temperature = simulate_ms_bubble_response(pressure_levels, config)
    return pressure_levels, radius_ratio, gas_temperature


def ms_pulse_cavitation_maps(
    normalized_intensity: np.ndarray,
    config: HistotripsyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pnp = pressure_from_normalized_intensity(normalized_intensity, config.ms_peak_negative_pressure_pa)
    pressure_levels, radius_ratio_curve, gas_temperature_curve = ms_pressure_response_curve(config)
    radius_ratio = np.interp(pnp.ravel(), pressure_levels, radius_ratio_curve).reshape(pnp.shape)
    gas_temperature = np.interp(pnp.ravel(), pressure_levels, gas_temperature_curve).reshape(pnp.shape)
    curve_peak = max(float(np.max(radius_ratio_curve) - 1.0), 1.0e-30)
    activity = np.log1p(np.maximum(radius_ratio - 1.0, 0.0)) / np.log1p(curve_peak)
    return radius_ratio, activity, gas_temperature


def focal_support_metrics(volume: np.ndarray, grid: VolumeGrid, level: float) -> dict[str, float]:
    if not (0.0 < level <= 1.0):
        raise ValueError("support level must lie in (0, 1]")
    peak_index = np.unravel_index(int(np.argmax(volume)), volume.shape)
    peak = float(volume[peak_index])
    if peak <= 0.0 or not np.isfinite(peak):
        raise ValueError("volume peak must be positive and finite")
    threshold = level * peak
    axial = volume[peak_index[0], peak_index[1], :]
    lateral_x = volume[:, peak_index[1], peak_index[2]]
    lateral_y = volume[peak_index[0], :, peak_index[2]]
    axial_support = np.flatnonzero(axial >= threshold)
    x_support = np.flatnonzero(lateral_x >= threshold)
    y_support = np.flatnonzero(lateral_y >= threshold)
    if axial_support.size == 0 or x_support.size == 0 or y_support.size == 0:
        raise ValueError("support level does not intersect all principal axes")
    lateral_diameter = max(
        float(grid.x[x_support[-1]] - grid.x[x_support[0]]),
        float(grid.y[y_support[-1]] - grid.y[y_support[0]]),
    )
    axial_length = float(grid.z[axial_support[-1]] - grid.z[axial_support[0]])
    return {
        "focus_x_mm": float(grid.x[peak_index[0]] * 1.0e3),
        "focus_y_mm": float(grid.y[peak_index[1]] * 1.0e3),
        "focus_z_mm": float(grid.z[peak_index[2]] * 1.0e3),
        "lateral_diameter_mm": lateral_diameter * 1.0e3,
        "axial_length_mm": axial_length * 1.0e3,
        "axial_to_lateral_ratio": axial_length / max(lateral_diameter, 1.0e-30),
    }


def volume_extent(axis_a: np.ndarray, axis_b: np.ndarray) -> list[float]:
    return [
        float(axis_b[0]) * 1e3,
        float(axis_b[-1]) * 1e3,
        float(axis_a[0]) * 1e3,
        float(axis_a[-1]) * 1e3,
    ]


def mip(volume: np.ndarray, axis: int) -> np.ndarray:
    return np.max(volume, axis=axis)


def save_volume_projection_figure(
    volume: np.ndarray,
    grid: VolumeGrid,
    title: str,
    output: Path,
    colorbar_label: str,
    threshold: float | None = None,
) -> None:
    projections = [
        ("xy", mip(volume, 2), volume_extent(grid.x, grid.y), "y [mm]", "x [mm]"),
        ("xz", mip(volume, 1), volume_extent(grid.x, grid.z), "z [mm]", "x [mm]"),
        ("yz", mip(volume, 0), volume_extent(grid.y, grid.z), "z [mm]", "y [mm]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, (plane, image, extent, xlabel, ylabel) in zip(axes, projections):
        im = ax.imshow(image.T, origin="lower", extent=extent, aspect="auto")
        if threshold is not None:
            ax.contour(
                image.T,
                levels=[threshold],
                colors="white",
                linewidths=1.0,
                extent=extent,
                origin="lower",
            )
        ax.set_title(f"{title} {plane}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_mechanism_comparison(
    probability: np.ndarray,
    ms_activity: np.ndarray,
    radius_ratio: np.ndarray,
    grid: VolumeGrid,
    config: HistotripsyConfig,
) -> None:
    cavitation_mask = probability >= 0.5
    ms_mask = radius_ratio >= config.inertial_radius_ratio_threshold
    overlap = np.logical_and(cavitation_mask, ms_mask).astype(float)
    panels = [
        ("intrinsic probability xz", mip(probability, 1), "probability"),
        ("ms-pulse collapse strength xz", mip(ms_activity, 1), "relative"),
        ("shared cavitation support xz", mip(overlap, 1), "overlap"),
    ]
    extent = volume_extent(grid.x, grid.z)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, (title, image, label) in zip(axes, panels):
        im = ax.imshow(image.T, origin="lower", extent=extent, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("x [mm]")
        fig.colorbar(im, ax=ax, label=label)
    fig.savefig(COMPARISON_FIGURE, dpi=160)
    plt.close(fig)


def save_pressure_response_figure(
    pressure_levels_pa: np.ndarray,
    radius_ratio: np.ndarray,
    gas_temperature_k: np.ndarray,
    config: HistotripsyConfig,
) -> None:
    pressure_mpa = pressure_levels_pa / 1.0e6
    fig, radius_ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    radius_ax.plot(pressure_mpa, radius_ratio, color="tab:blue", label="Rmax/R0")
    radius_ax.axhline(
        config.inertial_radius_ratio_threshold,
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
        label="inertial onset",
    )
    radius_ax.set_xlabel("peak negative pressure [MPa]")
    radius_ax.set_ylabel("maximum radius ratio Rmax/R0", color="tab:blue")
    radius_ax.tick_params(axis="y", labelcolor="tab:blue")
    radius_ax.grid(True, alpha=0.3)

    temperature_ax = radius_ax.twinx()
    temperature_ax.plot(pressure_mpa, gas_temperature_k, color="tab:red", label="collapse gas temperature")
    temperature_ax.set_ylabel("adiabatic gas temperature [K]", color="tab:red")
    temperature_ax.tick_params(axis="y", labelcolor="tab:red")
    temperature_ax.set_yscale("log")

    lines, labels = radius_ax.get_legend_handles_labels()
    temperature_lines, temperature_labels = temperature_ax.get_legend_handles_labels()
    radius_ax.legend(lines + temperature_lines, labels + temperature_labels, loc="upper left")
    radius_ax.set_title("Millisecond-pulse nucleus response")
    fig.savefig(PRESSURE_RESPONSE_FIGURE, dpi=160)
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = HistotripsyConfig()
    grid = build_volume_grid()
    intensity = axisymmetric_intensity_volume(grid)
    classical_pnp = pressure_from_normalized_intensity(
        intensity, config.classical_peak_negative_pressure_pa
    )
    probability = intrinsic_cavitation_probability(classical_pnp, config)
    pressure_levels, response_radius_ratio, response_temperature = ms_pressure_response_curve(config)
    radius_ratio, ms_activity, bubble_temperature = ms_pulse_cavitation_maps(intensity, config)
    support = focal_support_metrics(intensity, grid, level=0.5)

    save_volume_projection_figure(
        probability,
        grid,
        "intrinsic-threshold cavitation",
        CLASSICAL_FIGURE,
        "P(cavitation)",
        threshold=0.5,
    )
    save_volume_projection_figure(
        ms_activity,
        grid,
        "ms-pulse collapse strength",
        MS_CAVITATION_FIGURE,
        "relative log(Rmax/R0)",
        threshold=0.5,
    )
    save_volume_projection_figure(
        bubble_temperature,
        grid,
        "bubble internal temperature",
        BUBBLE_TEMPERATURE_FIGURE,
        "gas temperature [K]",
        threshold=None,
    )
    save_mechanism_comparison(probability, ms_activity, radius_ratio, grid, config)
    save_pressure_response_figure(pressure_levels, response_radius_ratio, response_temperature, config)

    dx = float(grid.x[1] - grid.x[0])
    dy = float(grid.y[1] - grid.y[0])
    dz = float(grid.z[1] - grid.z[0])
    voxel_volume = dx * dy * dz
    metrics = {
        "classical_peak_probability": float(np.max(probability)),
        "classical_p05_volume_mm3": float(np.sum(probability >= 0.5) * voxel_volume * 1e9),
        "ms_activity_peak": float(np.max(ms_activity)),
        "ms_activity_ge_0_5_volume_mm3": float(np.sum(ms_activity >= 0.5) * voxel_volume * 1e9),
        "ms_onset_radius_ratio_volume_mm3": float(
            np.sum(radius_ratio >= config.inertial_radius_ratio_threshold) * voxel_volume * 1e9
        ),
        "ms_max_radius_ratio": float(np.max(radius_ratio)),
        "bubble_internal_peak_temperature_k": float(np.max(bubble_temperature)),
        "focal_halfmax_axial_length_mm": support["axial_length_mm"],
        "focal_halfmax_lateral_diameter_mm": support["lateral_diameter_mm"],
        "focal_halfmax_axial_to_lateral_ratio": support["axial_to_lateral_ratio"],
        "focus_z_mm": support["focus_z_mm"],
        "classical_peak_negative_pressure_mpa": config.classical_peak_negative_pressure_pa / 1e6,
        "ms_peak_negative_pressure_mpa": config.ms_peak_negative_pressure_pa / 1e6,
        "figures": [
            str(CLASSICAL_FIGURE),
            str(MS_CAVITATION_FIGURE),
            str(BUBBLE_TEMPERATURE_FIGURE),
            str(COMPARISON_FIGURE),
            str(PRESSURE_RESPONSE_FIGURE),
        ],
    }
    REPORT_PATH.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
