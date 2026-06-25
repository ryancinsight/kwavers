"""Chapter 30: intravascular ultrasound imaging and therapy.

The chapter simulation is deterministic. It generates an IVUS-like coronary
cross-section, a dual-frequency catheter layout, radial B-mode data, and a
localized microbubble delivery field under docs/book/figures/ch30/.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge

try:
    import pykwavers as kw
    _HAS_PYKWAVERS = True
except ImportError:
    kw = None
    _HAS_PYKWAVERS = False


BOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BOOK_DIR.parents[3]
OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch30"

RHO_TISSUE_KG_M3 = 1060.0
C_TISSUE_M_S = 1540.0
CP_TISSUE_J_KG_K = 3600.0
T0_C = 37.0
DB_CM_MHZ_TO_NP_M_MHZ = 100.0 / 8.686


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    frame_shape: tuple[int, int]
    annotated_boundaries: tuple[str, ...]
    imaging_frequency_hz: float
    source_urls: tuple[str, ...]
    local_role: str


@dataclass(frozen=True)
class TransducerDesign:
    catheter_radius_m: float = 0.55e-3
    imaging_frequency_hz: float = 20.0e6
    imaging_elements: int = 64
    therapy_frequency_hz: float = 1.5e6
    therapy_pressure_pa: float = 300.0e3
    therapy_sector_count: int = 4
    therapy_azimuth_rad: float = -0.72
    therapy_sector_width_rad: float = 0.50
    therapy_sonication_s: float = 45.0
    therapy_duty_cycle: float = 0.05
    maximum_imaging_radius_m: float = 10.0e-3
    pullback_step_m: float = 0.5e-3


@dataclass(frozen=True)
class VesselPhantom:
    x_m: np.ndarray
    y_m: np.ndarray
    radius_m: np.ndarray
    theta_rad: np.ndarray
    labels: np.ndarray
    sound_speed_m_s: np.ndarray
    density_kg_m3: np.ndarray
    attenuation_db_cm_mhz: np.ndarray
    backscatter: np.ndarray
    lumen_mask: np.ndarray
    eel_mask: np.ndarray
    plaque_mask: np.ndarray
    fibrous_cap_mask: np.ndarray
    lipid_mask: np.ndarray
    calcium_mask: np.ndarray


def default_dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        name="IVUS-Net / IVUS Challenge public segmentation corpus",
        frame_shape=(384, 384),
        annotated_boundaries=("lumen", "media-adventitia"),
        imaging_frequency_hz=20.0e6,
        source_urls=(
            "https://github.com/Kulbear/ivus-segmentation-icsm2018",
            "https://arxiv.org/abs/1806.03583",
            "http://www.cvc.uab.es/IVUSchallenge2011/dataset.html",
        ),
        local_role=(
            "External validation target for lumen and vessel-wall contours; "
            "clinical frames are not redistributed by this repository."
        ),
    )


def angle_difference(a: np.ndarray | float, b: float) -> np.ndarray:
    return np.angle(np.exp(1j * (np.asarray(a) - b)))


def vessel_phantom(
    n: int = 384,
    fov_m: float = 12.0e-3,
    design: TransducerDesign | None = None,
    seed: int = 30,
) -> VesselPhantom:
    design = design or TransducerDesign()
    axis = np.linspace(-0.5 * fov_m, 0.5 * fov_m, n, dtype=float)
    x_m, y_m = np.meshgrid(axis, axis, indexing="ij")
    radius_m = np.hypot(x_m, y_m)
    theta_rad = np.arctan2(y_m, x_m)

    plaque_angle = angle_difference(theta_rad, design.therapy_azimuth_rad)
    plaque_weight = np.exp(-0.5 * (plaque_angle / 0.60) ** 2)
    lumen_boundary = (
        1.70e-3
        - 0.45e-3 * plaque_weight
        + 0.08e-3 * np.cos(3.0 * theta_rad + 0.30)
    )
    eel_boundary = (
        3.20e-3
        + 0.22e-3 * plaque_weight
        + 0.10e-3 * np.sin(2.0 * theta_rad - 0.40)
    )

    catheter = radius_m <= design.catheter_radius_m
    lumen = (radius_m > design.catheter_radius_m) & (radius_m <= lumen_boundary)
    wall = (radius_m > lumen_boundary) & (radius_m <= eel_boundary)
    plaque = wall & (plaque_weight > 0.28)
    cap = plaque & ((radius_m - lumen_boundary) < 0.24e-3)
    lipid = plaque & ((radius_m - lumen_boundary) > 0.45e-3) & (plaque_weight > 0.62)
    calcium_angle = angle_difference(theta_rad, 1.35)
    calcium = wall & (np.abs(calcium_angle) < 0.20) & (radius_m > eel_boundary - 0.42e-3)
    eel_mask = radius_m <= eel_boundary

    labels = np.zeros((n, n), dtype=np.uint8)
    labels[catheter] = 1
    labels[lumen] = 2
    labels[wall] = 3
    labels[plaque] = 4
    labels[cap] = 5
    labels[lipid] = 6
    labels[calcium] = 7

    sound_speed = np.full((n, n), 343.0, dtype=float)
    density = np.full((n, n), 1.2, dtype=float)
    attenuation = np.full((n, n), 0.02, dtype=float)
    sound_speed[lumen] = 1570.0
    density[lumen] = 1060.0
    attenuation[lumen] = 0.12
    sound_speed[wall] = 1585.0
    density[wall] = 1080.0
    attenuation[wall] = 0.65
    sound_speed[plaque] = 1520.0
    density[plaque] = 1040.0
    attenuation[plaque] = 0.95
    sound_speed[cap] = 1630.0
    density[cap] = 1120.0
    attenuation[cap] = 0.70
    sound_speed[lipid] = 1450.0
    density[lipid] = 980.0
    attenuation[lipid] = 1.15
    sound_speed[calcium] = 2900.0
    density[calcium] = 1850.0
    attenuation[calcium] = 4.0

    rng = np.random.default_rng(seed)
    impedance = sound_speed * density
    gx, gy = np.gradient(impedance)
    interface_echo = np.hypot(gx, gy)
    interface_echo /= max(float(interface_echo.max()), 1.0)
    speckle = rng.rayleigh(scale=0.18, size=(n, n))
    backscatter = interface_echo + speckle * (0.15 * lumen + 0.80 * wall + 0.55 * lipid)
    backscatter[catheter] = 0.0
    backscatter /= max(float(backscatter.max()), 1.0)

    return VesselPhantom(
        x_m=x_m,
        y_m=y_m,
        radius_m=radius_m,
        theta_rad=theta_rad,
        labels=labels,
        sound_speed_m_s=sound_speed,
        density_kg_m3=density,
        attenuation_db_cm_mhz=attenuation,
        backscatter=backscatter,
        lumen_mask=lumen,
        eel_mask=eel_mask,
        plaque_mask=plaque,
        fibrous_cap_mask=cap,
        lipid_mask=lipid,
        calcium_mask=calcium,
    )


def nearest_indices(phantom: VesselPhantom, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = phantom.x_m.shape[0]
    x0 = float(phantom.x_m[0, 0])
    y0 = float(phantom.y_m[0, 0])
    dx = float(phantom.x_m[1, 0] - phantom.x_m[0, 0])
    dy = float(phantom.y_m[0, 1] - phantom.y_m[0, 0])
    ix = np.clip(np.rint((x - x0) / dx).astype(int), 0, n - 1)
    iy = np.clip(np.rint((y - y0) / dy).astype(int), 0, n - 1)
    return ix, iy


def gaussian_kernel(samples: int, sigma: float) -> np.ndarray:
    x = np.arange(samples, dtype=float) - 0.5 * (samples - 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / float(kernel.sum())


# --- Physics delegated to the kwavers Rust core (PyO3 kernels) -----------------
# pykwavers is a thin PyO3 wrapper: the closed-form acoustic relations below run
# in Rust (kwavers_python::analytical_bindings::{thermal, imaging}). The Python
# fallbacks are bit-identical to the kernels and only run when the compiled
# extension is unavailable so the book can still build.


def intensity_w_m2(pressure: np.ndarray, rho: float, c: float) -> np.ndarray:
    """Time-averaged intensity I = p²/(2ρc) [W/m²] via kw.acoustic_intensity_from_amplitude."""
    if _HAS_PYKWAVERS:
        flat = np.ascontiguousarray(pressure.ravel(), dtype=np.float64)
        return np.asarray(
            kw.acoustic_intensity_from_amplitude(flat, float(rho), float(c))
        ).reshape(pressure.shape)
    return pressure**2 / (2.0 * rho * c)


def adiabatic_delta_t_kelvin(
    heat_source: np.ndarray, tau_s: float, rho: float, specific_heat: float
) -> np.ndarray:
    """No-perfusion ΔT = Q·τ/(ρ·c_p) [K] via kw.adiabatic_temperature_rise_kelvin."""
    if _HAS_PYKWAVERS:
        q_flat = np.ascontiguousarray(heat_source.ravel(), dtype=np.float64)
        tau = np.full(q_flat.size, float(tau_s), dtype=np.float64)
        return np.asarray(
            kw.adiabatic_temperature_rise_kelvin(q_flat, tau, float(rho), float(specific_heat))
        ).reshape(heat_source.shape)
    return heat_source * tau_s / (rho * specific_heat)


def bmode_log_compress(envelope: np.ndarray, reference: float, floor_db: float = -60.0) -> np.ndarray:
    """Fixed-reference log compression db = clip(20·log10(env/ref), floor, 0) via kw.bmode_db_fixed_reference."""
    if _HAS_PYKWAVERS:
        flat = np.ascontiguousarray(envelope.ravel(), dtype=np.float64)
        return np.asarray(
            kw.bmode_db_fixed_reference(flat, float(reference), float(floor_db))
        ).reshape(envelope.shape)
    db = 20.0 * np.log10(np.maximum(np.maximum(envelope, 0.0) / max(reference, 1.0e-300), 1.0e-12))
    return np.clip(db, floor_db, 0.0)


def simulate_bmode(
    phantom: VesselPhantom,
    design: TransducerDesign,
    radial_samples: int = 384,
    angle_samples: int = 384,
) -> dict[str, np.ndarray]:
    r_axis = np.linspace(design.catheter_radius_m, design.maximum_imaging_radius_m, radial_samples)
    theta_axis = np.linspace(-np.pi, np.pi, angle_samples, endpoint=False)
    rr, tt = np.meshgrid(r_axis, theta_axis, indexing="ij")
    ix, iy = nearest_indices(phantom, rr * np.cos(tt), rr * np.sin(tt))
    alpha_np = phantom.attenuation_db_cm_mhz[ix, iy] * DB_CM_MHZ_TO_NP_M_MHZ
    frequency_mhz = design.imaging_frequency_hz / 1.0e6
    attenuation = np.exp(-2.0 * alpha_np * frequency_mhz * (rr - design.catheter_radius_m))
    rf = phantom.backscatter[ix, iy] * attenuation
    rf += 0.10 * np.exp(-((rr - design.catheter_radius_m) / 0.22e-3) ** 2)

    # Hilbert-transform envelope |z(t)| (§9.1.3, Theorem 9.1) per RF line. The
    # analytic-signal magnitude comes from the Rust core; the Gaussian-smoothing
    # path is only a no-pykwavers fallback (not a true envelope).
    envelope = np.empty_like(rf)
    if _HAS_PYKWAVERS:
        for col in range(angle_samples):
            envelope[:, col] = np.asarray(kw.bmode_envelope(np.ascontiguousarray(rf[:, col])))
    else:
        kernel = gaussian_kernel(11, 1.8)
        for col in range(angle_samples):
            envelope[:, col] = np.convolve(rf[:, col], kernel, mode="same")
    envelope = np.maximum(envelope, 1.0e-9)
    # Log-compression in the Rust core (db is clamped to [-60, 0]); the final
    # normalized image is identical to the unclamped-then-clip form.
    db = bmode_log_compress(envelope, float(envelope.max()), -60.0)
    bmode = np.clip((db + 60.0) / 60.0, 0.0, 1.0)
    cartesian = scan_convert(bmode, r_axis, theta_axis, phantom)
    return {
        "r_axis_m": r_axis,
        "theta_axis_rad": theta_axis,
        "polar": bmode,
        "cartesian": cartesian,
        "db": db,
    }


def scan_convert(
    polar: np.ndarray,
    r_axis: np.ndarray,
    theta_axis: np.ndarray,
    phantom: VesselPhantom,
) -> np.ndarray:
    r = phantom.radius_m
    theta = phantom.theta_rad
    dr = float(r_axis[1] - r_axis[0])
    dtheta = float(theta_axis[1] - theta_axis[0])
    ri = np.clip(np.rint((r - r_axis[0]) / dr).astype(int), 0, r_axis.size - 1)
    ti = np.mod(np.rint((theta - theta_axis[0]) / dtheta).astype(int), theta_axis.size)
    image = polar[ri, ti]
    image[r < r_axis[0]] = 0.0
    image[r > r_axis[-1]] = 0.0
    return image


def simulate_therapy(phantom: VesselPhantom, design: TransducerDesign) -> dict[str, np.ndarray | float]:
    angle_gain = np.exp(
        -0.5 * (angle_difference(phantom.theta_rad, design.therapy_azimuth_rad) / design.therapy_sector_width_rad) ** 2
    )
    range_m = np.maximum(phantom.radius_m - design.catheter_radius_m, 0.0)
    pressure = design.therapy_pressure_pa * angle_gain * np.exp(-range_m / 3.2e-3)
    pressure[phantom.radius_m <= design.catheter_radius_m] = 0.0

    # Effective amplitude attenuation [Np/m] at the therapy frequency. It is a
    # spatially-varying tissue field, so the scalar-α acoustic_heat_source_density
    # kernel cannot consume it directly; instead I = p²/(2ρc) is computed by the
    # Rust intensity kernel and the per-voxel Beer–Lambert weighting (Q = 2αI) is
    # applied to that Rust intensity.
    alpha_np = phantom.attenuation_db_cm_mhz * DB_CM_MHZ_TO_NP_M_MHZ
    frequency_mhz = design.therapy_frequency_hz / 1.0e6
    alpha_eff_np_m = alpha_np * frequency_mhz
    intensity = intensity_w_m2(pressure, RHO_TISSUE_KG_M3, C_TISSUE_M_S)
    absorbed_power = 2.0 * alpha_eff_np_m * intensity * design.therapy_duty_cycle
    delta_t = adiabatic_delta_t_kelvin(
        absorbed_power, design.therapy_sonication_s, RHO_TISSUE_KG_M3, CP_TISSUE_J_KG_K
    )

    wall = phantom.eel_mask & ~phantom.lumen_mask
    wall_target = phantom.fibrous_cap_mask | phantom.lipid_mask
    radial_band = np.exp(-((range_m - 1.75e-3) / 1.2e-3) ** 2)
    acoustic_radiation_force = 2.0 * alpha_eff_np_m * intensity / C_TISSUE_M_S
    deposition = acoustic_radiation_force * radial_band * (0.20 * wall + 0.80 * wall_target)
    deposition /= max(float(deposition.max()), 1.0e-12)
    delivered_fraction = 1.0 - np.exp(-3.0 * deposition)
    # MI = p_neg [MPa] / sqrt(f [MHz]) via kw.mechanical_index(p_neg_pa, f_hz)
    peak_p_pa = float(np.max(pressure))
    mechanical_index = (kw.mechanical_index(peak_p_pa, design.therapy_frequency_hz)
                        if _HAS_PYKWAVERS
                        else peak_p_pa / 1.0e6 / np.sqrt(frequency_mhz))

    return {
        "pressure_pa": pressure,
        "intensity_w_m2": intensity,
        "temperature_c": T0_C + delta_t,
        "deposition": delivered_fraction,
        "mechanical_index": mechanical_index,
        "target_to_offtarget_ratio": target_to_offtarget_ratio(delivered_fraction, wall_target, phantom.plaque_mask),
        "peak_delta_t_c": float(np.max(delta_t)),
    }


def target_to_offtarget_ratio(field: np.ndarray, target: np.ndarray, plaque: np.ndarray) -> float:
    target_mean = float(np.mean(field[target]))
    off_target = (~plaque) & (field > 0.0)
    off_mean = float(np.mean(field[off_target]))
    return target_mean / max(off_mean, 1.0e-12)


def axis_extent_mm(phantom: VesselPhantom) -> list[float]:
    return [
        float(phantom.x_m.min() * 1.0e3),
        float(phantom.x_m.max() * 1.0e3),
        float(phantom.y_m.min() * 1.0e3),
        float(phantom.y_m.max() * 1.0e3),
    ]


def add_vessel_contours(ax: plt.Axes, phantom: VesselPhantom, colors: tuple[str, str] = ("cyan", "yellow")) -> None:
    extent = axis_extent_mm(phantom)
    x = np.linspace(extent[0], extent[1], phantom.labels.shape[0])
    y = np.linspace(extent[2], extent[3], phantom.labels.shape[1])
    ax.contour(x, y, phantom.lumen_mask.T.astype(float), levels=[0.5], colors=colors[0], linewidths=1.0)
    ax.contour(x, y, phantom.eel_mask.T.astype(float), levels=[0.5], colors=colors[1], linewidths=1.0)


def savefig(fig: plt.Figure, name: str) -> Path:
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dataset_and_phantom(spec: DatasetSpec, phantom: VesselPhantom) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    im = axes[0].imshow(phantom.labels.T, origin="lower", extent=axis_extent_mm(phantom), cmap="tab20")
    add_vessel_contours(axes[0], phantom)
    axes[0].set_title("Analytic IVUS frame geometry")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03, label="tissue label")

    axes[1].axis("off")
    text = (
        f"Dataset target: {spec.name}\n"
        f"Frame contract: {spec.frame_shape[0]} x {spec.frame_shape[1]} B-mode\n"
        f"Annotations: {', '.join(spec.annotated_boundaries)}\n"
        f"Frequency anchor: {spec.imaging_frequency_hz / 1.0e6:.0f} MHz\n"
        "Local executable input: deterministic vessel phantom\n"
        "External role: contour validation and domain shift checks"
    )
    axes[1].text(0.02, 0.92, text, va="top", ha="left", fontsize=10, family="monospace")
    axes[1].set_title("Dataset selection contract")
    return savefig(fig, "fig01_dataset_and_anatomy")


def plot_transducer_design(design: TransducerDesign, phantom: VesselPhantom) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    extent = axis_extent_mm(phantom)
    ax.imshow(np.where(phantom.plaque_mask, 1.0, np.nan).T, origin="lower", extent=extent, cmap="Reds", alpha=0.35)
    add_vessel_contours(ax, phantom, colors=("deepskyblue", "gold"))
    ax.add_patch(Circle((0.0, 0.0), design.catheter_radius_m * 1.0e3, color="#263238", zorder=3))
    ax.add_patch(Circle((0.18, -0.12), 0.11, color="white", zorder=4))
    angles = np.linspace(-np.pi, np.pi, design.imaging_elements, endpoint=False)
    elem_r = design.catheter_radius_m * 1.0e3 * 1.05
    ax.scatter(elem_r * np.cos(angles), elem_r * np.sin(angles), s=8, c="#80deea", zorder=5)
    start = np.rad2deg(design.therapy_azimuth_rad - design.therapy_sector_width_rad)
    stop = np.rad2deg(design.therapy_azimuth_rad + design.therapy_sector_width_rad)
    ax.add_patch(Wedge((0.0, 0.0), 4.2, start, stop, width=3.0, color="#ff7043", alpha=0.25))
    ax.plot([0.0, 4.0 * np.cos(design.therapy_azimuth_rad)], [0.0, 4.0 * np.sin(design.therapy_azimuth_rad)], color="#ff7043", lw=2)
    ax.set_title("Dual-frequency IVUS catheter design")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    return savefig(fig, "fig02_transducer_design")


def plot_bmode(phantom: VesselPhantom, bmode: dict[str, np.ndarray]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    axes[0].imshow(bmode["polar"], origin="lower", aspect="auto", cmap="gray")
    axes[0].set_title("Radial IVUS A-line frame")
    axes[0].set_xlabel("azimuth sample")
    axes[0].set_ylabel("range sample")
    axes[1].imshow(bmode["cartesian"].T, origin="lower", extent=axis_extent_mm(phantom), cmap="gray")
    add_vessel_contours(axes[1], phantom)
    axes[1].set_title("Scan-converted B-mode with contours")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    return savefig(fig, "fig03_ivus_bmode_simulation")


def plot_therapy(phantom: VesselPhantom, therapy: dict[str, np.ndarray | float]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), constrained_layout=True)
    fields = (
        ("pressure_pa", "Peak pressure [kPa]", "magma", 1.0e-3),
        ("deposition", "Delivered fraction", "viridis", 1.0),
        ("temperature_c", "Temperature [C]", "inferno", 1.0),
    )
    for ax, (key, title, cmap, scale) in zip(axes, fields):
        image = np.asarray(therapy[key], dtype=float) * scale
        im = ax.imshow(image.T, origin="lower", extent=axis_extent_mm(phantom), cmap=cmap)
        add_vessel_contours(ax, phantom)
        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return savefig(fig, "fig04_microbubble_therapy_map")


def plot_usage_workflow(phantom: VesselPhantom, bmode: dict[str, np.ndarray], therapy: dict[str, np.ndarray | float]) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.1), constrained_layout=True)
    axes[0].imshow(phantom.labels.T, origin="lower", extent=axis_extent_mm(phantom), cmap="tab20")
    axes[0].set_title("1 catheter crossing")
    axes[1].imshow(bmode["cartesian"].T, origin="lower", extent=axis_extent_mm(phantom), cmap="gray")
    axes[1].set_title("2 IVUS imaging")
    axes[2].imshow(np.asarray(therapy["deposition"]).T, origin="lower", extent=axis_extent_mm(phantom), cmap="viridis")
    axes[2].set_title("3 wall therapy")
    response = np.asarray(bmode["cartesian"]) + 0.45 * np.asarray(therapy["deposition"])
    axes[3].imshow(response.T, origin="lower", extent=axis_extent_mm(phantom), cmap="gray")
    axes[3].set_title("4 post-treatment check")
    for ax in axes:
        add_vessel_contours(ax, phantom)
        ax.set_xticks([])
        ax.set_yticks([])
    return savefig(fig, "fig05_intravascular_usage_sequence")


def write_metrics(
    spec: DatasetSpec,
    design: TransducerDesign,
    phantom: VesselPhantom,
    bmode: dict[str, np.ndarray],
    therapy: dict[str, np.ndarray | float],
    figures: list[Path],
) -> Path:
    payload = {
        "chapter": 30,
        "analysis": "intravascular ultrasound imaging plus localized microbubble therapy",
        "dataset": asdict(spec),
        "transducer": asdict(design),
        "computed_metrics": {
            "imaging_wavelength_um": C_TISSUE_M_S / design.imaging_frequency_hz * 1.0e6,
            "therapy_wavelength_mm": C_TISSUE_M_S / design.therapy_frequency_hz * 1.0e3,
            "lumen_area_mm2": float(np.count_nonzero(phantom.lumen_mask) * pixel_area_m2(phantom) * 1.0e6),
            "plaque_area_mm2": float(np.count_nonzero(phantom.plaque_mask) * pixel_area_m2(phantom) * 1.0e6),
            "bmode_dynamic_range_db": 60.0,
            "bmode_mean_lumen_intensity": float(np.mean(bmode["cartesian"][phantom.lumen_mask])),
            "bmode_mean_wall_intensity": float(np.mean(bmode["cartesian"][phantom.eel_mask & ~phantom.lumen_mask])),
            "therapy_mechanical_index": float(therapy["mechanical_index"]),
            "therapy_peak_delta_t_c": float(therapy["peak_delta_t_c"]),
            "therapy_target_to_offtarget_deposition_ratio": float(therapy["target_to_offtarget_ratio"]),
        },
        "figures": [str(path) for path in figures],
    }
    path = OUT_DIR / "metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def pixel_area_m2(phantom: VesselPhantom) -> float:
    dx = float(phantom.x_m[1, 0] - phantom.x_m[0, 0])
    dy = float(phantom.y_m[0, 1] - phantom.y_m[0, 0])
    return dx * dy


def run() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = default_dataset_spec()
    design = TransducerDesign()
    phantom = vessel_phantom(design=design)
    bmode = simulate_bmode(phantom, design)
    therapy = simulate_therapy(phantom, design)
    figures = [
        plot_dataset_and_phantom(spec, phantom),
        plot_transducer_design(design, phantom),
        plot_bmode(phantom, bmode),
        plot_therapy(phantom, therapy),
        plot_usage_workflow(phantom, bmode, therapy),
    ]
    metrics = write_metrics(spec, design, phantom, bmode, therapy, figures)
    return {"figures": [str(path) for path in figures], "metrics": str(metrics)}


if __name__ == "__main__" or __name__ == "ch30":
    run()
