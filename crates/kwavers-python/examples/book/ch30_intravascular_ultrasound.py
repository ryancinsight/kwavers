"""Chapter 30: intravascular ultrasound imaging and therapy.

The chapter simulation is deterministic. It generates an IVUS-like coronary
cross-section, a dual-frequency catheter layout, radial B-mode data, and a
localized microbubble delivery field under docs/book/figures/ch30/.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pykwavers as kw
from matplotlib.patches import Circle, Wedge


BOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BOOK_DIR.parents[3]
OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch30"

RHO_TISSUE_KG_M3 = 1060.0
C_TISSUE_M_S = 1540.0
CP_TISSUE_J_KG_K = 3600.0
T0_C = 37.0


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


def vessel_phantom(
    n: int = 384,
    fov_m: float = 12.0e-3,
    design: TransducerDesign | None = None,
    seed: int = 30,
) -> VesselPhantom:
    design = design or TransducerDesign()
    phantom = kw.ivus_vessel_phantom(
        int(n),
        float(fov_m),
        float(design.catheter_radius_m),
        float(design.therapy_azimuth_rad),
        int(seed),
    )

    return VesselPhantom(
        x_m=np.asarray(phantom["x_m"], dtype=float),
        y_m=np.asarray(phantom["y_m"], dtype=float),
        radius_m=np.asarray(phantom["radius_m"], dtype=float),
        theta_rad=np.asarray(phantom["theta_rad"], dtype=float),
        labels=np.asarray(phantom["labels"], dtype=np.uint8),
        sound_speed_m_s=np.asarray(phantom["sound_speed_m_s"], dtype=float),
        density_kg_m3=np.asarray(phantom["density_kg_m3"], dtype=float),
        attenuation_db_cm_mhz=np.asarray(phantom["attenuation_db_cm_mhz"], dtype=float),
        backscatter=np.asarray(phantom["backscatter"], dtype=float),
        lumen_mask=np.asarray(phantom["lumen_mask"], dtype=bool),
        eel_mask=np.asarray(phantom["eel_mask"], dtype=bool),
        plaque_mask=np.asarray(phantom["plaque_mask"], dtype=bool),
        fibrous_cap_mask=np.asarray(phantom["fibrous_cap_mask"], dtype=bool),
        lipid_mask=np.asarray(phantom["lipid_mask"], dtype=bool),
        calcium_mask=np.asarray(phantom["calcium_mask"], dtype=bool),
    )


def simulate_bmode(
    phantom: VesselPhantom,
    design: TransducerDesign,
    radial_samples: int = 384,
    angle_samples: int = 384,
) -> dict[str, np.ndarray]:
    r_axis = np.linspace(design.catheter_radius_m, design.maximum_imaging_radius_m, radial_samples)
    theta_axis = np.linspace(-np.pi, np.pi, angle_samples, endpoint=False)
    image = kw.ivus_bmode_image(
        np.ascontiguousarray(phantom.x_m, dtype=np.float64),
        np.ascontiguousarray(phantom.y_m, dtype=np.float64),
        np.ascontiguousarray(phantom.backscatter, dtype=np.float64),
        np.ascontiguousarray(phantom.attenuation_db_cm_mhz, dtype=np.float64),
        np.ascontiguousarray(r_axis, dtype=np.float64),
        np.ascontiguousarray(theta_axis, dtype=np.float64),
        np.ascontiguousarray(phantom.radius_m, dtype=np.float64),
        np.ascontiguousarray(phantom.theta_rad, dtype=np.float64),
        float(design.catheter_radius_m),
        float(design.imaging_frequency_hz),
        -60.0,
    )
    return {
        "r_axis_m": r_axis,
        "theta_axis_rad": theta_axis,
        "polar": np.asarray(image["polar"]).reshape((radial_samples, angle_samples)),
        "cartesian": np.asarray(image["cartesian"]).reshape(phantom.radius_m.shape),
        "db": np.asarray(image["db"]).reshape((radial_samples, angle_samples)),
    }


def simulate_therapy(phantom: VesselPhantom, design: TransducerDesign) -> dict[str, np.ndarray | float]:
    fields = kw.ivus_therapy_fields(
        np.ascontiguousarray(phantom.radius_m, dtype=np.float64),
        np.ascontiguousarray(phantom.theta_rad, dtype=np.float64),
        np.ascontiguousarray(phantom.attenuation_db_cm_mhz, dtype=np.float64),
        np.ascontiguousarray(phantom.eel_mask, dtype=bool),
        np.ascontiguousarray(phantom.lumen_mask, dtype=bool),
        np.ascontiguousarray(phantom.fibrous_cap_mask, dtype=bool),
        np.ascontiguousarray(phantom.lipid_mask, dtype=bool),
        np.ascontiguousarray(phantom.plaque_mask, dtype=bool),
        float(design.catheter_radius_m),
        float(design.therapy_pressure_pa),
        float(design.therapy_azimuth_rad),
        float(design.therapy_sector_width_rad),
        3.2e-3,
        float(design.therapy_frequency_hz),
        float(design.therapy_duty_cycle),
        float(design.therapy_sonication_s),
        RHO_TISSUE_KG_M3,
        C_TISSUE_M_S,
        CP_TISSUE_J_KG_K,
        1.75e-3,
        1.2e-3,
    )
    pressure = np.asarray(fields["pressure_pa"]).reshape(phantom.radius_m.shape)
    intensity = np.asarray(fields["intensity_w_m2"]).reshape(phantom.radius_m.shape)
    delta_t = np.asarray(fields["temperature_rise_k"]).reshape(phantom.radius_m.shape)
    delivered_fraction = np.asarray(fields["deposition"]).reshape(phantom.radius_m.shape)

    return {
        "pressure_pa": pressure,
        "intensity_w_m2": intensity,
        "temperature_c": T0_C + delta_t,
        "deposition": delivered_fraction,
        "mechanical_index": float(fields["mechanical_index"]),
        "target_to_offtarget_ratio": float(fields["target_to_offtarget_ratio"]),
        "peak_delta_t_c": float(fields["peak_delta_t_k"]),
    }


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
    computed_metrics = kw.ivus_chapter_metrics(
        np.ascontiguousarray(phantom.x_m, dtype=np.float64),
        np.ascontiguousarray(phantom.y_m, dtype=np.float64),
        np.ascontiguousarray(phantom.lumen_mask, dtype=bool),
        np.ascontiguousarray(phantom.eel_mask, dtype=bool),
        np.ascontiguousarray(phantom.plaque_mask, dtype=bool),
        np.ascontiguousarray(bmode["cartesian"], dtype=np.float64),
        C_TISSUE_M_S,
        float(design.imaging_frequency_hz),
        float(design.therapy_frequency_hz),
        60.0,
        float(therapy["mechanical_index"]),
        float(therapy["peak_delta_t_c"]),
        float(therapy["target_to_offtarget_ratio"]),
    )
    payload = {
        "chapter": 30,
        "analysis": "intravascular ultrasound imaging plus localized microbubble therapy",
        "dataset": asdict(spec),
        "transducer": asdict(design),
        "computed_metrics": dict(computed_metrics),
        "figures": [str(path) for path in figures],
    }
    path = OUT_DIR / "metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
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
