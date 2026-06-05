"""
Chapter 21b: Histotripsy treatment of hepatocellular carcinoma (HCC)
====================================================================

Simulates three clinical histotripsy scenarios on a 3-D liver phantom:

    1. Intrinsic-threshold histotripsy  (1 MHz, 30 MPa PNP, 2-cycle pulse)
    2. Boiling histotripsy              (1 MHz, 15 MPa PNP, 50-cycle shock)
    3. Sub-threshold millisecond        (500 kHz, 18 MPa PNP, 5-cycle pulse)

Physics:
    All acoustic propagation, nonlinear Westervelt FWI, cavitation source
    density, and electronic steering are delegated to the Rust solver via
    pykwavers.run_theranostic_nonlinear_3d_from_ritk.  Python is used only
    for thermal dose integration (pykwavers.ThermalSimulation) and plotting.

CT / segmentation:
    Set KWAVERS_CH21B_CT_NIFTI and KWAVERS_CH21B_SEG_NIFTI to load a real
    liver CT.  If the file is absent the Rust binding falls back to a
    synthetic abdominal-liver phantom automatically.

Outputs (PNG and PDF) under docs/book/figures/ch21b/:
    fig01_phantom_slices        — anatomy slices through transducer focus
    fig02_pressure_fields       — Westervelt peak pressure (3 scenarios)
    fig03_cavitation_maps       — cavitation source density (3 scenarios)
    fig04_thermal_dose          — CEM43 maps after full treatment
    fig05_lesion_envelope       — cavitation lesion vs target mask
    fig06_scenario_metrics      — bar chart of treatment metrics
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pykwavers as kw

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch21b")
os.makedirs(OUT_DIR, exist_ok=True)

CT_PATH = os.environ.get("KWAVERS_CH21B_CT_NIFTI", "__synthetic__")
SEG_PATH = os.environ.get("KWAVERS_CH21B_SEG_NIFTI", "__synthetic__")
GRID_SIZE = int(os.environ.get("KWAVERS_CH21B_GRID_SIZE", "48"))
ITERATIONS = int(os.environ.get("KWAVERS_CH21B_ITERATIONS", "2"))


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch21b/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 8, "lines.linewidth": 1.3,
})

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants (Duck 1990 / IT'IS v4.1 liver)
# ─────────────────────────────────────────────────────────────────────────────
RHO_LIVER = 1060.0  # kg/m³
C_LIVER = 1590.0    # m/s
ALPHA_LIVER = 9.0   # Np/m at 1 MHz
K_LIVER = 0.51      # W/(m·K)
CP_LIVER = 3600.0   # J/(kg·K)
WB_LIVER = 5e-3     # blood perfusion [1/s]
T_TREAT = 3.0       # s sonication time per raster point

# ─────────────────────────────────────────────────────────────────────────────
# Three histotripsy scenarios
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS: list[dict] = [
    {
        "label": "Intrinsic\n(1 MHz, 30 MPa, 2 cyc)",
        "anatomy": "liver",
        "frequency_hz": 1.0e6,
        "source_pressure_pa": 30.0e6,
        "cycles": 2.0,
    },
    {
        "label": "Boiling\n(1 MHz, 15 MPa, 50 cyc)",
        "anatomy": "liver",
        "frequency_hz": 1.0e6,
        "source_pressure_pa": 15.0e6,
        "cycles": 50.0,
    },
    {
        "label": "Sub-threshold\n(500 kHz, 18 MPa, 5 cyc)",
        "anatomy": "liver",
        "frequency_hz": 5.0e5,
        "source_pressure_pa": 18.0e6,
        "cycles": 5.0,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Run each scenario — Westervelt FWI + cavitation (Rust solver)
# ─────────────────────────────────────────────────────────────────────────────
print("Running 3 histotripsy scenarios via pykwavers.run_theranostic_nonlinear_3d_from_ritk …")
results: list[dict] = []
for i, sc in enumerate(SCENARIOS):
    print(f"  scenario {i+1}/3: {sc['label'].replace(chr(10), ' ')}")
    kwargs: dict = dict(
        ct_nifti_path=CT_PATH if (CT_PATH != "__synthetic__" and os.path.exists(CT_PATH)) else "__synthetic__",
        anatomy=sc["anatomy"],
        grid_size=GRID_SIZE,
        iterations=ITERATIONS,
        frequency_hz=sc["frequency_hz"],
        source_pressure_pa=sc["source_pressure_pa"],
        cycles=sc["cycles"],
    )
    if SEG_PATH != "__synthetic__" and os.path.exists(SEG_PATH):
        kwargs["segmentation_nifti_path"] = SEG_PATH
    results.append(kw.run_theranostic_nonlinear_3d_from_ritk(**kwargs))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: mid-slice index, mm coordinate
# ─────────────────────────────────────────────────────────────────────────────

def _midslice(vol: np.ndarray) -> tuple[int, int, int]:
    return tuple(s // 2 for s in vol.shape)


def _mm(arr: np.ndarray, spacing_m: float) -> np.ndarray:
    return arr * spacing_m * 1e3


# ─────────────────────────────────────────────────────────────────────────────
# Figure 01: CT phantom anatomy slices through transducer focus
# ─────────────────────────────────────────────────────────────────────────────
print("[fig01] CT phantom slices")
r0 = results[0]
ct = np.asarray(r0["ct_hu"])
label = np.asarray(r0["label"])
sp = float(r0["spacing_m"])
ix, iy, iz = _midslice(ct)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (sl, ttl) in zip(axes, [
    (ct[:, iy, :], "Sagittal (xz)"),
    (ct[ix, :, :], "Coronal (yz)"),
    (ct[:, :, iz], "Axial (xy)"),
]):
    ax.imshow(sl.T, cmap="gray", origin="lower", vmin=-150, vmax=200,
              extent=[0, _mm(np.arange(sl.shape[0]), sp)[-1],
                      0, _mm(np.arange(sl.shape[1]), sp)[-1]])
    ax.set_title(ttl); ax.set_xlabel("mm"); ax.set_ylabel("mm")
fig.suptitle("Liver HCC Phantom — CT Slices (HU)\n"
             f"{'Synthetic' if r0.get('synthetic_phantom', True) else 'Real CT'} anatomy, "
             f"spacing {sp*1e3:.2f} mm")
plt.tight_layout()
savefig("fig01_phantom_slices")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 02: Westervelt peak pressure fields — 3 scenarios (axial mid-slice)
# ─────────────────────────────────────────────────────────────────────────────
print("[fig02] Westervelt peak pressure fields")
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, r, sc in zip(axes, results, SCENARIOS):
    p = np.asarray(r["westervelt_peak_pressure_pa"])
    sp_r = float(r["spacing_m"])
    ix_r = p.shape[0] // 2
    sl = p[ix_r, :, :]
    ext = [0, _mm(np.arange(sl.shape[1]), sp_r)[-1],
           0, _mm(np.arange(sl.shape[0]), sp_r)[-1]]
    im = ax.imshow(sl, cmap="hot", origin="lower", extent=ext, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, label="Pa")
    ax.set_title(sc["label"])
    ax.set_xlabel("mm")
axes[0].set_ylabel("mm")
fig.suptitle("Westervelt Peak Pressure — 3 Histotripsy Scenarios\n"
             "(Nonlinear PSTD FWI via pykwavers, liver anatomy)")
plt.tight_layout()
savefig("fig02_pressure_fields")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 03: Cavitation source density — 3 scenarios
# ─────────────────────────────────────────────────────────────────────────────
print("[fig03] Cavitation source density maps")
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, r, sc in zip(axes, results, SCENARIOS):
    cav = np.asarray(r["cavitation_source_density"])
    sp_r = float(r["spacing_m"])
    ix_r = cav.shape[0] // 2
    sl = cav[ix_r, :, :]
    ext = [0, _mm(np.arange(sl.shape[1]), sp_r)[-1],
           0, _mm(np.arange(sl.shape[0]), sp_r)[-1]]
    im = ax.imshow(sl, cmap="inferno", origin="lower", extent=ext, aspect="auto",
                   vmin=0, vmax=cav.max() + 1e-30)
    plt.colorbar(im, ax=ax, fraction=0.04, label="a.u.")
    ax.set_title(sc["label"])
    ax.set_xlabel("mm")
axes[0].set_ylabel("mm")
fig.suptitle("Cavitation Source Density — Westervelt-Bubble Coupled Solver\n"
             "(pykwavers.run_theranostic_nonlinear_3d_from_ritk)")
plt.tight_layout()
savefig("fig03_cavitation_maps")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 04: CEM43 thermal dose after 3-s sonication (ThermalDiffusionSolver)
# Q = 2·α·p²/(2·ρ·c) from the Westervelt peak-pressure field.
# ─────────────────────────────────────────────────────────────────────────────
print("[fig04] CEM43 thermal dose (ThermalDiffusionSolver)")
DT_THERM = 0.1   # s — stable for typical tissue mesh spacing
N_STEPS_TH = int(T_TREAT / DT_THERM)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, r, sc in zip(axes, results, SCENARIOS):
    p_peak = np.asarray(r["westervelt_peak_pressure_pa"])
    sp_r = float(r["spacing_m"])
    nx, ny, nz = p_peak.shape
    # Heat source Q = α·p²/(ρ·c) [W/m³] via Pennes bioheat source term.
    # Derivation: I = p²/(2ρc), Q = 2α·I → Q = α·p²/(ρ·c).
    Q = np.asarray(
        kw.acoustic_heat_source_density(
            p_peak.ravel().astype(np.float64),
            float(ALPHA_LIVER),
            float(RHO_LIVER),
            float(C_LIVER),
        )
    ).reshape(p_peak.shape).astype(p_peak.dtype)
    sim_th = kw.ThermalSimulation(
        nx, ny, nz, sp_r, sp_r, sp_r,
        thermal_conductivity=K_LIVER, density=RHO_LIVER,
        specific_heat=CP_LIVER, enable_bioheat=True,
        perfusion_rate=WB_LIVER, initial_temperature=37.0,
        track_thermal_dose=True,
    )
    res_th = sim_th.run(N_STEPS_TH, DT_THERM, heat_source=Q)
    dose = np.asarray(res_th.thermal_dose)  # (nx, ny, nz) [min]
    ix_r = dose.shape[0] // 2
    sl = dose[ix_r, :, :]
    ext = [0, _mm(np.arange(sl.shape[1]), sp_r)[-1],
           0, _mm(np.arange(sl.shape[0]), sp_r)[-1]]
    im = ax.imshow(np.log10(np.maximum(sl, 1e-2)), cmap="inferno",
                   origin="lower", extent=ext, aspect="auto")
    ax.contour(sl, levels=[60.0, 240.0], colors=["yellow", "white"],
               extent=ext, origin="lower", linewidths=[0.8, 1.2])
    plt.colorbar(im, ax=ax, fraction=0.04, label="log₁₀ CEM43 [min]")
    ax.set_title(sc["label"])
    ax.set_xlabel("mm")
axes[0].set_ylabel("mm")
fig.suptitle(f"CEM43 Thermal Dose after {T_TREAT:.0f} s Sonication\n"
             "(Q from Westervelt pressure; kwavers ThermalDiffusionSolver)")
plt.tight_layout()
savefig("fig04_thermal_dose")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 05: Lesion envelope — cavitation density vs target mask
# ─────────────────────────────────────────────────────────────────────────────
print("[fig05] Lesion envelope vs target mask")
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, r, sc in zip(axes, results, SCENARIOS):
    cav = np.asarray(r["cavitation_source_density"])
    tgt = np.asarray(r["target_mask"]).astype(float)
    sp_r = float(r["spacing_m"])
    ix_r = cav.shape[0] // 2
    ext = [0, _mm(np.arange(cav.shape[2]), sp_r)[-1],
           0, _mm(np.arange(cav.shape[1]), sp_r)[-1]]
    ax.imshow(cav[ix_r].T, cmap="hot", origin="lower", extent=ext, aspect="auto")
    ax.contour(tgt[ix_r].T, levels=[0.5], colors=["cyan"], extent=ext,
               origin="lower", linewidths=1.5, linestyles=["--"])
    ax.set_title(sc["label"])
    ax.set_xlabel("mm")
axes[0].set_ylabel("mm")
fig.suptitle("Cavitation Lesion (hot) vs HCC Target Mask (cyan dashed)\n"
             "(pykwavers nonlinear 3-D theranostic solver)")
plt.tight_layout()
savefig("fig05_lesion_envelope")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 06: Scenario metrics bar chart
# ─────────────────────────────────────────────────────────────────────────────
print("[fig06] Scenario metrics")
metrics_labels = ["Peak pressure\n(MPa)", "Cavitation vol\n(mm³)", "Active voxels"]
metrics_data: list[list[float]] = []
for r, sc in zip(results, SCENARIOS):
    p_peak = np.asarray(r["westervelt_peak_pressure_pa"])
    cav = np.asarray(r["cavitation_source_density"])
    sp_r = float(r["spacing_m"])
    peak_mpa = float(p_peak.max()) / 1e6
    cav_vol = float((cav > 0.1 * cav.max()).sum()) * (sp_r * 1e3) ** 3
    active = int(r.get("active_voxels", int((r["body_mask"] > 0).sum())))
    metrics_data.append([peak_mpa, cav_vol, float(active)])

x = np.arange(len(SCENARIOS))
scenario_labels = [sc["label"].replace("\n", " ") for sc in SCENARIOS]
n_metrics = len(metrics_labels)
width = 0.25

fig, axes_m = plt.subplots(1, n_metrics, figsize=(12, 4))
for mi, (ax_m, ml) in enumerate(zip(axes_m, metrics_labels)):
    vals = [metrics_data[si][mi] for si in range(len(SCENARIOS))]
    bars = ax_m.bar(x, vals, width=0.6, color=["C0", "C1", "C2"])
    ax_m.set_title(ml)
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(["S1", "S2", "S3"])
    for bar, v in zip(bars, vals):
        ax_m.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                  f"{v:.2g}", ha="center", fontsize=8)

fig.suptitle("Scenario Metrics — 3 Histotripsy Modalities (Liver HCC)")
plt.tight_layout()
savefig("fig06_scenario_metrics")
plt.close()

print(f"\nChapter 21b figures written to: {os.path.relpath(OUT_DIR)}")
