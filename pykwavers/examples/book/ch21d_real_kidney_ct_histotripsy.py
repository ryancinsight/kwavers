"""
Chapter 21d: Histotripsy planning on a real abdominal CT — kidney tumour
========================================================================

Loads a patient abdominal CT (KiTS19 dataset format) and runs intrinsic-
threshold histotripsy planning for a renal-cell carcinoma (RCC) target.

CT / segmentation:
    Set KWAVERS_CH21D_CT_NIFTI (e.g. kits19_sample/case_00000.nii.gz) and
    KWAVERS_CH21D_SEG_NIFTI (segmentation_00000.nii.gz) to use real patient
    data.  If absent, the Rust binding falls back to a synthetic abdominal-
    kidney phantom automatically.

Physics:
    All acoustic propagation, nonlinear Westervelt FWI, cavitation source
    density, and electronic steering are delegated to the Rust solver via
    pykwavers.run_theranostic_nonlinear_3d_from_ritk (anatomy="kidney").
    Python is used only for thermal dose integration (ThermalSimulation)
    and plotting.

License (if using real data):
    KiTS19 CC-BY-NC-SA 4.0 (Heller 2019)
    https://kits19.sfo2.digitaloceanspaces.com/master_00000.nii.gz

Outputs (PNG and PDF) under docs/book/figures/ch21d/:
    fig13_ct_segmentation        — CT slices with tissue-label overlay
    fig14_pressure_cavitation    — peak pressure + cavitation density
    fig15_thermal_dose           — CEM43 map after treatment (ThermalDiffusionSolver)
    fig16_transducer_placement   — bowl elements on kidney surface (3-D)
    fig17_fwi_convergence        — FWI objective history
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — required for projection="3d"

import pykwavers as kw

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch21d")
os.makedirs(OUT_DIR, exist_ok=True)

_DEFAULT_CT = os.path.join(REPO_ROOT, "data", "kits19_sample", "case_00000.nii.gz")
_DEFAULT_SEG = os.path.join(REPO_ROOT, "data", "kits19_sample", "segmentation_00000.nii.gz")
CT_PATH = os.environ.get("KWAVERS_CH21D_CT_NIFTI", _DEFAULT_CT)
SEG_PATH = os.environ.get("KWAVERS_CH21D_SEG_NIFTI", _DEFAULT_SEG)
GRID_SIZE = int(os.environ.get("KWAVERS_CH21D_GRID_SIZE", "48"))
ITERATIONS = int(os.environ.get("KWAVERS_CH21D_ITERATIONS", "2"))

# Histotripsy scenario: intrinsic threshold (HistoSonics-style)
FREQUENCY_HZ = float(os.environ.get("KWAVERS_CH21D_FREQ_HZ", "1e6"))
SOURCE_PRESSURE_PA = float(os.environ.get("KWAVERS_CH21D_PRESSURE_PA", "30e6"))
CYCLES = float(os.environ.get("KWAVERS_CH21D_CYCLES", "2.0"))
T_TREAT = 3.0   # s, sonication time per raster point for thermal dose

# Duck 1990 / IT'IS v4.1 — kidney
RHO_KIDNEY = 1050.0
C_KIDNEY = 1560.0
ALPHA_KIDNEY = 8.5   # Np/m at 1 MHz
K_KIDNEY = 0.52
CP_KIDNEY = 3760.0
WB_KIDNEY = 8e-3


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch21d/{name}.{{pdf,png}}")


plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 8,
})

# ─────────────────────────────────────────────────────────────────────────────
# Run nonlinear 3-D histotripsy simulation — Rust solver
# ─────────────────────────────────────────────────────────────────────────────
print("Running kidney histotripsy simulation via pykwavers.run_theranostic_nonlinear_3d_from_ritk …")
ct_exists = CT_PATH != "__synthetic__" and os.path.exists(CT_PATH)
seg_exists = SEG_PATH != "__synthetic__" and os.path.exists(SEG_PATH)

kwargs: dict = dict(
    ct_nifti_path=CT_PATH if ct_exists else "__synthetic__",
    anatomy="kidney",
    grid_size=GRID_SIZE,
    iterations=ITERATIONS,
    frequency_hz=FREQUENCY_HZ,
    source_pressure_pa=SOURCE_PRESSURE_PA,
    cycles=CYCLES,
)
if seg_exists:
    kwargs["segmentation_nifti_path"] = SEG_PATH

r = kw.run_theranostic_nonlinear_3d_from_ritk(**kwargs)
print(f"  anatomy={r['anatomy']}, grid={r['grid_size']}, "
      f"elements={r.get('element_count', '?')}, spacing={float(r['spacing_m'])*1e3:.2f} mm")

ct = np.asarray(r["ct_hu"])
label = np.asarray(r["label"])
body_mask = np.asarray(r["body_mask"]).astype(bool)
target_mask = np.asarray(r["target_mask"]).astype(bool)
p_peak = np.asarray(r["westervelt_peak_pressure_pa"])
cav = np.asarray(r["cavitation_source_density"])
c_map = np.asarray(r["true_sound_speed_m_s"])
sp = float(r["spacing_m"])
therapy_pts = np.asarray(r["therapy_points_m"])   # (N, 3) [m]
fwi_obj = np.asarray(r["fwi_objective_history"])


def _mm(n: int) -> np.ndarray:
    return np.arange(n) * sp * 1e3


ix, iy, iz = (s // 2 for s in ct.shape)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 13: CT slices with tissue-label overlay
# ─────────────────────────────────────────────────────────────────────────────
print("[fig13] CT segmentation")
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
slices = [
    (ct[:, iy, :], label[:, iy, :], "Sagittal (xz)", _mm(ct.shape[0]), _mm(ct.shape[2])),
    (ct[ix, :, :], label[ix, :, :], "Coronal (yz)",  _mm(ct.shape[1]), _mm(ct.shape[2])),
    (ct[:, :, iz], label[:, :, iz], "Axial (xy)",    _mm(ct.shape[0]), _mm(ct.shape[1])),
]
LABEL_COLORS = {1: (0.8, 0.2, 0.2), 2: (0.2, 0.8, 0.2)}  # 1=kidney, 2=tumour
for ax, (ct_sl, lbl_sl, ttl, x_mm, y_mm) in zip(axes, slices):
    ext = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]
    ax.imshow(ct_sl.T, cmap="gray", origin="lower", vmin=-150, vmax=250, extent=ext)
    for lv, col in LABEL_COLORS.items():
        mask_sl = lbl_sl == lv
        if mask_sl.any():
            overlay = np.zeros((*mask_sl.shape, 4), dtype=float)
            overlay[mask_sl, :3] = col
            overlay[mask_sl, 3] = 0.4
            ax.imshow(overlay.transpose(1, 0, 2), origin="lower", extent=ext, aspect="auto")
    ax.set_title(ttl); ax.set_xlabel("mm"); ax.set_ylabel("mm")
synthetic_str = "Synthetic phantom" if r.get("synthetic_phantom", not ct_exists) else "KiTS19 CT"
fig.suptitle(f"Kidney CT — {synthetic_str}\nRed = kidney, green = RCC tumour target")
plt.tight_layout()
savefig("fig13_ct_segmentation")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 14: Westervelt peak pressure + cavitation source density
# ─────────────────────────────────────────────────────────────────────────────
print("[fig14] Pressure + cavitation maps")
fig, (ax_p, ax_c) = plt.subplots(1, 2, figsize=(12, 5))

ext_xz = [_mm(ct.shape[2])[0], _mm(ct.shape[2])[-1],
           _mm(ct.shape[0])[0], _mm(ct.shape[0])[-1]]

im_p = ax_p.imshow(p_peak[:, iy, :] / 1e6, cmap="hot", origin="lower", extent=ext_xz, aspect="auto")
plt.colorbar(im_p, ax=ax_p, label="Peak pressure (MPa)")
ax_p.set_title(f"Westervelt Peak Pressure (MPa)\n{FREQUENCY_HZ/1e6:.1f} MHz, {SOURCE_PRESSURE_PA/1e6:.0f} MPa")
ax_p.set_xlabel("z (mm)"); ax_p.set_ylabel("x (mm)")

im_c = ax_c.imshow(cav[:, iy, :], cmap="inferno", origin="lower", extent=ext_xz, aspect="auto")
plt.colorbar(im_c, ax=ax_c, label="Cavitation density (a.u.)")
if target_mask.any():
    ax_c.contour(target_mask[:, iy, :].astype(float), levels=[0.5], colors=["cyan"],
                 extent=ext_xz, origin="lower", linewidths=1.2, linestyles=["--"])
ax_c.set_title("Cavitation Source Density\n(Westervelt-bubble coupled solver; cyan = RCC)")
ax_c.set_xlabel("z (mm)"); ax_c.set_ylabel("x (mm)")

plt.tight_layout()
savefig("fig14_pressure_cavitation")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 15: CEM43 thermal dose — ThermalDiffusionSolver
# ─────────────────────────────────────────────────────────────────────────────
print("[fig15] CEM43 thermal dose (ThermalDiffusionSolver)")
DT_THERM = 0.1
N_STEPS_TH = int(T_TREAT / DT_THERM)
Q_field = ALPHA_KIDNEY * p_peak ** 2 / (RHO_KIDNEY * C_KIDNEY)  # W/m³
nx, ny, nz = p_peak.shape
sim_th = kw.ThermalSimulation(
    nx, ny, nz, sp, sp, sp,
    thermal_conductivity=K_KIDNEY, density=RHO_KIDNEY,
    specific_heat=CP_KIDNEY, enable_bioheat=True,
    perfusion_rate=WB_KIDNEY, initial_temperature=37.0,
    track_thermal_dose=True,
)
res_th = sim_th.run(N_STEPS_TH, DT_THERM, heat_source=Q_field)
dose = np.asarray(res_th.thermal_dose)

fig, ax_d = plt.subplots(figsize=(6, 5))
im_d = ax_d.imshow(np.log10(np.maximum(dose[:, iy, :], 1e-2)),
                   cmap="inferno", origin="lower", extent=ext_xz, aspect="auto")
ax_d.contour(dose[:, iy, :], levels=[60.0, 240.0], colors=["yellow", "white"],
             extent=ext_xz, origin="lower", linewidths=[0.8, 1.2])
plt.colorbar(im_d, ax=ax_d, label="log₁₀ CEM43 [min]")
ax_d.set_title(f"CEM43 Thermal Dose — {T_TREAT:.0f} s Sonication\n"
               "Yellow=60 min, White=240 min ablation contours")
ax_d.set_xlabel("z (mm)"); ax_d.set_ylabel("x (mm)")
plt.tight_layout()
savefig("fig15_thermal_dose")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 16: 3-D transducer bowl element placement on kidney surface
# ─────────────────────────────────────────────────────────────────────────────
print("[fig16] 3-D transducer placement")
fig = plt.figure(figsize=(8, 7))
ax3 = fig.add_subplot(111, projection="3d")
# Body surface (downsampled)
body_idx = np.argwhere(body_mask)
if len(body_idx) > 5000:
    body_idx = body_idx[:: len(body_idx) // 5000]
ax3.scatter(body_idx[:, 2] * sp * 1e3, body_idx[:, 0] * sp * 1e3, body_idx[:, 1] * sp * 1e3,
            s=0.3, c="lightblue", alpha=0.1)
# Target (RCC tumour)
if target_mask.any():
    tgt_idx = np.argwhere(target_mask)
    ax3.scatter(tgt_idx[:, 2] * sp * 1e3, tgt_idx[:, 0] * sp * 1e3, tgt_idx[:, 1] * sp * 1e3,
                s=2, c="red", alpha=0.6, label="RCC target")
# Therapy elements
if therapy_pts.size > 0:
    ax3.scatter(therapy_pts[:, 2] * 1e3, therapy_pts[:, 0] * 1e3, therapy_pts[:, 1] * 1e3,
                s=10, c="gold", alpha=0.9, label=f"Bowl elements ({len(therapy_pts)})")
ax3.set_xlabel("z (mm)"); ax3.set_ylabel("x (mm)"); ax3.set_zlabel("y (mm)")
ax3.set_title("HistoSonics-like Bowl on Kidney\n"
              "Gold = therapy elements, red = RCC, blue = body")
ax3.legend(fontsize=8)
plt.tight_layout()
savefig("fig16_transducer_placement")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 17: FWI objective convergence history
# ─────────────────────────────────────────────────────────────────────────────
print("[fig17] FWI convergence history")
fig, ax_fwi = plt.subplots(figsize=(7, 4))
ax_fwi.semilogy(np.arange(1, len(fwi_obj) + 1), fwi_obj, "C0-o", ms=4)
ax_fwi.set_xlabel("FWI iteration")
ax_fwi.set_ylabel("Objective")
ax_fwi.set_title("Multiparameter FWI Convergence — Kidney RCC\n"
                 "(Sound speed + nonlinearity β joint inversion)")
ax_fwi.grid(True, ls=":", alpha=0.4)
plt.tight_layout()
savefig("fig17_fwi_convergence")
plt.close()

print(f"\nChapter 21d figures written to: {os.path.relpath(OUT_DIR)}")
print(f"  CT source: {'real NIfTI' if ct_exists else 'synthetic phantom'}")
