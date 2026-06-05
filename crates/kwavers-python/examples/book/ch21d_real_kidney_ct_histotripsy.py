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

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
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
# Q(x,y,z) = α·p²/(ρ·c)  [W/m³] — Pennes bioheat source from pressure field.
# Delegated to kw.acoustic_heat_source_density (Pennes 1948, Duck 1990 §5.2).
nx, ny, nz = p_peak.shape
Q_field = np.asarray(
    kw.acoustic_heat_source_density(
        p_peak.ravel().astype(np.float64),
        float(ALPHA_KIDNEY),
        float(RHO_KIDNEY),
        float(C_KIDNEY),
    )
).reshape(p_peak.shape).astype(p_peak.dtype)
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
# Figure 16: 3-D transducer bowl element placement on kidney — skin interface
# ─────────────────────────────────────────────────────────────────────────────
# Extract the body skin surface (outermost voxel shell) via 1-iteration 6-
# connected morphological erosion.  Showing all interior body voxels as the
# previous implementation did produces a solid opaque blob that occludes the
# transducer elements and misrepresents the transducer-to-skin interface.
# The skin surface is the set of body voxels with at least one 6-connected
# face-adjacent neighbour that is NOT body tissue.
#
# Invariant verified: no therapy element may lie inside the body mask.
# The HistoSonics-like bowl sits on the anterior or lateral skin surface
# (hip/abdominal region), external to the patient.
print("[fig16] 3-D transducer placement on skin surface")
from scipy.ndimage import binary_erosion, generate_binary_structure as _gbs

_struct6 = _gbs(3, 1)          # 6-connected face-adjacent neighbourhood
_body_interior = binary_erosion(body_mask, structure=_struct6, iterations=1)
_body_surface_mask = body_mask & ~_body_interior
_surface_idx = np.argwhere(_body_surface_mask)
if len(_surface_idx) > 10_000:
    _surface_idx = _surface_idx[:: len(_surface_idx) // 10_000]
_surface_m = _surface_idx * sp          # physical coords (x, y, z) in metres

# Focus: centroid of RCC target voxels.
_focus_m: np.ndarray | None = None
if target_mask.any():
    _tgt_idx = np.argwhere(target_mask)
    _focus_m = _tgt_idx.mean(axis=0) * sp          # shape (3,) in metres

# Skin contact: body surface voxel nearest to the centroid of therapy elements.
_skin_m: np.ndarray | None = None
if therapy_pts.size > 0 and len(_surface_m) > 0:
    _elem_centroid = therapy_pts.mean(axis=0)
    _skin_m = _surface_m[
        np.linalg.norm(_surface_m - _elem_centroid, axis=1).argmin()
    ]

# Invariant check: no element inside body.
if therapy_pts.size > 0:
    for _e in therapy_pts:
        _ix = int(round(_e[0] / sp))
        _iy = int(round(_e[1] / sp))
        _iz = int(round(_e[2] / sp))
        if (0 <= _ix < body_mask.shape[0]
                and 0 <= _iy < body_mask.shape[1]
                and 0 <= _iz < body_mask.shape[2]):
            assert not body_mask[_ix, _iy, _iz], (
                f"Therapy element at voxel ({_ix},{_iy},{_iz}) is inside the body mask. "
                "Transducer placement invariant violated."
            )

fig = plt.figure(figsize=(9, 7))
ax3 = fig.add_subplot(111, projection="3d")

# Body skin surface — very transparent so elements in front remain visible.
# Axis ordering (z→X, x→Y, y→Z) matches the original file convention and
# places superior (z) on the horizontal axis, matching standard axial view.
ax3.scatter(
    _surface_m[:, 2] * 1e3, _surface_m[:, 0] * 1e3, _surface_m[:, 1] * 1e3,
    s=0.5, c="#9eb8d1", alpha=0.12, rasterized=True, label="Skin surface",
)

# RCC tumour (target)
if target_mask.any():
    _tgt_m = np.argwhere(target_mask) * sp
    ax3.scatter(
        _tgt_m[:, 2] * 1e3, _tgt_m[:, 0] * 1e3, _tgt_m[:, 1] * 1e3,
        s=3, c="#d62728", alpha=0.7, label="RCC target",
    )

# Therapy elements — outside body, at skin coupling interface.
if therapy_pts.size > 0:
    ax3.scatter(
        therapy_pts[:, 2] * 1e3, therapy_pts[:, 0] * 1e3, therapy_pts[:, 1] * 1e3,
        s=14, c="gold", alpha=0.9, label=f"Bowl elements ({len(therapy_pts)})",
        zorder=5,
    )

# Focus marker (target centroid).
if _focus_m is not None:
    ax3.scatter(
        _focus_m[2] * 1e3, _focus_m[0] * 1e3, _focus_m[1] * 1e3,
        s=90, c="cyan", marker="*", zorder=6, label="Focus (target centroid)",
    )

# Skin contact marker.
if _skin_m is not None:
    ax3.scatter(
        _skin_m[2] * 1e3, _skin_m[0] * 1e3, _skin_m[1] * 1e3,
        s=90, c="orange", marker="^", zorder=6, label="Skin contact",
    )
    # Beam lines from every 32nd element to focus (sparse, for clarity).
    if _focus_m is not None and therapy_pts.size > 0:
        _stride = max(1, len(therapy_pts) // 32)
        for _e in therapy_pts[::_stride]:
            ax3.plot(
                [_e[2] * 1e3, _focus_m[2] * 1e3],
                [_e[0] * 1e3, _focus_m[0] * 1e3],
                [_e[1] * 1e3, _focus_m[1] * 1e3],
                c="#555", lw=0.25, alpha=0.20,
            )

ax3.set_xlabel("z (mm)")
ax3.set_ylabel("x (mm)")
ax3.set_zlabel("y (mm)")
ax3.set_title(
    "HistoSonics-like Bowl on Kidney — Elements at Skin Interface\n"
    "Gold = therapy elements, red = RCC, cyan = focus, orange = skin contact",
)
ax3.legend(fontsize=7, markerscale=2, loc="upper right")
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
