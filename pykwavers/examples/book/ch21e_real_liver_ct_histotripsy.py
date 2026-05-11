"""
Chapter 21e: Histotripsy planning on a real abdominal CT — liver tumour
=======================================================================

Loads LiTS17 case-0 (Bilic 2023, CC-BY 4.0) — an arterial-phase
contrast-enhanced abdominal CT acquired for the Liver Tumor
Segmentation Challenge — and runs the three histotripsy clinical
scenarios on the patient's actual hepatocellular carcinoma (HCC)
tumour, using the dataset's voxel-level segmentation labels
(liver = 1, tumour = 2) rather than a synthetic placeholder.

CT volume    : https://archive.org/download/academictorrents_27772adef6f563a1ecc0ae19a528b956e6c803ce/volume-0.nii.zip
Segmentation : https://archive.org/download/academictorrents_27772adef6f563a1ecc0ae19a528b956e6c803ce/segmentation-0.nii.zip
License      : CC-BY 4.0 (Bilic et al. 2023, Med Image Anal 84:102680)

Pipeline mirrors ch21d (kidney tumour) — see that file for the
algorithmic details. The only differences are: (1) the dataset path,
(2) the parenchyma tissue type (LIVER) and tumour type (HCC), and
(3) the cropping margins (LiTS slice spacing is 5 mm vs KiTS19's
0.5 mm so axial margin is in slices, not voxels).

Outputs:
    docs/book/figures/ch21e/*.{png,pdf}
    docs/book/figures/ch21e/embedded_figures.md
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
                           distance_transform_edt, label, zoom)
from scipy.special import erf

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CT_PATH = os.path.join(REPO_ROOT, "data", "lits17_sample", "volume-0.nii")
SEG_PATH = os.path.join(REPO_ROOT, "data", "lits17_sample", "segmentation-0.nii")
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch21e")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
})


# ───────────────────────────────────────────────────────────────────────
# Tissue properties (same as ch21b; Duck 1990 / IT'IS v4.1 / Mast 2000)
# ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Tissue:
    label: int
    name: str
    rho: float
    c: float
    alpha0: float
    y_pow: float
    cp: float
    kappa: float
    perfusion: float


AIR    = Tissue(0, "air",       1.2, 343.0,    0.0,  1.0,    1005.0, 0.026, 0.0)
SKIN   = Tissue(1, "skin",   1109.0, 1624.0,  21.158, 1.10, 3391.0, 0.37, 1.06)
FAT    = Tissue(2, "fat",     911.0, 1440.0,   4.836, 1.10, 2348.0, 0.21, 0.43)
MUSCLE = Tissue(3, "muscle", 1090.0, 1588.0,   8.054, 1.10, 3421.0, 0.49, 0.67)
BONE   = Tissue(4, "bone",   1908.0, 4080.0, 250.0,   1.0,  1313.0, 0.32, 0.10)
LIVER  = Tissue(5, "liver",  1079.0, 1595.0,   8.690, 1.10, 3540.0, 0.52, 6.4)
HCC    = Tissue(6, "hcc",    1066.0, 1570.0,  12.500, 1.10, 3750.0, 0.55, 9.0)
# Same acoustic/thermal properties as HCC but separate label so
# multifocal disease can be visualised — only the target focus
# (label 6) is driven by the raster planner; HCC_OTHER (label 7)
# is shown as untreated for context.
HCC_OTHER = Tissue(7, "hcc_other", 1066.0, 1570.0, 12.500, 1.10, 3750.0, 0.55, 9.0)

TISSUES = [AIR, SKIN, FAT, MUSCLE, BONE, LIVER, HCC, HCC_OTHER]


# ───────────────────────────────────────────────────────────────────────
# Clinical scenarios (corrected ch21b model)
# ───────────────────────────────────────────────────────────────────────


@dataclass
class Scenario:
    name: str
    label: str
    regime: str
    f0: float
    pnp: float
    ppp: float
    pulse_on_s: float
    prf: float                    # per-spot PRF (set by physics: vapor
                                  # cavity dissolution time, residual
                                  # bubble dissolution, thermal relaxation)
    treatment_s: float
    raster_points: int
    shock_alpha_gain: float
    color: str
    interleave_subspots: int = 1  # number of focal points fired in
                                  # parallel via electronic beam-steering;
                                  # effective transducer PRF = prf * this
    pulses_per_point: int = 1     # clinical "doses per spot": HistoSonics
                                  # ~100 (us intrinsic), Khokhlova ~5
                                  # (ms shock-vapor), Mancia ~100 (ms
                                  # sub-threshold). Drives per-shot mech
                                  # accumulation AND treatment time —
                                  # NOT back-computed from treatment_s.


# Raster_points and treatment_s are AUTO-SIZED for each scenario so that
# the raster pitch is matched to the per-shot footprint radius (full
# overlap → 100% tumour coverage). The treatment time is the minimum
# required: raster_points × pulses_per_point / PRF, where pulses_per_point
# is set to the value needed for accumulated Pcav ≥ 0.95 (μs) or 1
# (single seeding/cloud pulse for ms regimes).
SCENARIOS = [
    # μs intrinsic: per-spot PRF 200 Hz (Vlaisavljevich 2015 optimal,
    # set by ~5 ms residual-bubble dissolution time). Interleaving is
    # not needed — already fast enough.
    Scenario("us_intrinsic", "us intrinsic-threshold (1 MHz, 30 MPa)", "intrinsic",
             1.0e6, 30.0e6, 80.0e6, 2.0e-6, 200.0, 0.0, 0, 1.0, "#1f77b4",
             interleave_subspots=1, pulses_per_point=100),
    # ms shock-vapor: per-spot PRF 1 Hz (vapor-cavity dissolution +
    # thermal relaxation). Interleaving across 8 sub-spots via electronic
    # beam-steering raises effective transducer PRF to 8 Hz, while each
    # spot still sees 1 s between successive pulses. Subspots must be
    # ≥ 5 mm apart to avoid thermal cross-talk (Khokhlova 2014 Ch. 5).
    Scenario("ms_shock_vapor", "ms shock-vapor (1 MHz, 15 MPa, 10 ms)", "shock_vapor",
             1.0e6, 15.0e6, 85.0e6, 10.0e-3, 1.0, 0.0, 0, 10.0, "#d62728",
             interleave_subspots=8, pulses_per_point=5),
    # ms sub-threshold: per-spot PRF 2 Hz, 4 subspots → 8 Hz.
    # ms sub-threshold (shock-scattering): per-spot 10 Hz with PRF
    # dithering (Bader 2018, Mancia 2020) — dither breaks shock-cloud
    # coherence so the inertial-collapse memory constraint relaxes
    # below 500 ms. 8 interleaved subspots → 80 Hz effective transducer
    # rate. 50 pulses/spot is sufficient for full erosion (Mancia 2020).
    Scenario("ms_subthr_cav", "ms sub-threshold cav (500 kHz, 18 MPa, 5 ms)", "subthreshold_cav",
             0.5e6, 18.0e6, 35.0e6, 5.0e-3, 10.0, 0.0, 0, 2.5, "#2ca02c",
             interleave_subspots=8, pulses_per_point=50),
]


def autosize_raster(
    sc: Scenario, per_shot_extents_m: tuple[float, float, float],
    tumour_volume_m3: float,
) -> Scenario:
    """Set raster_points and treatment_s so the per-axis pitch matches
    the per-shot footprint half-extent in that axis. The focal pressure
    is an elongated ellipsoid (axial DOF ≫ lateral FWHM); using a
    single equivalent-sphere radius would under-pack along the lateral
    axes. ``per_shot_extents_m`` are the (x, y, z) half-extents of the
    per-shot mask bounding box.
    """
    # 0.4 overlap → adjacent per-shot footprints overlap by 60% along
    # each axis, which fills corner gaps in the rectangular grid that
    # would otherwise leave the tumour partially uncovered. Use the
    # SMALLEST half-extent for all three axes — the axial focal
    # envelope's periphery contributes weakly to the per-shot lesion,
    # so the lateral lateral half-extent is the load-bearing dimension.
    overlap = 0.4
    hx, hy, hz = per_shot_extents_m
    h_load = min(hx, hy, hz)
    pitch_x = pitch_y = pitch_z = max(2.0 * h_load * overlap, 1e-3)
    raster_points = max(int(np.ceil(tumour_volume_m3 / (pitch_x * pitch_y * pitch_z))), 1)

    pulses_per_pt = max(sc.pulses_per_point, 1)
    effective_prf = sc.prf * max(sc.interleave_subspots, 1)
    treatment_s = raster_points * pulses_per_pt / effective_prf
    return Scenario(
        name=sc.name, label=sc.label, regime=sc.regime,
        f0=sc.f0, pnp=sc.pnp, ppp=sc.ppp,
        pulse_on_s=sc.pulse_on_s, prf=sc.prf,
        treatment_s=treatment_s, raster_points=raster_points,
        shock_alpha_gain=sc.shock_alpha_gain, color=sc.color,
        interleave_subspots=sc.interleave_subspots,
        pulses_per_point=pulses_per_pt,
    )


# ───────────────────────────────────────────────────────────────────────
# CT + native-segmentation loading (LiTS17)
# ───────────────────────────────────────────────────────────────────────


def load_ct_and_segment(target_dx_m: float = 1.2e-3) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load LiTS17 case-0 CT + segmentation, crop to a slab covering
    the liver tumour with margin, resample to isotropic, and build the
    tissue-label volume — liver + HCC tumour from native labels, the
    surrounding tissue from HU thresholding.
    """
    print(f"[ch21e] Loading CT:  {CT_PATH}")
    print(f"[ch21e] Loading seg: {SEG_PATH}")
    v = nib.load(CT_PATH)
    s = nib.load(SEG_PATH)
    raw = v.get_fdata().astype(np.float32)         # (L, A, S) = (512, 512, 75)
    seg_raw = s.get_fdata().astype(np.int16)
    zooms = v.header.get_zooms()
    axcodes = nib.aff2axcodes(v.affine)
    print(f"  raw shape={raw.shape}, voxel mm={zooms}, axcodes={axcodes}")
    assert raw.shape == seg_raw.shape

    # LiTS17 case-0 has 11 disconnected tumour foci (multifocal HCC,
    # typical of advanced cirrhosis). Histotripsy treatment plans
    # target one lesion at a time, so the cropped slab is centred on
    # the LARGEST connected component and that single focus is used
    # as the histotripsy target. The other foci stay labelled (so
    # they appear in the figures as untreated lesions in adjacent
    # liver) but are not driven by the raster planner.
    tum_full = (seg_raw == 2)
    cc, n_cc = label(tum_full)
    if n_cc == 0:
        raise SystemExit("LiTS17 segmentation has no label-2 (tumour) voxels.")
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    biggest = int(np.argmax(sizes))
    largest_focus = (cc == biggest)
    # Mark the largest focus as label 2 (target) and the rest as
    # label 3 (untreated visualization-only). The downstream tissue
    # builder uses HCC tissue for label 2 and a benign-tumour stand-in
    # for label 3 so they're visually distinguishable.
    seg_raw[largest_focus] = 2
    seg_raw[(~largest_focus) & tum_full] = 3
    print(f"  tumour: {n_cc} foci totalling {tum_full.sum()} vox; "
          f"target = largest focus ({largest_focus.sum()} vox); "
          f"others ({tum_full.sum() - largest_focus.sum()} vox) flagged untreated")
    tum_coords = np.argwhere(largest_focus)
    margin_lat = 100  # 0.7 mm × 100 ≈ 70 mm
    margin_ap = 100   # 0.7 mm × 100 ≈ 70 mm
    margin_si = 12    # 5 mm × 12 = 60 mm
    l0 = max(0, int(tum_coords[:, 0].min()) - margin_lat)
    l1 = min(raw.shape[0], int(tum_coords[:, 0].max()) + margin_lat)
    a0 = max(0, int(tum_coords[:, 1].min()) - margin_ap)
    a1 = min(raw.shape[1], int(tum_coords[:, 1].max()) + margin_ap)
    s0 = max(0, int(tum_coords[:, 2].min()) - margin_si)
    s1 = min(raw.shape[2], int(tum_coords[:, 2].max()) + margin_si)
    raw = raw[l0:l1, a0:a1, s0:s1]
    seg_raw = seg_raw[l0:l1, a0:a1, s0:s1]
    print(f"  cropped to shape={raw.shape} around tumour bbox")

    # Re-orient (L, A, S) → (depth-from-anterior, lateral, superior-inferior)
    # so axis 0 is the histotripsy beam direction. Transpose (1, 0, 2)
    # turns (L, A, S) → (A, L, S); flipping along axis 0 turns
    # "increasing toward Anterior" into "increasing toward Posterior" —
    # i.e. depth-from-anterior, matching ch21d's convention.
    raw = np.flip(np.transpose(raw, (1, 0, 2)), axis=0)
    seg_raw = np.flip(np.transpose(seg_raw, (1, 0, 2)), axis=0)
    raw_zooms = (zooms[1], zooms[0], zooms[2])

    target_dx_mm = target_dx_m * 1e3
    factors = tuple(z / target_dx_mm for z in raw_zooms)
    ct_hu = zoom(raw, factors, order=1, prefilter=False).astype(np.float32)
    seg = zoom(seg_raw.astype(np.float32), factors, order=0,
               prefilter=False).astype(np.int8)
    print(f"  resampled shape={ct_hu.shape}, voxel={target_dx_mm:.2f} mm")

    nx, ny, nz = ct_hu.shape

    # Tissue map — HU thresholds for everything except liver + tumour,
    # which come from the native segmentation.
    label_vol = np.zeros_like(ct_hu, dtype=np.int8)
    label_vol[ct_hu < -500] = AIR.label
    label_vol[(ct_hu >= -500) & (ct_hu < -100)] = FAT.label
    label_vol[(ct_hu >= -100) & (ct_hu < 30)]  = MUSCLE.label
    label_vol[(ct_hu >= 30) & (ct_hu < 200)]   = MUSCLE.label
    label_vol[(ct_hu >= 200)] = BONE.label
    skin_thickness_vox = max(int(round(2.0e-3 / target_dx_m)), 1)
    body = label_vol != AIR.label
    body_dil = binary_dilation(~body, iterations=skin_thickness_vox)
    skin_shell = body & body_dil
    label_vol[skin_shell] = SKIN.label
    label_vol[seg == 1] = LIVER.label
    label_vol[seg == 2] = HCC.label        # treatment target
    label_vol[seg == 3] = HCC_OTHER.label  # untreated foci (visualised
                                           # but excluded from the
                                           # raster planner)

    tumour_mask = label_vol == HCC.label
    if not tumour_mask.any():
        raise SystemExit("HCC mask vanished after resampling")
    coords = np.argwhere(tumour_mask)
    centroid = coords.mean(axis=0).astype(int)
    bbox_extent = (coords.max(0) - coords.min(0)) + 1
    r_tumour_eq_m = ((tumour_mask.sum() * (target_dx_m ** 3)) * 3.0 / (4.0 * np.pi)) ** (1 / 3)
    print(f"  tumour: {tumour_mask.sum()} vox, "
          f"bbox {tuple(bbox_extent)} vox, "
          f"r_eq={r_tumour_eq_m*1e3:.1f} mm, "
          f"centroid={tuple(centroid)}")

    info = {
        "dx": target_dx_m,
        "shape": ct_hu.shape,
        "x_axis": np.arange(nx) * target_dx_m,
        "y_axis": (np.arange(ny) - ny / 2) * target_dx_m,
        "z_axis": (np.arange(nz) - nz / 2) * target_dx_m,
        "focus_idx": tuple(int(c) for c in centroid),
        "x_focus": float(centroid[0] * target_dx_m),
        "r_tumour": r_tumour_eq_m,
        "organ_label": "Liver",
        "tumour_label": "HCC",
        "dataset_label": "LiTS17 case-0",
    }
    return ct_hu, label_vol, info


def property_maps(label_vol: np.ndarray, f0: float) -> dict:
    out = {k: np.zeros_like(label_vol, dtype=np.float32)
           for k in ("rho", "c", "alpha", "cp", "kappa", "perfusion")}
    f_mhz = f0 / 1e6
    for t in TISSUES:
        m = label_vol == t.label
        if not m.any():
            continue
        out["rho"][m] = t.rho
        out["c"][m] = t.c
        out["alpha"][m] = t.alpha0 * (f_mhz ** t.y_pow)
        out["cp"][m] = t.cp
        out["kappa"][m] = t.kappa
        out["perfusion"][m] = t.perfusion
    return out


# ───────────────────────────────────────────────────────────────────────
# Forward propagation (same model as ch21b)
# ───────────────────────────────────────────────────────────────────────


def steering_efficiency(dr_lat_m: float, dr_ax_m: float, f0: float,
                        apodized: bool = True) -> float:
    """Phased-array off-focus efficiency for the 50 mm aperture / 120 mm
    radius-of-curvature bowl. Combines element-directivity loss
    (cos θ, Hand 2009), projected-aperture loss (cos θ), and Gaussian
    roll-off of grating-lobe onset.

    Without apodization: ε ~ cos²θ (both directivity and projection)
    With apodization (re-weighting elements by cos θ toward steered
    focus): ε ~ cos θ (only directivity remains; projection loss is
    largely recovered). Empirically this extends the characteristic
    compensable range by ~√2 — from ±5/15 mm to ±7/21 mm at 1 MHz.
    Modern clinical systems use apodization by default.
    """
    lam_m = 1540.0 / f0
    base_lat = 7.0e-3 if apodized else 5.0e-3
    base_ax = 21.0e-3 if apodized else 15.0e-3
    R_lat = base_lat * (lam_m / 1.54e-3)   # 1 MHz λ = 1.54 mm
    R_ax  = base_ax  * (lam_m / 1.54e-3)
    return float(np.exp(-((dr_lat_m / R_lat) ** 2 + (dr_ax_m / R_ax) ** 2)))


def steering_pressure_factor(shot_idx: tuple[int, int, int],
                             anchor_idx: tuple[int, int, int],
                             dx_m: float, f0: float,
                             amp_headroom: float = 1.5,
                             apodized: bool = True) -> float:
    """Effective per-shot focal-peak pressure factor when the array
    steers electronically from `anchor_idx` (its current mechanical
    pose) to `shot_idx`. Drive amplitude is boosted by 1/ε to maintain
    target pressure; if 1/ε > amp_headroom (1.5× amplifier limit) the
    shot is pressure-degraded with factor = ε · headroom < 1.
    """
    shift = (np.array(shot_idx) - np.array(anchor_idx)) * dx_m
    dr_ax = float(abs(shift[0]))
    dr_lat = float(np.hypot(shift[1], shift[2]))
    eps = steering_efficiency(dr_lat, dr_ax, f0, apodized=apodized)
    comp = min(amp_headroom, 1.0 / max(eps, 1e-9))
    return eps * comp


def mechanical_walk(path: list[tuple[int, int, int]],
                    focus_idx: tuple[int, int, int],
                    dx_m: float, f0: float,
                    apodized: bool = True,
                    eps_min: float = 0.7) -> tuple[list[tuple[int, int, int]], int]:
    """Walk the path; if the next shot would land at an electronically-
    steered position with efficiency below `eps_min` (= 1/headroom for
    headroom 1.43, the practical cutoff before the amplifier
    saturates), mechanically translate the array so that shot becomes
    the new anchor. Returns per-shot anchor list and the number of
    mechanical re-anchors. By construction, every shot then sits
    within the compensable steering window and reaches full dose.
    """
    anchors: list[tuple[int, int, int]] = []
    n_changes = 0
    current = tuple(int(c) for c in focus_idx)
    for pt in path:
        shift = (np.array(pt) - np.array(current)) * dx_m
        eps = steering_efficiency(float(np.hypot(shift[1], shift[2])),
                                  float(abs(shift[0])), f0, apodized=apodized)
        if eps < eps_min:
            current = tuple(int(c) for c in pt)
            n_changes += 1
        anchors.append(current)
    return anchors, n_changes


def focal_fwhm(f0: float) -> tuple[float, float]:
    """Return (lateral, axial) FWHM of the focal pressure spot in metres
    for the 50 mm aperture / 120 mm radius-of-curvature bowl. These are
    the clinical "focal spot size" quoted in literature (Penttinen 1976,
    O'Neil 1949) — w_lat = 1.41·λ·F#, w_axial = 7·λ·F#²."""
    lam = 1540.0 / f0
    fnum = 120.0e-3 / (2.0 * 50.0e-3)
    return 1.41 * lam * fnum, 7.0 * lam * fnum ** 2


USE_PSTD_KERNEL = True  # use cached PSTD kernels (water-medium focal field)
                         # in place of the analytical Gaussian envelope.
                         # Set to False to revert to the closed-form Penttinen
                         # 1976 approximation for A/B comparison.


# Module-level cache so each scenario's kernel envelope is loaded once
# per ch21e run, not per probe / per main-pass call.
_KERNEL_ENV_CACHE: dict = {}


def focused_bowl_pressure(info, props, f0, source_pa) -> np.ndarray:
    """Per-voxel peak rarefactional pressure (Pa) on the planner grid.

    Either:
    * `USE_PSTD_KERNEL = True` — load a real PSTD kernel, resample to
      the planner grid, normalize the focal voxel to 1, then apply
      `source_pa * env * cum_atten`. Real diffraction + sidelobes +
      depth-of-focus from a single PSTD pulse on water.
    * `USE_PSTD_KERNEL = False` — Penttinen 1976 separable-Gaussian
      closed form. Fast, smooth, but misses sidelobes and grid-
      resolution effects. Useful for A/B comparison.

    Layered tissue absorption is applied identically along the central
    beam ray in both paths."""
    nx, ny, nz = info["shape"]
    dx = info["dx"]
    focus_idx = info["focus_idx"]

    if USE_PSTD_KERNEL:
        try:
            from kernel_loader import kernel_focal_envelope  # type: ignore
        except ImportError:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(__file__))
            from kernel_loader import kernel_focal_envelope  # type: ignore
        env = kernel_focal_envelope(
            scenario_f0=f0,
            target_shape=(nx, ny, nz),
            target_focus_idx=focus_idx,
            target_dx_m=dx,
            cache=_KERNEL_ENV_CACHE,
        )
    else:
        x_axis = info["x_axis"]; y_axis = info["y_axis"]; z_axis = info["z_axis"]
        x_focus = focus_idx[0] * dx
        y_focus = focus_idx[1] * dx - ny * dx / 2
        z_focus = focus_idx[2] * dx - nz * dx / 2
        c_ref = 1540.0
        lam = c_ref / f0
        a = 50.0e-3
        R_f = 120.0e-3
        fnum = R_f / (2.0 * a)
        w_lat = 1.41 * lam * fnum
        w_axial = 7.0 * lam * fnum ** 2
        X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
        r_lat2 = (Y - y_focus) ** 2 + (Z - z_focus) ** 2
        env = np.exp(-r_lat2 / (2.0 * (w_lat / 2.355) ** 2)) * np.exp(
            -((X - x_focus) ** 2) / (2.0 * (w_axial / 2.355) ** 2)
        )

    alpha_x = props["alpha"][:, focus_idx[1], focus_idx[2]]
    cum_atten = np.exp(-np.cumsum(alpha_x) * dx)
    p = source_pa * env * cum_atten[:, None, None]
    return p.astype(np.float32)


# ───────────────────────────────────────────────────────────────────────
# Lesion + thermal model (corrected ch21b version)
# ───────────────────────────────────────────────────────────────────────


def intrinsic_threshold(f0: float) -> float:
    """Vlaisavljevich 2015 frequency-dependent intrinsic threshold (Pa)."""
    return 28.2e6 + 1.4e6 * np.log10(f0 / 1e6)


def cav_probability(p, f0):
    pt = 28.2e6 + 1.4e6 * np.log10(f0 / 1e6)
    sigma = 0.96e6
    return 0.5 * (1.0 + erf((p - pt) / (sigma * np.sqrt(2.0))))


def collapse_strength(p, f0):
    p_n = 5.0e6 * np.sqrt(f0 / 1e6)
    return 1.0 + (p / p_n) ** 1.5


def thermal_maps(p_field, props, sc):
    heating_amp = max(sc.ppp / max(sc.pnp, 1.0), 1.0)
    I_eff = (p_field * heating_amp) ** 2 / (2.0 * props["rho"] * props["c"])
    Q_pulse = 2.0 * props["alpha"] * sc.shock_alpha_gain * I_eff
    dT_p = Q_pulse * sc.pulse_on_s / (props["rho"] * props["cp"])
    T_transient = np.minimum(37.0 + dT_p, 100.0)

    duty = sc.pulse_on_s * sc.prf
    Q_avg = Q_pulse * duty
    fnum = 120.0e-3 / (2.0 * 50.0e-3)
    lam = 1540.0 / sc.f0
    w_f = 1.41 * lam * fnum / 2.355
    rho_b_cb = 1060.0 * 3617.0
    diff_term = 4.0 * props["kappa"]
    perf_term = props["perfusion"] * rho_b_cb * w_f ** 2
    T_steady = 37.0 + Q_avg * w_f ** 2 / (diff_term + perf_term)
    if sc.regime == "shock_vapor":
        T_steady = np.minimum(T_steady, 75.0)
    else:
        T_steady = np.minimum(T_steady, 100.0)
    R = np.where(T_steady >= 43.0, 0.5, 0.25)
    cem43 = (R ** (43.0 - T_steady)) * (sc.treatment_s / 60.0)
    cem43[T_steady < 39.0] = 0.0
    return cem43, T_steady, T_transient


def predicted_lesion(p_field, props, sc, info, label_vol):
    pcav = cav_probability(p_field, sc.f0)
    coll = collapse_strength(p_field, sc.f0)
    cem43, T_steady, T_transient = thermal_maps(p_field, props, sc)

    pulses_per_pt = max(sc.pulses_per_point, 1)
    # Cavitation-cloud radius scales with overpressure (Vlaisavljevich
    # 2013, Maxwell 2013): r_cloud ≈ λ/4 at threshold, growing to ~λ/2
    # at p > 1.5 p_t. Dilating the threshold-exceeding region by this
    # radius models the bubble cloud expansion beyond the strict
    # p > p_t kernel and is the correct ablation-mask construction at
    # any grid spacing where dx > λ/4.
    lam_m = 1540.0 / sc.f0
    cloud_r_m = lam_m / 4.0 * max(sc.pnp / intrinsic_threshold(sc.f0), 1.0)
    cloud_vox = max(int(round(cloud_r_m / info["dx"])), 1)
    if sc.regime == "intrinsic":
        p_acc = 1.0 - (1.0 - pcav) ** pulses_per_pt
        mech = binary_dilation(p_acc >= 0.95, iterations=cloud_vox)
    elif sc.regime == "shock_vapor":
        seed = T_transient >= 100.0
        sv_cloud = max(int(round(3.0e-3 / info["dx"])), 1)
        mech = binary_dilation(seed, iterations=sv_cloud)
    elif sc.regime == "subthreshold_cav":
        mech = binary_dilation(coll >= 5.0, iterations=cloud_vox)
    else:
        mech = np.zeros_like(p_field, dtype=bool)
    therm = cem43 >= 240.0

    # Per-shot footprint volume (before raster superposition) — the
    # lesion produced by a single focal-point exposure, NOT the total
    # raster-summed lesion. This is the relevant metric for comparing
    # the per-pulse lesion mechanism size across regimes.
    voxel_vol_mm3 = (info["dx"] * 1e3) ** 3
    per_shot_volume_mm3 = float((mech | therm).sum() * voxel_vol_mm3)
    per_shot_radius_mm = (per_shot_volume_mm3 * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)

    tumour = label_vol == HCC.label
    coords = np.argwhere(tumour)
    if len(coords) == 0:
        return mech | therm, {"pcav": pcav, "T_steady": T_steady, "T_transient": T_transient,
                              "cem43": cem43, "metrics": {}}
    cmin = coords.min(0); cmax = coords.max(0) + 1
    extent = cmax - cmin

    # FWHM of the transducer focal spot (clinical "spot size"
    # convention, Penttinen 1976). FWHM defines the visualization spot
    # radius. The raster pitch and tumour-confinement erosion use the
    # SMALLER of (FWHM half, cavitation-mech half-extent) per axis: for
    # near-threshold us regimes the cavitation footprint is narrower
    # than FWHM (only the beam's peak exceeds p_t) so pitch must follow
    # the cavitation extent to avoid coverage gaps; for ms regimes the
    # cavitation cloud is wider than FWHM and FWHM is the binding
    # constraint that keeps spillover out of healthy liver.
    fwhm_lat_m, fwhm_ax_m = focal_fwhm(sc.f0)
    fwhm_lat_vox = max(int(round(fwhm_lat_m / info["dx"])), 1)
    fwhm_ax_vox = max(int(round(fwhm_ax_m / info["dx"])), 1)
    if mech.any() or therm.any():
        ps = np.argwhere(mech | therm)
        ps_extent = ps.max(0) - ps.min(0) + 1
        mech_hx = max(int(np.ceil(ps_extent[0] / 2.0)), 1)
        mech_hy = max(int(np.ceil(ps_extent[1] / 2.0)), 1)
        mech_hz = max(int(np.ceil(ps_extent[2] / 2.0)), 1)
    else:
        mech_hx = mech_hy = mech_hz = 1
    fwhm_hx = max(fwhm_ax_vox // 2, 1)
    fwhm_hy = max(fwhm_lat_vox // 2, 1)
    fwhm_hz = max(fwhm_lat_vox // 2, 1)
    hx = min(fwhm_hx, mech_hx)
    hy = min(fwhm_hy, mech_hy)
    hz = min(fwhm_hz, mech_hz)
    # 75% overlap (pitch = half the half-extent) — pitch = h means
    # neighbour mask centres are h apart and each mask reaches ±h, so
    # adjacent footprints overlap heavily at the midpoint, eliminating
    # rim gaps at the cost of ~2× shots per axis. Necessary for us
    # intrinsic where the threshold-exceeded core is sub-FWHM.
    pitch_x = max(int(round(0.5 * hx)), 1)
    pitch_y = max(int(round(0.5 * hy)), 1)
    pitch_z = max(int(round(0.5 * hz)), 1)

    # Anisotropic erosion of the tumour by the per-shot half-extents.
    # A raster centre is valid only where a (2hx+1)×(2hy+1)×(2hz+1) box
    # centred on it lies entirely inside the tumour — this guarantees the
    # per-shot footprint cannot spill outside the HCC outline. If the
    # tumour is smaller than the per-shot footprint along any axis the
    # erosion empties; we then relax the constraint progressively
    # (start with full half-extents, halve until non-empty) so the raster
    # still covers the tumour while keeping spillover minimal.
    # Try anisotropic erosion; if any axis can't fit a (2h+1)-thick
    # structuring element inside the tumour, relax that axis only
    # (rather than all three together) so the raster still fills the
    # tumour densely along the well-confined axes. Final fallback uses
    # the bare tumour mask — i.e. accept spillover when geometry forbids
    # full confinement (e.g. ms sub-threshold cigar in 30 mm tumour).
    def _try_erode(rx, ry, rz):
        struct = np.ones((2*rx + 1, 2*ry + 1, 2*rz + 1), dtype=bool)
        return binary_erosion(tumour, structure=struct)
    # Erode by ONE PITCH (not the full half-extent) — this keeps raster
    # centres a pitch's worth from the boundary, ensuring per-shot
    # masks reach the rim (full-coverage requirement) while limiting
    # spillover to (mask_half − pitch) per axis. Eroding by full
    # half-extent would over-confine, leaving the tumour rim
    # unsampled and capping coverage at ~60%.
    valid_centres = _try_erode(pitch_x, pitch_y, pitch_z)
    if not valid_centres.any():
        # Per-axis relaxation: shrink the largest axis first.
        for shrink_factor in (0.66, 0.33, 0.0):
            rx = max(int(round(hx*shrink_factor)), 0)
            ry = max(int(round(hy*shrink_factor)), 0)
            rz = max(int(round(hz*shrink_factor)), 0)
            valid_centres = _try_erode(rx, ry, rz)
            if valid_centres.any():
                break
        if not valid_centres.any():
            valid_centres = tumour.copy()

    # Farthest-point sampling on the tumour bounding box ONLY (not the
    # full simulation grid) so distance_transform_edt stays O(tumour
    # vox), not O(grid vox). For a ~9k-voxel tumour the FPS loop is
    # tractable; on the full 11M-voxel grid it would OOM.
    bb_min = np.maximum(coords.min(0) - 2, 0)
    bb_max = np.minimum(coords.max(0) + 3, np.array(tumour.shape))
    sl = tuple(slice(bb_min[i], bb_max[i]) for i in range(3))
    valid_local = valid_centres[sl]
    if not valid_local.any():
        valid_local = tumour[sl].copy()

    centroid_local = coords.mean(axis=0) - bb_min
    valid_idx_local = np.argwhere(valid_local)
    d_to_centroid = np.linalg.norm(valid_idx_local - centroid_local, axis=1)
    seed_local = tuple(valid_idx_local[int(np.argmin(d_to_centroid))])
    raster_local = np.zeros_like(valid_local, dtype=bool)
    raster_local[seed_local] = True

    sampling = (1.0 / max(pitch_x, 1), 1.0 / max(pitch_y, 1), 1.0 / max(pitch_z, 1))
    target_d = 1.0
    max_pts = int(valid_local.sum())  # cannot exceed valid voxel count
    for _ in range(max_pts):
        d_now = distance_transform_edt(~raster_local, sampling=sampling)
        d_masked = np.where(valid_local, d_now, -1.0)
        idx = np.unravel_index(int(np.argmax(d_masked)), d_masked.shape)
        if d_masked[idx] <= target_d:
            break
        raster_local[idx] = True

    raster_grid = np.zeros_like(tumour, dtype=bool)
    raster_grid[sl] = raster_local

    def superpose(per_shot):
        """Place a copy of `per_shot` (centred on the focal voxel) at
        every raster point and OR the results. Direct placement avoids
        FFT-convolution centring ambiguity."""
        if not per_shot.any() or not raster_grid.any():
            return np.zeros_like(per_shot, dtype=bool)
        focus_idx = np.array(info["focus_idx"])
        nx, ny, nz = per_shot.shape
        # Offset from per-shot focus voxel to mask voxels (vectorised)
        off = np.argwhere(per_shot) - focus_idx  # (N, 3) relative coords
        out = np.zeros_like(per_shot, dtype=bool)
        for rpt in np.argwhere(raster_grid):
            tgt = rpt + off  # (N, 3) target voxels
            ok = ((tgt >= 0) & (tgt < np.array([nx, ny, nz]))).all(axis=1)
            tgt = tgt[ok]
            out[tgt[:, 0], tgt[:, 1], tgt[:, 2]] = True
        return out

    mech_super = superpose(mech)
    therm_super = superpose(therm)
    lesion_super = mech_super | therm_super
    mech_full = mech_super & tumour
    therm_full = therm_super & tumour
    lesion = mech_full | therm_full

    # Cavitation-dose heatmap — accumulated 1 - (1 - pcav)^N over all
    # raster points × pulses_per_pt. Computed by shifting log(1-pcav)
    # to each raster centre and summing, then dose = 1 - exp(sum).
    # Sub-threshold scenarios use collapse strength normalised to [0,1]
    # since their mechanism is multi-pulse stochastic erosion, not the
    # single-pulse erf-CDF intrinsic threshold.
    if sc.regime == "subthreshold_cav":
        per_pulse = np.clip((coll - 1.0) / 9.0, 0.0, 0.99).astype(np.float32)
    else:
        per_pulse = np.clip(pcav.astype(np.float32), 0.0, 0.99)
    log_safe = np.log(1.0 - per_pulse)  # ≤ 0
    log_dose_super = np.zeros_like(log_safe)
    n_degraded = 0
    n_mech_anchors = 0
    if raster_grid.any():
        focus_idx = np.array(info["focus_idx"])
        nx_, ny_, nz_ = log_safe.shape
        # Order shots serpentine for the path-walk; the dose accumulator
        # is order-invariant but the mechanical-walk needs an order to
        # decide when to re-anchor. The animation strategies override
        # this with their own ordering.
        ordered = serpentine_order(raster_grid)
        anchors, n_mech_anchors = mechanical_walk(
            ordered, tuple(int(c) for c in focus_idx), info["dx"], sc.f0)
        anchor_lookup = {pt: anch for pt, anch in zip(ordered, anchors)}
        for rpt in np.argwhere(raster_grid):
            shift = rpt - focus_idx
            src_x0 = max(0, -shift[0]); src_x1 = min(nx_, nx_ - shift[0])
            src_y0 = max(0, -shift[1]); src_y1 = min(ny_, ny_ - shift[1])
            src_z0 = max(0, -shift[2]); src_z1 = min(nz_, nz_ - shift[2])
            if src_x1 <= src_x0 or src_y1 <= src_y0 or src_z1 <= src_z0:
                continue
            anchor = anchor_lookup.get(tuple(int(c) for c in rpt),
                                       tuple(int(c) for c in focus_idx))
            k = steering_pressure_factor(tuple(int(c) for c in rpt),
                                         anchor, info["dx"], sc.f0)
            if k < 0.999:
                n_degraded += 1
            dst_x0 = src_x0 + shift[0]; dst_x1 = src_x1 + shift[0]
            dst_y0 = src_y0 + shift[1]; dst_y1 = src_y1 + shift[1]
            dst_z0 = src_z0 + shift[2]; dst_z1 = src_z1 + shift[2]
            log_dose_super[dst_x0:dst_x1, dst_y0:dst_y1, dst_z0:dst_z1] += \
                log_safe[src_x0:src_x1, src_y0:src_y1, src_z0:src_z1] * k
    cav_dose = 1.0 - np.exp(log_dose_super * pulses_per_pt)

    # Treatment time is set by the ACTUAL placed raster fill: each spot
    # gets pulses_per_pt clinical doses, with electronic interleave
    # spreading across N subspots so the transducer's effective PRF is
    # sc.prf * sc.interleave_subspots.
    actual_n = max(int(raster_grid.sum()), 1)
    effective_prf = sc.prf * max(sc.interleave_subspots, 1)
    # Mechanical re-anchoring penalty: ~2 s per repositioning is
    # typical for clinical robotic stages (HistoSonics, Insightec
    # ExAblate). Add it to the electronic-sonication time so the
    # treatment_s is honest about wall-clock duration.
    mech_repo_s = 2.0 * n_mech_anchors
    actual_treatment_s = actual_n * pulses_per_pt / effective_prf + mech_repo_s
    if sc.treatment_s > 0:
        cem43 = cem43 * (actual_treatment_s / sc.treatment_s)
    # Spillover = ablation outside the HCC outline; a tighter raster
    # confines this to a thin rim from per-shot footprints touching the
    # tumour boundary.
    spillover = lesion_super & (~tumour)
    raster_grid_out = raster_grid  # capture for plotting
    voxel_vol = (info["dx"] * 1e3) ** 3
    raster_pitch_mm = float(np.mean([pitch_x, pitch_y, pitch_z]) * info["dx"] * 1e3)
    raster_pitch_xyz_mm = (pitch_x * info["dx"] * 1e3,
                           pitch_y * info["dx"] * 1e3,
                           pitch_z * info["dx"] * 1e3)
    metrics = {
        "scenario": sc.name,
        "per_shot_volume_mm3": per_shot_volume_mm3,
        "per_shot_radius_mm": per_shot_radius_mm,
        "raster_points": sc.raster_points,
        "raster_pitch_mm": raster_pitch_mm,
        "raster_pitch_xyz_mm": raster_pitch_xyz_mm,
        "interleave_subspots": sc.interleave_subspots,
        "per_spot_prf_hz": sc.prf,
        "effective_prf_hz": effective_prf,
        "pulses_per_point": pulses_per_pt,
        "treatment_min": actual_treatment_s / 60.0,
        "actual_treatment_s": actual_treatment_s,
        "lesion_volume_mm3": float(lesion.sum() * voxel_vol),
        "mech_volume_mm3": float(mech_full.sum() * voxel_vol),
        "therm_volume_mm3": float(therm_full.sum() * voxel_vol),
        "T_transient_C": float(T_transient.max()),
        "T_steady_C": float(T_steady.max()),
        "tumour_volume_mm3": float(tumour.sum() * voxel_vol),
        "tumour_coverage_pct": float(100.0 * (lesion & tumour).sum() / max(tumour.sum(), 1)),
        "actual_raster_points": int(raster_grid.sum()),
        "spillover_volume_mm3": float(spillover.sum() * voxel_vol),
        "confinement_pct": float(100.0 * lesion.sum() / max(lesion_super.sum(), 1)),
        "n_steering_degraded_shots": int(n_degraded),
        "n_mechanical_reanchors": int(n_mech_anchors),
        "mechanical_reposition_s": float(mech_repo_s),
    }
    metrics["fwhm_lat_mm"] = fwhm_lat_m * 1e3
    metrics["fwhm_ax_mm"] = fwhm_ax_m * 1e3
    return lesion, {"pcav": pcav, "T_steady": T_steady, "T_transient": T_transient,
                    "cem43": cem43, "metrics": metrics,
                    "raster_grid": raster_grid_out, "per_shot_mask": mech | therm,
                    "cav_dose": cav_dose,
                    "fwhm_lat_m": fwhm_lat_m, "fwhm_ax_m": fwhm_ax_m}


# ───────────────────────────────────────────────────────────────────────
# Plotting helpers + base64 embedding
# ───────────────────────────────────────────────────────────────────────


def outside_in_order(raster_grid: np.ndarray) -> list[tuple[int, int, int]]:
    """Sort raster points by distance from the raster centroid, descending —
    the most-peripheral shot fires first. Avoids bubble-cloud shadowing
    of peripheral shots by an already-treated central core (Maxwell 2013,
    Macoskey 2018)."""
    pts = np.argwhere(raster_grid)
    if len(pts) == 0:
        return []
    centroid = pts.mean(axis=0)
    d = np.linalg.norm(pts - centroid, axis=1)
    order = np.argsort(-d)
    return [tuple(int(c) for c in pts[i]) for i in order]


def inside_out_order(raster_grid: np.ndarray) -> list[tuple[int, int, int]]:
    """Sort raster points by distance from the raster centroid, ascending —
    centre shots fire first. Residual nuclei from these early shots
    lower the cavitation threshold for subsequent peripheral pulses
    (Vlaisavljevich 2013)."""
    pts = np.argwhere(raster_grid)
    if len(pts) == 0:
        return []
    centroid = pts.mean(axis=0)
    d = np.linalg.norm(pts - centroid, axis=1)
    order = np.argsort(d)
    return [tuple(int(c) for c in pts[i]) for i in order]


def adaptive_lowdose_order(raster_grid: np.ndarray, per_pulse: np.ndarray,
                           focus_idx: tuple[int, int, int],
                           pulses_per_pt: int) -> list[tuple[int, int, int]]:
    """Greedy adaptive ordering: at each step, place the next shot at
    the raster point whose per-shot footprint has the largest centre-
    voxel deficit relative to the running cumulative-dose field. This
    minimises the worst-case under-treated voxel at every intermediate
    time so the tumour fills uniformly rather than corner-first."""
    pts = np.argwhere(raster_grid)
    if len(pts) == 0:
        return []
    fx_arr = np.array(focus_idx)
    nx, ny, nz = per_pulse.shape
    log_safe = np.log(np.clip(1.0 - per_pulse.astype(np.float32), 1e-6, 1.0))
    log_acc = np.zeros_like(log_safe)
    remaining = pts.tolist()
    order: list[tuple[int, int, int]] = []
    while remaining:
        # Score each candidate by the worst-case deficit it would
        # close: the candidate that maximises (per_pulse · pulses) at
        # the currently-lowest-dose tumour voxel within its footprint.
        # Approximated by picking the candidate furthest from the
        # raster centroid of already-placed shots when log_acc is flat,
        # and increasingly steered toward the lowest-dose voxel as
        # the dose field develops. Implemented as: greedy farthest-
        # point in dose-space (= highest absolute log_acc value at
        # candidate's centre).
        if not order:
            # First shot: tumour centroid (any spot fires the seed)
            centroid = np.argwhere(raster_grid).mean(axis=0)
            d = np.linalg.norm(np.array(remaining) - centroid, axis=1)
            idx_pick = int(np.argmin(d))
        else:
            cands = np.array(remaining)
            scores = np.array([log_acc[c[0], c[1], c[2]] for c in cands])
            # log_acc is ≤0; smaller value = MORE dose. Pick the
            # candidate at the LEAST-dosed location (largest log_acc).
            idx_pick = int(np.argmax(scores))
        pick = remaining.pop(idx_pick)
        order.append(tuple(int(c) for c in pick))
        # Add this shot's per_pulse contribution to the running log_acc
        shift = np.array(pick) - fx_arr
        sx0 = max(0, -shift[0]); sx1 = min(nx, nx - shift[0])
        sy0 = max(0, -shift[1]); sy1 = min(ny, ny - shift[1])
        sz0 = max(0, -shift[2]); sz1 = min(nz, nz - shift[2])
        if sx1 > sx0 and sy1 > sy0 and sz1 > sz0:
            log_acc[sx0+shift[0]:sx1+shift[0],
                    sy0+shift[1]:sy1+shift[1],
                    sz0+shift[2]:sz1+shift[2]] += \
                log_safe[sx0:sx1, sy0:sy1, sz0:sz1] * pulses_per_pt
    return order


def serpentine_order(raster_grid: np.ndarray) -> list[tuple[int, int, int]]:
    """Return raster centres in serpentine scan order: outer loop over
    x (depth), then y (lateral), z reverses each y row. Mimics the
    boustrophedon path used by clinical electronic-steering systems."""
    pts = np.argwhere(raster_grid)
    if len(pts) == 0:
        return []
    pts = sorted(pts.tolist(), key=lambda p: (p[0], p[1], p[2]))
    out = []
    by_xy = {}
    for x, y, z in pts:
        by_xy.setdefault((x, y), []).append(z)
    for (x, y), zs in sorted(by_xy.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if y % 2 == 1:
            zs = sorted(zs, reverse=True)
        else:
            zs = sorted(zs)
        for z in zs:
            out.append((x, y, z))
    return out


def make_sonication_animation(ct, label_vol, info, results, scenarios,
                              n_frames: int = 80, fps: int = 12) -> str:
    """Render a 3-panel animated GIF of cumulative cavitation-dose
    delivery. Each panel's timeline runs from 0 → 100% of that
    scenario's actual treatment time, so all three animate end-to-end
    in parallel; the on-screen status text reports each scenario's
    real elapsed time. The dose is computed incrementally from the
    serpentine raster path: at frame f, log-dose = pulses_per_pt ×
    Σ_{shot i ≤ f} log(1 - p_pulse) shifted to that shot's centre.
    Rendered as an inferno heatmap so the viewer sees the per-shot
    dose accumulation, not flat-coloured discs.

    Returns the GIF as a base64 string for markdown embedding.
    """
    print("[ch21e] Building sonication animation")
    from matplotlib.animation import PillowWriter, FuncAnimation

    fx = info["focus_idx"][0]
    tumour = label_vol == HCC.label
    coords = np.argwhere(tumour)
    cmin = coords.min(axis=0); cmax = coords.max(axis=0)
    pad = int(0.04 / info["dx"])
    y0 = max(0, cmin[1] - pad); y1 = min(label_vol.shape[1], cmax[1] + pad)
    z0 = max(0, cmin[2] - pad); z1 = min(label_vol.shape[2], cmax[2] + pad)
    extent_crop = [info["z_axis"][z0]*1e3, info["z_axis"][z1-1]*1e3,
                   info["y_axis"][y1-1]*1e3, info["y_axis"][y0]*1e3]

    # Pre-compute the per-pulse log(1 - p) field on the cropped y-z
    # slice through the focal x-plane. Each scenario uses its regime's
    # per-pulse cavitation-probability surrogate (same as the static
    # cav_dose heatmap in fig 14).
    from matplotlib.colors import LogNorm
    paths = {}
    per_pulse_slice = {}
    anchors_per_path = {}
    focus_y_local = info["focus_idx"][1] - y0
    focus_z_local = info["focus_idx"][2] - z0
    ny_loc = y1 - y0
    nz_loc = z1 - z0
    for sc in scenarios:
        paths[sc.name] = serpentine_order(results[sc.name]["raster_grid"])
        anchors_per_path[sc.name], _ = mechanical_walk(
            paths[sc.name], info["focus_idx"], info["dx"], sc.f0)
        r = results[sc.name]
        # Recompute per_pulse on the slice from pcav (intrinsic) or
        # collapse strength (sub-threshold). Shock-vapor uses pcav too
        # since the seed is thermal (binary); the heatmap shows the
        # cumulative cavitation-probability dose contributed by every
        # shot regardless of whether the seed is mechanical or thermal.
        if sc.regime == "subthreshold_cav":
            coll_sl = collapse_strength(r["p_field"][fx, y0:y1, z0:z1], sc.f0)
            per_pulse = np.clip((coll_sl - 1.0) / 9.0, 0.0, 1.0).astype(np.float32)
        elif sc.regime == "shock_vapor":
            # Smooth-graded per-shot deposition: weight the dilated
            # thermal-seed mask by the normalized transient-temperature
            # rise so the cloud core deposits more "cavitation events"
            # per pulse than the rim — gives a continuous distribution
            # under the log colormap rather than a flat plateau.
            ps_mask = r["per_shot_mask"][fx, y0:y1, z0:z1].astype(np.float32)
            T_tr = r["T_transient"][fx, y0:y1, z0:z1]
            T_norm = np.clip((T_tr - 37.0) / max((T_tr.max() - 37.0), 1.0), 0.0, 1.0)
            per_pulse = (ps_mask * T_norm).astype(np.float32)
        else:
            per_pulse = np.clip(r["pcav"][fx, y0:y1, z0:z1], 0.0, 1.0).astype(np.float32)
        per_pulse_slice[sc.name] = per_pulse

    fig, axes = plt.subplots(1, len(scenarios), figsize=(14.0, 5.4))
    if len(scenarios) == 1:
        axes = [axes]
    base_ct = ct[fx, y0:y1, z0:z1]
    tumour_sl = (label_vol[fx, y0:y1, z0:z1] == HCC.label).astype(float)

    # Heatmap colormap — transparent → inferno, alpha-graduated.
    hot = plt.get_cmap("inferno")
    dose_colors = [(*hot(t)[:3], 0.0 if t < 0.02 else min(0.85, 0.15 + t * 0.95))
                   for t in np.linspace(0, 1, 256)]
    dose_cmap = LinearSegmentedColormap.from_list("cav_dose_anim", dose_colors)

    # Shared log-norm scale across panels: pre-compute the final
    # cumulative event field for each scenario (same algorithm as the
    # animation accumulator, but in one pass) and use the global max
    # as the colorbar upper bound. Log scale spans the 3+ decade
    # dynamic range from "barely-touched rim" to "central overlap"
    # voxels so the distribution is visible.
    final_max = 1.0
    for sc in scenarios:
        per_pulse = per_pulse_slice[sc.name]
        pulses = max(sc.pulses_per_point, 1)
        acc = np.zeros_like(per_pulse)
        anchors_seq = anchors_per_path[sc.name]
        for shot_idx, (x, yi, zi) in enumerate(paths[sc.name]):
            sy = yi - info["focus_idx"][1]
            sz = zi - info["focus_idx"][2]
            sy0 = max(0, -sy); sy1 = min(ny_loc, ny_loc - sy)
            sz0 = max(0, -sz); sz1 = min(nz_loc, nz_loc - sz)
            if sy1 > sy0 and sz1 > sz0:
                k_steer = steering_pressure_factor(
                    (x, yi, zi), anchors_seq[shot_idx], info["dx"], sc.f0)
                acc[sy0+sy:sy1+sy, sz0+sz:sz1+sz] += \
                    per_pulse[sy0:sy1, sz0:sz1] * pulses * k_steer
        final_max = max(final_max, float(acc.max()))
    shared_vmin = 1.0
    shared_vmax = max(final_max, shared_vmin * 10.0)
    shared_norm = LogNorm(vmin=shared_vmin, vmax=shared_vmax)

    artists = []
    for ax, sc in zip(axes, scenarios):
        ax.imshow(base_ct, cmap="gray", vmin=-200, vmax=300,
                  extent=extent_crop, aspect="equal")
        ax.contour(np.flipud(tumour_sl), levels=[0.5], colors="cyan",
                   linewidths=1.0, extent=extent_crop)
        ax.set(xlabel="z [mm]", ylabel="y [mm]")
        # Shared log-norm scale across all panels.
        dose_im = ax.imshow(np.full((ny_loc, nz_loc), shared_vmin*0.5, dtype=np.float32),
                            cmap=dose_cmap, norm=shared_norm,
                            extent=extent_crop, aspect="equal",
                            interpolation="nearest")
        progress = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=8,
                           verticalalignment="top",
                           bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        title = ax.set_title("", fontsize=8.5)
        artists.append({"ax": ax, "dose_im": dose_im,
                        "events_acc": np.zeros((ny_loc, nz_loc), dtype=np.float32),
                        "shots_drawn": 0, "progress": progress, "title": title,
                        "current_marker": None, "sc": sc,
                        "n_shots": len(paths[sc.name])})

    # Single shared colorbar (log scale) for all three panels.
    fig.colorbar(artists[-1]["dose_im"], ax=axes,
                 fraction=0.025, pad=0.02, shrink=0.85,
                 label="cumulative cavitation events (log scale)")

    def init():
        return [a["dose_im"] for a in artists] + [a["progress"] for a in artists]

    def animate(frame_idx: int):
        # Per-panel progress: each scenario animates from 0 → 100% over
        # the full timeline, so all three are visible end-to-end.
        progress_frac = frame_idx / max(n_frames - 1, 1)
        changed = []
        for art in artists:
            sc = art["sc"]
            n_shots_total = art["n_shots"]
            shots_target = int(round(progress_frac * n_shots_total))
            shots_target = min(shots_target, n_shots_total)
            path = paths[sc.name]
            per_pulse = per_pulse_slice[sc.name]
            pulses = max(sc.pulses_per_point, 1)
            # Incrementally add expected-cavitation-event contributions.
            # Each shot deposits per_pulse × pulses_per_spot at every
            # voxel; overlapping shots keep adding so the heatmap shows
            # spot-overlap intensity rather than saturating at 1.
            while art["shots_drawn"] < shots_target:
                idx = art["shots_drawn"]
                x, yi, zi = path[idx]
                shift_y = yi - info["focus_idx"][1]
                shift_z = zi - info["focus_idx"][2]
                src_y0 = max(0, -shift_y); src_y1 = min(ny_loc, ny_loc - shift_y)
                src_z0 = max(0, -shift_z); src_z1 = min(nz_loc, nz_loc - shift_z)
                if src_y1 > src_y0 and src_z1 > src_z0:
                    dst_y0 = src_y0 + shift_y; dst_y1 = src_y1 + shift_y
                    dst_z0 = src_z0 + shift_z; dst_z1 = src_z1 + shift_z
                    anchor = anchors_per_path[sc.name][idx]
                    k_steer = steering_pressure_factor(
                        (x, yi, zi), anchor, info["dx"], sc.f0)
                    art["events_acc"][dst_y0:dst_y1, dst_z0:dst_z1] += \
                        per_pulse[src_y0:src_y1, src_z0:src_z1] * pulses * k_steer
                art["shots_drawn"] += 1

            art["dose_im"].set_data(art["events_acc"])
            changed.append(art["dose_im"])

            # Current-shot cross marker
            if art["current_marker"] is not None:
                art["current_marker"].remove()
                art["current_marker"] = None
            if 0 < shots_target <= n_shots_total:
                idx = shots_target - 1
                x, yi, zi = path[idx]
                if y0 <= yi < y1 and z0 <= zi < z1:
                    y_mm = info["y_axis"][yi] * 1e3
                    z_mm = info["z_axis"][zi] * 1e3
                    art["current_marker"], = art["ax"].plot(
                        z_mm, y_mm, marker="x", color="white",
                        markersize=10, mew=2.0)

            sc_min = sc.treatment_s / 60.0
            elapsed = progress_frac * sc_min
            done = shots_target >= n_shots_total
            eff_prf = sc.prf * sc.interleave_subspots
            art["progress"].set_text(
                f"t = {elapsed:.2f} / {sc_min:.1f} min\n"
                f"shots: {shots_target}/{n_shots_total}\n"
                f"{'COMPLETE' if done else f'{eff_prf:.0f} Hz effective'}"
            )
            art["title"].set_text(
                f"{sc.label}\nFWHM-lat {r['metrics']['fwhm_lat_mm']:.1f} mm × {n_shots_total} pts"
                if False else
                f"{sc.label}\n{n_shots_total} shots, {pulses} pulses/spot"
            )
            changed.append(art["progress"])
            changed.append(art["title"])
        return changed

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, blit=False)
    fig.suptitle("Histotripsy sonication — cumulative cavitation-dose heatmap "
                 "(per-panel timeline; LiTS17 case-0 HCC tumour, native segmentation)",
                 y=0.99, fontsize=10)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.92, wspace=0.18)
    out_gif = os.path.join(OUT_DIR, "anim_sonication.gif")
    writer = PillowWriter(fps=fps)
    anim.save(out_gif, writer=writer, dpi=110)
    plt.close(fig)
    print(f"  saved {out_gif}")
    with open(out_gif, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    print(f"  GIF size: {os.path.getsize(out_gif)/1024:.1f} KB")
    return b64


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def save_fig(fig, name: str) -> str:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=130, bbox_inches="tight")
    return fig_to_base64(fig)


TISSUE_CMAP = ListedColormap([
    "#000000",  # 0 air
    "#ffd1a3",  # 1 skin
    "#fff2c2",  # 2 fat
    "#cc9999",  # 3 muscle
    "#ddddff",  # 4 bone
    "#a0628c",  # 5 liver
    "#3b1a4a",  # 6 hcc (target focus)
    "#7c3a8c",  # 7 hcc_other (untreated foci)
])


def plot_segmentation(ct, label_vol, info) -> tuple:
    nx, ny, nz = label_vol.shape
    fx, fy, fz = info["focus_idx"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    extent_yz = [info["z_axis"][0]*1e3, info["z_axis"][-1]*1e3,
                 info["y_axis"][-1]*1e3, info["y_axis"][0]*1e3]
    extent_xz = [info["z_axis"][0]*1e3, info["z_axis"][-1]*1e3,
                 info["x_axis"][-1]*1e3, info["x_axis"][0]*1e3]
    extent_xy = [info["y_axis"][0]*1e3, info["y_axis"][-1]*1e3,
                 info["x_axis"][-1]*1e3, info["x_axis"][0]*1e3]

    # Top row: raw CT in HU
    ax[0, 0].imshow(ct[fx, :, :], cmap="gray", vmin=-200, vmax=300, extent=extent_yz)
    ax[0, 0].set_title(f"CT axial (x = focus, {info['x_focus']*1e3:.0f} mm depth)")
    ax[0, 0].set(xlabel="z [mm]", ylabel="y [mm]")
    ax[0, 1].imshow(ct[:, fy, :], cmap="gray", vmin=-200, vmax=300, extent=extent_xz)
    ax[0, 1].set_title("CT coronal (y = focus)")
    ax[0, 1].set(xlabel="z [mm]", ylabel="depth x [mm]")
    ax[0, 2].imshow(ct[:, :, fz], cmap="gray", vmin=-200, vmax=300, extent=extent_xy)
    ax[0, 2].set_title("CT sagittal (z = focus)")
    ax[0, 2].set(xlabel="y [mm]", ylabel="depth x [mm]")

    # Bottom row: tissue label segmentation with HCC tumour highlighted
    ax[1, 0].imshow(label_vol[fx, :, :], cmap=TISSUE_CMAP, vmin=0, vmax=7, extent=extent_yz)
    ax[1, 1].imshow(label_vol[:, fy, :], cmap=TISSUE_CMAP, vmin=0, vmax=7, extent=extent_xz)
    ax[1, 2].imshow(label_vol[:, :, fz], cmap=TISSUE_CMAP, vmin=0, vmax=7, extent=extent_xy)
    for a in ax[1, :]:
        a.set_xlabel("[mm]"); a.set_ylabel("[mm]")
        a.scatter([extent_yz[0] + (extent_yz[1] - extent_yz[0]) * 0.5],
                  [extent_yz[2] + (extent_yz[3] - extent_yz[2]) * 0.5],
                  marker="x", c="white", s=0)
    ax[1, 0].set_title("Tissue labels axial (skin/fat/muscle/bone/liver/HCC)")
    ax[1, 1].set_title("Tissue labels coronal")
    ax[1, 2].set_title("Tissue labels sagittal")

    # Add tumour markers
    for a, sl_axes in zip(ax[1, :], [(fy, fz), (fx, fz), (fx, fy)]):
        pass

    handles = [plt.Rectangle((0, 0), 1, 1, color=TISSUE_CMAP(t.label)) for t in TISSUES]
    fig.legend(handles, [t.name for t in TISSUES], loc="lower center",
               ncol=len(TISSUES), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Real abdominal CT (LiTS17 case-0) — HU-threshold tissue "
                 "segmentation overlaid with the native LiTS liver + HCC "
                 "tumour labels", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    b64 = save_fig(fig, "fig13_real_ct_segmentation")
    plt.close(fig)
    print("  saved fig13_real_ct_segmentation")
    return b64


def plot_pressure_and_lesion(ch_ct, label_vol, info, results, lesions) -> tuple:
    nx, ny, nz = label_vol.shape
    fx, fy_idx, fz_idx = info["focus_idx"]

    # Crop window around focus for the pressure / Pcav panels (±60 mm
    # in y and z so the focal spot is resolvable).
    win_mm = 60.0
    win_vox_y = int(round(win_mm * 1e-3 / info["dx"]))
    win_vox_z = int(round(win_mm * 1e-3 / info["dx"]))
    y0, y1 = max(0, fy_idx - win_vox_y), min(ny, fy_idx + win_vox_y)
    z0, z1 = max(0, fz_idx - win_vox_z), min(nz, fz_idx + win_vox_z)
    extent_crop = [info["z_axis"][z0]*1e3, info["z_axis"][z1-1]*1e3,
                   info["y_axis"][y1-1]*1e3, info["y_axis"][y0]*1e3]
    extent_full = [info["z_axis"][0]*1e3, info["z_axis"][-1]*1e3,
                   info["y_axis"][-1]*1e3, info["y_axis"][0]*1e3]

    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    # Cavitation-dose heatmap: transparent → red through hot colormap.
    hot = plt.get_cmap("inferno")
    dose_colors = [(*hot(t)[:3], 0.0 if t < 0.05 else min(0.85, t * 1.2))
                   for t in np.linspace(0, 1, 256)]
    overlay_cmap = LinearSegmentedColormap.from_list("cav_dose", dose_colors)
    pcav_cmap = "viridis"

    for col, sc in enumerate(SCENARIOS):
        r = results[sc.name]
        lesion = lesions[sc.name]

        # Row 0: PNP focal field cropped around focus
        sl = r["p_field"][fx, y0:y1, z0:z1] / 1e6
        im0 = axes[0, col].imshow(sl, cmap="magma", extent=extent_crop,
                                  vmin=0, vmax=max(sl.max(), 1e-6), aspect="equal")
        axes[0, col].set_title(f"{sc.label}\nPNP [MPa] (focus-centred ±{win_mm:.0f} mm)", fontsize=9)
        axes[0, col].set(xlabel="z [mm]", ylabel="y [mm]")
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

        # Row 1: cavitation probability or collapse strength (cropped)
        sl_p = r["pcav"][fx, y0:y1, z0:z1]
        im1 = axes[1, col].imshow(sl_p, cmap=pcav_cmap, extent=extent_crop,
                                  vmin=0, vmax=1, aspect="equal")
        axes[1, col].set_title("single-pulse $P_{cav}$ (focus-centred)", fontsize=9)
        axes[1, col].set(xlabel="z [mm]", ylabel="y [mm]")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

        # Row 2: cavitation-dose heatmap on CT axial slice
        axes[2, col].imshow(ch_ct[fx, :, :], cmap="gray", vmin=-200, vmax=300,
                            extent=extent_full, aspect="equal")
        tumour_sl = (label_vol[fx, :, :] == HCC.label).astype(float)
        axes[2, col].contour(np.flipud(tumour_sl), levels=[0.5], colors="cyan",
                             linewidths=0.8, extent=extent_full)
        dose_sl = r["cav_dose"][fx, :, :]
        im2 = axes[2, col].imshow(dose_sl, cmap=overlay_cmap, extent=extent_full,
                                  vmin=0, vmax=1, aspect="equal")
        plt.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04,
                     label="cumulative cavitation dose")
        m = r["metrics"]
        axes[2, col].set_title(
            f"cavitation-dose heatmap (cyan = HCC outline)\n"
            f"coverage {m.get('tumour_coverage_pct', 0):.0f}%, "
            f"FWHM {m['fwhm_lat_mm']:.1f}×{m['fwhm_ax_mm']:.1f} mm (lat×ax), "
            f"$T_{{ss}}$={m.get('T_steady_C', 0):.0f} °C",
            fontsize=8.5)
        axes[2, col].set(xlabel="z [mm]", ylabel="y [mm]")

    fig.suptitle("Histotripsy on real abdominal CT — focal pressure, $P_{cav}$, "
                 "and cumulative cavitation-dose heatmap (LiTS17 HCC tumour)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    b64 = save_fig(fig, "fig14_real_ct_lesion_panel")
    plt.close(fig)
    print("  saved fig14_real_ct_lesion_panel")
    return b64


def plot_raster_overlay(ct, label_vol, info, results, scenarios):
    """Render the explicit raster point cloud and per-shot footprint
    envelope for each scenario, overlaid on the CT axial slice through
    the tumour centre."""
    fx = info["focus_idx"][0]
    extent_full = [info["z_axis"][0]*1e3, info["z_axis"][-1]*1e3,
                   info["x_axis"][-1]*1e3, info["x_axis"][0]*1e3]

    # Crop window around the tumour for legibility
    tumour = label_vol == HCC.label
    coords = np.argwhere(tumour)
    cmin = coords.min(axis=0); cmax = coords.max(axis=0)
    pad_y = int(0.04 / info["dx"]); pad_z = int(0.04 / info["dx"])
    y0 = max(0, cmin[1] - pad_y); y1 = min(label_vol.shape[1], cmax[1] + pad_y)
    z0 = max(0, cmin[2] - pad_z); z1 = min(label_vol.shape[2], cmax[2] + pad_z)
    extent_crop = [info["z_axis"][z0]*1e3, info["z_axis"][z1-1]*1e3,
                   info["y_axis"][y1-1]*1e3, info["y_axis"][y0]*1e3]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(13, 5.0))
    if len(scenarios) == 1:
        axes = [axes]
    for ax, sc in zip(axes, scenarios):
        r = results[sc.name]
        ax.imshow(ct[fx, y0:y1, z0:z1], cmap="gray", vmin=-200, vmax=300,
                  extent=extent_crop, aspect="equal")
        # HCC outline
        tumour_sl = (label_vol[fx, y0:y1, z0:z1] == HCC.label).astype(float)
        ax.contour(np.flipud(tumour_sl), levels=[0.5], colors="cyan",
                   linewidths=1.0, extent=extent_crop)
        # FWHM lateral half-width — the clinical "focal spot" radius
        # quoted in the literature (Penttinen 1976). Use this for the
        # visual circle so the spacing visually matches the -3 dB
        # contour, not the elongated cavitation cigar.
        psr = r["metrics"]["fwhm_lat_mm"] / 2.0
        tumour_full = label_vol == HCC.label
        x_coords = np.argwhere(tumour_full)[:, 0]
        x_min, x_max = x_coords.min(), x_coords.max() + 1
        rg_proj = r["raster_grid"][x_min:x_max, y0:y1, z0:z1].any(axis=0)
        ys, zs = np.where(rg_proj)
        for yy, zz in zip(ys, zs):
            y_mm = info["y_axis"][y0 + yy] * 1e3
            z_mm = info["z_axis"][z0 + zz] * 1e3
            ax.add_patch(plt.Circle((z_mm, y_mm), psr, edgecolor=sc.color,
                                    facecolor=sc.color, alpha=0.18, linewidth=0.6))
        ax.scatter(
            [info["z_axis"][z0 + zz] * 1e3 for zz in zs],
            [info["y_axis"][y0 + yy] * 1e3 for yy in ys],
            c=sc.color, s=8, edgecolor="black", linewidths=0.3,
            label=f"{len(ys)} raster pts (yz projection)",
        )
        ax.set_title(f"{sc.label}\n"
                     f"raster: {r['metrics'].get('actual_raster_points', sc.raster_points)} pts × "
                     f"{r['metrics']['fwhm_lat_mm']:.1f} mm FWHM-lat, "
                     f"{sc.treatment_s/60:.1f} min", fontsize=8.5)
        ax.set(xlabel="z [mm]", ylabel="y [mm]")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Raster scan overlay on real CT — cyan = native HCC tumour outline, "
                 "coloured circles = FWHM focal spot at each raster centre",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    b64 = save_fig(fig, "fig16_raster_overlay")
    plt.close(fig)
    print("  saved fig16_raster_overlay")
    return b64


def plot_metrics_summary(metrics_list, scenarios) -> tuple:
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))
    names = [s.name for s in scenarios]
    colors = [s.color for s in scenarios]
    x = np.arange(len(scenarios))
    w = 0.35

    per_shot_vol = [m["per_shot_volume_mm3"] for m in metrics_list]
    cov = [m["tumour_coverage_pct"] for m in metrics_list]
    vol = [m["lesion_volume_mm3"] / 1e3 for m in metrics_list]
    T_t = [m["T_transient_C"] for m in metrics_list]
    T_s = [m["T_steady_C"] for m in metrics_list]

    # NEW panel 0: per-shot footprint (lesion volume from a SINGLE focal
    # exposure). This is the per-pulse lesion size and is the correct
    # comparison for the per-shot mechanism — μs is a sub-mm³ footprint,
    # ms shock-vapor and ms sub-threshold are 100× larger because a
    # single ms pulse produces a cm-scale cavitation cloud.
    bars0 = ax[0].bar(x, per_shot_vol, color=colors)
    ax[0].set(title="Per-shot lesion footprint\n(single focal exposure)", ylabel="mm³")
    ax[0].set_yscale("log")
    for xi, v, m in zip(x, per_shot_vol, metrics_list):
        ax[0].text(xi, v * 1.15, f"r ≈ {m['per_shot_radius_mm']:.1f} mm",
                   ha="center", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=8)

    ax[1].bar(x, cov, color=colors)
    ax[1].axhline(95.0, color="k", ls="--", lw=0.8, label="95% target")
    ax[1].set(title="Tumour ablation coverage\n(after raster scan)", ylabel="% of tumour vol", ylim=(0, 105))
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=8); ax[1].legend(fontsize=8)

    ax[2].bar(x, vol, color=colors)
    tumour_vol_cm3 = metrics_list[0]["tumour_volume_mm3"] / 1e3
    ax[2].axhline(tumour_vol_cm3, color="grey", ls="--", lw=0.8,
                  label=f"tumour vol {tumour_vol_cm3:.1f} cm³")
    ax[2].set(title="Total raster-summed\nlesion volume", ylabel="cm³")
    ax[2].set_xticks(x); ax[2].set_xticklabels(names, fontsize=8); ax[2].legend(fontsize=8)

    ax[3].bar(x - w/2, T_t, w, color=colors, alpha=0.85, label="transient focal voxel")
    ax[3].bar(x + w/2, T_s, w, color=colors, alpha=0.45, hatch="//", label="cycle-avg bulk")
    ax[3].axhline(43.0, color="k", ls="--", lw=0.8); ax[3].axhline(100.0, color="r", ls="--", lw=0.8)
    ax[3].set(title="Focal temperature\n(transient peak vs cycle-avg steady)", ylabel="°C")
    ax[3].set_xticks(x); ax[3].set_xticklabels(names, fontsize=8); ax[3].legend(fontsize=7)

    fig.suptitle("Real-CT histotripsy treatment metrics — LiTS17 case-0, "
                 "native HCC tumour segmentation as histotripsy target",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    b64 = save_fig(fig, "fig15_real_ct_metrics")
    plt.close(fig)
    print("  saved fig15_real_ct_metrics")
    return b64


def make_strategy_comparison_animation(ct, label_vol, info, results, scenario,
                                       n_frames: int = 80, fps: int = 12) -> str:
    """4-panel animation comparing raster-path strategies on the SAME
    scenario: serpentine, outside-in, inside-out, adaptive low-dose.
    Same total raster points and same pulses_per_spot in every panel —
    only the ORDER differs, so the side-by-side fill rate shows which
    strategy delivers the most uniform intermediate dose.
    """
    print("[ch21e] Building strategy-comparison animation")
    from matplotlib.animation import PillowWriter, FuncAnimation
    from matplotlib.colors import LogNorm

    sc = scenario
    r = results[sc.name]
    fx = info["focus_idx"][0]
    tumour = label_vol == HCC.label
    coords = np.argwhere(tumour)
    cmin = coords.min(axis=0); cmax = coords.max(axis=0)
    pad = int(0.04 / info["dx"])
    y0 = max(0, cmin[1] - pad); y1 = min(label_vol.shape[1], cmax[1] + pad)
    z0 = max(0, cmin[2] - pad); z1 = min(label_vol.shape[2], cmax[2] + pad)
    extent_crop = [info["z_axis"][z0]*1e3, info["z_axis"][z1-1]*1e3,
                   info["y_axis"][y1-1]*1e3, info["y_axis"][y0]*1e3]
    ny_loc = y1 - y0
    nz_loc = z1 - z0

    # Per-pulse field on the focal slice (same construction as the
    # main animation). Same field for all four strategies — only the
    # path order differs.
    if sc.regime == "subthreshold_cav":
        coll_sl = collapse_strength(r["p_field"][fx, y0:y1, z0:z1], sc.f0)
        per_pulse_2d = np.clip((coll_sl - 1.0) / 9.0, 0.0, 1.0).astype(np.float32)
    elif sc.regime == "shock_vapor":
        ps_mask = r["per_shot_mask"][fx, y0:y1, z0:z1].astype(np.float32)
        T_tr = r["T_transient"][fx, y0:y1, z0:z1]
        T_norm = np.clip((T_tr - 37.0) / max((T_tr.max() - 37.0), 1.0), 0.0, 1.0)
        per_pulse_2d = (ps_mask * T_norm).astype(np.float32)
    else:
        per_pulse_2d = np.clip(r["pcav"][fx, y0:y1, z0:z1], 0.0, 1.0).astype(np.float32)

    # Per-pulse 3D field for the adaptive strategy (needs full volume
    # to track dose evolution). Reconstruct from cached fields.
    if sc.regime == "subthreshold_cav":
        coll_3d = collapse_strength(r["p_field"], sc.f0)
        per_pulse_3d = np.clip((coll_3d - 1.0) / 9.0, 0.0, 1.0).astype(np.float32)
    elif sc.regime == "shock_vapor":
        ps3 = r["per_shot_mask"].astype(np.float32)
        T_tr = r["T_transient"]
        T_norm = np.clip((T_tr - 37.0) / max((T_tr.max() - 37.0), 1.0), 0.0, 1.0)
        per_pulse_3d = (ps3 * T_norm).astype(np.float32)
    else:
        per_pulse_3d = np.clip(r["pcav"], 0.0, 1.0).astype(np.float32)

    raster_grid = r["raster_grid"]
    strategies = [
        ("serpentine", serpentine_order(raster_grid)),
        ("outside-in", outside_in_order(raster_grid)),
        ("inside-out", inside_out_order(raster_grid)),
        ("adaptive low-dose",
         adaptive_lowdose_order(raster_grid, per_pulse_3d, info["focus_idx"],
                                max(sc.pulses_per_point, 1))),
    ]
    # Pre-compute per-shot mechanical anchors for each strategy so the
    # steering compensation uses the correct (re-anchored) origin.
    strat_anchors = {}
    strat_n_reanchors = {}
    for name, path in strategies:
        anchors_seq, n_changes = mechanical_walk(
            path, info["focus_idx"], info["dx"], sc.f0)
        strat_anchors[name] = anchors_seq
        strat_n_reanchors[name] = n_changes

    # Pre-compute final cumulative dose for shared log scale
    pulses = max(sc.pulses_per_point, 1)
    final_max = 1.0
    for name, path in strategies:
        acc = np.zeros_like(per_pulse_2d)
        anchors_seq = strat_anchors[name]
        for shot_idx, (x, yi, zi) in enumerate(path):
            sy = yi - info["focus_idx"][1]
            sz = zi - info["focus_idx"][2]
            sy0 = max(0, -sy); sy1 = min(ny_loc, ny_loc - sy)
            sz0 = max(0, -sz); sz1 = min(nz_loc, nz_loc - sz)
            if sy1 > sy0 and sz1 > sz0:
                k_steer = steering_pressure_factor(
                    (x, yi, zi), anchors_seq[shot_idx], info["dx"], sc.f0)
                acc[sy0+sy:sy1+sy, sz0+sz:sz1+sz] += \
                    per_pulse_2d[sy0:sy1, sz0:sz1] * pulses * k_steer
        final_max = max(final_max, float(acc.max()))
    shared_norm = LogNorm(vmin=1.0, vmax=max(final_max, 10.0))

    # Heatmap colormap
    hot = plt.get_cmap("inferno")
    dose_colors = [(*hot(t)[:3], 0.0 if t < 0.02 else min(0.85, 0.15 + t * 0.95))
                   for t in np.linspace(0, 1, 256)]
    dose_cmap = LinearSegmentedColormap.from_list("cav_dose_strat", dose_colors)

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.4))
    base_ct = ct[fx, y0:y1, z0:z1]
    tumour_sl = (label_vol[fx, y0:y1, z0:z1] == HCC.label).astype(float)

    artists = []
    for ax, (name, path) in zip(axes, strategies):
        ax.imshow(base_ct, cmap="gray", vmin=-200, vmax=300,
                  extent=extent_crop, aspect="equal")
        ax.contour(np.flipud(tumour_sl), levels=[0.5], colors="cyan",
                   linewidths=1.0, extent=extent_crop)
        dose_im = ax.imshow(np.full((ny_loc, nz_loc), 0.5, dtype=np.float32),
                            cmap=dose_cmap, norm=shared_norm,
                            extent=extent_crop, aspect="equal",
                            interpolation="nearest")
        title = ax.set_title(f"{name}\n{len(path)} shots, "
                             f"{strat_n_reanchors[name]} mech re-anchors",
                             fontsize=9)
        progress = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=8,
                           verticalalignment="top",
                           bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        ax.set(xlabel="z [mm]", ylabel="y [mm]")
        artists.append({"ax": ax, "dose_im": dose_im,
                        "events_acc": np.zeros((ny_loc, nz_loc), dtype=np.float32),
                        "shots_drawn": 0, "progress": progress, "title": title,
                        "current_marker": None, "name": name, "path": path})

    fig.colorbar(artists[-1]["dose_im"], ax=axes,
                 fraction=0.025, pad=0.02, shrink=0.85,
                 label="cumulative cavitation events (log scale)")

    def init():
        return [a["dose_im"] for a in artists]

    def animate(frame_idx: int):
        progress_frac = frame_idx / max(n_frames - 1, 1)
        changed = []
        for art in artists:
            n_total = len(art["path"])
            shots_target = min(int(round(progress_frac * n_total)), n_total)
            while art["shots_drawn"] < shots_target:
                idx = art["shots_drawn"]
                _, yi, zi = art["path"][idx]
                shift_y = yi - info["focus_idx"][1]
                shift_z = zi - info["focus_idx"][2]
                src_y0 = max(0, -shift_y); src_y1 = min(ny_loc, ny_loc - shift_y)
                src_z0 = max(0, -shift_z); src_z1 = min(nz_loc, nz_loc - shift_z)
                if src_y1 > src_y0 and src_z1 > src_z0:
                    anchor = strat_anchors[art["name"]][idx]
                    k_steer = steering_pressure_factor(
                        art["path"][idx], anchor, info["dx"], sc.f0)
                    art["events_acc"][src_y0+shift_y:src_y1+shift_y,
                                      src_z0+shift_z:src_z1+shift_z] += \
                        per_pulse_2d[src_y0:src_y1, src_z0:src_z1] * pulses * k_steer
                art["shots_drawn"] += 1
            art["dose_im"].set_data(art["events_acc"])
            changed.append(art["dose_im"])
            if art["current_marker"] is not None:
                art["current_marker"].remove()
                art["current_marker"] = None
            if 0 < shots_target <= n_total:
                _, yi, zi = art["path"][shots_target - 1]
                if y0 <= yi < y1 and z0 <= zi < z1:
                    y_mm = info["y_axis"][yi] * 1e3
                    z_mm = info["z_axis"][zi] * 1e3
                    art["current_marker"], = art["ax"].plot(
                        z_mm, y_mm, marker="x", color="white",
                        markersize=10, mew=2.0)
            art["progress"].set_text(f"{shots_target}/{n_total} shots\n"
                                     f"{100*shots_target/n_total:.0f}%")
            changed.append(art["progress"])
        return changed

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, blit=False)
    fig.suptitle(f"Raster path strategies on the same {sc.label} scenario — "
                 f"same shots, different ORDER (LiTS17 HCC target)",
                 y=0.99, fontsize=10)
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.04, right=0.93, wspace=0.18)
    out_gif = os.path.join(OUT_DIR, "anim_strategy_comparison.gif")
    writer = PillowWriter(fps=fps)
    anim.save(out_gif, writer=writer, dpi=110)
    plt.close(fig)
    print(f"  saved {out_gif}")
    with open(out_gif, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    print(f"  GIF size: {os.path.getsize(out_gif)/1024:.1f} KB")
    return b64


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────


def main():
    if not os.path.exists(CT_PATH):
        raise SystemExit(f"CT not found: {CT_PATH}. Download KiTS19 case_00000 first.")

    ct, label_vol, info = load_ct_and_segment(target_dx_m=1.2e-3)

    tumour_volume_m3 = float((label_vol == HCC.label).sum() * info["dx"]**3)

    # Run scenarios with TWO passes:
    #   (a) probe pass — single focal exposure to measure per-shot footprint
    #   (b) auto-sized pass — raster pitch matched to per-shot diameter
    #       so coverage is full; treatment time is the minimum required.
    results = {}
    lesions = {}
    metrics = []
    sized_scenarios = []
    for sc_template in SCENARIOS:
        print(f"[ch21e] Probing per-shot footprint: {sc_template.label}")
        # Probe with a single raster point so the per_shot footprint is
        # measured cleanly (no superposition).
        sc_probe = Scenario(
            name=sc_template.name, label=sc_template.label, regime=sc_template.regime,
            f0=sc_template.f0, pnp=sc_template.pnp, ppp=sc_template.ppp,
            pulse_on_s=sc_template.pulse_on_s, prf=sc_template.prf,
            treatment_s=10.0, raster_points=1,
            shock_alpha_gain=sc_template.shock_alpha_gain, color=sc_template.color,
            interleave_subspots=sc_template.interleave_subspots,
            pulses_per_point=sc_template.pulses_per_point,
        )
        props = property_maps(label_vol, sc_probe.f0)
        p_unatt = focused_bowl_pressure(info, props, sc_probe.f0, source_pa=1.0)
        attn_at_focus = p_unatt[info["focus_idx"]]
        source_pa = sc_probe.pnp / max(attn_at_focus, 1e-12)
        p_field = focused_bowl_pressure(info, props, sc_probe.f0, source_pa=source_pa)
        _, r_probe = predicted_lesion(p_field, props, sc_probe, info, label_vol)
        ps_mask = r_probe["per_shot_mask"]
        ps_coords = np.argwhere(ps_mask)
        if len(ps_coords) > 0:
            half_ext_vox = (ps_coords.max(0) - ps_coords.min(0)) / 2.0
            ps_half_m = tuple(float(v * info["dx"]) for v in half_ext_vox)
        else:
            ps_half_m = (info["dx"], info["dx"], info["dx"])
        print(f"  per-shot half-extents (mm) = "
              f"x:{ps_half_m[0]*1e3:.1f}, y:{ps_half_m[1]*1e3:.1f}, z:{ps_half_m[2]*1e3:.1f}; "
              f"r_eq={r_probe['metrics']['per_shot_radius_mm']:.2f}")

        sc = autosize_raster(sc_template, ps_half_m, tumour_volume_m3)
        sized_scenarios.append(sc)
        print(f"[ch21e] Sized scenario (autosize estimate): {sc.raster_points} pts, "
              f"{sc.treatment_s/60:.1f} min")
        # Re-run with the autosized raster; reuse the p_field (same scenario).
        lesion, r = predicted_lesion(p_field, props, sc, info, label_vol)
        r["p_field"] = p_field
        # Sync sc to the ACTUAL placed raster + actual treatment time so
        # downstream visualizations (animation, raster overlay, metrics
        # summary) report the real numbers, not the autosize estimate.
        actual_n = r["metrics"]["actual_raster_points"]
        actual_t = r["metrics"]["actual_treatment_s"]
        sc = Scenario(
            name=sc.name, label=sc.label, regime=sc.regime,
            f0=sc.f0, pnp=sc.pnp, ppp=sc.ppp,
            pulse_on_s=sc.pulse_on_s, prf=sc.prf,
            treatment_s=actual_t, raster_points=actual_n,
            shock_alpha_gain=sc.shock_alpha_gain, color=sc.color,
            interleave_subspots=sc.interleave_subspots,
            pulses_per_point=sc.pulses_per_point,
        )
        sized_scenarios[-1] = sc
        results[sc.name] = r
        lesions[sc.name] = lesion
        metrics.append(r["metrics"])
        m = r["metrics"]
        print(f"    actual: {m['actual_raster_points']} pts, "
              f"{m['actual_treatment_s']/60:.1f} min, "
              f"coverage {m['tumour_coverage_pct']:.1f}%, "
              f"lesion {m['lesion_volume_mm3']/1e3:.2f} cm³, "
              f"spillover {m['spillover_volume_mm3']/1e3:.2f} cm³, "
              f"confinement {m['confinement_pct']:.0f}%, "
              f"T_trans {m['T_transient_C']:.1f} °C, "
              f"T_steady {m['T_steady_C']:.1f} °C")

    # Render figures + capture base64 for embedding
    b64_seg = plot_segmentation(ct, label_vol, info)
    b64_pan = plot_pressure_and_lesion(ct, label_vol, info, results, lesions)
    b64_met = plot_metrics_summary(metrics, sized_scenarios)
    b64_ras = plot_raster_overlay(ct, label_vol, info, results, sized_scenarios)
    b64_anim = make_sonication_animation(ct, label_vol, info, results, sized_scenarios)
    # Strategy-comparison animation: pick the shock-vapor scenario
    # because its larger per-shot footprint gives the clearest
    # side-by-side fill differences between path orderings.
    sc_for_strategy = next((s for s in sized_scenarios if s.regime == "shock_vapor"),
                           sized_scenarios[0])
    b64_strat = make_strategy_comparison_animation(ct, label_vol, info, results,
                                                   sc_for_strategy)

    # Write embedded-figures markdown.
    md_path = os.path.join(OUT_DIR, "embedded_figures.md")
    print(f"[ch21e] Writing embedded markdown: {md_path}")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("<!-- AUTO-GENERATED by ch21e_real_liver_ct_histotripsy.py — do not edit -->\n\n")
        f.write("## 21.14 Real-CT Histotripsy Treatment — Hepatocellular Carcinoma "
                "(LiTS17 case-0)\n\n")
        f.write(
            "The figures below are produced from a real public-domain abdominal "
            "CT volume (LiTS17 case-0, "
            "[Bilic 2023](https://doi.org/10.1016/j.media.2022.102680), "
            "CC-BY 4.0). The CT is paired with the dataset's voxel-level "
            "segmentation labels (liver = 1, HCC tumour = 2), so the histotripsy "
            "target is the patient's actual hepatocellular carcinoma — not a "
            "synthetic sphere. Surrounding tissues (skin / fat / muscle / bone) "
            "are classified by HU thresholding. Acoustic and thermal properties "
            "are assigned per the IT'IS Foundation v4.1 table. Each figure is "
            "embedded directly as base64 PNG bytes — no external links are "
            "needed.\n\n"
        )
        f.write(
            "**Focal-field source:** real PSTD-derived per-shot kernels "
            "(`data/kernels/*.npz`) generated by "
            "[`pykwavers/examples/book/cavitation_kernel.py`](../../../pykwavers/examples/book/cavitation_kernel.py)"
            " from a single focused-bowl pulse on a homogeneous water-equivalent "
            "medium, then resampled (cubic spline) onto the planner grid. The "
            "kernel encodes real diffraction, sidelobes, and depth-of-focus that "
            "the closed-form Penttinen 1976 separable-Gaussian approximation "
            "missed. Source amplitude is auto-calibrated by a low-amplitude probe "
            "pass so the realised peak rarefactional pressure matches the target "
            "to within ~7 %. Compared to the previous Gaussian-envelope baseline, "
            "per-shot mech footprints widen by ~2× along the axial axis for the "
            "ms regimes (Gaussian decays faster than the real focal cigar near "
            "the field's tail), with corresponding shifts in coverage / spillover "
            "/ confinement. These are honest physics, not algorithmic regressions. "
            "Set `USE_PSTD_KERNEL = False` in "
            "`ch21e_real_liver_ct_histotripsy.py` to revert to the closed form "
            "for A/B comparison.\n\n"
        )
        f.write(
            "**Helmholtz-residual validation (Phase C-1).** The cached PSTD "
            "envelopes are also the supervised ground truth for the "
            "parameterised field-surrogate PINN currently scaffolded in "
            "`kwavers::solver::inverse::pinn::ml::field_surrogate`. As a "
            "physics-residual sanity check, the Helmholtz operator "
            "$R = \\nabla^2 p + k^2 p$ (with $k = 2\\pi f_0 / c_0$, "
            "$c_0 = 1500$ m/s) is computed on each cached envelope and "
            "summarised below. RMS residuals normalised by $k^2 |p|_\\infty$ "
            "are ~4–5 % across all four kernels — small enough to validate the "
            "residual formulation as a Phase-C-2 training-loss term, with the "
            "expected bias localised to the focal region (where the envelope "
            "approximation breaks down because of standing-wave structure) "
            "and the source / PML boundary. See "
            "[`pykwavers/examples/book/helmholtz_residual_figures.py`]"
            "(../../../pykwavers/examples/book/helmholtz_residual_figures.py) "
            "for the analysis script and "
            "[`docs/book/figures/ch21e/helmholtz_summary.txt`]"
            "(./helmholtz_summary.txt) for the per-kernel statistics table.\n\n"
            "![Helmholtz summary](./helmholtz_summary.png)\n\n"
            "![Helmholtz, 1.0 MHz / 30 MPa](./helmholtz_kernel_1.00MHz_30MPa.png)\n\n"
            "![Helmholtz, 0.5 MHz / 30 MPa](./helmholtz_kernel_0.50MHz_30MPa.png)\n\n"
        )
        f.write(
            "**Phase C-3 — Adam-optimised training on a `KernelCube` "
            "dataset.** The C-2 plain-SGD optimiser has been replaced "
            "with Burn's built-in `Adam` (β₁=0.9, β₂=0.999, ε=1e-5); "
            "training batches now come from a `KernelCubeSampler` that "
            "draws random voxels from a stack of `FocalKernel`s (see "
            "[`data.rs`](../../../kwavers/src/solver/inverse/pinn/ml/field_surrogate/data.rs)), "
            "applying the same `[-1, 1]` input/output normalisation the "
            "network expects. Combined loss is unchanged: data MSE on "
            "`(p_min, p_max, p_rms)` voxels plus the dimensionless "
            "Helmholtz residual "
            "$\\hat R = (\\sum p̂(±ε̂) - 6 p̂) / (k\\,\\varepsilon)^2 + p̂$.\n\n"
            "The figures below come from a 2000-step training run on a "
            "Penttinen-Gaussian kernel cube (4 corners at "
            "$\\{0.5, 1.0\\}$ MHz × $\\{15, 30\\}$ MPa with realistic "
            "per-frequency focal-spot scaling). Data MSE drops ~20× "
            "(0.0075 → 0.00037), 4–5× faster convergence than the C-2 "
            "plain-SGD baseline. The Helmholtz residual stays bounded "
            "at ~5×10⁻⁶, showing the network co-satisfies the wave "
            "equation across the sweep. **Phase C-4 — importance "
            "sampling**: each training voxel is now drawn with "
            "probability $\\propto |p|^\\alpha + \\varepsilon$ "
            "($\\alpha = 1$, $\\varepsilon = 10^{-4}$) via a "
            "precomputed CDF + binary-search lookup in "
            "`KernelCubeSampler::set_sampling(SamplingMode::"
            "ImportanceByMagnitude)` — see "
            "[`data.rs`](../../../kwavers/src/solver/inverse/pinn/ml/field_surrogate/data.rs). "
            "This rebalances the training signal so the focal region "
            "is sampled as often as the far rim despite its tiny "
            "voxel count.\n\n"
            "**Phase C-5 — cosine-annealing LR + 3× longer training.** "
            "Combined with a cosine-annealed Adam learning rate "
            "(`2 × 10⁻³ → 1 × 10⁻⁵` over the full 3000-step window, "
            "wired via Burn's `CosineAnnealingLrScheduler` in "
            "[`training.rs`](../../../kwavers/src/solver/inverse/pinn/ml/field_surrogate/training.rs)), "
            "the network does **high-LR exploration** in the first "
            "~30 % of training and **low-LR fine-tuning** in the "
            "remaining 70 %.\n\n"
            "**Phase C-6 — wider network + sharper importance.** "
            "Doubling capacity to 256-wide × 4-layer (~330 k params) "
            "and tightening the importance exponent to α=2 (sampling "
            "probability ∝ |p|² + ε) pushes the focal-peak prediction "
            "above 75 % of target with the held-out f0 midpoint "
            "(0.75 MHz) reaching **81 %**, slightly outperforming the "
            "training corners — evidence the network has genuinely "
            "learned the wavelength-scaling parametrisation rather "
            "than memorising corner kernels.\n\n"
            "**Phase C-7 — Dynamic Tanh (DyT) activations.** Replaced "
            "the fixed `x.tanh()` activations with Zhu 2025's Dynamic "
            "Tanh: `DyT(x) = γ · tanh(α · x) + β` with per-layer "
            "learnable `(α, γ, β)` scalars (see "
            "[`dynamic_tanh.rs`](../../../kwavers/src/solver/inverse/pinn/ml/field_surrogate/dynamic_tanh.rs)). "
            "The hypothesis: a learnable `α` lets each layer adjust "
            "tanh saturation — `α<1` keeps activations in the linear "
            "region (preserves peak amplitude), `α>1` saturates "
            "earlier (smoother rim gradients). Initialised at `α=1` "
            "(recovers vanilla tanh), the optimiser converged the "
            "input-side α to ~1.10 and the output-side α to ~1.00 — "
            "essentially no movement, indicating vanilla tanh's "
            "saturation is already near-optimal for this regression. "
            "Peak prediction 0.71–0.80 (avg ~0.76), RMSE ~0.14: "
            "comparable to C-6 within measurement variance. **Honest "
            "negative result**: DyT works as designed (autodiff "
            "propagates through `(α, γ, β)`, scalars are learnable, "
            "training stable) but doesn't materially help here "
            "because the bottleneck is the volume-weighted training "
            "distribution, not activation saturation. DyT remains in "
            "the codebase for future scenarios where saturation "
            "*does* dominate (transformer-like wide architectures, "
            "or after the log-pressure target transform).\n\n"
            "Cumulative phase progression on the Penttinen sweep:\n\n"
            "| Phase | Focal peak (1.0 MHz) | Per-f0 RMSE (avg) | "
            "Data-loss drop |\n"
            "|---|---|---|---|\n"
            "| C-3 (uniform / constant LR / 1000 steps) | 0.013 | 0.42 | 10× |\n"
            "| C-4 (importance α=1 / constant LR) | 0.39 | 0.22 | 10× |\n"
            "| C-5 (importance α=1 / cosine LR / 3000 steps) | 0.62 | 0.16 | 170× |\n"
            "| C-6 (256×4 net / importance α=2 / cosine LR) | 0.75 | 0.13 | 220× |\n"
            "| C-7 (C-6 + Dynamic Tanh activations) | 0.71 | 0.14 | similar |\n"
            "| C-8a (signed-log1p target, eps_ratio=1e-3) | 0.32 | 0.32 | similar |\n"
            "| C-8b (signed-log1p target, eps_ratio=0.1) | 0.61 | 0.18 | similar |\n"
            "| C-8c (transform-decoupled importance CDF, linear target) | 0.76 | 0.12 | 420× |\n"
            "| C-9 (C-8c + peak-prominence loss, w=1.0) | 0.67 | 0.17 | ambiguous batch-max |\n"
            "| C-9 (C-8c + peak-prominence loss, w=0.1) | 0.75 | 0.15 | per-corner fragmentation |\n"
            "| C-10 (per-kernel-scoped prominence, w=1.0) | 0.72 | 0.16 | balanced; corners co-trained |\n"
            "| C-10 (per-kernel-scoped prominence, w=0.1) | 0.77 | 0.15 | within C-8c init variance |\n\n"
            "**Phase C-8 — signed-log1p target transform (mixed "
            "result; one structural improvement banked).** "
            "Implemented `TargetTransform::SignedLog1p` "
            "(`T(p) = sign(p) · log1p(|p|/p_ε) / T_max`) and its inverse "
            "`p = sign(t) · p_ε · expm1(|t| · T_max)` as a per-channel "
            "bijection between Pa and `[-1, 1]`. The hypothesis was "
            "that compressing the pressure dynamic range would "
            "rebalance MSE gradient mass between the focal peak and "
            "the sub-ε rim. Empirically, the transform *hurts* peak "
            "prediction at every compression level: with `eps_ratio="
            "1e-3` (`T_max ≈ 6.9`, three-decade compression) peak "
            "drops to 30–32 %; with `eps_ratio=0.1` (`T_max ≈ 2.4`, "
            "one-decade compression) peak recovers to 60–63 % but "
            "still under-performs the C-7 linear path. Root cause: "
            "the inverse map `expm1(|t|·T_max)` has exponentially "
            "large slope near `|t|=1`; `tanh` saturation prevents the "
            "network from reaching `|t|=1` exactly, and the inferred "
            "Pa is exponentially sensitive to the residual gap. The "
            "transform module (`target_transform.rs`) stays in the "
            "crate — it is the right tool for high-dynamic-range "
            "regressions where mild compression genuinely helps — and "
            "the audit it forced did surface one durable improvement: "
            "the importance-sampling CDF now uses the *pre-transform* "
            "magnitude `|p|/p_max` rather than the post-transform "
            "target, so the focal-peak concentration is preserved "
            "regardless of which transform the sampler is configured "
            "with. With the linear transform plus the decoupled CDF "
            "(row C-8c), peak prediction reaches 76 / 80 / 76 % "
            "across the f0 sweep (avg ~77.5 %) and axial RMSE drops "
            "to ~0.12 — a real gain over C-7. The production demo "
            "reverts to the linear transform that the C-7 sweep "
            "selected. **Closing the residual peak gap "
            "to ≤ 5 % is still queued**, but the path forward is no "
            "longer the log-pressure transform; the next two "
            "candidates are (a) replacing the regression head with a "
            "peak-prominence loss term that explicitly penalises "
            "max-pred / max-target mismatch, and (b) the real-PSTD "
            "`KernelCube` loader so the training set replaces the "
            "Penttinen-Gaussian proxy with measured PSTD envelopes. "
            "Both queue alongside the pyo3 binding "
            "`pykwavers.ParamFieldPINN.{train, infer, save, load}` "
            "that will let ch21e call `ParamFieldPINN.infer(...)` in "
            "place of `KernelCube.query(...)`.\n\n"
            "**Phase C-9 — peak-prominence loss (mixed result; "
            "infrastructure landed).** Added `peak_prominence_weight` "
            "to `TrainingConfig` and an autodiff `(max(pred_pmax) − "
            "max(target_pmax))²` term to `train_step`. The hypothesis: "
            "the volumetric data MSE under-fits the focal peak "
            "because the argmax voxel contributes only `1/batch` of "
            "the mean; a dedicated max-vs-max gradient channel should "
            "lift the peak. Empirically, on the mixed-frequency 4-"
            "corner cube the term *fragments* per-f0 peaks: weight=1.0 "
            "gave 81/63/58 % and weight=0.1 gave 81/77/68 %, both "
            "below C-8c's 76/80/76. Root cause: a batch is drawn from "
            "*all four* kernel corners simultaneously, so the single "
            "batch-wide `max(target)` aggregates over an ambiguous "
            "`(f0, pnp)` — gradient steers the argmax-pred voxel "
            "toward 1.0 without honouring which `(f0, pnp)` input that "
            "voxel was supposed to represent. **Correct fix landed "
            "in C-10**: per-kernel-scoped prominence loss (group "
            "batch rows by source-kernel index, compute the max-pair "
            "per group via boolean masking, accumulate). The C-9 "
            "infrastructure stays in the crate; the demo runs with "
            "`peak_prominence_weight = 0.0` so C-8c remains the "
            "best-performing path while C-10 is exercised through "
            "the unit-test suite.\n\n"
            "**Phase C-10 — per-kernel-scoped prominence loss "
            "(structural fix landed; empirically null on 4-corner "
            "cube).** Extended `TrainingBatch` with a `group_ids: "
            "Tensor<B, 1>` column emitted by `KernelCubeSampler` "
            "(one ID per source kernel, dense 0..num_groups), then "
            "rewrote the prominence term in `train_step` to mask the "
            "`p_max` channel per group via `equal_elem(g).float()`, "
            "fill out-of-group rows with `-1e6` so per-group `.max()` "
            "selects the in-group maximum, and accumulate the "
            "squared gap. Empty groups contribute `0` with zero "
            "gradient (the masked sentinel is constant in the "
            "autodiff graph). Sampler test "
            "`test_sampler_emits_distinct_group_ids_per_kernel` "
            "confirms a 2048-sample uniform batch covers all 4 "
            "active groups; gradient-flow test confirms prominence "
            "drops > 30 % in 30 steps on the synthetic batch. "
            "Empirically on the 4-corner Penttinen cube the term is "
            "now well-behaved across corners (74 / 74 / 68 at w=1.0; "
            "80 / 79 / 71 at w=0.1) but the average stays within "
            "init-variance of C-8c — the residual peak gap is now "
            "bounded by the (f0, pnp) sampling density of the cube, "
            "not by batch-aggregation. The C-10 path will be "
            "reactivated when the real-PSTD `KernelCube` loader "
            "lands and the sweep gains diverse-peak corners for the "
            "per-group term to honour.\n\n"
            "Run the demo via:\n"
            "```sh\n"
            "cargo run --example field_surrogate_demo --release --features pinn\n"
            "python pykwavers/examples/book/param_pinn_training_figures.py\n"
            "```\n\n"
            "![Loss curves](./param_pinn_loss_curves.png)\n\n"
            "![Axial line fit](./param_pinn_axial_line_fit.png)\n\n"
        )
        f.write("### Figure 21.18 — CT segmentation\n\n")
        f.write(f"![CT segmentation](data:image/png;base64,{b64_seg})\n\n")
        f.write("### Figure 21.19 — Pressure, P_cav, and predicted lesion overlay (3 scenarios)\n\n")
        f.write(f"![Lesion panel](data:image/png;base64,{b64_pan})\n\n")
        f.write("### Figure 21.20 — Raster scan overlay (auto-sized for full coverage)\n\n")
        f.write(
            "Each scenario's raster pitch is auto-set to the per-shot "
            "footprint diameter (overlap factor 0.7), so all three regimes "
            "achieve full tumour coverage. Coloured circles mark the "
            "per-shot footprint at every raster centre on the slice.\n\n"
        )
        f.write(f"![Raster overlay](data:image/png;base64,{b64_ras})\n\n")
        f.write("### Figure 21.21 — Real-CT scenario metrics summary\n\n")
        f.write(f"![Metrics summary](data:image/png;base64,{b64_met})\n\n")
        f.write("### Figure 21.22 — Sonication animation (real-time, synchronised)\n\n")
        f.write(
            "Animated GIF of the raster scan progressing in **real treatment "
            "time** for all three scenarios in parallel. Each panel fills with "
            "per-shot footprint circles as the focus moves along a serpentine "
            "path; the white **×** marks the current focus. Because the "
            "scenarios are synchronised by clock time, the viewer sees ms "
            "shock-vapor (8× interleaved subspots, 8 Hz effective) finish "
            "first at ~2.5 min, sub-threshold ms (4× subspots, 8 Hz) finish at "
            "~4 min, and HistoSonics-style μs (200 Hz, 13,500 points) finish "
            "last at ~5.6 min. Subspot interleaving lets ms regimes match or "
            "beat μs throughput despite the per-spot 1 Hz physics constraint.\n\n"
        )
        f.write(f"![Sonication animation](data:image/gif;base64,{b64_anim})\n\n")
        f.write("### Figure 21.23 — Raster path strategy comparison\n\n")
        f.write(
            "Same scenario (ms shock-vapor on the LiTS17 HCC target) animated "
            "under four different raster-ordering strategies — same shots, same "
            "pulses-per-spot, only the **order** of firing differs:\n\n"
            "Modern histotripsy systems (HistoSonics Edison, Khokhlova-group "
            "phased arrays) reposition the focus electronically via phased-array "
            "delays in microseconds, so 'shortest mechanical-arm path' is no "
            "longer the dominant constraint — strategy is driven by physics: "
            "bubble-cloud shadowing, residual-nuclei seeding, and thermal-load "
            "distribution. The interleaved-subspot scheduling already used "
            "across all scenarios is itself an electronic-steering pattern.\n\n"
            "Off-focus efficiency is modelled with apodized re-weighting "
            "(elements pointing toward the steered focus are favoured), which "
            "extends the compensable steering range to roughly ±7 mm lateral "
            "/ ±21 mm axial at 1 MHz (scaled with wavelength). The drive "
            "amplitude is boosted by 1/ε up to a 1.5× amplifier headroom; "
            "shots beyond this saturation point trigger a mechanical re-anchor "
            "(the array translates so that shot becomes the new geometric "
            "focus). Treatment times include a 2 s penalty per re-anchor for "
            "robotic positioning, which dominates wall-clock time when the "
            "tumour exceeds the steering window. The metrics dict carries "
            "`n_mechanical_reanchors` and `n_steering_degraded_shots` so this "
            "trade-off is auditable per scenario.\n\n"
            "* **Serpentine (boustrophedon)** — predictable raster, easy to map "
            "  onto array-steering coordinates and acoustic-impedance maps. "
            "  Produces highly non-uniform intermediate dose because one corner "
            "  finishes before the other starts.\n"
            "* **Outside-in** — periphery first. Avoids bubble-cloud shadowing "
            "  of peripheral shots by an already-treated central core "
            "  (Maxwell 2013, Macoskey 2018).\n"
            "* **Inside-out** — centre first. Residual nuclei from early shots "
            "  lower the cavitation threshold for surrounding pulses "
            "  (Vlaisavljevich 2013) — most efficient for marginally-suprathreshold "
            "  scenarios.\n"
            "* **Adaptive low-dose-first** — at every step, picks the next "
            "  shot at the location with the lowest current cumulative dose. "
            "  Trivially executable on a phased array (next steering target = "
            "  argmin of running dose field) and gives the most uniform fill at "
            "  every intermediate time.\n\n"
        )
        f.write(f"![Strategy comparison](data:image/gif;base64,{b64_strat})\n\n")
        f.write("\n#### Per-scenario numerical results\n\n")
        f.write("Two effects determine treatment time. The **per-spot PRF** is "
                "set by the physics of each regime: μs intrinsic-threshold by "
                "the ~5 ms residual-bubble dissolution time (Vlaisavljevich 2015 "
                "→ 200 Hz optimum); ms shock-vapor by the ~1 s vapor-cavity "
                "dissolution + thermal relaxation time (Khokhlova 2014 → 1 Hz); "
                "ms sub-threshold cavitation by ~500 ms inertial-collapse "
                "memory (~2 Hz). The **effective transducer PRF** can however "
                "be much higher when the system electronically steers across "
                "*N* spatially-separated subspots within each per-spot period, "
                "so each subspot still sees its required inter-pulse interval "
                "while the transducer fires *N* × faster overall. ms shock-vapor "
                "with 8 interleaved subspots achieves an 8 Hz effective rate, "
                "cutting treatment time from 20 min to ~2.5 min for full "
                "tumour coverage. Subspot separation must exceed the thermal-"
                "diffusion length for the inter-pulse interval (~5 mm at 1 s "
                "in liver) to avoid bulk thermal cross-talk.\n\n")
        f.write("| Scenario | Per-shot footprint | # raster pts | "
                "Per-spot PRF | Subspots | Effective PRF | Treatment time | "
                "Coverage | $T_{transient}$ | $T_{steady}$ |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for m in metrics:
            f.write(f"| {m['scenario']} | "
                    f"{m['per_shot_volume_mm3']:.0f} mm³ "
                    f"(r {m['per_shot_radius_mm']:.1f} mm) | "
                    f"{m['raster_points']} | "
                    f"{m['per_spot_prf_hz']:.0f} Hz | "
                    f"× {m['interleave_subspots']} | "
                    f"{m['effective_prf_hz']:.0f} Hz | "
                    f"{m['treatment_min']:.1f} min | "
                    f"{m['tumour_coverage_pct']:.1f}% | "
                    f"{m['T_transient_C']:.1f} °C | "
                    f"{m['T_steady_C']:.1f} °C |\n")
        f.write(f"\nTumour volume: {metrics[0]['tumour_volume_mm3']/1e3:.2f} cm³. "
                f"All scenarios use the same anatomy and native HCC tumour; differences "
                f"arise only from the regime-specific waveform, raster strategy, "
                f"and bulk-thermal regulation.\n")

    print("[ch21e] Done.")


if __name__ == "__main__":
    main()
