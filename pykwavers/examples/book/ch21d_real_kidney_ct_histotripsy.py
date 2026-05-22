"""
Chapter 21d: Histotripsy planning on a real abdominal CT — kidney tumour
========================================================================

Loads KiTS19 case_00000 (CC-BY-NC-SA 4.0; Heller 2019) — an arterial-
phase contrast-enhanced abdominal CT — and runs the three clinical
histotripsy scenarios on the patient's actual renal cell carcinoma
(RCC) tumour, using the dataset's voxel-level segmentation labels
(kidney = 1, tumour = 2) rather than a synthetic placeholder.

CT volume    : https://kits19.sfo2.digitaloceanspaces.com/master_00000.nii.gz
Segmentation : https://raw.githubusercontent.com/neheller/kits19/master/data/case_00000/segmentation.nii.gz
License      : CC-BY-NC-SA 4.0

Pipeline:
    1. Load CT + native segmentation.
    2. Crop to a slab covering the full tumour with a margin.
    3. Resample to ~1.2 mm isotropic.
    4. Build tissue-label volume — HU thresholds for air/fat/muscle/
       bone/skin; native labels override for kidney + tumour.
    5. Place the focus on the centroid of the actual tumour mask.
    6. Forward-propagate Rayleigh-Sommerfeld focused-bowl pressure
       through the layered patient geometry per scenario.
    7. Compute mech / thermal masks and the cumulative cavitation-dose
       heatmap; render the embedded-figures markdown.

Outputs:
    docs/book/figures/ch21d/*.{png,pdf}
    docs/book/figures/ch21d/embedded_figures.md
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — required for projection="3d"
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
                           distance_transform_edt, label, zoom)
from scipy.special import erf

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CT_PATH = os.path.join(REPO_ROOT, "data", "kits19_sample", "case_00000.nii.gz")
SEG_PATH = os.path.join(REPO_ROOT, "data", "kits19_sample", "segmentation_00000.nii.gz")
OUT_DIR = os.path.join(REPO_ROOT, "docs", "book", "figures", "ch21d")
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
    # Intrinsic cavitation threshold at 1 MHz (Pa).
    # Tissue-specific values from Vlaisavljevich 2015/2016 and Maxwell 2013.
    # float('inf') = no intrinsic cavitation (rigid scatterer or gas interface).
    pt_pa_1mhz: float = 28.2e6   # default: water-based global reference (Vlaisavljevich 2015)
    # 1σ width of the Maxwell 2013 erf-CDF threshold distribution (Pa).
    pt_sigma_pa: float = 0.96e6  # default: water-based σ (Maxwell 2013)


#  Acoustic/thermal: Duck 1990 / IT'IS Foundation v4.1 / Mast 2000
#  Cavitation thresholds: Vlaisavljevich 2015 (JASA 138:1864) — water baseline;
#    Vlaisavljevich 2016 (JASA 140:3504) — porcine tissue data; Maxwell 2013
#    (JASA 134:1765) — erf-CDF model; Hall 2007 (J Urol 178:2174) — in vivo
#    porcine kidney histotripsy; Roberts 2014 (BJU Int) — RCC clinical context.
AIR    = Tissue(0, "air",       1.2, 343.0,    0.0,  1.0,    1005.0, 0.026, 0.0,
                pt_pa_1mhz=float('inf'), pt_sigma_pa=1.0)
SKIN   = Tissue(1, "skin",   1109.0, 1624.0,  21.158, 1.10, 3391.0, 0.37, 1.06,
                # Water-rich dermis; no specific histotripsy data — assume near muscle.
                pt_pa_1mhz=26.0e6, pt_sigma_pa=3.0e6)
FAT    = Tissue(2, "fat",     911.0, 1440.0,   4.836, 1.10, 2348.0, 0.21, 0.43,
                # Vlaisavljevich 2015: lipid-rich tissues ~13–16 MPa.
                pt_pa_1mhz=14.0e6, pt_sigma_pa=2.0e6)
MUSCLE = Tissue(3, "muscle", 1090.0, 1588.0,   8.054, 1.10, 3421.0, 0.49, 0.67,
                # Vlaisavljevich 2015: skeletal muscle 23–27 MPa.
                pt_pa_1mhz=25.0e6, pt_sigma_pa=2.0e6)
BONE   = Tissue(4, "bone",   1908.0, 4080.0, 250.0,   1.0,  1313.0, 0.32, 0.10,
                # Cortical bone: no intrinsic cavitation.
                pt_pa_1mhz=float('inf'), pt_sigma_pa=1.0)
# Kidney parenchyma + RCC (renal cell carcinoma): properties from
# Duck 1990, IT'IS Foundation v4.1, Mast 2000. Kidney perfusion is
# extremely high (~58 ml/100 g/min ≈ 9.7 kg/m³/s) — among the highest
# of any organ — which gives renal histotripsy markedly different
# bulk-thermal behaviour vs liver despite similar acoustic properties.
KIDNEY = Tissue(5, "kidney", 1066.0, 1567.0,   8.000, 1.10, 3763.0, 0.53, 9.7,
                # Kidney parenchyma is highly vascular and water-rich.
                # Vlaisavljevich 2015 / Hall 2007 (J Urol 178:2174) suggest
                # threshold 22–26 MPa at 1 MHz; mean 24.0 MPa, σ = 3.0 MPa.
                pt_pa_1mhz=24.0e6, pt_sigma_pa=3.0e6)
TUMOR  = Tissue(6, "rcc",    1050.0, 1550.0,  12.000, 1.10, 3750.0, 0.55, 5.5,
                # RCC: variable histology (clear-cell, papillary, chromophobe).
                # Clear-cell RCC (~75 % of cases) has lipid-rich cytoplasm and
                # variable necrosis — lower threshold than normal parenchyma.
                # Inferred from Maxwell 2013 tumor data context: 21.5 MPa, σ=2 MPa.
                pt_pa_1mhz=21.5e6, pt_sigma_pa=2.0e6)

TISSUES = [AIR, SKIN, FAT, MUSCLE, BONE, KIDNEY, TUMOR]


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
# CT + native-segmentation loading
# ───────────────────────────────────────────────────────────────────────


def load_ct_and_segment(target_dx_m: float = 1.2e-3) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load KiTS19 case_00000 CT + segmentation, crop to a slab covering
    the renal tumour with margin, resample to isotropic, and build the
    tissue-label volume — kidney + tumour from native labels, the
    surrounding tissues from HU thresholding.

    Returns
    -------
    ct_hu     : resampled HU volume.
    label_vol : tissue-label volume (int8).
    info      : dict with dx, shape, axis arrays, focus index (real
                tumour centroid).
    """
    print(f"[ch21d] Loading CT:  {CT_PATH}")
    print(f"[ch21d] Loading seg: {SEG_PATH}")
    v = nib.load(CT_PATH)
    s = nib.load(SEG_PATH)
    raw = v.get_fdata().astype(np.float32)
    seg_raw = s.get_fdata().astype(np.int16)
    zooms = v.header.get_zooms()
    print(f"  raw shape={raw.shape}, voxel mm={zooms}, axcodes={nib.aff2axcodes(v.affine)}")
    assert raw.shape == seg_raw.shape, "CT and segmentation grids must match"

    # Find the tumour bounding box (label 2) and crop with margin so the
    # transducer's focal envelope and surrounding tissue layers fit.
    # Margin chosen large enough (~50 mm at 0.5 mm/vox = 100 vox) to
    # capture skin / fat / muscle / bone for the layered acoustic path.
    tum_coords = np.argwhere(seg_raw == 2)
    if len(tum_coords) == 0:
        raise SystemExit("KiTS19 segmentation has no label-2 (tumour) voxels.")
    margin_z = 60   # 0.5 mm × 60 = 30 mm along axis 0 (Inferior-axis)
    margin_y = 100  # 0.92 mm × 100 ≈ 92 mm along axis 1 (Anterior-axis)
    margin_x = 100  # 0.92 mm × 100 ≈ 92 mm along axis 2 (Lateral-axis)
    z0c = max(0, int(tum_coords[:, 0].min()) - margin_z)
    z1c = min(raw.shape[0], int(tum_coords[:, 0].max()) + margin_z)
    y0c = max(0, int(tum_coords[:, 1].min()) - margin_y)
    y1c = min(raw.shape[1], int(tum_coords[:, 1].max()) + margin_y)
    x0c = max(0, int(tum_coords[:, 2].min()) - margin_x)
    x1c = min(raw.shape[2], int(tum_coords[:, 2].max()) + margin_x)
    raw = raw[z0c:z1c, y0c:y1c, x0c:x1c]
    seg_raw = seg_raw[z0c:z1c, y0c:y1c, x0c:x1c]
    print(f"  cropped to shape={raw.shape} around tumour bbox")

    # Re-orient (z_si, y_ap, x_lat) → (y_ap, x_lat, z_si) so axis 0
    # is the depth-from-anterior — the histotripsy beam direction.
    raw = np.transpose(raw, (1, 2, 0))
    seg_raw = np.transpose(seg_raw, (1, 2, 0))
    raw_zooms = (zooms[1], zooms[2], zooms[0])

    target_dx_mm = target_dx_m * 1e3
    factors = tuple(z / target_dx_mm for z in raw_zooms)
    ct_hu = zoom(raw, factors, order=1, prefilter=False).astype(np.float32)
    seg = zoom(seg_raw.astype(np.float32), factors, order=0,
               prefilter=False).astype(np.int8)
    print(f"  resampled shape={ct_hu.shape}, voxel={target_dx_mm:.2f} mm")

    nx, ny, nz = ct_hu.shape

    # Tissue-label volume: HU thresholds for the surrounding tissues,
    # with native labels overriding for kidney + tumour. The native
    # segmentation is anatomically authoritative — HU-derived liver/
    # kidney parenchyma classes overlap and connected-component
    # heuristics are unreliable.
    label_vol = np.zeros_like(ct_hu, dtype=np.int8)
    label_vol[ct_hu < -500] = AIR.label
    label_vol[(ct_hu >= -500) & (ct_hu < -100)] = FAT.label
    label_vol[(ct_hu >= -100) & (ct_hu < 30)]  = MUSCLE.label
    label_vol[(ct_hu >= 30) & (ct_hu < 200)]   = MUSCLE.label  # soft tissue
    label_vol[(ct_hu >= 200)] = BONE.label
    skin_thickness_vox = max(int(round(2.0e-3 / target_dx_m)), 1)
    body = label_vol != AIR.label
    body_dil = binary_dilation(~body, iterations=skin_thickness_vox)
    skin_shell = body & body_dil
    label_vol[skin_shell] = SKIN.label

    # Native labels override (kidney parenchyma minus tumour, then tumour)
    label_vol[seg == 1] = KIDNEY.label
    label_vol[seg == 2] = TUMOR.label

    # Focal point = real tumour centroid
    tumour_mask = label_vol == TUMOR.label
    if not tumour_mask.any():
        raise SystemExit("Tumour mask vanished after resampling")
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
        "organ_label": "Kidney",
        "tumour_label": "RCC",
        "dataset_label": "KiTS19 case_00000",
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


def focal_fwhm(f0: float) -> tuple[float, float]:
    """Return (lateral, axial) FWHM of the focal pressure spot in metres
    for the 50 mm aperture / 120 mm radius-of-curvature bowl. These are
    the clinical "focal spot size" quoted in literature (Penttinen 1976,
    O'Neil 1949) — w_lat = 1.41·λ·F#, w_axial = 7·λ·F#²."""
    lam = 1540.0 / f0
    fnum = 120.0e-3 / (2.0 * 50.0e-3)
    return 1.41 * lam * fnum, 7.0 * lam * fnum ** 2


def focused_bowl_pressure(info, props, f0, source_pa) -> np.ndarray:
    nx, ny, nz = info["shape"]
    dx = info["dx"]
    x_axis = info["x_axis"]; y_axis = info["y_axis"]; z_axis = info["z_axis"]
    focus_idx = info["focus_idx"]
    x_focus = focus_idx[0] * dx; y_focus = focus_idx[1] * dx - ny * dx / 2
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
    # 3D heterogeneous attenuation: cumulative path integral along each (j,k)
    # column from the transducer face to each depth voxel.
    # For each lateral position (j,k), integrate α(x,j,k)·dx from x=0 inward.
    # Replaces the 1D central-axis approximation (focus_idx j,k applied to all
    # lateral positions) with per-ray integrals under the paraxial approximation
    # valid for the 50mm-aperture / 120mm-ROC bowl (f# = 1.2).
    # props["alpha"] shape: (Nx, Ny, Nz) in Np/m
    cum_atten_3d = np.exp(-np.cumsum(props["alpha"], axis=0) * dx)
    p = source_pa * env * cum_atten_3d
    return p.astype(np.float32)


# ───────────────────────────────────────────────────────────────────────
# Lesion + thermal model (corrected ch21b version)
# ───────────────────────────────────────────────────────────────────────


def intrinsic_threshold(f0: float) -> float:
    """Vlaisavljevich 2015 frequency-dependent intrinsic threshold (Pa).

    Global water-based reference: p_t(f) = 28.2 MPa + 1.4 MPa · log₁₀(f / 1 MHz).
    Use intrinsic_threshold_tissue_pa() for tissue-specific physics.
    """
    return 28.2e6 + 1.4e6 * np.log10(f0 / 1e6)


def intrinsic_threshold_tissue_pa(tissue: "Tissue", f0: float, T_C: float = 20.0) -> float:
    """Tissue-specific, frequency- and temperature-dependent intrinsic threshold.

    Model: Vlaisavljevich 2015/2016, Maxwell 2013.
      p_t(f, T) = p_{t,1MHz} + 1.4 MPa · log₁₀(f / 1 MHz) − 0.3 MPa · max(0, T − 20)

    Temperature correction: −0.3 MPa/°C empirical (Vlaisavljevich 2015 Fig. 7).
    Reference temperature 20 °C matches in-vitro calibration standard.

    Parameters
    ----------
    tissue : Tissue instance with pt_pa_1mhz field set
    f0     : driving frequency (Hz)
    T_C    : local temperature (°C, default 20.0 = in-vitro reference)

    Returns
    -------
    p_t : threshold magnitude (Pa); inf for tissue with no intrinsic cavitation
    """
    if not np.isfinite(tissue.pt_pa_1mhz):
        return float('inf')
    freq_shift = 1.4e6 * np.log10(max(f0, 1.0) / 1e6)
    temp_shift = -0.3e6 * max(0.0, T_C - 20.0)
    return tissue.pt_pa_1mhz + freq_shift + temp_shift


def cav_probability(p, f0):
    """Global water-based Maxwell 2013 erf-CDF (kept for backward compatibility).

    Prefer cav_probability_tissue() for tissue-specific physics.
    """
    pt = 28.2e6 + 1.4e6 * np.log10(f0 / 1e6)
    sigma = 0.96e6
    return 0.5 * (1.0 + erf((p - pt) / (sigma * np.sqrt(2.0))))


def cav_probability_tissue(
    p_field: np.ndarray,
    label_vol: np.ndarray,
    f0: float,
    T_field: np.ndarray | None = None,
) -> np.ndarray:
    """Per-voxel tissue-specific Maxwell 2013 erf-CDF cavitation probability.

    Each voxel is assigned a threshold p_t and width σ from its tissue label,
    then:  P_cav(x) = 0.5 · (1 + erf((|p(x)| − p_t(label, f, T)) / (σ · √2)))

    Labels absent from TISSUES use the global water reference (28.2 MPa, σ=0.96 MPa).
    Tissues with pt_pa_1mhz = inf (AIR, BONE) return P_cav = 0.

    Parameters
    ----------
    p_field   : (Nx, Ny, Nz) rarefactional pressure magnitudes (Pa, ≥ 0)
    label_vol : (Nx, Ny, Nz) int tissue labels matching TISSUES list
    f0        : driving frequency (Hz)
    T_field   : (Nx, Ny, Nz) temperature field (°C); None → 20 °C everywhere

    Returns
    -------
    pcav : (Nx, Ny, Nz) float32 in [0, 1]
    """
    freq_shift = 1.4e6 * np.log10(max(f0, 1.0) / 1e6)
    pt_field    = np.full(p_field.shape, 28.2e6 + freq_shift, dtype=np.float64)
    sigma_field = np.full(p_field.shape, 0.96e6,              dtype=np.float64)

    for tissue in TISSUES:
        mask = label_vol == tissue.label
        if not mask.any():
            continue
        if not np.isfinite(tissue.pt_pa_1mhz):
            pt_field[mask]    = 1.0e15
            sigma_field[mask] = 1.0
            continue
        pt_base = tissue.pt_pa_1mhz + freq_shift
        if T_field is not None:
            temp_corr = -0.3e6 * np.maximum(0.0, T_field[mask] - 20.0)
            pt_field[mask] = pt_base + temp_corr
        else:
            pt_field[mask] = pt_base
        sigma_field[mask] = tissue.pt_sigma_pa

    pcav = 0.5 * (1.0 + erf((p_field - pt_field) / (sigma_field * np.sqrt(2.0))))
    return pcav.astype(np.float32)


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
    # Thermal maps first: T_transient feeds the temperature correction for p_t.
    cem43, T_steady, T_transient = thermal_maps(p_field, props, sc)
    # Per-voxel tissue-specific cavitation probability with temperature correction.
    pcav = cav_probability_tissue(p_field, label_vol, sc.f0, T_field=T_transient)
    coll = collapse_strength(p_field, sc.f0)

    pulses_per_pt = max(sc.pulses_per_point, 1)
    # Cavitation-cloud radius scales with overpressure (Vlaisavljevich
    # 2013, Maxwell 2013): r_cloud ≈ λ/4 at threshold, growing to ~λ/2
    # at p > 1.5 p_t. Dilating the threshold-exceeding region by this
    # radius models the bubble cloud expansion beyond the strict
    # p > p_t kernel and is the correct ablation-mask construction at
    # any grid spacing where dx > λ/4.
    lam_m = 1540.0 / sc.f0
    # Use focal-tissue threshold, not the global water reference.
    _fi0, _fi1, _fi2 = info["focus_idx"]
    _focus_label   = int(label_vol[_fi0, _fi1, _fi2])
    _focus_tissue  = next((t for t in TISSUES if t.label == _focus_label), TUMOR)
    _focus_T_C     = float(T_transient[_fi0, _fi1, _fi2])
    pt_focal       = intrinsic_threshold_tissue_pa(_focus_tissue, sc.f0, _focus_T_C)
    if not np.isfinite(pt_focal):
        pt_focal = intrinsic_threshold(sc.f0)
    cloud_r_m = lam_m / 4.0 * max(sc.pnp / pt_focal, 1.0)
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

    tumour = label_vol == TUMOR.label
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
    if raster_grid.any():
        focus_idx = np.array(info["focus_idx"])
        nx_, ny_, nz_ = log_safe.shape
        offs = np.argwhere(np.ones_like(per_pulse, dtype=bool))  # all voxels
        # Direct shift-and-add via per-raster-point slicing (same pattern
        # as superpose). Sum log(1-p) shifted by (rpt - focus_idx).
        for rpt in np.argwhere(raster_grid):
            shift = rpt - focus_idx
            src_x0 = max(0, -shift[0]); src_x1 = min(nx_, nx_ - shift[0])
            src_y0 = max(0, -shift[1]); src_y1 = min(ny_, ny_ - shift[1])
            src_z0 = max(0, -shift[2]); src_z1 = min(nz_, nz_ - shift[2])
            if src_x1 <= src_x0 or src_y1 <= src_y0 or src_z1 <= src_z0:
                continue
            dst_x0 = src_x0 + shift[0]; dst_x1 = src_x1 + shift[0]
            dst_y0 = src_y0 + shift[1]; dst_y1 = src_y1 + shift[1]
            dst_z0 = src_z0 + shift[2]; dst_z1 = src_z1 + shift[2]
            log_dose_super[dst_x0:dst_x1, dst_y0:dst_y1, dst_z0:dst_z1] += \
                log_safe[src_x0:src_x1, src_y0:src_y1, src_z0:src_z1]
    cav_dose = 1.0 - np.exp(log_dose_super * pulses_per_pt)

    # Treatment time is set by the ACTUAL placed raster fill: each spot
    # gets pulses_per_pt clinical doses, with electronic interleave
    # spreading across N subspots so the transducer's effective PRF is
    # sc.prf * sc.interleave_subspots.
    actual_n = max(int(raster_grid.sum()), 1)
    effective_prf = sc.prf * max(sc.interleave_subspots, 1)
    actual_treatment_s = actual_n * pulses_per_pt / effective_prf
    # Rescale cem43 thermal dose to the actual treatment time.
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
    print("[ch21d] Building sonication animation")
    from matplotlib.animation import PillowWriter, FuncAnimation

    fx = info["focus_idx"][0]
    tumour = label_vol == TUMOR.label
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
    focus_y_local = info["focus_idx"][1] - y0
    focus_z_local = info["focus_idx"][2] - z0
    ny_loc = y1 - y0
    nz_loc = z1 - z0
    for sc in scenarios:
        paths[sc.name] = serpentine_order(results[sc.name]["raster_grid"])
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
    tumour_sl = (label_vol[fx, y0:y1, z0:z1] == TUMOR.label).astype(float)

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
        for x, yi, zi in paths[sc.name]:
            sy = yi - info["focus_idx"][1]
            sz = zi - info["focus_idx"][2]
            sy0 = max(0, -sy); sy1 = min(ny_loc, ny_loc - sy)
            sz0 = max(0, -sz); sz1 = min(nz_loc, nz_loc - sz)
            if sy1 > sy0 and sz1 > sz0:
                acc[sy0+sy:sy1+sy, sz0+sz:sz1+sz] += per_pulse[sy0:sy1, sz0:sz1] * pulses
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
                    art["events_acc"][dst_y0:dst_y1, dst_z0:dst_z1] += \
                        per_pulse[src_y0:src_y1, src_z0:src_z1] * pulses
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
                 "(per-panel timeline; KiTS19 case_00000 RCC tumour, native segmentation)",
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
    "#a0628c",  # 5 kidney
    "#3b1a4a",  # 6 rcc
])


def plot_transducer_placement_3d(label_vol: np.ndarray, info: dict) -> str:
    """3D scatter: HistoSonics 50 mm / 120 mm-ROC bowl placed anterior to the
    patient skin, targeting the RCC tumour centroid.

    The bowl element positions are computed analytically from the hemispherical
    shell at radius R_f = 120 mm centred on the focus.  All elements are at
    negative depth (outside the patient body, which begins at depth = 0).

    Refs: HistoSonics Edison system; Maxwell 2013 (JASA 134:1765);
    Vlaisavljevich 2015 (JASA 138:1864).
    """
    dx  = info["dx"]
    nx, ny, nz = info["shape"]
    x_axis = info["x_axis"]
    y_axis = info["y_axis"]
    z_axis = info["z_axis"]
    ix0, iy0, iz0 = info["focus_idx"]
    x_focus = ix0 * dx
    y_focus = iy0 * dx - ny * dx / 2.0
    z_focus = iz0 * dx - nz * dx / 2.0

    # ── Transducer bowl elements ──────────────────────────────────────────
    # 50 mm aperture diameter, 120 mm radius of curvature (HistoSonics class).
    # Bowl centred on focus; opens in +x direction (into the patient).
    # All elements sit at x < x_focus (anterior, outside the body when x_focus < R_f).
    R_f: float = 120.0e-3  # radius of curvature (m)
    a:   float = 25.0e-3   # aperture half-diameter (m)
    theta_max = float(np.arcsin(a / R_f))  # ≈ 12.0°

    ex_list: list[float] = [x_focus - R_f]   # apex element
    ey_list: list[float] = [y_focus]
    ez_list: list[float] = [z_focus]
    for ring in range(1, 9):
        th = theta_max * ring / 8.0
        # Circumferential element count proportional to ring arc length.
        n_phi = max(8, int(round(40.0 * np.sin(th) / np.sin(theta_max))))
        phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        ex_list.extend([float(x_focus - R_f * np.cos(th))] * n_phi)
        ey_list.extend((y_focus + R_f * np.sin(th) * np.cos(phis)).tolist())
        ez_list.extend((z_focus + R_f * np.sin(th) * np.sin(phis)).tolist())
    ex = np.array(ex_list)
    ey = np.array(ey_list)
    ez = np.array(ez_list)

    # ── Body (skin) surface — anterior face per lateral position ──────────
    # Vectorised: first non-air voxel along the depth axis (axis 0).
    body_bool = label_vol != AIR.label
    first_ix = np.argmax(body_bool, axis=0)   # shape (ny, nz)
    has_body  = body_bool.any(axis=0)          # shape (ny, nz)
    iy_grid, iz_grid = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
    sx_raw = x_axis[first_ix[has_body]]
    sy_raw = y_axis[iy_grid[has_body]]
    sz_raw = z_axis[iz_grid[has_body]]
    rng = np.random.default_rng(seed=0)
    if len(sx_raw) > 2500:
        sel = rng.choice(len(sx_raw), 2500, replace=False)
        sx_raw, sy_raw, sz_raw = sx_raw[sel], sy_raw[sel], sz_raw[sel]

    # ── Kidney + tumour surface voxels ────────────────────────────────────
    organ_mask = (label_vol == KIDNEY.label) | (label_vol == TUMOR.label)
    tumor_mask = label_vol == TUMOR.label
    organ_surf = organ_mask & ~binary_erosion(organ_mask, iterations=1)
    tumor_surf = tumor_mask & ~binary_erosion(tumor_mask, iterations=1)
    c_org = np.argwhere(organ_surf)
    c_tum = np.argwhere(tumor_surf)
    step_org = max(1, len(c_org) // 1500)
    step_tum = max(1, len(c_tum) // 800)
    c_org = c_org[::step_org]
    c_tum = c_tum[::step_tum]
    ox = x_axis[c_org[:, 0]]; oy = y_axis[c_org[:, 1]]; oz = z_axis[c_org[:, 2]]
    tx = x_axis[c_tum[:, 0]]; ty = y_axis[c_tum[:, 1]]; tz = z_axis[c_tum[:, 2]]

    # ── 3D scatter figure ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(sx_raw * 1e3, sy_raw * 1e3, sz_raw * 1e3,
               c="silver", s=1.5, alpha=0.06, linewidths=0, zorder=1,
               label="Patient skin surface")
    ax.scatter(ox * 1e3, oy * 1e3, oz * 1e3,
               c="#f5a623", s=4, alpha=0.45, linewidths=0, zorder=2, label="Kidney")
    ax.scatter(tx * 1e3, ty * 1e3, tz * 1e3,
               c="#c0392b", s=14, alpha=0.90, linewidths=0, zorder=3, label="RCC tumour")
    ax.scatter(ex * 1e3, ey * 1e3, ez * 1e3,
               c="#2c3e50", s=18, alpha=0.85, edgecolors="white", linewidths=0.4,
               zorder=4, label="Transducer bowl (50 mm / 120 mm-ROC)")
    ax.scatter([(x_focus - R_f) * 1e3], [y_focus * 1e3], [z_focus * 1e3],
               c="limegreen", s=80, marker="o", zorder=5, label="Bowl apex (coupling-gel contact)")
    ax.scatter([x_focus * 1e3], [y_focus * 1e3], [z_focus * 1e3],
               c="cyan", s=120, marker="+", linewidths=2.5, zorder=5, label="Focus (RCC centroid)")

    ax.set_xlabel("Depth (mm, anterior →)")
    ax.set_ylabel("Lateral Y (mm)")
    ax.set_zlabel("Sup–Inf Z (mm)")
    ax.set_title(
        "HistoSonics 50 mm / 120 mm-ROC Bowl — Kidney Placement\n"
        "(KiTS19 case_00000 · bowl anterior to skin · focus = RCC centroid)",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper right", markerscale=1.3)
    ax.view_init(elev=18, azim=-65)

    # Equal-aspect bounding cube enclosing all point clouds.
    all_x = np.concatenate([ex, ox, tx, sx_raw])
    all_y = np.concatenate([ey, oy, ty, sy_raw])
    all_z = np.concatenate([ez, oz, tz, sz_raw])
    cx, cy, cz = float(all_x.mean()), float(all_y.mean()), float(all_z.mean())
    r_eq = float(max(
        np.abs(all_x - cx).max(),
        np.abs(all_y - cy).max(),
        np.abs(all_z - cz).max(),
    )) * 1.08
    ax.set_xlim((cx - r_eq) * 1e3, (cx + r_eq) * 1e3)
    ax.set_ylim((cy - r_eq) * 1e3, (cy + r_eq) * 1e3)
    ax.set_zlim((cz - r_eq) * 1e3, (cz + r_eq) * 1e3)

    fig.tight_layout()
    return save_fig(fig, "fig_21d_transducer_placement_3d")


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

    # Bottom row: tissue label segmentation with RCC tumour highlighted
    ax[1, 0].imshow(label_vol[fx, :, :], cmap=TISSUE_CMAP, vmin=0, vmax=6, extent=extent_yz)
    ax[1, 1].imshow(label_vol[:, fy, :], cmap=TISSUE_CMAP, vmin=0, vmax=6, extent=extent_xz)
    ax[1, 2].imshow(label_vol[:, :, fz], cmap=TISSUE_CMAP, vmin=0, vmax=6, extent=extent_xy)
    for a in ax[1, :]:
        a.set_xlabel("[mm]"); a.set_ylabel("[mm]")
        a.scatter([extent_yz[0] + (extent_yz[1] - extent_yz[0]) * 0.5],
                  [extent_yz[2] + (extent_yz[3] - extent_yz[2]) * 0.5],
                  marker="x", c="white", s=0)
    ax[1, 0].set_title("Tissue labels axial (skin/fat/muscle/bone/kidney/RCC)")
    ax[1, 1].set_title("Tissue labels coronal")
    ax[1, 2].set_title("Tissue labels sagittal")

    # Add tumour markers
    for a, sl_axes in zip(ax[1, :], [(fy, fz), (fx, fz), (fx, fy)]):
        pass

    handles = [plt.Rectangle((0, 0), 1, 1, color=TISSUE_CMAP(t.label)) for t in TISSUES]
    fig.legend(handles, [t.name for t in TISSUES], loc="lower center",
               ncol=len(TISSUES), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Real abdominal CT (KiTS19 case_00000) — HU-threshold tissue "
                 "segmentation overlaid with the native KiTS19 kidney + RCC "
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
        tumour_sl = (label_vol[fx, :, :] == TUMOR.label).astype(float)
        axes[2, col].contour(np.flipud(tumour_sl), levels=[0.5], colors="cyan",
                             linewidths=0.8, extent=extent_full)
        dose_sl = r["cav_dose"][fx, :, :]
        im2 = axes[2, col].imshow(dose_sl, cmap=overlay_cmap, extent=extent_full,
                                  vmin=0, vmax=1, aspect="equal")
        plt.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04,
                     label="cumulative cavitation dose")
        m = r["metrics"]
        axes[2, col].set_title(
            f"cavitation-dose heatmap (cyan = RCC outline)\n"
            f"coverage {m.get('tumour_coverage_pct', 0):.0f}%, "
            f"FWHM {m['fwhm_lat_mm']:.1f}×{m['fwhm_ax_mm']:.1f} mm (lat×ax), "
            f"$T_{{ss}}$={m.get('T_steady_C', 0):.0f} °C",
            fontsize=8.5)
        axes[2, col].set(xlabel="z [mm]", ylabel="y [mm]")

    fig.suptitle("Histotripsy on real abdominal CT — focal pressure, $P_{cav}$, "
                 "and cumulative cavitation-dose heatmap (KiTS19 RCC tumour)",
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
    tumour = label_vol == TUMOR.label
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
        # RCC outline
        tumour_sl = (label_vol[fx, y0:y1, z0:z1] == TUMOR.label).astype(float)
        ax.contour(np.flipud(tumour_sl), levels=[0.5], colors="cyan",
                   linewidths=1.0, extent=extent_crop)
        # FWHM lateral half-width — the clinical "focal spot" radius
        # quoted in the literature (Penttinen 1976). Use this for the
        # visual circle so the spacing visually matches the -3 dB
        # contour, not the elongated cavitation cigar.
        psr = r["metrics"]["fwhm_lat_mm"] / 2.0
        tumour_full = label_vol == TUMOR.label
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
    fig.suptitle("Raster scan overlay on real CT — cyan = native RCC tumour outline, "
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

    fig.suptitle("Real-CT histotripsy treatment metrics — KiTS19 case_00000, "
                 "native RCC tumour segmentation as histotripsy target",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    b64 = save_fig(fig, "fig15_real_ct_metrics")
    plt.close(fig)
    print("  saved fig15_real_ct_metrics")
    return b64


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────


def main():
    if not os.path.exists(CT_PATH):
        raise SystemExit(f"CT not found: {CT_PATH}. Download KiTS19 case_00000 first.")

    ct, label_vol, info = load_ct_and_segment(target_dx_m=1.2e-3)

    tumour_volume_m3 = float((label_vol == TUMOR.label).sum() * info["dx"]**3)

    # Run scenarios with TWO passes:
    #   (a) probe pass — single focal exposure to measure per-shot footprint
    #   (b) auto-sized pass — raster pitch matched to per-shot diameter
    #       so coverage is full; treatment time is the minimum required.
    results = {}
    lesions = {}
    metrics = []
    sized_scenarios = []
    for sc_template in SCENARIOS:
        print(f"[ch21d] Probing per-shot footprint: {sc_template.label}")
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
        print(f"[ch21d] Sized scenario (autosize estimate): {sc.raster_points} pts, "
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
    b64_place = plot_transducer_placement_3d(label_vol, info)
    b64_seg   = plot_segmentation(ct, label_vol, info)
    b64_pan   = plot_pressure_and_lesion(ct, label_vol, info, results, lesions)
    b64_met   = plot_metrics_summary(metrics, sized_scenarios)
    b64_ras   = plot_raster_overlay(ct, label_vol, info, results, sized_scenarios)
    b64_anim  = make_sonication_animation(ct, label_vol, info, results, sized_scenarios)

    # Write embedded-figures markdown.
    md_path = os.path.join(OUT_DIR, "embedded_figures.md")
    print(f"[ch21d] Writing embedded markdown: {md_path}")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("<!-- AUTO-GENERATED by ch21d_real_kidney_ct_histotripsy.py — do not edit -->\n\n")
        f.write("## 21.13 Real-CT Histotripsy Treatment — Renal Cell Carcinoma "
                "(KiTS19 case_00000)\n\n")
        f.write(
            "The figures below are produced from a real public-domain abdominal "
            "CT volume (KiTS19 case_00000, "
            "[Heller 2019](https://kits19.grand-challenge.org/), CC-BY-NC-SA 4.0). "
            "The CT is paired with the dataset's voxel-level segmentation labels "
            "(kidney = 1, RCC tumour = 2), so the histotripsy target is the "
            "patient's actual renal cell carcinoma — not a synthetic sphere. "
            "Surrounding tissues (skin / fat / muscle / bone) are classified by "
            "HU thresholding. Acoustic and thermal properties are assigned per "
            "the IT'IS Foundation v4.1 table. Each figure is embedded directly "
            "as base64 PNG bytes — no external links are needed.\n\n"
        )
        f.write("### Figure 21.17 — 3D transducer placement (HistoSonics bowl on skin)\n\n")
        f.write(
            "3-D view of the 50 mm aperture / 120 mm radius-of-curvature "
            "hemispherical bowl positioned anterior to the patient skin surface "
            "(depth < 0) with coupling gel.  The bowl apex (lime circle) is the "
            "deepest point of the dish — closest to the patient — and the cyan "
            "cross marks the geometric focus inside the renal tumour.  All bowl "
            "elements sit outside the body; the skin surface is at depth = 0.\n\n"
        )
        f.write(f"![Transducer placement 3D](data:image/png;base64,{b64_place})\n\n")
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
                "in kidney) to avoid bulk thermal cross-talk.\n\n")
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
                f"All scenarios use the same anatomy and native RCC tumour; differences "
                f"arise only from the regime-specific waveform, raster strategy, "
                f"and bulk-thermal regulation.\n")

    print("[ch21d] Done.")


if __name__ == "__main__":
    main()
