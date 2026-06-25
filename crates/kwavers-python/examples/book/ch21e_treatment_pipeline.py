"""ch21e — Liver histotripsy treatment pipeline (whole-tumour, multi-region, CT).

A complete liver-cancer histotripsy treatment on a REAL liver CT (LiTS17), built
on the kwavers Rust physics core + domain tissue characterization (this script is
geometry + orchestration + plotting only — no domain math, no tissue constants):

  • The segmented HCC tumour is tiled into MULTIPLE SONICATION REGIONS
    (mechanical foci); each region is an electronic-steering SUB-SPOT grid; each
    sub-spot is lesioned by MANY pulses (fractionation). The focus is rastered
    region-by-region until the whole tumour is lesioned, while a sensitive
    structure (OAR) is spared by capping sub-spots near it.
  • Tissue properties (sound speed, density, attenuation, impedance, tensile
    yield stress, intrinsic cavitation threshold) are read from the kwavers-domain
    tissue database via ``kw.tissue_properties`` / ``kw.histotripsy_tissue_properties``.

Outputs (``docs/book/figures/ch21e/``):
  A  fig21_treatment_pipeline_pulsing.png — real pulse train (overview + one-region zoom).
  B  fig22_treatment_pipeline_monitor.png — sensor-recorded cavitation monitor.
  C  anim_treatment_console.gif — ANIMATED console: CT + tumour contour + OAR,
     sub-spots pulsing and dose map accumulating over time, with the live
     spectrum / power / dose monitor.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import pykwavers as kw

from scipy.ndimage import binary_dilation, distance_transform_edt, label, zoom

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from cavitation_dose_monitor import BubbleMedium, simulate_population_emission


def _find_repo_root(start):
    d = start
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "data", "lits17_sample")):
            return d
        d = os.path.dirname(d)
    return os.path.abspath(os.path.join(start, "..", "..", "..", ".."))


_REPO = _find_repo_root(_HERE)
CT_PATH = os.path.join(_REPO, "data", "lits17_sample", "volume-0.nii")
SEG_PATH = os.path.join(_REPO, "data", "lits17_sample", "segmentation-0.nii")

OUT_DIR = os.path.join(_REPO, "docs", "book", "figures", "ch21e")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Scenario (BOILING histotripsy: millisecond shock-heating pulses) ─────────
F0 = 1.0e6
TAU_MAX_S = 20e-3           # max therapy pulse duration [s] (boiling histotripsy
                            # uses ~1–20 ms shock-heating pulses; Khokhlova 2015)
PULSE_DURATION_S = TAU_MAX_S  # nominal pulse width for the schedule timeline
PRF_HZ = 10.0              # BH pulse-repetition frequency [Hz] (low — lets heat
                            # diffuse between pulses; inter-pulse 100 ms)
P_DRIVE_PA = 95e6           # representative delivered focal peak (field-shape demos)
P_MAX_PA = 130e6            # transducer source-pressure ceiling (100% power)
P_MIN_PA = 30e6             # minimum useful focal drive (shock/boiling floor)
P_TARGET_FOCAL_PA = 55e6    # prescribed focal peak we CONTROL toward (interior spots)
CONFORMAL_PRESSURE_TAPER = True   # also reduce focal pressure (not just pulse ms) near
#   the PTV/OAR boundary, so the boiling lesion is conformally tapered by BOTH power and
#   pulse duration; the floor is P_MIN/P_TARGET (shock-formation limit).
FOCAL_DEPTH_M = 0.08        # nominal focal depth in liver (shock-parameter path)
COVERAGE_TARGET = 0.95
REGION_MOVE_S = 0.3
# Hybrid firing: fired sub-spots are clustered into compact groups of this many spots,
# interleaved WITHIN each group (so a spot rests group_size/PRF between its pulses ≫
# residual-gas τ_d → no self-shielding) and the group is COMPLETED before the next.
# group_size · (1/PRF) must exceed τ_d (≈0.14 s): 12·0.1 s = 1.2 s gives ample margin
# while keeping per-group completion fast (~group_size·n_rep/PRF ≈ 10 s) for real-time
# region-by-region control. Larger → closer to global interleave; smaller → sequential.
HYBRID_GROUP_SIZE = 12
# Peak focal residual void fraction left immediately after a boiling-histotripsy
# pulse (surviving bubble-cloud remnant), scaled by the pulse's normalised inertial
# (broadband) emission. It then dissolves with Epstein–Plesset τ_d before the next
# visit; whether it shields the next pulse depends on the revisit interval vs τ_d
# (the interleaved-vs-sequential distinction). Bader/Duryea residual-cloud scale.
RESIDUAL_VOID_PEAK = 1.0e-4
VOID_FRACTION_MAX = 1.0e-2          # physical clip on transient focal void fraction
# Lesion-memory cavitation susceptibility (kwavers-physics lacuna model): tissue
# already fractionated by earlier pulses cavitates more readily. τ_LACUNA is the
# gas-evolution time over which dissolved gas diffuses into the liquefied void to
# form a lacuna (∼a²/2Dζ for a sub-mm cavity → ∼minutes intra-procedure; the
# delayed term is why re-treatment days later cavitates far more than the first
# pass). K_IMMEDIATE = prompt residual-nuclei threshold drop; K_LACUNA = the larger
# delayed gas-cavity enhancement.
TAU_LACUNA_S = 120.0
K_LESION_IMMEDIATE = 0.5
K_LACUNA = 4.0
# 3-D lacuna void-fraction field (coupled into propagation via the Rust
# ResidualGasField → Wood sound-speed + Commander–Prosperetti attenuation). The
# persistent gas cavity grows in fractionated tissue (lacuna_void_fraction) up to
# BETA_LACUNA_MAX and, as a strong gas/tissue impedance step, shields and aberrates
# any later pulse whose beam path crosses it (the "can't treat behind a lacuna").
BETA_LACUNA_MAX = 5.0e-3
LACUNA_RADIUS_M = 30.0e-6           # residual gas-pocket radius in the lacuna
C_GAS_M_S, RHO_GAS_KG_M3 = 343.0, 1.2
# Real-time B-mode monitoring (genuine receive-data → DAS reconstruction). A linear
# imaging array images the focal (lateral × SI) plane; the acoustic reflectivity
# evolves with the lesion — fractionated tissue turns hypoechoic, the residual-gas
# lacuna turns hyperechoic — so the reconstructed B-mode shows lesion formation.
IMAGING_F0_HZ = 5.0e6              # higher centre frequency → resolves the lesion shape
IMAGING_FS_HZ = 8.0 * IMAGING_F0_HZ
IMAGING_FRAC_BW = 0.7
IMAGING_N_ELEM = 96
BMODE_DR_DB = 45.0
IMAGING_UPSAMPLE = 3               # finer scatterer grid → anatomical speckle texture
R_LESION_HYPO = 0.92               # fractionated/liquefied tissue → hypoechoic (dark)
R_GAS_BRIGHT = 0.25                # residual-gas adds only sparse bright speckle
# Cell-kill dose–response (Weibull survival; kwavers-physics histotripsy_kill_fraction):
# kill = 1 − exp(−(D/d0)^k), the cumulative cell-survival form underlying radiobiology's
# biologically-effective dose — but the mechanism is MECHANICAL FRACTIONATION (cavitation
# liquefies tissue to acellular homogenate), not thermal ablation. Cumulative cavitation
# dose D ∝ p_focal³·pulses; d0 = characteristic dose (≈63 % kill) as a fraction of the peak
# per-voxel dose. The conformal pressure taper gives a radial kill gradient reported as
# iso-lethal LD25/LD50/LD75/LD100 levels (cell-kill %), the histotripsy analog of
# radiotherapy isodose contours. The sub-LD25 periphery is the viable, immune-primed zone.
KILL_D0_FRAC = 0.45
KILL_WEIBULL_K = 2.5
LD_LEVELS = (0.25, 0.50, 0.75, 0.99)   # LD25, LD50, LD75, LD100 (≈99 % kill, asymptotic)
T_BODY_C, T_BOIL_C = 37.0, 100.0
DELTA_T_BOIL = T_BOIL_C - T_BODY_C


def _reference_dose(dose_flat: np.ndarray, target_idx: np.ndarray) -> float:
    """Robust reference dose for the Weibull characteristic-dose (d0) calibration.

    Returns the MEDIAN of the non-zero cumulative dose over the fully-treated
    target voxels (the GTV core) — the dose a well-treated core voxel actually
    receives. This is robust to ellipsoid-overlap hot-spots, unlike the global
    peak: keying d0 to the peak would inflate the threshold and make the ablated
    core read far below its true ≈LD100 kill. Falls back to the global peak only
    when the target carries no dose.
    """
    if target_idx.size:
        td = dose_flat[target_idx]
        td = td[td > 0.0]
        if td.size:
            return float(np.median(td))
    return float(dose_flat.max())

# ── Tissue characterization from the kwavers-domain database ─────────────────
_C_LIVER, _RHO_LIVER, _ALPHA_DB, _BA, _Z_LIVER_MRAYL = kw.tissue_properties("liver")
_C_FAT, _RHO_FAT, _, _, _Z_FAT_MRAYL = kw.tissue_properties("fat")
C_LIVER, RHO_LIVER = _C_LIVER, _RHO_LIVER
Z_LIVER, Z_FAT = _Z_LIVER_MRAYL * 1e6, _Z_FAT_MRAYL * 1e6
# α at 1 MHz [Np/m] = α₀[dB/cm/MHz^y]·1^y · (ln10/20)·100.
ALPHA_LIVER_NP_M = _ALPHA_DB * (np.log(10.0) / 20.0) * 100.0
BETA_LIVER = 1.0 + _BA / 2.0                       # coefficient of nonlinearity
CP_LIVER, K_LIVER, _ = kw.tissue_thermal_properties("liver")   # [J/kgK], [W/mK]
YIELD_STRESS_PA, P_T_1MHZ_PA, _T_SLOPE, SIGMA_T_PA = kw.histotripsy_tissue_properties("liver")
R0_NUC = 3.0e-6
P0 = 101_325.0

MEDIUM = BubbleMedium(rho=RHO_LIVER, sigma=0.056, gamma=1.4, mu=2.0e-3, pv=2.3e3,
                      c_l=C_LIVER, p0=P0)


# Therapy transducer geometry — used identically by the analytic focal_fwhm AND
# the PSTD-nonlinear focused-bowl sim so the two field models are comparable.
TRANSDUCER_ROC_M = 30e-3
TRANSDUCER_DIAM_M = 30e-3
TRANSDUCER_PPW = 6          # points-per-wavelength for the PSTD focal sim
THERAPY_BOWL_RINGS = 10
THERAPY_BOWL_ELEMS_PER_RING = 24


def focal_fwhm(f0):
    lam = C_LIVER / f0
    fnum = TRANSDUCER_ROC_M / TRANSDUCER_DIAM_M        # F-number = ROC / aperture
    return 1.41 * lam * fnum, 7.0 * lam * fnum ** 2


FWHM_LAT, FWHM_AX = focal_fwhm(F0)         # Penttinen/O'Neil focal-spot size [m]
SIGMA_LAT = FWHM_LAT / 2.35482             # Gaussian focal-field σ (−6 dB → FWHM)
SIGMA_AX = FWHM_AX / 2.35482
NMAX_PULSES = 400                          # per-spot pulse cap


def _therapy_bowl_elements():
    cache = getattr(_therapy_bowl_elements, "_cache", None)
    if cache is None:
        cache = np.asarray(kw.focused_bowl_element_positions_3d(
            THERAPY_BOWL_RINGS, THERAPY_BOWL_ELEMS_PER_RING,
            0.5 * TRANSDUCER_DIAM_M, TRANSDUCER_ROC_M, FOCAL_DEPTH_M), dtype=float)
        _therapy_bowl_elements._cache = cache
    return cache


def _steered_transmit_profile(focus_xyz, p_spot, r):
    elem = _therapy_bowl_elements()
    focus = np.ascontiguousarray(focus_xyz, dtype=float)
    delays = np.asarray(kw.delay_law_focus_3d(elem, focus, C_LIVER), dtype=float)
    weights = np.ones(elem.shape[0], dtype=float)
    pts = np.column_stack([
        np.full_like(r, focus[0]),
        focus[1] + r,
        np.full_like(r, focus[2]),
    ])
    p = np.asarray(kw.steered_aperture_pressure_3d(
        np.ascontiguousarray(pts), elem, weights, delays, focus,
        F0, C_LIVER, ALPHA_LIVER_NP_M, float(p_spot)), dtype=float)
    return p / max(float(p_spot), 1e-30)


def simulate_boiling_lesion(p_spot, allowed, focus_xyz, field_b=None):
    """Simulate the boiling-histotripsy lesion extent and the conformal pulse.

    ``field_b(r)`` is the normalised transverse focal profile (default: analytic
    Penttinen Gaussian); pass the PSTD-nonlinear profile for the comparison.

    The focal pressure field is the analytic focused-bowl Gaussian (Penttinen
    1976) `p(r) = p_spot·exp(−r²/2σ²)`. The shocked wave deposits heat at the
    genuine shock-enhanced rate (kwavers `shock_heat_source_density`, with the
    Goldberg shock parameter σ = focal_depth / `shock_formation_distance`):
    ```
       Q(r) = gain(σ)·α·p(r)²/(ρc)        [W/m³]
    ```
    Boiling occurs at radius r once the focus reaches 100 °C within the pulse:
    ```
       t_boil(r) = ρ·C_p·(T_boil−T_body) / Q(r)
    ```
    so a pulse of duration τ liquefies out to the radius where `t_boil(r) ≤ τ`.
    The lesion extent is therefore the SIMULATED boiling region, and the
    conformal pulse duration `τ = t_boil(r_les)` with `r_les = min(allowed,
    natural boiling extent)` — shortening the millisecond pulse at the periphery
    genuinely shrinks the boiling lesion.

    Returns ``{pulses, les_lat, les_ax, pulse_ms}`` or ``None`` if the focus does
    not reach boiling within ``TAU_MAX_S``.
    """
    r = np.linspace(0.0, 2.0 * FWHM_LAT, 240)
    B = field_b(r) if field_b is not None else _steered_transmit_profile(focus_xyz, p_spot, r)
    plan = kw.boiling_lesion_from_pressure_profile(
        np.ascontiguousarray(r), np.ascontiguousarray(B), float(p_spot),
        FOCAL_DEPTH_M, F0, C_LIVER, RHO_LIVER, BETA_LIVER, ALPHA_LIVER_NP_M,
        CP_LIVER, DELTA_T_BOIL, TAU_MAX_S, SIGMA_AX / SIGMA_LAT, float(allowed),
        COVERAGE_TARGET)
    if plan is None:
        return None
    pulses, les_lat, les_ax, pulse_ms = plan
    return {"pulses": int(pulses), "les_lat": float(les_lat),
            "les_ax": float(les_ax), "pulse_ms": float(pulse_ms)}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry on the real CT: tumour mask, OAR, sonication-region / sub-spot grids
# ─────────────────────────────────────────────────────────────────────────────
def _load_ct_tumour_ritk(target_dx_m=1.5e-3):
    """Load the liver CT + tumour segmentation via the kwavers RITK NIfTI reader
    (``kw.load_ct_nifti`` → ritk-io), select the largest HCC focus, crop around
    it, and resample to an isotropic grid. No nibabel — medical-image I/O lives
    in the Rust core. Returns (ct_hu, tumour_mask, info)."""
    print(f"[ch21e] RITK-loading CT:  {CT_PATH}")
    ct_raw, sp = kw.load_ct_nifti(CT_PATH)
    seg_raw, _ = kw.load_ct_nifti(SEG_PATH)
    ct_raw = np.asarray(ct_raw, dtype=np.float32)
    seg_raw = np.asarray(seg_raw, dtype=np.float32)
    print(f"  RITK volume shape={ct_raw.shape}, voxel mm={tuple(round(s,3) for s in sp)}")

    # Tumour = label 2; keep the largest connected focus (multifocal HCC).
    tum = seg_raw > 1.5
    cc, n = label(tum)
    if n == 0:
        raise SystemExit("segmentation has no tumour (label-2) voxels")
    sizes = np.bincount(cc.ravel()); sizes[0] = 0
    focus = cc == int(np.argmax(sizes))
    coords = np.argwhere(focus)
    mlat = int(round(70e-3 / (sp[0] * 1e-3)))
    mele = int(round(70e-3 / (sp[1] * 1e-3)))
    msi = int(round(60e-3 / (sp[2] * 1e-3)))
    lo = np.maximum(coords.min(0) - [mlat, mele, msi], 0)
    hi = np.minimum(coords.max(0) + [mlat, mele, msi] + 1, ct_raw.shape)
    ct_c = ct_raw[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    seg_c = focus[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)

    target_mm = target_dx_m * 1e3
    fac = tuple(s / target_mm for s in sp)
    ct_hu = zoom(ct_c, fac, order=1, prefilter=False).astype(np.float32)
    tumour = zoom(seg_c, fac, order=0, prefilter=False) > 0.5
    if not tumour.any():
        raise SystemExit("tumour mask vanished after resampling")
    nx, ny, nz = ct_hu.shape
    centroid = np.argwhere(tumour).mean(0).astype(int)
    r_eq = ((tumour.sum() * target_dx_m ** 3) * 3.0 / (4.0 * np.pi)) ** (1 / 3)
    print(f"  resampled {ct_hu.shape} @ {target_mm:.2f} mm; tumour {int(tumour.sum())} vox, "
          f"r_eq={r_eq*1e3:.1f} mm")
    info = {
        "dx": target_dx_m, "shape": ct_hu.shape,
        "x_axis": np.arange(nx) * target_dx_m,
        "y_axis": (np.arange(ny) - ny / 2) * target_dx_m,
        "z_axis": (np.arange(nz) - nz / 2) * target_dx_m,
        "focus_idx": tuple(int(c) for c in centroid),
    }
    return ct_hu, tumour, info


def build_geometry():
    ct, tumour, info = _load_ct_tumour_ritk(target_dx_m=1.5e-3)
    dx = info["dx"]

    def vox(m):
        return max(int(round(m / dx)), 1)

    # Radiotherapy-style nested planning volumes (treat the PTV to fractionate the GTV
    # with margins): GTV = segmented tumour; CTV = GTV ⊕ 3 mm (microscopic
    # disease); PTV = CTV ⊕ 3 mm (motion / setup uncertainty).
    gtv = tumour
    ctv = binary_dilation(gtv, iterations=vox(3e-3))
    ptv = binary_dilation(ctv, iterations=vox(3e-3))
    coords = np.argwhere(ptv)
    cmin, cmax = coords.min(0), coords.max(0)
    fx = info["focus_idx"][0]                       # beam-axis (depth) focal slice

    # OAR: a sensitive vessel just lateral (+y) to the PTV, within the liver.
    oar_iy = min(int(np.argwhere(ptv)[:, 1].max()) + vox(3e-3), tumour.shape[1] - 1)
    safety_margin = 2.0e-3

    fwhm_lat, fwhm_ax = focal_fwhm(F0)
    les_lat = 0.7 * fwhm_lat
    les_ax = 0.4 * fwhm_ax

    # Distance from each PTV voxel to the PTV boundary [m] — the room a focal
    # lesion has before it would extend beyond the planned volume. Peripheral
    # spots use this to shorten their pulse (shrink the lesion) for conformality.
    edt_ptv_m = distance_transform_edt(ptv) * dx

    # Sonication-region lattice tiling the 3-D PTV, with spacing set by the LESION
    # FWHM so the focal lesions tile the whole volume across every image slice:
    #   * beam-axis / depth (axis 0): spacing = axial lesion extent les_ax (the focus
    #     is elongated along the beam, so one row of spots covers a depth band);
    #   * lateral (axis 1) and SI / image-slice (axis 2): spacing = lateral lesion
    #     diameter (2·les_lat), with electronic sub-spots stepping by the lesion radius
    #     so adjacent lesions overlap — covering multiple image slices in elevation.
    so = max(vox(les_lat), 1)              # in-plane sub-spot step ≈ lateral lesion radius
    rx_step = max(vox(les_ax), 1)          # depth spacing ≈ axial lesion extent
    rpl = max(2 * so, 1)                    # lateral / SI region spacing ≈ lesion diameter
    rx = range(cmin[0], cmax[0] + 1, rx_step)
    ry = range(cmin[1], cmax[1] + 1, rpl)
    rz = range(cmin[2], cmax[2] + 1, rpl)
    region_centers = [(ix, iy, iz) for iz in rz for iy in ry for ix in rx
                      if ptv[ix, iy, iz] or _near_tumour(ptv, ix, iy, iz, vox(2e-3))]
    # Fire regions in a multi-pass interleaved-lattice order (spread for the whole
    # treatment, not just the first half), so consecutive sonications stay far
    # apart — residual bubbles dissolve / heat diffuses before nearby tissue.
    region_centers = [region_centers[i] for i in _interleaved_lattice_order(region_centers, rpl, rpl)]
    # Electronic sub-spot offsets within a region (voxels). The focal spot is a
    # narrow TRANSVERSE dot elongated along the beam (depth, axis 0), so the raster
    # steps in the transverse lateral (axis 1) × SI (axis 2) plane; the beam-axis
    # extent is covered by the elongated focus. Interleave them too.
    subspot_offsets = [(0, oy, oz) for oz in (-so, 0, so) for oy in (-so, 0, so)]
    subspot_offsets = [subspot_offsets[i] for i in _farthest_point_order(subspot_offsets)]

    return {
        "ct": ct, "info": info, "dx": dx, "fx": fx,
        "tumour": tumour, "gtv": gtv, "ctv": ctv, "ptv": ptv,
        "oar_iy": oar_iy, "safety_margin": safety_margin, "edt_ptv_m": edt_ptv_m,
        "les_lat": les_lat, "les_ax": les_ax, "region_centers": region_centers,
        "subspot_offsets": subspot_offsets, "cmin": cmin, "cmax": cmax,
    }


def _interleaved_lattice_order(centers, rp, rpz, stride=2):
    """Multi-pass interleaved firing order for lattice-placed sonication regions.

    Regions are coloured into ``stride³`` interleaved sub-lattices (by their
    lattice coordinate mod `stride`); each sub-lattice is a SPARSE set spread
    across the whole tumour. Firing colour-by-colour (each colour ordered
    farthest-point) keeps consecutive sonications far apart for the ENTIRE
    treatment — unlike greedy farthest-point, which degenerates to gap-filling
    (locally sequential) in its second half. Each pass dots the whole volume; the
    next pass fills the interstitials — so a spot's neighbours are never treated
    back-to-back (residual bubbles dissolve / heat diffuses between visits)."""
    centers = list(centers)
    groups = {}
    for i, c in enumerate(centers):
        key = ((c[0] // rp) % stride, (c[1] // rp) % stride, (c[2] // rpz) % stride)
        groups.setdefault(key, []).append(i)
    order = []
    for key in sorted(groups):
        idxs = groups[key]
        fp = _farthest_point_order([centers[i] for i in idxs])
        order.extend(idxs[j] for j in fp)
    return order


def _farthest_point_order(points):
    """Greedy farthest-point ordering: start near a corner, then repeatedly pick
    the point farthest from all already-chosen. Consecutive entries are spatially
    distant (top-left → bottom-right → …), so a spot is never re-sonicated — nor
    its neighbour treated — before its residual bubble cloud dissolves and heat
    diffuses (avoids the unequal cavitation of a sequential raster)."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n <= 2:
        return list(range(n))
    start = int(np.argmin(pts.sum(axis=1)))           # a corner (min coord sum)
    order = [start]
    mind = np.linalg.norm(pts - pts[start], axis=1)
    for _ in range(n - 1):
        mind[order] = -np.inf
        nxt = int(np.argmax(mind))
        order.append(nxt)
        mind = np.minimum(mind, np.linalg.norm(pts - pts[nxt], axis=1))
    return order


def _near_tumour(mask, ix, iy, iz, r):
    sl = mask[max(0, ix - r):ix + r + 1, max(0, iy - r):iy + r + 1,
              max(0, iz - r):iz + r + 1]
    return bool(sl.any())


def _impedance_map(ct_hu):
    """Per-voxel specific acoustic impedance Z = ρ·c from the CT. Density from the
    standard Hounsfield relation ρ ≈ 1000·(1 + HU/1000) [kg/m³] (data preparation);
    the cavitation-relevant interface physics itself is the Rust
    `interface_pressure_enhancement`. Real anatomical boundaries (vessels, fat,
    parenchyma) therefore drive the interface cavitation enhancement."""
    rho = np.clip(1000.0 * (1.0 + np.asarray(ct_hu, float) / 1000.0), 5.0, 1200.0)
    return rho * C_LIVER


def _spot_interface_enhancement(zmap, ix, iy, iz):
    """Interface peak-pressure enhancement at a focal voxel: the largest-contrast
    6-neighbour impedance step evaluated by the Rust reflection law."""
    nx, ny, nz = zmap.shape
    zc = float(zmap[ix, iy, iz])
    nbz = [zmap[max(ix - 1, 0), iy, iz], zmap[min(ix + 1, nx - 1), iy, iz],
           zmap[ix, max(iy - 1, 0), iz], zmap[ix, min(iy + 1, ny - 1), iz],
           zmap[ix, iy, max(iz - 1, 0)], zmap[ix, iy, min(iz + 1, nz - 1)]]
    znb = max((float(z) for z in nbz), key=lambda z: abs(z - zc))
    return float(kw.interface_pressure_enhancement(zc, znb))


def _lesion_time_fields(geom, plan):
    """Per-voxel lesion-onset time t_start(x) and pulse-train duration over which
    fractionation completes. Used to grow the 3-D lacuna void field."""
    dx = geom["dx"]; shape = geom["tumour"].shape
    t_start = np.full(shape, np.inf); t_done = np.zeros(shape)
    for s in plan["spots"]:
        ix, iy, iz = s["idx"]; ll, la = s["les_lat"], s["les_ax"]
        tmp = np.zeros(shape, dtype=bool)
        _mark_ellipsoid(tmp, ix, iy, iz, ll, la, dx,
                        int(np.ceil(ll / dx)) + 1, int(np.ceil(la / dx)) + 1)
        ts = float(s.get("t_start", s.get("t_done", 0.0)))
        td = max(float(s.get("t_done", ts)), ts + 1e-9)
        upd = tmp & (ts < t_start)
        t_start[upd] = ts; t_done[upd] = td
    return t_start, t_done


def _lacuna_void_field(t, t_start, t_done):
    """3-D lacuna void fraction β(x,t) at treatment time t — the vectorised form of
    the Rust `lacuna_void_fraction`: β = β_max·f·(1−exp(−(t−t_start)/τ)) in tissue
    fractionated by time t (f = fractionation progress), zero elsewhere."""
    with np.errstate(invalid="ignore"):
        frac = np.clip((t - t_start) / np.maximum(t_done - t_start, 1e-9), 0.0, 1.0)
        age = np.where(np.isfinite(t_start), np.maximum(t - t_start, 0.0), 0.0)
        beta = np.vectorize(kw.lacuna_void_fraction)(
            frac, age, TAU_LACUNA_S, BETA_LACUNA_MAX)
    return np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)


def _anatomical_echogenicity(geom, fx, y0, y1, z0, z1):
    """Baseline acoustic reflectivity of the focal slice from the CT anatomy: a bulk
    tissue-backscatter term (HU-mapped: parenchyma mid, vessels/bile dark, fat/capsule
    bright) plus a specular boundary term (impedance gradient). Gives the B-mode its
    anatomical structure — vessels, capsule, tumour boundary — not a featureless field."""
    ct_sl = geom["ct"][fx, y0:y1, z0:z1].astype(float)
    hu = np.clip(ct_sl, -150.0, 200.0)
    bulk = (hu + 150.0) / 350.0                              # 0..1 tissue echogenicity
    zsl = _impedance_map(geom["ct"])[fx, y0:y1, z0:z1]
    gy, gz = np.gradient(zsl)
    spec = np.hypot(gy, gz); m = float(spec.max())
    spec = spec / m if m > 0 else spec                       # specular boundaries
    return 0.2 + 0.45 * bulk + 0.55 * spec


def _reflectivity_slice(geom, fx, t, t_start, t_done, y0, y1, z0, z1, echo=None):
    """Acoustic reflectivity of the focal slice at treatment time t = anatomical
    echogenicity modulated by lesion state: fractionated/liquefied tissue turns
    HYPOechoic (the histotripsy lesion is anechoic), and residual-gas pockets add
    sparse HYPERechoic speckle. The lesion therefore appears as the tumour-shaped dark
    zone developing within the liver anatomy — not a generic bright sphere."""
    if echo is None:
        echo = _anatomical_echogenicity(geom, fx, y0, y1, z0, z1)
    ts = t_start[fx, y0:y1, z0:z1]; td = t_done[fx, y0:y1, z0:z1]
    with np.errstate(invalid="ignore"):
        f = np.clip((t - ts) / np.maximum(td - ts, 1e-9), 0.0, 1.0)
    f = np.nan_to_num(f)
    beta = _lacuna_void_field(t, t_start, t_done)[fx, y0:y1, z0:z1]
    refl = echo * (1.0 - R_LESION_HYPO * f) + R_GAS_BRIGHT * echo * (beta / BETA_LACUNA_MAX)
    return np.clip(refl, 0.0, None)


def reconstruct_bmode_frames(geom, plan, times, seed=7):
    """Real-time B-mode of the focal plane at each `times` sample, reconstructed from
    GENUINE simulated receive data: the evolving reflectivity is sampled as point
    scatterers, a linear imaging array's channel RF is synthesised
    (`kw.simulate_receive_rf`, first-Born synthetic aperture), and delay-and-sum
    (`kw.beamform_image_delay_and_sum`) reconstructs the image. Envelope + log
    compression give the displayed B-mode (dB)."""
    fx = geom["fx"]; dx = geom["dx"]; info = geom["info"]
    coords = np.argwhere(geom["ptv"]); cmin, cmax = coords.min(0), coords.max(0)
    pad = int(round(8e-3 / dx))
    y0 = max(0, cmin[1] - pad); y1 = min(geom["ptv"].shape[1], cmax[1] + pad)
    z0 = max(0, cmin[2] - pad); z1 = min(geom["ptv"].shape[2], cmax[2] + pad)
    y_phys = info["y_axis"][y0:y1]; z_phys = info["z_axis"][z0:z1]
    t_start, t_done = _lesion_time_fields(geom, plan)
    ny, nz = y1 - y0, z1 - z0
    fy = np.linspace(y_phys[0], y_phys[-1], ny * IMAGING_UPSAMPLE)   # depth (lateral axis)
    fz = np.linspace(z_phys[0], z_phys[-1], nz * IMAGING_UPSAMPLE)   # lateral (SI axis)
    yy, zz = np.meshgrid(fy, fz, indexing="ij")
    scat_pos = np.column_stack([yy.ravel(), zz.ravel(), np.zeros(yy.size)])
    grid_pts = scat_pos                                              # recon at scatterer grid
    speckle = np.random.default_rng(seed).standard_normal(yy.shape)  # static tissue speckle
    y_arr = float(fy[0] - 4e-3)                                      # array sits proximal
    ze = np.linspace(float(fz[0]), float(fz[-1]), IMAGING_N_ELEM)
    elem_pos = np.column_stack([np.full(IMAGING_N_ELEM, y_arr), ze, np.zeros(IMAGING_N_ELEM)])
    maxd = float(np.hypot(fy[-1] - y_arr, fz[-1] - fz[0]))
    n_samples = int(maxd / C_LIVER * IMAGING_FS_HZ) + 128
    echo = _anatomical_echogenicity(geom, fx, y0, y1, z0, z1)        # static anatomy
    envelopes = []
    for t in times:
        refl = _reflectivity_slice(geom, fx, float(t), t_start, t_done, y0, y1, z0, z1, echo=echo)
        refl_f = zoom(refl, (yy.shape[0] / refl.shape[0], yy.shape[1] / refl.shape[1]), order=1)
        amp = np.ascontiguousarray((refl_f * speckle).ravel())
        rf = np.asarray(kw.simulate_receive_rf(scat_pos, amp, elem_pos, C_LIVER,
                                               IMAGING_FS_HZ, IMAGING_F0_HZ, n_samples, IMAGING_FRAC_BW))
        img = np.asarray(kw.beamform_image_delay_and_sum(
            rf, elem_pos, grid_pts, C_LIVER, IMAGING_FS_HZ, apodization="hann")).reshape(yy.shape)
        # Hilbert-transform envelope per axial line (§9.1.3) from the Rust core.
        env = np.empty_like(img)
        for col in range(img.shape[1]):
            env[:, col] = np.asarray(kw.bmode_envelope(np.ascontiguousarray(img[:, col])))
        envelopes.append(np.ascontiguousarray(env.ravel()))
    baseline = envelopes[0] if envelopes else np.ones(1)
    reference = float(np.max(baseline)) + 1e-30
    frames = [np.asarray(kw.bmode_db_fixed_reference(e, reference, -BMODE_DR_DB), float).reshape(yy.shape)
              for e in envelopes]
    delta_frames = [np.asarray(kw.delta_bmode_db(e, baseline, 1e-12), float).reshape(yy.shape)
                    for e in envelopes]
    extent = [z_phys[0] * 1e3, z_phys[-1] * 1e3, y_phys[-1] * 1e3, y_phys[0] * 1e3]
    return {"frames": frames, "delta_frames": delta_frames, "extent": extent,
            "times": np.asarray(times)}


def _conformal_target_pressure(allowed):
    """Target focal peak pressure for a spot with `allowed` room to the PTV/OAR
    boundary. Full target in the interior; linearly tapered to the P_MIN shock floor
    within ~2 lateral-FWHM of the boundary (lesion-shape control via pressure)."""
    if not CONFORMAL_PRESSURE_TAPER:
        return P_TARGET_FOCAL_PA
    floor = P_MIN_PA / P_TARGET_FOCAL_PA
    room = min(float(allowed) / max(2.0 * FWHM_LAT, 1e-9), 1.0)
    return P_TARGET_FOCAL_PA * (floor + (1.0 - floor) * room)


# ─────────────────────────────────────────────────────────────────────────────
# Treatment planning (genuine per-spot physics + volumetric coverage + schedule)
# ─────────────────────────────────────────────────────────────────────────────
def plan_treatment(geom, field_b=None):
    info, dx = geom["info"], geom["dx"]
    tumour = geom["tumour"]
    oar_y_m = info["y_axis"][geom["oar_iy"]]

    nx, ny, nz = tumour.shape
    lesioned = np.zeros_like(tumour, dtype=bool)
    edt = geom["edt_ptv_m"]
    zmap = _impedance_map(geom["ct"])     # CT-derived impedance for interface enhancement
    allowed_mask = geom["ptv"].copy()
    allowed_mask[:, geom["info"]["y_axis"] >= (oar_y_m - geom["safety_margin"]), :] = False
    allowed_flat = np.ascontiguousarray(allowed_mask.ravel())
    nx_m, ny_m, nz_m = allowed_mask.shape

    regions, spots = [], []
    for ridx, rc in enumerate(geom["region_centers"]):
        cx, cy, cz = rc
        reg = {"center": rc, "ridx": ridx, "spots": [], "n_fired": 0, "n_rep": 0}
        for off in geom["subspot_offsets"]:
            ix, iy, iz = cx + off[0], cy + off[1], cz + off[2]
            if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
                continue
            gy_m = info["y_axis"][iy]
            # Room the focal lesion has before extending beyond the PTV or
            # breaching the OAR safety margin.
            allowed = min(float(edt[ix, iy, iz]),
                          (oar_y_m - gy_m) - geom["safety_margin"])
            if allowed <= 0.0:
                reg["spots"].append({"idx": (ix, iy, iz), "fired": False, "pulses": 0})
                continue
            # Transverse electronic steering: lateral (axis 1) × SI (axis 2).
            steer_lat, steer_ele = off[1] * dx, off[2] * dx
            path = info["x_axis"][ix]
            focus_xyz = (float(path), float(steer_lat), float(steer_ele))
            eps = kw.electronic_steering_efficiency(float(steer_lat), float(steer_ele),
                                                    F0, C_LIVER, True)
            deliver = kw.forward_delivery_fraction(
                float(eps), Z_FAT, Z_LIVER, ALPHA_LIVER_NP_M, float(path),
                0.0, F0, R0_NUC, C_LIVER, RHO_LIVER, MEDIUM.mu, P0, MEDIUM.gamma)
            # CONFORMAL PRESSURE TAPER: interior spots are driven to the full target
            # focal pressure; spots within ~2 lateral-FWHM of the PTV/OAR boundary get
            # a reduced target (down to the P_MIN shock floor) so the boiling lesion is
            # tapered by BOTH focal pressure and pulse duration — controlling the lesion
            # shape so it conforms to the planning volume without breaching the margin.
            p_target = _conformal_target_pressure(allowed)
            # POWER CONTROL: source power to reach the (tapered) target scales as
            # 1/efficiency, so deep/steered spots draw more power. The transducer ceiling
            # P_MAX caps it; capped spots fall short of target (genuine power variation).
            req_power = min(max(p_target / max(deliver, 1e-3), P_MIN_PA), P_MAX_PA)
            delivered = req_power * deliver          # focal pressure actually reached
            les = simulate_boiling_lesion(delivered, allowed, focus_xyz, field_b)
            if les is None:
                reg["spots"].append({"idx": (ix, iy, iz), "fired": False, "pulses": 0})
                continue
            if not kw.ellipsoid_respects_allowed_mask(
                allowed_flat, nx_m, ny_m, nz_m, ix, iy, iz,
                float(les["les_lat"]), float(les["les_ax"]), dx
            ):
                reg["spots"].append({"idx": (ix, iy, iz), "fired": False, "pulses": 0})
                continue
            s = {"idx": (ix, iy, iz), "ridx": ridx, "fired": True, "pulses": les["pulses"],
                 "p_spot": delivered, "delivered": delivered, "deliver": float(deliver),
                 "p_target": float(p_target), "power_pct": req_power / P_MAX_PA * 100.0,
                 "e_int": _spot_interface_enhancement(zmap, ix, iy, iz),
                 "yz": (iy, iz), "x": ix, "focus_xyz": focus_xyz,
                 "les_lat": les["les_lat"], "les_ax": les["les_ax"], "pulse_ms": les["pulse_ms"]}
            reg["spots"].append(s)
            spots.append(s)
            rl_s = int(np.ceil(les["les_lat"] / dx)) + 1
            rz_s = int(np.ceil(les["les_ax"] / dx)) + 1
            _mark_ellipsoid(lesioned, ix, iy, iz, les["les_lat"], les["les_ax"], dx, rl_s, rz_s)
        fired = [s for s in reg["spots"] if s["fired"]]
        reg["n_fired"] = len(fired)
        reg["n_rep"] = max((s["pulses"] for s in fired), default=0)
        regions.append(reg)

    cov = float(np.count_nonzero(lesioned & tumour) / max(np.count_nonzero(tumour), 1))

    # HYBRID interleaved-group pulse timeline. Fired sub-spots are clustered into
    # compact local GROUPS (~HYBRID_GROUP_SIZE); within a group the spots fire
    # round-robin across repetitions — so each spot rests group_size/PRF ≫ residual-gas
    # τ_d (no self-shielding) — and the group is COMPLETED before the next. Groups are
    # visited farthest-point (consecutive groups far apart). This balances the global
    # interleave (rests every spot but each spot finishes only at end of treatment)
    # against a sequential raster (fast but shielded): each local sub-volume fully
    # lesions in ~group_size·n_rep/PRF s → real-time, region-by-region control.
    inter = 1.0 / PRF_HZ
    groups = _hybrid_groups(spots, dx, HYBRID_GROUP_SIZE)
    onsets, region_id, schedule, t = [], [], [], 0.0
    for group in groups:
        gmax = max((s["pulses"] for s in group), default=0)
        for p in range(gmax):
            for s in group:
                if p < s["pulses"]:
                    onsets.append(t); region_id.append(s["ridx"]); schedule.append((t, s))
                    if p == 0:
                        s["t_start"] = t             # first pulse (lesion initiation)
                    s["t_done"] = t; t += inter      # last pulse (fractionation complete)
        t += REGION_MOVE_S                           # per-group settle / verify dwell
    onsets = np.asarray(onsets); region_id = np.asarray(region_id, dtype=int)
    n_active = len({s["ridx"] for s in spots})

    return {
        "regions": regions, "spots": spots, "lesioned": lesioned, "coverage": cov,
        "onsets": onsets, "region_id": region_id, "treatment_s": float(t),
        "schedule": schedule, "n_regions_active": n_active, "n_groups": len(groups),
        "oar_y_m": oar_y_m,
    }


def _morton3(x, y, z):
    """3-D Morton (Z-order) code interleaving 10-bit integer coords — gives a
    spatial-locality sort so consecutive entries are spatial neighbours."""
    def part(n):
        n = int(n) & 0x3FF
        n = (n | (n << 16)) & 0x030000FF
        n = (n | (n << 8)) & 0x0300F00F
        n = (n | (n << 4)) & 0x030C30C3
        n = (n | (n << 2)) & 0x09249249
        return n
    return part(x) | (part(y) << 1) | (part(z) << 2)


def _hybrid_groups(spots, dx, group_size):
    """Cluster fired sub-spots into compact groups of ~group_size by Morton (Z-order)
    locality, then order the groups farthest-point so consecutive groups are spatially
    separated. Each group is treated to completion with internal interleaving."""
    if not spots:
        return []
    keys = np.array([_morton3(s["idx"][0], s["idx"][1], s["idx"][2]) for s in spots])
    ordered = [spots[i] for i in np.argsort(keys, kind="stable")]
    g = max(int(group_size), 1)
    groups = [ordered[i:i + g] for i in range(0, len(ordered), g)]
    if len(groups) > 2:
        cent = np.array([[np.mean([s["idx"][a] for s in grp]) for a in range(3)]
                         for grp in groups]) * dx
        groups = [groups[i] for i in _farthest_point_order(cent)]
    return groups


def _boiling_time_profile(p_spot, field_b):
    """Return (r [m], t_boil(r) [s]) for a focal field B(r) at peak `p_spot` —
    the Rust shock-heating boiling-onset time vs radius (for comparison plots)."""
    r = np.linspace(0.0, 2.0 * FWHM_LAT, 200)
    B = field_b(r) if field_b is not None else np.exp(-r * r / (2.0 * SIGMA_LAT * SIGMA_LAT))
    t_boil = np.asarray(kw.boiling_time_profile_from_pressure(
        np.ascontiguousarray(B), float(p_spot), FOCAL_DEPTH_M, F0, C_LIVER,
        RHO_LIVER, BETA_LIVER, ALPHA_LIVER_NP_M, CP_LIVER, DELTA_T_BOIL), float)
    return r, t_boil


def figure_field_comparison(geom, fields, meta, plans):
    """Compare the analytic vs PSTD-nonlinear focal-field models and their effect
    on the boiling lesion: focal profiles, boiling-onset time, lesion-vs-pulse,
    and the resulting whole-tumour coverage / treatment time."""
    rgrid = np.linspace(0.0, 2.0 * FWHM_LAT, 200)
    p_demo = 0.55 * P_DRIVE_PA          # representative delivered focal peak
    colors = {"analytic": "#4aa3ff", "pstd_nonlinear": "#ff8c42"}
    labels = {"analytic": "analytic (Penttinen Gaussian)",
              "pstd_nonlinear": "PSTD nonlinear (simulated)"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    # (A) normalised focal profiles.
    a = axes[0, 0]
    for k, B in fields.items():
        prof = B(rgrid)
        fwhm = 2.0 * np.interp(0.5, prof[::-1], rgrid[::-1]) * 1e3 if prof.min() < 0.5 else np.nan
        a.plot(rgrid * 1e3, prof, color=colors[k], lw=2, label=f"{labels[k]}  (FWHM≈{fwhm:.1f} mm)")
    a.set_xlabel("transverse radius [mm]"); a.set_ylabel("normalised pressure")
    a.set_title("Focal field profile"); a.legend(fontsize=8); a.set_xlim(0, 4)

    # (B) boiling-onset time vs radius at the demo drive.
    a = axes[0, 1]
    for k, B in fields.items():
        r, tb = _boiling_time_profile(p_demo, B)
        a.plot(r * 1e3, tb * 1e3, color=colors[k], lw=2, label=labels[k])
    a.axhline(TAU_MAX_S * 1e3, color="k", ls="--", lw=1, label=f"τ_max {TAU_MAX_S*1e3:.0f} ms")
    a.set_xlabel("transverse radius [mm]"); a.set_ylabel("boiling-onset time [ms]")
    a.set_title(f"Boiling time t_boil(r) at {p_demo/1e6:.0f} MPa")
    a.set_xlim(0, 4); a.set_ylim(0, 3 * TAU_MAX_S * 1e3); a.legend(fontsize=8)

    # (C) conformal lesion radius vs pulse duration.
    a = axes[1, 0]
    taus = np.linspace(1e-3, TAU_MAX_S, 30)
    for k, B in fields.items():
        r, tb = _boiling_time_profile(p_demo, B)
        radii = [np.interp(t, tb, r, right=r[-1]) * 1e3 if np.any(tb <= t) else 0.0 for t in taus]
        a.plot(taus * 1e3, radii, color=colors[k], lw=2, label=labels[k])
    a.set_xlabel("pulse duration [ms]"); a.set_ylabel("lesion radius [mm]")
    a.set_title(f"Lesion extent vs pulse duration at {p_demo/1e6:.0f} MPa"); a.legend(fontsize=8)

    # (D) whole-tumour outcome table.
    a = axes[1, 1]; a.axis("off")
    rows = [("metric", "analytic", "PSTD-nl")]
    pa, pp = plans["analytic"], plans["pstd_nonlinear"]
    ll_a = np.array([s["les_lat"] for s in pa["spots"]]) * 1e3
    ll_p = np.array([s["les_lat"] for s in pp["spots"]]) * 1e3
    rows += [
        ("GTV coverage [%]", f"{pa['coverage']*100:.1f}", f"{pp['coverage']*100:.1f}"),
        ("fired sub-spots", f"{len(pa['spots'])}", f"{len(pp['spots'])}"),
        ("total pulses", f"{pa['onsets'].size}", f"{pp['onsets'].size}"),
        ("treatment [s]", f"{pa['treatment_s']:.0f}", f"{pp['treatment_s']:.0f}"),
        ("mean lesion R [mm]", f"{ll_a.mean():.2f}", f"{ll_p.mean():.2f}"),
    ]
    tbl = a.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.scale(1, 1.6); tbl.set_fontsize(10)
    a.set_title("Whole-tumour outcome")
    note = (f"PSTD focal peak: +{meta['p_max_focus']/1e6:.0f} MPa (shock) / "
            f"−{meta['p_min_focus']/1e6:.0f} MPa (rarefaction)\n"
            f"→ asymmetry confirms nonlinear shock formation")
    a.text(0.5, 0.04, note, transform=a.transAxes, ha="center", fontsize=8, color="#444")

    fig.suptitle("ch21e — Focal-field model comparison: analytic vs PSTD-nonlinear "
                 "(boiling-histotripsy lesion)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fig23_focal_field_comparison")


def _mark_ellipsoid(vol, ix, iy, iz, les_lat, les_ax, dx, rl, rz):
    nx, ny, nz = vol.shape
    xa = np.arange(max(0, ix - rz), min(nx, ix + rz + 1))
    ya = np.arange(max(0, iy - rl), min(ny, iy + rl + 1))
    za = np.arange(max(0, iz - rl), min(nz, iz + rl + 1))
    XX, YY, ZZ = np.meshgrid(xa, ya, za, indexing="ij")
    ell = (((XX - ix) * dx / les_ax) ** 2 + ((YY - iy) * dx / les_lat) ** 2
           + ((ZZ - iz) * dx / les_lat) ** 2) <= 1.0
    vol[xa[0]:xa[-1] + 1, ya[0]:ya[-1] + 1, za[0]:za[-1] + 1] |= ell


def _pulse_icd(drive_pa, rng, n_bubbles=6):
    icds = []
    for _ in range(n_bubbles):
        r0 = float(rng.lognormal(np.log(1.5e-6), 0.4))
        _t, r, rdot, _e, _mc, _mm, _nc, _cv = kw.simulate_bubble_emission(
            r0, drive_pa, F0, 8.0, 4096, 5.0e-2,
            p0_pa=MEDIUM.p0, rho=MEDIUM.rho, c_liquid=MEDIUM.c_l, mu=MEDIUM.mu,
            sigma=MEDIUM.sigma, pv=MEDIUM.pv, gamma=MEDIUM.gamma)
        r = np.asarray(r, float); rdot = np.asarray(rdot, float)
        if r.size < 4 or not (np.all(np.isfinite(r)) and np.all(np.isfinite(rdot))):
            continue
        icds.append(float(kw.inertial_cavitation_dose(r, rdot, r0)))
    return float(np.mean(icds)) if icds else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Fig A: real pulse train
# ─────────────────────────────────────────────────────────────────────────────
def figure_pulsing_pattern(plan):
    onsets, region_id = plan["onsets"], plan["region_id"]
    sched = plan["schedule"]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))
    # Top: whole-treatment firing map — sub-spot lateral position vs time, coloured
    # by treatment time. The cloud stays vertically spread for the WHOLE duration
    # (global interleave), never collapsing to a sequential top→bottom sweep.
    if sched:
        ts = np.array([t for t, _ in sched])
        ylat = np.array([s["idx"][1] for _, s in sched])
        ax0.scatter(ts, ylat, c=ts, cmap="turbo", s=3, alpha=0.5)
    ax0.set_xlabel("treatment time [s]"); ax0.set_ylabel("sub-spot lateral index")
    ng = plan.get("n_groups", 0)
    ax0.set_title(f"Whole-tumour hybrid firing map — {ng} interleaved groups, "
                  f"{onsets.size} pulses, ③ Sonication Duration {plan['treatment_s']:.1f} s")
    ax0.set_xlim(0, max(plan["treatment_s"], 1e-9))
    # Bottom: ms boiling-histotripsy pulse pattern over the first group — each stem is
    # one fired pulse at its onset, height = boiling-onset-compensated pulse duration
    # [ms]; within the group consecutive pulses hop between its spots (internal
    # interleave), and the group completes before the next.
    K = min(60, len(sched))
    if K:
        on0 = np.array([sched[i][0] for i in range(K)])
        pm0 = np.array([sched[i][1]["pulse_ms"] for i in range(K)])
        pm = np.array([s["pulse_ms"] for s in plan["spots"]])
        pm_lo, pm_hi = float(pm.min()), float(pm.max())
        revisit_s = HYBRID_GROUP_SIZE / PRF_HZ          # within-group per-spot rest
        ax1.stem(on0, pm0, basefmt=" ", linefmt="#1f3c88", markerfmt="o")
        ax1.set_xlabel("treatment time [s]"); ax1.set_ylabel("pulse duration [ms]")
        ax1.set_title(f"Boiling-histotripsy pulsing (first {K} pulses) — "
                      f"① pulse duration {pm_lo:.1f}–{pm_hi:.1f} ms, "
                      f"② in-group rest {revisit_s:.1f} s ≫ residual-gas τ_d")
    fig.tight_layout(); _save(fig, "fig21_treatment_pipeline_pulsing")


# ─────────────────────────────────────────────────────────────────────────────
# Fig B: monitor (genuine per-pulse emission over the treatment)
# ─────────────────────────────────────────────────────────────────────────────
SPEC_F = np.linspace(0.0, 4.0e6, 240)


def _emission_tables(rng, p_grid):
    """Genuine emission (stable+broadband) and PSD spectrum vs delivered focal
    pressure, from microbubble-population Keller–Miksis sims (one per pressure).
    Returns (stable[p], broadband[p], spectra[p, SPEC_F])."""
    st = np.zeros(p_grid.size); br = np.zeros(p_grid.size)
    spec = np.zeros((p_grid.size, SPEC_F.size))
    for i, p in enumerate(p_grid):
        r = simulate_population_emission(kw, drive_pa=float(p), f0=F0, medium=MEDIUM,
                                         n_bubbles=14, rng=rng, n_cycles=8.0,
                                         n_out=4096, steps_per_cycle=1500)
        st[i] = r["stable"]; br[i] = r["broadband"]
        fr = np.asarray(r["freqs"]); ps = np.asarray(r["psd"])
        if fr.size > 4:
            spec[i] = np.interp(SPEC_F, fr, ps, left=0.0, right=0.0)
    return st, br, spec


def _gas_tables(path, cloud=3.0e-3):
    """β_focal → (forward gas transmission, two-way received fraction) tables."""
    bg = np.linspace(0.0, 2.0e-4, 200)
    fwd = np.array([float(np.exp(-kw.bubbly_cloud_attenuation(
        F0, b * cloud / path, R0_NUC, C_LIVER, RHO_LIVER, MEDIUM.mu, P0, MEDIUM.gamma) * path))
        for b in bg])
    recv = np.array([float(kw.received_signal_fraction(
        Z_FAT, Z_LIVER, ALPHA_LIVER_NP_M, path, b * cloud / path, F0, R0_NUC,
        C_LIVER, RHO_LIVER, MEDIUM.mu, P0, MEDIUM.gamma)) for b in bg])
    return bg, fwd, recv


def _pcd_receiver_positions(geom, n_elem=32):
    info = geom["info"]
    coords = np.argwhere(geom["ptv"])
    cmin, cmax = coords.min(0), coords.max(0)
    z0 = info["z_axis"][cmin[2]]
    z1 = info["z_axis"][cmax[2]]
    z_span = max(float(z1 - z0) + 12e-3, 12e-3)
    zc = 0.5 * float(z0 + z1)
    z = np.linspace(zc - 0.5 * z_span, zc + 0.5 * z_span, n_elem)
    return np.ascontiguousarray(np.column_stack([
        np.zeros(n_elem),
        np.full(n_elem, -10e-3),
        z,
    ]))


def _sequential_sequence(spots):
    """Comparison firing order: row-major over fired sub-spots, all repetitions at
    one sub-spot back-to-back (re-hit every 1/PRF — no rest, so residual gas from
    the previous pulse has not dissolved and shields the next)."""
    inter = 1.0 / PRF_HZ
    ordered = sorted(spots, key=lambda s: (s["idx"][0], s["idx"][1], s["idx"][2]))
    seq, t = [], 0.0
    for s in ordered:
        for _ in range(s["pulses"]):
            seq.append((t, s)); t += inter
    return seq


def _walk_sequence(seq, p_grid, emis_tot, emis_broad, spec_tab, bg, fwd_tab, recv_tab,
                   tau_d, cap, receiver_pos, lac=None):
    """Walk a fired-pulse sequence tracking per-spot residual gas (revisit-decay),
    closed-loop power control toward the target focal pressure, and the measured
    cavitation under forward+echo shielding. `lac` (optional) supplies the evolving
    3-D lacuna field coupling — per (spot, time-bin) forward transmission through the
    growing gas cavities and the local gas/tissue interface enhancement — so later
    pulses whose beam path crosses a lacuna are shielded and aberrated.

    Returns per-pulse records and, per spot, the achieved vs intended cavitation —
    their ratio is the delivery EFFICIENCY. The lacuna and interface terms enter
    both achieved and intended identically, so the ratio still isolates the residual
    bubble-cloud shielding (the interleaved-vs-sequential penalty)."""
    last = {}                       # spot key -> (beta, t_last) residual gas
    hist = {}                       # spot key -> (pulses_done, t_first) lesion memory
    ach = {}; intended = {}
    pmax = float(p_grid[-1])
    fwd0 = float(fwd_tab[0]); recv0 = float(recv_tab[0])    # unshielded (β=0) reference
    t_arr = np.empty(len(seq)); pwr = np.empty(len(seq)); sig = np.empty(len(seq))
    bet = np.empty(len(seq)); dlv = np.empty(len(seq))
    spectra = np.zeros((len(seq), spec_tab.shape[1]))
    n_bins = lac["edges"].size - 1 if lac is not None else 0
    for k, (t, s) in enumerate(seq):
        key = s["idx"]; base_eff = s["deliver"]; p_tgt = s.get("p_target", P_TARGET_FOCAL_PA)
        b_prev, t_last = last.get(key, (0.0, None))
        b = b_prev * np.exp(-(t - t_last) / tau_d) if t_last is not None else 0.0
        fwd_gas = float(np.interp(b, bg, fwd_tab))
        recv = float(np.interp(b, bg, recv_tab))
        # LACUNA FIELD: forward transmission through the growing 3-D gas cavities on
        # this spot's beam path, and the local gas/tissue interface enhancement.
        e_int = s.get("e_int", 1.0); fwd_lac = 1.0
        if lac is not None:
            si = lac["index"][key]
            kb = min(max(int(np.searchsorted(lac["edges"], t) - 1), 0), n_bins - 1)
            fwd_lac = float(lac["fwd"][si, kb])
            e_int = max(e_int, float(lac["e_dyn"][si, kb]))
        fwd_eff = base_eff * fwd_gas * fwd_lac
        req_power = min(max(p_tgt / max(fwd_eff, 1e-3), P_MIN_PA), P_MAX_PA)
        delivered = req_power * fwd_eff               # focal pressure actually reached
        # PROMPT LESION MEMORY: freshly fractionated tissue (this spot's prior passes)
        # carries residual nuclei that lower the cavitation threshold → more emission.
        n_done, t_first = hist.get(key, (0, None))
        f_self = n_done / max(s["pulses"], 1)
        suscept = 1.0 + K_LESION_IMMEDIATE * f_self
        d_eff = min(delivered * e_int, pmax)
        emis = float(np.interp(d_eff, p_grid, emis_tot)) * suscept
        broad = float(np.interp(d_eff, p_grid, emis_broad)) * suscept
        measured = emis * recv                        # sensor-recorded (two-way) signal
        source_psd = np.asarray(_spec_at(spec_tab, p_grid, d_eff), float) * suscept * recv
        channel_psd = np.asarray(kw.receiver_channel_psd_from_source(
            np.ascontiguousarray(source_psd),
            np.ascontiguousarray(s.get("focus_xyz", (0.0, 0.0, 0.0)), dtype=float),
            receiver_pos, ALPHA_LIVER_NP_M), float)
        spectra[k] = np.asarray(kw.integrate_channel_psd(channel_psd), float)
        # Unshielded reference: cloud-free (β=0) but SAME lacuna+interface+suscept, so
        # the efficiency ratio isolates the residual bubble-cloud shielding.
        fwd_eff0 = base_eff * fwd0 * fwd_lac
        deliv0 = min(max(p_tgt / max(fwd_eff0, 1e-3), P_MIN_PA), P_MAX_PA) * fwd_eff0
        measured0 = float(np.interp(min(deliv0 * e_int, pmax), p_grid, emis_tot)) * suscept * recv0
        ach[key] = ach.get(key, 0.0) + measured
        intended[key] = intended.get(key, 0.0) + measured0
        b = min(b + RESIDUAL_VOID_PEAK * (broad / cap), VOID_FRACTION_MAX)
        last[key] = (b, t)
        hist[key] = (n_done + 1, t_first if t_first is not None else t)
        t_arr[k] = t; pwr[k] = req_power / P_MAX_PA * 100.0
        sig[k] = measured; bet[k] = b; dlv[k] = d_eff
    eff = {k: ach[k] / intended[k] for k in ach if intended[k] > 0}
    return {"t": t_arr, "power_pct": pwr, "signal": sig, "beta": bet,
            "delivered": dlv, "spectra": spectra}, eff


def _coeff_var(values):
    v = np.asarray(list(values), float)
    return float(v.std() / v.mean()) if v.size and v.mean() > 0 else 0.0


def _evolve_lacuna_coupling(geom, plan, edges):
    """Evolve the 3-D lacuna void field over the treatment and, per (spot, time-bin),
    return the forward transmission through the gas cavities on each spot's beam path
    and the local gas/tissue interface enhancement — plus per-bin focal-slice void
    fraction for the animation. Uses the Rust ResidualGasField (Wood sound speed +
    Commander–Prosperetti attenuation) as the medium-coupling SSOT."""
    nx, ny, nz = geom["tumour"].shape
    dx = geom["dx"]; fx = geom["fx"]
    t_start, t_done = _lesion_time_fields(geom, plan)
    spots = plan["spots"]
    ijk = np.array([s["idx"] for s in spots], dtype=int)
    nb = edges.size - 1
    tc = 0.5 * (edges[:-1] + edges[1:])
    fwd = np.ones((len(spots), nb)); beta_loc = np.zeros((len(spots), nb))
    slices = []
    z_liver = RHO_LIVER * C_LIVER
    for kb, t in enumerate(tc):
        beta = _lacuna_void_field(t, t_start, t_done)
        rgf = kw.ResidualGasField(nx, ny, nz, LACUNA_RADIUS_M)
        rgf.deposit(beta)
        acp = np.asarray(rgf.attenuation_field(F0, C_LIVER, RHO_LIVER, MEDIUM.mu, P0, MEDIUM.gamma))
        apath = np.cumsum(acp, axis=0) * dx          # path-integrated attenuation along beam axis
        fwd[:, kb] = np.exp(-apath[ijk[:, 0], ijk[:, 1], ijk[:, 2]])
        beta_loc[:, kb] = beta[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        slices.append(beta[fx].copy())
    # Local gas/tissue interface enhancement from the lacuna Wood impedance (SSOT).
    c_wood = np.vectorize(kw.wood_sound_speed)(beta_loc, C_LIVER, RHO_LIVER, C_GAS_M_S, RHO_GAS_KG_M3)
    rho_mix = (1.0 - beta_loc) * RHO_LIVER + beta_loc * RHO_GAS_KG_M3
    z_lac = rho_mix * c_wood
    e_dyn = np.vectorize(kw.interface_pressure_enhancement)(z_liver, z_lac)
    index = {s["idx"]: i for i, s in enumerate(spots)}
    return {"edges": edges, "fwd": fwd, "e_dyn": e_dyn, "index": index,
            "beta_slices": slices, "peak_beta": float(max((b.max() for b in slices), default=0.0))}


def simulate_measured_sonication(plan, geom, n_pulses=80, seed=11):
    """Sensor-recorded cavitation over the treatment, driven by the GENUINE
    fired-pulse spot sequence: per-spot delivery efficiency sets the source power,
    residual gas from each spot's actual revisit interval shields the next visit, and
    a GROWING 3-D lacuna void field (Rust ResidualGasField → Wood/Commander–Prosperetti
    coupling) shields and aberrates later pulses whose beam path crosses it. Also
    walks the SEQUENTIAL order to quantify the residual-bubble cavitation inequality."""
    rng = np.random.default_rng(seed)
    tau_d = kw.epstein_plesset_dissolution_time(R0_NUC, 0.5)
    # Emission curve spans up to the interface-enhanced peak (delivered × interface
    # enhancement, which can exceed the conformal target near tissue boundaries).
    p_grid = np.linspace(P_MIN_PA, 1.7 * P_TARGET_FOCAL_PA, 10)
    emis_st, emis_br, spec_tab = _emission_tables(rng, p_grid)
    emis_tot = emis_st + emis_br
    cap = max(float(emis_br.max()), 1e-30)
    bg, fwd_tab, recv_tab = _gas_tables(FOCAL_DEPTH_M)

    treatment_s = max(plan["schedule"][-1][0], 1e-6) if plan["schedule"] else 1e-6
    edges = np.linspace(0.0, treatment_s, n_pulses + 1)
    tc = 0.5 * (edges[:-1] + edges[1:])
    lac = _evolve_lacuna_coupling(geom, plan, edges)   # 3-D lacuna field coupling
    receiver_pos = _pcd_receiver_positions(geom)

    seq_int = plan["schedule"]                       # canonical global interleaved timeline
    seq_seq = _sequential_sequence(plan["spots"])     # comparison: no-rest sequential raster
    rec_int, cav_int = _walk_sequence(seq_int, p_grid, emis_tot, emis_br, spec_tab, bg, fwd_tab,
                                      recv_tab, tau_d, cap, receiver_pos, lac=lac)
    _rec_seq, cav_seq = _walk_sequence(seq_seq, p_grid, emis_tot, emis_br, spec_tab, bg, fwd_tab,
                                       recv_tab, tau_d, cap, receiver_pos)

    # Downsample the interleaved per-pulse record to n_pulses display bins.
    idx = np.clip(np.searchsorted(edges, rec_int["t"], side="right") - 1, 0, n_pulses - 1)

    def _binmean(vals):
        out = np.zeros(n_pulses); cnt = np.zeros(n_pulses)
        np.add.at(out, idx, vals); np.add.at(cnt, idx, 1.0)
        return np.where(cnt > 0, out / np.maximum(cnt, 1), 0.0)

    def _binrange(vals):
        lo = np.full(n_pulses, np.inf); hi = np.full(n_pulses, -np.inf)
        np.minimum.at(lo, idx, vals); np.maximum.at(hi, idx, vals)
        lo[~np.isfinite(lo)] = 0.0; hi[~np.isfinite(hi)] = 0.0
        return lo, hi

    def _binspectra(vals):
        out = np.zeros((n_pulses, vals.shape[1])); cnt = np.zeros(n_pulses)
        for row, bi in zip(vals, idx):
            out[bi] += row
            cnt[bi] += 1.0
        return out / np.maximum(cnt[:, None], 1.0)

    sig = _binmean(rec_int["signal"]); pwr = _binmean(rec_int["power_pct"])
    beta = _binmean(rec_int["beta"]); dlv = _binmean(rec_int["delivered"])
    # Spot-to-spot power SPREAD per time-bin: with global interleaving each bin mixes
    # interior (high-power) and tapered-periphery (low-power) spots, so the band — not
    # the ~flat mean — is the visible signature of per-spot power control.
    pwr_lo, pwr_hi = _binrange(rec_int["power_pct"])
    dt = 1.0 / PRF_HZ
    cumulative_full = np.cumsum(rec_int["signal"]) * dt
    cumulative = np.interp(tc, rec_int["t"], cumulative_full)
    goal = sum(
        s["pulses"] * float(np.interp(min(s["p_target"] * s.get("e_int", 1.0), p_grid[-1]), p_grid, emis_tot))
        for s in plan["spots"]
    ) * dt
    goal = max(float(goal), 1.0)
    done = int(np.argmax(cumulative >= goal)) if np.any(cumulative >= goal) else n_pulses - 1
    spectra = _binspectra(rec_int["spectra"])

    return {"t": tc, "signal": sig, "power_pct": pwr, "power_lo": pwr_lo, "power_hi": pwr_hi,
            "cumulative": cumulative,
            "goal": goal, "beta": beta, "spectrum": (SPEC_F, spectra.mean(0)),
            "spec_f": SPEC_F, "spectra": spectra, "done_t": tc[done], "dt": treatment_s / n_pulses,
            "lacuna_slices": lac["beta_slices"], "lacuna_peak": lac["peak_beta"],
            "equality": {"interleaved": np.array(list(cav_int.values())),
                          "sequential": np.array(list(cav_seq.values())),
                          "cov_int": _coeff_var(cav_int.values()),
                          "cov_seq": _coeff_var(cav_seq.values())},
            "spot_eff": cav_int}


def _spec_at(spec_tab, p_grid, p):
    """Linear-interpolate the emission spectrum table along pressure at p."""
    j = int(np.clip(np.searchsorted(p_grid, p) - 1, 0, len(p_grid) - 2))
    w = (p - p_grid[j]) / max(p_grid[j + 1] - p_grid[j], 1e-30)
    w = min(max(w, 0.0), 1.0)
    return (1 - w) * spec_tab[j] + w * spec_tab[j + 1]


def figure_lacuna_field(ffm):
    """Full-3-D PSTD focal field with vs without a pre-focal lacuna gas inclusion —
    the resolved standing-wave / acoustic-shadow that the per-pulse path-integral
    coupling approximates. Uses the Rust ResidualGasField Wood coupling."""
    res = ffm.pstd_lacuna_focal_fields(F0, C_LIVER, RHO_LIVER, beta_lacuna=BETA_LACUNA_MAX)
    dx = float(res["dx"]); fx = int(res["fx"]); cx = int(res["cx"])
    clean = np.asarray(res["clean_slice"]); lac = np.asarray(res["lac_slice"])
    beta = np.asarray(res["beta_slice"])
    nx, nyl = clean.shape
    ext = [0, nyl * dx * 1e3, nx * dx * 1e3, 0]          # [SI/lateral mm, axial mm]
    vmax = float(clean.max()) + 1e-30
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.0))
    for a, fld, ttl in ((ax[0], clean / 1e6, "Clean focal field"),
                        (ax[1], lac / 1e6, "Through lacuna (gas void)")):
        im = a.imshow(fld, extent=ext, aspect="equal", cmap="inferno", vmax=vmax / 1e6)
        a.contour((beta > 0).astype(float), levels=[0.5], colors="cyan",
                  linewidths=1.2, extent=ext, origin="upper")
        a.axhline(fx * dx * 1e3, color="white", ls=":", lw=0.8)
        a.set_title(ttl); a.set_xlabel("lateral [mm]"); a.set_ylabel("beam axis [mm]")
        fig.colorbar(im, ax=a, fraction=0.046, label="peak pressure [MPa]")
    # Axial line through the focus: shadow + standing wave behind the lacuna.
    axial_clean = np.asarray(res["clean_axial"]) / 1e6
    axial_lac = np.asarray(res["lac_axial"]) / 1e6
    z = np.arange(axial_clean.size) * dx * 1e3
    ax[2].plot(z, axial_clean, color="#2ecc71", label="clean")
    ax[2].plot(z, axial_lac, color="#e74c3c", label="through lacuna")
    ax[2].axvline(cx * dx * 1e3, color="cyan", ls="--", lw=1.0, label="lacuna")
    ax[2].axvline(fx * dx * 1e3, color="k", ls=":", lw=0.8, label="focus")
    ax[2].set_xlabel("beam axis [mm]"); ax[2].set_ylabel("peak pressure [MPa]")
    shadow = 100.0 * (1.0 - axial_lac[fx] / max(axial_clean[fx], 1e-30))
    ax[2].set_title(f"On-axis profile (focus shielded {shadow:.0f}%)")
    ax[2].legend(fontsize=8)
    fig.suptitle("ch21e — Resolved standing-wave / shadow field through a lacuna "
                 "(full-3D PSTD, Wood-coupled gas void)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig25_lacuna_field")


def figure_raster_equality(mon):
    """Residual-bubble penalty: per-spot delivery efficiency for the HYBRID
    interleaved-group schedule (each spot rests group_size/PRF ≫ τ_d within its group)
    vs a SEQUENTIAL raster (re-sonicates a spot before its residual bubble cloud
    dissolves → shielding → unequal cavitation). The hybrid keeps near-full, equal
    cavitation while completing each local group fast."""
    eq = mon["equality"]
    ci, cs = eq["interleaved"], eq["sequential"]   # per-spot delivery efficiency
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    a = ax[0]
    bins = np.linspace(0, 1.05, 30)
    a.hist(cs, bins=bins, alpha=0.6, color="#e74c3c",
           label=f"sequential  (mean {cs.mean()*100:.0f}%, CoV {eq['cov_seq']*100:.0f}%)")
    a.hist(ci, bins=bins, alpha=0.6, color="#2ecc71",
           label=f"hybrid groups  (mean {ci.mean()*100:.0f}%, CoV {eq['cov_int']*100:.0f}%)")
    a.set_xlabel("per-spot delivery efficiency  (achieved ÷ unshielded cavitation)")
    a.set_ylabel("sub-spot count")
    a.set_title("Per-spot cavitation efficiency"); a.legend(fontsize=8)
    a = ax[1]
    under = lambda c: float(np.mean(c < 0.7) * 100.0)
    bars = a.bar(["sequential", "hybrid"], [under(cs), under(ci)],
                 color=["#e74c3c", "#2ecc71"])
    a.set_ylabel("% sub-spots under-dosed (efficiency < 70%)")
    a.set_title("Residual-bubble shielding penalty")
    for b, v in zip(bars, [under(cs), under(ci)]):
        a.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}%", ha="center", va="bottom")
    fig.suptitle("ch21e — Firing order: residual-bubble cavitation equality "
                 "(hybrid interleaved-groups vs sequential)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "fig24_raster_equality")


BIO_RINGS = {"GTV core": "#ff5a5a", "CTV ring": "#ffb24d", "PTV margin": "#4dd2ff"}
LD_BAND_COLORS = ["#3a86ff", "#4dd2ff", "#ffb24d", "#ff7a3d", "#d11313"]  # <LD25 … ≥LD100


def _draw_bioeffect_panel(ax, prog, km=None, label_color="white"):
    """Cell-kill panel: per-shell mean CELL-KILL (%) curves growing over the sonication,
    with dotted iso-lethal LD25/LD50/LD75/LD100 thresholds (radiotherapy-isodose style).
    The GTV core climbs to ≈LD100 (complete fractionation), the CTV ring to ~LD50–75
    (partial kill), and the PTV margin stays below LD25 (viable, immune-primed)."""
    t = prog["t"]
    for lv, key in zip(prog["ld_levels"], prog["ld_keys"]):
        pct = lv * 100.0
        ax.axhline(pct, ls=":", lw=1.1, color="#9aa7c7")
        ax.text(t[-1], pct, f" {key}", color="#9aa7c7", fontsize=6, va="center", ha="right")
    n = km if km is not None else len(t)
    for name, arr in prog["ring"].items():
        ax.plot(t[:n], arr[:n], color=BIO_RINGS[name], lw=2.0, label=name)
    ax.set_ylim(0, 105); ax.set_xlim(0, float(t[-1]))
    ax.set_xlabel("time [s]", color=label_color)
    ax.set_ylabel("cell kill [%]", color=label_color)
    ax.set_title("Cell kill → LD25 / LD50 / LD75 / LD100", color="orange")
    ax.legend(fontsize=6, loc="upper left", facecolor="#0b0f1a", labelcolor=label_color, framealpha=0.5)


def figure_monitor(mon, prog):
    """Static 3-panel monitor in PHYSICAL units (no 0–1 normalisation): measured
    emission PSD, cavitation signal + applied power, and the graded-bioeffect dose
    panel (per-shell dose curves with ablation/damage/priming phase thresholds)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4)); fig.patch.set_facecolor("black")
    for a in axes:
        a.set_facecolor("black"); a.tick_params(colors="white")
        for sp in a.spines.values():
            sp.set_color("white")
    # Spectrum [Pa²] — time-mean measured emission.
    a = axes[0]
    f, psd = mon["spectrum"]; m = f > 5e4
    a.fill_between(f[m] / 1e6, psd[m], color="orange", alpha=0.9)
    a.set_xlim(0, 4); a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    a.set_title("Acoustic Spectrum", color="orange")
    a.set_xlabel("frequency [MHz]", color="white"); a.set_ylabel("emission PSD [Pa²]", color="white")
    # Controls — cavitation signal [Pa²] + applied power [%].
    a = axes[1]; aR = a.twinx()
    a.bar(mon["t"], mon["signal"], width=mon["dt"] * 0.9, color="orange", alpha=0.85)
    a.set_xlabel("time [s]", color="white"); a.set_ylabel("cavitation signal [Pa²]", color="orange")
    a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    a.set_title("Acoustic Controls", color="orange")
    aR.fill_between(mon["t"], mon["power_lo"], mon["power_hi"], color="lime", alpha=0.25, lw=0)
    aR.plot(mon["t"], mon["power_pct"], color="lime", lw=1.6); aR.set_ylim(0, 105)
    aR.set_ylabel("power [%] (band = per-spot spread)", color="lime"); aR.tick_params(colors="lime")
    # Graded-bioeffect dose panel — per-shell dose curves + phase thresholds.
    _draw_bioeffect_panel(axes[2], prog, label_color="white")
    fig.suptitle("Sensor-recorded cavitation + graded bioeffect over the treatment", color="white")
    _save(fig, "fig22_treatment_pipeline_monitor", facecolor="black")


# ─────────────────────────────────────────────────────────────────────────────
# Fig C: ANIMATED treatment console (CT + tumour contour + raster + dose + monitor)
# ─────────────────────────────────────────────────────────────────────────────
def compute_bioeffect_progress(geom, plan, mon, n_steps=80):
    """Graded CELL-KILL and cumulative cavitation-dose progress vs treatment time.

    Boiling histotripsy fractionates PER PULSE. Each voxel accumulates delivered
    cavitation dose from the actual monitored delivery efficiency for its spot
    (residual shielding / lacuna coupling / receive-scaled cavitation), then the
    Rust Weibull histotripsy dose-response converts cumulative dose to cell kill.

    The conformal pressure taper gives a radial gradient — GTV core → near-complete
    kill (≈LD100), CTV ring → ~LD50–75, PTV margin → sub-LD25 (viable, immune-primed).
    Returns per-shell mean cell-kill curves, LD levels, planning-volume fraction past
    each LD level, per-time focal-slice cell-kill and cumulative-dose maps."""
    dx = geom["dx"]; shape = geom["tumour"].shape; fx = geom["fx"]
    gtv = geom["gtv"]; ctv = geom["ctv"]; ptv = geom["ptv"]
    spot_eff = mon.get("spot_eff", {})
    items = []
    for s in plan["spots"]:
        ix, iy, iz = s["idx"]; ll, la = s["les_lat"], s["les_ax"]
        tmp = np.zeros(shape, dtype=bool)
        _mark_ellipsoid(tmp, ix, iy, iz, ll, la, dx,
                        int(np.ceil(ll / dx)) + 1, int(np.ceil(la / dx)) + 1)
        n_pulse = max(int(s["pulses"]), 1)
        eff = float(spot_eff.get(s["idx"], 1.0))
        w = max(eff, 0.0) * (float(s["delivered"]) / P_TARGET_FOCAL_PA) ** 3
        t0 = s.get("t_start", s.get("t_done", 0.0)); t1 = max(s.get("t_done", t0), t0 + 1e-9)
        items.append((np.flatnonzero(tmp), n_pulse, w, t0, t1))
    nvox = int(np.prod(shape))
    shells = {"GTV core": gtv, "CTV ring": ctv & ~gtv, "PTV margin": ptv & ~ctv}
    flat = {k: np.flatnonzero(v) for k, v in shells.items()}
    ptv_flat = np.flatnonzero(ptv)
    ld_keys = ["LD25", "LD50", "LD75", "LD100"]
    ts = np.linspace(0.0, max(plan["treatment_s"], 1e-9), n_steps)
    ring = {k: [] for k in shells}
    tier_vol = {k: [] for k in ld_keys}
    kill_slices = []; dose_slices = []
    lsurv = np.zeros(nvox); dose = np.zeros(nvox)
    dose_peak = 0.0
    final_dose = np.zeros(nvox)
    for idx, n_pulse, w, _t0, _t1 in items:
        final_dose[idx] += w * n_pulse
    # Weibull characteristic dose d0 (≈63 % kill) is referenced to the dose the
    # FULLY-TREATED TARGET — the GTV core — actually receives, NOT the global peak.
    # The peak is an ellipsoid-overlap hot-spot; keying d0 to it inflates the
    # threshold and spuriously suppresses every typical core voxel (the core mean
    # then reads ≈LD25). Using the robust median of the GTV-core dose makes a
    # well-treated core voxel reach LD100 — as intended for the ablated target —
    # while the lower-dose CTV/PTV margins still grade down (radial gradient).
    d0 = max(KILL_D0_FRAC * _reference_dose(final_dose, flat["GTV core"]), 1e-30)
    for tcut in ts:
        lsurv[:] = 0.0; dose[:] = 0.0
        for idx, n_pulse, w, t0, t1 in items:
            frac = (tcut - t0) / (t1 - t0)
            if frac <= 0.0:
                continue
            n = n_pulse * min(frac, 1.0)                         # pulses delivered so far
            dose[idx] += w * n                                   # cumulative cavitation dose
        kf = np.asarray(kw.delivered_histotripsy_progress(
            np.ascontiguousarray(dose), d0, KILL_WEIBULL_K), float)
        dose_peak = max(dose_peak, float(dose.max()))
        for k, vi in flat.items():
            ring[k].append(float(kf[vi].mean()) * 100.0 if vi.size else 0.0)
        kp = kf[ptv_flat]
        for key, lv in zip(ld_keys, LD_LEVELS):
            tier_vol[key].append(float(np.mean(kp >= lv) * 100.0))
        kill_slices.append(kf.reshape(shape)[fx])
        dose_slices.append(dose.reshape(shape)[fx].copy())
    ring = {k: np.asarray(v) for k, v in ring.items()}
    tier_vol = {k: np.asarray(v) for k, v in tier_vol.items()}
    return {"t": ts, "ring": ring, "ld_levels": LD_LEVELS, "ld_keys": ld_keys,
            "tier_vol": tier_vol, "kill_slices": kill_slices,
            "dose_slices": dose_slices, "dose_peak": max(dose_peak, 1e-30)}


def _final_kill_volume(geom, plan):
    """Final 3-D cell-kill field over the whole grid: cumulative cavitation dose
    (∝ p_focal³·pulses) deposited into every fired sub-spot's lesion ellipsoid, mapped
    to cell kill by the Rust Weibull dose–response. The ellipsoids tile the PTV in 3-D
    (lattice spacing set by the lesion FWHM), so this is the 3-D lesioned volume."""
    dx = geom["dx"]; shape = geom["tumour"].shape
    dose = np.zeros(int(np.prod(shape)))
    for s in plan["spots"]:
        ix, iy, iz = s["idx"]; ll, la = s["les_lat"], s["les_ax"]
        tmp = np.zeros(shape, dtype=bool)
        _mark_ellipsoid(tmp, ix, iy, iz, ll, la, dx,
                        int(np.ceil(ll / dx)) + 1, int(np.ceil(la / dx)) + 1)
        w = (float(s["delivered"]) / P_TARGET_FOCAL_PA) ** 3
        dose[np.flatnonzero(tmp)] += w * max(int(s["pulses"]), 1)
    # d0 referenced to the GTV-core dose (the fully-treated target), robust to
    # ellipsoid-overlap hot-spots — consistent with _kill_curves so the final
    # lesion volume and the live curves use the same kill calibration.
    d0 = max(KILL_D0_FRAC * _reference_dose(dose, np.flatnonzero(geom["gtv"])), 1e-30)
    kf = np.asarray(kw.delivered_histotripsy_progress(
        np.ascontiguousarray(dose), d0, KILL_WEIBULL_K), float)
    return kf.reshape(shape)


def figure_volume_slices(geom, plan, n_slices=9):
    """Multi-slice montage of the final 3-D lesioned volume — the LD25/50/75/100
    cell-kill map (with GTV/CTV/PTV contours) on a stack of beam-axis depth slices
    spanning the tumour, demonstrating that the lesion FWHM-spaced raster fractionates
    the segmented tumour across every image slice (not just the focal plane)."""
    from matplotlib.colors import ListedColormap
    kill = _final_kill_volume(geom, plan)
    gtv = geom["gtv"]; ctv = geom["ctv"]; ptv = geom["ptv"]; ct = geom["ct"]
    info = geom["info"]; dx = geom["dx"]
    coords = np.argwhere(ptv); cmin, cmax = coords.min(0), coords.max(0)
    pad = int(round(6e-3 / dx))
    y0 = max(0, cmin[1] - pad); y1 = min(ct.shape[1], cmax[1] + pad)
    z0 = max(0, cmin[2] - pad); z1 = min(ct.shape[2], cmax[2] + pad)
    xs = np.unique(np.linspace(cmin[0], cmax[0], n_slices).round().astype(int))
    tier_cmap = ListedColormap([(0, 0, 0, 0), (0.30, 0.82, 1.0, 0.50),
                                (1.0, 0.70, 0.30, 0.58), (1.0, 0.48, 0.24, 0.66),
                                (0.82, 0.07, 0.07, 0.78)])
    ld = np.asarray(LD_LEVELS)
    ext = [info["z_axis"][z0] * 1e3, info["z_axis"][z1 - 1] * 1e3,
           info["y_axis"][y1 - 1] * 1e3, info["y_axis"][y0] * 1e3]
    ncol = 3; nrow = int(np.ceil(len(xs) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.2 * nrow))
    ax = np.atleast_1d(ax).ravel()
    for a in ax:
        a.axis("off")
    for k, xk in enumerate(xs):
        a = ax[k]; a.axis("on")
        ctsl = ct[xk, y0:y1, z0:z1]
        a.imshow(ctsl, cmap="gray", extent=ext, aspect="equal",
                 vmin=np.percentile(ctsl, 2), vmax=np.percentile(ctsl, 98))
        ksl = kill[xk, y0:y1, z0:z1]
        a.imshow(np.digitize(ksl, ld).astype(float), cmap=tier_cmap, extent=ext,
                 aspect="equal", origin="upper", vmin=0, vmax=4, interpolation="nearest")
        for m, c in ((ptv, "white"), (ctv, "yellow"), (gtv, "cyan")):
            a.contour(m[xk, y0:y1, z0:z1].astype(float), levels=[0.5], colors=c,
                      linewidths=1.0, extent=ext, origin="upper")
        gm = gtv[xk, y0:y1, z0:z1]
        cov = 100.0 * float(np.mean(ksl[gm] >= 0.5)) if gm.any() else 0.0
        a.set_title(f"depth {info['x_axis'][xk] * 1e3:.0f} mm | GTV≥LD50 {cov:.0f}%", fontsize=8)
        a.set_xticks([]); a.set_yticks([])
        a.set_xlabel("SI [mm]", fontsize=6); a.set_ylabel("lateral [mm]", fontsize=6)
    cov3 = 100.0 * float(np.mean(kill[gtv] >= 0.5)) if gtv.any() else 0.0
    fig.suptitle("ch21e — 3-D lesioned volume across depth slices "
                 f"(LD25/50/75/100 cell-kill; whole-GTV ≥LD50 = {cov3:.0f}%)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fig26_volume_slices")
    return cov3


def make_treatment_gif(geom, plan, mon, prog, bmode=None, n_frames=80, fps=12):
    info, dx, fx = geom["info"], geom["dx"], geom["fx"]
    ct = geom["ct"]
    coords = np.argwhere(geom["ptv"])
    cmin, cmax = coords.min(0), coords.max(0)
    pad = int(round(8e-3 / dx))
    y0 = max(0, cmin[1] - pad); y1 = min(ct.shape[1], cmax[1] + pad)
    z0 = max(0, cmin[2] - pad); z1 = min(ct.shape[2], cmax[2] + pad)
    extent = [info["z_axis"][z0] * 1e3, info["z_axis"][z1 - 1] * 1e3,
              info["y_axis"][y1 - 1] * 1e3, info["y_axis"][y0] * 1e3]
    ct_sl = ct[fx, y0:y1, z0:z1]
    gtv_sl = geom["gtv"][fx, y0:y1, z0:z1].astype(float)
    ctv_sl = geom["ctv"][fx, y0:y1, z0:z1].astype(float)
    ptv_sl = geom["ptv"][fx, y0:y1, z0:z1].astype(float)
    oar_sl_y = info["y_axis"][geom["oar_iy"]] * 1e3

    # Per-sub-spot 2-D dose footprint on the focal (y–z) slice, weighted by axial
    # proximity to the slice. The lesion builds up PER PULSE: one increment is
    # deposited at each fired pulse of the global interleaved schedule, so the dose
    # map fills in smoothly and spatially throughout the whole treatment (not all at
    # the end) and the active-spot marker hops across the tumour every pulse.
    nyl, nzl = y1 - y0, z1 - z0
    Yc, Zc = np.meshgrid(np.arange(nyl), np.arange(nzl), indexing="ij")
    fp_cache = {}
    for s in plan["spots"]:
        iy, iz = s["yz"]
        if not (y0 <= iy < y1 and z0 <= iz < z1):
            continue
        iyl, izl = iy - y0, iz - z0
        sig_lat = s["les_lat"] / dx
        axial = np.exp(-((s["x"] - fx) * dx) ** 2 / (2.0 * s["les_ax"] ** 2))
        fp = np.exp(-(((Yc - iyl) ** 2 + (Zc - izl) ** 2) / (2.0 * sig_lat ** 2)))
        fp_cache[id(s)] = (fp * axial, (iyl, izl))      # per-pulse footprint increment
    # One contribution per fired pulse, in global interleaved fire order.
    contrib = [(t, *fp_cache[id(s)]) for (t, s) in plan["schedule"] if id(s) in fp_cache]

    treatment_s = max(plan["treatment_s"], 1e-9)
    frame_t = np.linspace(0, treatment_s, n_frames)

    # Physical-unit scales (NOT normalised): actual simulated measured emission.
    spec_f = mon["spec_f"]; spectra = mon["spectra"]
    spec_mask = spec_f > 5e4
    spec_gmax = float(spectra[:, spec_mask].max()) + 1e-30
    sig_max = float(mon["signal"].max()) + 1e-30
    vol_colors = {"GTV": "cyan", "CTV": "yellow", "PTV": "white"}
    # LOG-scale cumulative cavitation-dose heat map (colourwash) + iso-lethal LD25/50/
    # 75/100 cell-kill contours (radiotherapy-isodose style). Log scale reveals the
    # per-pulse dose deposited from the first passes, not just the late high-dose core.
    dose_slices = prog["dose_slices"]; kill_slices = prog["kill_slices"]
    ld_edges = np.asarray(prog["ld_levels"])         # [0.25,0.5,0.75,0.99]
    ld_colors = ["#4dd2ff", "#ffd24d", "#ff7a3d", "#d11313"]
    dvmax = float(prog["dose_peak"]); dvmin = dvmax / 1e3
    dose_cmap = plt.get_cmap("turbo").copy(); dose_cmap.set_bad(alpha=0.0)

    fig = plt.figure(figsize=(13.5, 7.4)); fig.patch.set_facecolor("#05070d")
    gs = fig.add_gridspec(3, 2, width_ratios=[2.0, 1.05], hspace=0.6, wspace=0.32)
    axCT = fig.add_subplot(gs[:, 0])
    axSp = fig.add_subplot(gs[0, 1]); axPw = fig.add_subplot(gs[1, 1]); axDo = fig.add_subplot(gs[2, 1])
    axPwR = axPw.twinx()
    for a in (axCT, axSp, axPw, axPwR, axDo):
        a.set_facecolor("#05070d"); a.tick_params(colors="#9ecbff", labelsize=7)
        for sp in a.spines.values():
            sp.set_color("#33415c")

    axCT.imshow(ct_sl, cmap="gray", extent=extent, aspect="equal",
                vmin=np.percentile(ct_sl, 2), vmax=np.percentile(ct_sl, 98))
    # Planning-volume contours: GTV (cyan), CTV (yellow), PTV (white).
    for sl, col in ((ptv_sl, "white"), (ctv_sl, "yellow"), (gtv_sl, "cyan")):
        axCT.contour(sl, levels=[0.5], colors=col, linewidths=1.8,
                     extent=extent, origin="upper")
    axCT.axhline(oar_sl_y, color="red", lw=2.0)
    axCT.text(extent[0], oar_sl_y, " OAR (vessel)", color="red", fontsize=8, va="bottom")
    # Log-scale cumulative cavitation-dose heat map; LD cell-kill contours drawn per
    # frame in update().
    dose_im = axCT.imshow(np.full((nyl, nzl), np.nan), cmap=dose_cmap,
                          norm=LogNorm(vmin=dvmin, vmax=dvmax), extent=extent,
                          aspect="equal", origin="upper", alpha=0.8)
    cb = fig.colorbar(dose_im, ax=axCT, fraction=0.035, pad=0.01)
    cb.set_label("cumulative cavitation dose [a.u., log]", color="#9ecbff", fontsize=7)
    cb.ax.tick_params(colors="#9ecbff", labelsize=6)
    active = axCT.plot([], [], "x", color="white", ms=10, mew=2)[0]
    axCT.set_xlabel("superior–inferior [mm]", color="#9ecbff")
    axCT.set_ylabel("lateral [mm]", color="#9ecbff")
    from matplotlib.lines import Line2D
    bio_handles = [Line2D([], [], color=c, lw=2, label=k)
                   for k, c in zip(prog["ld_keys"], ld_colors)]
    axCT.legend(handles=[Line2D([], [], color=c, lw=2, label=k) for k, c in vol_colors.items()]
                + [Line2D([], [], color="red", lw=2, label="OAR")] + bio_handles,
                loc="upper right", fontsize=6.5, facecolor="#0b0f1a", labelcolor="white",
                framealpha=0.6)
    title = axCT.set_title("", color="white", fontsize=11)

    # Picture-in-picture: baseline-locked delta B-mode from the reconstructed focal
    # plane. Raw gain-locked B-mode frames remain in `bmode["frames"]`; the delta
    # view depicts lesion/gas evolution without per-frame peak renormalisation.
    bmode_frames = bmode["frames"] if bmode is not None else []
    delta_bmode_frames = bmode.get("delta_frames", []) if bmode is not None else []
    pip_im = None
    if delta_bmode_frames:
        ax_pip = axCT.inset_axes([0.625, 0.02, 0.355, 0.355])
        ax_pip.set_facecolor("#000000")
        pip_im = ax_pip.imshow(delta_bmode_frames[0], extent=bmode["extent"], aspect="equal",
                               cmap="seismic", vmin=-12.0, vmax=12.0, origin="upper")
        ax_pip.set_xticks([]); ax_pip.set_yticks([])
        for sp in ax_pip.spines.values():
            sp.set_color("#19e6c8"); sp.set_linewidth(1.4)
        ax_pip.set_title("delta B-mode [dB]", color="#19e6c8", fontsize=7, pad=2)

    state = {"i": 0}

    def update(fi):
        T = frame_t[fi]
        while state["i"] < len(contrib) and contrib[state["i"]][0] <= T:
            state["i"] += 1                         # advance fired-pulse cursor (active marker)
        if state["i"] > 0:
            _, _, (iyl, izl) = contrib[state["i"] - 1]
            active.set_data([info["z_axis"][z0 + izl] * 1e3], [info["y_axis"][y0 + iyl] * 1e3])
        km = max(1, min(int(np.searchsorted(mon["t"], T)), len(mon["t"])))
        # Log-scale cumulative cavitation-dose heat map + iso-lethal LD contours.
        kb = min(km - 1, len(dose_slices) - 1)
        dsl = np.asarray(dose_slices[kb])[y0:y1, z0:z1]
        dose_im.set_data(np.where(dsl > dvmin, dsl, np.nan))
        if state.get("ld_cs") is not None:
            try:
                state["ld_cs"].remove()
            except Exception:
                pass
            state["ld_cs"] = None
        ksl = np.asarray(kill_slices[kb])[y0:y1, z0:z1]
        if float(ksl.max()) >= ld_edges[0]:
            state["ld_cs"] = axCT.contour(ksl, levels=list(ld_edges), colors=ld_colors,
                                          linewidths=1.1, extent=extent, origin="upper")
        # Live B-mode reconstruction (PIP).
        if pip_im is not None:
            pip_im.set_data(delta_bmode_frames[min(km - 1, len(delta_bmode_frames) - 1)])

        # Acoustic Spectrum — actual measured emission PSD [Pa²] (changes per frame).
        axSp.clear(); axSp.set_facecolor("#05070d")
        axSp.fill_between(spec_f[spec_mask] / 1e6, spectra[km - 1][spec_mask], color="orange", alpha=0.9)
        axSp.set_xlim(0, 4); axSp.set_ylim(0, spec_gmax)
        axSp.set_title("Acoustic Spectrum", color="orange", fontsize=9)
        axSp.set_xlabel("frequency [MHz]", color="#9ecbff", fontsize=7)
        axSp.set_ylabel("emission PSD [Pa²]", color="#9ecbff", fontsize=7)
        axSp.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axSp.tick_params(colors="#9ecbff", labelsize=6)

        # Acoustic Controls — measured cavitation signal [Pa²] + applied power [%].
        axPw.clear(); axPwR.clear()
        axPw.set_facecolor("#05070d")
        axPw.bar(mon["t"][:km], mon["signal"][:km], width=mon["dt"] * 0.9, color="orange", alpha=0.85)
        axPw.set_xlim(0, treatment_s); axPw.set_ylim(0, sig_max * 1.05)
        axPw.set_xlabel("time [s]", color="#9ecbff", fontsize=7)
        axPw.set_ylabel("cavitation signal [Pa²]", color="orange", fontsize=7)
        axPw.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axPw.set_title("Acoustic Controls", color="orange", fontsize=9)
        axPw.tick_params(colors="#9ecbff", labelsize=6)
        axPwR.fill_between(mon["t"][:km], mon["power_lo"][:km], mon["power_hi"][:km],
                           color="lime", alpha=0.25, lw=0)
        axPwR.plot(mon["t"][:km], mon["power_pct"][:km], color="lime", lw=1.4)
        axPwR.set_ylim(0, 105); axPwR.set_ylabel("power [%] (band = per-spot spread)",
                                                  color="lime", fontsize=7)
        axPwR.tick_params(colors="lime", labelsize=6)

        # Graded-bioeffect dose panel — per-shell dose curves growing over time with
        # the ablation / damage / priming phase thresholds.
        axDo.clear(); axDo.set_facecolor("#05070d")
        _draw_bioeffect_panel(axDo, prog, km=km, label_color="#9ecbff")
        axDo.title.set_fontsize(9)
        axDo.tick_params(colors="#9ecbff", labelsize=6)
        for lab in (axDo.xaxis.label, axDo.yaxis.label):
            lab.set_fontsize(7)

        vol = prog["tier_vol"]; kk = min(km - 1, len(vol["LD100"]) - 1)
        title.set_text(f"Liver histotripsy — t={T:5.1f}s | pulses {state['i']}/{len(contrib)} | PTV "
                       f"≥LD100 {vol['LD100'][kk]:.0f}% · ≥LD50 {vol['LD50'][kk]:.0f}% · "
                       f"≥LD25 {vol['LD25'][kk]:.0f}%")
        return [dose_im, active] + ([pip_im] if pip_im is not None else [])

    fig.suptitle("kwavers — Liver Histotripsy Treatment Console (real CT, whole-tumour)",
                 color="white", fontsize=13)
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    path = os.path.join(OUT_DIR, "anim_treatment_console.gif")
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {path}")


def _save(fig, name, facecolor="white"):
    p = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor=facecolor)
    plt.close(fig); print(f"  wrote {p}")


def main():
    import focal_field_models as ffm

    print("Loading CT + segmenting tumour, building sonication raster...")
    geom = build_geometry()
    print(f"  {len(geom['region_centers'])} candidate regions, "
          f"{len(geom['subspot_offsets'])} sub-spots/region")

    # Two focal-field models: analytic (Penttinen Gaussian) and PSTD-nonlinear
    # (simulated focused bowl with shock steepening; cached after the first run).
    rc, prof, meta = ffm.pstd_nonlinear_profile(
        F0, C_LIVER, RHO_LIVER, _BA,
        roc=TRANSDUCER_ROC_M, diameter=TRANSDUCER_DIAM_M, ppw=TRANSDUCER_PPW)
    fields = {"analytic": ffm.gaussian_profile(SIGMA_LAT),
              "pstd_nonlinear": ffm.pstd_profile_callable(rc, prof)}

    print("Planning treatment with BOTH focal-field models...")
    plans = {}
    for name, B in fields.items():
        pl = plan_treatment(geom, B)
        plans[name] = pl
        grp_s = pl["treatment_s"] / max(pl.get("n_groups", 1), 1)
        print(f"  [{name}] {pl['n_regions_active']} regions, {len(pl['spots'])} spots, "
              f"{pl['onsets'].size} pulses; coverage {pl['coverage']*100:.1f}%; "
              f"treatment {pl['treatment_s']:.0f} s; {pl.get('n_groups', 0)} hybrid groups "
              f"(~{grp_s:.0f} s/group)")

    print("Fig - focal-field comparison (analytic vs PSTD-nonlinear)...")
    figure_field_comparison(geom, fields, meta, plans)

    print("Planning canonical treatment with steered 3-D aperture transmission...")
    plan = plan_treatment(geom, None)
    print(f"  [steered_aperture] {plan['n_regions_active']} regions, {len(plan['spots'])} spots, "
          f"{plan['onsets'].size} pulses; coverage {plan['coverage']*100:.1f}%; "
          f"treatment {plan['treatment_s']:.0f} s")
    print("Fig A - whole-tumour pulse train...")
    figure_pulsing_pattern(plan)
    print("Fig B - sensor-recorded cavitation monitor...")
    mon = simulate_measured_sonication(plan, geom)
    print("Computing graded cell-kill (LD25/LD50/LD75/LD100) progress...")
    prog = compute_bioeffect_progress(geom, plan, mon)
    figure_monitor(mon, prog)
    tv = prog["tier_vol"]
    rg = prog["ring"]
    print(f"  PTV cell-kill (final): >=LD100 {tv['LD100'][-1]:.0f}%, >=LD75 {tv['LD75'][-1]:.0f}%, "
          f">=LD50 {tv['LD50'][-1]:.0f}%, >=LD25 {tv['LD25'][-1]:.0f}%")
    print(f"  shell mean kill (final): GTV core {rg['GTV core'][-1]:.0f}%, "
          f"CTV ring {rg['CTV ring'][-1]:.0f}%, PTV margin {rg['PTV margin'][-1]:.0f}%")
    eq = mon["equality"]
    print(f"  residual-bubble equality (mean efficiency): hybrid {eq['interleaved'].mean()*100:.0f}% "
          f"vs sequential {eq['sequential'].mean()*100:.0f}%")
    figure_raster_equality(mon)
    print("Fig D - resolved standing-wave/shadow field through a lacuna (full-3D PSTD)...")
    figure_lacuna_field(ffm)
    print("Fig E - 3-D lesioned volume across image slices...")
    cov3 = figure_volume_slices(geom, plan)
    print(f"  3-D volume coverage: whole-GTV >=LD50 = {cov3:.0f}%")
    print("Reconstructing real-time B-mode frames (simulated receive -> DAS)...")
    bmode = reconstruct_bmode_frames(geom, plan, mon["t"])
    print("Fig C - animated treatment console (GIF, PSTD-nonlinear field + B-mode PIP)...")
    make_treatment_gif(geom, plan, mon, prog, bmode=bmode)
    print("Done.")


if __name__ == "__main__":
    main()
