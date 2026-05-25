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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — required for projection="3d"
import nibabel as nib
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
                           distance_transform_edt, label, zoom)
from scipy.special import erf

try:
    import pykwavers as kw
    _HAS_PYKWAVERS = True
except ImportError:
    kw = None
    _HAS_PYKWAVERS = False

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
    # Intrinsic cavitation threshold at 1 MHz (Pa).
    # Tissue-specific values from Vlaisavljevich 2015/2016 and Maxwell 2013.
    # float('inf') = no intrinsic cavitation (rigid scatterer or gas interface).
    pt_pa_1mhz: float = 28.2e6   # default: water-based global reference (Vlaisavljevich 2015)
    # 1σ width of the Maxwell 2013 erf-CDF threshold distribution (Pa).
    # Reflects biological variability within each tissue type.
    pt_sigma_pa: float = 0.96e6  # default: water-based σ (Maxwell 2013)


#  Acoustic/thermal: Duck 1990 / IT'IS Foundation v4.1 / Mast 2000
#  Cavitation thresholds: Vlaisavljevich 2015 (JASA 138:1864) — frequency-
#    dependent water baseline; Vlaisavljevich 2016 (JASA 140:3504) — tissue-
#    specific porcine liver, muscle, fat, n=30 each; Maxwell 2013 (JASA
#    134:1765) — erf-CDF model and σ values; HOPE4LIVER trial (NCT04573881)
#    HCC data context.
#  Temperature correction: −0.3 MPa/°C (Vlaisavljevich 2015 Fig. 7).
AIR    = Tissue(0, "air",       1.2, 343.0,    0.0,  1.0,    1005.0, 0.026, 0.0,
                pt_pa_1mhz=float('inf'), pt_sigma_pa=1.0)
SKIN   = Tissue(1, "skin",   1109.0, 1624.0,  21.158, 1.10, 3391.0, 0.37, 1.06,
                # Water-rich dermis; no dedicated histotripsy data.
                # Vlaisavljevich 2015: water-rich tissue ~28 MPa; assume close to water.
                pt_pa_1mhz=26.0e6, pt_sigma_pa=3.0e6)
FAT    = Tissue(2, "fat",     911.0, 1440.0,   4.836, 1.10, 2348.0, 0.21, 0.43,
                # Vlaisavljevich 2015 (JASA 138:1864): lipid-rich tissues threshold
                # ~13–16 MPa, approximately 50 % below the water baseline.
                # σ = 2 MPa reflects moderate fat-composition variability.
                pt_pa_1mhz=14.0e6, pt_sigma_pa=2.0e6)
MUSCLE = Tissue(3, "muscle", 1090.0, 1588.0,   8.054, 1.10, 3421.0, 0.49, 0.67,
                # Vlaisavljevich 2015: skeletal muscle 23–27 MPa; σ = 2 MPa.
                pt_pa_1mhz=25.0e6, pt_sigma_pa=2.0e6)
BONE   = Tissue(4, "bone",   1908.0, 4080.0, 250.0,   1.0,  1313.0, 0.32, 0.10,
                # Cortical bone: acoustic impedance mismatch reflects ultrasound;
                # intrinsic cavitation nucleation not observed (Vlaisavljevich 2017).
                pt_pa_1mhz=float('inf'), pt_sigma_pa=1.0)
LIVER  = Tissue(5, "liver",  1079.0, 1595.0,   8.690, 1.10, 3540.0, 0.52, 6.4,
                # Vlaisavljevich 2016 (JASA 140:3504): porcine liver, n=30,
                # mean 20.6 ± 4.6 MPa (1σ); corrected for 1 MHz.
                pt_pa_1mhz=20.6e6, pt_sigma_pa=4.6e6)
HCC    = Tissue(6, "hcc",    1066.0, 1570.0,  12.500, 1.10, 3750.0, 0.55, 9.0,
                # Maxwell 2013 (JASA 134:1765) / HOPE4LIVER clinical context:
                # HCC has denser cell packing and higher nuclear/cytoplasm ratio
                # than normal parenchyma → slightly higher threshold 22.2 MPa,
                # narrower distribution σ = 1.2 MPa (uniform pathological tissue).
                pt_pa_1mhz=22.2e6, pt_sigma_pa=1.2e6)
# Same acoustic/thermal properties as HCC but separate label so
# multifocal disease can be visualised — only the target focus
# (label 6) is driven by the raster planner; HCC_OTHER (label 7)
# is shown as untreated for context.
HCC_OTHER = Tissue(7, "hcc_other", 1066.0, 1570.0, 12.500, 1.10, 3750.0, 0.55, 9.0,
                   pt_pa_1mhz=22.2e6, pt_sigma_pa=1.2e6)

TISSUES = [AIR, SKIN, FAT, MUSCLE, BONE, LIVER, HCC, HCC_OTHER]


# ───────────────────────────────────────────────────────────────────────
# Organs-at-Risk (OAR) specification and derivation
# ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OARSpec:
    """Clinical constraint specification for one organ-at-risk structure.

    Attributes
    ----------
    name        : identifier matching the key in the derive_oar_masks dict
    colour      : matplotlib colour for contour / legend display
    prv_mm      : planning risk volume expansion (mm) applied by dilation
    is_absolute : True  = hard no-raster exclusion (structural damage risk);
                  False = dose-flag only (proximity warning, not auto-excluded)
    description : clinical justification and evidence reference
    """
    name: str
    colour: str | tuple
    prv_mm: float
    is_absolute: bool
    description: str


# Evidence-based OAR specifications for hepatic histotripsy.
# ──────────────────────────────────────────────────────────────────────
# Smolock 2018 (doi:10.1016/j.ultrasmedbio.2017.11.014): bile duct 5 mm porcine
# Roberts 2014 (doi:10.1002/jcu.22171): vessel-sparing at lumen >1-2 mm
# Wang 2020 (doi:10.1016/j.ebiom.2020.102965): 5 mm bile duct clinical protocol
# FDA IND (Edison/HistoSonics histotripsy): ≥10 mm bowel clearance
# Vlaisavljevich 2017 (doi:10.1148/radiol.2017162344): in vivo vessel sparing
# ──────────────────────────────────────────────────────────────────────
OAR_SPECS: list[OARSpec] = [
    OARSpec(
        name="large_vessels",
        colour="red",
        prv_mm=3.0,
        is_absolute=False,   # dose-flag only: histotripsy is vessel-sparing at
                             # lumen >2 mm inside liver (Roberts 2014); direct
                             # targeting within the GTV is an MDT decision, not
                             # auto-exclusion.  OAR dose monitored; alert if > 5%.
        description=(
            "Portal vein + hepatic veins + IVC.  Derived: HU > 160 within "
            "liver label at portal-phase CT (parenchyma ~80-130 HU; "
            "vessels ~160-220 HU); CC filter ≥ 27 mm³ retains only lumina "
            "≥ ~3 mm diameter.  Injury: portal vein → hepatic infarction; "
            "hepatic vein/IVC → haemorrhage.  Vessel-sparing for lumen >2 mm "
            "if wall is outside ablation zone (Roberts 2014, Vlaisavljevich 2017). "
            "3 mm PRV.  Non-absolute: dose-flagged, not auto-excluded from raster "
            "because histotripsy physics spares vessels traversed by the beam. "
            "GTV-encasing vessels require MDT review."
        ),
    ),
    OARSpec(
        name="gallbladder",
        colour="orange",
        prv_mm=5.0,
        is_absolute=True,
        description=(
            "Gallbladder.  Derived: pure-bile density (HU 0-25) within 1 mm "
            "of liver surface but outside liver+tumour labels, volume ≥ 500 mm³. "
            "The narrow HU window and minimum size filter exclude peri-hepatic "
            "ascites (cirrhosis is common in HCC patients) and small fluid pockets. "
            "Injury: transmural ablation → bile peritonitis, perforation. "
            "5 mm PRV (Smolock 2018, Wang 2020)."
        ),
    ),
    OARSpec(
        name="liver_capsule",
        colour="magenta",
        prv_mm=3.0,
        is_absolute=False,
        description=(
            "Outer 3 mm shell of the liver parenchyma.  Derived: liver mask "
            "minus its 3 mm erosion.  Injury: capsular disruption → "
            "hemoperitoneum, subcapsular haematoma.  Flagged but not "
            "auto-excluded — subcapsular HCC is common and the capsule PRV "
            "would exclude valid targets; requires case-by-case clinical review."
        ),
    ),
    OARSpec(
        name="extrahepatic_prv",
        colour=(1.0, 0.45, 0.0),   # deep orange
        prv_mm=10.0,
        is_absolute=True,
        description=(
            "Extrahepatic tissue within 10 mm of liver surface.  Derived: "
            "non-liver, non-air voxels within binary_dilation(liver, 10 mm).  "
            "Covers duodenum, stomach, transverse colon, diaphragm, and right "
            "kidney depending on geometry.  FDA IND (HistoSonics): ≥10 mm "
            "bowel clearance.  Diaphragm: ≥5 mm (pneumothorax risk)."
        ),
    ),
    OARSpec(
        name="bone_prv",
        colour="darkred",
        prv_mm=5.0,
        is_absolute=True,
        description=(
            "Bone (ribs, vertebral body, sternum) + 5 mm PRV.  Derived: BONE "
            "label dilation.  Injury: periosteal heating at high PRF → rib "
            "fracture (Xu 2016).  Cortical surface reflections shift the "
            "effective focus position unpredictably."
        ),
    ),
]


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


def derive_oar_masks(
    ct_hu: np.ndarray,
    label_vol: np.ndarray,
    info: dict,
) -> dict[str, np.ndarray]:
    """Derive organ-at-risk boolean masks from CT HU + native segmentation.

    LiTS17 provides only liver (seg=1) + tumour (seg=2) native labels.
    Every structure is approximated from HU thresholding and morphological
    filtering on the resampled isotropic volume.

    Returns
    -------
    dict with keys equal to OARSpec.name for each entry in OAR_SPECS, plus:
      ``"combined"``  — union of all absolute (hard no-lesion) OARs
      ``"all_oar"``   — union including non-absolute (flagged) OARs

    Each mask already incorporates the PRV expansion defined in OAR_SPECS.

    Derivation per structure
    ────────────────────────
    large_vessels   : HU > 160 inside liver label.  Portal-phase parenchyma
                      is ~80-140 HU; portal vein + hepatic veins + IVC branches
                      are ~160-220 HU.  Connected-component filter retains
                      only structures ≥ 27 mm³ (≈ 3 mm diameter at 1.2 mm grid)
                      to remove isolated bright parenchymal voxels.
                      Expanded by PRV = 3 mm.

    gallbladder     : HU 0 to 25 (pure bile; excludes enhanced parenchyma and
                      peri-hepatic ascites), outside liver+tumour, within 1 mm
                      of liver surface.  Minimum volume 500 mm³ (cirrhotic HCC
                      patients often have ascites — the narrow HU window and
                      1 mm adjacency band plus size filter exclude small fluid
                      pockets).  Expanded by PRV = 5 mm.

    liver_capsule   : Outer PRV_mm ring of the combined liver mask derived
                      as liver_all minus its morphological erosion.  Non-absolute
                      (subcapsular HCC is common; this is a proximity flag).

    extrahepatic_prv: Non-liver, non-air tissue within 10 mm of liver surface.
                      Covers bowel wall, stomach, diaphragm, kidney depending
                      on patient geometry.  Expanded by PRV = 10 mm.

    bone_prv        : BONE label (label_vol == 4) dilated by 5 mm.
    """
    dx_mm = float(info["dx"] * 1e3)
    liver    = label_vol == LIVER.label
    hcc      = (label_vol == HCC.label) | (label_vol == HCC_OTHER.label)
    bone     = label_vol == BONE.label
    air      = label_vol == AIR.label
    liver_all = liver | hcc   # all hepatic tissue

    def _dilate(mask: np.ndarray, mm: float) -> np.ndarray:
        n = max(int(round(mm / dx_mm)), 0)
        return binary_dilation(mask, iterations=n) if n > 0 else mask.copy()

    def _erode(mask: np.ndarray, mm: float) -> np.ndarray:
        n = max(int(round(mm / dx_mm)), 0)
        return binary_erosion(mask, iterations=n) if n > 0 else mask.copy()

    # ── 1. Large vessels ─────────────────────────────────────────────
    # Portal-phase enhancement: threshold HU > 160 to separate vessel
    # lumina (~160-220 HU) from enhanced parenchyma (~80-140 HU at
    # portal phase).  CC filter retains only structures ≥ 27 mm³
    # (≈ (3 mm)³ sphere, 16 voxels at 1.2 mm grid) to exclude isolated
    # bright parenchymal voxels.  Histotripsy is vessel-sparing inside
    # the liver (Roberts 2014), so this OAR is flagged (non-absolute)
    # rather than hard-excluded — dose is monitored, not blocked.
    v_raw = liver & (ct_hu > 160) & (ct_hu < 400)
    min_vox_v = max(int(np.ceil(27.0 / dx_mm ** 3)), 1)
    cc_v, _ = label(v_raw)
    sizes_v = np.bincount(cc_v.ravel()); sizes_v[0] = 0
    v_filt = np.isin(cc_v, np.where(sizes_v >= min_vox_v)[0]) & v_raw
    vessels_prv = _dilate(v_filt, OAR_SPECS[0].prv_mm)

    # ── 2. Gallbladder ───────────────────────────────────────────────
    # Pure bile density (HU 0-25) within 1 mm of liver surface.
    # Minimum 500 mm³ (cirrhotic patients often have ascites — the narrow
    # HU window and 1mm adjacency band plus size filter exclude isolated
    # peri-hepatic fluid pockets from cirrhotic ascites).
    liver_prox = _dilate(liver_all, 1.0)
    g_raw = (liver_prox & ~liver_all & ~bone & ~air
             & (ct_hu >= 0) & (ct_hu <= 25))
    min_vox_g = max(int(np.ceil(500.0 / dx_mm ** 3)), 1)
    cc_g, _ = label(g_raw)
    sizes_g = np.bincount(cc_g.ravel()); sizes_g[0] = 0
    g_filt = np.isin(cc_g, np.where(sizes_g >= min_vox_g)[0]) & g_raw
    gallbladder_prv = _dilate(g_filt, OAR_SPECS[1].prv_mm)

    # ── 3. Liver capsule ─────────────────────────────────────────────
    # Outer shell = liver_all minus its morphological erosion by prv_mm.
    liver_eroded = _erode(liver_all, OAR_SPECS[2].prv_mm)
    liver_capsule = liver_all & ~liver_eroded

    # ── 4. Extrahepatic PRV ──────────────────────────────────────────
    # Non-liver, non-air tissue inside the prv_mm dilation of the liver.
    liver_expanded = _dilate(liver_all, OAR_SPECS[3].prv_mm)
    extrahep_prv = liver_expanded & ~liver_all & ~air

    # ── 5. Bone PRV ──────────────────────────────────────────────────
    bone_prv = _dilate(bone, OAR_SPECS[4].prv_mm)

    masks: dict[str, np.ndarray] = {
        "large_vessels":    vessels_prv,
        "gallbladder":      gallbladder_prv,
        "liver_capsule":    liver_capsule,
        "extrahepatic_prv": extrahep_prv,
        "bone_prv":         bone_prv,
    }
    # "combined": union of ABSOLUTE OARs only — used for raster exclusion.
    # Non-absolute OARs (large_vessels, liver_capsule) are dose-flagged but
    # do not block raster point placement; their proximity is monitored via
    # the oar_*_dose_max metrics.
    abs_names = {spec.name for spec in OAR_SPECS if spec.is_absolute}
    combined = np.zeros_like(liver_all, dtype=bool)
    for name, m in masks.items():
        if name in abs_names:
            combined |= m
    masks["combined"] = combined
    # "all_oar": full union for display purposes.
    masks["all_oar"] = combined.copy()
    for m in masks.values():
        if m is not combined and m is not masks.get("all_oar"):
            masks["all_oar"] = masks["all_oar"] | m
    return masks


def _draw_oar_contours(
    ax,
    oar_masks: dict[str, np.ndarray],
    info: dict,
    fx: int,
    y0: int, y1: int,
    z0: int, z1: int,
    extent: list[float],
    lw: float = 0.9,
) -> None:
    """Overlay OAR contours on ax at the focal slice [fx, y0:y1, z0:z1].

    Absolute OARs are drawn solid; non-absolute (liver_capsule) are dashed.
    """
    for spec in OAR_SPECS:
        mask = oar_masks.get(spec.name)
        if mask is None:
            continue
        sl = mask[fx, y0:y1, z0:z1].astype(float)
        if sl.any():
            ax.contour(
                np.flipud(sl), levels=[0.5],
                colors=[spec.colour],
                linewidths=lw,
                linestyles="solid" if spec.is_absolute else "dashed",
                extent=extent,
                alpha=0.90,
            )


def _oar_legend_handles() -> list:
    """Return matplotlib Line2D handles for the OAR contour legend."""
    handles = []
    for spec in OAR_SPECS:
        ls = "solid" if spec.is_absolute else "dashed"
        lbl = f"{spec.name.replace('_', ' ')} PRV {spec.prv_mm:.0f} mm"
        handles.append(Line2D([0], [0], color=spec.colour,
                               lw=0.9, linestyle=ls, label=lbl))
    return handles


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

    # 3D heterogeneous attenuation: cumulative path integral along each (j,k)
    # column from the transducer face to each depth voxel.
    # For each lateral position (j,k), integrate α(x,j,k)·dx from x=0 inward.
    # This replaces the 1D central-axis approximation (which applied the
    # attenuation at focus_idx (j,k) to ALL lateral positions) with the
    # correct per-ray integral under the paraxial (small-angle) approximation
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

    Model (Vlaisavljevich 2015/2016, Maxwell 2013):
      p_t(f, T) = p_{t,1MHz} + 1.4 MPa · log₁₀(f / 1 MHz) − 0.3 MPa · max(0, T − 20)

    Frequency scaling: Vlaisavljevich 2015 JASA 138:1864, Eq. 4.
    Temperature correction: −0.3 MPa/°C empirical (Vlaisavljevich 2015 Fig. 7).
    Reference temperature 20 °C matches in-vitro calibration standard.
    In-vivo tissue at 37 °C applies a −5.1 MPa correction from 20 °C baseline.

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

    Assigns each voxel a threshold p_t and width σ from the tissue label,
    then evaluates:
      P_cav(x) = 0.5 · (1 + erf((|p(x)| − p_t(label, f, T)) / (σ · √2)))

    For labels absent from TISSUES the global water reference (28.2 MPa, σ=0.96 MPa)
    applies.  For tissues with pt_pa_1mhz = inf (AIR, BONE) the probability is
    set to 0 — no intrinsic cavitation is possible.

    Tissue values: Vlaisavljevich 2015/2016 and Maxwell 2013.
    Temperature correction: −0.3 MPa/°C from 20 °C in-vitro reference.

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

    # Initialise to global water-reference so unlabelled voxels degrade gracefully
    pt_field    = np.full(p_field.shape, 28.2e6 + freq_shift, dtype=np.float64)
    sigma_field = np.full(p_field.shape, 0.96e6,              dtype=np.float64)

    for tissue in TISSUES:
        mask = label_vol == tissue.label
        if not mask.any():
            continue
        if not np.isfinite(tissue.pt_pa_1mhz):
            # AIR / BONE: no intrinsic cavitation — threshold set to +∞ so
            # erf-CDF → 0 regardless of applied pressure.
            pt_field[mask]    = 1.0e15
            sigma_field[mask] = 1.0
            continue
        pt_base = tissue.pt_pa_1mhz + freq_shift
        if T_field is not None:
            # Per-voxel temperature correction (vectorised over the mask)
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
    # CEM43 thermal dose via kw.cem43_at_temperatures (Sapareto & Dewey 1984).
    # R = 0.5 for T ≥ 43 °C, R = 0.25 for T < 43 °C; returns [min].
    if _HAS_PYKWAVERS:
        cem43 = np.asarray(
            kw.cem43_at_temperatures(T_steady.ravel(), sc.treatment_s)
        ).reshape(T_steady.shape)
    else:
        R = np.where(T_steady >= 43.0, 0.5, 0.25)
        cem43 = (R ** (43.0 - T_steady)) * (sc.treatment_s / 60.0)
    cem43[T_steady < 39.0] = 0.0
    return cem43, T_steady, T_transient


def predicted_lesion(p_field, props, sc, info, label_vol,
                     oar_masks: dict[str, np.ndarray] | None = None):
    # Thermal maps first: T_transient feeds the temperature correction for p_t.
    cem43, T_steady, T_transient = thermal_maps(p_field, props, sc)
    # Per-voxel tissue-specific cavitation probability.
    # Each voxel uses the threshold and σ of the tissue at that position
    # (LIVER, HCC, FAT, MUSCLE …) plus a −0.3 MPa/°C temperature correction
    # derived from T_transient (per-pulse heating before steady-state diffusion).
    # This replaces the global 28.2 MPa / σ=0.96 MPa water reference.
    pcav = cav_probability_tissue(p_field, label_vol, sc.f0, T_field=T_transient)
    coll = collapse_strength(p_field, sc.f0)

    # Regime-appropriate single-pulse dose map for Figure 14 row 1.
    # pcav is near-zero for shock-vapor (PNP << 28.2 MPa intrinsic threshold)
    # and irrelevant for sub-threshold (mechanism is inertial collapse, not
    # threshold nucleation). Provide the physically meaningful metric:
    #   intrinsic       → P_cav erf-CDF (Vlaisavljevich 2015)
    #   shock-vapor     → normalised T_transient (vapor seeding temperature,
    #                     Khokhlova 2011: nucleation at T ≥ 100°C focal voxel)
    #   sub-threshold   → normalised collapse strength (Vlaisavljevich 2018)
    if sc.regime == "intrinsic":
        dose_map_regime = pcav.astype(np.float32)
        regime_label = "single-pulse $P_{cav}$\n(intrinsic threshold)"
    elif sc.regime == "shock_vapor":
        dose_map_regime = np.clip(
            (T_transient - 37.0) / 63.0, 0.0, 1.0
        ).astype(np.float32)
        regime_label = "$T_{transient}$ / 100°C\n(vapor-seed probability)"
    else:  # subthreshold_cav
        dose_map_regime = np.clip(
            (coll - 1.0) / 9.0, 0.0, 1.0
        ).astype(np.float32)
        regime_label = "collapse strength (norm)\n($S_c / 10$, sub-threshold erosion)"

    pulses_per_pt = max(sc.pulses_per_point, 1)
    # Cavitation-cloud radius scales with overpressure (Vlaisavljevich
    # 2013, Maxwell 2013): r_cloud ≈ λ/4 at threshold, growing to ~λ/2
    # at p > 1.5 p_t. Dilating the threshold-exceeding region by this
    # radius models the bubble cloud expansion beyond the strict
    # p > p_t kernel and is the correct ablation-mask construction at
    # any grid spacing where dx > λ/4.
    lam_m = 1540.0 / sc.f0
    # Cloud radius uses focal-tissue threshold, not the global water reference.
    # Identify tissue at the focal point; look up tissue-specific p_t with
    # temperature correction from T_transient at that voxel.
    _fi0, _fi1, _fi2 = info["focus_idx"]
    _focus_label = int(label_vol[_fi0, _fi1, _fi2])
    _focus_tissue = next((t for t in TISSUES if t.label == _focus_label), LIVER)
    _focus_T_C    = float(T_transient[_fi0, _fi1, _fi2])
    pt_focal      = intrinsic_threshold_tissue_pa(_focus_tissue, sc.f0, _focus_T_C)
    if not np.isfinite(pt_focal):
        pt_focal = intrinsic_threshold(sc.f0)   # fallback for bone/air
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

    tumour = label_vol == HCC.label
    coords = np.argwhere(tumour)
    if len(coords) == 0:
        return mech | therm, {"pcav": pcav, "T_steady": T_steady, "T_transient": T_transient,
                              "cem43": cem43, "metrics": {}}
    cmin = coords.min(0); cmax = coords.max(0) + 1
    extent = cmax - cmin

    fwhm_lat_m, fwhm_ax_m = focal_fwhm(sc.f0)
    if mech.any() or therm.any():
        ps = np.argwhere(mech | therm)
        ps_extent = ps.max(0) - ps.min(0) + 1
        mech_hx = max(int(np.ceil(ps_extent[0] / 2.0)), 1)
        mech_hy = max(int(np.ceil(ps_extent[1] / 2.0)), 1)
        mech_hz = max(int(np.ceil(ps_extent[2] / 2.0)), 1)
    else:
        mech_hx = mech_hy = mech_hz = 1

    # Pressure-/bubble-contour-based raster pitch.
    # The pitch is driven entirely by the per-shot mechanical footprint
    # half-extents, NOT by the FWHM (-3 dB amplitude contour).
    # Rationale: the FWHM is a diagnostic imaging convention (O'Neil
    # 1949, Penttinen 1976) that quantifies the beam width at -3 dB; it
    # does NOT describe where the acoustic pressure exceeds the regime-
    # specific cavitation threshold.  Using FWHM as an upper bound
    # artificially tightens the lateral pitch for ms regimes (whose
    # cavitation cloud from collapse_strength ≥ 5 or T ≥ 100°C is
    # WIDER than the -3 dB contour), under-populates the raster, and
    # leaves inter-spot gaps that the dose model would miss.
    # Load-bearing axis: lateral (minimum of the 3 half-extents), since
    # the axial focal cigar extends far beyond the therapeutic zone and
    # its periphery contributes negligibly to the per-shot lesion.  All
    # three axes share the same pitch so the raster grid is isotropic
    # and no direction is preferentially under-covered.
    # 50% overlap: adjacent footprints share their half-extent at the
    # midpoint, guaranteeing every inter-shot voxel is reached by at
    # least one shot's cloud.
    h_min = min(mech_hx, mech_hy, mech_hz)
    pitch_x = pitch_y = pitch_z = max(int(round(0.5 * h_min)), 1)

    # Erosion of the tumour by one pitch so that a raster centre
    # placed anywhere in valid_centres has its per-shot cloud
    # (half-extent = h_min) reaching the tumour boundary.  Eroding
    # by the PITCH (= 0.4 × h_min) rather than the full half-extent
    # keeps the raster centres close enough to the boundary that
    # the per-shot footprint can cover the rim; eroding by the full
    # half-extent over-confines, leaving the boundary unsampled.
    # Fallback: if the tumour is smaller than one pitch in any axis,
    # halve the erosion radius progressively and ultimately fall back
    # to the bare tumour mask (accepting spillover).
    def _try_erode(rx, ry, rz):
        if rx == 0 and ry == 0 and rz == 0:
            return tumour.copy()
        struct = np.ones((2*rx + 1, 2*ry + 1, 2*rz + 1), dtype=bool)
        return binary_erosion(tumour, structure=struct)
    valid_centres = _try_erode(pitch_x, pitch_y, pitch_z)
    if not valid_centres.any():
        # Progressively relax the erosion radius using the per-shot
        # mechanical half-extents so each shrink_factor still
        # corresponds to a physically meaningful spatial constraint.
        for shrink_factor in (0.66, 0.33, 0.0):
            rx = max(int(round(mech_hx * shrink_factor)), 0)
            ry = max(int(round(mech_hy * shrink_factor)), 0)
            rz = max(int(round(mech_hz * shrink_factor)), 0)
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

    # If erosion leaves fewer than 5 % of the tumour voxels as valid
    # centres, the raster would be trivially sparse (FPS terminates in
    # 1–2 shots because all remaining valid points cluster near the
    # centroid). Fall back to the full tumour interior so FPS fills it
    # properly. The boundary spillover this permits is bounded by the
    # per-shot cloud half-extent minus the pitch — acceptable when the
    # tumour is smaller than the per-shot footprint.
    min_valid_frac = max(10, int(0.05 * max(int(tumour.sum()), 1)))
    if int(valid_local.sum()) < min_valid_frac:
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

    # OAR dose-footprint accounting
    # ─────────────────────────────────────────────────────────────────
    # Raster points are NOT excluded based on OAR proximity.  Removing
    # shots reduces tumour coverage without eliminating OAR dose —
    # adjacent shots still deposit dose via their tails, so exclusion
    # trades coverage loss for negligible OAR benefit.
    # The correct instrument is dose monitoring: every FPS raster point
    # is retained, the accumulated cav_dose map is computed for the full
    # raster, and the per-OAR dose metrics below quantify what each
    # structure receives.  Non-zero OAR dose is flagged for clinical
    # review; the decision to reduce pulses or re-plan belongs to the MDT,
    # not to an automatic point-exclusion filter.
    #
    # n_oar_footprint_overlap: raster points whose per-shot mechanical
    # footprint could reach an absolute OAR voxel outside the GTV.
    # Reported in metrics; does not affect raster placement.
    n_oar_footprint_overlap = 0
    n_oar_encasing = 0
    oar_reach: np.ndarray | None = None
    if oar_masks is not None and raster_grid.any():
        combined_oar = oar_masks.get("combined",
                                      np.zeros_like(tumour, dtype=bool))
        oar_outside_gtv = combined_oar & ~tumour
        if oar_outside_gtv.any():
            se = np.ones((2 * mech_hx + 1,
                          2 * mech_hy + 1,
                          2 * mech_hz + 1), dtype=bool)
            oar_reach = binary_dilation(oar_outside_gtv, structure=se)
            n_oar_footprint_overlap = int((raster_grid & oar_reach).sum())
        # Count GTV-interior OAR overlap (vessel-encasing tumour flagging).
        oar_inside_gtv = combined_oar & tumour
        if oar_inside_gtv.any() and raster_grid.any():
            se_g = np.ones((2 * mech_hx + 1,
                            2 * mech_hy + 1,
                            2 * mech_hz + 1), dtype=bool)
            oar_reach_g = binary_dilation(oar_inside_gtv, structure=se_g)
            n_oar_encasing = int((raster_grid & oar_reach_g).sum())

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

    # Cavitation-dose heatmap — accumulated 1 - (1 - p_per_pulse)^N over
    # all raster points × pulses_per_pt.  Use the SAME regime-appropriate
    # per-pulse field as dose_map_regime so that the cav_dose map is
    # consistent with what row 1 of fig14 displays:
    #   intrinsic     → P_cav erf-CDF  (meaningful: PNP ≥ threshold)
    #   shock_vapor   → T_transient/100°C (vapour-seed probability;
    #                   pcav is near-zero because PNP << 28.2 MPa
    #                   intrinsic threshold — using pcav here produces
    #                   a flat-zero cav_dose map for this regime)
    #   sub-threshold → collapse strength / 10
    per_pulse = np.clip(dose_map_regime.astype(np.float32), 0.0, 0.99)
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
    # typical for clinical robotic stages. Add it to the electronic-sonication
    # time so the treatment_s is honest about wall-clock duration.
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
    raster_pitch_mm = float(pitch_x * info["dx"] * 1e3)  # isotropic by construction
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
        # Per-shot cavitation-cloud radius (load-bearing lateral half-extent).
        # Downstream plots use this in place of FWHM for overlay circles so
        # the displayed circles correspond to the actual ablation footprint.
        "mech_radius_mm": float(h_min * info["dx"] * 1e3),
        "mech_hx_mm": float(mech_hx * info["dx"] * 1e3),
        "mech_hy_mm": float(mech_hy * info["dx"] * 1e3),
        "mech_hz_mm": float(mech_hz * info["dx"] * 1e3),
        # OAR safety metrics — overlap count only; no raster points excluded
        "n_oar_footprint_overlap_pts": n_oar_footprint_overlap,
        "n_oar_encasing_raster_pts":   n_oar_encasing,
    }
    metrics["fwhm_lat_mm"] = fwhm_lat_m * 1e3
    metrics["fwhm_ax_mm"] = fwhm_ax_m * 1e3
    # Per-OAR dose: max cavitation dose received by each OAR.
    # Values > 0.05 warrant clinical review of the treatment plan.
    if oar_masks is not None:
        for spec in OAR_SPECS:
            m_oar = oar_masks.get(spec.name)
            if m_oar is not None and m_oar.any():
                metrics[f"oar_{spec.name}_dose_max"] = float(
                    cav_dose[m_oar].max())
                metrics[f"oar_{spec.name}_dose_mean"] = float(
                    cav_dose[m_oar].mean())
            else:
                metrics[f"oar_{spec.name}_dose_max"] = 0.0
                metrics[f"oar_{spec.name}_dose_mean"] = 0.0
    return lesion, {"pcav": pcav, "T_steady": T_steady, "T_transient": T_transient,
                    "coll": coll,
                    "cem43": cem43, "metrics": metrics,
                    "raster_grid": raster_grid_out, "per_shot_mask": mech | therm,
                    "cav_dose": cav_dose,
                    "dose_map_regime": dose_map_regime,
                    "regime_label": regime_label,
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
                           pulses_per_pt: int,
                           per_shot_mask: np.ndarray | None = None,
                           ) -> list[tuple[int, int, int]]:
    """Greedy adaptive ordering with footprint-aware pre-lesion accounting.

    Scores each candidate raster point by the MAXIMUM log_acc (= least
    accumulated dose = most urgently under-treated) across ALL voxels in
    its per-shot mechanical footprint, not just at its centre voxel.

    Why footprint-aware scoring matters
    ────────────────────────────────────
    The raster pitch is set to 50 % of the per-shot footprint half-extent,
    so adjacent footprints overlap substantially.  After several shots, a
    candidate's centre voxel may already be dosed by beam tails from
    neighbours while the candidate's footprint still covers genuinely
    under-treated rim tissue.  Centre-only scoring misses this: it ranks
    the candidate low (centre is dosed) even though the shot would deliver
    marginal benefit to the rim.  Footprint max-log_acc correctly captures
    the worst-dosed voxel within the candidate's reach.

    Pre-lesioned tissue: acoustic transparency, not shadowing
    ──────────────────────────────────────────────────────────
    Once a region accumulates cav_dose ≥ 0.95, the tissue is liquefied
    (lysed cell debris + saline).  Acoustically this debris is MORE
    transparent than intact HCC:
        α_ablated ≈ 0.5 dB/MHz/cm  vs  α_HCC ≈ 12.5 dB/MHz/cm
        Z_ablated ≈ 1.54 GPa·s/m³  vs  Z_HCC ≈ 1.67 GPa·s/m³
    Shots aimed at adjacent INTACT tissue through a pre-lesioned zone
    therefore experience reduced attenuation — not acoustic shadowing.
    The residual-bubble shadowing concern (Maxwell 2013) applies only to
    TRANSIENT bubble clouds during and immediately after each pulse
    (within 5–50 ms of the pulse), not to the static liquefied debris
    between shots.  Since clinical histotripsy PRF ≥ 1 Hz, each shot
    fires into acoustically cleared tissue; no inter-shot attenuation
    correction is applied to beam paths through pre-lesioned zones.

    Saturation de-prioritisation
    ─────────────────────────────
    Voxels already at cav_dose ≥ 0.995 have log_acc ≪ 0 (very negative),
    so candidates centred on or covering only saturated tissue rank last
    by construction — redundant shots are naturally suppressed without
    explicit exclusion logic.

    Parameters
    ----------
    raster_grid    : bool (Nx, Ny, Nz) — raster centre positions
    per_pulse      : float32 (Nx, Ny, Nz) — regime dose per pulse ∈ [0, 1]
    focus_idx      : (ix, iy, iz) nominal focal voxel
    pulses_per_pt  : pulses delivered per raster point
    per_shot_mask  : bool (Nx, Ny, Nz) — mechanical lesion footprint of
                     one shot centred on focus_idx (i.e. ``mech | therm``
                     from ``predicted_lesion``).  If None, falls back to
                     centre-voxel-only scoring (backward-compatible).
    """
    pts = np.argwhere(raster_grid)
    if len(pts) == 0:
        return []
    fx_arr = np.array(focus_idx)
    nx, ny, nz = per_pulse.shape
    grid_shape = np.array([nx, ny, nz])
    log_safe = np.log(np.clip(1.0 - per_pulse.astype(np.float32), 1e-6, 1.0))
    log_acc = np.zeros_like(log_safe)

    # Pre-compute per-shot footprint offsets relative to the focal voxel.
    # When a candidate raster point c is scored, its footprint voxels are
    # at c + fprint_off (clipped to the grid boundary).
    fprint_off: np.ndarray | None = None
    if per_shot_mask is not None and per_shot_mask.any():
        fprint_off = np.argwhere(per_shot_mask) - fx_arr   # (M, 3) offsets

    remaining = pts.tolist()
    order: list[tuple[int, int, int]] = []

    while remaining:
        if not order:
            # Seed: raster centroid (geometrically central, order-independent).
            centroid = np.argwhere(raster_grid).mean(axis=0)
            d = np.linalg.norm(np.array(remaining) - centroid, axis=1)
            idx_pick = int(np.argmin(d))
        else:
            cands = np.array(remaining)  # (K, 3)

            if fprint_off is not None:
                # Footprint-aware score for each candidate c:
                #   score(c) = max( log_acc[c + fprint_off ∩ grid] )
                # max log_acc is the LEAST-dosed footprint voxel (log_acc ≤ 0;
                # max = closest to 0 = fewest accumulated events = most urgent).
                # Choosing argmax(scores) selects the candidate whose footprint
                # covers the most under-treated tissue in the tumour.
                scores = np.empty(len(cands), dtype=np.float32)
                for j, c in enumerate(cands):
                    tgt = c + fprint_off                          # (M, 3)
                    ok = ((tgt >= 0) & (tgt < grid_shape)).all(axis=1)
                    tgt_ok = tgt[ok]
                    if len(tgt_ok):
                        scores[j] = float(
                            log_acc[tgt_ok[:, 0], tgt_ok[:, 1], tgt_ok[:, 2]].max())
                    else:
                        scores[j] = float(log_acc[c[0], c[1], c[2]])
            else:
                # Centre-voxel fallback (backward-compatible, no footprint).
                scores = np.array([log_acc[c[0], c[1], c[2]] for c in cands],
                                  dtype=np.float32)

            idx_pick = int(np.argmax(scores))

        pick = remaining.pop(idx_pick)
        order.append(tuple(int(c) for c in pick))

        # Accumulate log-space dose from this shot into the running field.
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


def _raster_pts_by_volume(raster_grid, gtv, ctv, ptv, y0, y1, z0, z1, info):
    """Return raster point yz-coordinates (projected over x) split by
    planning volume membership. All points in the current raster are
    placed inside the GTV by the FPS algorithm (the erosion enforces
    this); projecting to the focal slice shows the spatial relationship
    between the shot cloud and the planning volumes.

    Returns
    -------
    dict mapping volume name → (y_mm_array, z_mm_array, colour, size)
    """
    nx = raster_grid.shape[0]
    x_coords = np.argwhere(raster_grid.any(axis=(1, 2))).ravel()
    x_min = int(x_coords.min()) if len(x_coords) else 0
    x_max = int(x_coords.max()) + 1 if len(x_coords) else nx
    rg_crop = raster_grid[x_min:x_max, y0:y1, z0:z1]

    # Project to 2-D and label each projected voxel by the innermost
    # volume it belongs to (GTV ⊂ CTV ⊂ PTV).
    out = {}
    for vname, vmask, clr, sz in (
        ("GTV", gtv, "cyan",   22),
        ("CTV", ctv, "yellow", 16),
        ("PTV", ptv, "white",  12),
    ):
        vm_crop = vmask[x_min:x_max, y0:y1, z0:z1]
        proj = (rg_crop & vm_crop).any(axis=0)
        # Subtract inner volumes so each point is labelled once
        if vname == "CTV":
            inner = gtv[x_min:x_max, y0:y1, z0:z1]
            proj &= ~(rg_crop & inner).any(axis=0)
        elif vname == "PTV":
            inner = ctv[x_min:x_max, y0:y1, z0:z1]
            proj &= ~(rg_crop & inner).any(axis=0)
        ys, zs = np.where(proj)
        y_mm = np.array([info["y_axis"][y0 + yy] * 1e3 for yy in ys])
        z_mm = np.array([info["z_axis"][z0 + zz] * 1e3 for zz in zs])
        out[vname] = (y_mm, z_mm, clr, sz)
    return out


def _prelesion_mask_2d(
    log_acc_2d: np.ndarray,
    dose_threshold: float = 0.90,
) -> np.ndarray:
    """Boolean mask of substantially ablated voxels in a 2D log-dose slice.

    ``log_acc_2d`` holds the accumulated log-survival × pulses_per_pt for
    each 2-D voxel summed over all shots fired so far:

        log_acc_2d[y, z] = Σ_shots  log(1 − per_pulse[y,z]·k_steer) · pulses

    The cavitation-dose probability at each voxel is

        cav_dose(y, z) = 1 − exp( log_acc_2d[y, z] )   ∈ [0, 1]

    Voxels where cav_dose ≥ dose_threshold (default 0.90) are classed as
    pre-lesioned: tissue is expected to be liquefied and further shots at
    that location deliver zero marginal mechanical cavitation benefit.

    Acoustic note
    ─────────────
    Pre-lesioned tissue (liquefied HCC debris) is MORE acoustically
    transparent than intact HCC (α_ablated ≈ 0.5 vs 12.5 dB/MHz/cm;
    Z_ablated ≈ 1.54 vs 1.67 GPa·s/m³).  Shots aimed at adjacent intact
    tissue through a pre-lesioned zone experience REDUCED attenuation —
    not shadowing.  This mask is therefore used only for:
      1. Visual feedback in the animation (lime-green overlay).
      2. Scoring context in adaptive ordering — saturated voxels yield
         strongly negative log_acc, so they rank last without needing
         explicit exclusion.
    No attenuation penalty is applied to beam paths through ablated tissue.

    Parameters
    ----------
    log_acc_2d     : (Ny, Nz) float32 — accumulated log_safe × pulses; ≤ 0
    dose_threshold : cavitation-dose fraction for ablation call; default 0.90

    Returns
    -------
    mask : (Ny, Nz) bool — True where cav_dose ≥ dose_threshold
    """
    # log_acc_2d already incorporates pulses_per_pt so exp gives cav_dose directly.
    return (1.0 - np.exp(log_acc_2d)) >= dose_threshold


def make_sonication_animation(ct, label_vol, info, results, scenarios,
                              gtv, ctv, ptv,
                              oar_masks: dict[str, np.ndarray] | None = None,
                              n_frames: int = 80, fps: int = 12) -> str:
    """Render a 3-panel animated GIF of cumulative cavitation-dose
    delivery with GTV / CTV / PTV planning-volume overlays.

    Each panel's timeline runs from 0 → 100 % of that scenario's
    actual treatment time.  The dose is computed incrementally from
    the serpentine raster path.

    Overlays added per panel
    ────────────────────────
    • GTV contour (cyan) — the imaging-defined ablation target; all
      raster shots are placed WITHIN this boundary by the FPS planner.
    • CTV contour (yellow) — GTV + 3 mm spherical margin for
      microscopic disease; beam tails from GTV-placed shots deposit
      dose here, decaying with distance.
    • PTV contour (white) — CTV + 3 mm for residual motion /
      positioning uncertainty; receives only far beam-tail dose.
    • Ghost dots — all planned raster points projected to the focal
      y-z slice, colour-coded cyan (GTV) / yellow (CTV only) /
      white (PTV only).  Because shots are confined to the GTV
      by construction, ALL dots are cyan — making the volume-
      confinement of the raster plan immediately visible.
    • Active-shot × marker — tracks the current firing location.

    The progress box reports: elapsed time, shot counter, per-volume
    raster-point count, and whether any shots were steered to CTV/PTV
    (currently always 0 — confirming GTV confinement).

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

    # Pre-compute 2-D planning-volume slices for static contour overlays.
    gtv_sl = gtv[fx, y0:y1, z0:z1].astype(float)
    ctv_sl = ctv[fx, y0:y1, z0:z1].astype(float)
    ptv_sl = ptv[fx, y0:y1, z0:z1].astype(float)

    from matplotlib.colors import LogNorm
    paths = {}
    per_pulse_slice = {}
    anchors_per_path = {}
    ny_loc = y1 - y0
    nz_loc = z1 - z0
    for sc in scenarios:
        paths[sc.name] = serpentine_order(results[sc.name]["raster_grid"])
        anchors_per_path[sc.name], _ = mechanical_walk(
            paths[sc.name], info["focus_idx"], info["dx"], sc.f0)
        r = results[sc.name]
        if sc.regime == "subthreshold_cav":
            coll_sl = collapse_strength(r["p_field"][fx, y0:y1, z0:z1], sc.f0)
            per_pulse = np.clip((coll_sl - 1.0) / 9.0, 0.0, 1.0).astype(np.float32)
        elif sc.regime == "shock_vapor":
            ps_mask = r["per_shot_mask"][fx, y0:y1, z0:z1].astype(np.float32)
            T_tr = r["T_transient"][fx, y0:y1, z0:z1]
            T_norm = np.clip((T_tr - 37.0) / max((T_tr.max() - 37.0), 1.0), 0.0, 1.0)
            per_pulse = (ps_mask * T_norm).astype(np.float32)
        else:
            per_pulse = np.clip(r["pcav"][fx, y0:y1, z0:z1], 0.0, 1.0).astype(np.float32)
        per_pulse_slice[sc.name] = per_pulse

    # Shared log-norm scale.
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

    # Pre-compute log-safe 2D slices for the parallel log-space dose accumulator
    # used by the pre-lesion detector.  Stored per scenario because per_pulse
    # differs across regimes.
    log_safe_prelesion_slice: dict[str, np.ndarray] = {}
    for sc in scenarios:
        pp = per_pulse_slice[sc.name]
        log_safe_prelesion_slice[sc.name] = np.log(
            np.clip(1.0 - pp, 1e-9, 1.0))          # (ny_loc, nz_loc), ≤ 0

    hot = plt.get_cmap("inferno")
    dose_colors = [(*hot(t)[:3], 0.0 if t < 0.02 else min(0.85, 0.15 + t * 0.95))
                   for t in np.linspace(0, 1, 256)]
    dose_cmap = LinearSegmentedColormap.from_list("cav_dose_anim", dose_colors)

    base_ct = ct[fx, y0:y1, z0:z1]
    # GTV pixel count on focal slice (for ablation-progress percentage).
    gtv_sl_bool = gtv_sl.astype(bool)
    n_gtv_px = int(gtv_sl_bool.sum())

    fig, axes = plt.subplots(1, len(scenarios), figsize=(14.0, 5.8))
    if len(scenarios) == 1:
        axes = [axes]

    artists = []
    for ax, sc in zip(axes, scenarios):
        ax.imshow(base_ct, cmap="gray", vmin=-200, vmax=300,
                  extent=extent_crop, aspect="equal")

        # ── Static planning-volume contours ────────────────────────────
        # Drawn once; remain visible throughout the animation.
        for sl_mask, clr, lw in (
            (ptv_sl, "white",  0.7),
            (ctv_sl, "yellow", 0.9),
            (gtv_sl, "cyan",   1.1),
        ):
            if sl_mask.any():
                ax.contour(np.flipud(sl_mask), levels=[0.5],
                           colors=clr, linewidths=lw, extent=extent_crop)

        # ── Ghost raster-point cloud (colour = innermost volume) ───────
        # All current shots are inside the GTV by FPS construction;
        # dots outside GTV would indicate a bug in the raster planner.
        rpts = _raster_pts_by_volume(
            results[sc.name]["raster_grid"], gtv, ctv, ptv,
            y0, y1, z0, z1, info)
        n_gtv_pts = len(rpts["GTV"][0])
        n_ctv_pts = len(rpts["CTV"][0])
        n_ptv_pts = len(rpts["PTV"][0])
        for vname, (y_mm, z_mm, clr, sz) in rpts.items():
            if len(y_mm):
                ax.scatter(z_mm, y_mm, c=clr, s=sz, alpha=0.35,
                           edgecolors="none", zorder=4,
                           label=f"{vname}: {len(y_mm)} pts")

        # ── Dose heatmap (updated each frame) ─────────────────────────
        dose_im = ax.imshow(
            np.full((ny_loc, nz_loc), shared_vmin * 0.5, dtype=np.float32),
            cmap=dose_cmap, norm=shared_norm,
            extent=extent_crop, aspect="equal", interpolation="nearest",
            zorder=3)

        # ── Pre-lesion overlay (lime-green RGBA, updated each frame) ───
        # Marks voxels where cumulative cav_dose ≥ 0.90: tissue is fully
        # ablated, further shots deliver no marginal cavitation benefit.
        # Pre-lesioned debris is acoustically MORE transparent than intact
        # HCC (lower α, lower Z), so no beam-path attenuation penalty is
        # shown — only the "ablation completed" spatial information.
        prelesion_rgba = np.zeros((ny_loc, nz_loc, 4), dtype=np.float32)
        prelesion_im = ax.imshow(prelesion_rgba, extent=extent_crop,
                                 aspect="equal", interpolation="nearest",
                                 zorder=6)

        # ── Static OAR contours (drawn once; zorder above dose, below marker) ─
        if oar_masks is not None:
            _draw_oar_contours(ax, oar_masks, info, fx,
                               y0, y1, z0, z1, extent_crop, lw=0.80)

        progress = ax.text(
            0.02, 0.97, "", transform=ax.transAxes, fontsize=7.0,
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.65, edgecolor="none",
                      boxstyle="round,pad=0.3"),
            color="white")
        title = ax.set_title("", fontsize=8.5)
        ax.set(xlabel="z [mm]", ylabel="y [mm]")

        # Per-panel legend: GTV / CTV / PTV + shot counts + pre-lesion key + OARs.
        legend_handles = [
            Line2D([0], [0], color="cyan",              lw=1.1,
                   label=f"GTV ({n_gtv_pts} shots)"),
            Line2D([0], [0], color="yellow",            lw=0.9,
                   label=f"CTV +5 mm HOPE4LIVER ({n_ctv_pts} shots)"),
            Line2D([0], [0], color="white",             lw=0.7,
                   label=f"PTV +8 mm total ({n_ptv_pts} shots)"),
            Line2D([0], [0], color=(0.2, 0.9, 0.35, 1), lw=5.0,
                   label="pre-lesioned (dose ≥ 90%)"),
        ]
        if oar_masks is not None:
            legend_handles += _oar_legend_handles()
        ax.legend(handles=legend_handles, fontsize=6.0, loc="lower right",
                  framealpha=0.7,
                  facecolor="black", labelcolor="white", edgecolor="gray")

        artists.append({
            "ax": ax, "dose_im": dose_im, "prelesion_im": prelesion_im,
            "events_acc": np.zeros((ny_loc, nz_loc), dtype=np.float32),
            # log-space accumulator for pre-lesion detection:
            # log_acc_2d = Σ_shots log(1 − per_pulse·k) · pulses; ≤ 0
            # cav_dose = 1 − exp(log_acc_2d)
            "log_acc_2d": np.zeros((ny_loc, nz_loc), dtype=np.float32),
            "shots_drawn": 0, "progress": progress, "title": title,
            "current_marker": None, "sc": sc,
            "n_shots": len(paths[sc.name]),
            "n_gtv_pts": n_gtv_pts, "n_ctv_pts": n_ctv_pts, "n_ptv_pts": n_ptv_pts,
        })

    fig.colorbar(artists[-1]["dose_im"], ax=axes,
                 fraction=0.025, pad=0.02, shrink=0.85,
                 label="cumulative cavitation events (log scale)")

    def init():
        return ([a["dose_im"] for a in artists]
                + [a["prelesion_im"] for a in artists]
                + [a["progress"] for a in artists])

    def animate(frame_idx: int):
        progress_frac = frame_idx / max(n_frames - 1, 1)
        changed = []
        for art in artists:
            sc = art["sc"]
            n_shots_total = art["n_shots"]
            shots_target = min(int(round(progress_frac * n_shots_total)), n_shots_total)
            path = paths[sc.name]
            per_pulse = per_pulse_slice[sc.name]
            log_safe_pp = log_safe_prelesion_slice[sc.name]
            pulses = max(sc.pulses_per_point, 1)

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
                    # Parallel log-space accumulation for pre-lesion tracking.
                    # log(1 − per_pulse·k) used exactly (no first-order approx).
                    eff_pp = np.clip(
                        per_pulse[src_y0:src_y1, src_z0:src_z1] * k_steer,
                        0.0, 0.9999)
                    art["log_acc_2d"][dst_y0:dst_y1, dst_z0:dst_z1] += \
                        np.log1p(-eff_pp) * pulses
                art["shots_drawn"] += 1

            art["dose_im"].set_data(art["events_acc"])
            changed.append(art["dose_im"])

            # Pre-lesion overlay: lime-green where cav_dose ≥ 0.90.
            prelesion_mask = _prelesion_mask_2d(art["log_acc_2d"])
            prelesion_rgba = np.zeros((ny_loc, nz_loc, 4), dtype=np.float32)
            prelesion_rgba[prelesion_mask, 0] = 0.20
            prelesion_rgba[prelesion_mask, 1] = 0.90
            prelesion_rgba[prelesion_mask, 2] = 0.35
            prelesion_rgba[prelesion_mask, 3] = 0.38   # 38% opacity
            art["prelesion_im"].set_data(prelesion_rgba)
            changed.append(art["prelesion_im"])

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
                        markersize=9, mew=1.8, zorder=10)

            sc_min = sc.treatment_s / 60.0
            elapsed = progress_frac * sc_min
            done = shots_target >= n_shots_total
            eff_prf = sc.prf * sc.interleave_subspots
            n_ablated = int(prelesion_mask.sum())
            pct_ablated = 100.0 * n_ablated / max(n_gtv_px, 1)
            art["progress"].set_text(
                f"t = {elapsed:.2f} / {sc_min:.1f} min\n"
                f"shots: {shots_target}/{n_shots_total}\n"
                f"GTV: {art['n_gtv_pts']} pts  CTV: {art['n_ctv_pts']}  PTV: {art['n_ptv_pts']}\n"
                f"Pre-lesioned: {n_ablated}/{n_gtv_px} px ({pct_ablated:.0f}% GTV)\n"
                f"{'■ COMPLETE' if done else f'PRF {eff_prf:.0f} Hz (interleaved)'}"
            )
            art["title"].set_text(
                f"{sc.label}\n"
                f"{n_shots_total} shots inside GTV → beam tails cover CTV/PTV"
            )
            changed.append(art["progress"])
            changed.append(art["title"])
        return changed

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, blit=False)
    fig.suptitle(
        "Histotripsy sonication — cumulative cavitation-dose heatmap with GTV/CTV/PTV planning volumes\n"
        "cyan=GTV  yellow=CTV +5 mm (HOPE4LIVER)  white=PTV +8 mm  "
        "lime-green = pre-lesioned (cav_dose ≥ 90%, tissue liquefied)\n"
        "Ghost dots = all planned raster points (cyan = inside GTV).  "
        "Beam tails from GTV shots fill CTV/PTV with decaying dose.",
        y=1.01, fontsize=9)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.05, right=0.92, wspace=0.20)
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


def _sphere_kernel(r: int) -> np.ndarray:
    """Spherical binary structuring element of radius r voxels."""
    g = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (g[0] ** 2 + g[1] ** 2 + g[2] ** 2) <= (r + 0.5) ** 2


def margin_weighted_dose(
    cav_dose: np.ndarray,
    gtv: np.ndarray,
    ptv: np.ndarray,
    info: dict,
    ctv_margin_m: float = 5.0e-3,
    ptv_margin_m: float = 3.0e-3,
) -> np.ndarray:
    """Distance-weighted dose display for CTV / PTV margin annuli.

    Inside the GTV the raw accumulated cav_dose is returned unchanged —
    raster shots target the GTV so those voxels carry the full therapeutic
    dose.  In the CTV and PTV annuli a linear weight

        w(d) = 1 − d / (ctv_margin_m + ptv_margin_m)

    tapers the dose from the GTV surface (w = 1) to the PTV outer edge
    (w = 0).  This makes the clinical intent explicit: maximum treatment
    in the gross tumour, graded margin coverage for microscopic disease /
    setup uncertainty, and no dose shown outside the planning boundary.

    ``distance_transform_edt(~gtv)`` returns 0 for GTV voxels and the
    Euclidean distance to the nearest GTV voxel for everything outside,
    so ``w = 1`` everywhere inside the GTV by construction.

    Parameters
    ----------
    cav_dose        : (Nx, Ny, Nz) float32 raw log-accumulated cavitation dose
    gtv, ptv        : bool arrays shaped (Nx, Ny, Nz)
    info            : simulation info dict (key ``"dx"`` in metres)
    ctv_margin_m    : GTV → CTV isotropic margin (default 5 mm, HOPE4LIVER standard)
    ptv_margin_m    : CTV → PTV additional margin for motion/positioning (default 3 mm)

    Returns
    -------
    weighted : (Nx, Ny, Nz) float32 in [0, 1]
    """
    total_margin_m = ctv_margin_m + ptv_margin_m          # 6 mm
    dist_m = distance_transform_edt(~gtv) * info["dx"]    # 0 inside GTV

    weight = np.ones(cav_dose.shape, dtype=np.float32)
    outside_gtv_in_ptv = (~gtv) & ptv
    if outside_gtv_in_ptv.any():
        weight[outside_gtv_in_ptv] = np.clip(
            1.0 - dist_m[outside_gtv_in_ptv] / total_margin_m,
            0.0, 1.0,
        ).astype(np.float32)
    weight[~ptv] = 0.0

    return np.clip(cav_dose * weight, 0.0, 1.0)


def compute_gtv_ctv_ptv(
    label_vol: np.ndarray,
    dx_m: float,
    ctv_margin_m: float = 5.0e-3,
    ptv_margin_m: float = 3.0e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute radiotherapy-style target volumes for histotripsy planning.

    GTV — Gross Tumour Volume: the imaging-defined HCC region as labelled
          in the native segmentation (label == HCC.label). Hard ablation
          target — every voxel must receive therapeutic dose.

    CTV — Clinical Target Volume: GTV + ctv_margin_m spherical expansion
          for microscopic disease extension beyond the imaging contour.
          5 mm per HOPE4LIVER trial protocol (NCT04573881, Laimer 2021):
          minimum 5 mm margin to achieve 90 % local tumour control at 1 year
          for HCC and liver metastases. Image-guided systems with real-time
          ultrasound tracking can maintain this margin reliably (Worlikar 2020).

    PTV — Planning Target Volume: CTV + ptv_margin_m for residual
          geometric uncertainty (respiratory motion residual, mechanical
          positioning). 3 mm additional from CTV gives a total GTV-to-PTV
          margin of 8 mm for image-guided systems (Liu 2022).

    Parameters
    ----------
    label_vol : (Nx, Ny, Nz) int8
    dx_m      : voxel edge length (isotropic)
    ctv_margin_m : GTV → CTV isotropic expansion (default 5 mm, HOPE4LIVER)
    ptv_margin_m : CTV → PTV additional expansion (default 3 mm)

    Returns
    -------
    gtv, ctv, ptv : bool arrays shaped (Nx, Ny, Nz)
    """
    gtv = (label_vol == HCC.label)
    ctv_r = max(int(round(ctv_margin_m / dx_m)), 1)
    ptv_r = max(int(round((ctv_margin_m + ptv_margin_m) / dx_m)), 1)
    ctv = binary_dilation(gtv, structure=_sphere_kernel(ctv_r))
    ptv = binary_dilation(gtv, structure=_sphere_kernel(ptv_r))
    return gtv, ctv, ptv


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


def plot_transducer_placement_3d(label_vol: np.ndarray, info: dict) -> str:
    """3D scatter: HistoSonics 50 mm / 120 mm-ROC bowl placed anterior to the
    patient skin (sub-costal approach), targeting the HCC tumour centroid.

    All bowl elements are placed at depth < 0 (outside the patient body).
    The skin surface begins at depth = 0 mm (anterior face of the volume).

    Refs: HistoSonics Edison system; Vlaisavljevich 2015 (JASA 138:1864);
    Bilic 2023 (Med Image Anal 84:102680).
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
    R_f: float = 120.0e-3  # radius of curvature (m)
    a:   float = 25.0e-3   # aperture half-diameter (m)
    theta_max = float(np.arcsin(a / R_f))  # ≈ 12.0°

    ex_list: list[float] = [x_focus - R_f]
    ey_list: list[float] = [y_focus]
    ez_list: list[float] = [z_focus]
    for ring in range(1, 9):
        th = theta_max * ring / 8.0
        n_phi = max(8, int(round(40.0 * np.sin(th) / np.sin(theta_max))))
        phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        ex_list.extend([float(x_focus - R_f * np.cos(th))] * n_phi)
        ey_list.extend((y_focus + R_f * np.sin(th) * np.cos(phis)).tolist())
        ez_list.extend((z_focus + R_f * np.sin(th) * np.sin(phis)).tolist())
    ex = np.array(ex_list)
    ey = np.array(ey_list)
    ez = np.array(ez_list)

    # ── Body (skin) surface — anterior face per lateral position ──────────
    body_bool = label_vol != AIR.label
    first_ix  = np.argmax(body_bool, axis=0)
    has_body  = body_bool.any(axis=0)
    iy_grid, iz_grid = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
    sx_raw = x_axis[first_ix[has_body]]
    sy_raw = y_axis[iy_grid[has_body]]
    sz_raw = z_axis[iz_grid[has_body]]
    rng = np.random.default_rng(seed=0)
    if len(sx_raw) > 2500:
        sel = rng.choice(len(sx_raw), 2500, replace=False)
        sx_raw, sy_raw, sz_raw = sx_raw[sel], sy_raw[sel], sz_raw[sel]

    # ── Liver + HCC tumour surface voxels ─────────────────────────────────
    organ_mask = (label_vol == LIVER.label) | (label_vol == HCC.label)
    tumor_mask = label_vol == HCC.label
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
               c="#8b6914", s=4, alpha=0.40, linewidths=0, zorder=2, label="Liver")
    ax.scatter(tx * 1e3, ty * 1e3, tz * 1e3,
               c="#c0392b", s=14, alpha=0.90, linewidths=0, zorder=3, label="HCC tumour")
    ax.scatter(ex * 1e3, ey * 1e3, ez * 1e3,
               c="#2c3e50", s=18, alpha=0.85, edgecolors="white", linewidths=0.4,
               zorder=4, label="Transducer bowl (50 mm / 120 mm-ROC)")
    ax.scatter([(x_focus - R_f) * 1e3], [y_focus * 1e3], [z_focus * 1e3],
               c="limegreen", s=80, marker="o", zorder=5, label="Bowl apex (sub-costal coupling)")
    ax.scatter([x_focus * 1e3], [y_focus * 1e3], [z_focus * 1e3],
               c="cyan", s=120, marker="+", linewidths=2.5, zorder=5, label="Focus (HCC centroid)")

    ax.set_xlabel("Depth (mm, anterior →)")
    ax.set_ylabel("Lateral Y (mm)")
    ax.set_zlabel("Sup–Inf Z (mm)")
    ax.set_title(
        "HistoSonics 50 mm / 120 mm-ROC Bowl — Liver Placement\n"
        "(LiTS17 case-0 · sub-costal approach · bowl anterior to skin · focus = HCC centroid)",
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
    return save_fig(fig, "fig_21e_transducer_placement_3d")


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


def plot_pressure_and_lesion(ch_ct, label_vol, info, results, lesions,
                             gtv, ctv, ptv,
                             oar_masks: dict[str, np.ndarray] | None = None) -> tuple:
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

        # Row 1: regime-appropriate single-pulse dose/cavitation metric.
        # For shock-vapor: pcav is near-zero (PNP << 28.2 MPa threshold) —
        # show normalised T_transient instead. For sub-threshold: show
        # normalised collapse strength. For intrinsic: P_cav erf-CDF.
        dose_map_sl = r["dose_map_regime"][fx, y0:y1, z0:z1]
        regime_lbl = r["regime_label"]
        im1 = axes[1, col].imshow(dose_map_sl, cmap=pcav_cmap, extent=extent_crop,
                                  vmin=0, vmax=1, aspect="equal")
        axes[1, col].set_title(f"{regime_lbl}\n(focus-centred ±{win_mm:.0f} mm)",
                               fontsize=9)
        axes[1, col].set(xlabel="z [mm]", ylabel="y [mm]")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)
        # Add tumour, CTV, PTV contours on row 1
        for mask, clr, lw, lbl in (
            (gtv[fx, y0:y1, z0:z1].astype(float), "cyan",   0.9, "GTV"),
            (ctv[fx, y0:y1, z0:z1].astype(float), "yellow", 0.8, "CTV"),
            (ptv[fx, y0:y1, z0:z1].astype(float), "white",  0.7, "PTV"),
        ):
            if mask.any():
                axes[1, col].contour(np.flipud(mask), levels=[0.5],
                                     colors=clr, linewidths=lw,
                                     extent=extent_crop)

        # Row 2: cavitation-dose heatmap clipped to PTV on CT axial slice.
        # Displaying the full-field dose outside the PTV is misleading —
        # the therapeutic intent is within the planning volumes only.
        axes[2, col].imshow(ch_ct[fx, :, :], cmap="gray", vmin=-200, vmax=300,
                            extent=extent_full, aspect="equal")
        for mask, clr, lw in (
            (gtv[fx, :, :].astype(float), "cyan",   0.9),
            (ctv[fx, :, :].astype(float), "yellow", 0.8),
            (ptv[fx, :, :].astype(float), "white",  0.7),
        ):
            if mask.any():
                axes[2, col].contour(np.flipud(mask), levels=[0.5],
                                     colors=clr, linewidths=lw,
                                     extent=extent_full)
        # Distance-weighted dose: GTV carries full cav_dose; CTV annulus
        # decays linearly from the GTV surface to half-value at the CTV
        # boundary; PTV annulus decays to 0 at the outer PTV edge.
        # This makes the clinical intent visible — maximum ablation in
        # the gross target with graded margin coverage — without the
        # binary PTV clip that would show a hard-edged boundary.
        dose_weighted = margin_weighted_dose(
            r["cav_dose"], gtv, ptv, info,
            ctv_margin_m=5.0e-3, ptv_margin_m=3.0e-3,
        )
        dose_sl = dose_weighted[fx, :, :]
        im2 = axes[2, col].imshow(dose_sl, cmap=overlay_cmap, extent=extent_full,
                                  vmin=0, vmax=1, aspect="equal")
        plt.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04,
                     label="cavitation dose (GTV=full, margins=distance-weighted)")
        # OAR contours on the full-CT dose panel (row 2).
        # Absolute OARs are solid; liver_capsule (non-absolute) is dashed.
        if oar_masks is not None:
            ny_, nz_ = ch_ct.shape[1], ch_ct.shape[2]
            _draw_oar_contours(axes[2, col], oar_masks, info, fx,
                               0, ny_, 0, nz_, extent_full, lw=0.75)
        m = r["metrics"]
        oar_overlap = m.get("n_oar_footprint_overlap_pts", 0)
        axes[2, col].set_title(
            f"cavitation-dose (GTV→PTV gradient)\n"
            f"(cyan=GTV, yellow=CTV, white=PTV)\n"
            f"coverage {m.get('tumour_coverage_pct', 0):.0f}%, "
            f"cloud r={m.get('mech_radius_mm', 0):.1f} mm, "
            f"$T_{{ss}}$={m.get('T_steady_C', 0):.0f}°C"
            + (f"\nOAR footprint overlap: {oar_overlap} pts" if oar_overlap > 0 else ""),
            fontsize=8.0)
        axes[2, col].set(xlabel="z [mm]", ylabel="y [mm]")

    oar_note = ""
    if oar_masks is not None:
        oar_note = ("\nOAR contours: red=vessels, orange=gallbladder, "
                    "magenta=capsule (dashed), amber=extrahepatic, darkred=bone")
    fig.suptitle(
        "Histotripsy on real CT — focal pressure, regime-specific dose map, "
        "and distance-weighted cavitation-dose heatmap\n"
        "(GTV = full dose; CTV/PTV margins decay linearly to 0 at PTV edge; "
        "cyan=GTV, yellow=CTV +5 mm [HOPE4LIVER], white=PTV +8 mm total)" + oar_note,
        fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    b64 = save_fig(fig, "fig14_real_ct_lesion_panel")
    plt.close(fig)
    print("  saved fig14_real_ct_lesion_panel")
    return b64


def plot_raster_overlay(ct, label_vol, info, results, scenarios, gtv, ctv, ptv,
                        oar_masks: dict[str, np.ndarray] | None = None):
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
        # GTV / CTV / PTV contours (same colour convention as fig 14)
        for vol, clr, lw, lbl in (
            (gtv, "cyan",   1.0, "GTV"),
            (ctv, "yellow", 0.9, "CTV"),
            (ptv, "white",  0.7, "PTV"),
        ):
            sl = vol[fx, y0:y1, z0:z1].astype(float)
            if sl.any():
                ax.contour(np.flipud(sl), levels=[0.5], colors=clr,
                           linewidths=lw, extent=extent_crop)
        # Circle radius = per-shot mechanical cloud radius (not FWHM).
        # The FWHM is a -3 dB beam-width convention that describes the
        # transducer's amplitude pattern, not the cavitation footprint.
        # mech_radius_mm is the load-bearing lateral half-extent of the
        # regime-specific per-shot mask (T≥100°C seed cloud, collapse
        # cloud, or p≥p_t core), which is the quantity that actually
        # determines the raster pitch and coverage.
        psr = r["metrics"].get("mech_radius_mm",
                               r["metrics"]["fwhm_lat_mm"] / 2.0)
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
        # OAR contours on the raster overlay (cropped window).
        if oar_masks is not None:
            _draw_oar_contours(ax, oar_masks, info, fx,
                               y0, y1, z0, z1, extent_crop, lw=0.85)
        oar_overlap = r["metrics"].get("n_oar_footprint_overlap_pts", 0)
        oar_enc     = r["metrics"].get("n_oar_encasing_raster_pts", 0)
        oar_line = ""
        if oar_overlap > 0:
            oar_line += f"\nOAR footprint overlap: {oar_overlap} pts (dose monitored)"
        if oar_enc > 0:
            oar_line += f"  vessel-encasing: {oar_enc} pts (MDT review)"
        ax.set_title(
            f"{sc.label}\n"
            f"raster: {r['metrics'].get('actual_raster_points', sc.raster_points)} pts "
            f"× {psr:.1f} mm cloud r, "
            f"{sc.treatment_s/60:.1f} min\n"
            f"(cyan=GTV  yellow=CTV  white=PTV)" + oar_line,
            fontsize=7.5,
        )
        ax.set(xlabel="z [mm]", ylabel="y [mm]")
        # Legend: target volumes + OAR entries
        legend_handles = [
            Line2D([0], [0], color="cyan",   lw=1.0, label="GTV"),
            Line2D([0], [0], color="yellow", lw=0.9, label="CTV +5 mm (HOPE4LIVER)"),
            Line2D([0], [0], color="white",  lw=0.7, label="PTV +8 mm total"),
        ]
        if oar_masks is not None:
            legend_handles += _oar_legend_handles()
        ax.legend(handles=legend_handles, fontsize=6.5, loc="lower right",
                  framealpha=0.75, facecolor="black",
                  labelcolor="white", edgecolor="gray")
    oar_note = ""
    if oar_masks is not None:
        oar_note = ("\nOAR contours: red=vessels, orange=gallbladder, "
                    "magenta=capsule (dashed/flag), amber=extrahepatic, darkred=bone")
    fig.suptitle(
        "Raster scan overlay on real CT — coloured circles = per-shot "
        "cavitation-cloud radius (not FWHM)\n"
        "(cyan=GTV  yellow=CTV +5 mm [HOPE4LIVER]  white=PTV +8 mm total)" + oar_note,
        fontsize=9,
    )
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
                                       gtv, ctv, ptv,
                                       oar_masks: dict[str, np.ndarray] | None = None,
                                       n_frames: int = 80, fps: int = 12) -> str:
    """4-panel animation comparing raster-path strategies on the SAME
    scenario with GTV / CTV / PTV planning-volume overlays.

    Strategies compared: serpentine, outside-in, inside-out, adaptive
    low-dose.  Same total raster points and same pulses_per_spot in
    every panel — only the ORDER differs — so the side-by-side fill
    rate shows which strategy achieves the most uniform intermediate
    dose inside the GTV and how quickly it covers the CTV / PTV annuli
    through beam-tail deposition.

    GTV / CTV / PTV overlays
    ─────────────────────────
    Contours drawn once (static).  Ghost dots show the shared raster
    plan (all cyan = inside GTV) before any shot fires; the dose
    heatmap then fills frame-by-frame so the viewer can compare:

    * serpentine   — row-by-row, predictable but non-adaptive
    * outside-in   — peripheral shots first; avoids core shadowing
    * inside-out   — core first; residual nuclei lower threshold for
                     subsequent peripheral shots
    * adaptive     — lowest-dose voxel drives the next shot; maximises
                     uniformity at every intermediate time step

    The GTV confinement is identical across all four — only the fill
    ORDER changes, and the CTV / PTV beam-tail coverage follows.
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

    # Planning-volume 2D slices for static contours.
    gtv_sl = gtv[fx, y0:y1, z0:z1].astype(float)
    ctv_sl = ctv[fx, y0:y1, z0:z1].astype(float)
    ptv_sl = ptv[fx, y0:y1, z0:z1].astype(float)

    # Per-pulse field on the focal slice — same for all four strategies.
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

    # 3D field for the adaptive strategy.
    if sc.regime == "subthreshold_cav":
        coll_3d = collapse_strength(r["p_field"], sc.f0)
        per_pulse_3d = np.clip((coll_3d - 1.0) / 9.0, 0.0, 1.0).astype(np.float32)
    elif sc.regime == "shock_vapor":
        ps3 = r["per_shot_mask"].astype(np.float32)
        T_tr3 = r["T_transient"]
        T_norm3 = np.clip((T_tr3 - 37.0) / max((T_tr3.max() - 37.0), 1.0), 0.0, 1.0)
        per_pulse_3d = (ps3 * T_norm3).astype(np.float32)
    else:
        per_pulse_3d = np.clip(r["pcav"], 0.0, 1.0).astype(np.float32)

    raster_grid = r["raster_grid"]
    strategies = [
        ("serpentine",       serpentine_order(raster_grid)),
        ("outside-in",       outside_in_order(raster_grid)),
        ("inside-out",       inside_out_order(raster_grid)),
        ("adaptive low-dose",
         adaptive_lowdose_order(raster_grid, per_pulse_3d, info["focus_idx"],
                                max(sc.pulses_per_point, 1),
                                per_shot_mask=r["per_shot_mask"])),
    ]
    strat_anchors = {}
    strat_n_reanchors = {}
    for name, path in strategies:
        anchors_seq, n_changes = mechanical_walk(
            path, info["focus_idx"], info["dx"], sc.f0)
        strat_anchors[name] = anchors_seq
        strat_n_reanchors[name] = n_changes

    # Derive per-shot mechanical cluster IDs from anchor change events.
    # A new cluster begins whenever mechanical_walk() re-anchors (anchor
    # tuple changes between consecutive shots).  The same set of raster
    # POSITIONS is shared across all four strategies; what differs is the
    # ORDER of visits, which determines how consecutive shots group into
    # delivery clusters (one transducer mechanical pose per cluster).
    # Serpentine, outside-in, inside-out, and adaptive each produce a
    # different spatial grouping that is directly visible as colour bands.
    strat_cluster_ids: dict[str, list[int]] = {}
    for name, _ in strategies:
        anchors_seq = strat_anchors[name]
        ids: list[int] = []
        cluster_id = 0
        prev_anchor: tuple[int, int, int] | None = (
            anchors_seq[0] if anchors_seq else None)
        for a in anchors_seq:
            if a != prev_anchor:
                cluster_id += 1
                prev_anchor = a
            ids.append(cluster_id)
        strat_cluster_ids[name] = ids

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

    hot = plt.get_cmap("inferno")
    dose_colors = [(*hot(t)[:3], 0.0 if t < 0.02 else min(0.85, 0.15 + t * 0.95))
                   for t in np.linspace(0, 1, 256)]
    dose_cmap = LinearSegmentedColormap.from_list("cav_dose_strat", dose_colors)

    # Pre-compute ghost raster point positions (shared; order-independent).
    rpts = _raster_pts_by_volume(raster_grid, gtv, ctv, ptv, y0, y1, z0, z1, info)

    # Log-safe 2D field for parallel log-space accumulation (pre-lesion tracking).
    # Uses exact log(1 − per_pulse) so the cav_dose threshold is consistent with
    # predicted_lesion(); k_steer is applied per-shot inside animate().
    log_safe_2d = np.log(np.clip(1.0 - per_pulse_2d, 1e-9, 1.0))  # ≤ 0

    # GTV pixel count on focal slice for ablation-progress percentage.
    gtv_sl_bool = gtv_sl.astype(bool)
    n_gtv_px_strat = int(gtv_sl_bool.sum())

    fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.8))
    base_ct = ct[fx, y0:y1, z0:z1]

    artists = []
    for ax, (name, path) in zip(axes, strategies):
        ax.imshow(base_ct, cmap="gray", vmin=-200, vmax=300,
                  extent=extent_crop, aspect="equal")

        # Static GTV / CTV / PTV contours.
        for sl_mask, clr, lw in (
            (ptv_sl, "white",  0.7),
            (ctv_sl, "yellow", 0.9),
            (gtv_sl, "cyan",   1.1),
        ):
            if sl_mask.any():
                ax.contour(np.flipud(sl_mask), levels=[0.5],
                           colors=clr, linewidths=lw, extent=extent_crop)

        # Cluster-coloured ghost raster points.
        # POSITIONS are identical across all four panels (shared FPS raster).
        # COLOURS encode the mechanical delivery cluster: each colour = one
        # transducer pose.  A new cluster starts whenever mechanical_walk()
        # determines the next shot exceeds the electronic steering radius and
        # requires a physical table repositioning.  Different orderings
        # (serpentine, outside-in, inside-out, adaptive) group the same points
        # into spatially different clusters — the colour pattern makes that
        # grouping difference visible even though point positions are the same.
        path_yz = [(info["y_axis"][yi] * 1e3, info["z_axis"][zi] * 1e3)
                   for _, yi, zi in path]
        shot_y_mm = np.array([p[0] for p in path_yz], dtype=np.float32)
        shot_z_mm = np.array([p[1] for p in path_yz], dtype=np.float32)
        cids = np.array(strat_cluster_ids[name], dtype=np.float32)
        n_clusters = int(cids.max()) + 1 if len(cids) else 1
        ax.scatter(shot_z_mm, shot_y_mm,
                   c=cids, cmap="tab10", vmin=0, vmax=max(n_clusters - 1, 9),
                   s=20, alpha=0.45, edgecolors="none", zorder=4)

        dose_im = ax.imshow(np.full((ny_loc, nz_loc), 0.5, dtype=np.float32),
                            cmap=dose_cmap, norm=shared_norm,
                            extent=extent_crop, aspect="equal",
                            interpolation="nearest", zorder=3)

        # Pre-lesion overlay: lime-green for cav_dose ≥ 0.90.
        # Ablated tissue (liquefied HCC debris) has α ≈ 0.5 vs 12.5 dB/MHz/cm —
        # it is MORE acoustically transparent than intact HCC, so pre-lesioned
        # zones do not attenuate adjacent shots.  The overlay shows treatment
        # progress, not an acoustic obstacle.
        prelesion_rgba_init = np.zeros((ny_loc, nz_loc, 4), dtype=np.float32)
        prelesion_im = ax.imshow(prelesion_rgba_init, extent=extent_crop,
                                 aspect="equal", interpolation="nearest",
                                 zorder=6)

        # Static OAR contours — drawn once per panel on top of everything.
        if oar_masks is not None:
            _draw_oar_contours(ax, oar_masks, info, fx,
                               y0, y1, z0, z1, extent_crop, lw=0.80)

        title = ax.set_title(
            f"{name}\n{len(path)} shots in GTV, "
            f"{strat_n_reanchors[name]} mech re-anchors",
            fontsize=8.5)
        progress = ax.text(
            0.02, 0.97, "", transform=ax.transAxes, fontsize=7.0,
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.65, edgecolor="none",
                      boxstyle="round,pad=0.3"),
            color="white")
        ax.set(xlabel="z [mm]", ylabel="y [mm]")
        artists.append({
            "ax": ax, "dose_im": dose_im, "prelesion_im": prelesion_im,
            "events_acc": np.zeros((ny_loc, nz_loc), dtype=np.float32),
            # log_acc_2d = Σ_shots log(1−per_pulse·k)·pulses; ≤ 0
            # cav_dose_2d = 1 − exp(log_acc_2d)
            "log_acc_2d": np.zeros((ny_loc, nz_loc), dtype=np.float32),
            "shots_drawn": 0, "progress": progress, "title": title,
            "current_marker": None, "name": name, "path": path,
        })

    fig.colorbar(artists[-1]["dose_im"], ax=axes,
                 fraction=0.018, pad=0.02, shrink=0.85,
                 label="cumulative cavitation events (log scale)")

    # Shared legend (first panel only to avoid repetition).
    # n_gtv: use path length of any strategy — all share the same raster grid.
    n_gtv = len(strategies[0][1])
    # Sample tab10 colours to illustrate cluster encoding in the legend.
    _tab10 = matplotlib.colormaps["tab10"]
    legend_handles = [
        Line2D([0], [0], color="cyan",              lw=1.1,
               label=f"GTV contour ({n_gtv} shots — FPS positions shared by all)"),
        Line2D([0], [0], color="yellow",            lw=0.9, label="CTV +5 mm (HOPE4LIVER)"),
        Line2D([0], [0], color="white",             lw=0.7, label="PTV +8 mm total"),
        # Cluster-colour explanation: use the first three tab10 entries as swatches.
        mpatches.Patch(color=_tab10(0 / 9), alpha=0.7,
                       label="cluster 0 (initial transducer pose)"),
        mpatches.Patch(color=_tab10(1 / 9), alpha=0.7,
                       label="cluster 1 (1st mechanical re-anchor)"),
        mpatches.Patch(color=_tab10(2 / 9), alpha=0.7,
                       label="cluster 2+ (subsequent re-anchors)"),
        Line2D([0], [0], color=(0.2, 0.9, 0.35, 1), lw=5.0,
               label="pre-lesioned (cav_dose ≥ 90%)"),
    ]
    if oar_masks is not None:
        legend_handles += _oar_legend_handles()
    axes[0].legend(handles=legend_handles, fontsize=5.8, loc="lower right",
                   framealpha=0.75, facecolor="black",
                   labelcolor="white", edgecolor="gray")

    def init():
        return ([a["dose_im"] for a in artists]
                + [a["prelesion_im"] for a in artists])

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
                    dst_y0 = src_y0 + shift_y; dst_y1 = src_y1 + shift_y
                    dst_z0 = src_z0 + shift_z; dst_z1 = src_z1 + shift_z
                    anchor = strat_anchors[art["name"]][idx]
                    k_steer = steering_pressure_factor(
                        art["path"][idx], anchor, info["dx"], sc.f0)
                    art["events_acc"][dst_y0:dst_y1, dst_z0:dst_z1] += \
                        per_pulse_2d[src_y0:src_y1, src_z0:src_z1] * pulses * k_steer
                    # Parallel log-space accumulation: exact log(1−per_pulse·k).
                    # Pre-lesioned voxels accumulate strongly negative log_acc,
                    # naturally de-prioritising them in the adaptive scorer.
                    eff_pp = np.clip(
                        per_pulse_2d[src_y0:src_y1, src_z0:src_z1] * k_steer,
                        0.0, 0.9999)
                    art["log_acc_2d"][dst_y0:dst_y1, dst_z0:dst_z1] += \
                        np.log1p(-eff_pp) * pulses
                art["shots_drawn"] += 1

            art["dose_im"].set_data(art["events_acc"])
            changed.append(art["dose_im"])

            # Pre-lesion overlay: voxels with cav_dose ≥ 0.90 are lime-green.
            prelesion_mask = _prelesion_mask_2d(art["log_acc_2d"])
            prelesion_rgba = np.zeros((ny_loc, nz_loc, 4), dtype=np.float32)
            prelesion_rgba[prelesion_mask, 0] = 0.20
            prelesion_rgba[prelesion_mask, 1] = 0.90
            prelesion_rgba[prelesion_mask, 2] = 0.35
            prelesion_rgba[prelesion_mask, 3] = 0.38
            art["prelesion_im"].set_data(prelesion_rgba)
            changed.append(art["prelesion_im"])

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
                        markersize=9, mew=1.8, zorder=10)

            pct = 100.0 * shots_target / n_total if n_total else 0.0
            n_ablated = int(prelesion_mask.sum())
            pct_ablated = 100.0 * n_ablated / max(n_gtv_px_strat, 1)
            art["progress"].set_text(
                f"{shots_target}/{n_total} shots\n"
                f"GTV fill: {pct:.0f}%\n"
                f"Pre-lesioned: {n_ablated}/{n_gtv_px_strat} px "
                f"({pct_ablated:.0f}%)\n"
                f"{'■ DONE' if shots_target >= n_total else ''}"
            )
            changed.append(art["progress"])
        return changed

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, blit=False)
    fig.suptitle(
        f"Raster path strategies — {sc.label}  |  "
        f"{len(strategies[0][1])} shots · identical FPS positions across all panels · "
        f"strategy changes ORDER → different mechanical cluster groupings  |  LiTS17 HCC\n"
        "dot colour = mechanical delivery cluster (one transducer pose per colour; "
        "cluster count = mech re-anchors + 1 in panel title)  "
        "lime-green = pre-lesioned (cav_dose ≥ 90%)  "
        "acoustic note: ablated HCC is MORE transparent (α 12.5→0.5 dB/MHz/cm)\n"
        "Adaptive panel scores footprint worst-dosed voxel via log-space accumulation; "
        "saturated voxels rank last naturally.",
        y=1.01, fontsize=7.5)
    fig.subplots_adjust(top=0.83, bottom=0.10, left=0.04, right=0.94, wspace=0.18)
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


def plot_dvh_and_volumes(ct, label_vol, info, results, gtv, ctv, ptv,
                         scenarios) -> str:
    """Dose-Volume Histogram (DVH) and axial target-volume overlay.

    Panel layout
    ────────────
    Row 0 (3 columns): axial CT slice through the tumour centre with
        GTV / CTV / PTV contours and the per-scenario cumulative-dose
        heatmap (same PTV-clipped display as fig 14 row 2, but all
        three scenarios side by side for direct comparison).

    Row 1 (single wide panel): dose-volume histogram for each scenario
        evaluated on all three target volumes.  For each scenario s and
        volume V, the DVH gives the fraction of V that receives a
        cumulative cavitation dose ≥ d, plotted against d ∈ [0, 1].
        Key metrics annotated on the DVH:
            V95  — volume fraction receiving ≥ 0.95 dose (target ≥ 95 %)
            D95  — minimum dose received by 95 % of the volume
            Dmean — volume-average dose

    Clinical reference lines:
        d = 0.95 vertical dashed line (therapeutic coverage threshold)
        V = 0.95 horizontal dashed line (coverage target)

    The DVH quantifies how completely the dose fills each volume:
    a DVH curve that passes through (0.95, 0.95) satisfies the standard
    radiotherapy V95 ≥ 95 % coverage criterion, adapted here to
    cumulative cavitation-dose probability.
    """
    print("[ch21e] Building DVH + volumes figure")
    fx = info["focus_idx"][0]
    hot = plt.get_cmap("inferno")
    dose_colors = [(*hot(t)[:3], 0.0 if t < 0.05 else min(0.85, t * 1.2))
                   for t in np.linspace(0, 1, 256)]
    overlay_cmap = LinearSegmentedColormap.from_list("cav_dose_dvh", dose_colors)

    fig = plt.figure(figsize=(17, 11))
    gs_top = fig.add_gridspec(1, 3, top=0.92, bottom=0.54, wspace=0.22,
                              left=0.05, right=0.97)
    gs_bot = fig.add_gridspec(1, 1, top=0.46, bottom=0.07,
                              left=0.07, right=0.97)

    # ── Row 0: axial dose heatmaps ─────────────────────────────────────
    for col, sc in enumerate(scenarios):
        ax = fig.add_subplot(gs_top[0, col])
        r = results[sc.name]

        # Background CT
        tumour = gtv
        coords = np.argwhere(tumour)
        cmin = coords.min(axis=0); cmax = coords.max(axis=0)
        pad = int(0.04 / info["dx"])
        y0 = max(0, cmin[1] - pad); y1 = min(label_vol.shape[1], cmax[1] + pad)
        z0 = max(0, cmin[2] - pad); z1 = min(label_vol.shape[2], cmax[2] + pad)
        extent_crop = [info["z_axis"][z0] * 1e3, info["z_axis"][z1 - 1] * 1e3,
                       info["y_axis"][y1 - 1] * 1e3, info["y_axis"][y0] * 1e3]

        ax.imshow(ct[fx, y0:y1, z0:z1], cmap="gray", vmin=-200, vmax=300,
                  extent=extent_crop, aspect="equal")
        # GTV / CTV / PTV contours
        for vol, clr, lw in (
            (gtv, "cyan",   0.9),
            (ctv, "yellow", 0.8),
            (ptv, "white",  0.7),
        ):
            sl = vol[fx, y0:y1, z0:z1].astype(float)
            if sl.any():
                ax.contour(np.flipud(sl), levels=[0.5], colors=clr,
                           linewidths=lw, extent=extent_crop)
        # Distance-weighted dose overlay (same transform as fig 14 row 2):
        # GTV = full cav_dose; CTV/PTV annuli decay linearly to 0 at
        # the PTV outer edge so the clinical gradient is visible.
        dose_weighted = margin_weighted_dose(
            r["cav_dose"], gtv, ptv, info,
            ctv_margin_m=5.0e-3, ptv_margin_m=3.0e-3,
        )
        dose_sl = dose_weighted[fx, y0:y1, z0:z1]
        ax.imshow(dose_sl, cmap=overlay_cmap, extent=extent_crop,
                  vmin=0, vmax=1, aspect="equal")
        m = r["metrics"]
        ax.set_title(
            f"{sc.label}\n"
            f"GTV cov {m.get('tumour_coverage_pct', 0):.0f}% | "
            f"{m.get('actual_raster_points', 0)} shots | "
            f"{sc.treatment_s / 60:.1f} min",
            fontsize=8.5,
        )
        ax.set(xlabel="z [mm]", ylabel="y [mm]")
        if col == 0:
            legend_patches = [
                Line2D([0], [0], color="cyan",   lw=1.0, label="GTV (target)"),
                Line2D([0], [0], color="yellow", lw=0.9, label="CTV +5 mm (HOPE4LIVER)"),
                Line2D([0], [0], color="white",  lw=0.8, label="PTV +8 mm total"),
            ]
            ax.legend(handles=legend_patches, fontsize=7, loc="upper right",
                      framealpha=0.7)

    # ── Row 1: DVH ─────────────────────────────────────────────────────
    ax_dvh = fig.add_subplot(gs_bot[0, 0])
    d_vals = np.linspace(0, 1, 300)

    volume_specs = [
        ("GTV", gtv,  "-",  0.9),
        ("CTV", ctv,  "--", 0.7),
        ("PTV", ptv,  ":",  0.6),
    ]
    vname_of = {"GTV": gtv, "CTV": ctv, "PTV": ptv}

    for sc in scenarios:
        r = results[sc.name]
        # GTV uses raw cav_dose (honest physics — shots fired here).
        # CTV / PTV use the distance-weighted dose so the DVH curves
        # reflect the graded margin coverage intent rather than
        # stochastic beam-tail values: w=1 at GTV surface decaying
        # linearly to 0 at the PTV outer edge.  This makes the DVH
        # read as "what fraction of the margin receives therapeutic-
        # equivalent dose?" which is the clinically meaningful question.
        dose_raw = r["cav_dose"]
        dose_mw = margin_weighted_dose(
            dose_raw, gtv, ptv, info,
            ctv_margin_m=5.0e-3, ptv_margin_m=3.0e-3,
        )
        for vname, vmask, ls, alpha in volume_specs:
            n_vox = int(vmask.sum())
            if n_vox == 0:
                continue
            # GTV DVH: raw physics. CTV / PTV DVH: margin-weighted.
            voxel_doses = dose_raw[vmask] if vname == "GTV" else dose_mw[vmask]
            # DVH: fraction of volume receiving ≥ d
            dvh = np.array([(voxel_doses >= d).mean() for d in d_vals])
            ax_dvh.plot(d_vals, dvh, color=sc.color, linestyle=ls,
                        linewidth=1.4 * alpha, alpha=alpha,
                        label=f"{sc.name} / {vname}")

            # Annotate V95 and D95
            v95 = float((voxel_doses >= 0.95).mean())
            # D95: dose exceeded by 95% of the volume (5th percentile)
            d95 = float(np.percentile(voxel_doses, 5)) if n_vox >= 5 else 0.0
            if vname == "GTV":  # annotate GTV only to avoid clutter
                ax_dvh.annotate(
                    f"V95={v95 * 100:.0f}%\nD95={d95:.2f}",
                    xy=(d95, 0.95),
                    xytext=(d95 + 0.05, 0.95 - 0.08),
                    fontsize=7, color=sc.color,
                    arrowprops=dict(arrowstyle="->", color=sc.color, lw=0.6),
                )

    # Reference lines
    ax_dvh.axvline(0.95, color="k", ls="--", lw=0.8, label="d = 0.95 threshold")
    ax_dvh.axhline(0.95, color="gray", ls="--", lw=0.7, label="V95 target = 95 %")
    ax_dvh.set(
        xlabel="Cumulative cavitation dose (normalised, 0–1)",
        ylabel="Volume fraction receiving ≥ dose",
        xlim=(0, 1), ylim=(0, 1.02),
        title=(
            "Dose-Volume Histogram (DVH) — GTV / CTV / PTV\n"
            "line style: solid=GTV, dashed=CTV, dotted=PTV; "
            "colour = scenario"
        ),
    )
    ax_dvh.legend(fontsize=7, ncol=3, loc="lower left",
                  framealpha=0.85)

    fig.suptitle(
        "GTV / CTV / PTV — axial dose overlay and Dose-Volume Histogram\n"
        "(LiTS17 case-0 HCC; GTV = native segmentation, "
        "CTV = GTV + 5 mm [HOPE4LIVER NCT04573881], PTV = CTV + 3 mm)",
        fontsize=10,
    )
    b64 = save_fig(fig, "fig17_dvh_and_volumes")
    plt.close(fig)
    print("  saved fig17_dvh_and_volumes")
    return b64


def main():
    if not os.path.exists(CT_PATH):
        raise SystemExit(f"CT not found: {CT_PATH}. Download KiTS19 case_00000 first.")

    ct, label_vol, info = load_ct_and_segment(target_dx_m=1.2e-3)

    # GTV / CTV / PTV — computed once; threaded through all downstream
    # plotting and confinement calls so every figure uses consistent volumes.
    gtv, ctv, ptv = compute_gtv_ctv_ptv(label_vol, info["dx"],
                                         ctv_margin_m=5.0e-3,
                                         ptv_margin_m=3.0e-3)
    gtv_vol_mm3  = float(gtv.sum()  * (info["dx"] * 1e3) ** 3)
    ctv_vol_mm3  = float(ctv.sum()  * (info["dx"] * 1e3) ** 3)
    ptv_vol_mm3  = float(ptv.sum()  * (info["dx"] * 1e3) ** 3)
    print(f"[ch21e] Target volumes: "
          f"GTV={gtv_vol_mm3/1e3:.2f} cm³  "
          f"CTV={ctv_vol_mm3/1e3:.2f} cm³  "
          f"PTV={ptv_vol_mm3/1e3:.2f} cm³")

    tumour_volume_m3 = float((label_vol == HCC.label).sum() * info["dx"]**3)

    # Derive OAR masks once from CT HU + native segmentation.
    # These are passed to predicted_lesion() for raster exclusion and to
    # all plot/animation functions for contour display.
    print("[ch21e] Deriving organ-at-risk masks from CT HU + segmentation")
    oar_masks = derive_oar_masks(ct, label_vol, info)
    for spec in OAR_SPECS:
        n_vox = int(oar_masks[spec.name].sum())
        vol_cm3 = n_vox * (info["dx"] * 1e2) ** 3
        print(f"  OAR {spec.name}: {n_vox} vox, {vol_cm3:.1f} cm³"
              + (" [absolute no-lesion]" if spec.is_absolute else " [flagged]"))

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
        _, r_probe = predicted_lesion(p_field, props, sc_probe, info, label_vol,
                                      oar_masks=oar_masks)
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
        lesion, r = predicted_lesion(p_field, props, sc, info, label_vol,
                                     oar_masks=oar_masks)
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
        oar_overlap = m.get("n_oar_footprint_overlap_pts", 0)
        oar_enc     = m.get("n_oar_encasing_raster_pts", 0)
        print(f"    actual: {m['actual_raster_points']} pts, "
              f"{m['actual_treatment_s']/60:.1f} min, "
              f"coverage {m['tumour_coverage_pct']:.1f}%, "
              f"lesion {m['lesion_volume_mm3']/1e3:.2f} cm³, "
              f"spillover {m['spillover_volume_mm3']/1e3:.2f} cm³, "
              f"confinement {m['confinement_pct']:.0f}%, "
              f"T_trans {m['T_transient_C']:.1f} °C, "
              f"T_steady {m['T_steady_C']:.1f} °C")
        if oar_overlap > 0 or oar_enc > 0:
            print(f"    OAR: {oar_overlap} raster pts with OAR footprint overlap "
                  f"(dose monitored, not excluded), "
                  f"{oar_enc} pts inside vessel-encasing GTV (MDT review)")
        for spec in OAR_SPECS:
            dmax = m.get(f"oar_{spec.name}_dose_max", 0.0)
            if dmax > 0.05:
                print(f"    WARNING OAR {spec.name}: max dose {dmax:.3f} "
                      f"{'[ABSOLUTE]' if spec.is_absolute else '[flagged]'}")

    # Render figures + capture base64 for embedding
    b64_place = plot_transducer_placement_3d(label_vol, info)
    b64_seg   = plot_segmentation(ct, label_vol, info)
    b64_pan   = plot_pressure_and_lesion(ct, label_vol, info, results, lesions,
                                         gtv, ctv, ptv, oar_masks=oar_masks)
    b64_met = plot_metrics_summary(metrics, sized_scenarios)
    b64_ras = plot_raster_overlay(ct, label_vol, info, results, sized_scenarios,
                                  gtv, ctv, ptv, oar_masks=oar_masks)
    b64_dvh = plot_dvh_and_volumes(ct, label_vol, info, results,
                                   gtv, ctv, ptv, sized_scenarios)
    b64_anim = make_sonication_animation(ct, label_vol, info, results, sized_scenarios,
                                         gtv, ctv, ptv, oar_masks=oar_masks)
    # Strategy-comparison animation: pick the shock-vapor scenario
    # because its larger per-shot footprint gives the clearest
    # side-by-side fill differences between path orderings.
    sc_for_strategy = next((s for s in sized_scenarios if s.regime == "shock_vapor"),
                           sized_scenarios[0])
    b64_strat = make_strategy_comparison_animation(ct, label_vol, info, results,
                                                   sc_for_strategy, gtv, ctv, ptv,
                                                   oar_masks=oar_masks)

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
            "**Phase D-1 — real-PSTD `FocalKernel` `.npz` loader.** "
            "Added `kwavers::physics::field_surrogate::load_focal_kernel` "
            "and `::discover_focal_kernels`, backed by the "
            "`ndarray-npy` crate. The loader round-trips the schema "
            "`cavitation_kernel.py` writes (`p_min`, `dx`, `f0`, "
            "`pnp_realised`, `source_pa`, `focus_idx`, `fwhm_lat_m`, "
            "`fwhm_ax_m`), negating the signed `p_min` so the in-"
            "memory `FocalKernel.field` carries non-negative peak "
            "rarefactional pressure as the rest of the field-"
            "surrogate stack expects. Optional `target_pnp_pa` "
            "argument linearly rescales the field by `target / "
            "pnp_realised` — exact in water (B/A = 0). Five unit "
            "tests cover round-trip preservation, linear rescaling, "
            "missing-array rejection, focus-out-of-bounds rejection, "
            "and directory discovery sorted by filename. The "
            "`field_surrogate_demo` example now reads the env var "
            "`FIELD_SURROGATE_KERNEL_DIR`: when it points at a "
            "directory containing `kernel_*.npz` files, the trainer "
            "is supervised on the cached PSTD envelopes; otherwise "
            "the analytic Penttinen-Gaussian proxy is used. "
            "Reactivates the C-7 / C-8b / C-10 paths against the "
            "diverse-corner sweep once the cache is populated.\n\n"
            "Run the demo via:\n"
            "```sh\n"
            "cargo run --example field_surrogate_demo --release --features pinn\n"
            "python pykwavers/examples/book/param_pinn_training_figures.py\n"
            "```\n\n"
            "![Loss curves](./param_pinn_loss_curves.png)\n\n"
            "![Axial line fit](./param_pinn_axial_line_fit.png)\n\n"
        )
        f.write("### Figure 21.17 — 3D transducer placement (HistoSonics bowl on skin)\n\n")
        f.write(
            "3-D view of the 50 mm aperture / 120 mm radius-of-curvature "
            "hemispherical bowl placed via a sub-costal approach, anterior to "
            "the patient skin surface (depth < 0) with coupling gel.  The bowl "
            "apex (lime circle) marks the deepest point of the dish — closest "
            "to the patient — and the cyan cross marks the geometric focus "
            "inside the HCC tumour.  All bowl elements sit outside the body; "
            "the skin surface is at depth = 0.\n\n"
        )
        f.write(f"![Transducer placement 3D](data:image/png;base64,{b64_place})\n\n")
        f.write("### Figure 21.18 — CT segmentation\n\n")
        f.write(f"![CT segmentation](data:image/png;base64,{b64_seg})\n\n")
        f.write(
            "### Figure 21.19 — Focal pressure, regime dose map, "
            "and PTV-clipped cavitation-dose heatmap (3 scenarios)\n\n"
        )
        f.write(
            "**Row 0**: Peak rarefactional pressure [MPa] around the focus.\n\n"
            "**Row 1**: Regime-appropriate single-pulse dose metric (not P_cav "
            "for all regimes): "
            "*μs intrinsic* → $P_{cav}$ erf-CDF (Vlaisavljevich 2015); "
            "*ms shock-vapor* → normalised $T_{transient}$ / 100°C (vapor seed "
            "probability per Khokhlova 2011 — PNP=15 MPa is sub-intrinsic-"
            "threshold so $P_{cav}≈0$, but the shock delivers sufficient heat "
            "for boiling nucleation); "
            "*ms sub-threshold* → normalised collapse strength $S_c / 10$ "
            "(Vlaisavljevich 2018). "
            "GTV (cyan), CTV (yellow, +3 mm), and PTV (white, +6 mm) contours"
            "are overlaid.\n\n"
            "**Row 2**: Cumulative cavitation-dose heatmap clipped to the PTV. "
            "Dose outside the PTV is zeroed so the display reflects only the "
            "therapeutically relevant volume. Coverage, cloud radius, and "
            "steady-state temperature are annotated.\n\n"
        )
        f.write(f"![Lesion panel](data:image/png;base64,{b64_pan})\n\n")
        f.write("### Figure 21.20 — Raster scan overlay (pressure-contour based pitch)\n\n")
        f.write(
            "Raster pitch is derived from the per-shot mechanical cavitation-"
            "cloud half-extent (the region where T ≥ 100°C / P ≥ P_t / "
            "$S_c$ ≥ 5), NOT the FWHM beam width. FWHM is a diagnostic imaging "
            "convention (-3 dB amplitude, Penttinen 1976) unrelated to where "
            "the acoustic pressure crosses the therapeutic threshold. Using "
            "the true cavitation contour ensures adjacent shot footprints "
            "overlap by exactly 50 % so every inter-spot voxel receives a "
            "full exposure with no coverage gaps. Coloured circles show the "
            "per-shot cloud radius at each raster centre. GTV / CTV / PTV "
            "contours are drawn in the same colour convention as fig 14.\n\n"
        )
        f.write(f"![Raster overlay](data:image/png;base64,{b64_ras})\n\n")
        f.write("### Figure 21.21 — Real-CT scenario metrics summary\n\n")
        f.write(f"![Metrics summary](data:image/png;base64,{b64_met})\n\n")
        f.write(
            "### Figure 21.22 — GTV / CTV / PTV dose-volume histogram (DVH)\n\n"
        )
        f.write(
            "Dose-Volume Histogram evaluated on the three planning volumes "
            "(GTV = native HCC segmentation, CTV = GTV + 5 mm [HOPE4LIVER NCT04573881], "
            "PTV = CTV + 3 mm) for each scenario. The DVH gives the fraction "
            "of volume receiving ≥ d cumulative cavitation dose, plotted against "
            "d. **V95** (volume fraction with dose ≥ 0.95) and **D95** (dose "
            "at the 5th percentile of the volume) are annotated per scenario "
            "on the GTV curve. The standard radiotherapy target V95 ≥ 95 % "
            "(vertical d=0.95 and horizontal V=0.95 dashed lines) is carried "
            "forward as the histotripsy coverage criterion.\n\n"
            "The axial dose overlays (top row) confirm that PTV-clipping "
            "eliminates the apparent out-of-segmentation dose that was artefactual "
            "in previous figures — the displayed distribution is confined to "
            "the planning volumes only.\n\n"
        )
        f.write(f"![DVH and volumes](data:image/png;base64,{b64_dvh})\n\n")
        f.write("### Figure 21.23 — Sonication animation (real-time, synchronised)\n\n")
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
        f.write("### Figure 21.24 — Raster path strategy comparison\n\n")
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
